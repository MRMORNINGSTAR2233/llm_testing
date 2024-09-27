import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import wandb
import logging

class ChainOfThoughtDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 1024, split="train[:10]"):
        self.dataset = hf_dataset[split]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        document = item['document']
        summary = item['summary']

        # Create a chain of thought prompt
        input_text = f"Summarize the following text step by step:\n\n{document}\n\nStep-by-step summary:"
        target_text = f"Step 1: Identify main topics\nStep 2: Extract key information\nStep 3: Organize information\nStep 4: Condense and refine\nFinal summary: {summary}"

        inputs = self.tokenizer.encode_plus(
            input_text,
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Assuming ReasoningModule and AdvancedChainOfThoughtModel classes are defined elsewhere
class ReasoningModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

class AdvancedChainOfThoughtModel(nn.Module):
    def __init__(self, base_model_name: str, num_reasoning_steps: int = 2):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_reasoning_steps = num_reasoning_steps
        
        hidden_size = self.base_model.config.hidden_size
        self.reasoning_modules = nn.ModuleList([ReasoningModule(hidden_size) for _ in range(num_reasoning_steps)])
        
        self.reasoning_combiner = nn.Linear(hidden_size * (num_reasoning_steps + 1), hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.base_model.config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use the last hidden state from base_outputs
        hidden_states = [base_outputs.hidden_states[-1]]
        current_state = base_outputs.hidden_states[-1]
        
        for reasoning_module in self.reasoning_modules:
            reasoned_state = reasoning_module(current_state)
            hidden_states.append(reasoned_state)
            current_state = reasoned_state
        
        combined_state = torch.cat(hidden_states, dim=-1)
        final_state = self.reasoning_combiner(combined_state)
        
        lm_logits = self.output_layer(final_state)
        
        return lm_logits

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.base_model.generate(input_ids, attention_mask=attention_mask, **kwargs)

    def save_pretrained(self, save_directory):
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/cot_model.pt")

    @classmethod
    def from_pretrained(cls, load_directory):
        base_model = AutoModelForCausalLM.from_pretrained(load_directory)
        tokenizer = AutoTokenizer.from_pretrained(load_directory)
        model = cls(base_model.config.name_or_path)
        model.base_model = base_model
        model.tokenizer = tokenizer
        model.load_state_dict(torch.load(f"{load_directory}/cot_model.pt"))
        return model
    
def train_advanced_cot_model(
    model: AdvancedChainOfThoughtModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    epochs: int = 3,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
    logging_steps: int = 100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    if fp16:
        scaler = torch.cuda.amp.GradScaler()

    wandb.init(project="advanced-cot-training")
    
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for step, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=fp16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    if fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        wandb.log({
                            "train_loss": epoch_loss / (step + 1),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": global_step
                        })

                pbar.update(1)
                pbar.set_postfix({"loss": epoch_loss / (step + 1)})

        if val_dataloader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
            logging.info(f"Validation Loss: {val_loss}")

    wandb.finish()

# Load and prepare the dataset
def prepare_dataset(tokenizer, batch_size=6, train_subset_ratio=0.1):
    dataset = load_dataset("multi_news")
    train_dataset = ChainOfThoughtDataset(dataset, tokenizer, split="train")
    val_dataset = ChainOfThoughtDataset(dataset, tokenizer, split="validation")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader

# Main execution
def main():
    base_model_name = "gpt2"  # Using a smaller model for faster training
    cot_model = AdvancedChainOfThoughtModel(base_model_name, num_reasoning_steps=2)

    # Prepare the dataset
    train_dataloader, val_dataloader = prepare_dataset(cot_model.tokenizer)

    # Train the model
    train_advanced_cot_model(
        cot_model, 
        train_dataloader, 
        val_dataloader=val_dataloader,
        epochs=1,  # Reduced for demonstration
        lr=1e-5, 
        fp16=True
    )

    # Save the model
    cot_model.save_pretrained("cot_multi_news_model")

    # Example inference
    def generate_with_cot(model, prompt, max_length=300):
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Test the model
    test_document = """
    The United Nations Climate Change Conference, COP26, is set to take place in Glasgow, Scotland. 
    World leaders, experts, and activists will gather to discuss and negotiate global climate action. 
    The conference aims to accelerate progress towards the goals of the Paris Agreement and the UN Framework Convention on Climate Change.
    Key topics will include reducing greenhouse gas emissions, adapting to climate impacts, and financing climate action in developing countries.
    """

    prompt = f"Summarize the following text step by step:\n\n{test_document}\n\nStep-by-step summary:"
    response = generate_with_cot(cot_model, prompt)
    print(response)

if __name__ == "__main__":
    main()
