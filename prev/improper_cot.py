import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, PreTrainedModel,AutoConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import wandb
import logging
import json


class ChainOfThoughtDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 1024, split="train", num_examples: Optional[int] = None):
        self.dataset = hf_dataset[split]
        if num_examples is not None:
            self.dataset = self.dataset.select(range(min(num_examples, len(self.dataset))))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        document = item['document']
        summary = item['summary']

        input_text = f"Summarize the following text step by step:\n\n{document}\n\nStep-by-step summary:"
        target_text = f"Step 1: Identify main topics\nStep 2: Extract key information\nStep 3: Organize information\nStep 4: Condense and refine\nFinal summary: {summary}"

        full_text = input_text + target_text

        encoded = self.tokenizer.encode_plus(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        target_start = len(self.tokenizer.encode(input_text, add_special_tokens=False))

        labels = input_ids.clone()
        labels[:target_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

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

class AdvancedChainOfThoughtModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.base_model = AutoModelForCausalLM.from_config(config)
        
        hidden_size = config.hidden_size // 2
        self.num_reasoning_steps = getattr(config, 'num_reasoning_steps', 1)
        
        self.reasoning_modules = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_reasoning_steps)])
        
        self.reasoning_combiner = nn.Linear(hidden_size * (self.num_reasoning_steps + 1), hidden_size)
        self.output_layer = nn.Linear(hidden_size, config.vocab_size)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        num_reasoning_steps = kwargs.pop('num_reasoning_steps', 1)
        
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.num_reasoning_steps = num_reasoning_steps
        
        model = cls(config)
        model.base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)
        model.tokenizer.pad_token = model.tokenizer.eos_token
        
        return model

    def forward(self, input_ids, attention_mask=None):
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        hidden_states = [base_outputs.hidden_states[-1][:, :, :self.config.hidden_size//2]]
        current_state = hidden_states[0]
        
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

def train_advanced_cot_model(
    model: AdvancedChainOfThoughtModel,
    train_dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-4,
    warmup_steps: int = 1000,
    gradient_accumulation_steps: int = 16,
    max_grad_norm: float = 1.0,
    logging_steps: int = 100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.base_model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

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

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(outputs.float().view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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

                torch.cuda.empty_cache()

                pbar.update(1)
                pbar.set_postfix({"loss": epoch_loss / (step + 1)})

    wandb.finish()

def prepare_dataset(tokenizer, batch_size=3, max_length=512, train_examples=50):
    dataset = load_dataset("multi_news")
    train_dataset = ChainOfThoughtDataset(dataset, tokenizer, max_length=max_length, split="train", num_examples=train_examples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

def main():
    base_model_name = "gpt2" 
    cot_model = AdvancedChainOfThoughtModel(base_model_name, num_reasoning_steps=3)

    train_dataloader = prepare_dataset(
        cot_model.tokenizer,
        train_examples=50
    )

    train_advanced_cot_model(
        cot_model, 
        train_dataloader, 
        epochs=1,
        lr=2e-4, 
    )

    # Save the model using Hugging Face's save_pretrained method
    cot_model.base_model.save_pretrained("cot_multi_news_model")
    cot_model.tokenizer.save_pretrained("cot_multi_news_model")

    # Example inference
    def generate_with_cot(model, prompt, max_length=300):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=3,
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