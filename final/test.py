import torch
from transformers import AutoTokenizer
from prev.improper_cot import AdvancedChainOfThoughtModel  

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
            temperature=0.7,
            do_sample=True,
        )

    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    # Load the saved model
    model_path = "advanced_cot_multi_news_model"  # Path to the saved model
    cot_model = AdvancedChainOfThoughtModel.from_pretrained(model_path)
    cot_model.eval()  # Set the model to evaluation mode

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cot_model.to(device)

    # Test document
    test_document = """
    The United Nations Climate Change Conference, COP26, is set to take place in Glasgow, Scotland. 
    World leaders, experts, and activists will gather to discuss and negotiate global climate action. 
    The conference aims to accelerate progress towards the goals of the Paris Agreement and the UN Framework Convention on Climate Change.
    Key topics will include reducing greenhouse gas emissions, adapting to climate impacts, and financing climate action in developing countries.
    """

    prompt = f"Summarize the following text step by step:\n\n{test_document}\n\nStep-by-step summary:"
    
    # Generate a response using the loaded model
    response = generate_with_cot(cot_model, prompt)
    print("Generated summary:")
    print(response)

if __name__ == "__main__":
    main()