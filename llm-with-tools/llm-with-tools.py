import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)

# Load GSM8K test split
gsm8k = load_dataset("gsm8k", "main", split="test")

# Format prompt: Use standard CoT template
def format_prompt(example):
    return f"Q: {example['question']}\nA: Let's think step by step.\n"

# Generate answer
def generate_answer(prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run on a few samples
for i in range(5):  # change this to more if needed
    ex = gsm8k[i]
    prompt = format_prompt(ex)
    output = generate_answer(prompt)
    print("=" * 80)
    print(f"Question:\n{ex['question']}")
    print(f"Gold Answer:\n{ex['answer']}")
    print(f"Model Output:\n{output[len(prompt):].strip()}")