import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, PeftConfig, PeftModel
from tqdm import tqdm
import os
import random
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.data import ReasoningDataset, InferenceCollator
from copy import deepcopy


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
def evaluate_expression(expr: str):
    expr = expr.strip().split('=')[0]  # only take left side of =
    try:
        return str(eval(expr))
    except Exception as e:
        return f"[ERROR: {e}]"


def patch_expressions(text: str):
    pattern = r"<<(.*?)>>" # we force the patter to be <<expr>> in training
    def replacer(match):
        expr = match.group(1)
        result = evaluate_expression(expr)
        return f"<<{expr}={result}>>"
    return re.sub(pattern, replacer, text)


def evaluate_model(model, test_loader, tokenizer):
    model.eval()
    all_answers = []
    all_predictions = []

    with tqdm(test_loader, unit="batch", dynamic_ncols=True) as tepoch:
        tepoch.set_description(f"Testing")

        for batch, gold_answers in tepoch:
            for i in range(batch['input_ids'].size(0)):
                input_ids = batch['input_ids'][i].unsqueeze(0)
                attention_mask = batch['attention_mask'][i].unsqueeze(0)

                full_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                done = False
                max_steps = 5

                for _ in range(max_steps):
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=64,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

                    if "<<" in decoded and ">>" in decoded: # Extract and evaluate <<expr>> patterns
                        patched = patch_expressions(decoded)
                        full_text = patched
                        input_ids = tokenizer(patched, return_tensors='pt').input_ids.to(model.device)
                        attention_mask = tokenizer(patched, return_tensors='pt').attention_mask.to(model.device)
                    else:
                        full_text = decoded
                        break

                match = re.search(r"<\|answer\|>(.*?)" + re.escape(tokenizer.eos_token), full_text) # Extract final answer
                prediction = match.group(1).strip() if match else ''
                all_predictions.append(prediction)
                all_answers.append(gold_answers[i])

                print("=" * 80)
                print("Final Output:\n", full_text)
                print("Prediction:", prediction)
                print("Gold:", gold_answers[i])

    accuracy = accuracy_score(all_answers, all_predictions)
    precision = precision_score(all_answers, all_predictions, average='weighted')
    recall = recall_score(all_answers, all_predictions, average='weighted')
    f1 = f1_score(all_answers, all_predictions, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    model_path = "model/weights_1B_HF_loss/acc/model"  # Replace <epoch_num> with the checkpoint you want (e.g. model-5), none if use out of box model
    # model_path = None
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    BATCH_SIZE = 8
    LR = 1e-6
    NUM_EPOCHS = 30
    LLM_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    # LLM_NAME = "gpt2"

    pad_token = "<|pad|>"
    eos_token = "<|eos|>"
    question_token = "<|question|>"
    rationale_token = "<|rationale|>"
    answer_token = "<|answer|>"

    if model_path:
        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.add_special_tokens({
            "pad_token": pad_token,
            "additional_special_tokens": [
                question_token,
                rationale_token,
                answer_token,
                "<<", ">>"
            ]
        })
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, model_path).to(DEVICE) # Load the LoRA adapter
    else:
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        tokenizer.add_special_tokens({
            "pad_token": pad_token,
            "additional_special_tokens": [
                question_token,
                rationale_token,
                answer_token,
                "<<", ">>"
            ]
        })
        model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    question_token_id = tokenizer.convert_tokens_to_ids(question_token)
    rationale_token_id = tokenizer.convert_tokens_to_ids(rationale_token)
    answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    collator = InferenceCollator(tokenizer, DEVICE, question_token, rationale_token, answer_token)

    test_dataset = ReasoningDataset("data/gsm8k/gsm8k_test.json")

    test_load = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        shuffle=False
    )

    accuracy, precision, recall, f1 = evaluate_model(model, test_load, tokenizer)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")