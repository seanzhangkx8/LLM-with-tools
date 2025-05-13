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

from data import ReasoningDataset, InferenceCollator

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def evaluate_model(model, test_loader, tokenizer):
    model.eval()
    all_answers = []
    all_predictions = []
    with tqdm(test_loader, unit="batch", dynamic_ncols=True) as tepoch:
        tepoch.set_description(f"Testing")
        for batch, answer in tepoch:
            with torch.no_grad():
                input_ids = batch['input_ids']
                outputs = model.generate(input_ids=input_ids, attention_mask=batch['attention_mask'], max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                predictions = torch.cat((outputs, torch.tensor([tokenizer.convert_tokens_to_ids("<|answer|>") for _ in range(outputs.shape[0])]).unsqueeze(1).to(outputs.device)), dim=1)
                # print(predictions)
                new_attn_mask = torch.cat((batch['attention_mask'], torch.ones(outputs.shape[0], (outputs.shape[1] -  input_ids.shape[1] + 1)).to(outputs.device)), dim=1)
                outputs = model.generate(input_ids=predictions, attention_mask=new_attn_mask.to(predictions.device), max_new_tokens=10, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                print(predictions)
                predictions = [
                                    re.search(r"<\|answer\|>(.*?)" + re.escape(tokenizer.eos_token), words).group(1).strip()
                                    if re.search(r"<\|answer\|>(.*?)" + re.escape(tokenizer.eos_token), words) else ''
                                    for words in predictions
                                ]
                all_answers.extend(answer)
                all_predictions.extend(predictions)
    accuracy = accuracy_score(all_answers, all_predictions)

    precision = precision_score(all_answers, all_predictions, average='weighted')
    recall = recall_score(all_answers, all_predictions, average='weighted')
    f1 = f1_score(all_answers, all_predictions, average='weighted')
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    model_path = "model/weights/acc/model"  # Replace <epoch_num> with the checkpoint you want (e.g. model-5)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    BATCH_SIZE = 1
    LR = 1e-6
    NUM_EPOCHS = 30
    LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    # LLM_NAME = "gpt2"

    pad_token = "<|pad|>"
    eos_token = "<|eos|>"
    question_token = "<|question|>"
    rationale_token = "<|rationale|>"
    answer_token = "<|answer|>"

    # lora_config = LoraConfig(**LoraConfig.from_json_file("model/config/lora_config.json"))

    # model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    # model = get_peft_model(model, lora_config)

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

    # tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    # tokenizer.add_special_tokens({"pad_token": pad_token, "eos_token": eos_token, "additional_special_tokens": [question_token, rationale_token, answer_token]})
    # model.resize_token_embeddings(len(tokenizer))

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