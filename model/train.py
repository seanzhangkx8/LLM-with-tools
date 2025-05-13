import torch
import argparse
import wandb
from torch.utils.data import DataLoader, Subset
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import os
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from eval import evaluate_model

from data import ReasoningDataset, Collator, InferenceCollator

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train(model, train_loader, val_loader, test_loader, optimizer, tokenizer, scheduler, criterion, num_epochs, save_dir):
    best_metrics = {"acc" : -100, "precision" : -100, "recall" : -100, "f1" : -100, "loss" : 10000}
    for epoch in range(num_epochs):
        for phase, loader in zip(["train", "validate"], [train_loader, val_loader]):
            if phase == "train":
                model.train()
            else:
                model.eval()
            total_loss = 0.0

            with tqdm(loader, unit="batch", dynamic_ncols=True) as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs} {phase.capitalize()}")
                for batch in tepoch:
                    with torch.set_grad_enabled(phase == "train"):
                        input_ids = batch['input_ids']
                        outputs = model(input_ids=input_ids, attention_mask=batch['attention_mask'])
                        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))
                        loss = loss.mean()

                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                        loss_value = loss.item()
                        total_loss += loss_value
                        tepoch.set_postfix(loss=loss_value)

            avg_loss = total_loss / len(loader)
            wandb.log({
                                "epoch": epoch + 1,
                                f"{phase}/loss": loss_value,
                                **({f"{phase}/lr": scheduler.get_last_lr()[0]} if phase == "train" else {})
            }, commit=False)
            # }, commit=(phase == "validate"))
            print(f"Epoch [{epoch + 1}/{num_epochs}] Average {phase.capitalize()} Loss: {avg_loss:.4f}")

            val_loss = avg_loss if phase == "validate" else None

        acc, precision, recall, f1 = evaluate_model(model, test_loader, tokenizer)
        wandb.log({
            "epoch": epoch + 1,
            "test/accuracy": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1
        }, commit=True)
        train_state = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        for metric_name, metric in zip(['acc', 'precision', 'recall', 'f1', 'loss'], 
                                       [acc, precision, recall, f1, val_loss]):
            cur_save_dir = f"{save_dir}/{metric_name}/"
            os.makedirs(cur_save_dir, exist_ok=True)
            if metric >= best_metrics[metric_name]:
                minimal_state = {
                    'epoch': epoch + 1,
                    'eval' : {
                        'accuracy': acc,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'val_loss': val_loss
                    }
                }
                # best_metrics["loss"] = val_loss
                best_metrics[metric_name] = metric
                torch.save(train_state, f"{cur_save_dir}train_state.pt")
                torch.save(minimal_state, f"{cur_save_dir}minimal_state.pt")
                model.save_pretrained(f"{cur_save_dir}model")
                print(f"Saved checkpoint at epoch {epoch + 1} for {metric_name}.")
    print("Training completed!")
    print(f"Best metrics: {best_metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description='Script configuration')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else
                        'cpu',
                        help='Device to run on (cuda, mps, or cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--llm_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                        help='Name of the language model to use')
    parser.add_argument('--save_dir', type=str, default="weights/",
                        help='Directory to save model checkpoints')
    parser.add_argument('--data_dir', type=str, default="data/gsm8k")
    parser.add_argument('--wandb', type=str, default='online', help="Wandb mode")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    DEVICE = args.device
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_EPOCHS = args.num_epochs
    LLM_NAME = args.llm_name
    SAVE_DIR = args.save_dir
    DATA_DIR = args.data_dir

    pad_token = "<|pad|>"
    question_token = "<|question|>"
    rationale_token = "<|rationale|>"
    answer_token = "<|answer|>"

    lora_config = LoraConfig(**LoraConfig.from_json_file("model/config/lora_config.json"))

    model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    tokenizer.add_special_tokens({"pad_token": pad_token, "additional_special_tokens": [question_token, rationale_token, answer_token, '<<', '>>']})
    model.resize_token_embeddings(len(tokenizer))

    question_token_id = tokenizer.convert_tokens_to_ids(question_token)
    rationale_token_id = tokenizer.convert_tokens_to_ids(rationale_token)
    answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
    eos_token_id = tokenizer.eos_token_id

    collator = Collator(tokenizer, DEVICE, question_token, rationale_token, answer_token)
    inference_collator = InferenceCollator(tokenizer, DEVICE, question_token, rationale_token, answer_token)

    train_dataset = ReasoningDataset(f"{DATA_DIR}/gsm8k_train.json")
    val_dataset = ReasoningDataset(f"{DATA_DIR}/gsm8k_val.json")
    test_dataset = ReasoningDataset(f"{DATA_DIR}/gsm8k_test.json")
    test_subset = Subset(test_dataset, random.sample(range(len(test_dataset)), int(0.2 * len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=inference_collator, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, collate_fn=inference_collator, shuffle=False)


    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # set reduction to none, return shape [batch_size * seq_len]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * training_steps),
                                                num_training_steps=training_steps)

    wandb.init(mode=args.wandb, project="LLM-with-tools",
               config={"backbone": LLM_NAME, "epochs": NUM_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE})

    train(model, train_loader, val_loader, test_loader, optimizer, tokenizer, scheduler, criterion,
          num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR)

    wandb.finish()