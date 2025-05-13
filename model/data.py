from torch.utils.data import Dataset
import json

class ReasoningDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as reader:
            self.data = json.load(reader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            'question': example['question'],
            'rationale': example['reasoning'],
            'answer': example['answer'],
        }
    

class Collator:
    def __init__(self, tokenizer, device, question_token, rationale_token, answer_token):
        self.tokenizer = tokenizer
        self.device = device
        self.special_token_ids = tokenizer.all_special_ids
        self.question_token = "<|question|>"
        self.rationale_token = "<|rationale|>"
        self.answer_token = "<|answer|>"
        self.question_token_id = tokenizer.convert_tokens_to_ids(question_token)
        self.rationale_token_id = tokenizer.convert_tokens_to_ids(rationale_token)
        self.answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        self.token_idf = None

    def __call__(self, batch):
        prompts = [f"{self.question_token}{item['question']}Let's think step by step: {self.rationale_token}{item['rationale']}{self.answer_token}{item['answer']}{self.tokenizer.eos_token}" for item in batch]

        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            padding_side='left',
            return_tensors='pt'
        ).to(self.device)
        tokenized_prompts["labels"] = tokenized_prompts['input_ids'].masked_fill(~(tokenized_prompts['attention_mask'].bool()), -100).to(self.device)

        return tokenized_prompts


class InferenceCollator(Collator):
    def __init__(self, tokenizer, device, question_token, rationale_token, answer_token):
        super().__init__(
            tokenizer, device,
            question_token, rationale_token, answer_token
        )

    def __call__(self, batch):
        prompts = [f"{self.question_token}{item['question']}Let's think step by step: {self.rationale_token}" for item in batch]
        answers = [f"{item['answer']}" for item in batch]

        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            padding_side='left',
            return_tensors='pt'
        ).to(self.device)

        return tokenized_prompts, answers


if __name__ == '__main__':
    dataset = ReasoningDataset('data/dataset/gsm8k.json')
    print(dataset[0])