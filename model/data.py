from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
        with open("data/gsm8k/gsm8k_train.json", "r") as f: 
            self.examples = json.load(f)
        self.compare_model = SentenceTransformer("all-MiniLM-L6-v2")
        questions = [ex["question"] for ex in self.examples]
        self.question_embeddings = self.compare_model.encode(questions, convert_to_tensor=True, normalize_embeddings=True)


    def __call__(self, batch, with_example=True, with_dynamic_example=True):
        prompts = []
        answers = []

        for item in batch:
            if with_example:
                if with_dynamic_example:
                    examples = self.retrieve_similar_examples(item['question'], k=3)
                else:
                    examples = [
                        "Question: Stefan goes to a restaurant to eat dinner with his family. They order an appetizer that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for the waiter, what is the total amount of money that they spend at the restaurant? Rationale: The total cost of the entrees is 4 * $20 = $<<4*20=80>>80.\nThe total cost of the dinner is $80 + $10 = $<<80+10=90>>90.\nThe tip is $90 * 0.20 = $<<90*0.20=18>>18\nThe total cost with tip is $90 + $18 = $<<90+18=108>>108\n Answer: 108",
                        "Question: The gauge on a water tank shows that the tank is 1/3 full of water. To fill the tank, 16 gallons of water are added. How many gallons of water does the tank hold when full? Rationale: Given that the tank is 1/3 full of water, and that it requires 16 gallons to fill. 1 full tank -1/3 tank = 2/3 of the tank is empty, which is equal to 16 gallons.\nIf 2/3 of the tank equals 16 gallons, this means 1/3 of the tank is equal to 16 / 2 gallons = 8 gallons\nThe total capacity of the tank or how much the tanks holds = 16 + 8 = <<16+8=24>>24 gallons\n Answer: 24",
                        "Question: Ben has 8 apples more than Phillip does. Tom has three eighths as many apples at Ben has. If Phillip has 40 apples, how many apples does Tom have? Rationale: Ben has 40+8 = <<40+8=48>>48 apples.\nThere are 48/8 = <<48/8=6>>6 apples in every eighth.\nTom has 6*3 = <<6*3=18>>18 apples.\n Answer: 18"
                    ]

                joined_examples = [f"##########\nExample {i+1}: {example}##########\n" for i, example in enumerate(examples)]
                joined_examples_text = "".join(joined_examples)
                prompt = (
                    f"{self.question_token}{item['question']} "
                    f"Examples of how to solve similar questions: {joined_examples_text} "
                    f"Now, Let's think step by step: {self.rationale_token}"
                )
            else:
                prompt = f"{self.question_token}{item['question']}Let's think step by step: {self.rationale_token}"

            prompts.append(prompt)
            answers.append(item['answer'])

        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            padding_side='left',
            return_tensors='pt'
        ).to(self.device)

        return tokenized_prompts, answers
    
    def retrieve_similar_examples(self, query, k=3):
        query_embedding = self.compare_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = (self.question_embeddings @ query_embedding).cpu().numpy()  # cosine similarity via dot product
        top_indices = np.argsort(scores)[-k:][::-1]
        retrieved = [self.examples[i] for i in top_indices]
        return [self.format_gsm8k_example(ex) for ex in retrieved]
    
    def format_gsm8k_example(self, example):
        return (
            f"Question: {example['question']} "
            f"Rationale: {example['reasoning'].strip()} "
            f"Answer: {example['answer']}"
        )


if __name__ == '__main__':
    dataset = ReasoningDataset('data/dataset/gsm8k.json')
    print(dataset[0])