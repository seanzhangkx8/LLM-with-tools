import json
import random

# Load the split information
split = json.load(open("data/split/gsm8k__default.json", "r"))
train_split = set(split["train"])  # faster lookup
test_split = set(split["test"])

print(f"Train split size: {len(train_split)}")
print(f"Test split size: {len(test_split)}")

# Load the full dataset
full_data = json.load(open("data/dataset/gsm8k.json", "r"))

# Now split
train_data = []
test_data = []

for example in full_data['data']:
    example_id = example["sample_index"]
    if example_id in train_split:
        train_data.append(example)
    elif example_id in test_split:
        test_data.append(example)

# Random split train_data into train and val (90% train, 10% val)
random.seed(42)  # fix seed for reproducibility
random.shuffle(train_data)

split_idx = int(0.9 * len(train_data))
new_train_data = train_data[:split_idx]
val_data = train_data[split_idx:]

# Save to files
with open("data/gsm8k/gsm8k_train.json", "w") as f:
    json.dump(new_train_data, f, indent=4)

with open("data/gsm8k/gsm8k_val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open("data/gsm8k/gsm8k_test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print(f"Saved {len(new_train_data)} train examples, {len(val_data)} val examples, and {len(test_data)} test examples!")
