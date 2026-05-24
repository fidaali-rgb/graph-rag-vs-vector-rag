from datasets import load_dataset

# Load the CUAD dataset
dataset = load_dataset("theatticusproject/cuad-qa")

# Print dataset structure
print(dataset)

# Access splits
train_data = dataset["train"]
test_data = dataset["test"]

# Example sample
print(train_data[0])

# Save locally (optional)
train_data.to_json("cuad_train.json")
test_data.to_json("cuad_test.json")