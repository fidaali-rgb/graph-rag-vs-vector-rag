from datasets import load_dataset

dataset = load_dataset("dgslibisey/MuSiQue")

print(dataset)
print(dataset["train"][0])

dataset["train"].to_json("musique_train.json")
dataset["validation"].to_json("musique_validation.json")