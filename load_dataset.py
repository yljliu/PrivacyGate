from datasets import load_dataset

# Load the standard WildChat-1M dataset
ds = load_dataset("allenai/WildChat-1M", split="train")

print(ds)
print(ds[0].keys())

