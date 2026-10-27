from huggingface_hub import hf_hub_download, DatasetCard, HfApi

info = HfApi().dataset_info("allenai/WildChat-1M")
print(info.card_data.get("license"), info.tags)

card = DatasetCard.load("allenai/WildChat-1M")
print(card.content[:3000])     # read the license/terms and usage sections