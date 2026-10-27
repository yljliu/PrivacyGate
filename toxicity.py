from functools import lru_cache


TOXICITY_RULE = "toxic"
TOXICITY_MODEL = "s-nlp/roberta_toxicity_classifier"
TOXICITY_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load_toxicity_model():
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(TOXICITY_MODEL)
    model = RobertaForSequenceClassification.from_pretrained(TOXICITY_MODEL)
    model.eval()
    return tokenizer, model


def predict_toxicity(prompts):
    if not prompts:
        return []

    import torch

    tokenizer, model = _load_toxicity_model()
    predictions = []
    for start in range(0, len(prompts), TOXICITY_BATCH_SIZE):
        batch = tokenizer(
            prompts[start:start + TOXICITY_BATCH_SIZE],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            labels = model(**batch).logits.argmax(dim=-1).tolist()
        predictions.extend(label == 1 for label in labels)
    return predictions


def filter_toxicity(rows):
    predictions = predict_toxicity(
        [
            next(
                (turn.get("content") or "" for turn in row.get("conversation", []) if turn.get("role") == "user"),
                "",
            )
            for row in rows
        ]
    )
    kept, dropped = [], []
    for row, is_toxic in zip(rows, predictions):
        (dropped if is_toxic else kept).append(row)
    return kept, dropped