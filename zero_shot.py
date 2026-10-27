from functools import lru_cache

from keywords import keyword_match
from rules import first_prompt


PROMPT_CATEGORIES = (
    "writing or editing an email",
    "a health or medical question",
    "work or personal document",
    "something else",
)
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
MIN_CATEGORY_SCORE = 0.5


@lru_cache(maxsize=1)
def _load_zero_shot_classifier():
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


def add_prompt_categories(rows):
    if not rows:
        return rows

    rows_for_zero_shot = []
    for row in rows:
        keyword_matched = row.pop("_keyword_matched", None)
        if keyword_matched is None:
            keyword_matched = keyword_match(first_prompt(row))

        if keyword_matched:
            row.pop("category", None)
        else:
            rows_for_zero_shot.append(row)

    if not rows_for_zero_shot:
        return rows

    classifier = _load_zero_shot_classifier()
    predictions = classifier(
        [first_prompt(row) for row in rows_for_zero_shot],
        candidate_labels=list(PROMPT_CATEGORIES),
        hypothesis_template="This prompt is {}.",
        multi_label=False,
        batch_size=16,
    )
    if isinstance(predictions, dict):
        predictions = [predictions]

    for row, prediction in zip(rows_for_zero_shot, predictions):
        row["category"] = prediction["labels"][0]
        row["category_score"] = prediction["scores"][0]
    return rows