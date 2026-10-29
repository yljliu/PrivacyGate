import argparse
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from toxicity import TOXICITY_RULE, filter_toxicity


def first_prompt(row):
    for turn in row.get("conversation", []):
        if turn.get("role") == "user":
            return turn.get("content") or ""
    return ""


def is_first_user_turn(row):
    conversation = row.get("conversation") or []
    first_non_system_turn = next(
        (turn.get("role") for turn in conversation if turn.get("role") != "system"),
        None,
    )
    return first_non_system_turn == "user"


RULES = {
    "dataset_toxic": lambda row: row.get("toxic") is True,
    "not_first_user_turn": lambda row: not is_first_user_turn(row),
    "non_english": lambda row: str(row.get("language") or "").strip().casefold() != "english",
    "too_short": lambda row: len(first_prompt(row).split()) < 5,
    "too_long": lambda row: len(first_prompt(row).split()) > 2000,
}

ALL_RULES = (*RULES, TOXICITY_RULE)


def find_parquet_files():
    workspace_files = list(Path(__file__).parent.rglob("*.parquet"))
    workspace_files = [
        p for p in workspace_files
        if ".venv" not in p.parts and p.name != "filtered.parquet"
    ]
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_files = list((cache_root / "datasets--allenai--WildChat-1M").glob("snapshots/*/data/*.parquet"))
    return [str(path) for path in sorted(set(workspace_files + cache_files))]


# def load_sample(limit=10_000, seed=42, parquet_files=None):
#     files = parquet_files or find_parquet_files()
#     if not files:
#         raise FileNotFoundError("No parquet files found. Download WildChat or pass parquet paths.")

#     rng = random.Random(seed)
#     dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)
#     reservoir = []
#     for index, row in enumerate(dataset):
#         if index < limit:
#             reservoir.append(row)
#         else:
#             replacement = rng.randint(0, index)
#             if replacement < limit:
#                 reservoir[replacement] = row

#     rng.shuffle(reservoir)
#     return reservoir


def classify_rules(rows):
    drops, kept, dropped = Counter(), [], []
    for row in rows:
        hits = [name for rule_name, rule in RULES.items() if rule(row) for name in [rule_name]]
        for hit in hits:
            drops[hit] += 1
        if hits:
            dropped.append((row, hits))
        else:
            kept.append((row, []))
    return drops, kept, dropped


def classify(rows):
    drops, eligible, dropped = classify_rules(rows)
    kept_rows, toxic_rows = filter_toxicity([row for row, _ in eligible])
    for row in toxic_rows:
        drops[TOXICITY_RULE] += 1
        dropped.append((row, [TOXICITY_RULE]))
    kept = [(row, []) for row in kept_rows]
    return drops, kept, dropped


def run(limit=10_000, seed=42, parquet_files=None):
    sample = load_sample(limit, seed, parquet_files)
    drops, kept, dropped = classify(sample)
    files = parquet_files or find_parquet_files()
    print(f"loaded {len(sample)} rows from {len(files)} parquet file(s)")
    if not sample:
        print("No rows loaded.")
        return sample, drops, kept, dropped

    print(f"kept {len(kept) / len(sample):.1%}")
    for name, count in drops.most_common():
        print(f"{name:15s} {count / len(sample):.1%}")
    return sample, drops, kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Apply WildChat filtering rules to parquet data.")
    parser.add_argument("parquet_files", nargs="*", help="Parquet paths; cached WildChat shards are used by default.")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.limit, args.seed, args.parquet_files or None)
