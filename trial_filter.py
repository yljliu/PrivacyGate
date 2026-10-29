import argparse
import hashlib
import random
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from rules import RULES, classify_rules, find_parquet_files, first_prompt
from toxicity import filter_toxicity
from zero_shot import add_prompt_categories
from keywords import keyword_match

BATCH_SIZE = 32


def prompt_key(row):
    """Dedupe key: hash of the whitespace/case-normalized first user prompt."""
    text = " ".join(first_prompt(row).split()).casefold()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def write_filtered(rows, output_path):
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    print(f"wrote {len(rows)} rows to {output_path}")


def write_annotator_file(rows, output_path):
    """Annotators get only an ID and the first user prompt: no metadata, no replies."""
    slim = [
        {"id": row.get("conversation_hash", str(i)), "prompt": first_prompt(row)}
        for i, row in enumerate(rows)
    ]
    pq.write_table(pa.Table.from_pylist(slim), output_path)
    print(f"wrote {len(slim)} annotator rows to {output_path}")


# ---------------------------------------------------------------------------
# Stage 1: cheap pass over the whole dataset (basic rules + keywords + dedupe).
# Reservoir-samples up to `pool_size` unique survivors with the fixed seed.
# ---------------------------------------------------------------------------
def build_candidate_pool(files, pool_size, seed):
    rng = random.Random(seed)
    dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)

    pool = {}        # prompt_key -> row
    pool_slots = []  # slot index -> prompt_key
    seen_keys = set()
    rule_drops = Counter()
    scanned = passed_rules = passed_keywords = duplicates = 0
    batch = []

    def consider_batch(rows):
        nonlocal scanned, passed_rules, passed_keywords, duplicates
        scanned += len(rows)
        drops, kept, _ = classify_rules(rows)
        rule_drops.update(drops)
        for row, _ in kept:
            passed_rules += 1
            if not keyword_match(first_prompt(row)):
                continue

            key = prompt_key(row)
            if key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(key)
            passed_keywords += 1  # unique survivors only, keeps reservoir math correct

            if len(pool_slots) < pool_size:
                pool[key] = row
                pool_slots.append(key)
            else:
                j = rng.randint(0, passed_keywords - 1)
                if j < pool_size:
                    del pool[pool_slots[j]]
                    pool[key] = row
                    pool_slots[j] = key

    for row in dataset:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            consider_batch(batch)
            batch = []
            if scanned % 50_000 < BATCH_SIZE:
                print(f"scanned {scanned:,} rows; {passed_keywords:,} unique survivors")
    if batch:
        consider_batch(batch)

    stats = {
        "scanned": scanned,
        "passed_rules": passed_rules,
        "duplicates_skipped": duplicates,
        "passed_keywords_unique": passed_keywords,
        "pool_size": len(pool),
    }
    print(f"stage 1 done: {stats}")
    print("stage 1 filtered by rule:")
    for rule_name in RULES:
        print(f"  {rule_name}: {rule_drops[rule_name]:,}")
    return pool, stats


# ---------------------------------------------------------------------------
# Stage 2: dedupe the pool by text, then run the slow models on a seeded
# shuffle, stopping once `limit` prompts are accepted.
# ---------------------------------------------------------------------------
def select_sample(pool, limit, seed, min_score):
    rows = pool.values() if isinstance(pool, dict) else pool

    # 1) Scan the pool for duplicate texts (safety net; stage 1 should already prevent them).
    unique = {}
    for row in rows:
        unique.setdefault(prompt_key(row), row)
    duplicates_removed = len(pool) - len(unique)

    # 2) Seeded shuffle of the unique prompts so stopping early is a fair random draw.
    keys = list(unique)
    random.Random(seed).shuffle(keys)

    # 3) Toxicity -> zero-shot, batch by batch, until we have enough.
    accepted = {}
    processed = toxic_removed = category_removed = 0

    for start in range(0, len(keys), BATCH_SIZE):
        batch = [unique[k] for k in keys[start:start + BATCH_SIZE]]
        processed += len(batch)

        batch, toxic = filter_toxicity(batch)
        toxic_removed += len(toxic)
        if not batch:
            continue

        batch = add_prompt_categories(batch)
        for row in batch:
            score = row.get("category_score")  # set in zero_shot.py to enable the threshold
            low_score = score is not None and score < min_score
            if row.get("category") == "something else" or low_score:
                category_removed += 1
            else:
                accepted.setdefault(prompt_key(row), row)

        if len(accepted) >= limit:
            break

    accepted = dict(list(accepted.items())[:limit])
    stats = {
        "pool_duplicates_removed": duplicates_removed,
        "stage2_processed": processed,
        "toxic_removed": toxic_removed,
        "category_removed": category_removed,
        "accepted": len(accepted),
    }
    print(f"stage 2 done: {stats}")
    if len(accepted) < limit:
        print(
            f"WARNING: only {len(accepted)} prompts accepted (wanted {limit}). "
            "Increase --pool-size or loosen the keyword/threshold rules."
        )
    return accepted, stats


def main():
    parser = argparse.ArgumentParser(description="Filter WildChat prompts and write a seeded sample.")
    parser.add_argument("parquet_files", nargs="*")
    parser.add_argument("--limit", type=int, default=170, help="Final sample size.")
    parser.add_argument("--pool-size", type=int, default=3000,
                        help="Max unique keyword survivors to keep for the slow models.")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum zero-shot score (used only if rows have category_score).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="filtered.parquet", help="Full rows (for you).")
    parser.add_argument("--annotator-output", default="annotator_prompts.parquet",
                        help="ID + prompt only (for annotators).")
    args = parser.parse_args()

    files = args.parquet_files or find_parquet_files()
    if not files:
        raise FileNotFoundError("No parquet files found. Download WildChat or pass parquet paths.")

    pool, _ = build_candidate_pool(files, args.pool_size, args.seed)
    sample, _ = select_sample(pool, args.limit, args.seed, args.min_score)

    sample_rows = list(sample.values())
    assert len({prompt_key(r) for r in sample_rows}) == len(sample_rows), "duplicate prompts in sample"
    write_filtered(sample_rows, args.output)
    write_annotator_file(sample_rows, args.annotator_output)


if __name__ == "__main__":
    main()
