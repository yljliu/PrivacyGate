import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from PrivacyGate.rules import classify, find_parquet_files, first_prompt, load_sample, RULES


def preview(rows, count=30, seed=42):
    random.seed(seed)
    selected = random.sample(rows, min(count, len(rows)))
    for row, hits in selected:
        print(hits, "|", first_prompt(row)[:200].replace("\n", " "))


def write_filtered(rows, output_path):
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    print(f"wrote {len(rows)} filtered rows to {output_path}")


def load_filtered_sample(limit=170, seed=42, parquet_files=None):
    files = parquet_files or find_parquet_files()
    if not files:
        raise FileNotFoundError("No parquet files found. Download WildChat or pass parquet paths.")

    rng = random.Random(seed)
    dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)
    reservoir = []
    eligible_count = 0

    for row in dataset:
        _, _, dropped = classify([row])
        if dropped:
            continue
        eligible_count += 1
        if len(reservoir) < limit:
            reservoir.append(row)
        else:
            replacement = rng.randint(0, eligible_count - 1)
            if replacement < limit:
                reservoir[replacement] = row

    rng.shuffle(reservoir)
    print(f"found {eligible_count} prompts passing all {len(RULES)} rules")
    return reservoir


def main():
    parser = argparse.ArgumentParser(description="Preview prompts matched by filtering rules.")
    parser.add_argument("parquet_files", nargs="*")
    parser.add_argument("--limit", type=int, default=170)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="filtered.parquet", help="Write prompts that pass all rules to this parquet file.")
    args = parser.parse_args()

    kept = load_filtered_sample(args.limit, args.seed, args.parquet_files or None)
    preview([(row, []) for row in kept], seed=args.seed)
    write_filtered(kept, args.output)


if __name__ == "__main__":
    main()