import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import pandas as pd

MODELS = ("llama3:8b", "qwen3.5:4b", "llama3.2:1b")
OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(model, value):
    payload = json.dumps({
        "model": model,
        "prompt": str(value),
        "stream": True
    }).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    full_response = []

    with urlopen(request, timeout=900) as response:
        for line in response:
            if not line.strip():
                continue

            chunk = json.loads(line.decode("utf-8"))

            # Ollama sends generated text in the "response" field
            text = chunk.get("response", "")
            print(text, end="", flush=True)
            full_response.append(text)

            if chunk.get("done", False):
                break

    print()  # Newline after the completed response
    return "".join(full_response).strip()


def ask_csv():
    df = pd.read_csv("untitled.csv", header=None)

    for model in MODELS:
        print(f"\n--- Model: {model} ---")

        generate_response(model, df.iloc[0, 0])


ask_csv()



# def process_csv(input_path, output_path, column, prompt_template):
#     with input_path.open("r", newline="", encoding="utf-8-sig") as source:
#         reader = csv.DictReader(source)
#         if not reader.fieldnames:
#             raise ValueError("The input CSV must have a header row.")

#         selected_column = column or reader.fieldnames[0]
#         if selected_column not in reader.fieldnames:
#             raise ValueError(
#                 f"Column '{selected_column}' not found. Available columns: "
#                 + ", ".join(reader.fieldnames)
#             )

#         rows = list(reader)
#         fieldnames = list(reader.fieldnames)

#     if input_path.resolve() == output_path.resolve():
#         raise ValueError("The output path must be different from the input path.")

#     response_columns = {model: f"{model}_response" for model in MODELS}
#     for response_column in response_columns.values():
#         if response_column not in fieldnames:
#             fieldnames.append(response_column)

#     for index, row in enumerate(rows, start=1):
#         value = row.get(selected_column, "")
#         for model in MODELS:
#             row[response_columns[model]] = generate_response(
#                 model, value, prompt_template
#             )
#             print(f"Processed row {index}/{len(rows)} with {model}.")

#     with output_path.open("w", newline="", encoding="utf-8") as destination:
#         writer = csv.DictWriter(destination, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)


# def main():





# if __name__ == "__main__":
#     raise SystemExit(main())