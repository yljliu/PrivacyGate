import ast

import pandas as pd


def convert_parquet_to_csv():
    # Load the Parquet file
    df = pd.read_parquet('filtered.parquet')

    # Save as CSV (index=False prevents an extra row-number column)
    df.to_csv('x.csv', index=False)


def fetch_content_from_Csv(input_file='output.csv', output_file='conversations.csv'):
    df = pd.read_csv(input_file)
    conversation_column = next(
        (column for column in ('conversations', 'conversation') if column in df.columns),
        None,
    )
    if conversation_column is None:
        raise ValueError("CSV must contain a 'conversations' or 'conversation' column")

    def extract_content(value):
        if pd.isna(value):
            return ''
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError('Conversation entries must be dictionaries or lists of dictionaries')
        return '\n'.join(
            turn['content'] for turn in value
            if isinstance(turn, dict) and 'content' in turn
        )

    df[conversation_column] = df[conversation_column].map(extract_content)
    df.to_csv(output_file, index=False)
    return df



convert_parquet_to_csv()