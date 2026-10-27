import pandas as pd

# Load the Parquet file
df = pd.read_parquet('filtered.parquet')

# Save as CSV (index=False prevents an extra row-number column)
df.to_csv('output.csv', index=False)