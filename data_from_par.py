import pandas as pd

df = pd.read_parquet("0000.parquet")

print(df.head())
print(len(df))