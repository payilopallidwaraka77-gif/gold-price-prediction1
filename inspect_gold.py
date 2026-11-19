
import pandas as pd

df = pd.read_parquet("dataset/GOLD_INR.parquet")

print("Total records:", len(df))
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nDate range:")
print(df.index.min(), "to", df.index.max())

print("\n24K Gold Price per Gram:")
print(df["Gold_24K_1g"].describe())
