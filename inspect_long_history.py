
import pandas as pd

path = "dataset/gold_daily_source.csv"

df = pd.read_csv(path)

print("TOTAL RECORDS:", len(df))
print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nLAST 5 ROWS:")
print(df.tail())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)
