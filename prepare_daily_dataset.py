
import pandas as pd
from pathlib import Path

INPUT = Path("dataset/gold_daily_source.csv")
OUTPUT = Path("dataset/gold_prices_daily.csv")

# Read source data
df = pd.read_csv(INPUT)

# Select required columns
df = df[
    [
        "date",
        "24k_inr_per_gram",
        "gold_usd_per_troy_oz",
        "inr_per_usd"
    ]
].copy()

# Rename columns
df.columns = [
    "Date",
    "Gold_24K_1g",
    "Gold_USD_Ounce",
    "USD_INR"
]

# Convert dates
df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce"
)

# Convert prices to numbers
for col in ["Gold_24K_1g", "Gold_USD_Ounce", "USD_INR"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid records
df = df.dropna()
df = df[
    (df["Gold_24K_1g"] > 0) &
    (df["Gold_USD_Ounce"] > 0) &
    (df["USD_INR"] > 0)
]

# Remove duplicate dates and sort
df = df.drop_duplicates(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Check data frequency
df["Gap_Days"] = df["Date"].diff().dt.days

print("\nDATE GAP DISTRIBUTION:")
print(df["Gap_Days"].value_counts().sort_index().head(20))

print("\nLARGEST DATE GAP:")
print(df["Gap_Days"].max())

# Save original observations without filling gaps
df = df.drop(columns=["Gap_Days"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nDATASET PREPARATION SUCCESSFUL")
print("Total records:", len(df))
print("Start date:", df["Date"].min().date())
print("End date:", df["Date"].max().date())

print("\nFIRST FIVE RECORDS:")
print(df.head().to_string(index=False))

print("\nLAST FIVE RECORDS:")
print(df.tail().to_string(index=False))

print("\nSaved to:", OUTPUT)
