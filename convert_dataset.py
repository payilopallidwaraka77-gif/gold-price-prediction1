
import pandas as pd
from pathlib import Path

# File locations
INPUT_FILE = Path("dataset/GOLD_INR.parquet")
OUTPUT_FILE = Path("dataset/gold_prices_clean.csv")

def convert_dataset():

    print("\nGOLD PRICE DATASET PREPARATION")
    print("-" * 40)

    # Load Parquet dataset
    df = pd.read_parquet(INPUT_FILE)

    # Date is stored as the index in this dataset
    df = df.reset_index()

    # Standardize date column name
    df = df.rename(columns={df.columns[0]: "Date"})

    # Select 24K gold price per gram
    df = df[["Date", "Gold_24K_1g"]].copy()

    # Clean and convert data types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Gold_24K_1g"] = pd.to_numeric(
        df["Gold_24K_1g"], errors="coerce"
    )

    # Remove invalid records
    df = df.dropna()
    df = df[df["Gold_24K_1g"] > 0]

    # Remove duplicate dates and sort
    df = df.drop_duplicates(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # Display dataset information
    print("\nConversion successful!")
    print("Total records:", len(df))
    print("Start date:", df["Date"].min().date())
    print("End date:", df["Date"].max().date())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nLast five records:")
    print(df.tail().to_string(index=False))

    # Check date intervals
    gaps = df["Date"].diff().dt.days

    print("\nDate interval distribution:")
    print(gaps.value_counts().sort_index().head(15))

    daily_count = ((gaps >= 1) & (gaps <= 7)).sum()

    print("\nObservations following a gap of 1-7 days:", daily_count)
    print("\nSaved file:", OUTPUT_FILE)

if __name__ == "__main__":
    convert_dataset()
