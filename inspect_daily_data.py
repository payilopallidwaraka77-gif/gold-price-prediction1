
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/gold_prices_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Display recent observations
print("\nRECENT GOLD PRICE DATA")
print(df.tail(20).to_string(index=False))

# Calculate date gaps
df["Gap_Days"] = df["Date"].diff().dt.days

print("\nDATE GAP ANALYSIS")
print(df["Gap_Days"].value_counts().sort_index())

# Display observations with short date gaps
daily_data = df[
    (df["Gap_Days"] >= 1) &
    (df["Gap_Days"] <= 7)
]

print("\nOBSERVATIONS WITH 1-7 DAY GAPS:")
print(daily_data.to_string(index=False))

print("\nTotal observations:", len(daily_data))
