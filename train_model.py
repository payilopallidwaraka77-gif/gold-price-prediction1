
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("dataset/gold_prices_daily.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("GOLD PRICE PREDICTION - MODEL TRAINING")
print("=" * 55)

# Load data
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df["Gold_24K_1g"] = pd.to_numeric(
    df["Gold_24K_1g"], errors="coerce"
)

df = df.dropna(subset=["Date", "Gold_24K_1g"])
df = df[df["Gold_24K_1g"] > 0]
df = df.drop_duplicates(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Predict the next recorded observation using
# prices available up to the current observation.
df["Price_Current"] = df["Gold_24K_1g"]
df["Price_Lag1"] = df["Gold_24K_1g"].shift(1)
df["Price_Lag2"] = df["Gold_24K_1g"].shift(2)
df["Price_Lag3"] = df["Gold_24K_1g"].shift(3)
df["Price_Lag7"] = df["Gold_24K_1g"].shift(7)

df["MA_3"] = df["Gold_24K_1g"].rolling(3).mean()
df["MA_7"] = df["Gold_24K_1g"].rolling(7).mean()

df["Return_1"] = df["Gold_24K_1g"].pct_change()

# Next observed price is the target
df["Target"] = df["Gold_24K_1g"].shift(-1)
df["Next_Date"] = df["Date"].shift(-1)

# Keep pairs where next observation is within 7 days
df["Next_Gap"] = (
    df["Next_Date"] - df["Date"]
).dt.days

features = [
    "Price_Current",
    "Price_Lag1",
    "Price_Lag2",
    "Price_Lag3",
    "Price_Lag7",
    "MA_3",
    "MA_7",
    "Return_1"
]

# Save latest features for future prediction
latest_features = df[features].iloc[[-1]].copy()

# Prepare training examples
data = df.dropna(
    subset=features + ["Target", "Next_Gap"]
).copy()

data = data[
    (data["Next_Gap"] >= 1) &
    (data["Next_Gap"] <= 7)
].reset_index(drop=True)

if len(data) < 100:
    raise ValueError(
        "Insufficient training examples. "
        "Please obtain more historical observations."
    )

X = data[features]
y = data["Target"]

# Chronological 80/20 split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Baseline: next price equals current price
baseline_pred = X_test["Price_Current"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_pred
)

print(f"\nLast-price baseline MAE: Rs.{baseline_mae:.2f}")

best_model = None
best_name = None
best_mae = float("inf")

for name, model in models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"MAE  : Rs.{mae:.2f}")
    print(f"RMSE : Rs.{rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_model = model

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)

# Refit selected model using all available
# labelled examples after evaluating the test set
best_model.fit(X, y)

joblib.dump(
    {
        "model": best_model,
        "features": features,
        "model_name": best_name,
        "test_mae": best_mae,
        "baseline_mae": baseline_mae
    },
    MODEL_DIR / "gold_price_model.pkl"
)

# Estimate the next observed gold price
prediction = best_model.predict(latest_features)[0]

print("\n" + "=" * 55)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nBest ML model:", best_name)
print(f"Test MAE: Rs.{best_mae:.2f}")
print(f"Baseline MAE: Rs.{baseline_mae:.2f}")

print("\nLatest available date:", df["Date"].iloc[-1].date())
print(f"Latest gold price: Rs.{df['Gold_24K_1g'].iloc[-1]:.2f}/gram")
print(f"Next observed price estimate: Rs.{prediction:.2f}/gram")
print("Exact next observation date is not known.")

if best_mae >= baseline_mae:
    print(
        "\nWarning: Selected ML model did not beat "
        "the last-price baseline on the test set."
    )

print("\nModel saved to models/gold_price_model.pkl")
print("Comparison saved to models/model_comparison.csv")
