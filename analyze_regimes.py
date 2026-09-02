import pandas as pd
import numpy as np

# Load final test predictions
df = pd.read_csv(
    "data/xgboost_regression_final_test.csv",
    parse_dates=["Date"]
)

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

# Calculate market regime using 20-day price momentum.
# This uses only information available up to each date.
df["Market_Return_20D"] = (
    df["Close"] / df["Close"].shift(20) - 1
)

# Define simple regimes
def classify_regime(value):
    if pd.isna(value):
        return "Unknown"
    elif value >= 0.05:
        return "Strong Up"
    elif value >= 0.00:
        return "Up"
    elif value <= -0.05:
        return "Strong Down"
    else:
        return "Down"

df["Regime"] = df["Market_Return_20D"].apply(classify_regime)

# Remove the initial rows where 20-day momentum cannot be calculated
df = df[df["Regime"] != "Unknown"].copy()

print("\n========== MARKET REGIME ANALYSIS ==========")

print(f"Observations analyzed: {len(df)}")

# -----------------------------
# Performance by regime
# -----------------------------

print("\n========== PERFORMANCE BY REGIME ==========")

for regime in ["Strong Down", "Down", "Up", "Strong Up"]:

    subset = df[df["Regime"] == regime]

    if len(subset) == 0:
        continue

    actual = subset["Future_Return_5D"]
    predicted = subset["Predicted_Return"]

    correlation = predicted.corr(actual)

    positive_rate = (actual > 0).mean() * 100

    print(f"\n{regime}")
    print(f"Observations: {len(subset)}")
    print(f"Average predicted return: {predicted.mean() * 100:.2f}%")
    print(f"Average actual 5D return: {actual.mean() * 100:.2f}%")
    print(f"Median actual 5D return: {actual.median() * 100:.2f}%")
    print(f"Positive-return rate: {positive_rate:.2f}%")
    print(f"Prediction/actual correlation: {correlation:.4f}")


# -----------------------------
# Overall comparison
# -----------------------------

print("\n========== REGIME SUMMARY ==========")

summary = (
    df.groupby("Regime", observed=True)
    .agg(
        Observations=("Future_Return_5D", "count"),
        Avg_Predicted=("Predicted_Return", "mean"),
        Avg_Actual=("Future_Return_5D", "mean"),
        Median_Actual=("Future_Return_5D", "median")
    )
)

for regime, row in summary.iterrows():

    print(
        f"{regime:12s} | "
        f"Count: {int(row['Observations']):3d} | "
        f"Predicted: {row['Avg_Predicted'] * 100:6.2f}% | "
        f"Actual: {row['Avg_Actual'] * 100:6.2f}% | "
        f"Median: {row['Median_Actual'] * 100:6.2f}%"
    )

print("\n===========================================\n")