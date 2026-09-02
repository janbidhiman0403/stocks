import pandas as pd
import numpy as np

# Load the untouched V3 final-test predictions
df = pd.read_csv("data/xgboost_regression_final_test.csv")

df["Date"] = pd.to_datetime(df["Date"])

pred = df["Predicted_Return"]
actual = df["Future_Return_5D"]

print("V3 Signal Analysis")
print("------------------")
print("Rows:", len(df))
print()

# Basic signal statistics
print("Prediction statistics:")
print(f"Mean predicted return: {pred.mean() * 100:.2f}%")
print(f"Mean actual return:    {actual.mean() * 100:.2f}%")
print(f"Prediction std:        {pred.std() * 100:.2f}%")
print(f"Actual std:            {actual.std() * 100:.2f}%")
print()

# Prediction buckets
df["Signal"] = pd.cut(
    pred,
    bins=[-np.inf, -0.02, 0, 0.02, np.inf],
    labels=[
        "Strong Negative",
        "Weak Negative",
        "Weak Positive",
        "Strong Positive"
    ]
)

print("Signal bucket performance")
print("-------------------------")

summary = df.groupby(
    "Signal",
    observed=False
).agg(
    Signals=("Future_Return_5D", "count"),
    Avg_Predicted=("Predicted_Return", "mean"),
    Avg_Actual=("Future_Return_5D", "mean"),
    Median_Actual=("Future_Return_5D", "median"),
    Positive_Rate=(
        "Future_Return_5D",
        lambda x: (x > 0).mean()
    )
)

summary["Avg_Predicted"] *= 100
summary["Avg_Actual"] *= 100
summary["Median_Actual"] *= 100
summary["Positive_Rate"] *= 100

print(
    summary.to_string(
        formatters={
            "Avg_Predicted": "{:.2f}%".format,
            "Avg_Actual": "{:.2f}%".format,
            "Median_Actual": "{:.2f}%".format,
            "Positive_Rate": "{:.2f}%".format,
        }
    )
)

print()

# Test different positive prediction thresholds
print("Positive signal thresholds")
print("--------------------------")

thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

for threshold in thresholds:
    signals = df[df["Predicted_Return"] >= threshold]

    if len(signals) == 0:
        continue

    avg_actual = signals["Future_Return_5D"].mean()
    win_rate = (
        signals["Future_Return_5D"] > 0
    ).mean()

    print(
        f">= {threshold * 100:.1f}%: "
        f"{len(signals)} signals | "
        f"Actual avg: {avg_actual * 100:.2f}% | "
        f"Win rate: {win_rate * 100:.2f}%"
    )

print()

# Top prediction decile
top_10_cutoff = pred.quantile(0.90)
top_10 = df[pred >= top_10_cutoff]

print("Top 10% predictions")
print("-------------------")
print(f"Cutoff: {top_10_cutoff * 100:.2f}%")
print(f"Signals: {len(top_10)}")
print(
    f"Actual average: "
    f"{top_10['Future_Return_5D'].mean() * 100:.2f}%"
)
print(
    f"Positive rate: "
    f"{(top_10['Future_Return_5D'] > 0).mean() * 100:.2f}%"
)
