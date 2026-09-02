import pandas as pd
import numpy as np

# Load final test predictions
df = pd.read_csv(
    "data/xgboost_regression_final_test.csv",
    parse_dates=["Date"]
)

# -----------------------------
# Basic prediction statistics
# -----------------------------

prediction = df["Predicted_Return"]
actual = df["Future_Return_5D"]

print("\n========== PREDICTION SIGNAL ANALYSIS ==========")

print(f"Number of predictions: {len(df)}")

print("\nPrediction Statistics:")
print(f"Mean predicted return: {prediction.mean() * 100:.2f}%")
print(f"Mean actual return: {actual.mean() * 100:.2f}%")
print(f"Prediction std: {prediction.std() * 100:.2f}%")
print(f"Actual std: {actual.std() * 100:.2f}%")

# Correlation
correlation = prediction.corr(actual)

print(f"\nPrediction / Actual Correlation: {correlation:.4f}")


# -----------------------------
# Prediction buckets
# -----------------------------
# Divide predictions into five groups.
# This tests whether stronger predictions
# correspond to better actual outcomes.

df["Prediction_Bucket"] = pd.qcut(
    df["Predicted_Return"],
    q=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    duplicates="drop"
)

bucket_analysis = (
    df.groupby("Prediction_Bucket", observed=True)
    .agg(
        Count=("Future_Return_5D", "count"),
        Average_Predicted=("Predicted_Return", "mean"),
        Average_Actual=("Future_Return_5D", "mean"),
        Median_Actual=("Future_Return_5D", "median")
    )
)

print("\n========== PREDICTION BUCKET ANALYSIS ==========")

for bucket, row in bucket_analysis.iterrows():
    print(
        f"{bucket:10s} | "
        f"Count: {int(row['Count']):3d} | "
        f"Predicted: {row['Average_Predicted'] * 100:6.2f}% | "
        f"Actual: {row['Average_Actual'] * 100:6.2f}% | "
        f"Median: {row['Median_Actual'] * 100:6.2f}%"
    )


# -----------------------------
# Top prediction signals
# -----------------------------

top_10 = df.nlargest(
    max(1, int(len(df) * 0.10)),
    "Predicted_Return"
)

top_10_avg_actual = top_10["Future_Return_5D"].mean()
top_10_win_rate = (
    (top_10["Future_Return_5D"] > 0).mean() * 100
)

print("\n========== TOP 10% SIGNALS ==========")

print(f"Signals: {len(top_10)}")
print(f"Average predicted return: {top_10['Predicted_Return'].mean() * 100:.2f}%")
print(f"Average actual return: {top_10_avg_actual * 100:.2f}%")
print(f"Win rate: {top_10_win_rate:.2f}%")

print("\n===============================================\n")