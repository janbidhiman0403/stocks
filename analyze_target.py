import pandas as pd
import numpy as np

# Load the regression dataset
df = pd.read_csv(
    "data/ml_ready_TCS_regression.csv",
    parse_dates=["Date"]
)

returns = df["Future_Return_5D"]

print("\n========== TARGET ANALYSIS ==========")

print(f"Number of observations: {len(returns)}")

print("\nBasic Statistics:")
print(f"Mean:   {returns.mean() * 100:.2f}%")
print(f"Median: {returns.median() * 100:.2f}%")
print(f"Std:    {returns.std() * 100:.2f}%")
print(f"Min:    {returns.min() * 100:.2f}%")
print(f"Max:    {returns.max() * 100:.2f}%")

# -----------------------------
# Return distribution
# -----------------------------

print("\n========== RETURN DISTRIBUTION ==========")

thresholds = [
    (-0.10, "Below -10%"),
    (-0.05, "-10% to -5%"),
    (-0.02, "-5% to -2%"),
    (0.00, "-2% to 0%"),
    (0.02, "0% to +2%"),
    (0.05, "+2% to +5%"),
    (float("inf"), "Above +5%")
]

lower = -float("inf")

for upper, label in thresholds:
    count = ((returns > lower) & (returns <= upper)).sum()
    percentage = count / len(returns) * 100

    print(
        f"{label:15s} | "
        f"{count:4d} observations | "
        f"{percentage:6.2f}%"
    )

    lower = upper


# -----------------------------
# Classification thresholds
# -----------------------------

print("\n========== ±2% CLASSIFICATION ==========")

avoid = (returns <= -0.02).sum()
hold = ((returns > -0.02) & (returns < 0.02)).sum()
buy = (returns >= 0.02).sum()

print(f"AVOID (<= -2%): {avoid} ({avoid / len(returns) * 100:.2f}%)")
print(f"HOLD  (-2% to +2%): {hold} ({hold / len(returns) * 100:.2f}%)")
print(f"BUY   (>= +2%): {buy} ({buy / len(returns) * 100:.2f}%)")


# -----------------------------
# Alternative thresholds
# -----------------------------

print("\n========== ALTERNATIVE THRESHOLDS ==========")

for threshold in [0.01, 0.015, 0.02, 0.025, 0.03]:

    negative = (returns <= -threshold).sum()
    neutral = (
        (returns > -threshold) &
        (returns < threshold)
    ).sum()
    positive = (returns >= threshold).sum()

    print(
        f"±{threshold * 100:.1f}% | "
        f"Negative: {negative:4d} | "
        f"Neutral: {neutral:4d} | "
        f"Positive: {positive:4d}"
    )


# -----------------------------
# Percentiles
# -----------------------------

print("\n========== RETURN PERCENTILES ==========")

percentiles = [5, 10, 25, 50, 75, 90, 95]

for p in percentiles:
    value = np.percentile(returns, p)

    print(
        f"{p:2d}th percentile: "
        f"{value * 100:.2f}%"
    )

print("========================================\n")