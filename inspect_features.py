import pandas as pd
import numpy as np

# Load the exact dataset used for regression
df = pd.read_csv(
    "data/ml_ready_TCS_regression.csv",
    parse_dates=["Date"]
)

print("\n========== FEATURE INSPECTION ==========")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumn Names:")
for column in df.columns:
    print(f"- {column}")

# -----------------------------
# Missing values
# -----------------------------

print("\nMissing Values:")

missing = df.isnull().sum()

for column, value in missing.items():
    if value > 0:
        print(f"{column}: {value}")

if missing.sum() == 0:
    print("No missing values found.")


# -----------------------------
# Infinite values
# -----------------------------

numeric_df = df.select_dtypes(include=np.number)

infinite_count = np.isinf(numeric_df).sum().sum()

print(f"\nInfinite numeric values: {infinite_count}")


# -----------------------------
# Feature statistics
# -----------------------------

print("\n========== NUMERIC FEATURE RANGES ==========")

for column in numeric_df.columns:

    minimum = numeric_df[column].min()
    maximum = numeric_df[column].max()
    mean = numeric_df[column].mean()
    std = numeric_df[column].std()

    print(
        f"{column:25s} | "
        f"Min: {minimum:12.4f} | "
        f"Max: {maximum:12.4f} | "
        f"Mean: {mean:12.4f} | "
        f"Std: {std:12.4f}"
    )


# -----------------------------
# Correlation with target
# -----------------------------

print("\n========== CORRELATION WITH TARGET ==========")

target = "Future_Return_5D"

correlations = (
    numeric_df
    .corr()[target]
    .drop(target)
    .sort_values(ascending=False)
)

for feature, correlation in correlations.items():
    print(f"{feature:25s} | {correlation: .4f}")


# -----------------------------
# Duplicate / constant features
# -----------------------------

print("\n========== FEATURE QUALITY CHECK ==========")

constant_features = []

for column in numeric_df.columns:
    if numeric_df[column].nunique() <= 1:
        constant_features.append(column)

if constant_features:
    print("Constant features:")
    for feature in constant_features:
        print(f"- {feature}")
else:
    print("No constant features found.")


# -----------------------------
# Final summary
# -----------------------------

print("\n========== SUMMARY ==========")

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Missing values: {missing.sum()}")
print(f"Infinite values: {infinite_count}")
print(
    f"Target mean: "
    f"{df[target].mean() * 100:.2f}%"
)
print(
    f"Target standard deviation: "
    f"{df[target].std() * 100:.2f}%"
)

print("============================================\n")