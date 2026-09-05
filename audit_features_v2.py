import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/ml_targets_v2.csv")

print("=" * 70)
print("ALPHALENS V2 FEATURE LEAKAGE AUDIT")
print("=" * 70)

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nRows   :", len(df))
print("Columns:", len(df.columns))
print(
    "Date   :",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

# ============================================================
# 1. TARGET COLUMNS
# ============================================================

TARGETS = [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Direction_1D",
    "Target_Direction_5D",
    "Target_Regime_1D",
    "Target_Volatility_20",
]

print("\n" + "=" * 70)
print("1. TARGET COLUMNS")
print("=" * 70)

for col in TARGETS:
    if col in df.columns:
        print(" -", col)

# ============================================================
# 2. POSSIBLE LEAKAGE BY COLUMN NAME
# ============================================================

print("\n" + "=" * 70)
print("2. POSSIBLE TARGET-LIKE FEATURES")
print("=" * 70)

suspicious = []

keywords = [
    "target",
    "future",
    "forward",
    "next",
    "lead"
]

for col in df.columns:
    if col in TARGETS:
        continue

    name = col.lower()

    if any(k in name for k in keywords):
        suspicious.append(col)

if suspicious:
    for col in suspicious:
        print("WARNING:", col)
else:
    print("PASS: No suspicious feature names.")

# ============================================================
# 3. NUMERIC FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("3. FEATURE/TARGET CORRELATION")
print("=" * 70)

feature_cols = [
    c for c in df.columns
    if c not in TARGETS + ["Date"]
]

numeric_features = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
]

corr = (
    df[numeric_features + ["Target_Direction_1D"]]
    .corr()["Target_Direction_1D"]
    .drop("Target_Direction_1D")
    .sort_values(key=abs, ascending=False)
)

print("\nTop 30 absolute correlations:")

print(
    corr.head(30).to_string()
)

# ============================================================
# 4. EXTREME CORRELATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. STRONG CORRELATION CHECK")
print("=" * 70)

strong = corr[abs(corr) >= 0.20]

if len(strong) == 0:
    print("PASS: No feature has |correlation| >= 0.20")
else:
    print("WARNING: Strong correlations found:")
    print(strong.to_string())

# ============================================================
# 5. CONSTANT / NEAR-CONSTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. LOW-VARIANCE FEATURES")
print("=" * 70)

low_variance = []

for col in numeric_features:
    unique = df[col].nunique(dropna=True)

    if unique <= 2:
        low_variance.append((col, unique))

if low_variance:
    for col, unique in low_variance:
        print(f"{col}: {unique} unique values")
else:
    print("PASS: No obvious constant features.")

# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. MISSING VALUES")
print("=" * 70)

missing = df[feature_cols].isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("PASS: No missing feature values.")
else:
    print(missing.to_string())

# ============================================================
# 7. NEWS FEATURES
# ============================================================

NEWS_FEATURES = [
    c for c in df.columns
    if "News" in c
]

print("\n" + "=" * 70)
print("7. NEWS FEATURES")
print("=" * 70)

print("News feature count:", len(NEWS_FEATURES))

for col in NEWS_FEATURES:
    print(" -", col)

# ============================================================
# 8. NEWS COVERAGE
# ============================================================

if "Has_News" in df.columns:

    print("\n" + "=" * 70)
    print("8. NEWS COVERAGE")
    print("=" * 70)

    coverage = (
        df.groupby(df["Date"].dt.year)
        .agg(
            Rows=("Date", "size"),
            News_Days=("Has_News", "sum"),
            Avg_News_Count=("News_Count", "mean"),
            Avg_Sentiment=("News_Sentiment", "mean")
        )
    )

    coverage["Coverage_%"] = (
        coverage["News_Days"]
        / coverage["Rows"]
        * 100
    )

    print(
        coverage.to_string()
    )

# ============================================================
# 9. NEWS / TARGET RELATIONSHIP
# ============================================================

if "Has_News" in df.columns:

    print("\n" + "=" * 70)
    print("9. TARGET RATE BY NEWS")
    print("=" * 70)

    print(
        df.groupby("Has_News")["Target_Direction_1D"]
        .agg(["count", "mean"])
        .to_string()
    )

# ============================================================
# 10. DATE ORDER CHECK
# ============================================================

print("\n" + "=" * 70)
print("10. TIME ORDER")
print("=" * 70)

is_sorted = df["Date"].is_monotonic_increasing

print("Dates sorted:", is_sorted)

duplicates = df["Date"].duplicated().sum()

print("Duplicate dates:", duplicates)

if is_sorted and duplicates == 0:
    print("PASS: Chronological dataset.")
else:
    print("WARNING: Date ordering problem.")

# ============================================================
# 11. FINAL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AUDIT COMPLETE")
print("=" * 70)

print("""
IMPORTANT:
This script checks statistical properties and obvious leakage.

It does NOT yet prove that news timestamps are leakage-free.

News timestamp validation is the next stage if publication
timestamps are available in the raw news dataset.
""")

print("PASS: Feature audit completed.")
print("=" * 70)