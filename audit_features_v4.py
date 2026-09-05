import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ALPHALENS V4 FEATURE LEAKAGE AUDIT
# ============================================================

INPUT = Path("data/ml_features_v4.csv")

print("=" * 70)
print("ALPHALENS V4 FEATURE LEAKAGE AUDIT")
print("=" * 70)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing: {INPUT}")

df = pd.read_csv(INPUT)

print("\nDATASET")
print("-" * 70)
print("Rows   :", len(df))
print("Columns:", len(df))

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    print(
        "Date   :",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )

# ============================================================
# 1. TARGET-LIKE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("1. TARGET-LIKE / FUTURE-LOOKING FEATURE NAMES")
print("=" * 70)

target_words = [
    "target",
    "future",
    "forward",
    "next",
    "label",
    "y_true",
    "outcome"
]

suspects = []

for col in df.columns:
    name = col.lower()

    if any(word in name for word in target_words):
        suspects.append(col)

if suspects:
    for col in suspects:
        print("WARNING:", col)
else:
    print("PASS: No target-like feature names found.")

# ============================================================
# 2. DUPLICATES / CHRONOLOGY
# ============================================================

print("\n" + "=" * 70)
print("2. CHRONOLOGY")
print("=" * 70)

if "Date" in df.columns:

    sorted_ok = df["Date"].is_monotonic_increasing
    duplicates = df["Date"].duplicated().sum()

    print("Dates sorted :", sorted_ok)
    print("Duplicate dates:", duplicates)

    if sorted_ok and duplicates == 0:
        print("PASS: Chronological dataset.")
    else:
        print("WARNING: Chronology problem detected.")

# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("PASS: No missing values.")
else:
    print("WARNING: Missing values:")
    print(missing.to_string())

# ============================================================
# 4. CONSTANT / LOW VARIANCE
# ============================================================

print("\n" + "=" * 70)
print("4. LOW-VARIANCE FEATURES")
print("=" * 70)

feature_cols = [
    c for c in df.columns
    if c != "Date"
]

low_variance = []

for col in feature_cols:

    if not pd.api.types.is_numeric_dtype(df[col]):
        continue

    unique = df[col].nunique(dropna=False)

    if unique <= 2:
        low_variance.append((col, unique))

if low_variance:

    for col, unique in low_variance:
        print(f"{col}: {unique} unique values")

else:
    print("PASS: No binary/constant features detected.")

# ============================================================
# 5. INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("5. INFINITE VALUES")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns

inf_counts = np.isinf(df[numeric_cols]).sum()
inf_counts = inf_counts[inf_counts > 0]

if len(inf_counts) == 0:
    print("PASS: No infinite values.")
else:
    print("WARNING:")
    print(inf_counts.to_string())

# ============================================================
# 6. NUMERIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURE TYPES")
print("=" * 70)

non_numeric = []

for col in feature_cols:

    if not pd.api.types.is_numeric_dtype(df[col]):
        non_numeric.append(col)

if non_numeric:

    print("WARNING: Non-numeric columns:")
    for col in non_numeric:
        print(" -", col)

else:
    print("PASS: All features are numeric.")

# ============================================================
# 7. STRONG CORRELATION WITH NEXT-DAY RETURN
# ============================================================

print("\n" + "=" * 70)
print("7. CORRELATION AUDIT")
print("=" * 70)

# Reconstruct next-day return from Close.
# This is ONLY an audit variable and is NOT used as a feature.

if "Close" in df.columns:

    audit_target = df["Close"].shift(-1) / df["Close"] - 1

    correlations = {}

    for col in numeric_cols:

        if col == "Close":
            continue

        x = df[col]

        valid = x.notna() & audit_target.notna()

        if valid.sum() < 30:
            continue

        corr = x[valid].corr(audit_target[valid])

        if pd.notna(corr):
            correlations[col] = corr

    corr_series = (
        pd.Series(correlations)
        .abs()
        .sort_values(ascending=False)
    )

    print("\nTop 30 absolute correlations with NEXT-DAY return:")

    for col in corr_series.head(30).index:
        raw_corr = correlations[col]
        print(f"{col:35s} {raw_corr: .6f}")

    strong = corr_series[corr_series >= 0.20]

    print("\nStrong correlation threshold: |corr| >= 0.20")

    if len(strong) == 0:
        print("PASS: No feature has |correlation| >= 0.20.")
    else:
        print("WARNING: Strongly correlated features:")
        for col in strong.index:
            print(
                f" - {col}: "
                f"{correlations[col]:.6f}"
            )

# ============================================================
# 8. KNOWN DANGEROUS FEATURES
# ============================================================

print("\n" + "=" * 70)
print("8. DANGEROUS FEATURE PATTERN CHECK")
print("=" * 70)

danger_patterns = [
    "future",
    "forward",
    "next_return",
    "future_return",
    "target_",
    "y_true",
    "label"
]

dangerous = []

for col in feature_cols:

    name = col.lower()

    for pattern in danger_patterns:

        if pattern in name:
            dangerous.append(col)
            break

if dangerous:

    print("WARNING: Potentially dangerous features:")

    for col in sorted(set(dangerous)):
        print(" -", col)

else:
    print("PASS: No dangerous naming patterns detected.")

# ============================================================
# 9. ROLLING FEATURE NAME AUDIT
# ============================================================

print("\n" + "=" * 70)
print("9. ROLLING / LOOKBACK FEATURE AUDIT")
print("=" * 70)

rolling_keywords = [
    "rolling",
    "sma",
    "ema",
    "volatility",
    "zscore",
    "percentile",
    "mean_",
    "std_",
    "skew",
    "kurtosis"
]

rolling_features = []

for col in feature_cols:

    name = col.lower()

    if any(k in name for k in rolling_keywords):
        rolling_features.append(col)

print("Rolling/statistical features:", len(rolling_features))

for col in rolling_features:
    print(" -", col)

# ============================================================
# 10. PRICE FEATURE SANITY
# ============================================================

print("\n" + "=" * 70)
print("10. PRICE SANITY")
print("=" * 70)

price_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in price_cols:

    if col in df.columns:

        print(
            f"{col:10s}",
            "min=",
            df[col].min(),
            "max=",
            df[col].max()
        )

if all(c in df.columns for c in ["High", "Low", "Open", "Close"]):

    bad_high = (
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["High"] < df["Low"])
    ).sum()

    bad_low = (
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"]) |
        (df["Low"] > df["High"])
    ).sum()

    print("\nInvalid OHLC rows:")
    print("High violations:", bad_high)
    print("Low violations :", bad_low)

    if bad_high == 0 and bad_low == 0:
        print("PASS: OHLC relationships are valid.")
    else:
        print("WARNING: Invalid OHLC relationships found.")

# ============================================================
# 11. NEWS FEATURES
# ============================================================

print("\n" + "=" * 70)
print("11. NEWS FEATURES")
print("=" * 70)

news_cols = [
    c for c in df.columns
    if "news" in c.lower()
]

print("News feature count:", len(news_cols))

for col in news_cols:
    print(" -", col)

if news_cols:

    print("\nNews coverage:")

    for col in news_cols:

        if pd.api.types.is_numeric_dtype(df[col]):

            nonzero = (df[col].fillna(0) != 0).sum()

            if nonzero > 0:
                print(
                    f"{col:35s}"
                    f" nonzero={nonzero}"
                    f" ({nonzero / len(df) * 100:.2f}%)"
                )

# ============================================================
# 12. FEATURE / TARGET SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("12. FEATURE / TARGET SEPARATION")
print("=" * 70)

target_candidates = [
    c for c in df.columns
    if c.lower().startswith("target")
]

if target_candidates:

    print("Target-like columns present:")
    for col in target_candidates:
        print(" -", col)

    print(
        "\nIMPORTANT: These columns must NOT be passed "
        "to model training as features."
    )

else:

    print(
        "No target columns found in V4 feature file."
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("V4 FEATURE AUDIT COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:\n"
    "This audit checks statistical and structural leakage.\n"
    "It cannot prove timestamp-level news leakage without\n"
    "raw publication timestamps."
)

print("\nPASS: V4 audit completed.")
print("=" * 70)