import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ALPHALENS V3 TARGET LEAKAGE AUDIT
# ============================================================

INPUT = Path("data/ml_targets_v3.csv")

print("=" * 70)
print("ALPHALENS V3 TARGET LEAKAGE AUDIT")
print("=" * 70)

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values("Date").reset_index(drop=True)

print("\nDATASET")
print("-" * 70)
print("Rows   :", len(df))
print("Columns:", len(df.columns))
print(
    "Date   :",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

# ============================================================
# TARGET COLUMNS
# ============================================================

target_cols = [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Volatility_20",
    "Target_Return_Vol_Adj",
    "Target_Direction_V3",
    "Target_Up_V3",
    "Target_Direction_5D_V3",
    "Target_Direction_10D_V3",
    "Target_Excess_Return",
]

print("\n" + "=" * 70)
print("1. TARGET COLUMNS")
print("=" * 70)

for col in target_cols:

    if col in df.columns:
        print(" -", col)
    else:
        print("MISSING:", col)

# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET-LIKE COLUMNS")
print("=" * 70)

suspicious = []

for col in df.columns:

    if col == "Date":
        continue

    if (
        "Target" in col
        or "Future" in col
        or "Forward" in col
    ):
        if col not in target_cols:
            suspicious.append(col)

if suspicious:

    print("WARNING: possible target-like columns")

    for col in suspicious:
        print(" -", col)

else:

    print("PASS: No unexpected target-like columns.")

# ============================================================
# TARGET MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. TARGET MISSING VALUES")
print("=" * 70)

missing = df[target_cols].isna().sum()

print(missing.to_string())

if missing.sum() == 0:
    print("\nPASS: No missing target values.")

# ============================================================
# CHRONOLOGY
# ============================================================

print("\n" + "=" * 70)
print("4. CHRONOLOGY")
print("=" * 70)

print(
    "Dates sorted:",
    df["Date"].is_monotonic_increasing
)

print(
    "Duplicate dates:",
    df["Date"].duplicated().sum()
)

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. TARGET DISTRIBUTIONS")
print("=" * 70)

for col in [
    "Target_Direction_V3",
    "Target_Direction_5D_V3",
    "Target_Direction_10D_V3",
    "Target_Up_V3",
]:

    if col in df.columns:

        print("\n" + col)

        print(
            df[col]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

# ============================================================
# RETURN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("6. FUTURE RETURN STATISTICS")
print("=" * 70)

for col in [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
]:

    print("\n" + col)

    print(
        df[col]
        .describe()
        .to_string()
    )

# ============================================================
# VOLATILITY
# ============================================================

print("\n" + "=" * 70)
print("7. VOLATILITY VALIDATION")
print("=" * 70)

print(
    df["Target_Volatility_20"]
    .describe()
    .to_string()
)

if (df["Target_Volatility_20"] <= 0).any():

    print(
        "\nWARNING: Non-positive volatility detected."
    )

else:

    print(
        "\nPASS: Volatility values are positive."
    )

# ============================================================
# MATHEMATICAL CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("8. MATHEMATICAL CONSISTENCY")
print("=" * 70)

expected_vol_adj = (
    df["Target_Return_1D"] /
    df["Target_Volatility_20"]
)

difference = (
    df["Target_Return_Vol_Adj"] -
    expected_vol_adj
).abs()

max_difference = difference.max()

print(
    "Max volatility-adjustment difference:",
    max_difference
)

if max_difference < 1e-10:

    print(
        "PASS: Volatility-adjusted target is mathematically consistent."
    )

else:

    print(
        "WARNING: Volatility-adjusted target mismatch."
    )

# ============================================================
# IMPORTANT LEAKAGE TEST
#
# Target_Volatility_20 must be based only on historical returns.
#
# Reconstruct it directly from Close and compare.
# ============================================================

print("\n" + "=" * 70)
print("9. HISTORICAL VOLATILITY RECONSTRUCTION")
print("=" * 70)

if "Close" in df.columns:

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    historical_return = close.pct_change()

    reconstructed_vol = (
        historical_return
        .rolling(20)
        .std()
    )

    difference = (
        df["Target_Volatility_20"] -
        reconstructed_vol
    ).abs()

    valid = difference.dropna()

    if len(valid) > 0:

        print(
            "Max difference:",
            valid.max()
        )

        print(
            "Mean difference:",
            valid.mean()
        )

        if valid.max() < 1e-10:

            print(
                "PASS: Volatility uses historical information."
            )

        else:

            print(
                "WARNING: Volatility reconstruction mismatch."
            )

else:

    print(
        "WARNING: Close column unavailable for reconstruction."
    )

# ============================================================
# TARGET / CURRENT PRICE SANITY
# ============================================================

print("\n" + "=" * 70)
print("10. TARGET HORIZON SANITY")
print("=" * 70)

if "Close" in df.columns:

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    expected_1d = close.shift(-1) / close - 1

    diff = (
        df["Target_Return_1D"] -
        expected_1d
    ).abs()

    valid = diff.dropna()

    print(
        "1D return max difference:",
        valid.max()
    )

    if valid.max() < 1e-10:

        print(
            "PASS: 1D target correctly represents next-day return."
        )

    else:

        print(
            "WARNING: 1D target mismatch."
        )

# ============================================================
# EXTREME TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("11. EXTREME TARGET CHECK")
print("=" * 70)

for col in [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
]:

    values = df[col].dropna()

    print(
        f"{col}:",
        "min=",
        round(values.min(), 6),
        "max=",
        round(values.max(), 6)
    )

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    "\nV3 target dataset is ready for model experimentation "
    "if all PASS checks above succeed."
)

print("=" * 70)