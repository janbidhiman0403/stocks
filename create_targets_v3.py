import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ALPHALENS V3 TARGET ENGINE
# Volatility-adjusted targets
# ============================================================

INPUT = Path("data/ml_ready_TCS_news_clean.csv")
OUTPUT = Path("data/ml_targets_v3.csv")

print("=" * 70)
print("ALPHALENS V3 TARGET ENGINE")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values("Date").reset_index(drop=True)

print("\nInput:")
print("Rows:", len(df))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

# ------------------------------------------------------------
# PRICE
# ------------------------------------------------------------

if "Close" not in df.columns:
    raise ValueError("Close column not found.")

close = pd.to_numeric(df["Close"], errors="coerce")

# ------------------------------------------------------------
# RETURNS
# ------------------------------------------------------------

df["Target_Return_1D"] = close.shift(-1) / close - 1.0

df["Target_Return_5D"] = close.shift(-5) / close - 1.0

df["Target_Return_10D"] = close.shift(-10) / close - 1.0

# ------------------------------------------------------------
# VOLATILITY
#
# IMPORTANT:
# Shift by one day.
#
# The target at day t must only use volatility information
# that was available at the end of day t.
# ------------------------------------------------------------

daily_return = close.pct_change()

df["Target_Volatility_20"] = (
    daily_return
    .rolling(20)
    .std()
)

# ------------------------------------------------------------
# VOLATILITY-ADJUSTED RETURN
# ------------------------------------------------------------

df["Target_Return_Vol_Adj"] = (
    df["Target_Return_1D"] /
    df["Target_Volatility_20"]
)

# ------------------------------------------------------------
# V3 DIRECTION
#
# Ignore movements smaller than a volatility-scaled threshold.
#
# +1 = meaningful upward move
# -1 = meaningful downward move
#  0 = noise / neutral
# ------------------------------------------------------------

THRESHOLD = 0.50

df["Target_Direction_V3"] = np.select(
    [
        df["Target_Return_1D"] >
        THRESHOLD * df["Target_Volatility_20"],

        df["Target_Return_1D"] <
        -THRESHOLD * df["Target_Volatility_20"]
    ],
    [
        1,
        -1
    ],
    default=0
)

# ------------------------------------------------------------
# BINARY V3 TARGET
#
# Useful for binary XGBoost / Logistic models.
#
# 1 = meaningful positive move
# 0 = everything else
# ------------------------------------------------------------

df["Target_Up_V3"] = (
    df["Target_Direction_V3"] == 1
).astype(int)

# ------------------------------------------------------------
# 5D V3 DIRECTION
# ------------------------------------------------------------

df["Target_Direction_5D_V3"] = np.select(
    [
        df["Target_Return_5D"] >
        THRESHOLD * df["Target_Volatility_20"] * np.sqrt(5),

        df["Target_Return_5D"] <
        -THRESHOLD * df["Target_Volatility_20"] * np.sqrt(5)
    ],
    [
        1,
        -1
    ],
    default=0
)

# ------------------------------------------------------------
# 10D V3 DIRECTION
# ------------------------------------------------------------

df["Target_Direction_10D_V3"] = np.select(
    [
        df["Target_Return_10D"] >
        THRESHOLD * df["Target_Volatility_20"] * np.sqrt(10),

        df["Target_Return_10D"] <
        -THRESHOLD * df["Target_Volatility_20"] * np.sqrt(10)
    ],
    [
        1,
        -1
    ],
    default=0
)

# ------------------------------------------------------------
# EXCESS RETURN
#
# Compare future TCS return against its recent volatility.
# ------------------------------------------------------------

df["Target_Excess_Return"] = (
    df["Target_Return_1D"] -
    df["Target_Volatility_20"] * THRESHOLD
)

# ------------------------------------------------------------
# CLEAN
# ------------------------------------------------------------

target_columns = [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Volatility_20",
    "Target_Return_Vol_Adj",
    "Target_Direction_V3",
    "Target_Up_V3",
    "Target_Direction_5D_V3",
    "Target_Direction_10D_V3",
    "Target_Excess_Return"
]

# We need valid future targets.
df = df.dropna(
    subset=[
        "Target_Return_1D",
        "Target_Return_5D",
        "Target_Return_10D"
    ]
).copy()

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V3 TARGET VALIDATION")
print("=" * 70)

print("\n1D return:")
print(
    df["Target_Return_1D"]
    .describe()
    .to_string()
)

print("\n20D volatility:")
print(
    df["Target_Volatility_20"]
    .describe()
    .to_string()
)

print("\nV3 direction distribution:")
print(
    df["Target_Direction_V3"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nV3 direction percentages:")
print(
    (
        df["Target_Direction_V3"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )
    .round(2)
    .to_string()
)

print("\nBinary Up V3:")
print(
    df["Target_Up_V3"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n5D V3:")
print(
    df["Target_Direction_5D_V3"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n10D V3:")
print(
    df["Target_Direction_10D_V3"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# CHECK TARGET BALANCE
# ------------------------------------------------------------

up_pct = df["Target_Up_V3"].mean()

print("\nPositive class rate:")
print(f"{up_pct:.2%}")

if up_pct < 0.20 or up_pct > 0.80:
    print(
        "\nWARNING: Binary V3 target is highly imbalanced."
    )
else:
    print(
        "\nPASS: Binary V3 target has reasonable class balance."
    )

# ------------------------------------------------------------
# CHRONOLOGY
# ------------------------------------------------------------

print("\nChronology:")
print("Dates sorted:", df["Date"].is_monotonic_increasing)
print("Duplicate dates:", df["Date"].duplicated().sum())

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("V3 TARGET ENGINE COMPLETE")
print("=" * 70)

print("\nOutput:", OUTPUT)
print("Rows:", len(df))
print(
    "Date:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

print("\nTargets created:")

for col in target_columns:
    print(" -", col)

print("\nPASS: V3 targets created.")
print("=" * 70)