import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/ml_ready_TCS_news_clean.csv")
OUTPUT = Path("data/ml_targets_v2.csv")

print("=" * 70)
print("ALPHALENS V2 TARGET ENGINE")
print("=" * 70)

print("\nLoading:", INPUT)

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Rows:", len(df))
print(
    "Date:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

# ============================================================
# 1. NEXT-DAY RETURN
# ============================================================

df["Target_Return_1D"] = (
    df["Close"].shift(-1) / df["Close"] - 1
)

# ============================================================
# 2. FUTURE 5-DAY RETURN
# ============================================================

df["Target_Return_5D"] = (
    df["Close"].shift(-5) / df["Close"] - 1
)

# ============================================================
# 3. FUTURE 10-DAY RETURN
# ============================================================

df["Target_Return_10D"] = (
    df["Close"].shift(-10) / df["Close"] - 1
)

# ============================================================
# 4. HISTORICAL VOLATILITY
# ============================================================

historical_vol = (
    df["Close"]
    .pct_change()
    .rolling(20)
    .std()
)

df["Target_Volatility_20"] = historical_vol

# ============================================================
# 5. SIMPLE NEXT-DAY DIRECTION
# ============================================================

df["Target_Direction_1D"] = np.where(
    df["Target_Return_1D"].notna(),
    (df["Target_Return_1D"] > 0).astype(int),
    np.nan
)

# ============================================================
# 6. VOLATILITY-ADJUSTED 3-CLASS TARGET
# ============================================================

threshold = 0.5 * df["Target_Volatility_20"]

df["Target_Regime_1D"] = np.nan

valid = (
    df["Target_Return_1D"].notna()
    & threshold.notna()
)

df.loc[
    valid & (df["Target_Return_1D"] > threshold),
    "Target_Regime_1D"
] = 1

df.loc[
    valid
    & (df["Target_Return_1D"] < -threshold),
    "Target_Regime_1D"
] = -1

df.loc[
    valid
    & (df["Target_Return_1D"].abs() <= threshold),
    "Target_Regime_1D"
] = 0

# ============================================================
# 7. REMOVE ROWS WITHOUT FUTURE TARGETS
# ============================================================

df = df.iloc[:-10].copy()

# ============================================================
# 8. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

print("\n1D return statistics:")
print(
    df["Target_Return_1D"]
    .describe()
    .to_string()
)

print("\n1D direction:")
print(
    df["Target_Direction_1D"]
    .value_counts()
    .sort_index()
)

print("\nVolatility-adjusted regime:")
print(
    df["Target_Regime_1D"]
    .value_counts()
    .sort_index()
)

print("\nMissing targets:")

target_columns = [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Direction_1D",
    "Target_Regime_1D",
]

print(
    df[target_columns]
    .isna()
    .sum()
)

# ============================================================
# 9. SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("TARGET ENGINE COMPLETE")
print("=" * 70)

print("\nOutput:")
print(OUTPUT)

print("Rows:", len(df))

print(
    "Date:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

print("\nPASS: V2 targets created.")
print("=" * 70)