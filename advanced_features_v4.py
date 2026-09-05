import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ALPHALENS V4 — ADVANCED FEATURE ENGINEERING
# ============================================================

INPUT = Path("data/ml_targets_v3.csv")
OUTPUT = Path("data/ml_features_v4.csv")

print("=" * 70)
print("ALPHALENS V4 ADVANCED FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nInput rows:", len(df))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

# ============================================================
# BASIC PRICE FEATURES
# ============================================================

df["Return_1"] = df["Close"].pct_change(1)
df["Return_3"] = df["Close"].pct_change(3)
df["Return_5"] = df["Close"].pct_change(5)
df["Return_10"] = df["Close"].pct_change(10)
df["Return_20"] = df["Close"].pct_change(20)
df["Return_60"] = df["Close"].pct_change(60)

# ============================================================
# TREND FEATURES
# ============================================================

df["Trend_5_20"] = df["SMA_5"] / df["SMA_20"] - 1
df["Trend_10_50"] = df["SMA_10"] / df["SMA_50"] - 1
df["Trend_20_50"] = df["SMA_20"] / df["SMA_50"] - 1
df["Trend_50_200"] = df["SMA_50"] / df["SMA_200"] - 1

df["EMA_5_20"] = df["EMA_5"] / df["EMA_20"] - 1
df["EMA_20_50"] = df["EMA_20"] / df["EMA_50"] - 1
df["EMA_50_200"] = df["EMA_50"] / df["EMA_200"] - 1

df["Price_vs_EMA20"] = df["Close"] / df["EMA_20"] - 1
df["Price_vs_EMA50"] = df["Close"] / df["EMA_50"] - 1
df["Price_vs_EMA200"] = df["Close"] / df["EMA_200"] - 1

# ============================================================
# MOMENTUM FEATURES
# ============================================================

df["Momentum_3"] = df["Close"] / df["Close"].shift(3) - 1
df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

# Momentum acceleration
df["Momentum_Acceleration_5"] = (
    df["Momentum_5"] - df["Momentum_5"].shift(5)
)

df["Momentum_Acceleration_10"] = (
    df["Momentum_10"] - df["Momentum_10"].shift(10)
)

# ============================================================
# RSI FEATURES
# ============================================================

df["RSI_Distance_50"] = df["RSI_14"] - 50
df["RSI_Overbought"] = (df["RSI_14"] > 70).astype(int)
df["RSI_Oversold"] = (df["RSI_14"] < 30).astype(int)

# ============================================================
# MACD FEATURES
# ============================================================

df["MACD_Distance"] = (
    df["MACD"] - df["MACD_Signal"]
)

df["MACD_Positive"] = (
    df["MACD_Distance"] > 0
).astype(int)

df["MACD_Momentum"] = (
    df["MACD_Histogram"] -
    df["MACD_Histogram"].shift(1)
)

# ============================================================
# VOLATILITY REGIME
# ============================================================

df["Vol_Ratio_5_20"] = (
    df["Volatility_5"] /
    df["Volatility_20"]
)

df["Vol_Ratio_10_20"] = (
    df["Volatility_10"] /
    df["Volatility_20"]
)

df["Volatility_Expansion"] = (
    df["Volatility_5"] -
    df["Volatility_20"]
)

df["Volatility_Percentile_60"] = (
    df["Volatility_20"]
    .rolling(60)
    .rank(pct=True)
)

df["High_Volatility_Regime"] = (
    df["Volatility_20"] >
    df["Volatility_20"].rolling(60).median()
).astype(int)

# ============================================================
# VOLUME REGIME
# ============================================================

df["Volume_Ratio_5_20"] = (
    df["Volume_SMA_5"] /
    df["Volume_SMA_20"]
)

df["Volume_Surge"] = (
    df["Volume_Ratio"] > 1.5
).astype(int)

df["Volume_Dry"] = (
    df["Volume_Ratio"] < 0.7
).astype(int)

# ============================================================
# PRICE POSITION / BREAKOUT
# ============================================================

df["Breakout_Distance_20"] = (
    df["Close"] /
    df["Rolling_High_20"] - 1
)

df["Breakdown_Distance_20"] = (
    df["Close"] /
    df["Rolling_Low_20"] - 1
)

df["Breakout_Distance_50"] = (
    df["Close"] /
    df["Rolling_High_50"] - 1
)

df["Breakdown_Distance_50"] = (
    df["Close"] /
    df["Rolling_Low_50"] - 1
)

df["Near_20D_High"] = (
    df["Price_Position_20"] > 0.9
).astype(int)

df["Near_20D_Low"] = (
    df["Price_Position_20"] < 0.1
).astype(int)

df["Near_50D_High"] = (
    df["Price_Position_50"] > 0.9
).astype(int)

df["Near_50D_Low"] = (
    df["Price_Position_50"] < 0.1
).astype(int)

# ============================================================
# CANDLE STRUCTURE
# ============================================================

df["Body_Size"] = (
    df["Close"] - df["Open"]
) / df["Open"]

df["Upper_Wick"] = (
    df["High"] -
    df[["Open", "Close"]].max(axis=1)
) / df["Open"]

df["Lower_Wick"] = (
    df[["Open", "Close"]].min(axis=1) -
    df["Low"]
) / df["Open"]

df["Body_to_Range"] = (
    abs(df["Close"] - df["Open"]) /
    (df["High"] - df["Low"] + 1e-10)
)

df["Bullish_Candle"] = (
    df["Close"] > df["Open"]
).astype(int)

# ============================================================
# MEAN REVERSION
# ============================================================

df["Distance_SMA20"] = (
    df["Close"] / df["SMA_20"] - 1
)

df["Distance_SMA50"] = (
    df["Close"] / df["SMA_50"] - 1
)

df["ZScore_20"] = (
    (df["Close"] - df["Close"].rolling(20).mean()) /
    df["Close"].rolling(20).std()
)

df["ZScore_60"] = (
    (df["Close"] - df["Close"].rolling(60).mean()) /
    df["Close"].rolling(60).std()
)

# ============================================================
# RETURN DISTRIBUTION
# ============================================================

df["Return_Mean_20"] = (
    df["Return_1"].rolling(20).mean()
)

df["Return_Std_20"] = (
    df["Return_1"].rolling(20).std()
)

df["Return_Skew_60"] = (
    df["Return_1"].rolling(60).skew()
)

df["Return_Kurtosis_60"] = (
    df["Return_1"].rolling(60).kurt()
)

# ============================================================
# NEWS FEATURES — ONLY PAST INFORMATION
# ============================================================

if "News_Sentiment" in df.columns:

    df["News_Sentiment_Lag1_V4"] = (
        df["News_Sentiment"].shift(1)
    )

    df["News_Sentiment_Momentum"] = (
        df["News_Sentiment_Lag1_V4"] -
        df["News_Sentiment"].shift(2)
    )

    df["News_Activity_Surge"] = (
        df["News_Count"] >
        df["News_Count"].rolling(20).mean() * 2
    ).astype(int)

# ============================================================
# REMOVE ORIGINAL TARGETS FROM FEATURE DATA
# ============================================================

target_columns = [
    c for c in df.columns
    if c.startswith("Target_")
]

# Keep Date, original features and engineered features,
# but remove targets because this is the feature dataset.

features_df = df.drop(
    columns=target_columns,
    errors="ignore"
)

# ============================================================
# CLEAN
# ============================================================

numeric_columns = features_df.select_dtypes(
    include=[np.number]
).columns

features_df[numeric_columns] = (
    features_df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

# Forward fill is NOT used.
# Missing values at the beginning are expected because
# rolling indicators require historical observations.

features_df = features_df.dropna().reset_index(drop=True)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("V4 FEATURE DATASET")
print("=" * 70)

print("Rows:", len(features_df))
print("Columns:", len(features_df.columns))
print(
    "Date:",
    features_df["Date"].min().date(),
    "to",
    features_df["Date"].max().date()
)

print("\nNew engineered features:")

original_features = set(df.columns)

new_features = [
    c for c in features_df.columns
    if c not in original_features
]

for i, feature in enumerate(new_features, 1):
    print(f"{i:02d}. {feature}")

print("\nMissing values:")
print(
    features_df.isna().sum()
    .sum()
)

# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

features_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("V4 FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("Output:", OUTPUT)

print("\nPASS: V4 features created.")
print("=" * 70)