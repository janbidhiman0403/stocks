import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ALPHALENS V4 FEATURE ENGINEERING
# ============================================================

INPUT = Path("data/ml_ready_TCS_news_clean.csv")
OUTPUT = Path("data/ml_features_v4.csv")

print("=" * 70)
print("ALPHALENS V4 FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nInput:")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def add_feature(name, values):
    df[name] = values

# ------------------------------------------------------------
# 1. RETURNS / PRICE DYNAMICS
# ------------------------------------------------------------

close = df["Close"]
open_ = df["Open"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

add_feature("Return_1D", close.pct_change(1))
add_feature("Return_2D", close.pct_change(2))
add_feature("Return_3D", close.pct_change(3))
add_feature("Return_5D", close.pct_change(5))
add_feature("Return_10D", close.pct_change(10))
add_feature("Return_20D", close.pct_change(20))
add_feature("Return_60D", close.pct_change(60))

add_feature("Return_1D_Lag1", df["Return_1D"].shift(1))
add_feature("Return_1D_Lag2", df["Return_1D"].shift(2))
add_feature("Return_1D_Lag3", df["Return_1D"].shift(3))
add_feature("Return_1D_Lag5", df["Return_1D"].shift(5))

# ------------------------------------------------------------
# 2. CANDLE FEATURES
# ------------------------------------------------------------

add_feature(
    "Body_Pct",
    (close - open_) / open_
)

add_feature(
    "Upper_Shadow_Pct",
    (high - np.maximum(open_, close)) / close
)

add_feature(
    "Lower_Shadow_Pct",
    (np.minimum(open_, close) - low) / close
)

add_feature(
    "Close_Position_Day",
    (close - low) / (high - low + 1e-10)
)

add_feature(
    "Range_Pct",
    (high - low) / close
)

# ------------------------------------------------------------
# 3. TREND / MOVING AVERAGE DISTANCE
# ------------------------------------------------------------

for window in [5, 10, 20, 50, 100, 200]:

    sma = close.rolling(window).mean()

    add_feature(
        f"Distance_SMA_{window}",
        close / sma - 1
    )

    add_feature(
        f"Slope_SMA_{window}",
        sma.pct_change(5)
    )

# EMA distances

for window in [10, 20, 50, 200]:

    ema = close.ewm(span=window, adjust=False).mean()

    add_feature(
        f"Distance_EMA_{window}",
        close / ema - 1
    )

# ------------------------------------------------------------
# 4. MOMENTUM
# ------------------------------------------------------------

for window in [3, 5, 10, 20, 60]:

    momentum = close / close.shift(window) - 1

    add_feature(
        f"Momentum_{window}D",
        momentum
    )

    add_feature(
        f"Momentum_{window}D_Lag1",
        momentum.shift(1)
    )

# ------------------------------------------------------------
# 5. VOLATILITY
# ------------------------------------------------------------

returns = close.pct_change()

for window in [5, 10, 20, 30, 60]:

    vol = returns.rolling(window).std()

    add_feature(
        f"Rolling_Volatility_{window}",
        vol
    )

# Volatility regime

vol20 = returns.rolling(20).std()

add_feature(
    "Volatility_Ratio_5_20",
    returns.rolling(5).std() / (vol20 + 1e-10)
)

add_feature(
    "Volatility_Ratio_10_20",
    returns.rolling(10).std() / (vol20 + 1e-10)
)

# ------------------------------------------------------------
# 6. ATR
# ------------------------------------------------------------

previous_close = close.shift(1)

true_range = pd.concat(
    [
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ],
    axis=1
).max(axis=1)

for window in [5, 14, 20]:

    atr = true_range.rolling(window).mean()

    add_feature(
        f"ATR_{window}",
        atr
    )

    add_feature(
        f"ATR_{window}_Pct",
        atr / close
    )

# ------------------------------------------------------------
# 7. BREAKOUT FEATURES
# ------------------------------------------------------------

for window in [10, 20, 50]:

    previous_high = high.shift(1).rolling(window).max()
    previous_low = low.shift(1).rolling(window).min()

    add_feature(
        f"Breakout_High_{window}",
        (close > previous_high).astype(int)
    )

    add_feature(
        f"Breakout_Low_{window}",
        (close < previous_low).astype(int)
    )

    add_feature(
        f"Distance_High_{window}",
        close / previous_high - 1
    )

    add_feature(
        f"Distance_Low_{window}",
        close / previous_low - 1
    )

# ------------------------------------------------------------
# 8. PRICE POSITION
# ------------------------------------------------------------

for window in [10, 20, 50, 100]:

    rolling_high = high.shift(1).rolling(window).max()
    rolling_low = low.shift(1).rolling(window).min()

    position = (
        (close - rolling_low) /
        (rolling_high - rolling_low + 1e-10)
    )

    add_feature(
        f"Price_Position_{window}",
        position
    )

# ------------------------------------------------------------
# 9. VOLUME FEATURES
# ------------------------------------------------------------

for window in [5, 10, 20, 50]:

    volume_ma = volume.rolling(window).mean()

    add_feature(
        f"Volume_Ratio_{window}",
        volume / (volume_ma + 1e-10)
    )

    add_feature(
        f"Volume_Change_{window}",
        volume.pct_change(window)
    )

# Volume-price confirmation

add_feature(
    "Volume_Return_Interaction",
    df["Return_1D"] * np.log1p(volume)
)

add_feature(
    "Volume_Momentum_Interaction",
    df["Momentum_5D"] * df["Volume_Ratio_20"]
)

# ------------------------------------------------------------
# 10. RSI
# ------------------------------------------------------------

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / (avg_loss + 1e-10)

rsi = 100 - (100 / (1 + rs))

add_feature("RSI_V4", rsi)
add_feature("RSI_Change", rsi.diff())
add_feature("RSI_Momentum", rsi.diff(5))

# RSI regime

add_feature(
    "RSI_Oversold",
    (rsi < 30).astype(int)
)

add_feature(
    "RSI_Overbought",
    (rsi > 70).astype(int)
)

# ------------------------------------------------------------
# 11. MACD
# ------------------------------------------------------------

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()

macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()
hist = macd - signal

add_feature("MACD_V4", macd)
add_feature("MACD_Signal_V4", signal)
add_feature("MACD_Hist_V4", hist)
add_feature("MACD_Hist_Change", hist.diff())

add_feature(
    "MACD_Bullish",
    (macd > signal).astype(int)
)

# ------------------------------------------------------------
# 12. TREND REGIME
# ------------------------------------------------------------

sma20 = close.rolling(20).mean()
sma50 = close.rolling(50).mean()
sma200 = close.rolling(200).mean()

add_feature(
    "Trend_20_50",
    sma20 / sma50 - 1
)

add_feature(
    "Trend_50_200",
    sma50 / sma200 - 1
)

add_feature(
    "Bull_Trend_Regime",
    (
        (close > sma50) &
        (sma50 > sma200)
    ).astype(int)
)

add_feature(
    "Bear_Trend_Regime",
    (
        (close < sma50) &
        (sma50 < sma200)
    ).astype(int)
)

# ------------------------------------------------------------
# 13. VOLATILITY REGIME
# ------------------------------------------------------------

vol20 = returns.rolling(20).std()

vol_long = vol20.rolling(100).mean()

add_feature(
    "High_Volatility_Regime",
    (vol20 > vol_long).astype(int)
)

add_feature(
    "Low_Volatility_Regime",
    (vol20 < vol_long).astype(int)
)

add_feature(
    "Volatility_Regime_Strength",
    vol20 / (vol_long + 1e-10)
)

# ------------------------------------------------------------
# 14. MARKET STATE SCORE
# ------------------------------------------------------------

trend_score = (
    (close > sma20).astype(int)
    + (sma20 > sma50).astype(int)
    + (sma50 > sma200).astype(int)
)

momentum_score = (
    (df["Momentum_5D"] > 0).astype(int)
    + (df["Momentum_10D"] > 0).astype(int)
    + (df["Momentum_20D"] > 0).astype(int)
)

add_feature(
    "Trend_Score",
    trend_score
)

add_feature(
    "Momentum_Score",
    momentum_score
)

add_feature(
    "Market_State_Score",
    trend_score + momentum_score
)

# ------------------------------------------------------------
# 15. NEWS FEATURES
# ------------------------------------------------------------

news_columns = [
    "News_Sentiment",
    "News_Count",
    "Has_News",
    "News_Sentiment_Lag1",
    "News_Sentiment_MA3",
    "News_Sentiment_MA5",
    "News_Count_MA3",
    "News_Sentiment_Change",
    "News_Sentiment_Lag2",
    "News_Sentiment_Lag3",
    "News_Sentiment_MA5_Trading",
    "News_Sentiment_MA10",
    "News_Count_MA5",
    "News_Count_MA10",
    "News_Sentiment_Abs",
    "News_Positive",
    "News_Negative",
    "News_Neutral",
    "News_Return_Interaction",
    "News_Momentum_Interaction",
    "News_Volume_Interaction"
]

for col in news_columns:

    if col in df.columns:

        # Explicitly lag current-day news features
        # so the model cannot accidentally use information
        # that arrived after the prediction cutoff.

        if col in [
            "News_Sentiment",
            "News_Count",
            "Has_News",
            "News_Positive",
            "News_Negative",
            "News_Neutral"
        ]:

            df[f"{col}_V4"] = df[col].shift(1)

# ------------------------------------------------------------
# 16. REMOVE ORIGINAL TARGET COLUMNS
# ------------------------------------------------------------

target_keywords = [
    "Target_",
    "target",
    "Future",
    "future"
]

target_columns = []

for col in df.columns:

    if any(
        keyword in col
        for keyword in target_keywords
    ):

        target_columns.append(col)

if target_columns:

    print("\nRemoving target-like columns:")

    for col in target_columns:
        print(" -", col)

    df = df.drop(columns=target_columns)

# ------------------------------------------------------------
# 17. CLEAN
# ------------------------------------------------------------

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

feature_columns = [
    c for c in df.columns
    if c != "Date"
]

# Keep only rows where engineered features are available

before = len(df)

df = df.dropna(
    subset=feature_columns
).reset_index(drop=True)

removed = before - len(df)

# ------------------------------------------------------------
# 18. VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V4 FEATURE VALIDATION")
print("=" * 70)

print("\nRows after cleaning:", len(df))
print("Rows removed:", removed)
print("Feature count:", len(feature_columns))

print("\nDate:")
print(df["Date"].min().date(), "to", df["Date"].max().date())

print("\nMissing values:")

missing = df[feature_columns].isna().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("PASS: No missing feature values.")
else:
    print(missing.to_string())

print("\nInfinite values:")

infinite_count = np.isinf(
    df[feature_columns].select_dtypes(include=np.number)
).sum().sum()

print(infinite_count)

if infinite_count == 0:
    print("PASS: No infinite values.")

print("\nDuplicate dates:")

duplicates = df["Date"].duplicated().sum()

print(duplicates)

if duplicates == 0:
    print("PASS: No duplicate dates.")

print("\nChronological order:")

chronological = df["Date"].is_monotonic_increasing

print(chronological)

if chronological:
    print("PASS: Dates sorted chronologically.")

# ------------------------------------------------------------
# FEATURE LIST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("V4 FEATURE LIST")
print("=" * 70)

for i, col in enumerate(feature_columns, 1):

    print(
        f"{i:03d}. {col}"
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("V4 FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("Output:", OUTPUT)
print("Rows:", len(df))
print("Features:", len(feature_columns))

print("\nPASS: V4 feature dataset created.")
print("=" * 70)