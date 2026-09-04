import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# ALPHALENS - NEWS + TECHNICAL FEATURE ENGINEERING
# ============================================================

INPUT_FILE = "data/ml_ready_TCS_news.csv"
OUTPUT_FILE = "data/ml_ready_TCS_news_features.csv"

print("=" * 60)
print("ALPHALENS NEWS FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# 1. CHECK INPUT
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print()
    print("ERROR: Input file not found:")
    print(INPUT_FILE)
    print()
    print("Make sure integrate_news_features.py was completed first.")
    sys.exit(1)

print()
print("Loading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))
print("Columns loaded:", len(df.columns))

# ------------------------------------------------------------
# 2. DATE HANDLING
# ------------------------------------------------------------

if "Date" not in df.columns:
    raise ValueError("Input dataset does not contain a Date column.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = df["Date"].isna().sum()

if invalid_dates > 0:
    print("WARNING: Removing invalid dates:", invalid_dates)
    df = df.dropna(subset=["Date"])

df = df.sort_values("Date").reset_index(drop=True)

# ------------------------------------------------------------
# 3. CHECK PRICE COLUMNS
# ------------------------------------------------------------

required_price_columns = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

for column in required_price_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required price column missing: {column}"
        )

# ------------------------------------------------------------
# 4. ENSURE NUMERIC PRICE DATA
# ------------------------------------------------------------

for column in required_price_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ------------------------------------------------------------
# 5. BASIC PRICE FEATURES
# ------------------------------------------------------------

print()
print("Creating price features...")

df["Daily_Return"] = df["Close"].pct_change()

df["Daily_Return_Pct"] = df["Daily_Return"] * 100

df["Price_Change"] = df["Close"].diff()

df["High_Low_Range"] = df["High"] - df["Low"]

df["High_Low_Range_Pct"] = (
    (df["High"] - df["Low"]) / df["Close"]
) * 100

df["Open_Close_Change"] = df["Close"] - df["Open"]

df["Open_Close_Change_Pct"] = (
    (df["Close"] - df["Open"]) / df["Open"]
) * 100

# ------------------------------------------------------------
# 6. MOVING AVERAGES
# ------------------------------------------------------------

print("Creating moving averages...")

df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_100"] = df["Close"].rolling(100).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()

df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

# ------------------------------------------------------------
# 7. PRICE VS MOVING AVERAGES
# ------------------------------------------------------------

df["Close_vs_SMA20"] = (
    (df["Close"] - df["SMA_20"]) / df["SMA_20"]
)

df["Close_vs_SMA50"] = (
    (df["Close"] - df["SMA_50"]) / df["SMA_50"]
)

df["Close_vs_SMA200"] = (
    (df["Close"] - df["SMA_200"]) / df["SMA_200"]
)

# ------------------------------------------------------------
# 8. RSI
# ------------------------------------------------------------

print("Creating RSI...")

delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["RSI_14"] = 100 - (100 / (1 + rs))

# ------------------------------------------------------------
# 9. MACD
# ------------------------------------------------------------

print("Creating MACD...")

ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema_12 - ema_26

df["MACD_Signal"] = (
    df["MACD"].ewm(span=9, adjust=False).mean()
)

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)

# ------------------------------------------------------------
# 10. VOLATILITY
# ------------------------------------------------------------

print("Creating volatility features...")

df["Volatility_5"] = (
    df["Daily_Return"].rolling(5).std()
)

df["Volatility_10"] = (
    df["Daily_Return"].rolling(10).std()
)

df["Volatility_20"] = (
    df["Daily_Return"].rolling(20).std()
)

df["Volatility_20_Pct"] = df["Volatility_20"] * 100

# ------------------------------------------------------------
# 11. VOLUME FEATURES
# ------------------------------------------------------------

print("Creating volume features...")

df["Volume_SMA_5"] = df["Volume"].rolling(5).mean()
df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()

df["Volume_Ratio"] = (
    df["Volume"] / df["Volume_SMA_20"]
)

df["Volume_Change"] = df["Volume"].pct_change()

# ------------------------------------------------------------
# 12. MOMENTUM
# ------------------------------------------------------------

print("Creating momentum features...")

df["Momentum_5"] = (
    df["Close"] / df["Close"].shift(5) - 1
)

df["Momentum_10"] = (
    df["Close"] / df["Close"].shift(10) - 1
)

df["Momentum_20"] = (
    df["Close"] / df["Close"].shift(20) - 1
)

df["Momentum_60"] = (
    df["Close"] / df["Close"].shift(60) - 1
)

# ------------------------------------------------------------
# 13. PRICE POSITION / BREAKOUT FEATURES
# ------------------------------------------------------------

print("Creating price-position features...")

df["Rolling_High_20"] = (
    df["High"].rolling(20).max()
)

df["Rolling_Low_20"] = (
    df["Low"].rolling(20).min()
)

df["Price_Position_20"] = (
    (df["Close"] - df["Rolling_Low_20"]) /
    (df["Rolling_High_20"] - df["Rolling_Low_20"])
)

df["Rolling_High_50"] = (
    df["High"].rolling(50).max()
)

df["Rolling_Low_50"] = (
    df["Low"].rolling(50).min()
)

df["Price_Position_50"] = (
    (df["Close"] - df["Rolling_Low_50"]) /
    (df["Rolling_High_50"] - df["Rolling_Low_50"])
)

# ------------------------------------------------------------
# 14. NEWS FEATURES
# ------------------------------------------------------------

print("Checking news features...")

news_columns = [
    "News_Sentiment",
    "News_Count",
    "Has_News",
    "News_Sentiment_Lag1",
    "News_Sentiment_MA3",
    "News_Sentiment_MA5",
    "News_Count_MA3",
    "News_Sentiment_Change"
]

for column in news_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required news feature missing: {column}"
        )

# ------------------------------------------------------------
# 15. ADDITIONAL NEWS FEATURES
# ------------------------------------------------------------

print("Creating additional news features...")

df["News_Sentiment_Lag2"] = (
    df["News_Sentiment"].shift(2)
)

df["News_Sentiment_Lag3"] = (
    df["News_Sentiment"].shift(3)
)

df["News_Sentiment_MA5_Trading"] = (
    df["News_Sentiment"].rolling(5).mean()
)

df["News_Sentiment_MA10"] = (
    df["News_Sentiment"].rolling(10).mean()
)

df["News_Count_MA5"] = (
    df["News_Count"].rolling(5).mean()
)

df["News_Count_MA10"] = (
    df["News_Count"].rolling(10).mean()
)

df["News_Sentiment_Abs"] = (
    df["News_Sentiment"].abs()
)

df["News_Positive"] = (
    (df["News_Sentiment"] > 0).astype(int)
)

df["News_Negative"] = (
    (df["News_Sentiment"] < 0).astype(int)
)

df["News_Neutral"] = (
    (df["News_Sentiment"] == 0).astype(int)
)

# ------------------------------------------------------------
# 16. NEWS × PRICE FEATURES
# ------------------------------------------------------------

print("Creating news-price interaction features...")

df["News_Return_Interaction"] = (
    df["News_Sentiment"] *
    df["Daily_Return"]
)

df["News_Momentum_Interaction"] = (
    df["News_Sentiment"] *
    df["Momentum_5"]
)

df["News_Volume_Interaction"] = (
    df["News_Sentiment"] *
    df["Volume_Ratio"]
)

# ------------------------------------------------------------
# 17. FORWARD TARGETS
# ------------------------------------------------------------

print("Creating prediction targets...")

# IMPORTANT:
# These are targets, not input features.
# They use FUTURE prices and therefore must NOT be used
# as model inputs.

df["Target_Return_1D"] = (
    df["Close"].shift(-1) / df["Close"] - 1
)

df["Target_Return_5D"] = (
    df["Close"].shift(-5) / df["Close"] - 1
)

df["Target_Return_10D"] = (
    df["Close"].shift(-10) / df["Close"] - 1
)

df["Target_Direction_1D"] = (
    df["Target_Return_1D"] > 0
).astype(int)

df["Target_Direction_5D"] = (
    df["Target_Return_5D"] > 0
).astype(int)

# ------------------------------------------------------------
# 18. CLEAN INFINITE VALUES
# ------------------------------------------------------------

print("Cleaning infinite values...")

df = df.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# 19. DO NOT DROP ROWS YET
# ------------------------------------------------------------
#
# Early technical indicators naturally contain NaN values.
# We keep them here so the complete historical dataset
# remains available.
#
# Later train/validation/test preparation can remove the
# appropriate rows after selecting the feature columns.

# ------------------------------------------------------------
# 20. SAVE
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# 21. REPORT
# ------------------------------------------------------------

print()
print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Date range:")
print("First:", df["Date"].min().date())
print("Last :", df["Date"].max().date())

print()
print("News coverage:")

news_days = int(df["Has_News"].sum())
total_days = len(df)

coverage = (
    news_days / total_days * 100
    if total_days > 0
    else 0
)

print("Trading days:", total_days)
print("Trading days with news:", news_days)
print("News coverage: {:.2f} %".format(coverage))

print()
print("News sentiment statistics:")
print(df["News_Sentiment"].describe())

print()
print("Missing values in major features:")

check_columns = [
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "RSI_14",
    "MACD",
    "Volatility_20",
    "Momentum_20",
    "News_Sentiment",
    "News_Count",
    "News_Sentiment_Lag1",
    "News_Sentiment_MA3",
    "News_Sentiment_MA5"
]

print(df[check_columns].isna().sum())

print()
print("Latest rows:")

print(
    df[
        [
            "Date",
            "Close",
            "News_Sentiment",
            "News_Count",
            "Has_News",
            "RSI_14",
            "MACD",
            "Momentum_20",
            "Target_Return_1D"
        ]
    ].tail(10).to_string(index=False)
)

print()
print("=" * 60)
print("PASS: Technical + news features created successfully.")
print("=" * 60)