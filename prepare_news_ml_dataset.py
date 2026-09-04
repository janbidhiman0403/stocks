import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# ALPHALENS
# PREPARE CLEAN ML DATASET WITH NEWS FEATURES
# ============================================================

INPUT_FILE = "data/ml_ready_TCS_news_features.csv"
OUTPUT_FILE = "data/ml_ready_TCS_news_clean.csv"

print("=" * 60)
print("ALPHALENS ML DATASET PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. CHECK INPUT
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print()
    print("ERROR: Input file not found:")
    print(INPUT_FILE)
    sys.exit(1)

print()
print("Loading:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# ------------------------------------------------------------
# 2. DATE
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(subset=["Date"])

df = df.sort_values("Date").reset_index(drop=True)

# ------------------------------------------------------------
# 3. FEATURE COLUMNS
# ------------------------------------------------------------

feature_columns = [
    # Price
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",

    # Returns / range
    "Daily_Return",
    "Daily_Return_Pct",
    "Price_Change",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Open_Close_Change",
    "Open_Close_Change_Pct",

    # Moving averages
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",

    "EMA_5",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",

    # Price vs averages
    "Close_vs_SMA20",
    "Close_vs_SMA50",
    "Close_vs_SMA200",

    # RSI
    "RSI_14",

    # MACD
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # Volatility
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Volatility_20_Pct",

    # Volume
    "Volume_SMA_5",
    "Volume_SMA_20",
    "Volume_Ratio",
    "Volume_Change",

    # Momentum
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Momentum_60",

    # Price position
    "Rolling_High_20",
    "Rolling_Low_20",
    "Price_Position_20",
    "Rolling_High_50",
    "Rolling_Low_50",
    "Price_Position_50",

    # Existing news features
    "News_Sentiment",
    "News_Count",
    "Has_News",
    "News_Sentiment_Lag1",
    "News_Sentiment_MA3",
    "News_Sentiment_MA5",
    "News_Count_MA3",
    "News_Sentiment_Change",

    # Additional news features
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

    # News × price
    "News_Return_Interaction",
    "News_Momentum_Interaction",
    "News_Volume_Interaction",
]

# ------------------------------------------------------------
# 4. TARGET COLUMNS
# ------------------------------------------------------------

target_columns = [
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Direction_1D",
    "Target_Direction_5D",
]

# ------------------------------------------------------------
# 5. VERIFY COLUMNS
# ------------------------------------------------------------

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

missing_targets = [
    column
    for column in target_columns
    if column not in df.columns
]

if missing_features:
    print()
    print("ERROR: Missing feature columns:")
    for column in missing_features:
        print(" -", column)
    sys.exit(1)

if missing_targets:
    print()
    print("ERROR: Missing target columns:")
    for column in missing_targets:
        print(" -", column)
    sys.exit(1)

print()
print("Feature columns:", len(feature_columns))
print("Target columns :", len(target_columns))

# ------------------------------------------------------------
# 6. SELECT DATA
# ------------------------------------------------------------

keep_columns = (
    ["Date"]
    + feature_columns
    + target_columns
)

ml_df = df[keep_columns].copy()

# ------------------------------------------------------------
# 7. CLEAN INFINITE VALUES
# ------------------------------------------------------------

ml_df = ml_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ------------------------------------------------------------
# 8. REPORT MISSING VALUES BEFORE CLEANING
# ------------------------------------------------------------

print()
print("Missing values before cleaning:")

missing_before = ml_df.isna().sum()

print(
    missing_before[
        missing_before > 0
    ]
)

# ------------------------------------------------------------
# 9. DROP ROWS WITH MISSING FEATURE VALUES
# ------------------------------------------------------------
#
# We only remove rows where predictor features are unavailable.
#
# This removes the initial rolling-indicator startup period.
#
# We also require all target values because these rows are
# intended for supervised learning.
# ------------------------------------------------------------

required_columns = (
    feature_columns
    + target_columns
)

before_rows = len(ml_df)

ml_df = ml_df.dropna(
    subset=required_columns
).reset_index(drop=True)

removed_rows = before_rows - len(ml_df)

print()
print("Rows removed because of missing ML values:", removed_rows)

# ------------------------------------------------------------
# 10. DUPLICATE DATE CHECK
# ------------------------------------------------------------

duplicate_dates = ml_df["Date"].duplicated().sum()

print()
print("Duplicate dates:", duplicate_dates)

if duplicate_dates > 0:
    print("ERROR: Duplicate dates detected.")
    sys.exit(1)

# ------------------------------------------------------------
# 11. SORT AGAIN
# ------------------------------------------------------------

ml_df = ml_df.sort_values(
    "Date"
).reset_index(drop=True)

# ------------------------------------------------------------
# 12. SAVE
# ------------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# 13. FINAL REPORT
# ------------------------------------------------------------

print()
print("=" * 60)
print("CLEAN ML DATASET CREATED")
print("=" * 60)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("Rows:", len(ml_df))
print("Columns:", len(ml_df.columns))

print()
print("Date range:")
print("First:", ml_df["Date"].min().date())
print("Last :", ml_df["Date"].max().date())

print()
print("Remaining missing values:")

remaining_missing = ml_df.isna().sum()

if remaining_missing.sum() == 0:
    print("NONE")
else:
    print(
        remaining_missing[
            remaining_missing > 0
        ]
    )

# ------------------------------------------------------------
# 14. NEWS COVERAGE
# ------------------------------------------------------------

news_days = int(
    ml_df["Has_News"].sum()
)

total_days = len(ml_df)

coverage = (
    news_days / total_days * 100
    if total_days > 0
    else 0
)

print()
print("News coverage:")
print("Trading rows:", total_days)
print("Rows with news:", news_days)
print(
    "News coverage: {:.2f} %".format(
        coverage
    )
)

# ------------------------------------------------------------
# 15. TARGET DISTRIBUTION
# ------------------------------------------------------------

print()
print("Target Direction 1D distribution:")

print(
    ml_df[
        "Target_Direction_1D"
    ].value_counts()
    .sort_index()
)

print()
print("Target Direction 5D distribution:")

print(
    ml_df[
        "Target_Direction_5D"
    ].value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 16. LATEST DATA
# ------------------------------------------------------------

print()
print("Latest rows:")

display_columns = [
    "Date",
    "Close",
    "News_Sentiment",
    "News_Count",
    "Has_News",
    "RSI_14",
    "MACD",
    "Momentum_20",
    "Target_Return_1D",
    "Target_Direction_1D",
]

print(
    ml_df[
        display_columns
    ].tail(10).to_string(index=False)
)

# ------------------------------------------------------------
# 17. FINAL PASS
# ------------------------------------------------------------

print()
print("=" * 60)
print("PASS: CLEAN ML DATASET READY")
print("=" * 60)