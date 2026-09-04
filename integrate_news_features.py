import pandas as pd
from pathlib import Path

# ============================================================
# ALPHALENS - NEWS FEATURE INTEGRATION
# ============================================================

PRICE_FILE = Path("data/raw/TCS.csv")
NEWS_FILE = Path("data/news/TCS_news_aligned.csv")
OUTPUT_FILE = Path("data/ml_ready_TCS_news.csv")


print("=" * 60)
print("ALPHALENS NEWS FEATURE INTEGRATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. CHECK FILES
# ------------------------------------------------------------

print("\nChecking input files...")

if not PRICE_FILE.exists():
    raise FileNotFoundError(
        f"Price file not found: {PRICE_FILE}"
    )

if not NEWS_FILE.exists():
    raise FileNotFoundError(
        f"News file not found: {NEWS_FILE}"
    )

print("Price file :", PRICE_FILE)
print("News file  :", NEWS_FILE)


# ------------------------------------------------------------
# 2. LOAD PRICE DATA
# ------------------------------------------------------------

print("\nLoading price data...")

price_df = pd.read_csv(PRICE_FILE)

print("Price columns:")
print(list(price_df.columns))
print("Price rows:", len(price_df))


# ------------------------------------------------------------
# 3. HANDLE PRICE DATE COLUMN
# ------------------------------------------------------------

# Your TCS.csv currently uses the first column as the date.
# The column is named "Price".

if "Date" in price_df.columns:

    price_df["Date"] = pd.to_datetime(
        price_df["Date"],
        errors="coerce"
    )

else:

    print("\nPrice dataset has no Date column.")
    print("Using 'Price' column as the date column.")

    price_df = price_df.rename(
        columns={"Price": "Date"}
    )

    price_df["Date"] = pd.to_datetime(
        price_df["Date"],
        errors="coerce"
    )


# Remove invalid dates

invalid_price_dates = price_df["Date"].isna().sum()

if invalid_price_dates > 0:
    print(
        f"Removing {invalid_price_dates} rows "
        "with invalid price dates."
    )

    price_df = price_df.dropna(
        subset=["Date"]
    )


# Remove duplicate dates

duplicate_price_dates = price_df["Date"].duplicated().sum()

if duplicate_price_dates > 0:

    print(
        f"Removing {duplicate_price_dates} "
        "duplicate price dates."
    )

    price_df = price_df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )


# ------------------------------------------------------------
# 4. LOAD NEWS DATA
# ------------------------------------------------------------

print("\nLoading news data...")

news_df = pd.read_csv(NEWS_FILE)

print("News columns:")
print(list(news_df.columns))
print("News rows:", len(news_df))


# ------------------------------------------------------------
# 5. VALIDATE NEWS COLUMNS
# ------------------------------------------------------------

required_news_columns = [
    "Date",
    "News_Sentiment",
    "News_Count"
]

for column in required_news_columns:

    if column not in news_df.columns:

        raise ValueError(
            f"News dataset is missing required column: "
            f"{column}"
        )


# ------------------------------------------------------------
# 6. CLEAN NEWS DATES
# ------------------------------------------------------------

news_df["Date"] = pd.to_datetime(
    news_df["Date"],
    errors="coerce"
)

invalid_news_dates = news_df["Date"].isna().sum()

if invalid_news_dates > 0:

    print(
        f"Removing {invalid_news_dates} "
        "news rows with invalid dates."
    )

    news_df = news_df.dropna(
        subset=["Date"]
    )


# ------------------------------------------------------------
# 7. CLEAN NEWS FEATURES
# ------------------------------------------------------------

news_df["News_Sentiment"] = pd.to_numeric(
    news_df["News_Sentiment"],
    errors="coerce"
)

news_df["News_Count"] = pd.to_numeric(
    news_df["News_Count"],
    errors="coerce"
)

news_df["News_Sentiment"] = (
    news_df["News_Sentiment"].fillna(0.0)
)

news_df["News_Count"] = (
    news_df["News_Count"].fillna(0).astype(int)
)


# ------------------------------------------------------------
# 8. REMOVE DUPLICATE NEWS DATES
# ------------------------------------------------------------

duplicate_news_dates = news_df["Date"].duplicated().sum()

if duplicate_news_dates > 0:

    print(
        f"WARNING: {duplicate_news_dates} duplicate "
        "news dates found."
    )

    # If duplicates exist, aggregate them safely.

    news_df = (
        news_df
        .groupby("Date", as_index=False)
        .agg({
            "News_Sentiment": "mean",
            "News_Count": "sum"
        })
    )


# ------------------------------------------------------------
# 9. SORT DATA
# ------------------------------------------------------------

price_df = price_df.sort_values(
    "Date"
).reset_index(drop=True)

news_df = news_df.sort_values(
    "Date"
).reset_index(drop=True)


# ------------------------------------------------------------
# 10. MERGE PRICE + NEWS
# ------------------------------------------------------------

print("\nMerging price and news data...")

merged_df = pd.merge(
    price_df,
    news_df[
        [
            "Date",
            "News_Sentiment",
            "News_Count"
        ]
    ],
    on="Date",
    how="left"
)


# ------------------------------------------------------------
# 11. CREATE NEWS INDICATOR
# ------------------------------------------------------------

merged_df["Has_News"] = (
    merged_df["News_Count"].fillna(0) > 0
).astype(int)


# ------------------------------------------------------------
# 12. FILL DAYS WITHOUT NEWS
# ------------------------------------------------------------

merged_df["News_Sentiment"] = (
    merged_df["News_Sentiment"]
    .fillna(0.0)
)

merged_df["News_Count"] = (
    merged_df["News_Count"]
    .fillna(0)
    .astype(int)
)


# ------------------------------------------------------------
# 13. CREATE ADDITIONAL NEWS FEATURES
# ------------------------------------------------------------

# Previous trading day's news sentiment

merged_df["News_Sentiment_Lag1"] = (
    merged_df["News_Sentiment"].shift(1)
)


# 3-day rolling news sentiment

merged_df["News_Sentiment_MA3"] = (
    merged_df["News_Sentiment"]
    .rolling(window=3, min_periods=1)
    .mean()
)


# 5-day rolling news sentiment

merged_df["News_Sentiment_MA5"] = (
    merged_df["News_Sentiment"]
    .rolling(window=5, min_periods=1)
    .mean()
)


# 3-day news count

merged_df["News_Count_MA3"] = (
    merged_df["News_Count"]
    .rolling(window=3, min_periods=1)
    .mean()
)


# News sentiment change

merged_df["News_Sentiment_Change"] = (
    merged_df["News_Sentiment"]
    .diff()
)


# Fill first-row NaN values

merged_df["News_Sentiment_Lag1"] = (
    merged_df["News_Sentiment_Lag1"]
    .fillna(0.0)
)

merged_df["News_Sentiment_Change"] = (
    merged_df["News_Sentiment_Change"]
    .fillna(0.0)
)


# ------------------------------------------------------------
# 14. REORDER COLUMNS
# ------------------------------------------------------------

base_columns = [
    "Date",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

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


# Only include columns that actually exist.

final_columns = [
    column
    for column in base_columns + news_columns
    if column in merged_df.columns
]

# Preserve any additional original columns

remaining_columns = [
    column
    for column in merged_df.columns
    if column not in final_columns
]

merged_df = merged_df[
    final_columns + remaining_columns
]


# ------------------------------------------------------------
# 15. SAVE OUTPUT
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. REPORT
# ------------------------------------------------------------

trading_days = len(merged_df)

days_with_news = (
    merged_df["Has_News"].sum()
)

coverage = (
    days_with_news / trading_days * 100
    if trading_days > 0
    else 0
)


print("\n" + "=" * 60)
print("INTEGRATION COMPLETE")
print("=" * 60)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nRows:", len(merged_df))

print(
    "Date range:",
    merged_df["Date"].min().date(),
    "to",
    merged_df["Date"].max().date()
)

print("\nNews alignment:")
print("Trading days:", trading_days)
print("Trading days with news:", days_with_news)
print("News coverage: {:.2f} %".format(coverage))

print("\nNews feature columns:")

for column in news_columns:

    if column in merged_df.columns:
        print(" -", column)


print("\nMissing values in news features:")

print(
    merged_df[news_columns]
    .isna()
    .sum()
)


print("\nLatest rows:")

print(
    merged_df[
        [
            "Date",
            "Close",
            "News_Sentiment",
            "News_Count",
            "Has_News"
        ]
    ]
    .tail(20)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("PASS: News features successfully integrated.")
print("=" * 60)