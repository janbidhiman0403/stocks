import pandas as pd
from pathlib import Path


# ============================================================
# AlphaLens - TCS News / Price Alignment Check
# ============================================================

PRICE_FILE = "data/raw/TCS.csv"
NEWS_FILE = "data/news/TCS_daily_sentiment.csv"
OUTPUT_FILE = "data/news/TCS_news_aligned.csv"


print("Loading price and news data...")


# ============================================================
# 1. Check files
# ============================================================

if not Path(PRICE_FILE).exists():
    raise FileNotFoundError(f"Price file not found: {PRICE_FILE}")

if not Path(NEWS_FILE).exists():
    raise FileNotFoundError(f"News file not found: {NEWS_FILE}")


# ============================================================
# 2. Load datasets
# ============================================================

price_df = pd.read_csv(PRICE_FILE)
news_df = pd.read_csv(NEWS_FILE)


print("\nPrice columns:")
print(list(price_df.columns))

print("\nNews columns:")
print(list(news_df.columns))

print("\nPrice rows:", len(price_df))
print("News rows:", len(news_df))


# ============================================================
# 3. Price date handling
# ============================================================

# Your TCS.csv uses "Price" as the date column.
if "Date" in price_df.columns:
    price_df["Date"] = pd.to_datetime(
        price_df["Date"],
        errors="coerce"
    ).dt.normalize()

elif "Price" in price_df.columns:
    print("\nPrice column detected as date/index column.")

    price_df["Date"] = pd.to_datetime(
        price_df["Price"],
        errors="coerce"
    ).dt.normalize()

else:
    raise ValueError(
        "Price dataset contains neither 'Date' nor 'Price' column."
    )


# ============================================================
# 4. News date handling
# ============================================================

if "Date" not in news_df.columns:
    raise ValueError(
        "News dataset does not contain a 'Date' column."
    )

news_df["Date"] = pd.to_datetime(
    news_df["Date"],
    errors="coerce"
).dt.normalize()


# ============================================================
# 5. Validate news columns
# ============================================================

required_news_columns = [
    "News_Sentiment",
    "News_Count"
]

for column in required_news_columns:
    if column not in news_df.columns:
        raise ValueError(
            f"News dataset missing required column: {column}"
        )


# ============================================================
# 6. Remove invalid dates
# ============================================================

invalid_price_dates = price_df["Date"].isna().sum()
invalid_news_dates = news_df["Date"].isna().sum()

if invalid_price_dates > 0:
    print(
        f"\nWARNING: Removing {invalid_price_dates} "
        "price rows with invalid dates."
    )

if invalid_news_dates > 0:
    print(
        f"WARNING: Removing {invalid_news_dates} "
        "news rows with invalid dates."
    )

price_df = price_df.dropna(subset=["Date"]).copy()
news_df = news_df.dropna(subset=["Date"]).copy()


# ============================================================
# 7. Sort
# ============================================================

price_df = price_df.sort_values("Date").reset_index(drop=True)
news_df = news_df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 8. Check duplicate news dates
# ============================================================

duplicate_news_dates = news_df["Date"].duplicated().sum()

print("\nDuplicate news dates:", duplicate_news_dates)

if duplicate_news_dates > 0:

    print(
        "Aggregating multiple news records occurring "
        "on the same date..."
    )

    news_df = (
        news_df
        .groupby("Date", as_index=False)
        .agg(
            News_Sentiment=("News_Sentiment", "mean"),
            News_Count=("News_Count", "sum")
        )
    )


# ============================================================
# 9. Date ranges
# ============================================================

print("\nPrice date range:")

print(
    "First:",
    price_df["Date"].min().date()
)

print(
    "Last:",
    price_df["Date"].max().date()
)


print("\nNews date range:")

print(
    "First:",
    news_df["Date"].min().date()
)

print(
    "Last:",
    news_df["Date"].max().date()
)


# ============================================================
# 10. Align news with trading days
# ============================================================

aligned = price_df.merge(
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


# ============================================================
# 11. News availability flag
# ============================================================

aligned["Has_News"] = (
    aligned["News_Sentiment"].notna()
)


# ============================================================
# 12. Fill missing news
# ============================================================

aligned["News_Sentiment"] = (
    aligned["News_Sentiment"].fillna(0.0)
)

aligned["News_Count"] = (
    aligned["News_Count"].fillna(0)
)


# ============================================================
# 13. Alignment statistics
# ============================================================

total_trading_days = len(aligned)

trading_days_with_news = int(
    aligned["Has_News"].sum()
)

missing_news_days = (
    total_trading_days -
    trading_days_with_news
)

if total_trading_days > 0:
    coverage = (
        trading_days_with_news /
        total_trading_days
    ) * 100
else:
    coverage = 0.0


print("\n" + "=" * 60)
print("ALIGNMENT RESULTS")
print("=" * 60)

print(
    "Total trading rows:",
    total_trading_days
)

print(
    "Trading days with news:",
    trading_days_with_news
)

print(
    f"News coverage: {coverage:.2f} %"
)

print(
    "Missing news rows:",
    missing_news_days
)


# ============================================================
# 14. News coverage by year
# ============================================================

print("\n" + "=" * 60)
print("NEWS COVERAGE BY YEAR")
print("=" * 60)

aligned_with_news = aligned[
    aligned["Has_News"]
].copy()

if len(aligned_with_news) > 0:

    coverage_by_year = (
        aligned_with_news["Date"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print(coverage_by_year)

else:
    print("No trading days with news found.")


# ============================================================
# 15. News dataset by year
# ============================================================

print("\n" + "=" * 60)
print("NEWS DATASET BY YEAR")
print("=" * 60)

news_by_year = (
    news_df["Date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print(news_by_year)


# ============================================================
# 16. 2026 alignment
# ============================================================

print("\n" + "=" * 60)
print("2026 NEWS ALIGNMENT")
print("=" * 60)

aligned_2026 = aligned[
    (aligned["Date"].dt.year == 2026)
    &
    (aligned["Has_News"])
].copy()

if len(aligned_2026) > 0:

    print(
        aligned_2026[
            [
                "Date",
                "Close",
                "News_Sentiment",
                "News_Count"
            ]
        ]
        .tail(30)
        .to_string(index=False)
    )

else:

    print(
        "WARNING: No 2026 news is aligned "
        "with the trading dataset."
    )


# ============================================================
# 17. Latest news alignment
# ============================================================

print("\n" + "=" * 60)
print("LATEST NEWS ALIGNMENT CHECK")
print("=" * 60)

latest_news_dates = (
    news_df["Date"]
    .drop_duplicates()
    .sort_values()
    .tail(10)
)

latest_check = aligned[
    aligned["Date"].isin(latest_news_dates)
][
    [
        "Date",
        "Close",
        "News_Sentiment",
        "News_Count",
        "Has_News"
    ]
]

if len(latest_check) > 0:

    print(
        latest_check
        .sort_values("Date")
        .to_string(index=False)
    )

else:

    print(
        "No latest news dates matched "
        "the trading dataset."
    )


# ============================================================
# 18. September 2026 check
# ============================================================

print("\n" + "=" * 60)
print("SEPTEMBER 2026 CHECK")
print("=" * 60)

sept_2026 = aligned[
    (aligned["Date"] >= "2026-09-01")
    &
    (aligned["Date"] <= "2026-09-30")
]

if len(sept_2026) > 0:

    print(
        sept_2026[
            [
                "Date",
                "Close",
                "News_Sentiment",
                "News_Count",
                "Has_News"
            ]
        ]
        .to_string(index=False)
    )

else:

    print(
        "No September 2026 trading rows found."
    )


# ============================================================
# 19. Sample aligned rows
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE ALIGNED ROWS")
print("=" * 60)

sample = aligned[
    aligned["Has_News"]
][
    [
        "Date",
        "Close",
        "News_Sentiment",
        "News_Count"
    ]
].tail(20)

if len(sample) > 0:
    print(
        sample.to_string(index=False)
    )
else:
    print("No aligned rows found.")


# ============================================================
# 20. Save aligned dataset
# ============================================================

aligned.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("OUTPUT")
print("=" * 60)

print(
    "Aligned dataset saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 21. Final status
# ============================================================

print("\n" + "=" * 60)

if trading_days_with_news > 0:
    print(
        "PASS: News successfully aligned "
        "with trading data."
    )
else:
    print(
        "WARNING: No news successfully aligned "
        "with trading data."
    )

print("=" * 60)

print("\nDONE")