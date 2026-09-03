import pandas as pd

PRICE_FILE = "data/ml_ready_TCS_v3.csv"
NEWS_FILE = "data/news/TCS_daily_sentiment.csv"

print("Loading price and news data...")

price = pd.read_csv(PRICE_FILE)
news = pd.read_csv(NEWS_FILE)

price["Date"] = pd.to_datetime(price["Date"])
news["Date"] = pd.to_datetime(news["Date"])

# Keep only dates where news exists
merged = price.merge(
    news,
    on="Date",
    how="left"
)

# News coverage statistics
news_days = merged["News_Sentiment"].notna().sum()
total_days = len(merged)

print("\nAlignment results:")
print("Total trading rows:", total_days)
print("Trading days with news:", news_days)
print(
    "News coverage:",
    round(news_days / total_days * 100, 2),
    "%"
)

print("\nMissing news rows:", merged["News_Sentiment"].isna().sum())

print("\nNews coverage by year:")
print(
    merged[merged["News_Sentiment"].notna()]
    .assign(Year=lambda x: x["Date"].dt.year)
    ["Year"]
    .value_counts()
    .sort_index()
)

print("\nSample aligned rows:")
print(
    merged[
        ["Date", "Close", "News_Sentiment", "News_Count"]
    ]
    .dropna(subset=["News_Sentiment"])
    .head(15)
)