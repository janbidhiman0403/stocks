import os
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# ALPHALENS - TCS HISTORICAL NEWS DOWNLOADER
# ============================================================

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

TICKER = "TCS.NSE"

START_DATE = "2024-01-01"
END_DATE = "2026-08-21"

OUTPUT = Path("data/raw_tcs_news.csv")

BASE_URL = "https://www.alphavantage.co/query"

print("=" * 70)
print("ALPHALENS TCS NEWS DOWNLOADER")
print("=" * 70)

# ------------------------------------------------------------
# API KEY CHECK
# ------------------------------------------------------------

if not API_KEY:
    raise ValueError(
        "ALPHA_VANTAGE_API_KEY environment variable is not set."
    )

print("API key loaded: YES")
print("Ticker:", TICKER)
print("Date range:", START_DATE, "to", END_DATE)

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Convert dates
# ------------------------------------------------------------

start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

all_articles = []

current = start

# ------------------------------------------------------------
# Download in monthly windows
# ------------------------------------------------------------

while current <= end:

    month_end = min(
        current + timedelta(days=30),
        end
    )

    time_from = current.strftime("%Y%m%dT0000")
    time_to = month_end.strftime("%Y%m%dT2359")

    print("\n" + "-" * 70)
    print("Downloading:")
    print("From:", time_from)
    print("To  :", time_to)

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": TICKER,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "EARLIEST",
        "limit": 1000,
        "apikey": API_KEY
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

    except Exception as e:
        print("ERROR:", e)
        current = month_end + timedelta(days=1)
        continue

    # --------------------------------------------------------
    # API error handling
    # --------------------------------------------------------

    if "Error Message" in data:
        print("API ERROR:")
        print(data["Error Message"])

        current = month_end + timedelta(days=1)
        continue

    if "Note" in data:
        print("API LIMIT / NOTE:")
        print(data["Note"])
        print("\nStopping.")
        break

    if "Information" in data:
        print("API INFORMATION:")
        print(data["Information"])
        print("\nStopping.")
        break

    feed = data.get("feed", [])

    print("Articles:", len(feed))

    # --------------------------------------------------------
    # Extract article metadata
    # --------------------------------------------------------

    for article in feed:

        row = {
            "title": article.get("title"),
            "summary": article.get("summary"),
            "url": article.get("url"),
            "source": article.get("source"),
            "published_at": article.get("time_published"),
            "authors": article.get("authors"),
            "category": article.get("category"),
            "overall_sentiment_score":
                article.get("overall_sentiment_score"),
            "overall_sentiment_label":
                article.get("overall_sentiment_label"),
            "ticker_relevance": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None
        }

        # ----------------------------------------------------
        # Ticker-specific sentiment
        # ----------------------------------------------------

        for item in article.get("ticker_sentiment", []):

            symbol = str(
                item.get("ticker", "")
            ).upper()

            if symbol in ["TCS.NSE", "TCS"]:

                row["ticker_relevance"] = item.get(
                    "relevance_score"
                )

                row["ticker_sentiment_score"] = item.get(
                    "ticker_sentiment_score"
                )

                row["ticker_sentiment_label"] = item.get(
                    "ticker_sentiment_label"
                )

                break

        all_articles.append(row)

    # --------------------------------------------------------
    # Move to next window
    # --------------------------------------------------------

    current = month_end + timedelta(days=1)

    time.sleep(1)

# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING NEWS DATA")
print("=" * 70)

if not all_articles:

    raise RuntimeError(
        "No news articles were downloaded.\n"
        "Check the API key, ticker, date range, "
        "and Alpha Vantage API limits."
    )

news = pd.DataFrame(all_articles)

# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

before = len(news)

news = news.drop_duplicates(
    subset=["url"],
    keep="first"
)

after = len(news)

print("Duplicates removed:", before - after)

# ------------------------------------------------------------
# Parse timestamps
# ------------------------------------------------------------

news["published_at"] = pd.to_datetime(
    news["published_at"],
    errors="coerce",
    format="%Y%m%dT%H%M%S",
    utc=True
)

# ------------------------------------------------------------
# Remove completely invalid timestamps
# ------------------------------------------------------------

invalid = news["published_at"].isna().sum()

print("Invalid timestamps:", invalid)

news = news.dropna(
    subset=["published_at"]
).copy()

# ------------------------------------------------------------
# Sort chronologically
# ------------------------------------------------------------

news = news.sort_values(
    "published_at"
).reset_index(drop=True)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

news.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("Output :", OUTPUT)
print("Rows   :", len(news))
print("Columns:", len(news.columns))

print(
    "First:",
    news["published_at"].min()
)

print(
    "Last :",
    news["published_at"].max()
)

print(
    "Valid timestamps:",
    news["published_at"].notna().sum()
)

print("\nSources:")
print(
    news["source"]
    .value_counts()
    .head(15)
    .to_string()
)

print("\nSentiment labels:")
print(
    news["ticker_sentiment_label"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nSample:")
print(
    news[
        [
            "published_at",
            "source",
            "title",
            "ticker_sentiment_score"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("PASS: RAW NEWS DATASET CREATED")
print("=" * 70)