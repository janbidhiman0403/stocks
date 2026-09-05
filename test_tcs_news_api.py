import os
import requests
import json

print("=" * 70)
print("ALPHALENS TCS NEWS API DIAGNOSTIC")
print("=" * 70)

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ALPHA_VANTAGE_API_KEY environment variable is not set."
    )

print("API key loaded: YES")
print("Ticker: TCS")
print("Period: January 2026")

url = "https://www.alphavantage.co/query"

params = {
    "function": "NEWS_SENTIMENT",
    "tickers": "TCS",
    "time_from": "20260101T0000",
    "time_to": "20260131T2359",
    "sort": "EARLIEST",
    "limit": 10,
    "apikey": API_KEY,
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

print("HTTP status:", response.status_code)

response.raise_for_status()

data = response.json()

print("\nRESPONSE KEYS:")
print(list(data.keys()))

if "Note" in data:
    print("\nAPI NOTE:")
    print(data["Note"])

if "Error Message" in data:
    print("\nAPI ERROR:")
    print(data["Error Message"])

feed = data.get("feed", [])

print("\n" + "=" * 70)
print("NEWS RESULT")
print("=" * 70)

print("Articles returned:", len(feed))

if not feed:
    print("\nNo articles returned.")

    print("\nFull API response:")
    print(json.dumps(data, indent=2)[:5000])

else:

    for i, article in enumerate(feed, 1):

        print("\n" + "-" * 70)
        print(f"ARTICLE {i}")

        print("Title:")
        print(article.get("title"))

        print("Published:")
        print(article.get("time_published"))

        print("Source:")
        print(article.get("source"))

        print("URL:")
        print(article.get("url"))

        print("Overall sentiment:")
        print(article.get("overall_sentiment_score"))

        print("Ticker sentiment:")

        found = False

        for item in article.get("ticker_sentiment", []):

            ticker = item.get("ticker", "")

            if ticker.upper() == "TCS":

                found = True

                print(
                    "  Relevance:",
                    item.get("relevance_score")
                )

                print(
                    "  Sentiment:",
                    item.get("ticker_sentiment_score")
                )

                print(
                    "  Label:",
                    item.get("ticker_sentiment_label")
                )

        if not found:
            print("  TCS ticker sentiment not found.")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)