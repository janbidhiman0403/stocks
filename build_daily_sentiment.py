import pandas as pd

INPUT_FILE = "data/news/TCS_news_sentiment.csv"
OUTPUT_FILE = "data/news/TCS_daily_sentiment.csv"

print("Loading sentiment dataset...")

df = pd.read_csv(INPUT_FILE)

# Convert sentiment into a signed numerical score
def calculate_sentiment(row):
    score = row["Sentiment_Score"]

    if row["Sentiment"] == "positive":
        return score
    elif row["Sentiment"] == "negative":
        return -score
    else:
        return 0.0


df["Sentiment_Value"] = df.apply(calculate_sentiment, axis=1)

# Aggregate articles by date
daily = (
    df.groupby("Date")
    .agg(
        News_Sentiment=("Sentiment_Value", "mean"),
        News_Count=("Headline", "count")
    )
    .reset_index()
)

# Save daily sentiment dataset
daily.to_csv(OUTPUT_FILE, index=False)

print("\nDaily sentiment created successfully!")
print("Saved:", OUTPUT_FILE)

print("\nNumber of dates:", len(daily))

print("\nDaily sentiment range:")
print("Minimum:", round(daily["News_Sentiment"].min(), 4))
print("Maximum:", round(daily["News_Sentiment"].max(), 4))

print("\nSample:")
print(daily.head(10))