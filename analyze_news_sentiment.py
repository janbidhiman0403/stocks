import pandas as pd
from transformers import pipeline

INPUT_FILE = "data/news/TCS_news.csv"
OUTPUT_FILE = "data/news/TCS_news_sentiment.csv"

print("Loading FinBERT...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT loaded successfully!")

# Load news dataset
df = pd.read_csv(INPUT_FILE)

print("Articles loaded:", len(df))

# Analyze headlines
sentiments = []

for headline in df["Headline"]:
    result = sentiment_pipeline(headline)[0]

    sentiments.append({
        "Sentiment": result["label"],
        "Sentiment_Score": result["score"]
    })

# Add sentiment columns
sentiment_df = pd.DataFrame(sentiments)

df["Sentiment"] = sentiment_df["Sentiment"]
df["Sentiment_Score"] = sentiment_df["Sentiment_Score"]

# Save result
df.to_csv(OUTPUT_FILE, index=False)

print("\nSentiment analysis complete!")
print("Saved:", OUTPUT_FILE)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nSample results:")
print(df[["Date", "Headline", "Sentiment", "Sentiment_Score"]].head(10))