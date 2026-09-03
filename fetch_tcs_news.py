import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

OUTPUT_FILE = Path("data/news/TCS_news.csv")

queries = [
    "TCS Tata Consultancy Services 2024",
    "TCS Tata Consultancy Services 2025",
    "TCS Tata Consultancy Services 2026"
]

articles = []

print("Fetching expanded TCS news history...")

for query in queries:

    rss_url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    print("\nQuery:", query)

    feed = feedparser.parse(rss_url)

    print("Articles found:", len(feed.entries))

    for entry in feed.entries:

        title = entry.get("title", "").strip()
        url = entry.get("link", "").strip()

        source = ""

        if "source" in entry:
            source = entry.source.get("title", "").strip()

        published = entry.get("published", "")

        try:
            published_date = pd.to_datetime(published).date()
        except:
            published_date = datetime.now().date()

        if title:
            articles.append({
                "Date": published_date,
                "Headline": title,
                "Source": source,
                "URL": url
            })


new_data = pd.DataFrame(articles)

# Remove duplicate URLs
new_data = new_data.drop_duplicates(subset=["URL"])

# Load existing data if present
if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:

    existing_data = pd.read_csv(OUTPUT_FILE)

    combined = pd.concat(
        [existing_data, new_data],
        ignore_index=True
    )

    combined = combined.drop_duplicates(subset=["URL"])

else:

    combined = new_data


combined["Date"] = pd.to_datetime(
    combined["Date"]
).dt.date

combined = combined.sort_values("Date")

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nExpanded news dataset saved!")
print("File:", OUTPUT_FILE)
print("Total unique articles:", len(combined))

print("\nDate range:")
print("First:", combined["Date"].min())
print("Last:", combined["Date"].max())