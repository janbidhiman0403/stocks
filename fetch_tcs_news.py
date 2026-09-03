import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

RSS_URL = "https://news.google.com/rss/search?q=TCS+Tata+Consultancy+Services&hl=en-IN&gl=IN&ceid=IN:en"

OUTPUT_FILE = Path("data/news/TCS_news.csv")

# ==============================
# FETCH NEWS
# ==============================

print("Fetching TCS news...")

feed = feedparser.parse(RSS_URL)

print("Articles found:", len(feed.entries))

# ==============================
# EXTRACT ARTICLES
# ==============================

articles = []

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

# ==============================
# SAVE CSV
# ==============================

new_data = pd.DataFrame(articles)

if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:

    existing_data = pd.read_csv(OUTPUT_FILE)

    combined = pd.concat(
        [existing_data, new_data],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["URL"]
    )

else:

    combined = new_data

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)
print("Total unique articles:", len(combined))