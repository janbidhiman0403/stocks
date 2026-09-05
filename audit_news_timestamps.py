import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("ALPHALENS NEWS TIMESTAMP / FEATURE ALIGNMENT AUDIT")
print("=" * 70)

# ============================================================
# CONFIG
# ============================================================

PRICE_FILE = Path("data/ml_ready_TCS_news_clean.csv")

# Try likely raw-news files automatically
CANDIDATE_NEWS_FILES = [
    Path("data/news.csv"),
    Path("data/tcs_news.csv"),
    Path("data/TCS_news.csv"),
    Path("data/raw_news.csv"),
    Path("data/news_data.csv"),
    Path("data/tcs_news_data.csv"),
]

# ============================================================
# LOAD PRICE DATA
# ============================================================

if not PRICE_FILE.exists():
    raise FileNotFoundError(f"Missing: {PRICE_FILE}")

price = pd.read_csv(PRICE_FILE)

print("\nPRICE DATA")
print("-" * 70)
print("Rows:", len(price))
print("Columns:", len(price.columns))

price["Date"] = pd.to_datetime(price["Date"], errors="coerce")

print("Date range:",
      price["Date"].min().date(),
      "to",
      price["Date"].max().date())

# ============================================================
# FIND NEWS DATA
# ============================================================

news_file = None

for candidate in CANDIDATE_NEWS_FILES:
    if candidate.exists():
        news_file = candidate
        break

if news_file is None:

    # Search all CSV files in data/
    csv_files = list(Path("data").glob("*.csv"))

    print("\nNo standard news filename found.")
    print("\nCSV files found in data/:")

    for f in csv_files:
        print(" -", f)

    print("\nIMPORTANT:")
    print("Open the raw news CSV and identify its filename.")
    print("Then change CANDIDATE_NEWS_FILES at the top of this script.")

    raise FileNotFoundError(
        "Raw news dataset could not be located automatically."
    )

# ============================================================
# LOAD NEWS
# ============================================================

news = pd.read_csv(news_file)

print("\nNEWS DATA")
print("-" * 70)
print("File:", news_file)
print("Rows:", len(news))
print("Columns:", len(news.columns))

print("\nNews columns:")

for i, col in enumerate(news.columns, 1):
    print(f"{i:02d}. {col}")

# ============================================================
# DETECT DATE/TIME COLUMNS
# ============================================================

time_candidates = []

for col in news.columns:

    name = col.lower()

    if any(
        keyword in name
        for keyword in [
            "timestamp",
            "datetime",
            "published",
            "publication",
            "pub_date",
            "publish",
            "date",
            "time"
        ]
    ):
        time_candidates.append(col)

print("\nPOSSIBLE NEWS TIME COLUMNS")
print("-" * 70)

if time_candidates:
    for col in time_candidates:
        print(" -", col)
else:
    print("NONE FOUND")

# ============================================================
# DETECT TITLE / TEXT COLUMNS
# ============================================================

text_candidates = []

for col in news.columns:

    name = col.lower()

    if any(
        keyword in name
        for keyword in [
            "title",
            "headline",
            "description",
            "summary",
            "content",
            "text",
            "article"
        ]
    ):
        text_candidates.append(col)

print("\nPOSSIBLE NEWS TEXT COLUMNS")
print("-" * 70)

if text_candidates:
    for col in text_candidates:
        print(" -", col)
else:
    print("NONE FOUND")

# ============================================================
# SAMPLE DATA
# ============================================================

print("\nNEWS SAMPLE")
print("-" * 70)

print(news.head(5).to_string())

# ============================================================
# TIMESTAMP ANALYSIS
# ============================================================

if time_candidates:

    best_time_col = None
    best_valid = -1

    for col in time_candidates:

        parsed = pd.to_datetime(
            news[col],
            errors="coerce",
            utc=True
        )

        valid = parsed.notna().sum()

        if valid > best_valid:
            best_valid = valid
            best_time_col = col

    print("\nSELECTED TIME COLUMN")
    print("-" * 70)
    print("Column:", best_time_col)
    print("Valid timestamps:", best_valid)
    print(
        "Timestamp coverage:",
        f"{best_valid / len(news) * 100:.2f}%"
    )

    news["_timestamp"] = pd.to_datetime(
        news[best_time_col],
        errors="coerce",
        utc=True
    )

    print("\nNEWS TIMESTAMP RANGE")
    print("-" * 70)

    valid_times = news["_timestamp"].dropna()

    if len(valid_times):

        print("First:", valid_times.min())
        print("Last :", valid_times.max())

        print("\nTimezone information:")
        print(valid_times.dt.tz)

# ============================================================
# DAILY NEWS DISTRIBUTION
# ============================================================

if "_timestamp" in news.columns:

    valid_news = news.dropna(subset=["_timestamp"]).copy()

    valid_news["News_Date"] = (
        valid_news["_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    daily_news = (
        valid_news
        .groupby("News_Date")
        .size()
        .reset_index(name="News_Count")
    )

    print("\nDAILY NEWS DISTRIBUTION")
    print("-" * 70)

    print("News days:", len(daily_news))
    print("Average articles/news day:",
          round(daily_news["News_Count"].mean(), 3))
    print("Maximum articles/day:",
          daily_news["News_Count"].max())

    print("\nTop news days:")

    print(
        daily_news
        .sort_values("News_Count", ascending=False)
        .head(20)
        .to_string(index=False)
    )

# ============================================================
# CHECK WHETHER NEWS EXISTS BEFORE 2024
# ============================================================

if "_timestamp" in news.columns:

    news_dates = (
        news["_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    print("\nNEWS BY YEAR")
    print("-" * 70)

    print(
        news_dates
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )

# ============================================================
# PRICE/NEWS OVERLAP
# ============================================================

if "_timestamp" in news.columns:

    news_dates = (
        news["_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.normalize()
        .dt.tz_localize(None)
    )

    price_dates = set(price["Date"].dropna())

    news_dates_set = set(news_dates.dropna())

    overlap = price_dates.intersection(news_dates_set)

    print("\nPRICE / NEWS DATE OVERLAP")
    print("-" * 70)
    print("Trading dates:", len(price_dates))
    print("News dates   :", len(news_dates_set))
    print("Overlap      :", len(overlap))

    if len(price_dates):

        print(
            "Overlap % of trading dates:",
            f"{len(overlap) / len(price_dates) * 100:.2f}%"
        )

# ============================================================
# WEEKEND NEWS
# ============================================================

if "_timestamp" in news.columns:

    news_dates = (
        news["_timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    weekday_counts = (
        news_dates
        .dt.day_name()
        .value_counts()
    )

    print("\nNEWS BY WEEKDAY")
    print("-" * 70)

    print(weekday_counts.to_string())

# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSIS")
print("=" * 70)

if not time_candidates:

    print("""
FAIL:
No usable news timestamp column was detected.

Without publication timestamps we cannot prove that the
news features available on a trading day were actually
known before the prediction point.
""")

elif best_valid / len(news) < 0.90:

    print("""
WARNING:
A significant percentage of news records have invalid
timestamps.

Timestamp cleaning is required before model training.
""")

else:

    print("""
PASS:
A usable news timestamp column was detected.

NEXT STEP:
We should rebuild the news features using strict
point-in-time alignment.
""")

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)