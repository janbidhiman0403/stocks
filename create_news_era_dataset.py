import pandas as pd
from pathlib import Path

INPUT = Path("data/ml_ready_TCS_news_clean.csv")
OUTPUT = Path("data/ml_ready_TCS_news_era.csv")

print("=" * 60)
print("ALPHALENS NEWS-ERA DATASET")
print("=" * 60)

print("\nLoading:", INPUT)

df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])

print("Rows:", len(df))
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())

# ------------------------------------------------------------
# NEWS ERA
# ------------------------------------------------------------
# The diagnostic showed that meaningful news coverage starts
# in 2024. Earlier data contains essentially no news.
#
# We therefore create a dedicated news-era dataset while
# preserving the original clean dataset.
# ------------------------------------------------------------

NEWS_ERA_START = pd.Timestamp("2024-01-01")

era = df[df["Date"] >= NEWS_ERA_START].copy()

print("\nNews-era start:", NEWS_ERA_START.date())
print("News-era rows:", len(era))
print(
    "News-era dates:",
    era["Date"].min().date(),
    "to",
    era["Date"].max().date()
)

# ------------------------------------------------------------
# VERIFY NEWS COVERAGE
# ------------------------------------------------------------

news_days = int(era["Has_News"].sum())
total_days = len(era)
coverage = news_days / total_days * 100 if total_days else 0

print("\n" + "=" * 60)
print("NEWS COVERAGE")
print("=" * 60)

print("Trading days :", total_days)
print("News days    :", news_days)
print("Coverage     : {:.2f}%".format(coverage))

# ------------------------------------------------------------
# YEARLY COVERAGE
# ------------------------------------------------------------

era["Year"] = era["Date"].dt.year

year_stats = (
    era.groupby("Year")
    .agg(
        Rows=("Date", "size"),
        News_Days=("Has_News", "sum"),
        Total_News=("News_Count", "sum"),
        Avg_Sentiment=("News_Sentiment", "mean"),
    )
)

year_stats["Coverage_%"] = (
    year_stats["News_Days"] / year_stats["Rows"] * 100
)

print("\nYEARLY NEWS COVERAGE")
print(year_stats.round(3).to_string())

# Remove temporary Year column
era.drop(columns=["Year"], inplace=True)

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\n1D:")
print(era["Target_Direction_1D"].value_counts().sort_index())

print("\n5D:")
print(era["Target_Direction_5D"].value_counts().sort_index())

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

era.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("NEWS-ERA DATASET CREATED")
print("=" * 60)

print("\nOutput:")
print(OUTPUT)

print("Rows:", len(era))
print(
    "Date range:",
    era["Date"].min().date(),
    "to",
    era["Date"].max().date()
)

print("\nPASS: News-era dataset ready.")
print("=" * 60)