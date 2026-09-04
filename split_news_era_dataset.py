import pandas as pd
from pathlib import Path

INPUT = Path("data/ml_ready_TCS_news_era.csv")

TRAIN_OUTPUT = Path("data/train_TCS_news_era.csv")
VAL_OUTPUT = Path("data/validation_TCS_news_era.csv")
TEST_OUTPUT = Path("data/test_TCS_news_era.csv")

print("=" * 60)
print("ALPHALENS NEWS-ERA TIME-SERIES SPLIT")
print("=" * 60)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("\nLoading:", INPUT)

df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("Rows loaded   :", len(df))
print("Columns loaded:", len(df))
print(
    "Date range    :",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

# ------------------------------------------------------------
# DATE CHECKS
# ------------------------------------------------------------

print("\nDuplicate dates:", df["Date"].duplicated().sum())
print("Missing values :", int(df.isna().sum().sum()))

if df["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

if df.isna().any().any():
    raise ValueError("Missing values found.")

# ------------------------------------------------------------
# TIME SPLIT
# ------------------------------------------------------------
#
# Train:
#   2024-01-01 through 2025-12-31
#
# Validation:
#   2026-01-01 through 2026-05-31
#
# Test:
#   2026-06-01 onward
#
# This gives the model historical exposure to the news
# feature distribution before evaluating on later periods.
# ------------------------------------------------------------

train_end = pd.Timestamp("2025-12-31")
validation_end = pd.Timestamp("2026-05-31")

train = df[df["Date"] <= train_end].copy()

validation = df[
    (df["Date"] > train_end)
    & (df["Date"] <= validation_end)
].copy()

test = df[df["Date"] > validation_end].copy()

# ------------------------------------------------------------
# VALIDATE SPLIT
# ------------------------------------------------------------

if len(train) == 0:
    raise ValueError("Training dataset is empty.")

if len(validation) == 0:
    raise ValueError("Validation dataset is empty.")

if len(test) == 0:
    raise ValueError("Test dataset is empty.")

if train["Date"].max() >= validation["Date"].min():
    raise ValueError("Train/validation overlap detected.")

if validation["Date"].max() >= test["Date"].min():
    raise ValueError("Validation/test overlap detected.")

# ------------------------------------------------------------
# DISPLAY SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TIME-SERIES SPLIT")
print("=" * 60)

print("\nTRAIN")
print("-" * 60)
print("Rows       :", len(train))
print("First date :", train["Date"].min().date())
print("Last date  :", train["Date"].max().date())

print("\nVALIDATION")
print("-" * 60)
print("Rows       :", len(validation))
print("First date :", validation["Date"].min().date())
print("Last date  :", validation["Date"].max().date())

print("\nTEST")
print("-" * 60)
print("Rows       :", len(test))
print("First date :", test["Date"].min().date())
print("Last date  :", test["Date"].max().date())

# ------------------------------------------------------------
# NEWS COVERAGE FUNCTION
# ------------------------------------------------------------

def news_stats(name, data):
    total = len(data)
    news_days = int(data["Has_News"].sum())
    coverage = news_days / total * 100 if total else 0

    print(
        f"{name:<12} "
        f"rows={total:<5} "
        f"news_days={news_days:<4} "
        f"coverage={coverage:.2f}%"
    )

# ------------------------------------------------------------
# NEWS COVERAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NEWS COVERAGE")
print("=" * 60)

news_stats("TRAIN", train)
news_stats("VALIDATION", validation)
news_stats("TEST", test)

# ------------------------------------------------------------
# NEWS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NEWS DISTRIBUTION")
print("=" * 60)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print("-" * 40)

    print(
        "Sentiment mean:",
        round(data["News_Sentiment"].mean(), 4)
    )

    print(
        "Sentiment std :",
        round(data["News_Sentiment"].std(), 4)
    )

    print(
        "News count avg:",
        round(data["News_Count"].mean(), 4)
    )

    print(
        "News days     :",
        int(data["Has_News"].sum())
    )

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

for name, data in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print("-" * 40)

    print("1D:")
    print(
        data["Target_Direction_1D"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n5D:")
    print(
        data["Target_Direction_5D"]
        .value_counts()
        .sort_index()
        .to_string()
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

train.to_csv(TRAIN_OUTPUT, index=False)
validation.to_csv(VAL_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)

# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print("\nTRAIN:")
print(TRAIN_OUTPUT)

print("\nVALIDATION:")
print(VAL_OUTPUT)

print("\nTEST:")
print(TEST_OUTPUT)

print("\n" + "=" * 60)
print("PASS: NEWS-ERA DATASETS READY")
print("=" * 60)