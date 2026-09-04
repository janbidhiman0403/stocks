import pandas as pd
import os

INPUT_FILE = "data/ml_ready_TCS_news_clean.csv"

TRAIN_FILE = "data/train_TCS_news.csv"
VALIDATION_FILE = "data/validation_TCS_news.csv"
TEST_FILE = "data/test_TCS_news.csv"

print("=" * 60)
print("ALPHALENS TIME-SERIES ML DATASET SPLIT")
print("=" * 60)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print()
print("Loading:", INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

if df["Date"].isna().any():
    raise ValueError("Invalid dates found in dataset.")

df = df.sort_values("Date").reset_index(drop=True)

print("Rows loaded   :", len(df))
print("Columns loaded:", len(df.columns))
print(
    "Date range    :",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

# ------------------------------------------------------------
# CHECK DUPLICATES
# ------------------------------------------------------------

duplicates = df["Date"].duplicated().sum()

print()
print("Duplicate dates:", duplicates)

if duplicates > 0:
    raise ValueError(
        "Duplicate dates detected. Fix duplicates before splitting."
    )

# ------------------------------------------------------------
# CHECK MISSING VALUES
# ------------------------------------------------------------

missing_total = df.isna().sum().sum()

print()
print("Missing values:", missing_total)

if missing_total > 0:
    print(df.isna().sum()[df.isna().sum() > 0])
    raise ValueError(
        "Missing values detected. Clean the dataset before splitting."
    )

# ------------------------------------------------------------
# TIME-ORDERED SPLIT
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
# ------------------------------------------------------------

total_rows = len(df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

# ------------------------------------------------------------
# SAVE DATASETS
# ------------------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print()
print("=" * 60)
print("TIME-SERIES SPLIT COMPLETE")
print("=" * 60)

print()
print("TRAIN")
print("-" * 60)
print("Rows       :", len(train_df))
print("First date :", train_df["Date"].min().date())
print("Last date  :", train_df["Date"].max().date())

print()
print("VALIDATION")
print("-" * 60)
print("Rows       :", len(validation_df))
print("First date :", validation_df["Date"].min().date())
print("Last date  :", validation_df["Date"].max().date())

print()
print("TEST")
print("-" * 60)
print("Rows       :", len(test_df))
print("First date :", test_df["Date"].min().date())
print("Last date  :", test_df["Date"].max().date())

# ------------------------------------------------------------
# NEWS COVERAGE
# ------------------------------------------------------------

print()
print("=" * 60)
print("NEWS COVERAGE")
print("=" * 60)

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:
    news_days = int(data["Has_News"].sum())
    coverage = news_days / len(data) * 100

    print(
        f"{name:<12} rows={len(data):<5} "
        f"news_days={news_days:<5} "
        f"coverage={coverage:.2f}%"
    )

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print()
print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    print()
    print(name)

    if "Target_Direction_1D" in data.columns:
        print("1D:")
        print(
            data["Target_Direction_1D"]
            .value_counts()
            .sort_index()
        )

    if "Target_Direction_5D" in data.columns:
        print("5D:")
        print(
            data["Target_Direction_5D"]
            .value_counts()
            .sort_index()
        )

# ------------------------------------------------------------
# VERIFY NO OVERLAP
# ------------------------------------------------------------

train_dates = set(train_df["Date"])
validation_dates = set(validation_df["Date"])
test_dates = set(test_df["Date"])

if train_dates & validation_dates:
    raise ValueError("TRAIN and VALIDATION dates overlap.")

if train_dates & test_dates:
    raise ValueError("TRAIN and TEST dates overlap.")

if validation_dates & test_dates:
    raise ValueError("VALIDATION and TEST dates overlap.")

print()
print("=" * 60)
print("PASS: NO DATE OVERLAP")
print("=" * 60)

print()
print("Output files:")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)

print()
print("=" * 60)
print("PASS: ML DATASETS READY")
print("=" * 60)