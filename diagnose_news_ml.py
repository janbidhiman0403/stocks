import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/ml_ready_TCS_news_clean.csv"

print("=" * 60)
print("ALPHALENS NEWS ML DIAGNOSTIC")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

print(f"\nLoading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df = df.sort_values("Date").reset_index(drop=True)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"First   : {df['Date'].min().date()}")
print(f"Last    : {df['Date'].max().date()}")

# ------------------------------------------------------------
# 1. BASIC DATA CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("1. BASIC DATA CHECK")
print("=" * 60)

print("Duplicate dates:", df["Date"].duplicated().sum())
print("Missing values :", int(df.isna().sum().sum()))

if "Target_Direction_1D" not in df.columns:
    raise ValueError("Target_Direction_1D not found.")

if "Target_Return_1D" not in df.columns:
    raise ValueError("Target_Return_1D not found.")

# ------------------------------------------------------------
# 2. FEATURE / TARGET LEAKAGE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. POSSIBLE TARGET LEAKAGE CHECK")
print("=" * 60)

target_cols = [
    c for c in df.columns
    if c.startswith("Target_")
]

print("\nTarget columns:")
for c in target_cols:
    print(" -", c)

feature_cols = [
    c for c in df.columns
    if c not in ["Date"] + target_cols
]

suspicious = []

target = df["Target_Direction_1D"]

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        x = df[col]

        if x.nunique(dropna=True) <= 1:
            continue

        corr = x.corr(target)

        if pd.notna(corr) and abs(corr) >= 0.20:
            suspicious.append((col, corr))

print("\nFeatures with absolute correlation >= 0.20:")
if suspicious:
    for col, corr in sorted(
        suspicious,
        key=lambda x: abs(x[1]),
        reverse=True
    ):
        print(f"{col:35s} {corr:+.4f}")
else:
    print("None")

# ------------------------------------------------------------
# 3. DIRECT TARGET-LIKE FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. TARGET-LIKE FEATURE CHECK")
print("=" * 60)

target_like_words = [
    "target",
    "future",
    "next",
    "forward",
]

target_like_features = []

for col in feature_cols:
    name = col.lower()

    if any(word in name for word in target_like_words):
        target_like_features.append(col)

if target_like_features:
    print("WARNING: suspicious feature names:")
    for col in target_like_features:
        print(" -", col)
else:
    print("PASS: No obvious future/target feature names.")

# ------------------------------------------------------------
# 4. PRICE FEATURE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. PRICE FEATURE CORRELATIONS")
print("=" * 60)

price_features = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "Daily_Return_Pct",
    "Price_Change",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Open_Close_Change",
    "Open_Close_Change_Pct",
]

for col in price_features:
    if col in df.columns:
        corr = df[col].corr(target)
        print(f"{col:30s} {corr:+.6f}")

# ------------------------------------------------------------
# 5. NEWS FEATURE CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. NEWS FEATURE CORRELATIONS")
print("=" * 60)

news_features = [
    c for c in feature_cols
    if c.startswith("News_") or c == "Has_News"
]

print("News features:", len(news_features))

for col in news_features:
    if pd.api.types.is_numeric_dtype(df[col]):
        corr = df[col].corr(target)

        print(f"{col:35s} {corr:+.6f}")

# ------------------------------------------------------------
# 6. NEWS COVERAGE BY PERIOD
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. NEWS COVERAGE BY TIME PERIOD")
print("=" * 60)

df["Year"] = df["Date"].dt.year

year_stats = (
    df.groupby("Year")
    .agg(
        Rows=("Date", "size"),
        News_Days=("Has_News", "sum"),
        Avg_Sentiment=("News_Sentiment", "mean"),
        Total_News=("News_Count", "sum"),
    )
)

year_stats["Coverage_%"] = (
    year_stats["News_Days"] /
    year_stats["Rows"] * 100
)

print(year_stats.to_string())

# ------------------------------------------------------------
# 7. TRAIN / VALIDATION / TEST
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. TRAIN / VALIDATION / TEST DISTRIBUTION")
print("=" * 60)

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

periods = {
    "TRAIN": train,
    "VALIDATION": validation,
    "TEST": test,
}

for name, part in periods.items():

    print(f"\n{name}")
    print("-" * 40)

    print("Rows:", len(part))
    print(
        "Dates:",
        part["Date"].min().date(),
        "to",
        part["Date"].max().date()
    )

    news_days = int(part["Has_News"].sum())
    coverage = news_days / len(part) * 100

    print("News days:", news_days)
    print(f"News coverage: {coverage:.2f}%")

    print("\nTarget distribution:")
    print(
        part["Target_Direction_1D"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMean target:")
    print(
        f"{part['Target_Direction_1D'].mean():.4f}"
    )

# ------------------------------------------------------------
# 8. NEWS DISTRIBUTION BY PERIOD
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. NEWS DISTRIBUTION")
print("=" * 60)

for name, part in periods.items():

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Sentiment mean:",
        round(part["News_Sentiment"].mean(), 6)
    )

    print(
        "Sentiment std :",
        round(part["News_Sentiment"].std(), 6)
    )

    print(
        "Sentiment min :",
        round(part["News_Sentiment"].min(), 6)
    )

    print(
        "Sentiment max :",
        round(part["News_Sentiment"].max(), 6)
    )

    print(
        "Average news count:",
        round(part["News_Count"].mean(), 4)
    )

# ------------------------------------------------------------
# 9. NEWS VS TARGET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("9. NEWS VS TARGET DIRECTION")
print("=" * 60)

if "Has_News" in df.columns:

    grouped = (
        df.groupby("Has_News")["Target_Direction_1D"]
        .agg(["count", "mean"])
    )

    print("\nTarget probability by Has_News:")
    print(grouped)

if "News_Sentiment" in df.columns:

    df["Sentiment_Bucket"] = pd.cut(
        df["News_Sentiment"],
        bins=[
            -np.inf,
            -0.50,
            -0.10,
            0.10,
            0.50,
            np.inf,
        ],
        labels=[
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ],
    )

    sentiment_target = (
        df.groupby(
            "Sentiment_Bucket",
            observed=False
        )["Target_Direction_1D"]
        .agg(["count", "mean"])
    )

    print("\nTarget probability by sentiment:")
    print(sentiment_target)

# ------------------------------------------------------------
# 10. EXTREME CORRELATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("10. STRONGEST FEATURE/TARGET CORRELATIONS")
print("=" * 60)

correlations = []

for col in feature_cols:

    if pd.api.types.is_numeric_dtype(df[col]):

        corr = df[col].corr(target)

        if pd.notna(corr):
            correlations.append(
                (col, corr)
            )

corr_df = pd.DataFrame(
    correlations,
    columns=["Feature", "Correlation"]
)

corr_df["Abs_Correlation"] = (
    corr_df["Correlation"].abs()
)

corr_df = corr_df.sort_values(
    "Abs_Correlation",
    ascending=False
)

print(
    corr_df[
        ["Feature", "Correlation"]
    ]
    .head(30)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 11. CHECK CURRENT XGBOOST PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("11. EXISTING XGBOOST PREDICTION CHECK")
print("=" * 60)

prediction_file = "data/xgboost_news_predictions.csv"

if os.path.exists(prediction_file):

    pred = pd.read_csv(prediction_file)

    print("Prediction file found.")
    print("Rows:", len(pred))
    print("Columns:", list(pred.columns))

    print("\nPrediction sample:")
    print(pred.tail(10).to_string(index=False))

else:

    print(
        "Prediction file not found:"
        f" {prediction_file}"
    )

# ------------------------------------------------------------
# 12. FINAL DIAGNOSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DIAGNOSIS")
print("=" * 60)

train_news = train["Has_News"].mean() * 100
test_news = test["Has_News"].mean() * 100

print(
    f"\nTraining news coverage    : {train_news:.2f}%"
)

print(
    f"Testing news coverage     : {test_news:.2f}%"
)

if test_news > train_news * 2:
    print(
        "\nWARNING: News distribution is heavily shifted "
        "between training and testing."
    )

if suspicious:
    print(
        "\nWARNING: Some features have relatively strong "
        "correlation with the target."
    )

if target_like_features:
    print(
        "\nWARNING: Target/future-like feature names detected."
    )

print(
    "\n============================================================"
)

print(
    "DIAGNOSTIC COMPLETE"
)

print(
    "Do NOT retrain the model yet."
)

print(
    "Use this report to determine the next model design."
)

print(
    "============================================================"
)