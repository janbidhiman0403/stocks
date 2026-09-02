import pandas as pd

# ==============================
# LOAD VALIDATION PREDICTIONS
# ==============================

file_path = "data/xgboost_regression_validation_v2.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ==============================
# RANK PREDICTIONS
# ==============================

print("========== VALIDATION RANKING ANALYSIS ==========")

print("Validation rows:", len(df))

print(
    "Prediction range:",
    round(df["Predicted_Return"].min() * 100, 2),
    "% to",
    round(df["Predicted_Return"].max() * 100, 2),
    "%"
)


# ==============================
# TOP 10%
# ==============================

cutoff_10 = df["Predicted_Return"].quantile(0.90)

top_10 = df[
    df["Predicted_Return"] >= cutoff_10
].copy()

print("\n========== TOP 10% ==========")

print(
    "Prediction cutoff:",
    round(cutoff_10 * 100, 3),
    "%"
)

print("Signals:", len(top_10))

print(
    "Actual average 5D return:",
    round(top_10["Future_Return_5D"].mean() * 100, 3),
    "%"
)

print(
    "Actual median 5D return:",
    round(top_10["Future_Return_5D"].median() * 100, 3),
    "%"
)

print(
    "Win rate:",
    round(
        (top_10["Future_Return_5D"] > 0).mean() * 100,
        2
    ),
    "%"
)


# ==============================
# TOP 20%
# ==============================

cutoff_20 = df["Predicted_Return"].quantile(0.80)

top_20 = df[
    df["Predicted_Return"] >= cutoff_20
].copy()

print("\n========== TOP 20% ==========")

print(
    "Prediction cutoff:",
    round(cutoff_20 * 100, 3),
    "%"
)

print("Signals:", len(top_20))

print(
    "Actual average 5D return:",
    round(top_20["Future_Return_5D"].mean() * 100, 3),
    "%"
)

print(
    "Actual median 5D return:",
    round(top_20["Future_Return_5D"].median() * 100, 3),
    "%"
)

print(
    "Win rate:",
    round(
        (top_20["Future_Return_5D"] > 0).mean() * 100,
        2
    ),
    "%"
)


# ==============================
# TOP 30%
# ==============================

cutoff_30 = df["Predicted_Return"].quantile(0.70)

top_30 = df[
    df["Predicted_Return"] >= cutoff_30
].copy()

print("\n========== TOP 30% ==========")

print(
    "Prediction cutoff:",
    round(cutoff_30 * 100, 3),
    "%"
)

print(
    "Prediction cutoff:",
    round(cutoff_30 * 100, 3),
    "%"
)

print("Signals:", len(top_30))

print(
    "Actual average 5D return:",
    round(top_30["Future_Return_5D"].mean() * 100, 3),
    "%"
)

print(
    "Actual median 5D return:",
    round(top_30["Future_Return_5D"].median() * 100, 3),
    "%"
)

print(
    "Win rate:",
    round(
        (top_30["Future_Return_5D"] > 0).mean() * 100,
        2
    ),
    "%"
)


# ==============================
# BEST PREDICTIONS
# ==============================

print("\n========== TOP 10 PREDICTIONS ==========")

display_columns = [
    "Date",
    "Close",
    "Predicted_Return",
    "Future_Return_5D"
]

print(
    top_10
    .sort_values(
        "Predicted_Return",
        ascending=False
    )[display_columns]
    .head(10)
    .to_string(index=False)
)