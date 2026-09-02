import pandas as pd
import numpy as np

# ==============================
# LOAD FINAL TEST PREDICTIONS
# ==============================

file_path = "data/xgboost_regression_final_test.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])


# ==============================
# BASIC METRICS
# ==============================

actual = df["Future_Return_5D"]
predicted = df["Predicted_Return"]

mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(
    np.mean((actual - predicted) ** 2)
)

direction_accuracy = (
    np.sign(actual) == np.sign(predicted)
).mean()

correlation = actual.corr(predicted)


print("========== FINAL TEST ANALYSIS ==========")

print(
    "Test period:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)

print("Test rows:", len(df))

print(
    "\nMAE:",
    round(mae * 100, 3),
    "%"
)

print(
    "RMSE:",
    round(rmse * 100, 3),
    "%"
)

print(
    "Direction Accuracy:",
    round(direction_accuracy * 100, 2),
    "%"
)

print(
    "Prediction / Actual Correlation:",
    round(correlation, 4)
)


# ==============================
# TOP 10% RANKING
# ==============================

cutoff_10 = predicted.quantile(0.90)

top_10 = df[
    predicted >= cutoff_10
].copy()

print("\n========== FINAL TEST TOP 10% ==========")

print(
    "Prediction cutoff:",
    round(cutoff_10 * 100, 3),
    "%"
)

print(
    "Signals:",
    len(top_10)
)

print(
    "Actual average 5D return:",
    round(
        top_10["Future_Return_5D"].mean() * 100,
        3
    ),
    "%"
)

print(
    "Actual median 5D return:",
    round(
        top_10["Future_Return_5D"].median() * 100,
        3
    ),
    "%"
)

print(
    "Win rate:",
    round(
        (
            top_10["Future_Return_5D"] > 0
        ).mean() * 100,
        2
    ),
    "%"
)


# ==============================
# TOP 20% RANKING
# ==============================

cutoff_20 = predicted.quantile(0.80)

top_20 = df[
    predicted >= cutoff_20
].copy()

print("\n========== FINAL TEST TOP 20% ==========")

print(
    "Prediction cutoff:",
    round(cutoff_20 * 100, 3),
    "%"
)

print(
    "Signals:",
    len(top_20)
)

print(
    "Actual average 5D return:",
    round(
        top_20["Future_Return_5D"].mean() * 100,
        3
    ),
    "%"
)

print(
    "Actual median 5D return:",
    round(
        top_20["Future_Return_5D"].median() * 100,
        3
    ),
    "%"
)

print(
    "Win rate:",
    round(
        (
            top_20["Future_Return_5D"] > 0
        ).mean() * 100,
        2
    ),
    "%"
)


# ==============================
# TOP 10 PREDICTIONS
# ==============================

print("\n========== TOP 10 PREDICTIONS ==========")

columns = [
    "Date",
    "Close",
    "Predicted_Return",
    "Future_Return_5D"
]

print(
    df.sort_values(
        "Predicted_Return",
        ascending=False
    )[columns]
    .head(10)
    .to_string(index=False)
)