import pandas as pd

trades = pd.read_csv(
    "data/regression_backtest_trades.csv"
)

trades["Entry_Date"] = pd.to_datetime(
    trades["Entry_Date"]
)

trades["Exit_Date"] = pd.to_datetime(
    trades["Exit_Date"]
)

print("========== REGRESSION TRADE ANALYSIS ==========")

print(
    trades[
        [
            "Entry_Date",
            "Exit_Date",
            "Entry_Price",
            "Exit_Price",
            "Predicted_Return",
            "Actual_Return"
        ]
    ].to_string(index=False)
)

print("\n========== TRADE STATISTICS ==========")

print(
    "Average Predicted Return:",
    round(
        trades["Predicted_Return"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Average Actual Return:",
    round(
        trades["Actual_Return"].mean() * 100,
        2
    ),
    "%"
)

print(
    "Total Positive Trades:",
    (
        trades["Actual_Return"] > 0
    ).sum()
)

print(
    "Total Negative Trades:",
    (
        trades["Actual_Return"] < 0
    ).sum()
)