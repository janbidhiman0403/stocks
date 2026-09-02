import pandas as pd

# ==============================
# LOAD FINAL TEST TRADES
# ==============================

file_path = "data/regression_final_test_trades.csv"

trades = pd.read_csv(file_path)

trades["Entry_Date"] = pd.to_datetime(trades["Entry_Date"])
trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])


# ==============================
# BASIC TRADE ANALYSIS
# ==============================

print("========== FINAL TEST TRADE ANALYSIS ==========")

print("Total trades:", len(trades))

print(
    "Average predicted return:",
    round(
        trades["Predicted_Return"].mean() * 100,
        3
    ),
    "%"
)

print(
    "Average actual trade return:",
    round(
        trades["Actual_Return"].mean() * 100,
        3
    ),
    "%"
)

print(
    "Best trade:",
    round(
        trades["Actual_Return"].max() * 100,
        3
    ),
    "%"
)

print(
    "Worst trade:",
    round(
        trades["Actual_Return"].min() * 100,
        3
    ),
    "%"
)


# ==============================
# WINNERS / LOSERS
# ==============================

winners = trades[
    trades["Actual_Return"] > 0
]

losers = trades[
    trades["Actual_Return"] <= 0
]

print("\n========== WINNERS ==========")

print(
    "Winning trades:",
    len(winners)
)

if len(winners) > 0:

    print(
        "Average winning return:",
        round(
            winners["Actual_Return"].mean() * 100,
            3
        ),
        "%"
    )


print("\n========== LOSERS ==========")

print(
    "Losing trades:",
    len(losers)
)

if len(losers) > 0:

    print(
        "Average losing return:",
        round(
            losers["Actual_Return"].mean() * 100,
            3
        ),
        "%"
    )


# ==============================
# ALL TRADES
# ==============================

print("\n========== ALL TRADES ==========")

display_columns = [
    "Entry_Date",
    "Exit_Date",
    "Predicted_Return",
    "Actual_Return"
]

print(
    trades[display_columns]
    .to_string(index=False)
)