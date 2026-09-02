import pandas as pd

results = pd.read_csv(
    "data/xgboost_regression_predictions.csv"
)

results["Date"] = pd.to_datetime(results["Date"])

# Top 10% predicted returns
cutoff = results["Predicted_Return_5D"].quantile(0.90)

initial_capital = 100000
capital = initial_capital

trades = []

i = 0

while i < len(results):

    row = results.iloc[i]

    if row["Predicted_Return_5D"] >= cutoff:

        entry_date = row["Date"]
        entry_price = row["Close"]

        exit_index = i + 5

        if exit_index < len(results):

            exit_row = results.iloc[exit_index]

            exit_date = exit_row["Date"]
            exit_price = exit_row["Close"]

            trade_return = (
                exit_price / entry_price
            ) - 1

            capital = capital * (
                1 + trade_return
            )

            trades.append({
                "Entry_Date": entry_date,
                "Exit_Date": exit_date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Predicted_Return": row[
                    "Predicted_Return_5D"
                ],
                "Actual_Return": trade_return
            })

            i = exit_index + 1
            continue

    i += 1


trades_df = pd.DataFrame(trades)

print(
    "========== REGRESSION TOP 10% BACKTEST =========="
)

print(
    "Prediction Cutoff:",
    round(cutoff * 100, 2),
    "%"
)

print(
    "Initial Capital:",
    initial_capital
)

print(
    "Final Capital:",
    round(capital, 2)
)

total_return = (
    capital / initial_capital
) - 1

print(
    "Total Return:",
    round(total_return * 100, 2),
    "%"
)

print(
    "Number of Trades:",
    len(trades_df)
)

if len(trades_df) > 0:

    winners = (
        trades_df["Actual_Return"] > 0
    ).sum()

    win_rate = (
        winners / len(trades_df)
    )

    print(
        "Winning Trades:",
        winners
    )

    print(
        "Win Rate:",
        round(win_rate * 100, 2),
        "%"
    )

    print(
        "Average Trade Return:",
        round(
            trades_df["Actual_Return"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Best Trade:",
        round(
            trades_df["Actual_Return"].max() * 100,
            2
        ),
        "%"
    )

    print(
        "Worst Trade:",
        round(
            trades_df["Actual_Return"].min() * 100,
            2
        ),
        "%"
    )

    trades_df.to_csv(
        "data/regression_backtest_trades.csv",
        index=False
    )

    print(
        "\nSaved to: data/regression_backtest_trades.csv"
    )