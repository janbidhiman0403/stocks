import pandas as pd

# ==============================
# LOAD VALIDATION PREDICTIONS
# ==============================

results = pd.read_csv(
    "data/xgboost_regression_validation_v2.csv"
)

results["Date"] = pd.to_datetime(results["Date"])


# ==============================
# STRATEGY SETTINGS
# ==============================

# We use the top 20% of model predictions.
prediction_cutoff = results[
    "Predicted_Return"
].quantile(0.80)

initial_capital = 100000
capital = initial_capital

trades = []

i = 0


# ==============================
# BACKTEST
# ==============================

while i < len(results):

    row = results.iloc[i]

    # BUY only when prediction is in the
    # top 20% of validation predictions.
    if row["Predicted_Return"] >= prediction_cutoff:

        entry_date = row["Date"]
        entry_price = row["Close"]

        # Hold for 5 trading rows.
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
                "Predicted_Return": row["Predicted_Return"],
                "Actual_Return": trade_return,
                "Capital_After": capital
            })

            # Prevent overlapping trades.
            i = exit_index + 1
            continue

    i += 1


# ==============================
# RESULTS
# ==============================

trades_df = pd.DataFrame(trades)

print("========== VALIDATION BACKTEST ==========")

print(
    "Prediction cutoff:",
    round(prediction_cutoff * 100, 3),
    "%"
)

print(
    "Initial Capital:",
    round(initial_capital, 2)
)

print(
    "Final Capital:",
    round(capital, 2)
)

total_return = (
    capital / initial_capital - 1
)

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

    average_trade = (
        trades_df["Actual_Return"].mean()
    )

    print(
        "Winners:",
        winners
    )

    print(
        "Win Rate:",
        round(win_rate * 100, 2),
        "%"
    )

    print(
        "Average Trade Return:",
        round(average_trade * 100, 2),
        "%"
    )


# ==============================
# SAVE TRADES
# ==============================

output_file = (
    "data/regression_validation_backtest_v2.csv"
)

trades_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)