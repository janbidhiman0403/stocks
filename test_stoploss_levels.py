import pandas as pd

results = pd.read_csv(
    "data/xgboost_regression_predictions.csv"
)

results["Date"] = pd.to_datetime(results["Date"])

cutoff = results["Predicted_Return_5D"].quantile(0.90)

stop_losses = [-0.02, -0.03, -0.04]

print("========== STOP-LOSS COMPARISON ==========")
print("Prediction Cutoff:", round(cutoff * 100, 2), "%")

for stop_loss in stop_losses:

    initial_capital = 100000
    capital = initial_capital

    trades = []

    i = 0

    while i < len(results):

        row = results.iloc[i]

        if row["Predicted_Return_5D"] >= cutoff:

            entry_price = row["Close"]

            exit_index = i + 5

            if exit_index < len(results):

                exit_price = results.iloc[
                    exit_index
                ]["Close"]

                actual_return = (
                    exit_price / entry_price
                ) - 1

                trade_return = max(
                    actual_return,
                    stop_loss
                )

                capital *= (
                    1 + trade_return
                )

                trades.append(trade_return)

                i = exit_index + 1
                continue

        i += 1

    if len(trades) > 0:

        winners = sum(
            r > 0 for r in trades
        )

        win_rate = (
            winners / len(trades)
        )

        total_return = (
            capital / initial_capital
        ) - 1

        average_trade = (
            sum(trades) / len(trades)
        )

        print(
            f"\nStop Loss: {stop_loss * 100:.0f}%"
        )

        print(
            "Trades:",
            len(trades)
        )

        print(
            "Win Rate:",
            round(win_rate * 100, 2),
            "%"
        )

        print(
            "Average Trade:",
            round(average_trade * 100, 2),
            "%"
        )

        print(
            "Total Return:",
            round(total_return * 100, 2),
            "%"
        )

        print(
            "Final Capital:",
            round(capital, 2)
        )