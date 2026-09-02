import pandas as pd

results = pd.read_csv("data/xgboost_predictions.csv")
results["Date"] = pd.to_datetime(results["Date"])

initial_capital = 100000
capital = initial_capital

trades = []

i = 0

while i < len(results):

    row = results.iloc[i]

    # BUY signal
    if row["Predicted_Target"] == 1:

        entry_date = row["Date"]
        entry_price = row["Close"]

        # Exit after 5 trading days
        exit_index = i + 5

        if exit_index < len(results):

            exit_row = results.iloc[exit_index]

            exit_date = exit_row["Date"]
            exit_price = exit_row["Close"]

            trade_return = (exit_price / entry_price) - 1

            capital = capital * (1 + trade_return)

            trades.append({
                "Entry_Date": entry_date,
                "Exit_Date": exit_date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Return": trade_return
            })

            # Skip ahead so trades do not overlap
            i = exit_index + 1
            continue

    i += 1


trades_df = pd.DataFrame(trades)

print("========== CORRECTED BACKTEST ==========")

print("Initial Capital: ₹", initial_capital)
print("Final Capital: ₹", round(capital, 2))

total_return = (capital / initial_capital) - 1

print("Total Return:", round(total_return * 100, 2), "%")

print("\nNumber of Trades:", len(trades_df))

if len(trades_df) > 0:

    winning_trades = (trades_df["Return"] > 0).sum()

    win_rate = winning_trades / len(trades_df)

    print("Winning Trades:", winning_trades)
    print("Win Rate:", round(win_rate * 100, 2), "%")

    print("\n========== TRADE RESULTS ==========")
    print(trades_df.head(10).to_string(index=False))

output_file = "data/xgboost_backtest_v2.csv"

trades_df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)