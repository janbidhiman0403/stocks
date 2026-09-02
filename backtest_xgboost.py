import pandas as pd

results = pd.read_csv("data/xgboost_predictions.csv")

# Start with ₹100,000
initial_capital = 100000

capital = initial_capital

# Simple strategy:
# BUY  -> invest the full capital
# HOLD -> stay in cash
# AVOID -> stay in cash

for _, row in results.iterrows():

    signal = row["Predicted_Target"]
    future_return = row["Future_Return_5D"]

    if signal == 1:
        capital = capital * (1 + future_return)

print("========== BACKTEST RESULTS ==========")

total_return = (capital / initial_capital) - 1

print("Initial Capital: ₹", initial_capital)
print("Final Capital: ₹", round(capital, 2))
print("Total Return:", round(total_return * 100, 2), "%")

# Buy-and-hold comparison
first_price = results.iloc[0]["Close"]
last_price = results.iloc[-1]["Close"]

buy_hold_return = (last_price / first_price) - 1

buy_hold_capital = initial_capital * (1 + buy_hold_return)

print("\n========== BUY & HOLD ==========")
print("Final Capital: ₹", round(buy_hold_capital, 2))
print("Total Return:", round(buy_hold_return * 100, 2), "%")