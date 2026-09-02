import pandas as pd

results = pd.read_csv("data/xgboost_predictions_v3.csv")
results["Date"] = pd.to_datetime(results["Date"])

initial_capital = 100000

entry_price = results.iloc[0]["Close"]
exit_price = results.iloc[-1]["Close"]

buy_hold_return = (exit_price / entry_price) - 1
final_capital = initial_capital * (1 + buy_hold_return)

print("========== BUY & HOLD BENCHMARK ==========")

print("Test Period:")
print(results.iloc[0]["Date"].date(), "to", results.iloc[-1]["Date"].date())

print("Initial Capital:", initial_capital)
print("Entry Price:", round(entry_price, 2))
print("Exit Price:", round(exit_price, 2))

print("Final Capital:", round(final_capital, 2))
print("Total Return:", round(buy_hold_return * 100, 2), "%")