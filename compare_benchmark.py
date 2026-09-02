import pandas as pd

# Load final test predictions and final strategy trades
predictions = pd.read_csv(
    "data/xgboost_regression_final_test.csv",
    parse_dates=["Date"]
)

trades = pd.read_csv(
    "data/regression_final_test_trades.csv",
    parse_dates=["Entry_Date", "Exit_Date"]
)

# Initial capital
initial_capital = 100000

# -----------------------------
# AlphaLens strategy
# -----------------------------

strategy_final = initial_capital

for _, trade in trades.iterrows():
    trade_return = trade["Actual_Return"]
    strategy_final *= (1 + trade_return)

strategy_return = (strategy_final / initial_capital - 1) * 100


# -----------------------------
# TCS Buy & Hold benchmark
# -----------------------------

first_price = predictions.iloc[0]["Close"]
last_price = predictions.iloc[-1]["Close"]

benchmark_final = initial_capital * (last_price / first_price)

benchmark_return = (benchmark_final / initial_capital - 1) * 100


# -----------------------------
# Comparison
# -----------------------------

difference = strategy_return - benchmark_return

print("\n========== BENCHMARK COMPARISON ==========")

print(f"Test period:")
print(f"{predictions.iloc[0]['Date'].date()} → {predictions.iloc[-1]['Date'].date()}")

print("\nInitial Capital:")
print(f"₹{initial_capital:,.2f}")

print("\nAlphaLens Strategy:")
print(f"Final Capital: ₹{strategy_final:,.2f}")
print(f"Total Return: {strategy_return:.2f}%")
print(f"Number of Trades: {len(trades)}")

print("\nTCS Buy & Hold:")
print(f"Final Capital: ₹{benchmark_final:,.2f}")
print(f"Total Return: {benchmark_return:.2f}%")

print("\nPerformance Difference:")
print(f"AlphaLens - Buy & Hold: {difference:.2f} percentage points")

if difference > 0:
    print("\nResult: AlphaLens outperformed Buy & Hold.")
elif difference < 0:
    print("\nResult: AlphaLens underperformed Buy & Hold.")
else:
    print("\nResult: AlphaLens matched Buy & Hold.")

print("===========================================\n")