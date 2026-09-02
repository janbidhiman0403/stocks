import pandas as pd
import numpy as np

# Load final test trades
trades = pd.read_csv(
    "data/regression_final_test_trades.csv",
    parse_dates=["Entry_Date", "Exit_Date"]
)

initial_capital = 100000

# -----------------------------
# Basic trade statistics
# -----------------------------

returns = trades["Actual_Return"]

num_trades = len(trades)
winning_trades = (returns > 0).sum()
losing_trades = (returns < 0).sum()

win_rate = winning_trades / num_trades * 100

average_trade = returns.mean() * 100
best_trade = returns.max() * 100
worst_trade = returns.min() * 100


# -----------------------------
# Equity curve
# -----------------------------

equity = initial_capital * (1 + returns).cumprod()

final_capital = equity.iloc[-1]

total_return = (final_capital / initial_capital - 1) * 100


# -----------------------------
# Maximum Drawdown
# -----------------------------

running_max = equity.cummax()

drawdown = (equity - running_max) / running_max

max_drawdown = drawdown.min() * 100


# -----------------------------
# Sharpe Ratio
# -----------------------------

if returns.std() != 0:
    sharpe_ratio = (
        returns.mean() / returns.std()
    ) * np.sqrt(num_trades)
else:
    sharpe_ratio = 0


# -----------------------------
# Profit Factor
# -----------------------------

gross_profit = returns[returns > 0].sum()
gross_loss = abs(returns[returns < 0].sum())

if gross_loss != 0:
    profit_factor = gross_profit / gross_loss
else:
    profit_factor = float("inf")


# -----------------------------
# Print results
# -----------------------------

print("\n========== ALPHALENS FINAL METRICS ==========")

print(f"Number of Trades: {num_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate:.2f}%")

print("\nCapital Performance:")
print(f"Initial Capital: ₹{initial_capital:,.2f}")
print(f"Final Capital: ₹{final_capital:,.2f}")
print(f"Total Return: {total_return:.2f}%")

print("\nTrade Performance:")
print(f"Average Trade: {average_trade:.2f}%")
print(f"Best Trade: {best_trade:.2f}%")
print(f"Worst Trade: {worst_trade:.2f}%")

print("\nRisk Metrics:")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")

print("=============================================\n")