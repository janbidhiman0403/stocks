import pandas as pd
import numpy as np

trades = pd.read_csv("data/xgboost_backtest_v2.csv")

trades["Return"] = pd.to_numeric(trades["Return"])

initial_capital = 100000

# Build portfolio value after every trade
capital = initial_capital
portfolio_values = [capital]

for trade_return in trades["Return"]:
    capital = capital * (1 + trade_return)
    portfolio_values.append(capital)

portfolio = pd.Series(portfolio_values)

# Maximum drawdown
running_peak = portfolio.cummax()
drawdown = (portfolio - running_peak) / running_peak
max_drawdown = drawdown.min()

# Trade statistics
average_trade = trades["Return"].mean()
best_trade = trades["Return"].max()
worst_trade = trades["Return"].min()

# Sharpe ratio based on trade returns
sharpe_ratio = (
    trades["Return"].mean() /
    trades["Return"].std()
) * np.sqrt(len(trades))

print("========== RISK METRICS ==========")

print("Maximum Drawdown:",
      round(max_drawdown * 100, 2), "%")

print("Sharpe Ratio:",
      round(sharpe_ratio, 2))

print("Average Trade Return:",
      round(average_trade * 100, 2), "%")

print("Best Trade:",
      round(best_trade * 100, 2), "%")

print("Worst Trade:",
      round(worst_trade * 100, 2), "%")

print("Total Trades:",
      len(trades))