import pandas as pd

trades = pd.read_csv(
    "data/regression_backtest_trades.csv"
)

# Recalculate using the -2% stop-loss
stop_loss = -0.02

trades["Strategy_Return"] = trades[
    "Actual_Return"
].clip(lower=stop_loss)

initial_capital = 100000

capital = initial_capital
equity = [capital]

for trade_return in trades["Strategy_Return"]:

    capital *= (1 + trade_return)
    equity.append(capital)

equity_series = pd.Series(equity)

running_max = equity_series.cummax()

drawdown = (
    equity_series / running_max
) - 1

max_drawdown = drawdown.min()

print("========== REGRESSION STRATEGY RISK ==========")

print(
    "Stop Loss:",
    round(stop_loss * 100, 2),
    "%"
)

print(
    "Final Capital:",
    round(capital, 2)
)

print(
    "Total Return:",
    round(
        (capital / initial_capital - 1) * 100,
        2
    ),
    "%"
)

print(
    "Maximum Drawdown:",
    round(max_drawdown * 100, 2),
    "%"
)

print(
    "Number of Trades:",
    len(trades)
)