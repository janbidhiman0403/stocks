import pandas as pd
import yfinance as yf

# -----------------------------
# Load AlphaLens final test data
# -----------------------------

alpha = pd.read_csv(
    "data/xgboost_regression_final_test.csv",
    parse_dates=["Date"]
)

start_date = alpha["Date"].min()
end_date = alpha["Date"].max()

initial_capital = 100000


# -----------------------------
# Download NIFTY 50 data
# -----------------------------

print("Downloading NIFTY 50 data...")

nifty = yf.download(
    "^NSEI",
    start=start_date.strftime("%Y-%m-%d"),
    end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False
)

# Handle yfinance multi-level columns
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty = nifty.reset_index()

nifty["Date"] = pd.to_datetime(nifty["Date"]).dt.tz_localize(None)

nifty = nifty.dropna(subset=["Close"])

# Keep dates inside the exact final-test period
nifty = nifty[
    (nifty["Date"] >= start_date) &
    (nifty["Date"] <= end_date)
].copy()


# -----------------------------
# NIFTY Buy & Hold
# -----------------------------

first_price = float(nifty.iloc[0]["Close"])
last_price = float(nifty.iloc[-1]["Close"])

nifty_final = initial_capital * (last_price / first_price)

nifty_return = (nifty_final / initial_capital - 1) * 100


# -----------------------------
# AlphaLens Strategy
# -----------------------------

trades = pd.read_csv(
    "data/regression_final_test_trades.csv"
)

strategy_final = initial_capital

for _, trade in trades.iterrows():
    strategy_final *= (1 + trade["Actual_Return"])

strategy_return = (strategy_final / initial_capital - 1) * 100

difference = strategy_return - nifty_return


# -----------------------------
# Print results
# -----------------------------

print("\n========== NIFTY 50 BENCHMARK ==========")

print(
    f"Test period: "
    f"{start_date.date()} → {end_date.date()}"
)

print(f"\nInitial Capital: ₹{initial_capital:,.2f}")

print("\nAlphaLens Strategy:")
print(f"Final Capital: ₹{strategy_final:,.2f}")
print(f"Total Return: {strategy_return:.2f}%")

print("\nNIFTY 50 Buy & Hold:")
print(f"Final Capital: ₹{nifty_final:,.2f}")
print(f"Total Return: {nifty_return:.2f}%")

print("\nPerformance Difference:")
print(
    f"AlphaLens - NIFTY 50: "
    f"{difference:.2f} percentage points"
)

if difference > 0:
    print("\nResult: AlphaLens outperformed NIFTY 50.")
elif difference < 0:
    print("\nResult: AlphaLens underperformed NIFTY 50.")
else:
    print("\nResult: AlphaLens matched NIFTY 50.")

print("=========================================\n")