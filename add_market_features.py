import pandas as pd
import yfinance as yf

# Load our existing ML-ready TCS dataset
tcs = pd.read_csv("data/ml_ready_TCS.csv")

tcs["Date"] = pd.to_datetime(tcs["Date"])
tcs = tcs.sort_values("Date").reset_index(drop=True)

# Download NIFTY 50 data
nifty = yf.download(
    "^NSEI",
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Handle yfinance multi-level columns
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty = nifty.reset_index()
nifty["Date"] = pd.to_datetime(nifty["Date"])

# Keep only the market price
nifty = nifty[["Date", "Close"]].copy()
nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

# Market returns
nifty["NIFTY_Return"] = nifty["NIFTY_Close"].pct_change()

# Market momentum
nifty["NIFTY_Momentum_20"] = (
    nifty["NIFTY_Close"] / nifty["NIFTY_Close"].shift(20) - 1
)

# Market volatility
nifty["NIFTY_Volatility_20"] = (
    nifty["NIFTY_Return"].rolling(20).std()
)

# Merge market features with TCS data
df = tcs.merge(nifty, on="Date", how="left")

# Remove rows where market features are unavailable
df = df.dropna().reset_index(drop=True)

# Save
output_file = "data/ml_ready_TCS_market.csv"
df.to_csv(output_file, index=False)

print("Market features added successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("New market features:")
print("- NIFTY_Close")
print("- NIFTY_Return")
print("- NIFTY_Momentum_20")
print("- NIFTY_Volatility_20")
print()
print("Missing values:")
print(df[
    ["NIFTY_Close", "NIFTY_Return",
     "NIFTY_Momentum_20", "NIFTY_Volatility_20"]
].isna().sum())
print()
print("Saved to:", output_file)