import pandas as pd

# Load the V3 ML-ready dataset that already contains
# the relative-price and lag features
tcs = pd.read_csv("data/ml_ready_TCS_v3.csv")

tcs["Date"] = pd.to_datetime(tcs["Date"])
tcs = tcs.sort_values("Date").reset_index(drop=True)

# Download NIFTY 50 data
import yfinance as yf

nifty = yf.download(
    "^NSEI",
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Handle yfinance MultiIndex columns
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty = nifty.reset_index()
nifty["Date"] = pd.to_datetime(nifty["Date"])

# Keep required market data
nifty = nifty[["Date", "Close"]].copy()
nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

# Market return
nifty["NIFTY_Return"] = nifty["NIFTY_Close"].pct_change()

# 20-day market momentum
nifty["NIFTY_Momentum_20"] = (
    nifty["NIFTY_Close"] /
    nifty["NIFTY_Close"].shift(20) - 1
)

# 20-day market volatility
nifty["NIFTY_Volatility_20"] = (
    nifty["NIFTY_Return"].rolling(20).std()
)

# Merge TCS V3 data with NIFTY data
df = tcs.merge(nifty, on="Date", how="left")

# Remove rows where market features aren't available
df = df.dropna().reset_index(drop=True)

# Verify the important V3 features exist
required_v3_features = [
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "High_Low_Range",
    "Open_Close_Change",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_5",
    "RSI_Lag_1",
    "Volatility_Lag_1",
    "Volume_Ratio_Lag_1",
]

missing = [col for col in required_v3_features if col not in df.columns]

if missing:
    raise ValueError(f"V3 features are missing: {missing}")

# Verify market features exist
market_features = [
    "NIFTY_Close",
    "NIFTY_Return",
    "NIFTY_Momentum_20",
    "NIFTY_Volatility_20",
]

missing_market = [col for col in market_features if col not in df.columns]

if missing_market:
    raise ValueError(f"Market features are missing: {missing_market}")

# Save corrected dataset
output_file = "data/ml_ready_TCS_market_v2.csv"
df.to_csv(output_file, index=False)

print("Corrected market dataset created successfully.")
print()
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()
print("V3 features verified:", len(required_v3_features))
print("Market features verified:", len(market_features))
print()
print("Missing values in market features:")
print(df[market_features].isna().sum())
print()
print("Saved to:", output_file)