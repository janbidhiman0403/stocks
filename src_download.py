import yfinance as yf

ticker = "TCS.NS"

print("Downloading TCS historical data...")

data = yf.download(
    ticker,
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True
)

print("\nDownload complete!")
print("Number of rows:", len(data))
print("\nFirst 5 rows:")
print(data.head())

data.to_csv("data/raw/TCS.csv")

print("\nSaved to: data/raw/TCS.csv")