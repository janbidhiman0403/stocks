import pandas as pd

input_file = "data/ml_ready_TCS.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# Price relative to moving averages
df["Price_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
df["Price_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1

# Daily trading range
df["High_Low_Range"] = (
    df["High"] - df["Low"]
) / df["Close"]

# Intraday price movement
df["Open_Close_Change"] = (
    df["Close"] - df["Open"]
) / df["Open"]

print("========== RELATIVE FEATURES ==========")

print(
    df[
        [
            "Date",
            "Close",
            "SMA_20",
            "SMA_50",
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "High_Low_Range",
            "Open_Close_Change"
        ]
    ].tail(10)
)

print("\n========== MISSING VALUES ==========")
print(
    df[
        [
            "Price_vs_SMA20",
            "Price_vs_SMA50",
            "High_Low_Range",
            "Open_Close_Change"
        ]
    ].isnull().sum()
)

output_file = "data/ml_ready_TCS_v2.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)