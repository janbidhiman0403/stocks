import pandas as pd

input_file = "data/processed_TCS_dataset_volatility.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# 20-day average trading volume
df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()

# Today's volume relative to 20-day average
df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

print("========== VOLUME FEATURES ==========")
print(
    df[
        [
            "Date",
            "Volume",
            "Volume_SMA_20",
            "Volume_Ratio"
        ]
    ].tail(10)
)

print("\n========== MISSING VALUES ==========")
print(
    df[
        [
            "Volume_SMA_20",
            "Volume_Ratio"
        ]
    ].isnull().sum()
)

output_file = "data/processed_TCS_dataset_volume.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)