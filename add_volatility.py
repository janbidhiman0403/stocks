import pandas as pd

input_file = "data/processed_TCS_dataset.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# 20-day rolling volatility
df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

print("========== VOLATILITY VALUES ==========")
print(
    df[
        [
            "Date",
            "Close",
            "Daily_Return",
            "Volatility_20"
        ]
    ].tail(10)
)

print("\n========== MISSING VALUES ==========")
print(df["Volatility_20"].isnull().sum())

output_file = "data/processed_TCS_dataset_volatility.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)