import pandas as pd

input_file = "data/processed_TCS_features_macd.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# Calculate the closing price 5 trading days into the future
df["Future_Close_5D"] = df["Close"].shift(-5)

# Calculate future 5-day return
df["Future_Return_5D"] = (
    df["Future_Close_5D"] / df["Close"] - 1
)

# Create target labels
df["Target"] = 0

df.loc[df["Future_Return_5D"] >= 0.02, "Target"] = 1
df.loc[df["Future_Return_5D"] <= -0.02, "Target"] = -1

print("========== TARGET VALUES ==========")
print(
    df[
        [
            "Date",
            "Close",
            "Future_Close_5D",
            "Future_Return_5D",
            "Target"
        ]
    ].tail(15)
)

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Target"].value_counts().sort_index())

output_file = "data/processed_TCS_dataset.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)