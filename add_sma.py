import pandas as pd

# Load our returns dataset
input_file = "data/processed_TCS_returns.csv"

df = pd.read_csv(input_file)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Calculate Simple Moving Averages
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# Show the result
print("========== SMA FEATURES ==========")
print(df[["Date", "Close", "SMA_20", "SMA_50"]].tail(10))

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df[["SMA_20", "SMA_50"]].isnull().sum())

# Save the updated dataset
output_file = "data/processed_TCS_features.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)