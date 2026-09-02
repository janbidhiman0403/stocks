import pandas as pd
from ta.momentum import RSIIndicator

# Load our current feature dataset
input_file = "data/processed_TCS_features.csv"

df = pd.read_csv(input_file)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Calculate 14-day RSI
rsi_indicator = RSIIndicator(
    close=df["Close"],
    window=14
)

df["RSI_14"] = rsi_indicator.rsi()

# Show the latest RSI values
print("========== RSI VALUES ==========")
print(df[["Date", "Close", "RSI_14"]].tail(10))

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df["RSI_14"].isnull().sum())

# Save updated dataset
output_file = "data/processed_TCS_features_rsi.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)