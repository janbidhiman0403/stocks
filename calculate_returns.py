import pandas as pd

# Load cleaned TCS data
file_path = "data/processed_TCS.csv"

df = pd.read_csv(file_path)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Calculate daily percentage return
df["Daily_Return"] = df["Close"].pct_change()

# Convert return to percentage
df["Daily_Return_Percent"] = df["Daily_Return"] * 100

# Display the result
print("========== DAILY RETURNS ==========")
print(df[["Date", "Close", "Daily_Return", "Daily_Return_Percent"]].head(10))

print("\n========== RETURN STATISTICS ==========")
print(df["Daily_Return"].describe())

# Save the new dataset
output_file = "data/processed_TCS_returns.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)