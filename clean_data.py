import pandas as pd

# Load raw data
input_file = "data/raw/TCS.csv"

df = pd.read_csv(input_file)

# Remove the first two incorrect rows
df = df.iloc[2:].copy()

# Rename the first column to Date
df.rename(columns={"Price": "Date"}, inplace=True)

# Convert Date column to actual datetime
df["Date"] = pd.to_datetime(df["Date"])

# Convert numerical columns to numbers
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows with missing values
df.dropna(inplace=True)

# Sort by date
df.sort_values("Date", inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
output_file = "data/processed_TCS.csv"

df.to_csv(output_file, index=False)

print("========== CLEANING COMPLETE ==========")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\nSaved to:", output_file)