import pandas as pd

input_file = "data/processed_TCS_dataset_volume.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

print("========== DATASET BEFORE CLEANING ==========")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# Remove rows with missing values
df = df.dropna().copy()

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

print("\n========== DATASET AFTER CLEANING ==========")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Target"].value_counts().sort_index())

output_file = "data/ml_ready_TCS.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)