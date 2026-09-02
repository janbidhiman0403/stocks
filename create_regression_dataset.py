import pandas as pd

input_file = "data/ml_ready_TCS_v3.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

df["Future_Close_5D"] = df["Close"].shift(-5)

df["Future_Return_5D"] = (
    df["Future_Close_5D"] / df["Close"] - 1
)

df = df.dropna().copy()
df = df.sort_values("Date").reset_index(drop=True)

print("========== REGRESSION DATASET ==========")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n========== TARGET STATISTICS ==========")

print(df["Future_Return_5D"].describe())

print("\nMean 5D Return:",
      round(df["Future_Return_5D"].mean() * 100, 2), "%")

print("Median 5D Return:",
      round(df["Future_Return_5D"].median() * 100, 2), "%")

output_file = "data/ml_ready_TCS_regression.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)