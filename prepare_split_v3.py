import pandas as pd

input_file = "data/ml_ready_TCS_v3.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

print("========== BEFORE CLEANING ==========")
print("Rows:", len(df))

# Remove rows created by lagging
df = df.dropna().copy()

# Make absolutely sure the data is chronological
df = df.sort_values("Date").reset_index(drop=True)

print("\n========== AFTER CLEANING ==========")
print("Rows:", len(df))
print("Missing values:", df.isnull().sum().sum())

# 80/20 chronological split
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("\n========== SPLIT ==========")
print("Total:", len(df))
print("Training:", len(train))
print("Testing:", len(test))

print("\nTraining dates:")
print(train["Date"].min(), "to", train["Date"].max())

print("\nTesting dates:")
print(test["Date"].min(), "to", test["Date"].max())

print("\n========== TRAIN TARGET ==========")
print(train["Target"].value_counts().sort_index())

print("\n========== TEST TARGET ==========")
print(test["Target"].value_counts().sort_index())

train.to_csv("data/train_TCS_v3.csv", index=False)
test.to_csv("data/test_TCS_v3.csv", index=False)

print("\nSaved:")
print("data/train_TCS_v3.csv")
print("data/test_TCS_v3.csv")