import pandas as pd

input_file = "data/ml_ready_TCS.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# Make sure data is chronological
df = df.sort_values("Date").reset_index(drop=True)

# 80% training, 20% testing
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("========== DATA SPLIT ==========")
print("Total rows:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))

print("\n========== TRAINING PERIOD ==========")
print("Start:", train["Date"].min())
print("End:", train["Date"].max())

print("\n========== TESTING PERIOD ==========")
print("Start:", test["Date"].min())
print("End:", test["Date"].max())

print("\n========== TRAIN TARGET DISTRIBUTION ==========")
print(train["Target"].value_counts().sort_index())

print("\n========== TEST TARGET DISTRIBUTION ==========")
print(test["Target"].value_counts().sort_index())

train.to_csv("data/train_TCS.csv", index=False)
test.to_csv("data/test_TCS.csv", index=False)

print("\nSaved:")
print("data/train_TCS.csv")
print("data/test_TCS.csv")