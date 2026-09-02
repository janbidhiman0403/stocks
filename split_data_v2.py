import pandas as pd

input_file = "data/ml_ready_TCS_v2.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("========== V2 DATA SPLIT ==========")
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

train.to_csv("data/train_TCS_v2.csv", index=False)
test.to_csv("data/test_TCS_v2.csv", index=False)

print("\nSaved:")
print("data/train_TCS_v2.csv")
print("data/test_TCS_v2.csv")