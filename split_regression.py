import pandas as pd

input_file = "data/ml_ready_TCS_regression.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("========== REGRESSION SPLIT ==========")

print("Total rows:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))

print("\nTraining period:")
print(train["Date"].min().date(), "to", train["Date"].max().date())

print("\nTesting period:")
print(test["Date"].min().date(), "to", test["Date"].max().date())

print("\nTraining target mean:",
      round(train["Future_Return_5D"].mean() * 100, 2), "%")

print("Testing target mean:",
      round(test["Future_Return_5D"].mean() * 100, 2), "%")

train.to_csv("data/train_TCS_regression.csv", index=False)
test.to_csv("data/test_TCS_regression.csv", index=False)

print("\nSaved:")
print("data/train_TCS_regression.csv")
print("data/test_TCS_regression.csv")