import pandas as pd

input_file = "data/ml_ready_TCS_regression.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

total_rows = len(df)

train_end = int(total_rows * 0.60)
validation_end = int(total_rows * 0.80)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

print("========== TRAIN / VALIDATION / TEST SPLIT ==========")

print("Total rows:", len(df))

print("\nTraining rows:", len(train))
print(
    "Training period:",
    train["Date"].min().date(),
    "to",
    train["Date"].max().date()
)

print("\nValidation rows:", len(validation))
print(
    "Validation period:",
    validation["Date"].min().date(),
    "to",
    validation["Date"].max().date()
)

print("\nTesting rows:", len(test))
print(
    "Testing period:",
    test["Date"].min().date(),
    "to",
    test["Date"].max().date()
)

train.to_csv(
    "data/train_TCS_regression_v2.csv",
    index=False
)

validation.to_csv(
    "data/validation_TCS_regression.csv",
    index=False
)

test.to_csv(
    "data/test_TCS_regression_v2.csv",
    index=False
)

print("\nSaved:")
print("data/train_TCS_regression_v2.csv")
print("data/validation_TCS_regression.csv")
print("data/test_TCS_regression_v2.csv")