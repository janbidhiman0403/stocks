import pandas as pd

# Load corrected V3 + NIFTY dataset
df = pd.read_csv("data/ml_ready_TCS_market_v2.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Chronological 60/20/20 split
n = len(df)

train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()

# Save datasets
train.to_csv("data/train_TCS_market_v2.csv", index=False)
validation.to_csv("data/validation_TCS_market_v2.csv", index=False)
test.to_csv("data/test_TCS_market_v2.csv", index=False)

print("Corrected market datasets split successfully.")
print()
print("Total rows:", len(df))
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))
print()
print("Training period:")
print(train["Date"].min().date(), "to", train["Date"].max().date())
print()
print("Validation period:")
print(validation["Date"].min().date(), "to", validation["Date"].max().date())
print()
print("Testing period:")
print(test["Date"].min().date(), "to", test["Date"].max().date())
print()
print("Files created:")
print("- data/train_TCS_market_v2.csv")
print("- data/validation_TCS_market_v2.csv")
print("- data/test_TCS_market_v2.csv")