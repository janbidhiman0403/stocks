import pandas as pd

# Load validation predictions
df = pd.read_csv("data/xgboost_market_validation_v2.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Calculate prediction error
df["Error"] = (
    df["Predicted_Return_5D"] - df["Future_Return_5D"]
)

print("Market-context validation analysis")
print("-----------------------------------")
print("Rows:", len(df))
print()

print("Prediction statistics:")
print(
    df[
        ["Predicted_Return_5D", "Future_Return_5D"]
    ].describe()
)
print()

# Correlation between each prediction and actual return
correlation = df[
    ["Predicted_Return_5D", "Future_Return_5D"]
].corr().iloc[0, 1]

print(
    f"Prediction / actual correlation: "
    f"{correlation:.4f}"
)
print()

# Check prediction direction
direction_accuracy = (
    df["Predicted_Return_5D"].apply(lambda x: 1 if x > 0 else -1)
    ==
    df["Future_Return_5D"].apply(lambda x: 1 if x > 0 else -1)
).mean()

print(
    f"Direction accuracy: "
    f"{direction_accuracy * 100:.2f}%"
)
print()

# Show strongest positive and negative predictions
print("Top 10 predicted returns:")
print(
    df.nlargest(10, "Predicted_Return_5D")[
        ["Date", "Predicted_Return_5D", "Future_Return_5D"]
    ].to_string(index=False)
)

print()
print("Bottom 10 predicted returns:")
print(
    df.nsmallest(10, "Predicted_Return_5D")[
        ["Date", "Predicted_Return_5D", "Future_Return_5D"]
    ].to_string(index=False)
)