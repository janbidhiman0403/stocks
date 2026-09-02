import pandas as pd

results = pd.read_csv(
    "data/xgboost_regression_predictions.csv"
)

print("========== REGRESSION ANALYSIS ==========")

print("\nActual Return Statistics:")
print(
    results["Future_Return_5D"]
    .describe()
)

print("\nPredicted Return Statistics:")
print(
    results["Predicted_Return_5D"]
    .describe()
)

correlation = results[
    ["Future_Return_5D", "Predicted_Return_5D"]
].corr().iloc[0, 1]

print(
    "\nActual vs Predicted Correlation:",
    round(correlation, 4)
)

print("\n========== TOP PREDICTED RETURNS ==========")

top = results.sort_values(
    "Predicted_Return_5D",
    ascending=False
).head(10)

print(
    top[
        [
            "Date",
            "Future_Return_5D",
            "Predicted_Return_5D"
        ]
    ].to_string(index=False)
)

print("\n========== WORST PREDICTED RETURNS ==========")

bottom = results.sort_values(
    "Predicted_Return_5D"
).head(10)

print(
    bottom[
        [
            "Date",
            "Future_Return_5D",
            "Predicted_Return_5D"
        ]
    ].to_string(index=False)
)