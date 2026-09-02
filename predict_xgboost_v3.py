import pandas as pd
from xgboost import XGBClassifier

train = pd.read_csv("data/train_TCS_v3.csv")
test = pd.read_csv("data/test_TCS_v3.csv")

features = [
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_20",
    "Volume_Ratio",
    "Price_vs_SMA20",
    "Price_vs_SMA50",
    "High_Low_Range",
    "Open_Close_Change",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_5",
    "RSI_Lag_1",
    "Volatility_Lag_1",
    "Volume_Ratio_Lag_1"
]

X_train = train[features]
X_test = test[features]

y_train = train["Target"] + 1

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)

predicted_class = model.predict(X_test)

predicted_target = predicted_class - 1

results = test[
    [
        "Date",
        "Close",
        "Future_Return_5D",
        "Target"
    ]
].copy()

results["Predicted_Target"] = predicted_target

results["AVOID_Probability"] = probabilities[:, 0]
results["HOLD_Probability"] = probabilities[:, 1]
results["BUY_Probability"] = probabilities[:, 2]

results["Confidence"] = probabilities.max(axis=1)

print("========== V3 SAMPLE PREDICTIONS ==========")
print(results.head(15).to_string(index=False))

print("\n========== V3 CONFIDENCE STATISTICS ==========")
print(results["Confidence"].describe())

print("\n========== PREDICTION DISTRIBUTION ==========")
print(results["Predicted_Target"].value_counts().sort_index())

output_file = "data/xgboost_predictions_v3.csv"

results.to_csv(output_file, index=False)

print("\nSaved to:", output_file)