import pandas as pd
from xgboost import XGBClassifier

train = pd.read_csv("data/train_TCS.csv")
test = pd.read_csv("data/test_TCS.csv")

features = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_20",
    "Volume_SMA_20",
    "Volume_Ratio"
]

X_train = train[features]
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

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("========== FEATURE IMPORTANCE ==========")
print(importance.to_string(index=False))