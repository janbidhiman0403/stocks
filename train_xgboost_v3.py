import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Convert:
# -1 AVOID -> 0
#  0 HOLD  -> 1
#  1 BUY   -> 2

y_train = train["Target"] + 1
y_test = test["Target"] + 1

print("========== TRAINING XGBOOST V3 ==========")

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

print("Training complete!")

y_pred = model.predict(X_test)

y_pred_original = y_pred - 1

accuracy = accuracy_score(
    test["Target"],
    y_pred_original
)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", round(accuracy, 4))

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        test["Target"],
        y_pred_original,
        target_names=[
            "AVOID",
            "HOLD",
            "BUY"
        ]
    )
)