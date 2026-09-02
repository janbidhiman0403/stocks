import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load train and test data
train = pd.read_csv("data/train_TCS.csv")
test = pd.read_csv("data/test_TCS.csv")

# Features used by the model
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
X_test = test[features]

# XGBoost requires class labels starting from 0
# Original labels:
# -1 = AVOID
#  0 = HOLD
#  1 = BUY
#
# Convert them to:
#  0 = AVOID
#  1 = HOLD
#  2 = BUY

y_train = train["Target"] + 1
y_test = test["Target"] + 1

print("========== TRAINING XGBOOST ==========")

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

# Predictions
y_pred = model.predict(X_test)

# Convert predictions back:
# 0 -> -1
# 1 -> 0
# 2 -> 1
y_pred_original = y_pred - 1

print("\n========== MODEL ACCURACY ==========")

accuracy = accuracy_score(
    test["Target"],
    y_pred_original
)

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