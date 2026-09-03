import pandas as pd
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================

train_file = "data/train_TCS_regression_v2.csv"
validation_file = "data/validation_TCS_regression.csv"
test_file = "data/test_TCS_regression_v2.csv"

train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)
test = pd.read_csv(test_file)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])
test["Date"] = pd.to_datetime(test["Date"])


# ==============================
# FEATURES
# ==============================

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

target = "Future_Return_5D"


# ==============================
# COMBINE TRAIN + VALIDATION
# ==============================

development = pd.concat(
    [train, validation],
    ignore_index=True
)

X_development = development[features]
y_development = development[target]

X_test = test[features]
y_test = test[target]


# ==============================
# TRAIN FINAL MODEL
# ==============================

print("========== FINAL MODEL TRAINING ==========")

print(
    "Training + validation rows:",
    len(development)
)

print(
    "Final test rows:",
    len(test)
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42
)

model.fit(
    X_development,
    y_development
)
joblib.dump(
    model,
    "models/xgboost_regression_final.pkl"
)

print("Model saved: models/xgboost_regression_final.pkl")

# ==============================
# FINAL TEST PREDICTIONS
# ==============================

predictions = model.predict(X_test)


# ==============================
# SAVE RESULTS
# ==============================

results = test[
    ["Date", "Close", "Future_Return_5D"]
].copy()

results["Predicted_Return"] = predictions

output_file = (
    "data/xgboost_regression_final_test.csv"
)

results.to_csv(
    output_file,
    index=False
)


# ==============================
# BASIC INFORMATION
# ==============================

print("\n========== FINAL TEST ==========")

print(
    "Test period:",
    test["Date"].min().date(),
    "to",
    test["Date"].max().date()
)

print(
    "Test rows:",
    len(test)
)

print(
    "Predicted return range:",
    round(predictions.min() * 100, 2),
    "%",
    "to",
    round(predictions.max() * 100, 2),
    "%"
)

print("\nSaved:")
print(output_file)