import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ==============================
# LOAD DATA
# ==============================

train_file = "data/train_TCS_regression_v2.csv"
validation_file = "data/validation_TCS_regression.csv"

train = pd.read_csv(train_file)
validation = pd.read_csv(validation_file)

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])


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
# PREPARE X AND Y
# ==============================

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]


# ==============================
# TRAIN MODEL
# ==============================

print("========== TRAINING XGBOOST REGRESSION V2 ==========")

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

model.fit(X_train, y_train)


# ==============================
# VALIDATION PREDICTIONS
# ==============================

predictions = model.predict(X_validation)


# ==============================
# EVALUATION
# ==============================

mae = mean_absolute_error(
    y_validation,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        predictions
    )
)

direction_actual = np.sign(y_validation)
direction_predicted = np.sign(predictions)

direction_accuracy = (
    direction_actual == direction_predicted
).mean()


print("\n========== VALIDATION RESULTS ==========")

print(
    "MAE:",
    round(mae * 100, 3),
    "%"
)

print(
    "RMSE:",
    round(rmse * 100, 3),
    "%"
)

print(
    "Direction Accuracy:",
    round(direction_accuracy * 100, 2),
    "%"
)


# ==============================
# SAMPLE PREDICTIONS
# ==============================

results = validation[
    ["Date", "Close", "Future_Return_5D"]
].copy()

results["Predicted_Return"] = predictions

print("\n========== SAMPLE PREDICTIONS ==========")

print(
    results.head(10).to_string(index=False)
)


# ==============================
# SAVE VALIDATION PREDICTIONS
# ==============================

output_file = "data/xgboost_regression_validation_v2.csv"

results.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)