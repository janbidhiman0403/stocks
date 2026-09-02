import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

train_file = "data/train_TCS_regression.csv"
test_file = "data/test_TCS_regression.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

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
y_train = train["Future_Return_5D"]

X_test = test[features]
y_test = test["Future_Return_5D"]

print("========== XGBOOST REGRESSION ==========")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", len(features))

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

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

direction_actual = y_test > 0
direction_predicted = predictions > 0

direction_accuracy = (
    direction_actual == direction_predicted
).mean()

print("\n========== RESULTS ==========")

print("MAE:", round(mae * 100, 3), "%")
print("RMSE:", round(rmse * 100, 3), "%")
print(
    "Direction Accuracy:",
    round(direction_accuracy * 100, 2),
    "%"
)

output = test[
    ["Date", "Close", "Future_Close_5D", "Future_Return_5D"]
].copy()

output["Predicted_Return_5D"] = predictions

output.to_csv(
    "data/xgboost_regression_predictions.csv",
    index=False
)

print("\nSaved to:")
print("data/xgboost_regression_predictions.csv")