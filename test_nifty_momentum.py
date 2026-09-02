import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load corrected V3 + NIFTY datasets
train = pd.read_csv("data/train_TCS_market_v2.csv")
validation = pd.read_csv("data/validation_TCS_market_v2.csv")

# Features from our previous V3 model
features = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "Daily_Return_Percent",
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Volatility_20",
    "Volume_SMA_20",
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
    "Volume_Ratio_Lag_1",

    # Only one market-context feature
    "NIFTY_Momentum_20",
]

target = "Future_Return_5D"

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

print("Testing V3 + NIFTY Momentum model...")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Number of features:", len(features))
print()

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_validation)

mae = mean_absolute_error(y_validation, predictions)
rmse = np.sqrt(mean_squared_error(y_validation, predictions))

direction_accuracy = (
    np.sign(predictions) == np.sign(y_validation)
).mean()

correlation = np.corrcoef(
    predictions,
    y_validation
)[0, 1]

print("Validation Results")
print("------------------")
print(f"MAE: {mae * 100:.3f}%")
print(f"RMSE: {rmse * 100:.3f}%")
print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")
print(f"Prediction/Actual Correlation: {correlation:.4f}")

results = validation[
    ["Date", "Close", target]
].copy()

results["Predicted_Return_5D"] = predictions

results.to_csv(
    "data/xgboost_nifty_momentum_validation.csv",
    index=False
)

print()
print(
    "Saved to: "
    "data/xgboost_nifty_momentum_validation.csv"
)