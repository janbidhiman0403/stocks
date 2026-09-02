import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load corrected V3 + NIFTY training and validation data
train = pd.read_csv("data/train_TCS_market_v2.csv")
validation = pd.read_csv("data/validation_TCS_market_v2.csv")

train["Date"] = pd.to_datetime(train["Date"])
validation["Date"] = pd.to_datetime(validation["Date"])

# V3 features + NIFTY market-context features
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
    "NIFTY_Close",
    "NIFTY_Return",
    "NIFTY_Momentum_20",
    "NIFTY_Volatility_20",
]

target = "Future_Return_5D"

# Verify required columns
required_columns = features + [target]

missing_train = [
    col for col in required_columns
    if col not in train.columns
]

missing_validation = [
    col for col in required_columns
    if col not in validation.columns
]

if missing_train:
    raise ValueError(
        f"Missing columns in training data: {missing_train}"
    )

if missing_validation:
    raise ValueError(
        f"Missing columns in validation data: {missing_validation}"
    )

# Prepare training data
X_train = train[features]
y_train = train[target]

# Prepare validation data
X_validation = validation[features]
y_validation = validation[target]

print("Training market-context regression model...")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Number of features:", len(features))
print()

print("Market features:")
print("- NIFTY_Close")
print("- NIFTY_Return")
print("- NIFTY_Momentum_20")
print("- NIFTY_Volatility_20")
print()

# XGBoost regression model
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train ONLY on training data
model.fit(X_train, y_train)

# Predict validation data
predictions = model.predict(X_validation)

# Calculate metrics
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

direction_accuracy = (
    np.sign(predictions) ==
    np.sign(y_validation)
).mean()

print("Validation Results")
print("------------------")
print(f"MAE: {mae * 100:.3f}%")
print(f"RMSE: {rmse * 100:.3f}%")
print(
    f"Direction Accuracy: "
    f"{direction_accuracy * 100:.2f}%"
)

# Save validation predictions
results = validation[
    ["Date", "Close", target]
].copy()

results["Predicted_Return_5D"] = predictions

output_file = "data/xgboost_market_validation_v2.csv"

results.to_csv(
    output_file,
    index=False
)

print()
print("Saved to:", output_file)