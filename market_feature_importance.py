import pandas as pd
from xgboost import XGBRegressor

# Load training and validation data
train = pd.read_csv("data/train_TCS_market_v2.csv")
validation = pd.read_csv("data/validation_TCS_market_v2.csv")

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

X_train = train[features]
y_train = train[target]

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

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print("Feature Importance")
print("------------------")
print(
    importance.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)

print()
print("NIFTY Feature Importance")
print("------------------------")

nifty_features = [
    "NIFTY_Close",
    "NIFTY_Return",
    "NIFTY_Momentum_20",
    "NIFTY_Volatility_20",
]

print(
    importance[
        importance["Feature"].isin(nifty_features)
    ].to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)