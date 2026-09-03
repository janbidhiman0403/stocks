import pandas as pd
import yfinance as yf
import joblib

TICKER = "TCS.NS"

print("Downloading latest TCS data...")

df = yf.download(
    TICKER,
    start="2018-01-01",
    end="2026-09-03",
    auto_adjust=True,
    progress=False
)

# Flatten Yahoo Finance multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna(subset=["Close", "High", "Low", "Open"])

print("\nColumns:")
print(df.columns.tolist())

print("\nLatest 5 rows:")
print(df.tail())

print("\nRows downloaded:", len(df))
latest = df.iloc[-1]

print("\nLatest valid trading day:")
print("Date:", df.index[-1])
print("Close:", latest["Close"])
print("Volume:", latest["Volume"])
df["Daily_Return"] = df["Close"].pct_change()
df["Daily_Return_Percent"] = df["Daily_Return"] * 100

latest = df.iloc[-1]

print("\nLatest Daily Return:")
print("Daily Return:", latest["Daily_Return"])
print("Daily Return %:", latest["Daily_Return_Percent"])
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

latest = df.iloc[-1]

print("\nLatest Moving Averages:")
print("SMA 20:", latest["SMA_20"])
print("SMA 50:", latest["SMA_50"])
from ta.momentum import RSIIndicator

rsi_indicator = RSIIndicator(
    close=df["Close"],
    window=14
)

df["RSI_14"] = rsi_indicator.rsi()

latest = df.iloc[-1]

print("\nLatest RSI:")
print("RSI 14:", latest["RSI_14"])
from ta.trend import MACD

macd_indicator = MACD(
    close=df["Close"],
    window_slow=26,
    window_fast=12,
    window_sign=9
)

df["MACD"] = macd_indicator.macd()
df["MACD_Signal"] = macd_indicator.macd_signal()
df["MACD_Histogram"] = macd_indicator.macd_diff()

latest = df.iloc[-1]

print("\nLatest MACD:")
print("MACD:", latest["MACD"])
print("MACD Signal:", latest["MACD_Signal"])
print("MACD Histogram:", latest["MACD_Histogram"])
df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

latest = df.iloc[-1]

print("\nLatest Volatility:")
print("Volatility 20:", latest["Volatility_20"])
print("Volatility 20 %:", latest["Volatility_20"] * 100)
df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]

latest = df.iloc[-1]

print("\nLatest Volume Features:")
print("Volume SMA 20:", latest["Volume_SMA_20"])
print("Volume Ratio:", latest["Volume_Ratio"])
df["Price_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
df["Price_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1
df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
df["Open_Close_Change"] = (df["Close"] - df["Open"]) / df["Open"]

latest = df.iloc[-1]

print("\nLatest Relative Price Features:")
print("Price vs SMA20:", latest["Price_vs_SMA20"])
print("Price vs SMA50:", latest["Price_vs_SMA50"])
print("High-Low Range:", latest["High_Low_Range"])
print("Open-Close Change:", latest["Open_Close_Change"])
df["Return_Lag_1"] = df["Daily_Return"].shift(1)
df["Return_Lag_2"] = df["Daily_Return"].shift(2)
df["Return_Lag_5"] = df["Daily_Return"].shift(5)

df["RSI_Lag_1"] = df["RSI_14"].shift(1)
df["Volatility_Lag_1"] = df["Volatility_20"].shift(1)
df["Volume_Ratio_Lag_1"] = df["Volume_Ratio"].shift(1)

latest = df.iloc[-1]

print("\nLatest Lag Features:")
print("Return Lag 1:", latest["Return_Lag_1"])
print("Return Lag 2:", latest["Return_Lag_2"])
print("Return Lag 5:", latest["Return_Lag_5"])
print("RSI Lag 1:", latest["RSI_Lag_1"])
print("Volatility Lag 1:", latest["Volatility_Lag_1"])
print("Volume Ratio Lag 1:", latest["Volume_Ratio_Lag_1"])
model_features = [
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

missing_features = [
    feature for feature in model_features
    if feature not in df.columns
]

print("\nModel Feature Check:")
print("Expected features:", len(model_features))
print("Missing features:", missing_features)

if not missing_features:
    print("PASS: All 19 model features are present.")
else:
    print("FAIL: Some model features are missing.")
    import joblib

model = joblib.load("models/xgboost_regression_final.pkl")

latest_features = df[model_features].iloc[[-1]]

prediction = model.predict(latest_features)[0]

print("\nLatest Model Prediction:")
print("Predicted 5D Return:", prediction)
print("Predicted 5D Return %:", prediction * 100)
# ==============================
# ALPHALENS SIGNAL
# ==============================

buy_threshold = 0.00443

if prediction >= buy_threshold:
    signal = "BUY"
elif prediction <= -buy_threshold:
    signal = "AVOID"
else:
    signal = "HOLD"

print("\nAlphaLens Signal:")
print("Signal:", signal)
print("Threshold:", buy_threshold * 100, "%")
# ==============================
# SIGNAL STRENGTH
# ==============================

prediction_percent = prediction * 100

if abs(prediction_percent) >= 3:
    signal_strength = "Very Strong"
elif abs(prediction_percent) >= 2:
    signal_strength = "Strong"
elif abs(prediction_percent) >= 1:
    signal_strength = "Moderate"
else:
    signal_strength = "Weak"

print("Signal Strength:", signal_strength)
# ==============================
# CONFIDENCE SCORE
# ==============================

# Historical directional accuracy of the final test
base_confidence = 50.39

# Increase confidence when prediction is farther from the signal threshold
prediction_strength = abs(prediction) / buy_threshold

confidence_bonus = min(prediction_strength * 5, 15)

confidence = min(base_confidence + confidence_bonus, 65)

print("\nConfidence Assessment:")
print("Confidence:", round(confidence, 2), "%")
# ==============================
# RISK SCORE
# ==============================

volatility = latest["Volatility_20"]

if volatility < 0.01:
    risk_score = 20
elif volatility < 0.015:
    risk_score = 35
elif volatility < 0.02:
    risk_score = 50
elif volatility < 0.025:
    risk_score = 65
elif volatility < 0.03:
    risk_score = 80
else:
    risk_score = 95

if risk_score <= 35:
    risk_category = "Low"
elif risk_score <= 65:
    risk_category = "Moderate"
elif risk_score <= 80:
    risk_category = "High"
else:
    risk_category = "Very High"

print("\nRisk Assessment:")
print("Risk Score:", risk_score, "/ 100")
print("Risk Category:", risk_category)