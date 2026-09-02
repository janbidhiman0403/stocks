import pandas as pd
from ta.trend import MACD

input_file = "data/processed_TCS_features_rsi.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

macd_indicator = MACD(
    close=df["Close"],
    window_slow=26,
    window_fast=12,
    window_sign=9
)

df["MACD"] = macd_indicator.macd()
df["MACD_Signal"] = macd_indicator.macd_signal()
df["MACD_Histogram"] = macd_indicator.macd_diff()

print("========== MACD VALUES ==========")
print(
    df[
        [
            "Date",
            "Close",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram"
        ]
    ].tail(10)
)

print("\n========== MISSING VALUES ==========")
print(
    df[
        [
            "MACD",
            "MACD_Signal",
            "MACD_Histogram"
        ]
    ].isnull().sum()
)

output_file = "data/processed_TCS_features_macd.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)