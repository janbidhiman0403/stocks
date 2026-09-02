import pandas as pd

input_file = "data/ml_ready_TCS_v2.csv"

df = pd.read_csv(input_file)
df["Date"] = pd.to_datetime(df["Date"])

# Short-term return history
df["Return_Lag_1"] = df["Daily_Return"].shift(1)
df["Return_Lag_2"] = df["Daily_Return"].shift(2)
df["Return_Lag_5"] = df["Daily_Return"].shift(5)

# Previous-day indicator values
df["RSI_Lag_1"] = df["RSI_14"].shift(1)
df["Volatility_Lag_1"] = df["Volatility_20"].shift(1)
df["Volume_Ratio_Lag_1"] = df["Volume_Ratio"].shift(1)

print("========== LAG FEATURES ==========")

print(
    df[
        [
            "Date",
            "Daily_Return",
            "Return_Lag_1",
            "Return_Lag_2",
            "Return_Lag_5",
            "RSI_Lag_1",
            "Volatility_Lag_1",
            "Volume_Ratio_Lag_1"
        ]
    ].tail(10)
)

print("\n========== MISSING VALUES ==========")

print(
    df[
        [
            "Return_Lag_1",
            "Return_Lag_2",
            "Return_Lag_5",
            "RSI_Lag_1",
            "Volatility_Lag_1",
            "Volume_Ratio_Lag_1"
        ]
    ].isnull().sum()
)

output_file = "data/ml_ready_TCS_v3.csv"

df.to_csv(output_file, index=False)

print("\nSaved to:", output_file)