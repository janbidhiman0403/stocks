import pandas as pd

# Load AlphaLens signals
signals = pd.read_csv("data/alphalens_signals.csv")

# Load V3 + market dataset to obtain TCS volatility
market_data = pd.read_csv("data/ml_ready_TCS_market_v2.csv")

signals["Date"] = pd.to_datetime(signals["Date"])
market_data["Date"] = pd.to_datetime(market_data["Date"])

# Keep only the volatility feature we need
volatility = market_data[
    ["Date", "Volatility_20"]
].copy()

# Merge volatility into signal dataset
df = signals.merge(
    volatility,
    on="Date",
    how="left"
)

# Convert daily volatility into percentage
df["Volatility_Percent"] = (
    df["Volatility_20"] * 100
)

# Transparent rule-based risk score
#
# Lower 20-day volatility = lower risk
# Higher 20-day volatility = higher risk
def calculate_risk_score(volatility):
    volatility_percent = volatility * 100

    if volatility_percent < 1.0:
        return 20
    elif volatility_percent < 1.5:
        return 35
    elif volatility_percent < 2.0:
        return 50
    elif volatility_percent < 2.5:
        return 65
    elif volatility_percent < 3.0:
        return 80
    else:
        return 95


df["Risk_Score"] = df["Volatility_20"].apply(
    calculate_risk_score
)

# Convert numerical score into human-readable category
def risk_category(score):
    if score <= 35:
        return "Low"
    elif score <= 65:
        return "Moderate"
    elif score <= 80:
        return "High"
    else:
        return "Very High"


df["Risk_Level"] = df["Risk_Score"].apply(
    risk_category
)

# Latest available signal
latest = df.iloc[-1]

print("AlphaLens Risk Engine")
print("---------------------")
print()
print("Latest available assessment:")
print("Date:", latest["Date"].date())
print("TCS Close:", round(latest["Close"], 2))
print(
    "Predicted 5D return:",
    f"{latest['Predicted_Return_Percent']:.2f}%"
)
print("Signal:", latest["Signal"])
print("Signal strength:", latest["Signal_Strength"])
print(
    "20-day volatility:",
    f"{latest['Volatility_Percent']:.2f}%"
)
print("Risk score:", latest["Risk_Score"], "/ 100")
print("Risk level:", latest["Risk_Level"])

print()
print("Risk distribution:")
print(df["Risk_Level"].value_counts())

# Save final signal + risk dataset
output_file = "data/alphalens_signals_risk.csv"

df.to_csv(
    output_file,
    index=False
)

print()
print("Saved to:", output_file)