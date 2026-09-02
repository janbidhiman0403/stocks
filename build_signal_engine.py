import pandas as pd

# Load final V3 model predictions
df = pd.read_csv("data/xgboost_regression_final_test.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Validation-selected threshold.
# IMPORTANT: This threshold was selected using validation data,
# not the final test data.
buy_threshold = 0.00443

# Create investment signals
def generate_signal(predicted_return):
    if predicted_return >= buy_threshold:
        return "BUY"
    elif predicted_return <= -buy_threshold:
        return "AVOID"
    else:
        return "HOLD"


df["Signal"] = df["Predicted_Return"].apply(generate_signal)

# Signal strength
def signal_strength(predicted_return):
    magnitude = abs(predicted_return)

    if magnitude >= 0.03:
        return "Very Strong"
    elif magnitude >= 0.02:
        return "Strong"
    elif magnitude >= 0.01:
        return "Moderate"
    else:
        return "Weak"


df["Signal_Strength"] = df["Predicted_Return"].apply(
    signal_strength
)

# Convert predicted return to percentage
df["Predicted_Return_Percent"] = (
    df["Predicted_Return"] * 100
)

# Display latest signal
latest = df.iloc[-1]

print("AlphaLens Signal Engine")
print("-----------------------")
print()
print("Latest available prediction:")
print("Date:", latest["Date"].date())
print("TCS Close:", round(latest["Close"], 2))
print(
    "Predicted 5D return:",
    f"{latest['Predicted_Return_Percent']:.2f}%"
)
print("Signal:", latest["Signal"])
print("Signal strength:", latest["Signal_Strength"])

print()
print("Signal distribution:")
print(df["Signal"].value_counts())

print()
print("Signal percentages:")
print(
    (
        df["Signal"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

# Save signal dataset
output_file = "data/alphalens_signals.csv"

df.to_csv(output_file, index=False)

print()
print("Saved to:", output_file)