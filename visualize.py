import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/processed_TCS.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Create the chart
plt.figure(figsize=(14, 6))

plt.plot(df["Date"], df["Close"])

plt.title("TCS Stock Price — 2018 to 2025")
plt.xlabel("Date")
plt.ylabel("Price (₹)")

plt.grid(True)
plt.tight_layout()

plt.show()