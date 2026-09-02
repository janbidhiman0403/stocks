import pandas as pd

# Load the feature dataset
df = pd.read_csv(
    "data/ml_ready_TCS_regression.csv",
    parse_dates=["Date"]
)

# Find zero-volume rows
zero_volume = df[df["Volume"] == 0].copy()

print("\n========== VOLUME CHECK ==========")

print(f"Total rows: {len(df)}")
print(f"Zero-volume rows: {len(zero_volume)}")
print(
    f"Zero-volume percentage: "
    f"{len(zero_volume) / len(df) * 100:.2f}%"
)

if len(zero_volume) > 0:

    print("\nZero-volume dates:")

    for date in zero_volume["Date"]:
        print(date.date())

    print("\nCorresponding Volume Ratios:")

    print(
        zero_volume[
            ["Date", "Volume", "Volume_SMA_20", "Volume_Ratio"]
        ].to_string(index=False)
    )

else:
    print("\nNo zero-volume rows found.")

print("=================================\n")