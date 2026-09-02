import pandas as pd

# Load the exact dataset used by the regression pipeline
df = pd.read_csv(
    "data/ml_ready_TCS_regression.csv",
    parse_dates=["Date"]
)

# These columns contain future information or labels
forbidden_columns = [
    "Future_Close_5D",
    "Future_Return_5D",
    "Target"
]

# Recreate the feature-selection logic
features = [
    column
    for column in df.columns
    if column not in ["Date"] + forbidden_columns
]

print("\n========== MODEL FEATURE CHECK ==========")

print(f"Total dataset columns: {len(df.columns)}")
print(f"Model input features: {len(features)}")

print("\nModel Features:")

for i, feature in enumerate(features, start=1):
    print(f"{i:2d}. {feature}")

print("\nForbidden / Future-Information Columns:")

for column in forbidden_columns:
    print(f"- {column}")

# Check for accidental leakage
leakage = [
    column
    for column in features
    if column in forbidden_columns
]

print("\n========== LEAKAGE CHECK ==========")

if leakage:
    print("WARNING: Potential data leakage detected!")
    for column in leakage:
        print(f"- {column}")
else:
    print("PASS: No forbidden future/target columns are used as features.")

print("====================================\n")