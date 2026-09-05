import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("=" * 70)
print("ALPHALENS V5 WALK-FORWARD VALIDATION")
print("=" * 70)

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "data/ml_features_v4.csv"

HORIZON = 10
TARGET_THRESHOLD = 0.01

TRAIN_SIZE = 800
TEST_SIZE = 100

OUTPUT_FILE = "data/walk_forward_v5_results.csv"

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading V4 features...")

df = pd.read_csv(FEATURE_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Date    : {df['Date'].min().date()} to {df['Date'].max().date()}")

# ============================================================
# CREATE 10-DAY TARGET
# ============================================================

print()
print("=" * 70)
print("CREATING 10-DAY TARGET")
print("=" * 70)

# Future close relative to today's close.
# IMPORTANT:
# shift(-HORIZON) uses FUTURE PRICE ONLY for TARGET CREATION.
# It is NOT included as a feature.

future_close = df["Close"].shift(-HORIZON)

future_return = (future_close / df["Close"]) - 1.0

df["Target_10D_1pct"] = (
    future_return >= TARGET_THRESHOLD
).astype(int)

# Remove rows where future target cannot be calculated
df = df.iloc[:-HORIZON].copy()

print(f"Horizon             : {HORIZON} trading days")
print(f"Target threshold    : {TARGET_THRESHOLD:.2%}")

print()
print("Target distribution:")
print(df["Target_10D_1pct"].value_counts())

print()
print("Target percentages:")
print(
    (df["Target_10D_1pct"].value_counts(normalize=True) * 100)
    .round(2)
)

# ============================================================
# FEATURE SELECTION
# ============================================================

TARGET = "Target_10D_1pct"

EXCLUDE_COLUMNS = [
    "Date",
    TARGET,
]

# These are raw price/market columns that were present in V4.
# We keep the same general V4 feature universe used in the
# successful V5 experiment.
feature_columns = [
    c for c in df.columns
    if c not in EXCLUDE_COLUMNS
    and pd.api.types.is_numeric_dtype(df[c])
]

# Safety check:
# No future target columns should accidentally enter X.
dangerous = []

for col in feature_columns:
    name = col.lower()

    if "target" in name:
        dangerous.append(col)

    if "future" in name:
        dangerous.append(col)

if dangerous:
    print()
    print("ERROR: Dangerous feature columns detected:")
    for col in dangerous:
        print(" -", col)
    raise ValueError(
        "Future/target-like columns detected in feature set."
    )

print()
print("=" * 70)
print("FEATURE SET")
print("=" * 70)

print(f"Features : {len(feature_columns)}")

# ============================================================
# DATA ARRAYS
# ============================================================

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Replace any accidental infinities
X = X.replace([np.inf, -np.inf], np.nan)

# V4 audit reported no missing values, but keep this safety step.
if X.isna().any().any():
    print()
    print("Missing values detected. Filling with forward/backward values.")
    X = X.ffill().bfill()

# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

print()
print("=" * 70)
print("WALK-FORWARD WINDOWS")
print("=" * 70)

print()
print(
    f"TRAIN={TRAIN_SIZE} TEST={TEST_SIZE} "
    f"HORIZON={HORIZON} DAYS"
)

results = []

window_number = 0
train_end = TRAIN_SIZE

all_predictions = []

while train_end < len(df):

    test_start = train_end
    test_end = min(test_start + TEST_SIZE, len(df))

    if test_start >= test_end:
        break

    window_number += 1

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    train_dates = df["Date"].iloc[:train_end]
    test_dates = df["Date"].iloc[test_start:test_end]

    print()
    print("-" * 70)

    print(
        f"Window {window_number}: "
        f"TRAIN={len(X_train)} "
        f"TEST={len(X_test)} "
        f"{test_dates.iloc[0].date()} "
        f"to "
        f"{test_dates.iloc[-1].date()}"
    )

    print(
        f"Train target rate: {y_train.mean():.3f} | "
        f"Test target rate: {y_test.mean():.3f}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        reg_alpha=0.10,
        reg_lambda=1.50,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(X_test)[:, 1]

    prediction = (probability >= 0.50).astype(int)

    # ========================================================
    # METRICS
    # ========================================================

    if y_test.nunique() >= 2:
        auc = roc_auc_score(y_test, probability)
        bal = balanced_accuracy_score(y_test, prediction)
    else:
        auc = np.nan
        bal = np.nan

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    accuracy = (prediction == y_test).mean()

    print(
        f"XGBoost "
        f"AUC={auc:.4f} "
        f"BAL={bal:.4f} "
        f"F1={f1:.4f} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f}"
    )

    # ========================================================
    # STORE WINDOW RESULT
    # ========================================================

    results.append({
        "Window": window_number,
        "Train_Size": len(X_train),
        "Test_Size": len(X_test),
        "Train_Start": train_dates.iloc[0],
        "Train_End": train_dates.iloc[-1],
        "Test_Start": test_dates.iloc[0],
        "Test_End": test_dates.iloc[-1],
        "Train_Target_Rate": y_train.mean(),
        "Test_Target_Rate": y_test.mean(),
        "Accuracy": accuracy,
        "Balanced_Accuracy": bal,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
    })

    # ========================================================
    # STORE INDIVIDUAL PREDICTIONS
    # ========================================================

    window_predictions = pd.DataFrame({
        "Date": test_dates.values,
        "Actual": y_test.values,
        "Probability_Up": probability,
        "Prediction": prediction,
        "Window": window_number,
    })

    all_predictions.append(window_predictions)

    train_end = test_end

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

print()
print("Windows tested :", len(results_df))
print("Samples tested :", len(predictions_df))

print()
print("Per-window results:")
print()

display_columns = [
    "Window",
    "Test_Start",
    "Test_End",
    "ROC_AUC",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1",
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)

# ============================================================
# AGGREGATE OUT-OF-SAMPLE METRICS
# ============================================================

actual = predictions_df["Actual"]
probability = predictions_df["Probability_Up"]
prediction = predictions_df["Prediction"]

if actual.nunique() >= 2:

    overall_auc = roc_auc_score(
        actual,
        probability
    )

    overall_balanced = balanced_accuracy_score(
        actual,
        prediction
    )

else:
    overall_auc = np.nan
    overall_balanced = np.nan

overall_accuracy = (
    prediction == actual
).mean()

overall_precision = precision_score(
    actual,
    prediction,
    zero_division=0
)

overall_recall = recall_score(
    actual,
    prediction,
    zero_division=0
)

overall_f1 = f1_score(
    actual,
    prediction,
    zero_division=0
)

print()
print("=" * 70)
print("AGGREGATED OUT-OF-SAMPLE PERFORMANCE")
print("=" * 70)

print()
print(f"Samples             : {len(predictions_df)}")
print(f"ROC-AUC             : {overall_auc:.4f}")
print(f"Accuracy            : {overall_accuracy:.4f}")
print(f"Balanced Accuracy   : {overall_balanced:.4f}")
print(f"Precision           : {overall_precision:.4f}")
print(f"Recall              : {overall_recall:.4f}")
print(f"F1                  : {overall_f1:.4f}")

# ============================================================
# WINDOW STABILITY
# ============================================================

valid_auc = results_df["ROC_AUC"].dropna()

print()
print("=" * 70)
print("WALK-FORWARD STABILITY")
print("=" * 70)

print()
print(f"Mean Window AUC     : {valid_auc.mean():.4f}")
print(f"Median Window AUC   : {valid_auc.median():.4f}")
print(f"Std Window AUC      : {valid_auc.std():.4f}")
print(f"Best Window AUC     : {valid_auc.max():.4f}")
print(f"Worst Window AUC    : {valid_auc.min():.4f}")

positive_auc_windows = (
    valid_auc > 0.50
).sum()

print(
    f"Windows AUC > 0.50  : "
    f"{positive_auc_windows}/{len(valid_auc)}"
)

strong_auc_windows = (
    valid_auc >= 0.55
).sum()

print(
    f"Windows AUC >= 0.55 : "
    f"{strong_auc_windows}/{len(valid_auc)}"
)

# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if overall_auc >= 0.60 and positive_auc_windows >= len(valid_auc) * 0.60:

    print()
    print(
        "STRONG: The 10-day V5 signal appears "
        "reasonably persistent across walk-forward windows."
    )

elif overall_auc >= 0.55 and positive_auc_windows >= len(valid_auc) * 0.50:

    print()
    print(
        "PROMISING: The 10-day V5 signal survives "
        "walk-forward testing but remains moderate."
    )

elif overall_auc > 0.50:

    print()
    print(
        "WEAK POSITIVE: Some out-of-sample signal exists, "
        "but stability is insufficient for confidence."
    )

else:

    print()
    print(
        "FAIL: The apparent V5 test-set advantage "
        "does not survive walk-forward validation."
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

prediction_output = (
    "data/walk_forward_v5_predictions.csv"
)

predictions_df.to_csv(
    prediction_output,
    index=False
)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print()
print(f"Window results : {OUTPUT_FILE}")
print(f"Predictions    : {prediction_output}")

print()
print("=" * 70)
print("V5 WALK-FORWARD VALIDATION COMPLETE")
print("=" * 70)

print()
print("PASS: V5 walk-forward validation completed.")
print("=" * 70)