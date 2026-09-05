import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# ALPHALENS V4 THRESHOLD ANALYSIS
# ============================================================

INPUT_FILE = "data/v4_model_predictions.csv"
OUTPUT_FILE = "data/threshold_analysis_v4.csv"

THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.01), 2)


def find_column(df, candidates):
    """
    Find a column using exact match first, then case-insensitive match.
    """
    for name in candidates:
        if name in df.columns:
            return name

    lower_map = {str(col).lower(): col for col in df.columns}

    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def calculate_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics for one probability threshold.
    """
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Predicted_Positive_Rate": predictions.mean(),
        "Number_of_Positive_Predictions": int(predictions.sum()),
    }


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ALPHALENS V4 THRESHOLD ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Make sure train_v4_models.py has been run successfully."
    )

df = pd.read_csv(INPUT_FILE)

print()
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print()
print("Columns:")
for col in df.columns:
    print(f" - {col}")


# ============================================================
# IDENTIFY TARGET AND PROBABILITY COLUMNS
# ============================================================

target_col = find_column(
    df,
    [
        "Target_Up_V3",
        "Target_Up_V4",
        "Actual",
        "y_true",
        "target",
        "Target",
    ],
)

probability_col = find_column(
    df,
    [
        "Probability_Up",
        "Probability",
        "Predicted_Probability",
        "Probability_1",
        "Prob_Up",
        "proba",
    ],
)

prediction_col = find_column(
    df,
    [
        "Prediction",
        "Predicted",
        "y_pred",
        "Predicted_Class",
    ],
)

date_col = find_column(
    df,
    [
        "Date",
        "date",
    ],
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

if target_col is None:
    raise ValueError(
        "\nCould not find target column.\n"
        "Expected one of:\n"
        "Target_Up_V3\n"
        "Target_Up_V4\n"
        "Actual\n"
        "y_true\n"
        "target\n"
        "Target"
    )

if probability_col is None:
    raise ValueError(
        "\nCould not find probability column.\n"
        "Expected one of:\n"
        "Probability_Up\n"
        "Probability\n"
        "Predicted_Probability\n"
        "Probability_1\n"
        "Prob_Up\n"
        "proba"
    )


# ============================================================
# CLEAN DATA
# ============================================================

work = df.copy()

work[target_col] = pd.to_numeric(
    work[target_col],
    errors="coerce",
)

work[probability_col] = pd.to_numeric(
    work[probability_col],
    errors="coerce",
)

work = work.dropna(
    subset=[
        target_col,
        probability_col,
    ]
).copy()

work[target_col] = work[target_col].astype(int)

y_true = work[target_col].to_numpy()

probabilities = work[probability_col].to_numpy()


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(y_true) == 0:
    raise ValueError("No valid rows remain after cleaning.")

unique_targets = sorted(np.unique(y_true))

if not set(unique_targets).issubset({0, 1}):
    raise ValueError(
        f"Target must contain only 0 and 1. "
        f"Found: {unique_targets}"
    )

if np.any(probabilities < 0) or np.any(probabilities > 1):
    raise ValueError(
        "Probability column contains values outside [0, 1]."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

print_section("DATASET")

print(f"Rows used           : {len(work)}")
print(f"Target column       : {target_col}")
print(f"Probability column  : {probability_col}")

if prediction_col is not None:
    print(f"Prediction column   : {prediction_col}")

if date_col is not None:
    try:
        dates = pd.to_datetime(work[date_col], errors="coerce")
        valid_dates = dates.dropna()

        if len(valid_dates) > 0:
            print(
                f"Date                : "
                f"{valid_dates.min().date()} to "
                f"{valid_dates.max().date()}"
            )
    except Exception:
        pass


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print_section("TARGET DISTRIBUTION")

target_counts = pd.Series(y_true).value_counts().sort_index()

print(target_counts)

print()
print("Percentages:")

target_percentages = (
    pd.Series(y_true)
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(target_percentages.round(2))


# ============================================================
# BASELINE @ 0.50
# ============================================================

baseline_threshold = 0.50

baseline_predictions = (
    probabilities >= baseline_threshold
).astype(int)

baseline_auc = roc_auc_score(
    y_true,
    probabilities,
)

print_section("BASELINE @ 0.50")

print(f"Rows                 : {len(y_true)}")
print(f"ROC-AUC              : {baseline_auc:.4f}")
print(
    f"Accuracy             : "
    f"{accuracy_score(y_true, baseline_predictions):.4f}"
)
print(
    f"Balanced Accuracy    : "
    f"{balanced_accuracy_score(y_true, baseline_predictions):.4f}"
)
print(
    f"Precision            : "
    f"{precision_score(y_true, baseline_predictions, zero_division=0):.4f}"
)
print(
    f"Recall               : "
    f"{recall_score(y_true, baseline_predictions, zero_division=0):.4f}"
)
print(
    f"F1                   : "
    f"{f1_score(y_true, baseline_predictions, zero_division=0):.4f}"
)


# ============================================================
# THRESHOLD GRID
# ============================================================

results = []

for threshold in THRESHOLDS:
    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold,
    )

    results.append(metrics)


threshold_df = pd.DataFrame(results)


# ============================================================
# TOP 10 — BALANCED ACCURACY
# ============================================================

top_balanced = (
    threshold_df
    .sort_values(
        "Balanced_Accuracy",
        ascending=False,
    )
    .head(10)
)

print_section("TOP 10 THRESHOLDS — BALANCED ACCURACY")

print(
    top_balanced.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# TOP 10 — F1
# ============================================================

top_f1 = (
    threshold_df
    .sort_values(
        "F1",
        ascending=False,
    )
    .head(10)
)

print_section("TOP 10 THRESHOLDS — F1")

print(
    top_f1.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# TOP 10 — PRECISION
# ============================================================

top_precision = (
    threshold_df
    .sort_values(
        [
            "Precision",
            "Recall",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(10)
)

print_section("TOP 10 THRESHOLDS — PRECISION")

print(
    top_precision.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# TOP 10 — RECALL
# ============================================================

top_recall = (
    threshold_df
    .sort_values(
        "Recall",
        ascending=False,
    )
    .head(10)
)

print_section("TOP 10 THRESHOLDS — RECALL")

print(
    top_recall.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# RECOMMENDED THRESHOLDS
# ============================================================

best_balanced = threshold_df.loc[
    threshold_df["Balanced_Accuracy"].idxmax()
]

best_f1 = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_precision = threshold_df.loc[
    threshold_df["Precision"].idxmax()
]


print_section("RECOMMENDED THRESHOLDS")

print()
print("BEST BALANCED ACCURACY")
print("-" * 22)

for col in threshold_df.columns:
    print(
        f"{col:<35}: "
        f"{best_balanced[col]:.6f}"
        if isinstance(
            best_balanced[col],
            (float, np.floating),
        )
        else
        f"{col:<35}: "
        f"{best_balanced[col]}"
    )


print()
print("BEST F1")
print("-" * 22)

for col in threshold_df.columns:
    print(
        f"{col:<35}: "
        f"{best_f1[col]:.6f}"
        if isinstance(
            best_f1[col],
            (float, np.floating),
        )
        else
        f"{col:<35}: "
        f"{best_f1[col]}"
    )


print()
print("BEST PRECISION")
print("-" * 22)

for col in threshold_df.columns:
    print(
        f"{col:<35}: "
        f"{best_precision[col]:.6f}"
        if isinstance(
            best_precision[col],
            (float, np.floating),
        )
        else
        f"{col:<35}: "
        f"{best_precision[col]}"
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

print_section("PROBABILITY DISTRIBUTION")

print(
    pd.Series(probabilities).describe()
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

threshold_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print_section("THRESHOLD ANALYSIS COMPLETE")

print(f"Output: {OUTPUT_FILE}")

print()
print("PASS: V4 threshold analysis completed.")
print("=" * 70) 