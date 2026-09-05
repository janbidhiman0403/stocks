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
    confusion_matrix
)

print("=" * 70)
print("ALPHALENS V5 THRESHOLD ANALYSIS")
print("=" * 70)

INPUT_FILE = "data/walk_forward_v5_predictions.csv"
OUTPUT_FILE = "data/threshold_analysis_v5.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Missing input file: {INPUT_FILE}\n"
        "Run walk_forward_v5.py first."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDATASET")
print("-" * 70)
print("Rows   :", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for c in df.columns:
    print(" -", c)

# ------------------------------------------------------------
# Detect columns
# ------------------------------------------------------------

actual_candidates = [
    "Actual",
    "Target_10D_1pct",
    "Target"
]

probability_candidates = [
    "Probability_Up",
    "Probability",
    "Predicted_Probability",
    "Prob_Up"
]

actual_col = next(
    (c for c in actual_candidates if c in df.columns),
    None
)

prob_col = next(
    (c for c in probability_candidates if c in df.columns),
    None
)

if actual_col is None:
    raise ValueError(
        "Could not find actual target column. "
        f"Available columns: {df.columns.tolist()}"
    )

if prob_col is None:
    raise ValueError(
        "Could not find probability column. "
        f"Available columns: {df.columns.tolist()}"
    )

# ------------------------------------------------------------
# Clean data
# ------------------------------------------------------------

df = df[[c for c in df.columns if c in ["Date", actual_col, prob_col, "Prediction", "Window"]]].copy()

df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

df = df.dropna(subset=[actual_col, prob_col])

df[actual_col] = df[actual_col].astype(int)

df = df[
    (df[actual_col].isin([0, 1])) &
    (df[prob_col] >= 0) &
    (df[prob_col] <= 1)
].copy()

if len(df) == 0:
    raise ValueError("No valid rows remain after cleaning.")

# ------------------------------------------------------------
# Date
# ------------------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

# ------------------------------------------------------------
# Dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print("Rows used           :", len(df))
print("Target column       :", actual_col)
print("Probability column  :", prob_col)

if "Date" in df.columns and df["Date"].notna().any():
    print(
        "Date                :",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )

print("\nTarget distribution:")
print(df[actual_col].value_counts().sort_index())

print("\nTarget percentages:")
print(
    (df[actual_col].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

# ------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------

if df[actual_col].nunique() == 2:
    auc = roc_auc_score(df[actual_col], df[prob_col])
else:
    auc = np.nan

# ------------------------------------------------------------
# Threshold evaluation
# ------------------------------------------------------------

def evaluate_threshold(data, threshold):
    y_true = data[actual_col].values
    probabilities = data[prob_col].values

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Predicted_Positive_Rate": np.mean(y_pred),
        "Number_of_Positive_Predictions": int(y_pred.sum()),
        "True_Positive": int(tp),
        "False_Positive": int(fp),
        "True_Negative": int(tn),
        "False_Negative": int(fn)
    }


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline = evaluate_threshold(df, 0.50)

print("\n" + "=" * 70)
print("BASELINE @ 0.50")
print("=" * 70)

print("Rows                 :", len(df))
print("ROC-AUC              :", round(auc, 4))
print("Accuracy             :", round(baseline["Accuracy"], 4))
print(
    "Balanced Accuracy    :",
    round(baseline["Balanced_Accuracy"], 4)
)
print("Precision            :", round(baseline["Precision"], 4))
print("Recall               :", round(baseline["Recall"], 4))
print("F1                   :", round(baseline["F1"], 4))

# ------------------------------------------------------------
# Threshold grid
# ------------------------------------------------------------

thresholds = np.round(
    np.arange(0.10, 0.91, 0.01),
    2
)

results = []

for threshold in thresholds:
    results.append(
        evaluate_threshold(df, threshold)
    )

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# Top 10 helper
# ------------------------------------------------------------

def print_top(title, column):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    cols = [
        "Threshold",
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Predicted_Positive_Rate",
        "Number_of_Positive_Predictions"
    ]

    top = results_df.sort_values(
        column,
        ascending=False
    ).head(10)

    print(
        top[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ------------------------------------------------------------
# Rankings
# ------------------------------------------------------------

print_top(
    "TOP 10 THRESHOLDS — BALANCED ACCURACY",
    "Balanced_Accuracy"
)

print_top(
    "TOP 10 THRESHOLDS — F1",
    "F1"
)

print_top(
    "TOP 10 THRESHOLDS — PRECISION",
    "Precision"
)

print_top(
    "TOP 10 THRESHOLDS — RECALL",
    "Recall"
)

# ------------------------------------------------------------
# Recommended thresholds
# ------------------------------------------------------------

best_balanced = results_df.loc[
    results_df["Balanced_Accuracy"].idxmax()
]

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
]

best_precision = results_df.loc[
    results_df["Precision"].idxmax()
]

print("\n" + "=" * 70)
print("RECOMMENDED THRESHOLDS")
print("=" * 70)

def print_recommendation(title, row):
    print("\n" + title)
    print("-" * 22)

    print(
        "Threshold                          :",
        f"{row['Threshold']:.2f}"
    )

    print(
        "Accuracy                           :",
        f"{row['Accuracy']:.4f}"
    )

    print(
        "Balanced_Accuracy                  :",
        f"{row['Balanced_Accuracy']:.4f}"
    )

    print(
        "Precision                          :",
        f"{row['Precision']:.4f}"
    )

    print(
        "Recall                             :",
        f"{row['Recall']:.4f}"
    )

    print(
        "F1                                 :",
        f"{row['F1']:.4f}"
    )

    print(
        "Predicted_Positive_Rate            :",
        f"{row['Predicted_Positive_Rate']:.4f}"
    )

    print(
        "Number_of_Positive_Predictions     :",
        int(row["Number_of_Positive_Predictions"])
    )

    print(
        "TP / FP / TN / FN                  :",
        int(row["True_Positive"]),
        "/",
        int(row["False_Positive"]),
        "/",
        int(row["True_Negative"]),
        "/",
        int(row["False_Negative"])
    )


print_recommendation(
    "BEST BALANCED ACCURACY",
    best_balanced
)

print_recommendation(
    "BEST F1",
    best_f1
)

print_recommendation(
    "BEST PRECISION",
    best_precision
)

# ------------------------------------------------------------
# Practical thresholds
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRACTICAL THRESHOLD CHECK")
print("=" * 70)

practical_thresholds = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]

practical_rows = []

for threshold in practical_thresholds:
    row = evaluate_threshold(df, threshold)
    practical_rows.append(row)

practical_df = pd.DataFrame(practical_rows)

print(
    practical_df[
        [
            "Threshold",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Predicted_Positive_Rate",
            "Number_of_Positive_Predictions"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Probability distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION")
print("=" * 70)

print(
    df[prob_col].describe().to_string()
)

# ------------------------------------------------------------
# Probability separation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY BY ACTUAL CLASS")
print("=" * 70)

grouped = (
    df.groupby(actual_col)[prob_col]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    )
)

print(
    grouped.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# Window-level analysis
# ------------------------------------------------------------

if "Window" in df.columns:

    print("\n" + "=" * 70)
    print("WALK-FORWARD WINDOW THRESHOLD CHECK")
    print("=" * 70)

    window_rows = []

    for window, group in df.groupby("Window"):

        if len(group) < 10:
            continue

        if group[actual_col].nunique() < 2:
            window_auc = np.nan
        else:
            window_auc = roc_auc_score(
                group[actual_col],
                group[prob_col]
            )

        row = evaluate_threshold(group, 0.50)

        window_rows.append(
            {
                "Window": window,
                "Rows": len(group),
                "ROC_AUC": window_auc,
                "Balanced_Accuracy": row["Balanced_Accuracy"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "Predicted_Positive_Rate":
                    row["Predicted_Positive_Rate"]
            }
        )

    if window_rows:

        window_df = pd.DataFrame(window_rows)

        print(
            window_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_df = results_df.copy()

output_df["Model"] = "XGBoost"
output_df["Horizon"] = 10
output_df["Target"] = "Return_AtLeast_1.0pct"

output_df = output_df[
    [
        "Model",
        "Horizon",
        "Target",
        "Threshold",
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Predicted_Positive_Rate",
        "Number_of_Positive_Predictions",
        "True_Positive",
        "False_Positive",
        "True_Negative",
        "False_Negative"
    ]
]

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    f"ROC-AUC: {auc:.4f}"
)

print(
    f"Best balanced accuracy: "
    f"{best_balanced['Balanced_Accuracy']:.4f} "
    f"at threshold {best_balanced['Threshold']:.2f}"
)

print(
    f"Best F1: "
    f"{best_f1['F1']:.4f} "
    f"at threshold {best_f1['Threshold']:.2f}"
)

print(
    f"Best precision: "
    f"{best_precision['Precision']:.4f} "
    f"at threshold {best_precision['Threshold']:.2f}"
)

if auc >= 0.60:
    print(
        "PROMISING: Out-of-sample ranking signal is "
        "meaningfully above random."
    )
elif auc >= 0.55:
    print(
        "WEAK POSITIVE: Some ranking signal exists, "
        "but it is not yet strong."
    )
else:
    print(
        "WEAK: Ranking signal is close to random. "
        "Threshold optimization should not be treated "
        "as evidence of profitability."
    )

print("\n" + "=" * 70)
print("V5 THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print("Output:", OUTPUT_FILE)

print("\nPASS: V5 threshold analysis completed.")
print("=" * 70)