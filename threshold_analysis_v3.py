import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# ALPHALENS V3 THRESHOLD ANALYSIS
# ============================================================

INPUT = Path("data/v3_model_predictions.csv")
OUTPUT = Path("data/threshold_analysis_v3.csv")

print("=" * 70)
print("ALPHALENS V3 THRESHOLD ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found: {INPUT}\n"
        "Run train_v3_models.py first."
    )

df = pd.read_csv(INPUT)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for col in df.columns:
    print(" -", col)

# ------------------------------------------------------------
# EXPECTED COLUMNS
# ------------------------------------------------------------

TARGET_COL = "Actual"
PROB_COL = "Probability_Up"

if TARGET_COL not in df.columns:
    raise ValueError(
        f"\nMissing target column: {TARGET_COL}"
    )

if PROB_COL not in df.columns:
    raise ValueError(
        f"\nMissing probability column: {PROB_COL}"
    )

# ------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------

work = df[
    [c for c in ["Date", "Actual", "Prediction", "Probability_Up", "Model"]
     if c in df.columns]
].copy()

work["Actual"] = pd.to_numeric(
    work["Actual"],
    errors="coerce"
)

work["Probability_Up"] = pd.to_numeric(
    work["Probability_Up"],
    errors="coerce"
)

work = work.dropna(
    subset=["Actual", "Probability_Up"]
)

y_true = work["Actual"].astype(int).values
prob = work["Probability_Up"].values

# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

auc = roc_auc_score(
    y_true,
    prob
)

default_pred = (
    prob >= 0.50
).astype(int)

print("\n" + "=" * 70)
print("BASELINE @ 0.50")
print("=" * 70)

print("Rows                 :", len(work))
print("ROC-AUC              :", round(auc, 4))
print("Accuracy             :", round(
    accuracy_score(y_true, default_pred), 4
))
print("Balanced Accuracy    :", round(
    balanced_accuracy_score(y_true, default_pred), 4
))
print("Precision            :", round(
    precision_score(y_true, default_pred, zero_division=0), 4
))
print("Recall               :", round(
    recall_score(y_true, default_pred, zero_division=0), 4
))
print("F1                  :", round(
    f1_score(y_true, default_pred, zero_division=0), 4
))

# ------------------------------------------------------------
# THRESHOLD SEARCH
# ------------------------------------------------------------

results = []

thresholds = np.arange(
    0.20,
    0.81,
    0.01
)

for threshold in thresholds:

    pred = (
        prob >= threshold
    ).astype(int)

    results.append({
        "Threshold": round(float(threshold), 2),

        "Accuracy": accuracy_score(
            y_true,
            pred
        ),

        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            pred
        ),

        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Predicted_Positive_Rate": pred.mean(),

        "Number_of_Positive_Predictions": pred.sum()
    })

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# TOP BALANCED ACCURACY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 THRESHOLDS — BALANCED ACCURACY")
print("=" * 70)

print(
    results_df
    .sort_values(
        "Balanced_Accuracy",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# TOP F1
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 THRESHOLDS — F1")
print("=" * 70)

print(
    results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# TOP PRECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 THRESHOLDS — PRECISION")
print("=" * 70)

print(
    results_df
    .sort_values(
        "Precision",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# TOP RECALL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 THRESHOLDS — RECALL")
print("=" * 70)

print(
    results_df
    .sort_values(
        "Recall",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# BEST THRESHOLDS
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

print("\nBEST BALANCED ACCURACY")
print("----------------------")
print(best_balanced.to_string())

print("\nBEST F1")
print("----------------------")
print(best_f1.to_string())

print("\nBEST PRECISION")
print("----------------------")
print(best_precision.to_string())

# ------------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(prob).describe().to_string()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print("Output:", OUTPUT)

print("\nPASS: Threshold analysis completed.")