import os
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
roc_auc_score,
accuracy_score,
balanced_accuracy_score,
precision_score,
recall_score,
f1_score,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ======================================================================

# CONFIGURATION

# ======================================================================

DATA_DIR = "data"

FEATURE_FILE = os.path.join(DATA_DIR, "ml_features_v4.csv")
TARGET_FILE = os.path.join(DATA_DIR, "ml_targets_v3.csv")
SAFE_FILE = os.path.join(
DATA_DIR,
"feature_stability_v6_safe_features.csv",
)

RESULT_FILE = os.path.join(
DATA_DIR,
"walk_forward_v6_results.csv",
)

PRED_FILE = os.path.join(
DATA_DIR,
"walk_forward_v6_predictions.csv",
)

DATE_COLUMN = "Date"

# This column EXISTS in ml_targets_v3.csv

TARGET_COLUMN = "Target_Direction_10D_V3"

TRAIN_SIZE = 1000
TEST_SIZE = 100
STEP_SIZE = 100

RANDOM_STATE = 42

# ======================================================================

# DISPLAY HELPERS

# ======================================================================

def header(text):
print()
print("=" * 70)
print(text)
print("=" * 70)

def subheader(text):
print()
print(text)
print("-" * 70)

# ======================================================================

# MODEL HELPERS

# ======================================================================

def make_logistic_model():
return Pipeline(
[
(
"scale",
StandardScaler(),
),
(
"model",
LogisticRegression(
max_iter=3000,
class_weight="balanced",
random_state=RANDOM_STATE,
),
),
]
)

def make_xgb_model():
return XGBClassifier(
n_estimators=250,
max_depth=3,
learning_rate=0.03,
subsample=0.80,
colsample_bytree=0.80,
min_child_weight=5,
reg_alpha=0.10,
reg_lambda=1.0,
objective="binary:logistic",
eval_metric="logloss",
random_state=RANDOM_STATE,
n_jobs=-1,
)

# ======================================================================

# METRICS

# ======================================================================

def calculate_metrics(y_true, probability, threshold=0.50):
y_true = np.asarray(y_true).astype(int)
probability = np.asarray(probability).astype(float)

```
prediction = (
    probability >= threshold
).astype(int)

if len(np.unique(y_true)) > 1:
    auc = roc_auc_score(
        y_true,
        probability,
    )
else:
    auc = np.nan

return {
    "AUC": auc,
    "Accuracy": accuracy_score(
        y_true,
        prediction,
    ),
    "Balanced_Accuracy": balanced_accuracy_score(
        y_true,
        prediction,
    ),
    "Precision": precision_score(
        y_true,
        prediction,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_true,
        prediction,
        zero_division=0,
    ),
    "F1": f1_score(
        y_true,
        prediction,
        zero_division=0,
    ),
    "Predicted_Positive_Rate": prediction.mean(),
    "Number_of_Predicted_Positive": int(
        prediction.sum()
    ),
}
```

# ======================================================================

# LOAD SAFE FEATURES

# ======================================================================

def load_safe_features():
if not os.path.exists(SAFE_FILE):
raise FileNotFoundError(
f"Could not find SAFE feature file:\n{SAFE_FILE}"
)

```
safe_df = pd.read_csv(SAFE_FILE)

if "Feature" in safe_df.columns:
    features = (
        safe_df["Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )
else:
    # Fallback for files containing one feature name per row
    features = (
        safe_df.iloc[:, 0]
        .dropna()
        .astype(str)
        .tolist()
    )

features = list(
    dict.fromkeys(features)
)

if len(features) == 0:
    raise ValueError(
        "SAFE feature file contains no features."
    )

return features
```

# ======================================================================

# LOAD DATA

# ======================================================================

def load_inputs():
if not os.path.exists(FEATURE_FILE):
raise FileNotFoundError(
f"Could not find:\n{FEATURE_FILE}"
)

```
if not os.path.exists(TARGET_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{TARGET_FILE}"
    )

features_df = pd.read_csv(
    FEATURE_FILE
)

targets_df = pd.read_csv(
    TARGET_FILE
)

return features_df, targets_df
```

# ======================================================================

# PREPARE DATASET

# ======================================================================

def prepare_dataset():
header("ALPHALENS V6 WALK-FORWARD VALIDATION")

```
print("Loading V6 inputs...")

features_df, targets_df = load_inputs()

print(
    f"Feature rows    : {len(features_df)}"
)

print(
    f"Feature columns : {len(features_df.columns)}"
)

print(
    f"Target rows     : {len(targets_df)}"
)

print(
    f"Target columns  : {len(targets_df.columns)}"
)

# --------------------------------------------------------------
# DATE CHECK
# --------------------------------------------------------------

subheader("DATE ALIGNMENT")

if DATE_COLUMN not in features_df.columns:
    raise ValueError(
        f"{DATE_COLUMN} missing from feature file."
    )

if DATE_COLUMN not in targets_df.columns:
    raise ValueError(
        f"{DATE_COLUMN} missing from target file."
    )

features_df[DATE_COLUMN] = pd.to_datetime(
    features_df[DATE_COLUMN],
    errors="coerce",
)

targets_df[DATE_COLUMN] = pd.to_datetime(
    targets_df[DATE_COLUMN],
    errors="coerce",
)

features_df = features_df.dropna(
    subset=[DATE_COLUMN]
)

targets_df = targets_df.dropna(
    subset=[DATE_COLUMN]
)

features_df = features_df.sort_values(
    DATE_COLUMN
)

targets_df = targets_df.sort_values(
    DATE_COLUMN
)

print(
    "Feature date range : "
    f"{features_df[DATE_COLUMN].min().date()} "
    f"to "
    f"{features_df[DATE_COLUMN].max().date()}"
)

print(
    "Target date range  : "
    f"{targets_df[DATE_COLUMN].min().date()} "
    f"to "
    f"{targets_df[DATE_COLUMN].max().date()}"
)

print(
    f"Target             : {TARGET_COLUMN}"
)

# --------------------------------------------------------------
# TARGET CHECK
# --------------------------------------------------------------

if TARGET_COLUMN not in targets_df.columns:
    print()
    print(
        "AVAILABLE TARGET COLUMNS:"
    )

    for column in targets_df.columns:
        if (
            "Target" in column
            or "target" in column
        ):
            print(
                f" - {column}"
            )

    raise ValueError(
        f"\n{TARGET_COLUMN} missing from target file."
    )

# --------------------------------------------------------------
# SAFE FEATURES
# --------------------------------------------------------------

subheader("SAFE FEATURES")

safe_features = load_safe_features()

print(
    f"SAFE features listed : {len(safe_features)}"
)

usable_features = []

for feature in safe_features:
    if feature in features_df.columns:
        usable_features.append(feature)

print(
    f"SAFE features usable : {len(usable_features)}"
)

if len(usable_features) == 0:
    raise ValueError(
        "No SAFE features exist in ml_features_v4.csv."
    )

# --------------------------------------------------------------
# MERGE
# --------------------------------------------------------------

subheader("MERGE")

feature_part = features_df[
    [DATE_COLUMN] + usable_features
].copy()

target_part = targets_df[
    [DATE_COLUMN, TARGET_COLUMN]
].copy()

feature_part = feature_part.drop_duplicates(
    subset=[DATE_COLUMN],
    keep="last",
)

target_part = target_part.drop_duplicates(
    subset=[DATE_COLUMN],
    keep="last",
)

data = pd.merge(
    feature_part,
    target_part,
    on=DATE_COLUMN,
    how="inner",
)

data = data.sort_values(
    DATE_COLUMN
).reset_index(drop=True)

print(
    f"Merged rows : {len(data)}"
)

if len(data) == 0:
    raise ValueError(
        "Merged dataset contains zero rows."
    )

# --------------------------------------------------------------
# TARGET CLEANING
# --------------------------------------------------------------

data[TARGET_COLUMN] = pd.to_numeric(
    data[TARGET_COLUMN],
    errors="coerce",
)

data = data.dropna(
    subset=[TARGET_COLUMN]
).copy()

data[TARGET_COLUMN] = (
    data[TARGET_COLUMN]
    .astype(int)
)

# Keep only binary target rows
data = data[
    data[TARGET_COLUMN].isin([0, 1])
].copy()

data = data.reset_index(
    drop=True
)

print()
print("Target distribution:")

print(
    data[TARGET_COLUMN].value_counts()
)

print()

print(
    "Target rate: "
    f"{data[TARGET_COLUMN].mean():.4f}"
)

if len(data) < (
    TRAIN_SIZE + TEST_SIZE
):
    raise ValueError(
        "Not enough rows for walk-forward validation."
    )

return (
    data,
    usable_features,
)
```

# ======================================================================

# WALK-FORWARD

# ======================================================================

def run_walk_forward(
data,
features,
model_name,
model_factory,
):
header(
f"V6 WALK-FORWARD — {model_name}"
)

```
all_predictions = []
fold_results = []

n_rows = len(data)

fold_number = 0

train_end = TRAIN_SIZE

while (
    train_end + TEST_SIZE
    <= n_rows
):
    fold_number += 1

    test_start = train_end
    test_end = (
        test_start + TEST_SIZE
    )

    train_data = data.iloc[
        :train_end
    ].copy()

    test_data = data.iloc[
        test_start:test_end
    ].copy()

    X_train = train_data[
        features
    ].copy()

    y_train = train_data[
        TARGET_COLUMN
    ].copy()

    X_test = test_data[
        features
    ].copy()

    y_test = test_data[
        TARGET_COLUMN
    ].copy()

    # ----------------------------------------------------------
    # NUMERIC CLEANING
    # ----------------------------------------------------------

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Prevent future information from
    # entering preprocessing.
    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_train = X_train.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    # Any column still completely invalid
    # receives zero using training-only logic.
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    # ----------------------------------------------------------
    # TARGET CHECK
    # ----------------------------------------------------------

    if (
        len(np.unique(y_train))
        < 2
    ):
        print(
            f"Fold {fold_number:02d}: "
            "SKIPPED — training target has one class."
        )

        train_end += STEP_SIZE
        continue

    if (
        len(np.unique(y_test))
        < 2
    ):
        print(
            f"Fold {fold_number:02d}: "
            "WARNING — test target has one class."
        )

    # ----------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------

    model = model_factory()

    model.fit(
        X_train,
        y_train,
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        probability,
        threshold=0.50,
    )

    train_probability = model.predict_proba(
        X_train
    )[:, 1]

    if len(
        np.unique(y_train)
    ) > 1:
        train_auc = roc_auc_score(
            y_train,
            train_probability,
        )
    else:
        train_auc = np.nan

    # ----------------------------------------------------------
    # FOLD OUTPUT
    # ----------------------------------------------------------

    train_start_date = (
        train_data[DATE_COLUMN].iloc[0]
    )

    train_end_date = (
        train_data[DATE_COLUMN].iloc[-1]
    )

    test_start_date = (
        test_data[DATE_COLUMN].iloc[0]
    )

    test_end_date = (
        test_data[DATE_COLUMN].iloc[-1]
    )

    fold_row = {
        "Model": model_name,
        "Fold": fold_number,
        "Train_Rows": len(train_data),
        "Test_Rows": len(test_data),
        "Train_Start": train_start_date,
        "Train_End": train_end_date,
        "Test_Start": test_start_date,
        "Test_End": test_end_date,
        "Train_AUC": train_auc,
        "Test_AUC": metrics["AUC"],
        "Accuracy": metrics["Accuracy"],
        "Balanced_Accuracy": metrics[
            "Balanced_Accuracy"
        ],
        "Precision": metrics[
            "Precision"
        ],
        "Recall": metrics[
            "Recall"
        ],
        "F1": metrics["F1"],
        "Predicted_Positive_Rate": metrics[
            "Predicted_Positive_Rate"
        ],
        "Number_of_Predicted_Positive": metrics[
            "Number_of_Predicted_Positive"
        ],
        "Train_Target_Rate": y_train.mean(),
        "Test_Target_Rate": y_test.mean(),
    }

    fold_results.append(
        fold_row
    )

    # ----------------------------------------------------------
    # PREDICTIONS
    # ----------------------------------------------------------

    for i in range(
        len(test_data)
    ):
        all_predictions.append(
            {
                "Model": model_name,
                "Fold": fold_number,
                "Date": test_data[
                    DATE_COLUMN
                ].iloc[i],
                "Actual": int(
                    y_test.iloc[i]
                ),
                "Probability_Up": float(
                    probability[i]
                ),
                "Prediction": int(
                    probability[i] >= 0.50
                ),
            }
        )

    print(
        f"Fold {fold_number:02d} | "
        f"Train {len(train_data):4d} | "
        f"Test {len(test_data):3d} | "
        f"Train AUC {train_auc:.4f} | "
        f"Test AUC {metrics['AUC']:.4f} | "
        f"BAL {metrics['Balanced_Accuracy']:.4f} | "
        f"F1 {metrics['F1']:.4f}"
    )

    train_end += STEP_SIZE

return (
    pd.DataFrame(fold_results),
    pd.DataFrame(all_predictions),
)
```

# ======================================================================

# MAIN

# ======================================================================

def main():
data, usable_features = (
prepare_dataset()
)

```
header(
    "V6 WALK-FORWARD CONFIGURATION"
)

print(
    f"Target              : {TARGET_COLUMN}"
)

print(
    f"SAFE features        : {len(usable_features)}"
)

print(
    f"Training window      : {TRAIN_SIZE}"
)

print(
    f"Test window          : {TEST_SIZE}"
)

print(
    f"Step size            : {STEP_SIZE}"
)

print(
    f"Total usable rows    : {len(data)}"
)

# --------------------------------------------------------------
# RUN LOGISTIC
# --------------------------------------------------------------

logistic_results, logistic_predictions = (
    run_walk_forward(
        data,
        usable_features,
        "Logistic",
        make_logistic_model,
    )
)

# --------------------------------------------------------------
# RUN XGBOOST
# --------------------------------------------------------------

xgb_results, xgb_predictions = (
    run_walk_forward(
        data,
        usable_features,
        "XGBoost",
        make_xgb_model,
    )
)

# --------------------------------------------------------------
# COMBINE
# --------------------------------------------------------------

results = pd.concat(
    [
        logistic_results,
        xgb_results,
    ],
    ignore_index=True,
)

predictions = pd.concat(
    [
        logistic_predictions,
        xgb_predictions,
    ],
    ignore_index=True,
)

# --------------------------------------------------------------
# SAVE
# --------------------------------------------------------------

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)

results.to_csv(
    RESULT_FILE,
    index=False,
)

predictions.to_csv(
    PRED_FILE,
    index=False,
)

# --------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------

header(
    "V6 WALK-FORWARD SUMMARY"
)

if len(results) == 0:
    raise ValueError(
        "No walk-forward folds were successfully evaluated."
    )

summary_rows = []

for model_name in [
    "Logistic",
    "XGBoost",
]:
    model_results = results[
        results["Model"]
        == model_name
    ].copy()

    if len(model_results) == 0:
        continue

    summary_rows.append(
        {
            "Model": model_name,
            "Folds": len(model_results),
            "Mean_Train_AUC": model_results[
                "Train_AUC"
            ].mean(),
            "Mean_Test_AUC": model_results[
                "Test_AUC"
            ].mean(),
            "Median_Test_AUC": model_results[
                "Test_AUC"
            ].median(),
            "Mean_Balanced_Accuracy": model_results[
                "Balanced_Accuracy"
            ].mean(),
            "Mean_Precision": model_results[
                "Precision"
            ].mean(),
            "Mean_Recall": model_results[
                "Recall"
            ].mean(),
            "Mean_F1": model_results[
                "F1"
            ].mean(),
            "Std_Test_AUC": model_results[
                "Test_AUC"
            ].std(),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

print()

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# --------------------------------------------------------------
# BEST MODEL
# --------------------------------------------------------------

if len(summary) > 0:
    best_index = summary[
        "Mean_Test_AUC"
    ].idxmax()

    best_model = summary.loc[
        best_index,
        "Model",
    ]

    best_auc = summary.loc[
        best_index,
        "Mean_Test_AUC",
    ]

    best_balanced = summary.loc[
        best_index,
        "Mean_Balanced_Accuracy",
    ]

    print()
    print(
        f"Best walk-forward model : {best_model}"
    )

    print(
        f"Mean Test AUC           : {best_auc:.4f}"
    )

    print(
        "Mean Balanced Accuracy  : "
        f"{best_balanced:.4f}"
    )

    if best_auc >= 0.60:
        print(
            "SIGNAL: Potentially useful "
            "out-of-sample ranking signal."
        )
    elif best_auc >= 0.55:
        print(
            "WEAK POSITIVE: Some out-of-sample "
            "ranking signal may exist."
        )
    else:
        print(
            "WEAK: Walk-forward ranking signal "
            "is close to random."
        )

# --------------------------------------------------------------
# FINAL INTERPRETATION
# --------------------------------------------------------------

header(
    "V6 INTERPRETATION"
)

print(
    "This is a strict chronological "
    "walk-forward validation."
)

print(
    "Each fold trains only on dates before "
    "the corresponding test period."
)

print(
    "The SAFE feature list comes from the "
    "V6 feature stability audit."
)

print()
print(
    "IMPORTANT:"
)

print(
    "Walk-forward AUC is not profitability."
)

print(
    "Transaction costs, slippage, position "
    "sizing and trading rules still need "
    "separate evaluation."
)

print()
print(
    "OUTPUT FILES"
)

print("-" * 70)

print(
    f"Results     : {RESULT_FILE}"
)

print(
    f"Predictions : {PRED_FILE}"
)

print()
print(
    "PASS: V6 walk-forward validation completed."
)
```

# ======================================================================

# ENTRY POINT

# ======================================================================

if **name** == "**main**":
main()
