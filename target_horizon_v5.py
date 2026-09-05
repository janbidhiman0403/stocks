import os
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# ALPHALENS V5 TARGET HORIZON EXPERIMENT
# ============================================================

FEATURE_FILE = "data/ml_features_v4.csv"
OUTPUT_FILE = "data/target_horizon_v5_results.csv"

HORIZONS = [1, 3, 5, 10]

# Binary direction:
#   1 = forward return > 0
#   0 = forward return <= 0
#
# Minimum-return targets:
#   1 = forward return >= threshold
#   0 = otherwise
RETURN_THRESHOLDS = [
    0.000,
    0.005,
    0.010,
]


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, probabilities):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return np.nan


def evaluate(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "AUC": safe_auc(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
    }


def build_logistic():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            ),
        ]
    )


def build_xgb():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=8,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ]
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ALPHALENS V5 TARGET HORIZON EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"\nCould not find:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print()
print("Loading V4 features...")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# DATE
# ============================================================

if "Date" not in df.columns:
    raise ValueError(
        "Date column not found in V4 feature dataset."
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# PRICE COLUMN
# ============================================================

if "Close" not in df.columns:
    raise ValueError(
        "Close column not found in V4 feature dataset."
    )


# ============================================================
# IDENTIFY FEATURES
# ============================================================

exclude_columns = {
    "Date",
}

feature_columns = [
    col
    for col in df.columns
    if col not in exclude_columns
]

# Keep numeric features only.
numeric_features = []

for col in feature_columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)

feature_columns = numeric_features

print()
print(f"Usable numeric features: {len(feature_columns)}")


# ============================================================
# REMOVE ANY OBVIOUS TARGET-LIKE COLUMNS
# ============================================================

target_like = []

for col in feature_columns:
    name = col.lower()

    if (
        "target" in name
        or "future" in name
        or "forward" in name
        or "next_day" in name
        or "y_true" in name
    ):
        target_like.append(col)

if target_like:
    print()
    print("Removing target-like columns:")
    for col in target_like:
        print(f" - {col}")

    feature_columns = [
        col
        for col in feature_columns
        if col not in target_like
    ]


# ============================================================
# IMPORTANT:
# USE CLOSE[-1] / CLOSE[T] FOR FORWARD RETURNS
# ============================================================

close = df["Close"].astype(float)

print()
print("Price range:")
print(
    f"Close: {close.min():.2f} -> {close.max():.2f}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
#
# 60% TRAIN
# 20% VALIDATION
# 20% TEST
# ============================================================

n = len(df)

train_end = int(n * 0.60)
validation_end = int(n * 0.80)

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"TRAIN      : {train_end}"
    f" {df.loc[0, 'Date'].date()}"
    f" to "
    f"{df.loc[train_end - 1, 'Date'].date()}"
)

print(
    f"VALIDATION : {validation_end - train_end}"
    f" {df.loc[train_end, 'Date'].date()}"
    f" to "
    f"{df.loc[validation_end - 1, 'Date'].date()}"
)

print(
    f"TEST       : {n - validation_end}"
    f" {df.loc[validation_end, 'Date'].date()}"
    f" to "
    f"{df.loc[n - 1, 'Date'].date()}"
)


# ============================================================
# EXPERIMENT
# ============================================================

results = []


for horizon in HORIZONS:

    print()
    print("=" * 70)
    print(f"HORIZON: {horizon} DAY(S)")
    print("=" * 70)

    # Forward return:
    #
    # close[t+h] / close[t] - 1
    #
    # This is future information and is used ONLY as target.
    forward_return = (
        close.shift(-horizon) / close - 1.0
    )

    for return_threshold in RETURN_THRESHOLDS:

        if return_threshold == 0:
            target_name = "Direction"
        else:
            target_name = (
                f"Return_AtLeast_{return_threshold * 100:.1f}pct"
            )

        print()
        print("-" * 60)
        print(
            f"Target: {target_name}"
        )
        print(
            f"Horizon: {horizon} day(s)"
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = (
            forward_return >= return_threshold
        ).astype(float)

        target[forward_return.isna()] = np.nan

        valid = (
            target.notna()
            & df["Date"].notna()
        )

        experiment = df.loc[
            valid,
            ["Date"] + feature_columns
        ].copy()

        experiment["Target"] = target.loc[
            valid
        ].astype(int)

        experiment["Forward_Return"] = (
            forward_return.loc[valid]
        )

        # ----------------------------------------------------
        # SPLIT USING ORIGINAL ROW ORDER
        # ----------------------------------------------------

        train_mask = (
            experiment.index < train_end
        )

        val_mask = (
            (experiment.index >= train_end)
            & (experiment.index < validation_end)
        )

        test_mask = (
            experiment.index >= validation_end
        )

        X_train = experiment.loc[
            train_mask,
            feature_columns
        ]

        y_train = experiment.loc[
            train_mask,
            "Target"
        ]

        X_val = experiment.loc[
            val_mask,
            feature_columns
        ]

        y_val = experiment.loc[
            val_mask,
            "Target"
        ]

        X_test = experiment.loc[
            test_mask,
            feature_columns
        ]

        y_test = experiment.loc[
            test_mask,
            "Target"
        ]

        # Remove any rows where target was unavailable.
        train_valid = y_train.notna()
        val_valid = y_val.notna()
        test_valid = y_test.notna()

        X_train = X_train.loc[train_valid]
        y_train = y_train.loc[train_valid].astype(int)

        X_val = X_val.loc[val_valid]
        y_val = y_val.loc[val_valid].astype(int)

        X_test = X_test.loc[test_valid]
        y_test = y_test.loc[test_valid].astype(int)

        # ----------------------------------------------------
        # TARGET DISTRIBUTION
        # ----------------------------------------------------

        train_rate = y_train.mean()
        val_rate = y_val.mean()
        test_rate = y_test.mean()

        print()
        print(
            f"Train UP rate : {train_rate:.3f}"
        )
        print(
            f"Val UP rate   : {val_rate:.3f}"
        )
        print(
            f"Test UP rate  : {test_rate:.3f}"
        )

        # ----------------------------------------------------
        # SKIP EXTREME / INVALID TARGETS
        # ----------------------------------------------------

        if len(np.unique(y_train)) < 2:
            print(
                "SKIP: Training target contains only one class."
            )
            continue

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        models = {
            "Logistic": build_logistic(),
            "XGBoost": build_xgb(),
        }

        for model_name, model in models.items():

            print()
            print(
                f"Training {model_name}..."
            )

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            train_prob = model.predict_proba(
                X_train
            )[:, 1]

            val_prob = model.predict_proba(
                X_val
            )[:, 1]

            test_prob = model.predict_proba(
                X_test
            )[:, 1]

            train_metrics = evaluate(
                y_train,
                train_prob
            )

            val_metrics = evaluate(
                y_val,
                val_prob
            )

            test_metrics = evaluate(
                y_test,
                test_prob
            )

            print(
                f"{model_name:<10}"
                f" Train AUC={train_metrics['AUC']:.4f}"
                f" Val AUC={val_metrics['AUC']:.4f}"
                f" Test AUC={test_metrics['AUC']:.4f}"
                f" Test BAL={test_metrics['Balanced_Accuracy']:.4f}"
                f" Test F1={test_metrics['F1']:.4f}"
            )

            results.append(
                {
                    "Horizon": horizon,
                    "Target": target_name,
                    "Return_Threshold": return_threshold,
                    "Model": model_name,

                    "Train_Samples": len(y_train),
                    "Validation_Samples": len(y_val),
                    "Test_Samples": len(y_test),

                    "Train_UP_Rate": train_rate,
                    "Validation_UP_Rate": val_rate,
                    "Test_UP_Rate": test_rate,

                    "Train_AUC": train_metrics["AUC"],
                    "Validation_AUC": val_metrics["AUC"],
                    "Test_AUC": test_metrics["AUC"],

                    "Train_Balanced_Accuracy":
                        train_metrics["Balanced_Accuracy"],

                    "Validation_Balanced_Accuracy":
                        val_metrics["Balanced_Accuracy"],

                    "Test_Balanced_Accuracy":
                        test_metrics["Balanced_Accuracy"],

                    "Train_Accuracy":
                        train_metrics["Accuracy"],

                    "Validation_Accuracy":
                        val_metrics["Accuracy"],

                    "Test_Accuracy":
                        test_metrics["Accuracy"],

                    "Test_Precision":
                        test_metrics["Precision"],

                    "Test_Recall":
                        test_metrics["Recall"],

                    "Test_F1":
                        test_metrics["F1"],
                }
            )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

if len(results_df) == 0:
    raise RuntimeError(
        "No experiments completed."
    )


# ============================================================
# SORT BY TEST AUC
# ============================================================

results_sorted = results_df.sort_values(
    "Test_AUC",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("V5 TARGET HORIZON RESULTS")
print("=" * 70)

display_columns = [
    "Horizon",
    "Target",
    "Model",
    "Train_AUC",
    "Validation_AUC",
    "Test_AUC",
    "Test_Balanced_Accuracy",
    "Test_F1",
]

print(
    results_sorted[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST TEST AUC
# ============================================================

best = results_sorted.iloc[0]

print()
print("=" * 70)
print("BEST OUT-OF-SAMPLE CONFIGURATION")
print("=" * 70)

print(
    f"Horizon              : "
    f"{int(best['Horizon'])} days"
)

print(
    f"Target               : "
    f"{best['Target']}"
)

print(
    f"Model                : "
    f"{best['Model']}"
)

print(
    f"Train AUC            : "
    f"{best['Train_AUC']:.4f}"
)

print(
    f"Validation AUC       : "
    f"{best['Validation_AUC']:.4f}"
)

print(
    f"Test AUC             : "
    f"{best['Test_AUC']:.4f}"
)

print(
    f"Test Balanced Acc.   : "
    f"{best['Test_Balanced_Accuracy']:.4f}"
)

print(
    f"Test F1              : "
    f"{best['Test_F1']:.4f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

best_auc = float(best["Test_AUC"])

if best_auc >= 0.55:
    print(
        "PROMISING: At least one target horizon shows "
        "meaningful out-of-sample ranking signal."
    )
elif best_auc >= 0.52:
    print(
        "WEAK SIGNAL: The best horizon is above random, "
        "but the signal is not yet strong."
    )
elif best_auc >= 0.48:
    print(
        "NO ROBUST SIGNAL: Results remain close to random."
    )
else:
    print(
        "NEGATIVE SIGNAL: The best target formulation "
        "is below random on the test period."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("V5 TARGET HORIZON EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"Output: {OUTPUT_FILE}"
)

print()
print(
    "PASS: Target horizon experiment completed."
)
print("=" * 70)