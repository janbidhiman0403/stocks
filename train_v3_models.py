import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    )


# ============================================================
# ALPHALENS V3 MODEL BENCHMARK
# ============================================================

DATA_PATH = Path("data/ml_targets_v3.csv")

OUTPUT_RESULTS = Path(
    "data/v3_model_results.csv"
)

OUTPUT_PREDICTIONS = Path(
    "data/v3_model_predictions.csv"
)


print("=" * 70)
print("ALPHALENS V3 MODEL BENCHMARK")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("Rows:", len(df))
print(
    "Date:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


# ============================================================
# REMOVE OLD TARGET COLUMNS
# ============================================================

TARGET_PREFIX = "Target_"

target_columns = [
    c for c in df.columns
    if c.startswith(TARGET_PREFIX)
]

print("\nTarget columns detected:")

for c in target_columns:
    print(" -", c)


# ============================================================
# PRIMARY TARGET
# ============================================================

TARGET = "Target_Up_V3"

if TARGET not in df.columns:
    raise ValueError(
        f"Missing target: {TARGET}"
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

# Never allow target-derived columns into X.
feature_columns = [
    c for c in df.columns
    if c != "Date"
    and not c.startswith("Target_")
]


print("\nFeatures:", len(feature_columns))

print("\nFeature list:")

for i, c in enumerate(feature_columns, 1):
    print(f"{i:02d}. {c}")


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

df = df.dropna(
    subset=[TARGET]
).reset_index(drop=True)


X = df[feature_columns].copy()

y = df[TARGET].astype(int)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

non_numeric = [
    c for c in X.columns
    if c not in numeric_features
]

if non_numeric:

    print(
        "WARNING: non-numeric features detected:"
    )

    for c in non_numeric:
        print(" -", c)

else:

    print(
        "All features numeric: PASS"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    y.value_counts()
    .sort_index()
)

print("\nPercentages:")

print(
    (y.value_counts(normalize=True)
       .sort_index() * 100)
    .round(2)
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
validation_end = int(n * 0.80)

train_idx = slice(
    0,
    train_end
)

validation_idx = slice(
    train_end,
    validation_end
)

test_idx = slice(
    validation_end,
    n
)


X_train = X.iloc[train_idx]
X_val = X.iloc[validation_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[validation_idx]
y_test = y.iloc[test_idx]


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    "TRAIN:",
    len(X_train),
    df["Date"].iloc[0].date(),
    "to",
    df["Date"].iloc[train_end - 1].date()
)

print(
    "VALIDATION:",
    len(X_val),
    df["Date"].iloc[train_end].date(),
    "to",
    df["Date"].iloc[validation_end - 1].date()
)

print(
    "TEST:",
    len(X_test),
    df["Date"].iloc[validation_end].date(),
    "to",
    df["Date"].iloc[-1].date()
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic":

        Pipeline([
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
                    C=0.1,
                    class_weight="balanced"
                )
            )
        ]),


    "RandomForest":

        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=5,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),


    "XGBoost":

        Pipeline([
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
                    min_child_weight=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    name
):

    predictions = model.predict(X_data)

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    accuracy = accuracy_score(
        y_data,
        predictions
    )

    balanced = balanced_accuracy_score(
        y_data,
        predictions
    )

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_data,
            probabilities
        )

    except ValueError:

        auc = np.nan


    print(f"\n{name}")

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced:.4f}"
    )

    print(
        f"Precision         : {precision:.4f}"
    )

    print(
        f"Recall            : {recall:.4f}"
    )

    print(
        f"F1                : {f1:.4f}"
    )

    print(
        f"ROC AUC           : {auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_data,
            predictions
        )
    )

    return {

        "Model": name,

        "Accuracy": accuracy,

        "Balanced_Accuracy": balanced,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": auc
    }, predictions, probabilities


# ============================================================
# TRAINING
# ============================================================

results = []

prediction_rows = []


print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for name, model in models.items():

    print("\n" + "-" * 60)

    print(
        "TRAINING:",
        name
    )

    print("-" * 60)

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTRAIN")

    train_result, _, _ = evaluate_model(
        model,
        X_train,
        y_train,
        name
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nVALIDATION")

    val_result, _, _ = evaluate_model(
        model,
        X_val,
        y_val,
        name
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nTEST")

    test_result, predictions, probabilities = evaluate_model(
        model,
        X_test,
        y_test,
        name
    )


    results.append({

        "Model": name,

        "Train_Accuracy":
            train_result["Accuracy"],

        "Validation_Accuracy":
            val_result["Accuracy"],

        "Validation_Balanced_Accuracy":
            val_result["Balanced_Accuracy"],

        "Validation_F1":
            val_result["F1"],

        "Validation_ROC_AUC":
            val_result["ROC_AUC"],

        "Test_Accuracy":
            test_result["Accuracy"],

        "Test_Balanced_Accuracy":
            test_result["Balanced_Accuracy"],

        "Test_Precision":
            test_result["Precision"],

        "Test_Recall":
            test_result["Recall"],

        "Test_F1":
            test_result["F1"],

        "Test_ROC_AUC":
            test_result["ROC_AUC"]
    })


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    temp = pd.DataFrame({

        "Date":
            df["Date"].iloc[
                validation_end:
            ].values,

        "Actual":
            y_test.values,

        "Prediction":
            predictions,

        "Probability_Up":
            probabilities,

        "Model":
            name
    })

    prediction_rows.append(temp)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True
)


# ============================================================
# SORT MODELS
# ============================================================

results_df = results_df.sort_values(
    "Test_ROC_AUC",
    ascending=False
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_RESULTS.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("FINAL V3 MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

best = results_df.iloc[0]

print(
    "Best by Test ROC-AUC:",
    best["Model"]
)

print(
    f"Test Accuracy       : "
    f"{best['Test_Accuracy']:.4f}"
)

print(
    f"Test Balanced Acc.  : "
    f"{best['Test_Balanced_Accuracy']:.4f}"
)

print(
    f"Test F1             : "
    f"{best['Test_F1']:.4f}"
)

print(
    f"Test ROC-AUC        : "
    f"{best['Test_ROC_AUC']:.4f}"
)


print("\nOutput files:")

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_PREDICTIONS
)


print("\n" + "=" * 70)
print("PASS: V3 MODEL BENCHMARK COMPLETE")
print("=" * 70)