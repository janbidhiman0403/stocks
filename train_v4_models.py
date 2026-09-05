
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("=" * 70)
print("ALPHALENS V4 MODEL BENCHMARK")
print("=" * 70)

FEATURE_FILE = "data/ml_features_v4.csv"
TARGET_FILE = "data/ml_targets_v3.csv"

OUTPUT_RESULTS = "data/v4_model_results.csv"
OUTPUT_PREDICTIONS = "data/v4_model_predictions.csv"

TARGET = "Target_Up_V3"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading V4 features...")
features = pd.read_csv(FEATURE_FILE)

print("Loading V3 targets...")
targets = pd.read_csv(TARGET_FILE)

features["Date"] = pd.to_datetime(features["Date"])
targets["Date"] = pd.to_datetime(targets["Date"])

features = features.sort_values("Date").reset_index(drop=True)
targets = targets.sort_values("Date").reset_index(drop=True)

print(f"V4 feature rows : {len(features)}")
print(f"Target rows     : {len(targets)}")

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

df = features.merge(
    targets[["Date", TARGET]],
    on="Date",
    how="inner"
)

df = df.sort_values("Date").reset_index(drop=True)

print(f"Merged rows     : {len(df)}")
print(f"Date            : {df['Date'].min().date()} to {df['Date'].max().date()}")

if TARGET not in df.columns:
    raise ValueError(f"Target {TARGET} not found after merge.")

# ------------------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------------------

exclude = {
    "Date",
    TARGET,
}

feature_columns = [
    c for c in df.columns
    if c not in exclude
]

# Keep only numeric features
numeric_features = []

for c in feature_columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_features.append(c)

feature_columns = numeric_features

print("\nFeatures:", len(feature_columns))

# ------------------------------------------------------------
# REMOVE INVALID VALUES
# ------------------------------------------------------------

X = df[feature_columns].copy()
y = df[TARGET].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain all NaN
X = X.dropna(axis=1, how="all")

# Forward-fill only historical feature values
X = X.ffill()

# Any remaining NaNs are filled with training-independent zero
X = X.fillna(0)

feature_columns = X.columns.tolist()

print("Usable features:", len(feature_columns))

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts().sort_index())

print("\nPercentages:")
print((y.value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------------------

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_idx = np.arange(0, train_end)
val_idx = np.arange(train_end, val_end)
test_idx = np.arange(val_end, n)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]
y_test = y.iloc[test_idx]

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"TRAIN:      {len(train_idx)} "
    f"{df.iloc[train_idx]['Date'].min().date()} "
    f"to {df.iloc[train_idx]['Date'].max().date()}"
)

print(
    f"VALIDATION: {len(val_idx)} "
    f"{df.iloc[val_idx]['Date'].min().date()} "
    f"to {df.iloc[val_idx]['Date'].max().date()}"
)

print(
    f"TEST:       {len(test_idx)} "
    f"{df.iloc[test_idx]['Date'].min().date()} "
    f"to {df.iloc[test_idx]['Date'].max().date()}"
)

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

models = {
    "Logistic": Pipeline([
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
        )
    ]),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def evaluate_model(name, model, X_data, y_data, split_name):

    probability = model.predict_proba(X_data)[:, 1]

    prediction = (probability >= 0.50).astype(int)

    result = {
        "Model": name,
        "Split": split_name,
        "Accuracy": accuracy_score(y_data, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_data,
            prediction
        ),
        "Precision": precision_score(
            y_data,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            y_data,
            prediction,
            zero_division=0
        ),
        "F1": f1_score(
            y_data,
            prediction,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_data,
            probability
        )
    }

    print(f"\n{split_name}")
    print(f"\n{name}")
    print(f"Accuracy          : {result['Accuracy']:.4f}")
    print(
        f"Balanced Accuracy : "
        f"{result['Balanced_Accuracy']:.4f}"
    )
    print(f"Precision         : {result['Precision']:.4f}")
    print(f"Recall            : {result['Recall']:.4f}")
    print(f"F1                : {result['F1']:.4f}")
    print(f"ROC AUC           : {result['ROC_AUC']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_data, prediction))

    return result, probability, prediction


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

all_results = []
prediction_rows = []

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"TRAINING: {name}")
    print("-" * 60)

    model.fit(X_train, y_train)

    # TRAIN
    train_result, _, _ = evaluate_model(
        name,
        model,
        X_train,
        y_train,
        "TRAIN"
    )

    all_results.append(train_result)

    # VALIDATION
    val_result, _, _ = evaluate_model(
        name,
        model,
        X_val,
        y_val,
        "VALIDATION"
    )

    all_results.append(val_result)

    # TEST
    test_result, test_prob, test_pred = evaluate_model(
        name,
        model,
        X_test,
        y_test,
        "TEST"
    )

    all_results.append(test_result)

    # Save test predictions
    test_dates = df.iloc[test_idx]["Date"].values

    for date, actual, prob, pred in zip(
        test_dates,
        y_test.values,
        test_prob,
        test_pred
    ):
        prediction_rows.append({
            "Date": date,
            "Actual": int(actual),
            "Prediction": int(pred),
            "Probability_Up": float(prob),
            "Model": name
        })

# ------------------------------------------------------------
# RESULTS TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

os.makedirs("data", exist_ok=True)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

predictions_df = pd.DataFrame(prediction_rows)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

# ------------------------------------------------------------
# FINAL COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL V4 MODEL COMPARISON")
print("=" * 70)

comparison = results_df[
    results_df["Split"] == "TEST"
].copy()

comparison = comparison.sort_values(
    "ROC_AUC",
    ascending=False
)

print(
    comparison[
        [
            "Model",
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

best = comparison.iloc[0]

print("\n" + "=" * 70)
print("BEST V4 MODEL")
print("=" * 70)

print(f"Model               : {best['Model']}")
print(f"Test Accuracy       : {best['Accuracy']:.4f}")
print(
    f"Test Balanced Acc.  : "
    f"{best['Balanced_Accuracy']:.4f}"
)
print(f"Test Precision      : {best['Precision']:.4f}")
print(f"Test Recall         : {best['Recall']:.4f}")
print(f"Test F1             : {best['F1']:.4f}")
print(f"Test ROC-AUC        : {best['ROC_AUC']:.4f}")

print("\nOutput files:")
print(OUTPUT_RESULTS)
print(OUTPUT_PREDICTIONS)

print("\n" + "=" * 70)
print("V4 MODEL BENCHMARK COMPLETE")
print("=" * 70)

print("PASS: V4 models trained and evaluated.")
