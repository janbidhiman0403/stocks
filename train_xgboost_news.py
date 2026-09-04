import pandas as pd
import numpy as np
import os
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/train_TCS_news.csv"
VALIDATION_FILE = "data/validation_TCS_news.csv"
TEST_FILE = "data/test_TCS_news.csv"

MODEL_FILE = "data/xgboost_news_model.pkl"
PREDICTIONS_FILE = "data/xgboost_news_predictions.csv"

TARGET = "Target_Direction_1D"

# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("ALPHALENS XGBOOST NEWS + TECHNICAL MODEL")
print("=" * 60)

# ============================================================
# CHECK FILES
# ============================================================

for file in [TRAIN_FILE, VALIDATION_FILE, TEST_FILE]:
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("Train      :", len(train))
print("Validation :", len(validation))
print("Test       :", len(test))

# ============================================================
# DATE HANDLING
# ============================================================

for df in [train, validation, test]:
    df["Date"] = pd.to_datetime(df["Date"])

# ============================================================
# IDENTIFY FEATURES
# ============================================================

# Columns that must never be used as model inputs
EXCLUDE_COLUMNS = [
    "Date",

    # Future targets
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Direction_1D",
    "Target_Direction_5D",

    # Any other possible leakage columns
    "Future_Return",
    "Future_Close",
    "Next_Close"
]

feature_columns = [
    col
    for col in train.columns
    if col not in EXCLUDE_COLUMNS
]

# Keep only numeric features
numeric_features = []

for col in feature_columns:
    if pd.api.types.is_numeric_dtype(train[col]):
        numeric_features.append(col)

feature_columns = numeric_features

print()
print("Features selected:", len(feature_columns))

print()
print("Feature list:")
for i, col in enumerate(feature_columns, 1):
    print(f"{i:02d}. {col}")

# ============================================================
# CHECK FEATURE CONSISTENCY
# ============================================================

missing_validation_features = [
    col for col in feature_columns
    if col not in validation.columns
]

missing_test_features = [
    col for col in feature_columns
    if col not in test.columns
]

if missing_validation_features:
    raise ValueError(
        "Validation dataset missing features: "
        + str(missing_validation_features)
    )

if missing_test_features:
    raise ValueError(
        "Test dataset missing features: "
        + str(missing_test_features)
    )

# ============================================================
# CREATE X / Y
# ============================================================

X_train = train[feature_columns].copy()
y_train = train[TARGET].astype(int)

X_validation = validation[feature_columns].copy()
y_validation = validation[TARGET].astype(int)

X_test = test[feature_columns].copy()
y_test = test[TARGET].astype(int)

# ============================================================
# SAFETY CHECK
# ============================================================

print()
print("Checking feature values...")

for name, X in [
    ("Train", X_train),
    ("Validation", X_validation),
    ("Test", X_test)
]:

    if X.isna().sum().sum() > 0:
        raise ValueError(
            f"{name} contains missing feature values."
        )

    if np.isinf(X.values).any():
        raise ValueError(
            f"{name} contains infinite feature values."
        )

print("Feature values: PASS")

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print()
print("TRAIN:")
print(y_train.value_counts().sort_index())

print()
print("VALIDATION:")
print(y_validation.value_counts().sort_index())

print()
print("TEST:")
print(y_test.value_counts().sort_index())

# ============================================================
# NEWS FEATURE CHECK
# ============================================================

news_features = [
    col for col in feature_columns
    if "News" in col or "news" in col
]

print()
print("=" * 60)
print("NEWS FEATURES")
print("=" * 60)

print("News features used:", len(news_features))

for col in news_features:
    print(" -", col)

# ============================================================
# MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print()
print("Training rows:", len(X_train))
print("Features     :", len(feature_columns))

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation)
    ],
    verbose=False
)

print()
print("Training complete.")

# ============================================================
# PREDICTIONS
# ============================================================

print()
print("Generating predictions...")

train_probability = model.predict_proba(X_train)[:, 1]
validation_probability = model.predict_proba(X_validation)[:, 1]
test_probability = model.predict_proba(X_test)[:, 1]

train_prediction = (train_probability >= 0.5).astype(int)
validation_prediction = (validation_probability >= 0.5).astype(int)
test_prediction = (test_probability >= 0.5).astype(int)

# ============================================================
# METRICS FUNCTION
# ============================================================

def evaluate_model(name, y_true, prediction, probability):

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probability
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            prediction
        )
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            prediction,
            digits=4,
            zero_division=0
        )
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    }

# ============================================================
# EVALUATE
# ============================================================

train_metrics = evaluate_model(
    "TRAIN RESULTS",
    y_train,
    train_prediction,
    train_probability
)

validation_metrics = evaluate_model(
    "VALIDATION RESULTS",
    y_validation,
    validation_prediction,
    validation_probability
)

test_metrics = evaluate_model(
    "FINAL TEST RESULTS",
    y_test,
    test_prediction,
    test_probability
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.head(25).to_string(
        index=False
    )
)

# ============================================================
# CREATE PREDICTION FILE
# ============================================================

predictions = test[
    [
        "Date",
        "Close",
        "News_Sentiment",
        "News_Count",
        "Has_News"
    ]
].copy()

predictions["Actual"] = y_test.values
predictions["Prediction"] = test_prediction
predictions["Probability_Up"] = test_probability
predictions["Probability_Down"] = 1 - test_probability

predictions["Correct"] = (
    predictions["Actual"]
    == predictions["Prediction"]
)

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    MODEL_FILE
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print()
print("Model:")
print(MODEL_FILE)

print()
print("Predictions:")
print(PREDICTIONS_FILE)

print()
print("FINAL TEST PERFORMANCE")
print("-" * 60)

print(
    f"Accuracy : {test_metrics['Accuracy']:.4f}"
)

print(
    f"Precision: {test_metrics['Precision']:.4f}"
)

print(
    f"Recall   : {test_metrics['Recall']:.4f}"
)

print(
    f"F1 Score : {test_metrics['F1']:.4f}"
)

print(
    f"ROC AUC  : {test_metrics['ROC_AUC']:.4f}"
)

print()
print("=" * 60)
print("PASS: NEWS + TECHNICAL XGBOOST MODEL READY")
print("=" * 60)