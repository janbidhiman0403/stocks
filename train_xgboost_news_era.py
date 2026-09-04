import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = Path("data/train_TCS_news_era.csv")
VAL_FILE = Path("data/validation_TCS_news_era.csv")
TEST_FILE = Path("data/test_TCS_news_era.csv")

MODEL_FILE = Path("data/xgboost_news_era_model.pkl")
PRED_FILE = Path("data/xgboost_news_era_predictions.csv")

TARGET = "Target_Direction_1D"

# ============================================================
# FEATURE SELECTION
# ============================================================

EXCLUDE_COLUMNS = {
    "Date",
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Direction_1D",
    "Target_Direction_5D",
}

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ALPHALENS NEWS-ERA XGBOOST MODEL")
print("=" * 60)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print("Train      :", len(train))
print("Validation :", len(validation))
print("Test       :", len(test))

# ============================================================
# PREPARE FEATURES
# ============================================================

feature_columns = [
    c for c in train.columns
    if c not in EXCLUDE_COLUMNS
]

# Keep only numeric features
feature_columns = [
    c for c in feature_columns
    if pd.api.types.is_numeric_dtype(train[c])
]

print("\nFeatures selected:", len(feature_columns))

print("\nFeature list:")

for i, feature in enumerate(feature_columns, 1):
    print(f"{i:02d}. {feature}")

# ============================================================
# VALIDATE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("FEATURE VALIDATION")
print("=" * 60)

for name, df in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    missing = [c for c in feature_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing features: {missing}"
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            f"{name} contains missing feature values."
        )

    if not np.isfinite(
        df[feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{name} contains infinite feature values."
        )

print("Feature values: PASS")

# ============================================================
# X / Y
# ============================================================

X_train = train[feature_columns]
y_train = train[TARGET]

X_val = validation[feature_columns]
y_val = validation[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

for name, y in [
    ("TRAIN", y_train),
    ("VALIDATION", y_val),
    ("TEST", y_test),
]:
    print(f"\n{name}:")
    print(y.value_counts().sort_index())

# ============================================================
# CLASS BALANCE
# ============================================================

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print("\nTraining class balance:")
print("Down:", negative)
print("Up  :", positive)
print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4)
)

# ============================================================
# NEWS FEATURES
# ============================================================

news_features = [
    c for c in feature_columns
    if c.startswith("News_") or c == "Has_News"
]

print("\n" + "=" * 60)
print("NEWS FEATURES")
print("=" * 60)

print("News features:", len(news_features))

for feature in news_features:
    print(" -", feature)

# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.80,
    colsample_bytree=0.80,
    gamma=0.10,
    reg_alpha=0.10,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

print("\nTraining rows:", len(X_train))
print("Features     :", len(feature_columns))

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    verbose=False,
)

print("\nTraining complete.")

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X, y):

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y, predictions, zero_division=0
    )
    recall = recall_score(
        y, predictions, zero_division=0
    )
    f1 = f1_score(
        y, predictions, zero_division=0
    )
    auc = roc_auc_score(y, probabilities)

    cm = confusion_matrix(y, predictions)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return probabilities, predictions

# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

train_prob, train_pred = evaluate_model(
    "TRAIN RESULTS",
    model,
    X_train,
    y_train
)

val_prob, val_pred = evaluate_model(
    "VALIDATION RESULTS",
    model,
    X_val,
    y_val
)

test_prob, test_pred = evaluate_model(
    "FINAL TEST RESULTS",
    model,
    X_test,
    y_test
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
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
    importance.head(25).to_string(index=False)
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = test[
    [
        "Date",
        "Close",
        "News_Sentiment",
        "News_Count",
        "Has_News",
        TARGET,
    ]
].copy()

predictions["Actual"] = predictions[TARGET]
predictions["Prediction"] = test_pred
predictions["Probability_Up"] = test_prob
predictions["Probability_Down"] = 1 - test_prob
predictions["Correct"] = (
    predictions["Actual"]
    == predictions["Prediction"]
)

predictions.drop(
    columns=[TARGET],
    inplace=True
)

predictions.to_csv(
    PRED_FILE,
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

model_package = {
    "model": model,
    "features": feature_columns,
    "target": TARGET,
    "threshold": 0.50,
    "training_period": (
        train["Date"].min(),
        train["Date"].max()
    ),
    "validation_period": (
        validation["Date"].min(),
        validation["Date"].max()
    ),
    "test_period": (
        test["Date"].min(),
        test["Date"].max()
    ),
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model_package, f)

# ============================================================
# FINAL SUMMARY
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_prob
)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_FILE)

print("\nPredictions:")
print(PRED_FILE)

print("\nFINAL TEST PERFORMANCE")
print("-" * 60)
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC AUC  : {test_auc:.4f}")

print("\n" + "=" * 60)
print("PASS: NEWS-ERA XGBOOST MODEL READY")
print("=" * 60)