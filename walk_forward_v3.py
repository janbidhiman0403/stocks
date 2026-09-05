import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ============================================================
# ALPHALENS V3 — WALK-FORWARD VALIDATION
# ============================================================

INPUT = Path("data/ml_targets_v3.csv")
OUTPUT = Path("data/walk_forward_v3_results.csv")

TARGET = "Target_Up_V3"

print("=" * 70)
print("ALPHALENS V3 WALK-FORWARD VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\nRows:", len(df))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

exclude = {
    "Date",
    "Target_Return_1D",
    "Target_Return_5D",
    "Target_Return_10D",
    "Target_Volatility_20",
    "Target_Return_Vol_Adj",
    "Target_Direction_V3",
    "Target_Up_V3",
    "Target_Direction_5D_V3",
    "Target_Direction_10D_V3",
    "Target_Excess_Return",
    "Target_Direction_1D",
    "Target_Direction_5D",
}

features = [
    c for c in df.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(df[c])
]

print("\nFeatures:", len(features))

X = df[features].copy()
y = df[TARGET].astype(int)

# Safety
X = X.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# WALK-FORWARD SETTINGS
# ------------------------------------------------------------

MIN_TRAIN = 800
TEST_SIZE = 100
STEP = 100

results = []

print("\n" + "=" * 70)
print("WALK-FORWARD WINDOWS")
print("=" * 70)

window = 0
start_test = MIN_TRAIN

while start_test < len(df):

    end_test = min(start_test + TEST_SIZE, len(df))

    train_idx = np.arange(0, start_test)
    test_idx = np.arange(start_test, end_test)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # --------------------------------------------------------
    # Drop missing rows separately
    # --------------------------------------------------------

    train_valid = X_train.notna().all(axis=1)
    test_valid = X_test.notna().all(axis=1)

    X_train = X_train.loc[train_valid]
    y_train = y_train.loc[train_valid]

    X_test = X_test.loc[test_valid]
    y_test = y_test.loc[test_valid]

    if len(y_test) < 20:
        start_test += STEP
        continue

    if y_test.nunique() < 2:
        start_test += STEP
        continue

    print(
        f"\nWindow {window + 1}: "
        f"TRAIN={len(X_train)} "
        f"TEST={len(X_test)} "
        f"{df.iloc[start_test]['Date'].date()} "
        f"to {df.iloc[end_test - 1]['Date'].date()}"
    )

    # ========================================================
    # MODEL 1 — LOGISTIC REGRESSION
    # ========================================================

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1
            )
        )
    ])

    logistic.fit(X_train, y_train)

    p_log = logistic.predict_proba(X_test)[:, 1]
    pred_log = (p_log >= 0.50).astype(int)

    # ========================================================
    # MODEL 2 — XGBOOST
    # ========================================================

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8,
        reg_alpha=0.5,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    p_xgb = xgb.predict_proba(X_test)[:, 1]
    pred_xgb = (p_xgb >= 0.50).astype(int)

    # ========================================================
    # SAVE BOTH MODELS
    # ========================================================

    for model_name, probabilities, predictions in [
        ("Logistic", p_log, pred_log),
        ("XGBoost", p_xgb, pred_xgb)
    ]:

        result = pd.DataFrame({
            "Date": df.iloc[test_idx][test_valid]["Date"].values,
            "Actual": y_test.values,
            "Probability_Up": probabilities,
            "Prediction": predictions,
            "Model": model_name,
            "Window": window + 1
        })

        results.append(result)

        auc = roc_auc_score(y_test, probabilities)
        acc = accuracy_score(y_test, predictions)
        bal = balanced_accuracy_score(y_test, predictions)
        prec = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        rec = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            f"{model_name:10s} "
            f"AUC={auc:.4f} "
            f"BAL={bal:.4f} "
            f"F1={f1:.4f}"
        )

    window += 1
    start_test += STEP


# ============================================================
# COMBINE RESULTS
# ============================================================

if not results:
    raise RuntimeError("No walk-forward results generated.")

final = pd.concat(results, ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

final.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

for model in final["Model"].unique():

    d = final[final["Model"] == model]

    auc = roc_auc_score(
        d["Actual"],
        d["Probability_Up"]
    )

    pred = d["Prediction"]

    acc = accuracy_score(
        d["Actual"],
        pred
    )

    bal = balanced_accuracy_score(
        d["Actual"],
        pred
    )

    prec = precision_score(
        d["Actual"],
        pred,
        zero_division=0
    )

    rec = recall_score(
        d["Actual"],
        pred,
        zero_division=0
    )

    f1 = f1_score(
        d["Actual"],
        pred,
        zero_division=0
    )

    print("\n" + model)
    print("-" * 40)
    print("Samples             :", len(d))
    print("ROC-AUC             :", round(auc, 4))
    print("Accuracy            :", round(acc, 4))
    print("Balanced Accuracy   :", round(bal, 4))
    print("Precision           :", round(prec, 4))
    print("Recall              :", round(rec, 4))
    print("F1                  :", round(f1, 4))


print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(OUTPUT)

print("\nPASS: WALK-FORWARD VALIDATION COMPLETE")
print("=" * 70)