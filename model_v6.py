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
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FEATURE_FILE = os.path.join(DATA_DIR, "ml_features_v4.csv")
TARGET_FILE = os.path.join(DATA_DIR, "ml_targets_v3.csv")
SAFE_FILE = os.path.join(DATA_DIR, "feature_stability_v6_safe_features.csv")

RESULTS_FILE = os.path.join(DATA_DIR, "model_v6_results.csv")
PRED_FILE = os.path.join(DATA_DIR, "model_v6_predictions.csv")
FEATURES_USED_FILE = os.path.join(DATA_DIR, "model_v6_features_used.csv")


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def metric_row(y_true, prob, threshold=0.50):
    pred = (prob >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def load_safe_features(path):
    safe = pd.read_csv(path)

    # Accept the normal V6 audit format and a few reasonable variants.
    candidates = ["Feature", "feature", "Feature_Name", "feature_name"]
    col = next((c for c in candidates if c in safe.columns), None)
    if col is None:
        raise ValueError(
            f"Could not identify feature-name column in {path}. "
            f"Columns found: {list(safe.columns)}"
        )

    names = (
        safe[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("V6 SAFE feature file contains no features.")

    return names


def chronological_split(df, train_frac=0.60, val_frac=0.20):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def fit_models(X_train, y_train):
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            C=0.10,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=5,
            reg_lambda=5.0,
            reg_alpha=0.25,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    logistic.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {"Logistic": logistic, "XGBoost": xgb}


def main():
    header("ALPHALENS V6 CHRONOLOGICAL MODEL")

    print("Loading V6 inputs...")

    for path in [FEATURE_FILE, TARGET_FILE, SAFE_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run the V6 feature stability audit first."
            )

    features = pd.read_csv(FEATURE_FILE)
    targets = pd.read_csv(TARGET_FILE)

    print(f"Feature rows    : {len(features)}")
    print(f"Feature columns : {len(features.columns)}")
    print(f"Target rows     : {len(targets)}")
    print(f"Target columns  : {len(targets.columns)}")

    header("DATE ALIGNMENT")

    if "Date" not in features.columns or "Date" not in targets.columns:
        raise ValueError("Both input files must contain a Date column.")

    features["Date"] = pd.to_datetime(features["Date"])
    targets["Date"] = pd.to_datetime(targets["Date"])

    features = features.sort_values("Date").drop_duplicates("Date")
    targets = targets.sort_values("Date").drop_duplicates("Date")

    safe_features = load_safe_features(SAFE_FILE)

    target_candidates = [
        "Target_10D_1pct",
        "Target_Return_AtLeast_1pct_10D",
        "Target_Return_AtLeast_1.0pct_10D",
    ]

    target_col = next((c for c in target_candidates if c in targets.columns), None)

    # If the target is not already named as the experiment target,
    # derive it from Target_Return_10D.
    if target_col is None and "Target_Return_10D" in targets.columns:
        targets["Target_10D_1pct"] = (
            pd.to_numeric(targets["Target_Return_10D"], errors="coerce") >= 0.01
        ).astype(int)
        target_col = "Target_10D_1pct"

    if target_col is None:
        raise ValueError(
            "Could not find or derive the 10-day >=1% target. "
            f"Available target columns include: {list(targets.columns)}"
        )

    print(f"Feature date range : {features['Date'].min().date()} to {features['Date'].max().date()}")
    print(f"Target date range  : {targets['Date'].min().date()} to {targets['Date'].max().date()}")
    print(f"Target             : {target_col}")

    header("MERGE")

    df = features.merge(
        targets[["Date", target_col]],
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values("Date").reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df[target_col] = df[target_col].astype(int)

    print(f"Merged rows : {len(df)}")
    print("\nTarget distribution:")
    print(df[target_col].value_counts().sort_index())
    print("\nTarget rate:", round(df[target_col].mean(), 4))

    header("V6 SAFE FEATURE SET")

    usable = [f for f in safe_features if f in df.columns]

    missing_safe = [f for f in safe_features if f not in df.columns]

    print(f"SAFE features listed : {len(safe_features)}")
    print(f"SAFE features usable : {len(usable)}")

    if missing_safe:
        print("\nSAFE features not found in feature dataset:")
        for f in missing_safe:
            print(" -", f)

    if len(usable) < 5:
        raise ValueError(
            "Too few V6 SAFE features are present in ml_features_v4.csv."
        )

    X = df[usable].apply(pd.to_numeric, errors="coerce")
    y = df[target_col].astype(int)

    # Remove columns that are entirely missing after numeric conversion.
    all_missing = [c for c in X.columns if X[c].notna().sum() == 0]
    if all_missing:
        print("\nDropping entirely-missing features:")
        for f in all_missing:
            print(" -", f)
        X = X.drop(columns=all_missing)
        usable = list(X.columns)

    pd.DataFrame({"Feature": usable}).to_csv(FEATURES_USED_FILE, index=False)

    header("CHRONOLOGICAL SPLIT")

    work = pd.concat(
        [df[["Date", target_col]].reset_index(drop=True),
         X.reset_index(drop=True)],
        axis=1,
    )

    train, val, test = chronological_split(work)

    print(
        f"TRAIN      : {len(train)} "
        f"{train['Date'].min().date()} to {train['Date'].max().date()}"
    )
    print(
        f"VALIDATION : {len(val)} "
        f"{val['Date'].min().date()} to {val['Date'].max().date()}"
    )
    print(
        f"TEST       : {len(test)} "
        f"{test['Date'].min().date()} to {test['Date'].max().date()}"
    )

    X_train = train[usable]
    y_train = train[target_col]

    X_val = val[usable]
    y_val = val[target_col]

    X_test = test[usable]
    y_test = test[target_col]

    print("\nTarget rates:")
    print(f"Train : {y_train.mean():.3f}")
    print(f"Val   : {y_val.mean():.3f}")
    print(f"Test  : {y_test.mean():.3f}")

    header("TRAINING V6 MODELS")

    models = fit_models(X_train, y_train)

    results = []
    prediction_frames = []

    for name, model in models.items():
        print(f"\nTraining/evaluating {name}...")

        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        train_m = metric_row(y_train, train_prob)
        val_m = metric_row(y_val, val_prob)
        test_m = metric_row(y_test, test_prob)

        print(
            f"{name:<10} "
            f"Train AUC={train_m['ROC_AUC']:.4f} "
            f"Val AUC={val_m['ROC_AUC']:.4f} "
            f"Test AUC={test_m['ROC_AUC']:.4f} "
            f"Test BAL={test_m['Balanced_Accuracy']:.4f} "
            f"Test F1={test_m['F1']:.4f}"
        )

        results.append({
            "Model": name,
            "Features": len(usable),
            "Train_AUC": train_m["ROC_AUC"],
            "Validation_AUC": val_m["ROC_AUC"],
            "Test_AUC": test_m["ROC_AUC"],
            "Test_Accuracy": test_m["Accuracy"],
            "Test_Balanced_Accuracy": test_m["Balanced_Accuracy"],
            "Test_Precision": test_m["Precision"],
            "Test_Recall": test_m["Recall"],
            "Test_F1": test_m["F1"],
        })

        pred = pd.DataFrame({
            "Date": test["Date"].values,
            "Actual": y_test.values,
            "Probability_Up": test_prob,
            "Prediction_0.50": (test_prob >= 0.50).astype(int),
            "Model": name,
        })
        prediction_frames.append(pred)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    results_df.to_csv(RESULTS_FILE, index=False)
    predictions_df.to_csv(PRED_FILE, index=False)

    header("V6 MODEL RESULTS")

    display_cols = [
        "Model",
        "Features",
        "Train_AUC",
        "Validation_AUC",
        "Test_AUC",
        "Test_Balanced_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
    ]

    print(
        results_df[display_cols]
        .sort_values("Test_AUC", ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    best = results_df.sort_values("Test_AUC", ascending=False).iloc[0]

    header("V6 INTERPRETATION")

    print(f"Best model       : {best['Model']}")
    print(f"Test AUC         : {best['Test_AUC']:.4f}")
    print(f"Validation AUC   : {best['Validation_AUC']:.4f}")
    print(f"Test Balanced Acc: {best['Test_Balanced_Accuracy']:.4f}")

    if best["Test_AUC"] >= 0.60:
        print("\nPROMISING: Test ranking signal reaches the V6 review threshold.")
        print("Next stage: V6 walk-forward validation.")
    elif best["Test_AUC"] >= 0.55:
        print("\nWEAK POSITIVE: Some ranking signal exists.")
        print("Do NOT treat this as proof of profitability.")
        print("Next stage: V6 walk-forward validation.")
    else:
        print("\nWEAK/NEAR RANDOM: Test ranking signal is insufficient.")
        print("Do NOT proceed to trading deployment.")
        print("Review feature construction and target definition.")

    header("OUTPUT")

    print(f"Results        : {RESULTS_FILE}")
    print(f"Predictions    : {PRED_FILE}")
    print(f"Features used  : {FEATURES_USED_FILE}")

    header("V6 CHRONOLOGICAL MODEL COMPLETE")
    print("PASS: model_v6.py completed successfully.")


if __name__ == "__main__":
    main()
