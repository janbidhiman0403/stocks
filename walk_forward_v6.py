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

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "ml_features_v4.csv",
)

TARGET_FILE = os.path.join(
    DATA_DIR,
    "ml_targets_v3.csv",
)

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


DATE_COL = "Date"


# Target priority:
#
# 1. Target_10D_1pct
# 2. Target_Direction_10D_V3
# 3. Target_Return_10D converted to binary
#
TARGET_PRIORITY = [
    "Target_10D_1pct",
    "Target_Direction_10D_V3",
]


# Walk-forward parameters
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


def print_dataset_info(df, name):
    print()
    print(f"{name} rows    : {len(df)}")
    print(f"{name} columns : {len(df.columns)}")


# ======================================================================
# METRICS
# ======================================================================

def calculate_scores(y_true, probabilities):
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    unique_values = np.unique(y_true)

    if len(unique_values) > 1:
        auc = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        auc = np.nan

    return {
        "AUC": auc,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Predicted_Positive_Rate": predictions.mean(),
    }


# ======================================================================
# MODEL DEFINITIONS
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
                    max_iter=2000,
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
        reg_lambda=1.00,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ======================================================================
# LOAD SAFE FEATURES
# ======================================================================

def load_safe_features():

    if not os.path.exists(SAFE_FILE):
        raise FileNotFoundError(
            f"SAFE feature file not found:\n{SAFE_FILE}"
        )

    safe_df = pd.read_csv(SAFE_FILE)

    print()
    print(f"SAFE file columns: {list(safe_df.columns)}")

    possible_columns = [
        "Feature",
        "feature",
        "Feature_Name",
        "feature_name",
        "Column",
        "column",
        "Name",
        "name",
    ]

    feature_column = None

    for column in possible_columns:
        if column in safe_df.columns:
            feature_column = column
            break

    if feature_column is None:

        # If the CSV has exactly one column, use it.
        if len(safe_df.columns) == 1:
            feature_column = safe_df.columns[0]

        else:
            raise ValueError(
                "Could not identify the feature-name column in:\n"
                f"{SAFE_FILE}\n\n"
                f"Columns found: {list(safe_df.columns)}"
            )

    features = (
        safe_df[feature_column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Remove duplicates while preserving order.
    features = list(
        dict.fromkeys(features)
    )

    print()
    print(f"SAFE features loaded : {len(features)}")

    return features


# ======================================================================
# TARGET SELECTION
# ======================================================================

def select_target(target_df):

    # --------------------------------------------------------------
    # First try the preferred binary targets.
    # --------------------------------------------------------------

    for target_name in TARGET_PRIORITY:

        if target_name in target_df.columns:

            series = target_df[target_name]

            # Convert numeric values if possible.
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            unique = set(
                numeric.dropna().unique()
            )

            # Accept a genuine binary target.
            if unique.issubset({0, 1}) and len(unique) == 2:

                print()
                print(
                    f"Target selected : {target_name}"
                )
                print(
                    "Target type     : existing binary target"
                )

                return target_df, target_name

    # --------------------------------------------------------------
    # Fallback: derive binary target from Target_Return_10D.
    #
    # Positive 10-day return = 1
    # Non-positive 10-day return = 0
    # --------------------------------------------------------------

    return_col = "Target_Return_10D"

    if return_col in target_df.columns:

        print()
        print(
            f"{TARGET_PRIORITY[0]} not found as usable binary target."
        )

        print(
            "Creating binary target from Target_Return_10D."
        )

        target_df = target_df.copy()

        target_df["Target_WF_10D"] = (
            pd.to_numeric(
                target_df[return_col],
                errors="coerce",
            ) > 0
        ).astype(float)

        # Preserve missing values.
        raw_return = pd.to_numeric(
            target_df[return_col],
            errors="coerce",
        )

        target_df.loc[
            raw_return.isna(),
            "Target_WF_10D",
        ] = np.nan

        print()
        print(
            "Target selected : Target_WF_10D"
        )
        print(
            "Target source   : Target_Return_10D"
        )
        print(
            "Definition      : 10D return > 0"
        )

        return target_df, "Target_WF_10D"

    # --------------------------------------------------------------
    # Nothing usable.
    # --------------------------------------------------------------

    raise ValueError(
        "No usable target found.\n\n"
        "Expected one of:\n"
        "  Target_10D_1pct\n"
        "  Target_Direction_10D_V3\n"
        "  Target_Return_10D\n\n"
        f"Available target columns:\n{list(target_df.columns)}"
    )


# ======================================================================
# PREPARE DATASET
# ======================================================================

def prepare_dataset():

    header(
        "ALPHALENS V6 WALK-FORWARD VALIDATION"
    )

    # --------------------------------------------------------------
    # Check files.
    # --------------------------------------------------------------

    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(
            f"Target file not found:\n{TARGET_FILE}"
        )

    # --------------------------------------------------------------
    # Load data.
    # --------------------------------------------------------------

    print()
    print("Loading V6 inputs...")

    features_df = pd.read_csv(
        FEATURE_FILE
    )

    target_df = pd.read_csv(
        TARGET_FILE
    )

    print_dataset_info(
        features_df,
        "Feature",
    )

    print_dataset_info(
        target_df,
        "Target",
    )

    # --------------------------------------------------------------
    # Validate Date.
    # --------------------------------------------------------------

    if DATE_COL not in features_df.columns:
        raise ValueError(
            f"{DATE_COL} missing from feature file."
        )

    if DATE_COL not in target_df.columns:
        raise ValueError(
            f"{DATE_COL} missing from target file."
        )

    features_df[DATE_COL] = pd.to_datetime(
        features_df[DATE_COL],
        errors="coerce",
    )

    target_df[DATE_COL] = pd.to_datetime(
        target_df[DATE_COL],
        errors="coerce",
    )

    features_df = features_df.dropna(
        subset=[DATE_COL]
    )

    target_df = target_df.dropna(
        subset=[DATE_COL]
    )

    # --------------------------------------------------------------
    # Select target.
    # --------------------------------------------------------------

    target_df, target_col = select_target(
        target_df
    )

    # --------------------------------------------------------------
    # Date alignment.
    # --------------------------------------------------------------

    header(
        "DATE ALIGNMENT"
    )

    print(
        "Feature date range : "
        f"{features_df[DATE_COL].min().date()} "
        f"to "
        f"{features_df[DATE_COL].max().date()}"
    )

    print(
        "Target date range  : "
        f"{target_df[DATE_COL].min().date()} "
        f"to "
        f"{target_df[DATE_COL].max().date()}"
    )

    print(
        f"Target             : {target_col}"
    )

    # --------------------------------------------------------------
    # Load SAFE features.
    # --------------------------------------------------------------

    header(
        "V6 SAFE FEATURE SET"
    )

    safe_features = load_safe_features()

    usable_features = [
        feature
        for feature in safe_features
        if feature in features_df.columns
    ]

    missing_features = [
        feature
        for feature in safe_features
        if feature not in features_df.columns
    ]

    print(
        f"SAFE features listed : {len(safe_features)}"
    )

    print(
        f"SAFE features usable : {len(usable_features)}"
    )

    if missing_features:
        print()
        print(
            "Missing SAFE features:"
        )

        for feature in missing_features:
            print(
                f"  - {feature}"
            )

    if len(usable_features) == 0:
        raise ValueError(
            "No SAFE features are present in "
            "ml_features_v4.csv."
        )

    # --------------------------------------------------------------
    # Merge.
    # --------------------------------------------------------------

    header(
        "MERGE"
    )

    feature_part = features_df[
        [DATE_COL] + usable_features
    ].copy()

    target_part = target_df[
        [DATE_COL, target_col]
    ].copy()

    # Avoid duplicate dates.
    feature_part = (
        feature_part
        .sort_values(DATE_COL)
        .drop_duplicates(
            subset=[DATE_COL],
            keep="last",
        )
    )

    target_part = (
        target_part
        .sort_values(DATE_COL)
        .drop_duplicates(
            subset=[DATE_COL],
            keep="last",
        )
    )

    data = pd.merge(
        feature_part,
        target_part,
        on=DATE_COL,
        how="inner",
    )

    data = (
        data
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    print(
        f"Merged rows : {len(data)}"
    )

    if len(data) == 0:
        raise ValueError(
            "Merged rows = 0.\n"
            "Check Date formats in both files."
        )

    # --------------------------------------------------------------
    # Convert features to numeric.
    # --------------------------------------------------------------

    for feature in usable_features:

        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

    data[target_col] = pd.to_numeric(
        data[target_col],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Remove invalid target rows.
    # --------------------------------------------------------------

    data = data.dropna(
        subset=[target_col]
    ).copy()

    # --------------------------------------------------------------
    # Make sure target is binary.
    # --------------------------------------------------------------

    unique_target = sorted(
        data[target_col]
        .dropna()
        .unique()
        .tolist()
    )

    print()
    print(
        "Target unique values:"
    )
    print(
        unique_target
    )

    if not set(unique_target).issubset({0, 1}):

        raise ValueError(
            "Selected target is not binary.\n"
            f"Target: {target_col}\n"
            f"Values: {unique_target}"
        )

    data[target_col] = (
        data[target_col]
        .astype(int)
    )

    # --------------------------------------------------------------
    # Target distribution.
    # --------------------------------------------------------------

    print()
    print(
        "Target distribution:"
    )

    print(
        data[target_col]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Target rate: "
        f"{data[target_col].mean():.4f}"
    )

    # --------------------------------------------------------------
    # Missing feature handling.
    #
    # Important:
    # Imputation is fitted separately inside every walk-forward
    # training window to avoid using future information.
    # --------------------------------------------------------------

    return (
        data,
        usable_features,
        target_col,
    )


# ======================================================================
# WALK-FORWARD MODEL EVALUATION
# ======================================================================

def run_walk_forward(
    data,
    features,
    target_col,
    model_name,
    model_factory,
):

    print()
    print(
        "-" * 70
    )

    print(
        f"WALK-FORWARD MODEL: {model_name}"
    )

    print(
        "-" * 70
    )

    fold_rows = []
    prediction_rows = []

    total_rows = len(data)

    fold_number = 0

    train_start = 0

    while (
        train_start + TRAIN_SIZE + TEST_SIZE
        <= total_rows
    ):

        train_end = (
            train_start
            + TRAIN_SIZE
        )

        test_end = (
            train_end
            + TEST_SIZE
        )

        train_data = data.iloc[
            train_start:train_end
        ].copy()

        test_data = data.iloc[
            train_end:test_end
        ].copy()

        # ----------------------------------------------------------
        # Prepare X/y.
        # ----------------------------------------------------------

        X_train = train_data[
            features
        ].copy()

        X_test = test_data[
            features
        ].copy()

        y_train = train_data[
            target_col
        ].astype(int)

        y_test = test_data[
            target_col
        ].astype(int)

        # ----------------------------------------------------------
        # Remove constant / all-missing features within TRAIN ONLY.
        # ----------------------------------------------------------

        valid_features = []

        for feature in features:

            train_series = pd.to_numeric(
                X_train[feature],
                errors="coerce",
            )

            if train_series.notna().sum() == 0:
                continue

            if train_series.nunique(
                dropna=True
            ) <= 1:
                continue

            valid_features.append(
                feature
            )

        if len(valid_features) == 0:

            print(
                f"Fold {fold_number + 1}: "
                "no usable features. Skipping."
            )

            train_start += STEP_SIZE
            fold_number += 1

            continue

        X_train = X_train[
            valid_features
        ]

        X_test = X_test[
            valid_features
        ]

        # ----------------------------------------------------------
        # Median imputation using TRAIN ONLY.
        # ----------------------------------------------------------

        medians = X_train.median(
            numeric_only=True
        )

        X_train = X_train.fillna(
            medians
        )

        X_test = X_test.fillna(
            medians
        )

        # Any remaining NaN means the feature has no usable
        # training median. Replace with zero.
        X_train = X_train.fillna(0.0)
        X_test = X_test.fillna(0.0)

        # ----------------------------------------------------------
        # Check both classes in training data.
        # ----------------------------------------------------------

        if y_train.nunique() < 2:

            print(
                f"Fold {fold_number + 1}: "
                "training target has one class. Skipping."
            )

            train_start += STEP_SIZE
            fold_number += 1

            continue

        # ----------------------------------------------------------
        # Train.
        # ----------------------------------------------------------

        model = model_factory()

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------------
        # Predict probabilities.
        # ----------------------------------------------------------

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_scores(
            y_test,
            probabilities,
        )

        fold_number += 1

        train_start_date = (
            train_data[DATE_COL].min()
        )

        train_end_date = (
            train_data[DATE_COL].max()
        )

        test_start_date = (
            test_data[DATE_COL].min()
        )

        test_end_date = (
            test_data[DATE_COL].max()
        )

        fold_result = {
            "Model": model_name,
            "Fold": fold_number,
            "Train_Rows": len(train_data),
            "Test_Rows": len(test_data),
            "Train_Start": train_start_date,
            "Train_End": train_end_date,
            "Test_Start": test_start_date,
            "Test_End": test_end_date,
            "Features_Used": len(valid_features),
            "Train_Positive_Rate": y_train.mean(),
            "Test_Positive_Rate": y_test.mean(),
        }

        fold_result.update(metrics)

        fold_rows.append(
            fold_result
        )

        # ----------------------------------------------------------
        # Save individual predictions.
        # ----------------------------------------------------------

        fold_predictions = pd.DataFrame(
            {
                "Model": model_name,
                "Fold": fold_number,
                "Date": test_data[DATE_COL].values,
                "Actual": y_test.values,
                "Probability": probabilities,
                "Prediction": (
                    probabilities >= 0.50
                ).astype(int),
            }
        )

        prediction_rows.append(
            fold_predictions
        )

        # ----------------------------------------------------------
        # Console output.
        # ----------------------------------------------------------

        print(
            f"Fold {fold_number:02d} | "
            f"Train {train_start_date.date()} "
            f"to {train_end_date.date()} | "
            f"Test {test_start_date.date()} "
            f"to {test_end_date.date()} | "
            f"AUC={metrics['AUC']:.4f} | "
            f"BAL={metrics['Balanced_Accuracy']:.4f} | "
            f"F1={metrics['F1']:.4f}"
        )

        train_start += STEP_SIZE

    if len(fold_rows) == 0:

        raise ValueError(
            "No walk-forward folds were produced."
        )

    fold_df = pd.DataFrame(
        fold_rows
    )

    if prediction_rows:

        predictions_df = pd.concat(
            prediction_rows,
            ignore_index=True,
        )

    else:

        predictions_df = pd.DataFrame()

    return (
        fold_df,
        predictions_df,
    )


# ======================================================================
# SUMMARY
# ======================================================================

def create_summary(
    fold_results,
):

    summary_rows = []

    for model_name, group in (
        fold_results
        .groupby("Model")
    ):

        summary_rows.append(
            {
                "Model": model_name,
                "Folds": len(group),
                "Mean_Test_AUC": group[
                    "AUC"
                ].mean(),
                "Median_Test_AUC": group[
                    "AUC"
                ].median(),
                "Std_Test_AUC": group[
                    "AUC"
                ].std(),
                "Mean_Balanced_Accuracy": group[
                    "Balanced_Accuracy"
                ].mean(),
                "Mean_Precision": group[
                    "Precision"
                ].mean(),
                "Mean_Recall": group[
                    "Recall"
                ].mean(),
                "Mean_F1": group[
                    "F1"
                ].mean(),
                "Mean_Predicted_Positive_Rate": group[
                    "Predicted_Positive_Rate"
                ].mean(),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    # --------------------------------------------------------------
    # Prepare data.
    # --------------------------------------------------------------

    data, features, target_col = (
        prepare_dataset()
    )

    # --------------------------------------------------------------
    # Basic chronological information.
    # --------------------------------------------------------------

    header(
        "WALK-FORWARD CONFIGURATION"
    )

    print(
        f"Total usable rows : {len(data)}"
    )

    print(
        f"Train size        : {TRAIN_SIZE}"
    )

    print(
        f"Test size         : {TEST_SIZE}"
    )

    print(
        f"Step size         : {STEP_SIZE}"
    )

    print(
        f"Features          : {len(features)}"
    )

    print(
        f"Target            : {target_col}"
    )

    if len(data) < (
        TRAIN_SIZE + TEST_SIZE
    ):

        raise ValueError(
            "Not enough rows for walk-forward validation.\n"
            f"Rows available : {len(data)}\n"
            f"Rows required  : "
            f"{TRAIN_SIZE + TEST_SIZE}"
        )

    # --------------------------------------------------------------
    # Chronological confirmation.
    # --------------------------------------------------------------

    header(
        "CHRONOLOGICAL ORDER"
    )

    print(
        "First date : "
        f"{data[DATE_COL].min().date()}"
    )

    print(
        "Last date  : "
        f"{data[DATE_COL].max().date()}"
    )

    # --------------------------------------------------------------
    # Run models.
    # --------------------------------------------------------------

    all_fold_results = []
    all_predictions = []

    # --------------------------------------------------------------
    # Logistic Regression.
    # --------------------------------------------------------------

    fold_df, prediction_df = run_walk_forward(
        data=data,
        features=features,
        target_col=target_col,
        model_name="Logistic",
        model_factory=make_logistic_model,
    )

    all_fold_results.append(
        fold_df
    )

    if len(prediction_df) > 0:

        all_predictions.append(
            prediction_df
        )

    # --------------------------------------------------------------
    # XGBoost.
    # --------------------------------------------------------------

    fold_df, prediction_df = run_walk_forward(
        data=data,
        features=features,
        target_col=target_col,
        model_name="XGBoost",
        model_factory=make_xgb_model,
    )

    all_fold_results.append(
        fold_df
    )

    if len(prediction_df) > 0:

        all_predictions.append(
            prediction_df
        )

    # --------------------------------------------------------------
    # Combine results.
    # --------------------------------------------------------------

    fold_results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Summary.
    # --------------------------------------------------------------

    header(
        "V6 WALK-FORWARD SUMMARY"
    )

    summary = create_summary(
        fold_results
    )

    if len(summary) > 0:

        display_columns = [
            "Model",
            "Folds",
            "Mean_Test_AUC",
            "Median_Test_AUC",
            "Std_Test_AUC",
            "Mean_Balanced_Accuracy",
            "Mean_Precision",
            "Mean_Recall",
            "Mean_F1",
            "Mean_Predicted_Positive_Rate",
        ]

        print(
            summary[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.4f}"
                ),
            )
        )

    # --------------------------------------------------------------
    # Best model.
    # --------------------------------------------------------------

    header(
        "BEST MODEL"
    )

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

        print(
            f"Best walk-forward model : "
            f"{best_model}"
        )

        print(
            f"Mean Test AUC           : "
            f"{best_auc:.4f}"
        )

        print(
            f"Mean Balanced Accuracy  : "
            f"{best_balanced:.4f}"
        )

        if best_auc >= 0.60:

            print()
            print(
                "SIGNAL: Potentially useful "
                "out-of-sample ranking signal."
            )

        elif best_auc >= 0.55:

            print()
            print(
                "WEAK POSITIVE: Some out-of-sample "
                "ranking signal may exist."
            )

        else:

            print()
            print(
                "WEAK: Walk-forward ranking signal "
                "is close to random."
            )

    # --------------------------------------------------------------
    # Save results.
    # --------------------------------------------------------------

    header(
        "SAVING OUTPUTS"
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    fold_results.to_csv(
        RESULT_FILE,
        index=False,
    )

    predictions.to_csv(
        PRED_FILE,
        index=False,
    )

    # Also save summary next to the results.
    summary_file = os.path.join(
        DATA_DIR,
        "walk_forward_v6_summary.csv",
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------------
    # Final interpretation.
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
        "Missing feature values are imputed "
        "using training data only."
    )

    print(
        "The SAFE feature list is used as the "
        "starting feature set."
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

    # --------------------------------------------------------------
    # Output.
    # --------------------------------------------------------------

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 70
    )

    print(
        f"Fold results : {RESULT_FILE}"
    )

    print(
        f"Predictions  : {PRED_FILE}"
    )

    print(
        f"Summary      : {summary_file}"
    )

    print()
    print(
        "PASS: V6 walk-forward validation "
        "completed successfully."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()