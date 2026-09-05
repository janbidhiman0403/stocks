import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# ALPHALENS V3 FEATURE SELECTION EXPERIMENT
# ============================================================

DATA_PATH = Path("data/ml_targets_v3.csv")

OUTPUT = Path(
    "data/feature_selection_v3_results.csv"
)

print("=" * 70)
print("ALPHALENS V3 FEATURE SELECTION EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

TARGET = "Target_Up_V3"

print("\nRows:", len(df))
print(
    "Date:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


# ============================================================
# FEATURE GROUPS
# ============================================================

CORE = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "Daily_Return_Pct",
    "Price_Change",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Open_Close_Change",
    "Open_Close_Change_Pct",
]


MOMENTUM = [
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "EMA_5",
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "Close_vs_SMA20",
    "Close_vs_SMA50",
    "Close_vs_SMA200",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Momentum_60",
]


VOLATILITY_VOLUME = [
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Volatility_20_Pct",
    "Volume_SMA_5",
    "Volume_SMA_20",
    "Volume_Ratio",
    "Volume_Change",
    "Rolling_High_20",
    "Rolling_Low_20",
    "Price_Position_20",
    "Rolling_High_50",
    "Rolling_Low_50",
    "Price_Position_50",
]


NEWS = [
    "News_Sentiment",
    "News_Count",
    "Has_News",
    "News_Sentiment_Lag1",
    "News_Sentiment_MA3",
    "News_Sentiment_MA5",
    "News_Count_MA3",
    "News_Sentiment_Change",
    "News_Sentiment_Lag2",
    "News_Sentiment_Lag3",
    "News_Sentiment_MA5_Trading",
    "News_Sentiment_MA10",
    "News_Count_MA5",
    "News_Count_MA10",
    "News_Sentiment_Abs",
    "News_Positive",
    "News_Negative",
    "News_Neutral",
    "News_Return_Interaction",
    "News_Momentum_Interaction",
    "News_Volume_Interaction",
]


FEATURE_SETS = {

    "Core":
        CORE,

    "Core_Momentum":
        CORE + MOMENTUM,

    "Technical":
        CORE + MOMENTUM + VOLATILITY_VOLUME,

    "Technical_News":
        CORE + MOMENTUM + VOLATILITY_VOLUME + NEWS
}


# ============================================================
# VALIDATE FEATURES
# ============================================================

for name, features in FEATURE_SETS.items():

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:

        print(
            f"\nWARNING {name}: missing features"
        )

        for f in missing:
            print(" -", f)


# ============================================================
# DATA
# ============================================================

df = df.dropna(
    subset=[TARGET]
).reset_index(drop=True)

y = df[TARGET].astype(int)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)

val_end = int(n * 0.80)


train = df.iloc[:train_end]

validation = df.iloc[
    train_end:val_end
]

test = df.iloc[
    val_end:
]


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    "Train:",
    len(train),
    train["Date"].min().date(),
    "to",
    train["Date"].max().date()
)

print(
    "Validation:",
    len(validation),
    validation["Date"].min().date(),
    "to",
    validation["Date"].max().date()
)

print(
    "Test:",
    len(test),
    test["Date"].min().date(),
    "to",
    test["Date"].max().date()
)


# ============================================================
# MODELS
# ============================================================

def make_logistic():

    return Pipeline([

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
                C=0.05,
                max_iter=3000,
                class_weight="balanced"
            )
        )
    ])


def make_xgb():

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            XGBClassifier(

                n_estimators=200,

                max_depth=2,

                learning_rate=0.025,

                min_child_weight=12,

                subsample=0.75,

                colsample_bytree=0.70,

                reg_alpha=1.0,

                reg_lambda=5.0,

                objective="binary:logistic",

                eval_metric="logloss",

                random_state=42,

                n_jobs=-1
            )
        )
    ])


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    X,
    y
):

    pred = model.predict(X)

    prob = model.predict_proba(X)[:, 1]

    return {

        "Accuracy":
            accuracy_score(y, pred),

        "Balanced_Accuracy":
            balanced_accuracy_score(y, pred),

        "Precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y,
                prob
            )
    }


# ============================================================
# EXPERIMENT
# ============================================================

results = []


for feature_set, features in FEATURE_SETS.items():

    features = [
        f for f in features
        if f in df.columns
    ]

    print("\n" + "-" * 70)

    print(
        "FEATURE SET:",
        feature_set
    )

    print(
        "Features:",
        len(features)
    )

    X_train = train[features]
    y_train = train[TARGET]

    X_val = validation[features]
    y_val = validation[TARGET]

    X_test = test[features]
    y_test = test[TARGET]


    models = {

        "Logistic":
            make_logistic(),

        "XGBoost":
            make_xgb()
    }


    for model_name, model in models.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )


        train_metrics = evaluate(
            model,
            X_train,
            y_train
        )

        val_metrics = evaluate(
            model,
            X_val,
            y_val
        )

        test_metrics = evaluate(
            model,
            X_test,
            y_test
        )


        row = {

            "Feature_Set":
                feature_set,

            "Model":
                model_name,

            "Features":
                len(features),

            "Train_AUC":
                train_metrics["ROC_AUC"],

            "Validation_AUC":
                val_metrics["ROC_AUC"],

            "Test_AUC":
                test_metrics["ROC_AUC"],

            "Validation_Balanced_Accuracy":
                val_metrics[
                    "Balanced_Accuracy"
                ],

            "Test_Balanced_Accuracy":
                test_metrics[
                    "Balanced_Accuracy"
                ],

            "Test_Accuracy":
                test_metrics[
                    "Accuracy"
                ],

            "Test_Precision":
                test_metrics[
                    "Precision"
                ],

            "Test_Recall":
                test_metrics[
                    "Recall"
                ],

            "Test_F1":
                test_metrics[
                    "F1"
                ]
        }


        results.append(row)


        print(
            f"Train AUC : "
            f"{train_metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Val AUC   : "
            f"{val_metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Test AUC  : "
            f"{test_metrics['ROC_AUC']:.4f}"
        )

        print(
            f"Test Bal. : "
            f"{test_metrics['Balanced_Accuracy']:.4f}"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    [
        "Test_AUC",
        "Test_Balanced_Accuracy"
    ],
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best = results_df.iloc[0]


print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    "Feature set:",
    best["Feature_Set"]
)

print(
    "Model:",
    best["Model"]
)

print(
    "Features:",
    best["Features"]
)

print(
    f"Validation AUC: "
    f"{best['Validation_AUC']:.4f}"
)

print(
    f"Test AUC: "
    f"{best['Test_AUC']:.4f}"
)

print(
    f"Test Balanced Accuracy: "
    f"{best['Test_Balanced_Accuracy']:.4f}"
)

print(
    f"Test F1: "
    f"{best['Test_F1']:.4f}"
)


print("\nOutput:")
print(OUTPUT)

print("\n" + "=" * 70)
print("PASS: FEATURE SELECTION COMPLETE")
print("=" * 70)