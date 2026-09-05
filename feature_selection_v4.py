import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

FEATURE_FILE = "data/ml_features_v4.csv"
TARGET_FILE = "data/ml_targets_v3.csv"
OUTPUT_FILE = "data/feature_selection_v4_results.csv"

TARGET = "Target_Up_V3"

print("=" * 70)
print("ALPHALENS V4 FEATURE SELECTION EXPERIMENT")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

features = pd.read_csv(FEATURE_FILE)
targets = pd.read_csv(TARGET_FILE)

features["Date"] = pd.to_datetime(features["Date"])
targets["Date"] = pd.to_datetime(targets["Date"])

df = features.merge(
    targets[["Date", TARGET]],
    on="Date",
    how="inner"
)

df = df.sort_values("Date").reset_index(drop=True)

print()
print(f"Rows: {len(df)}")
print(f"Date: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ------------------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------------------

base_features = [
    "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Daily_Return_Pct",
    "Price_Change", "High_Low_Range",
    "High_Low_Range_Pct", "Open_Close_Change",
    "Open_Close_Change_Pct"
]

trend_features = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_200",
    "Close_vs_SMA20", "Close_vs_SMA50", "Close_vs_SMA200",
    "EMA_5_20", "EMA_20_50", "EMA_50_200",
    "Price_vs_EMA20", "Price_vs_EMA50", "Price_vs_EMA200",
    "Trend_5_20", "Trend_10_50", "Trend_20_50", "Trend_50_200",
    "Distance_SMA20", "Distance_SMA50"
]

momentum_features = [
    "RSI_14", "RSI_Distance_50",
    "RSI_Overbought", "RSI_Oversold",
    "Momentum_3", "Momentum_5", "Momentum_10",
    "Momentum_20", "Momentum_60",
    "Momentum_Acceleration_5",
    "Momentum_Acceleration_10",
    "MACD", "MACD_Signal", "MACD_Histogram",
    "MACD_Distance", "MACD_Positive", "MACD_Momentum"
]

volatility_features = [
    "Volatility_5", "Volatility_10", "Volatility_20",
    "Volatility_20_Pct",
    "Vol_Ratio_5_20", "Vol_Ratio_10_20",
    "Volatility_Expansion",
    "Volatility_Percentile_60",
    "High_Volatility_Regime",
    "ZScore_20", "ZScore_60",
    "Return_Mean_20", "Return_Std_20",
    "Return_Skew_60", "Return_Kurtosis_60"
]

volume_features = [
    "Volume_SMA_5", "Volume_SMA_20",
    "Volume_Ratio", "Volume_Change",
    "Volume_Ratio_5_20",
    "Volume_Surge", "Volume_Dry"
]

price_structure_features = [
    "Rolling_High_20", "Rolling_Low_20",
    "Price_Position_20",
    "Rolling_High_50", "Rolling_Low_50",
    "Price_Position_50",
    "Breakout_Distance_20",
    "Breakdown_Distance_20",
    "Breakout_Distance_50",
    "Breakdown_Distance_50",
    "Near_20D_High", "Near_20D_Low",
    "Near_50D_High", "Near_50D_Low",
    "Body_Size", "Upper_Wick",
    "Lower_Wick", "Body_to_Range",
    "Bullish_Candle"
]

news_features = [
    "News_Sentiment", "News_Count", "Has_News",
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
    "News_Sentiment_Lag1_V4",
    "News_Sentiment_Momentum",
    "News_Activity_Surge"
]

feature_sets = {
    "Base": base_features,
    "Base_Trend": base_features + trend_features,
    "Base_Momentum": base_features + momentum_features,
    "Base_Volatility": base_features + volatility_features,
    "Base_Volume": base_features + volume_features,
    "Base_PriceStructure": base_features + price_structure_features,
    "Technical_Core": (
        base_features
        + trend_features
        + momentum_features
        + volatility_features
    ),
    "Technical_No_Volume": (
        base_features
        + trend_features
        + momentum_features
        + volatility_features
        + price_structure_features
    ),
    "Technical_News": (
        base_features
        + trend_features
        + momentum_features
        + volatility_features
        + volume_features
        + price_structure_features
        + news_features
    ),
}

# Remove columns that do not exist in the dataset.
for name in feature_sets:
    feature_sets[name] = [
        f for f in feature_sets[name]
        if f in df.columns
    ]

# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# SAME 60/20/20 LOGIC AS V4 BENCHMARK
# ------------------------------------------------------------

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Train:      {len(train)} "
    f"{train['Date'].min().date()} to {train['Date'].max().date()}"
)

print(
    f"Validation: {len(val)} "
    f"{val['Date'].min().date()} to {val['Date'].max().date()}"
)

print(
    f"Test:       {len(test)} "
    f"{test['Date'].min().date()} to {test['Date'].max().date()}"
)

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def evaluate(y_true, probability, threshold=0.50):
    prediction = (probability >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, prediction
        ),
        "Precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_true, prediction, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, probability),
    }


results = []

print()
print("=" * 70)
print("FEATURE SET EXPERIMENTS")
print("=" * 70)

for feature_set_name, feature_list in feature_sets.items():

    print()
    print("-" * 70)
    print(f"FEATURE SET: {feature_set_name}")
    print(f"Features: {len(feature_list)}")
    print("-" * 70)

    X_train = train[feature_list].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    X_val = val[feature_list].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    X_test = test[feature_list].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    y_train = train[TARGET].astype(int)
    y_val = val[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    # --------------------------------------------------------
    # LOGISTIC
    # --------------------------------------------------------

    print("Training Logistic...")

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    logistic.fit(X_train, y_train)

    p_train = logistic.predict_proba(X_train)[:, 1]
    p_val = logistic.predict_proba(X_val)[:, 1]
    p_test = logistic.predict_proba(X_test)[:, 1]

    train_metrics = evaluate(y_train, p_train)
    val_metrics = evaluate(y_val, p_val)
    test_metrics = evaluate(y_test, p_test)

    print(f"Train AUC : {train_metrics['ROC_AUC']:.4f}")
    print(f"Val AUC   : {val_metrics['ROC_AUC']:.4f}")
    print(f"Test AUC  : {test_metrics['ROC_AUC']:.4f}")
    print(
        f"Test Bal. : "
        f"{test_metrics['Balanced_Accuracy']:.4f}"
    )

    results.append({
        "Feature_Set": feature_set_name,
        "Model": "Logistic",
        "Features": len(feature_list),
        "Train_AUC": train_metrics["ROC_AUC"],
        "Validation_AUC": val_metrics["ROC_AUC"],
        "Test_AUC": test_metrics["ROC_AUC"],
        "Validation_Balanced_Accuracy":
            val_metrics["Balanced_Accuracy"],
        "Test_Balanced_Accuracy":
            test_metrics["Balanced_Accuracy"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
    })

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    print()
    print("Training XGBoost...")

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    p_train = xgb.predict_proba(X_train)[:, 1]
    p_val = xgb.predict_proba(X_val)[:, 1]
    p_test = xgb.predict_proba(X_test)[:, 1]

    train_metrics = evaluate(y_train, p_train)
    val_metrics = evaluate(y_val, p_val)
    test_metrics = evaluate(y_test, p_test)

    print(f"Train AUC : {train_metrics['ROC_AUC']:.4f}")
    print(f"Val AUC   : {val_metrics['ROC_AUC']:.4f}")
    print(f"Test AUC  : {test_metrics['ROC_AUC']:.4f}")
    print(
        f"Test Bal. : "
        f"{test_metrics['Balanced_Accuracy']:.4f}"
    )

    results.append({
        "Feature_Set": feature_set_name,
        "Model": "XGBoost",
        "Features": len(feature_list),
        "Train_AUC": train_metrics["ROC_AUC"],
        "Validation_AUC": val_metrics["ROC_AUC"],
        "Test_AUC": test_metrics["ROC_AUC"],
        "Validation_Balanced_Accuracy":
            val_metrics["Balanced_Accuracy"],
        "Test_Balanced_Accuracy":
            test_metrics["Balanced_Accuracy"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
    })


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["Validation_AUC", "Test_AUC"],
    ascending=False
)

print()
print("=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# BEST CONFIGURATION
# ------------------------------------------------------------

best = results_df.iloc[0]

print()
print("=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(f"Feature set       : {best['Feature_Set']}")
print(f"Model              : {best['Model']}")
print(f"Features            : {int(best['Features'])}")
print(f"Validation AUC     : {best['Validation_AUC']:.4f}")
print(f"Test AUC           : {best['Test_AUC']:.4f}")
print(
    f"Test Balanced Acc. : "
    f"{best['Test_Balanced_Accuracy']:.4f}"
)
print(f"Test F1            : {best['Test_F1']:.4f}")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

os.makedirs("data", exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT_FILE}")
print("PASS: V4 feature selection completed.")
print("=" * 70)