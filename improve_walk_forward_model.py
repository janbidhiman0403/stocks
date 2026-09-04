import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

INPUT = "data/ml_ready_TCS_news_era.csv"
OUTPUT_RESULTS = "data/improved_walk_forward_results.csv"
OUTPUT_PRED = "data/improved_walk_forward_predictions.csv"

TARGET = "Target_Direction_1D"

NEWS_FEATURES = [
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

TECHNICAL_FEATURES = [
    "Daily_Return",
    "Daily_Return_Pct",
    "Price_Change",
    "High_Low_Range",
    "High_Low_Range_Pct",
    "Open_Close_Change",
    "Open_Close_Change_Pct",
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
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Volatility_20_Pct",
    "Volume_SMA_5",
    "Volume_SMA_20",
    "Volume_Ratio",
    "Volume_Change",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "Momentum_60",
    "Rolling_High_20",
    "Rolling_Low_20",
    "Price_Position_20",
    "Rolling_High_50",
    "Rolling_Low_50",
    "Price_Position_50",
]

# IMPORTANT:
# Do NOT use raw Close/Open/High/Low/Volume.
# We use normalized/derived features to reduce scale/regime dependence.
SAFE_TECHNICAL = TECHNICAL_FEATURES

MODELS = {
    "Logistic_Technical": "logistic_technical",
    "Logistic_Tech_News": "logistic_news",
    "RandomForest_Technical": "rf_technical",
    "RandomForest_Tech_News": "rf_news",
    "XGBoost_Technical": "xgb_technical",
    "XGBoost_Tech_News": "xgb_news",
}

print("=" * 70)
print("ALPHALENS IMPROVED WALK-FORWARD MODEL")
print("=" * 70)

print("\nLoading:", INPUT)

df = pd.read_csv(INPUT)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Rows:", len(df))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

required = SAFE_TECHNICAL + NEWS_FEATURES + [TARGET, "Close"]

missing = [c for c in required if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    for c in missing:
        print(" -", c)
    raise SystemExit(1)

df = df.replace([np.inf, -np.inf], np.nan)

before = len(df)
df = df.dropna(subset=required).reset_index(drop=True)

print("Rows after cleaning:", len(df))
print("Rows removed:", before - len(df))

# ------------------------------------------------------------
# FEATURE SETS
# ------------------------------------------------------------

feature_sets = {
    "Technical": SAFE_TECHNICAL,
    "Tech_News": SAFE_TECHNICAL + NEWS_FEATURES,
}

print("\nFeature counts:")
for name, cols in feature_sets.items():
    print(f"{name:12s}: {len(cols)}")

# ------------------------------------------------------------
# WALK-FORWARD SETTINGS
# ------------------------------------------------------------

INITIAL_TRAIN = 365
TEST_WINDOW = 60
STEP = 60

print("\n" + "=" * 70)
print("WALK-FORWARD SETTINGS")
print("=" * 70)

print("Initial training:", INITIAL_TRAIN)
print("Test window     :", TEST_WINDOW)
print("Step            :", STEP)
print("Training type   : EXPANDING")

# ------------------------------------------------------------
# MODEL FACTORY
# ------------------------------------------------------------

def make_model(name, y_train):

    positive = max(int((y_train == 1).sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)

    weight = negative / positive

    if name == "logistic_technical" or name == "logistic_news":

        return LogisticRegression(
            C=0.1,
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )

    if name == "rf_technical" or name == "rf_news":

        return RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    if name == "xgb_technical" or name == "xgb_news":

        return XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=weight,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(name)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(y_true, pred, prob):

    result = {}

    result["Accuracy"] = accuracy_score(y_true, pred)
    result["Balanced_Accuracy"] = balanced_accuracy_score(y_true, pred)
    result["Precision"] = precision_score(
        y_true, pred, zero_division=0
    )
    result["Recall"] = recall_score(
        y_true, pred, zero_division=0
    )
    result["F1"] = f1_score(
        y_true, pred, zero_division=0
    )

    try:
        result["ROC_AUC"] = roc_auc_score(y_true, prob)
    except Exception:
        result["ROC_AUC"] = np.nan

    return result


# ------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------

def backtest_returns(close, pred):

    close = np.asarray(close, dtype=float)
    pred = np.asarray(pred)

    returns = close[1:] / close[:-1] - 1

    # Prediction at day t is used for t -> t+1.
    signal = pred[:-1]

    strategy = np.where(signal == 1, returns, -returns)

    equity = np.cumprod(1 + strategy)

    if len(equity) == 0:
        return np.nan, np.nan, np.nan

    total_return = equity[-1] - 1

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    if strategy.std() > 0:
        sharpe = (
            strategy.mean() /
            strategy.std()
        ) * np.sqrt(252)
    else:
        sharpe = 0.0

    return total_return, max_drawdown, sharpe


# ------------------------------------------------------------
# WALK FORWARD
# ------------------------------------------------------------

all_results = []
all_predictions = []

fold = 0
start = INITIAL_TRAIN

while start < len(df):

    test_end = min(start + TEST_WINDOW, len(df))

    train = df.iloc[:start].copy()
    test = df.iloc[start:test_end].copy()

    if len(test) < 20:
        break

    fold += 1

    print("\n" + "-" * 70)
    print(f"FOLD {fold}")
    print("-" * 70)

    print(
        "Train:",
        train["Date"].min().date(),
        "to",
        train["Date"].max().date(),
        f"({len(train)} rows)"
    )

    print(
        "Test :",
        test["Date"].min().date(),
        "to",
        test["Date"].max().date(),
        f"({len(test)} rows)"
    )

    print(
        "News coverage:",
        f"{test['Has_News'].mean() * 100:.2f}%"
    )

    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    for dataset_name, features in feature_sets.items():

        X_train = train[features]
        X_test = test[features]

        for model_label, model_type in MODELS.items():

            if dataset_name == "Technical":
                if "_News" in model_label:
                    continue
            else:
                if "_Technical" in model_label:
                    continue

            model = make_model(model_type, y_train)

            model.fit(X_train, y_train)

            prob = model.predict_proba(X_test)[:, 1]

            pred = (prob >= 0.50).astype(int)

            metrics = calculate_metrics(
                y_test,
                pred,
                prob
            )

            strategy_return, max_dd, sharpe = backtest_returns(
                test["Close"].values,
                pred
            )

            row = {
                "Fold": fold,
                "Model": model_label,
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "Test_Start": test["Date"].min(),
                "Test_End": test["Date"].max(),
                "News_Coverage": test["Has_News"].mean(),
                **metrics,
                "Strategy_Return": strategy_return,
                "Max_Drawdown": max_dd,
                "Sharpe": sharpe,
            }

            all_results.append(row)

            for i in range(len(test)):

                all_predictions.append({
                    "Fold": fold,
                    "Model": model_label,
                    "Date": test.iloc[i]["Date"],
                    "Close": test.iloc[i]["Close"],
                    "Actual": int(y_test.iloc[i]),
                    "Prediction": int(pred[i]),
                    "Probability_Up": prob[i],
                    "Correct": bool(pred[i] == y_test.iloc[i])
                })

            print(
                f"{model_label:24s} "
                f"| Acc={metrics['Accuracy']:.3f} "
                f"| AUC={metrics['ROC_AUC']:.3f} "
                f"| Return={strategy_return:.2%} "
                f"| Sharpe={sharpe:.2f}"
            )

    start += STEP


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame(all_results)
predictions = pd.DataFrame(all_predictions)

results.to_csv(
    OUTPUT_RESULTS,
    index=False
)

predictions.to_csv(
    OUTPUT_PRED,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

summary = (
    results
    .groupby("Model")
    .agg(
        Folds=("Fold", "count"),
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
        Mean_Precision=("Precision", "mean"),
        Mean_Recall=("Recall", "mean"),
        Mean_F1=("F1", "mean"),
        Mean_ROC_AUC=("ROC_AUC", "mean"),
        Median_ROC_AUC=("ROC_AUC", "median"),
        Mean_Strategy_Return=("Strategy_Return", "mean"),
        Mean_Max_Drawdown=("Max_Drawdown", "mean"),
        Worst_Drawdown=("Max_Drawdown", "min"),
        Mean_Sharpe=("Sharpe", "mean"),
    )
    .sort_values(
        ["Mean_ROC_AUC", "Mean_Sharpe"],
        ascending=False
    )
)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# OVERALL OOS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL OUT-OF-SAMPLE")
print("=" * 70)

for model_name in predictions["Model"].unique():

    p = predictions[
        predictions["Model"] == model_name
    ].copy()

    y = p["Actual"]
    pred = p["Prediction"]
    prob = p["Probability_Up"]

    metrics = calculate_metrics(
        y,
        pred,
        prob
    )

    print("\n" + model_name)
    print("-" * 50)

    print(f"OOS rows           : {len(p)}")
    print(f"Accuracy           : {metrics['Accuracy']:.4f}")
    print(
        f"Balanced Accuracy  : "
        f"{metrics['Balanced_Accuracy']:.4f}"
    )
    print(f"Precision          : {metrics['Precision']:.4f}")
    print(f"Recall             : {metrics['Recall']:.4f}")
    print(f"F1                 : {metrics['F1']:.4f}")
    print(f"ROC-AUC            : {metrics['ROC_AUC']:.4f}")

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

best = summary.index[0]

print("\n" + "=" * 70)
print("MODEL DECISION")
print("=" * 70)

print("Best model by mean walk-forward ROC-AUC:")
print(best)

print("\nFull ranking:")
print(
    summary[
        [
            "Mean_ROC_AUC",
            "Mean_Balanced_Accuracy",
            "Mean_Sharpe",
            "Mean_Strategy_Return",
            "Worst_Drawdown",
        ]
    ].to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOutput files:")
print(OUTPUT_RESULTS)
print(OUTPUT_PRED)

print("\n" + "=" * 70)
print("PASS: IMPROVED WALK-FORWARD EXPERIMENT COMPLETE")
print("=" * 70)