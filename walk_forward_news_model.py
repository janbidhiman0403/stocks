import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

print("=" * 65)
print("ALPHALENS WALK-FORWARD VALIDATION")
print("=" * 65)

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/ml_ready_TCS_news_era.csv"

RESULT_FILE = "data/walk_forward_results.csv"
PRED_FILE = "data/walk_forward_predictions.csv"

TARGET = "Target_Direction_1D"

# ============================================================
# FEATURES
# ============================================================

technical_features = [
    "Close", "High", "Low", "Open", "Volume",
    "Daily_Return", "Daily_Return_Pct", "Price_Change",
    "High_Low_Range", "High_Low_Range_Pct",
    "Open_Close_Change", "Open_Close_Change_Pct",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "SMA_100", "SMA_200",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_200",
    "Close_vs_SMA20", "Close_vs_SMA50", "Close_vs_SMA200",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Histogram",
    "Volatility_5", "Volatility_10", "Volatility_20",
    "Volatility_20_Pct",
    "Volume_SMA_5", "Volume_SMA_20",
    "Volume_Ratio", "Volume_Change",
    "Momentum_5", "Momentum_10", "Momentum_20", "Momentum_60",
    "Rolling_High_20", "Rolling_Low_20", "Price_Position_20",
    "Rolling_High_50", "Rolling_Low_50", "Price_Position_50",
]

news_features = [
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

technical_features = [
    f for f in technical_features
    if f in pd.read_csv(DATA_FILE, nrows=1).columns
]

news_features = [
    f for f in news_features
    if f in pd.read_csv(DATA_FILE, nrows=1).columns
]

feature_sets = {
    "Technical_Only": technical_features,
    "Technical_News": technical_features + news_features,
}

# ============================================================
# LOAD
# ============================================================

print("\nLoading:", DATA_FILE)

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Rows:", len(df))
print("Date:", df["Date"].min().date(), "to", df["Date"].max().date())

# ============================================================
# CLEAN
# ============================================================

required_columns = list(
    set(
        technical_features
        + news_features
        + [
            TARGET,
            "Target_Return_1D",
            "Close",
            "Date",
            "News_Sentiment",
            "News_Count",
            "Has_News",
        ]
    )
)

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print("Rows after cleaning:", len(df))

# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

# Expanding training window.
#
# Example:
#
# Train: first 365 days
# Test : next 60 days
#
# Then:
#
# Train: first 425 days
# Test : next 60 days
#
# etc.

INITIAL_TRAIN = 365
TEST_WINDOW = 60
STEP = 60

print("\n" + "=" * 65)
print("WALK-FORWARD SETTINGS")
print("=" * 65)

print("Initial training rows:", INITIAL_TRAIN)
print("Test window:", TEST_WINDOW)
print("Step:", STEP)
print("Training type: EXPANDING")

# ============================================================
# MODEL
# ============================================================

def make_model():

    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probability):

    prediction = (
        probability >= 0.50
    ).astype(int)

    result = {}

    result["Accuracy"] = accuracy_score(
        y_true,
        prediction
    )

    result["Balanced_Accuracy"] = balanced_accuracy_score(
        y_true,
        prediction
    )

    result["Precision"] = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    result["Recall"] = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    result["F1"] = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    try:
        result["ROC_AUC"] = roc_auc_score(
            y_true,
            probability
        )
    except:
        result["ROC_AUC"] = np.nan

    return result, prediction


# ============================================================
# WALK FORWARD
# ============================================================

all_results = []
all_predictions = []

fold = 0

start = INITIAL_TRAIN

while start < len(df):

    fold += 1

    train_end = start
    test_end = min(
        start + TEST_WINDOW,
        len(df)
    )

    train_df = df.iloc[:train_end]
    test_df = df.iloc[start:test_end]

    if len(test_df) < 20:
        break

    print("\n" + "-" * 65)
    print(f"FOLD {fold}")
    print("-" * 65)

    print(
        "Train:",
        train_df["Date"].min().date(),
        "to",
        train_df["Date"].max().date(),
        f"({len(train_df)} rows)"
    )

    print(
        "Test :",
        test_df["Date"].min().date(),
        "to",
        test_df["Date"].max().date(),
        f"({len(test_df)} rows)"
    )

    print(
        "News coverage:",
        f"{test_df['Has_News'].mean() * 100:.2f}%"
    )

    for model_name, features in feature_sets.items():

        X_train = train_df[features]
        y_train = train_df[TARGET]

        X_test = test_df[features]
        y_test = test_df[TARGET]

        model = make_model()

        model.fit(
            X_train,
            y_train
        )

        probability = model.predict_proba(
            X_test
        )[:, 1]

        metrics, prediction = calculate_metrics(
            y_test,
            probability
        )

        strategy_returns = np.where(
            prediction == 1,
            test_df["Target_Return_1D"].values,
            0
        )

        buy_hold_returns = (
            test_df["Target_Return_1D"].values
        )

        strategy_return = (
            np.prod(1 + strategy_returns) - 1
        )

        buy_hold_return = (
            np.prod(1 + buy_hold_returns) - 1
        )

        cumulative = np.cumprod(
            1 + strategy_returns
        )

        running_max = np.maximum.accumulate(
            cumulative
        )

        drawdown = (
            cumulative / running_max
        ) - 1

        max_drawdown = drawdown.min()

        if np.std(strategy_returns) > 0:
            sharpe = (
                np.mean(strategy_returns)
                / np.std(strategy_returns)
            ) * np.sqrt(252)
        else:
            sharpe = 0

        row = {
            "Fold": fold,
            "Model": model_name,

            "Train_Start":
                train_df["Date"].min(),

            "Train_End":
                train_df["Date"].max(),

            "Test_Start":
                test_df["Date"].min(),

            "Test_End":
                test_df["Date"].max(),

            "Train_Rows":
                len(train_df),

            "Test_Rows":
                len(test_df),

            "News_Coverage":
                test_df["Has_News"].mean(),

            "Accuracy":
                metrics["Accuracy"],

            "Balanced_Accuracy":
                metrics["Balanced_Accuracy"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1":
                metrics["F1"],

            "ROC_AUC":
                metrics["ROC_AUC"],

            "Strategy_Return":
                strategy_return,

            "Buy_Hold_Return":
                buy_hold_return,

            "Max_Drawdown":
                max_drawdown,

            "Sharpe":
                sharpe,
        }

        all_results.append(row)

        for i in range(len(test_df)):

            all_predictions.append({
                "Date":
                    test_df.iloc[i]["Date"],

                "Fold":
                    fold,

                "Model":
                    model_name,

                "Close":
                    test_df.iloc[i]["Close"],

                "News_Sentiment":
                    test_df.iloc[i]["News_Sentiment"],

                "News_Count":
                    test_df.iloc[i]["News_Count"],

                "Has_News":
                    test_df.iloc[i]["Has_News"],

                "Actual":
                    y_test.iloc[i],

                "Prediction":
                    prediction[i],

                "Probability_Up":
                    probability[i],

                "Target_Return_1D":
                    test_df.iloc[i]["Target_Return_1D"],

                "Strategy_Return":
                    strategy_returns[i],
            })

        print(
            f"{model_name:18s} | "
            f"Acc={metrics['Accuracy']:.3f} | "
            f"AUC={metrics['ROC_AUC']:.3f} | "
            f"Return={strategy_return * 100:.2f}% | "
            f"Sharpe={sharpe:.2f}"
        )

    start += STEP


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

predictions_df.to_csv(
    PRED_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("WALK-FORWARD SUMMARY")
print("=" * 65)

for model_name in feature_sets:

    r = results_df[
        results_df["Model"] == model_name
    ]

    print("\n" + model_name)
    print("-" * 40)

    print(
        "Folds:",
        len(r)
    )

    print(
        "Mean Accuracy:",
        f"{r['Accuracy'].mean():.4f}"
    )

    print(
        "Mean Balanced Accuracy:",
        f"{r['Balanced_Accuracy'].mean():.4f}"
    )

    print(
        "Mean F1:",
        f"{r['F1'].mean():.4f}"
    )

    print(
        "Mean ROC-AUC:",
        f"{r['ROC_AUC'].mean():.4f}"
    )

    print(
        "Median ROC-AUC:",
        f"{r['ROC_AUC'].median():.4f}"
    )

    print(
        "Mean Strategy Return/Fold:",
        f"{r['Strategy_Return'].mean() * 100:.2f}%"
    )

    print(
        "Mean Buy-Hold Return/Fold:",
        f"{r['Buy_Hold_Return'].mean() * 100:.2f}%"
    )

    print(
        "Mean Sharpe:",
        f"{r['Sharpe'].mean():.3f}"
    )

    print(
        "Worst Drawdown:",
        f"{r['Max_Drawdown'].min() * 100:.2f}%"
    )

# ============================================================
# OVERALL OOS PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("OVERALL OUT-OF-SAMPLE PERFORMANCE")
print("=" * 65)

for model_name in feature_sets:

    p = predictions_df[
        predictions_df["Model"] == model_name
    ].copy()

    y_true = p["Actual"].values

    probability = p["Probability_Up"].values

    prediction = (
        probability >= 0.50
    ).astype(int)

    strategy_returns = (
        p["Strategy_Return"].values
    )

    buy_hold_returns = (
        p["Target_Return_1D"].values
    )

    total_strategy = (
        np.prod(1 + strategy_returns) - 1
    )

    total_buy_hold = (
        np.prod(1 + buy_hold_returns) - 1
    )

    cumulative = np.cumprod(
        1 + strategy_returns
    )

    running_max = np.maximum.accumulate(
        cumulative
    )

    drawdown = (
        cumulative / running_max
    ) - 1

    if np.std(strategy_returns) > 0:
        sharpe = (
            np.mean(strategy_returns)
            / np.std(strategy_returns)
        ) * np.sqrt(252)
    else:
        sharpe = 0

    print("\n" + model_name)
    print("-" * 40)

    print(
        "OOS rows:",
        len(p)
    )

    print(
        "Accuracy:",
        f"{accuracy_score(y_true, prediction):.4f}"
    )

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy_score(y_true, prediction):.4f}"
    )

    print(
        "ROC-AUC:",
        f"{roc_auc_score(y_true, probability):.4f}"
    )

    print(
        "Total Strategy Return:",
        f"{total_strategy * 100:.2f}%"
    )

    print(
        "Total Buy & Hold:",
        f"{total_buy_hold * 100:.2f}%"
    )

    print(
        "Max Drawdown:",
        f"{drawdown.min() * 100:.2f}%"
    )

    print(
        "Sharpe:",
        f"{sharpe:.3f}"
    )

# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 65)
print("MODEL DECISION")
print("=" * 65)

summary = (
    results_df
    .groupby("Model")
    .agg({
        "ROC_AUC": "mean",
        "Balanced_Accuracy": "mean",
        "Sharpe": "mean",
        "Strategy_Return": "mean",
    })
    .sort_values(
        "ROC_AUC",
        ascending=False
    )
)

print(
    summary.to_string()
)

best = summary.index[0]

print("\nBest model by mean walk-forward ROC-AUC:")
print(best)

print("\nOutput files:")
print(RESULT_FILE)
print(PRED_FILE)

print("\n" + "=" * 65)
print("PASS: WALK-FORWARD VALIDATION COMPLETE")
print("=" * 65)