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
    confusion_matrix,
)

print("=" * 60)
print("ALPHALENS MODEL COMPARISON + BACKTEST")
print("=" * 60)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_DIR = "data"

TRAIN_FILE = os.path.join(DATA_DIR, "train_TCS_news_era.csv")
VAL_FILE = os.path.join(DATA_DIR, "validation_TCS_news_era.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_TCS_news_era.csv")

OUTPUT_FILE = os.path.join(DATA_DIR, "model_comparison_results.csv")
PRED_FILE = os.path.join(DATA_DIR, "model_comparison_predictions.csv")

TARGET = "Target_Direction_1D"

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train      : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test       : {len(test)}")

# ------------------------------------------------------------
# FEATURE GROUPS
# ------------------------------------------------------------

technical_features = [
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

# Only keep columns that actually exist
technical_features = [
    x for x in technical_features
    if x in train.columns
]

news_features = [
    x for x in news_features
    if x in train.columns
]

feature_sets = {
    "Technical_Only": technical_features,
    "News_Only": news_features,
    "Technical_News": technical_features + news_features,
}

# ------------------------------------------------------------
# MODEL FUNCTION
# ------------------------------------------------------------

def train_model(X_train, y_train):

    model = XGBClassifier(
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

    model.fit(X_train, y_train)

    return model


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def evaluate_model(model, X, y):

    probability = model.predict_proba(X)[:, 1]

    prediction = (probability >= 0.50).astype(int)

    accuracy = accuracy_score(y, prediction)

    balanced = balanced_accuracy_score(y, prediction)

    precision = precision_score(
        y,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y,
        prediction,
        zero_division=0
    )

    try:
        auc = roc_auc_score(y, probability)
    except:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "prediction": prediction,
        "probability": probability,
    }


# ------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------

def backtest(test_df, prediction):

    df = test_df.copy()

    # Actual next-day return
    returns = df["Target_Return_1D"].values

    # Long if prediction = 1
    # Cash if prediction = 0
    strategy_returns = np.where(
        prediction == 1,
        returns,
        0
    )

    # Buy and hold
    buy_hold_returns = returns.copy()

    strategy_curve = np.cumprod(
        1 + strategy_returns
    )

    buy_hold_curve = np.cumprod(
        1 + buy_hold_returns
    )

    strategy_total = strategy_curve[-1] - 1

    buy_hold_total = buy_hold_curve[-1] - 1

    # Drawdown
    running_max = np.maximum.accumulate(strategy_curve)

    drawdown = (
        strategy_curve / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # Sharpe
    if np.std(strategy_returns) > 0:
        sharpe = (
            np.mean(strategy_returns)
            / np.std(strategy_returns)
        ) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        "strategy_return": strategy_total,
        "buy_hold_return": buy_hold_total,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


# ------------------------------------------------------------
# RUN MODELS
# ------------------------------------------------------------

results = []
prediction_output = test[
    [
        "Date",
        "Close",
        "News_Sentiment",
        "News_Count",
        "Has_News",
        TARGET,
        "Target_Return_1D",
    ]
].copy()

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, features in feature_sets.items():

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print("Features:", len(features))

    X_train = train[features]
    y_train = train[TARGET]

    X_val = val[features]
    y_val = val[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    print("Training...")

    model = train_model(
        X_train,
        y_train
    )

    train_result = evaluate_model(
        model,
        X_train,
        y_train
    )

    val_result = evaluate_model(
        model,
        X_val,
        y_val
    )

    test_result = evaluate_model(
        model,
        X_test,
        y_test
    )

    backtest_result = backtest(
        test,
        test_result["prediction"]
    )

    print("\nTRAIN")
    print(
        f"Accuracy          : "
        f"{train_result['accuracy']:.4f}"
    )

    print("\nVALIDATION")
    print(
        f"Accuracy          : "
        f"{val_result['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy : "
        f"{val_result['balanced_accuracy']:.4f}"
    )
    print(
        f"F1                : "
        f"{val_result['f1']:.4f}"
    )
    print(
        f"ROC AUC           : "
        f"{val_result['roc_auc']:.4f}"
    )

    print("\nTEST")
    print(
        f"Accuracy          : "
        f"{test_result['accuracy']:.4f}"
    )
    print(
        f"Balanced Accuracy : "
        f"{test_result['balanced_accuracy']:.4f}"
    )
    print(
        f"Precision         : "
        f"{test_result['precision']:.4f}"
    )
    print(
        f"Recall            : "
        f"{test_result['recall']:.4f}"
    )
    print(
        f"F1                : "
        f"{test_result['f1']:.4f}"
    )
    print(
        f"ROC AUC           : "
        f"{test_result['roc_auc']:.4f}"
    )

    print("\nBACKTEST")
    print(
        f"Strategy Return   : "
        f"{backtest_result['strategy_return'] * 100:.2f}%"
    )
    print(
        f"Buy & Hold Return : "
        f"{backtest_result['buy_hold_return'] * 100:.2f}%"
    )
    print(
        f"Max Drawdown      : "
        f"{backtest_result['max_drawdown'] * 100:.2f}%"
    )
    print(
        f"Sharpe Ratio      : "
        f"{backtest_result['sharpe']:.3f}"
    )

    results.append({
        "Model": model_name,

        "Train_Accuracy":
            train_result["accuracy"],

        "Validation_Accuracy":
            val_result["accuracy"],

        "Validation_Balanced_Accuracy":
            val_result["balanced_accuracy"],

        "Validation_F1":
            val_result["f1"],

        "Validation_ROC_AUC":
            val_result["roc_auc"],

        "Test_Accuracy":
            test_result["accuracy"],

        "Test_Balanced_Accuracy":
            test_result["balanced_accuracy"],

        "Test_Precision":
            test_result["precision"],

        "Test_Recall":
            test_result["recall"],

        "Test_F1":
            test_result["f1"],

        "Test_ROC_AUC":
            test_result["roc_auc"],

        "Strategy_Return":
            backtest_result["strategy_return"],

        "Buy_Hold_Return":
            backtest_result["buy_hold_return"],

        "Max_Drawdown":
            backtest_result["max_drawdown"],

        "Sharpe":
            backtest_result["sharpe"],
    })

    prediction_output[
        f"{model_name}_Prediction"
    ] = test_result["prediction"]

    prediction_output[
        f"{model_name}_Probability"
    ] = test_result["probability"]


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)

majority_class = train[TARGET].mode()[0]

baseline_prediction = np.full(
    len(test),
    majority_class
)

baseline_accuracy = accuracy_score(
    test[TARGET],
    baseline_prediction
)

baseline_balanced = balanced_accuracy_score(
    test[TARGET],
    baseline_prediction
)

print(
    f"Majority class : {majority_class}"
)

print(
    f"Test accuracy   : "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Balanced acc.   : "
    f"{baseline_balanced:.4f}"
)

results.append({
    "Model": "Majority_Baseline",
    "Train_Accuracy": np.nan,
    "Validation_Accuracy": np.nan,
    "Validation_Balanced_Accuracy": np.nan,
    "Validation_F1": np.nan,
    "Validation_ROC_AUC": np.nan,
    "Test_Accuracy": baseline_accuracy,
    "Test_Balanced_Accuracy": baseline_balanced,
    "Test_Precision": np.nan,
    "Test_Recall": np.nan,
    "Test_F1": np.nan,
    "Test_ROC_AUC": np.nan,
    "Strategy_Return": np.nan,
    "Buy_Hold_Return": np.nan,
    "Max_Drawdown": np.nan,
    "Sharpe": np.nan,
})

# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

prediction_output.to_csv(
    PRED_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL TABLE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

display_columns = [
    "Model",
    "Validation_Accuracy",
    "Validation_ROC_AUC",
    "Test_Accuracy",
    "Test_Balanced_Accuracy",
    "Test_F1",
    "Test_ROC_AUC",
    "Strategy_Return",
    "Buy_Hold_Return",
    "Max_Drawdown",
    "Sharpe",
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

valid_models = results_df[
    results_df["Model"] != "Majority_Baseline"
].copy()

best_model = valid_models.loc[
    valid_models["Test_ROC_AUC"].idxmax(),
    "Model"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Best by Test ROC-AUC: {best_model}"
)

print("\nOutput files:")
print(OUTPUT_FILE)
print(PRED_FILE)

print("\n" + "=" * 60)
print("PASS: MODEL COMPARISON COMPLETE")
print("=" * 60)