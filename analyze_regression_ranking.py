import pandas as pd

results = pd.read_csv(
    "data/xgboost_regression_predictions.csv"
)

print("========== REGRESSION RANKING ANALYSIS ==========")

quantiles = [0.10, 0.20, 0.30]

for q in quantiles:

    cutoff = results["Predicted_Return_5D"].quantile(
        1 - q
    )

    selected = results[
        results["Predicted_Return_5D"] >= cutoff
    ]

    actual_mean = selected[
        "Future_Return_5D"
    ].mean()

    profitable = (
        selected["Future_Return_5D"] > 0
    ).sum()

    win_rate = profitable / len(selected)

    print(
        f"\nTop {int(q * 100)}% Predictions"
    )

    print(
        "Signals:",
        len(selected)
    )

    print(
        "Prediction Cutoff:",
        round(cutoff * 100, 2),
        "%"
    )

    print(
        "Actual Average 5D Return:",
        round(actual_mean * 100, 2),
        "%"
    )

    print(
        "Win Rate:",
        round(win_rate * 100, 2),
        "%"
    )