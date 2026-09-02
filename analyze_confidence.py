import pandas as pd

results = pd.read_csv("data/xgboost_predictions_v3.csv")

buy_signals = results[
    results["Predicted_Target"] == 1
].copy()

print("========== CONFIDENCE THRESHOLD ANALYSIS ==========")

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

for threshold in thresholds:

    selected = buy_signals[
        buy_signals["BUY_Probability"] >= threshold
    ]

    if len(selected) == 0:
        continue

    profitable = (
        selected["Future_Return_5D"] > 0
    ).sum()

    win_rate = profitable / len(selected)

    average_return = selected[
        "Future_Return_5D"
    ].mean()

    print(
        f"\nThreshold >= {threshold:.2f}"
    )

    print(
        "Signals:",
        len(selected)
    )

    print(
        "Win Rate:",
        round(win_rate * 100, 2),
        "%"
    )

    print(
        "Average 5D Return:",
        round(average_return * 100, 2),
        "%"
    )