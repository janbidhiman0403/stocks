import pandas as pd

results = pd.read_csv("data/xgboost_predictions_v3.csv")

buy_signals = results[results["Predicted_Target"] == 1].copy()

print("========== BUY SIGNAL ANALYSIS ==========")

print("Total BUY signals:", len(buy_signals))

if len(buy_signals) > 0:

    print(
        "\nActual Future Return Statistics:"
    )

    print(
        buy_signals["Future_Return_5D"].describe()
    )

    profitable = (
        buy_signals["Future_Return_5D"] > 0
    ).sum()

    print(
        "\nProfitable BUY signals:",
        profitable
    )

    print(
        "BUY signal win rate:",
        round(
            profitable / len(buy_signals) * 100,
            2
        ),
        "%"
    )

    print("\n========== BUY SIGNALS ==========")

    print(
        buy_signals[
            [
                "Date",
                "Close",
                "Future_Return_5D",
                "BUY_Probability",
                "Confidence"
            ]
        ].to_string(index=False)
    )