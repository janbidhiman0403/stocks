import os
import numpy as np
import pandas as pd


# ======================================================================
# ALPHALENS V10 REGIME ANALYSIS
# ======================================================================

PRED_FILE = os.path.join("data", "walk_forward_v6_predictions.csv")
PRICE_FILE = os.path.join("data", "ml_ready_TCS_news_clean.csv")

RESULT_FILE = os.path.join("data", "v10_regime_results.csv")
DAILY_FILE = os.path.join("data", "v10_regime_daily.csv")


MODEL = "XGBoost"
THRESHOLD = 0.70

INITIAL_CAPITAL = 100000.0
TRANSACTION_COST = 0.0010
SLIPPAGE = 0.0005


# ======================================================================
# HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def find_price_column(df):
    candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
        "Price",
        "price",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a price column. "
        "Expected one of: Close, Adj Close, Price."
    )


def find_date_column(df):
    candidates = [
        "Date",
        "date",
        "Datetime",
        "datetime",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("Could not find a date column.")


def classify_regime(row):
    trend = row["Trend_Return_20D"]
    vol = row["Volatility_20D"]

    if pd.isna(trend) or pd.isna(vol):
        return "Unknown"

    if trend >= 0.05 and vol <= row["Volatility_Median"]:
        return "Strong_Uptrend_LowVol"

    if trend >= 0.05 and vol > row["Volatility_Median"]:
        return "Strong_Uptrend_HighVol"

    if trend <= -0.05 and vol <= row["Volatility_Median"]:
        return "Strong_Downtrend_LowVol"

    if trend <= -0.05 and vol > row["Volatility_Median"]:
        return "Strong_Downtrend_HighVol"

    if trend > 0:
        return "Weak_Uptrend"

    if trend < 0:
        return "Weak_Downtrend"

    return "Sideways"


# ======================================================================
# MAIN
# ======================================================================

def main():

    header("ALPHALENS V10 REGIME ANALYSIS")

    print("Model           :", MODEL)
    print("Entry threshold :", f"{THRESHOLD:.2f}")
    print("Transaction cost:", f"{TRANSACTION_COST:.3%}")
    print("Slippage        :", f"{SLIPPAGE:.3%}")

    # ------------------------------------------------------------------
    # LOAD PREDICTIONS
    # ------------------------------------------------------------------

    header("LOADING V6 OUT-OF-SAMPLE PREDICTIONS")

    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"Missing prediction file: {PRED_FILE}"
        )

    pred = pd.read_csv(PRED_FILE)

    pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce")

    pred = pred[
        pred["Model"].astype(str).str.lower()
        == MODEL.lower()
    ].copy()

    pred["Probability"] = safe_numeric(pred["Probability"])
    pred["Actual"] = safe_numeric(pred["Actual"])

    pred = pred.dropna(
        subset=["Date", "Probability", "Actual"]
    )

    pred = pred.sort_values("Date").reset_index(drop=True)

    print("Model           :", MODEL)
    print("Prediction rows :", len(pred))

    if len(pred) > 0:
        print(
            "Date range      :",
            pred["Date"].min().date(),
            "to",
            pred["Date"].max().date(),
        )

    # ------------------------------------------------------------------
    # LOAD PRICE DATA
    # ------------------------------------------------------------------

    header("LOADING PRICE DATA")

    if not os.path.exists(PRICE_FILE):
        raise FileNotFoundError(
            f"Missing price/feature file: {PRICE_FILE}"
        )

    price = pd.read_csv(PRICE_FILE)

    date_col = find_date_column(price)
    price_col = find_price_column(price)

    price[date_col] = pd.to_datetime(
        price[date_col],
        errors="coerce"
    )

    price[price_col] = safe_numeric(price[price_col])

    price = price[
        [date_col, price_col]
    ].copy()

    price.columns = ["Date", "Close"]

    price = price.dropna(
        subset=["Date", "Close"]
    )

    price = price.sort_values("Date")
    price = price.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    print("Price rows      :", len(price))

    if len(price) > 0:
        print(
            "Price date range:",
            price["Date"].min().date(),
            "to",
            price["Date"].max().date(),
        )

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    header("ALIGNING PREDICTIONS WITH PRICES")

    df = pd.merge(
        pred,
        price,
        on="Date",
        how="inner"
    )

    df = df.sort_values("Date").reset_index(drop=True)

    print("Merged rows:", len(df))

    if len(df) < 50:
        raise ValueError(
            "Too few aligned rows for regime analysis."
        )

    # ------------------------------------------------------------------
    # MARKET REGIME FEATURES
    # ------------------------------------------------------------------

    header("BUILDING MARKET REGIME FEATURES")

    df["Return_1D"] = df["Close"].pct_change()

    df["Trend_Return_20D"] = (
        df["Close"] / df["Close"].shift(20) - 1.0
    )

    df["Trend_Return_60D"] = (
        df["Close"] / df["Close"].shift(60) - 1.0
    )

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    df["Volatility_Median"] = (
        df["Volatility_20D"]
        .expanding(min_periods=20)
        .median()
    )

    df["Forward_Return_10D"] = (
        df["Close"].shift(-10)
        / df["Close"]
        - 1.0
    )

    df["Signal"] = (
        df["Probability"] >= THRESHOLD
    ).astype(int)

    df["Regime"] = df.apply(
        classify_regime,
        axis=1
    )

    df = df.dropna(
        subset=[
            "Trend_Return_20D",
            "Volatility_20D",
            "Forward_Return_10D",
        ]
    ).copy()

    print("Usable rows:", len(df))

    # ------------------------------------------------------------------
    # SIGNAL QUALITY
    # ------------------------------------------------------------------

    header("SIGNAL QUALITY BY REGIME")

    results = []

    for regime, group in df.groupby(
        "Regime",
        sort=True
    ):

        signal_rows = group[
            group["Signal"] == 1
        ].copy()

        all_rows = group.copy()

        if len(signal_rows) > 0:

            mean_forward = (
                signal_rows["Forward_Return_10D"]
                .mean()
            )

            median_forward = (
                signal_rows["Forward_Return_10D"]
                .median()
            )

            win_rate = (
                signal_rows["Forward_Return_10D"] > 0
            ).mean()

        else:

            mean_forward = np.nan
            median_forward = np.nan
            win_rate = np.nan

        baseline_forward = (
            all_rows["Forward_Return_10D"]
            .mean()
        )

        results.append(
            {
                "Regime": regime,
                "Rows": len(group),
                "Signal_Rows": len(signal_rows),
                "Signal_Rate": (
                    len(signal_rows) / len(group)
                    if len(group) > 0
                    else np.nan
                ),
                "Mean_Forward_10D_Return": mean_forward,
                "Median_Forward_10D_Return": median_forward,
                "Signal_Win_Rate": win_rate,
                "Baseline_Mean_Forward_10D_Return":
                    baseline_forward,
                "Signal_Excess_Forward_Return":
                    mean_forward - baseline_forward
                    if pd.notna(mean_forward)
                    else np.nan,
            }
        )

    regime_results = pd.DataFrame(results)

    print(
        regime_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ------------------------------------------------------------------
    # TRADING PERFORMANCE BY REGIME
    # ------------------------------------------------------------------

    header("TRADING PERFORMANCE BY REGIME")

    trading_rows = []

    for regime, group in df.groupby(
        "Regime",
        sort=True
    ):

        signal = group[
            group["Signal"] == 1
        ].copy()

        if len(signal) == 0:
            trading_rows.append(
                {
                    "Regime": regime,
                    "Trades": 0,
                    "Mean_Return": np.nan,
                    "Win_Rate": np.nan,
                    "Profit_Factor": np.nan,
                    "Total_Return": np.nan,
                }
            )
            continue

        returns = signal[
            "Forward_Return_10D"
        ].dropna()

        if len(returns) == 0:
            continue

        net_returns = (
            returns
            - TRANSACTION_COST
            - SLIPPAGE
        )

        positive = net_returns[
            net_returns > 0
        ].sum()

        negative = -net_returns[
            net_returns < 0
        ].sum()

        if negative > 0:
            profit_factor = (
                positive / negative
            )
        else:
            profit_factor = np.inf

        compounded = (
            np.prod(1.0 + net_returns)
            - 1.0
        )

        trading_rows.append(
            {
                "Regime": regime,
                "Trades": len(returns),
                "Mean_Return": net_returns.mean(),
                "Win_Rate": (
                    net_returns > 0
                ).mean(),
                "Profit_Factor": profit_factor,
                "Total_Return": compounded,
            }
        )

    trading_results = pd.DataFrame(
        trading_rows
    )

    print(
        trading_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ------------------------------------------------------------------
    # COMBINE RESULTS
    # ------------------------------------------------------------------

    final_results = pd.merge(
        regime_results,
        trading_results,
        on="Regime",
        how="left"
    )

    # ------------------------------------------------------------------
    # OVERALL SIGNAL TEST
    # ------------------------------------------------------------------

    header("OVERALL SIGNAL TEST")

    signal_all = df[
        df["Signal"] == 1
    ].copy()

    baseline_all = df[
        "Forward_Return_10D"
    ].dropna()

    signal_returns = signal_all[
        "Forward_Return_10D"
    ].dropna()

    print(
        "Signal rows             :",
        len(signal_returns)
    )

    print(
        "Signal mean 10D return  :",
        f"{signal_returns.mean():.4%}"
    )

    print(
        "Baseline mean 10D return:",
        f"{baseline_all.mean():.4%}"
    )

    print(
        "Signal excess            :",
        f"{signal_returns.mean() - baseline_all.mean():.4%}"
    )

    print(
        "Signal win rate          :",
        f"{(signal_returns > 0).mean():.2%}"
    )

    # ------------------------------------------------------------------
    # BEST REGIME
    # ------------------------------------------------------------------

    header("BEST REGIME")

    valid = final_results[
        final_results["Trades"] > 0
    ].copy()

    if len(valid) > 0:

        best_idx = valid[
            "Mean_Return"
        ].idxmax()

        best_regime = valid.loc[
            best_idx,
            "Regime"
        ]

        best_return = valid.loc[
            best_idx,
            "Mean_Return"
        ]

        best_pf = valid.loc[
            best_idx,
            "Profit_Factor"
        ]

        print(
            "Best regime        :",
            best_regime
        )

        print(
            "Mean trade return  :",
            f"{best_return:.4%}"
        )

        print(
            "Profit factor      :",
            f"{best_pf:.4f}"
        )

    # ------------------------------------------------------------------
    # WORST REGIME
    # ------------------------------------------------------------------

    header("WORST REGIME")

    if len(valid) > 0:

        worst_idx = valid[
            "Mean_Return"
        ].idxmin()

        worst_regime = valid.loc[
            worst_idx,
            "Regime"
        ]

        worst_return = valid.loc[
            worst_idx,
            "Mean_Return"
        ]

        print(
            "Worst regime       :",
            worst_regime
        )

        print(
            "Mean trade return  :",
            f"{worst_return:.4%}"
        )

    # ------------------------------------------------------------------
    # SAVE DAILY DATA
    # ------------------------------------------------------------------

    header("SAVING V10 OUTPUTS")

    os.makedirs(
        os.path.dirname(RESULT_FILE),
        exist_ok=True
    )

    df.to_csv(
        DAILY_FILE,
        index=False
    )

    final_results.to_csv(
        RESULT_FILE,
        index=False
    )

    print("Results :", RESULT_FILE)
    print("Daily   :", DAILY_FILE)

    # ------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------

    header("V10 INTERPRETATION")

    if len(signal_returns) == 0:

        print(
            "No XGBoost signals reached the "
            "0.70 threshold."
        )

    elif signal_returns.mean() > 0:

        print(
            "The XGBoost 0.70 signal has a "
            "positive average forward 10D return "
            "in the analyzed sample."
        )

    else:

        print(
            "The XGBoost 0.70 signal has a "
            "negative average forward 10D return "
            "in the analyzed sample."
        )

    print()
    print(
        "Regime analysis is diagnostic research."
    )

    print(
        "It does not establish future profitability."
    )

    print(
        "V6 out-of-sample predictions remain "
        "unchanged."
    )

    print()
    print(
        "PASS: V10 regime analysis completed."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()