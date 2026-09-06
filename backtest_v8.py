import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================================================================
# ALPHALENS V8 THRESHOLD ROBUSTNESS
# ======================================================================

DATA = "data"

PRED_FILE = os.path.join(
    DATA,
    "walk_forward_v6_predictions.csv"
)

RESULT_FILE = os.path.join(
    DATA,
    "backtest_v8_threshold_results.csv"
)

TRADE_FILE = os.path.join(
    DATA,
    "backtest_v8_trades.csv"
)

INITIAL_CAPITAL = 100000.0

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def load_predictions():
    header("LOADING V6 OUT-OF-SAMPLE PREDICTIONS")

    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"Missing file: {PRED_FILE}"
        )

    df = pd.read_csv(PRED_FILE)

    required = [
        "Model",
        "Fold",
        "Date",
        "Actual",
        "Probability",
        "Prediction",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Probability"] = pd.to_numeric(
        df["Probability"],
        errors="coerce"
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "Probability",
            "Actual",
        ]
    )

    df = df.sort_values(
        ["Model", "Date"]
    ).reset_index(drop=True)

    print(f"Rows   : {len(df)}")
    print(
        f"Models : {df['Model'].unique().tolist()}"
    )

    return df


def load_prices():
    header("LOADING PRICE DATA")

    candidates = [
        "data/ml_features_v4.csv",
        "data/features_v4.csv",
        "data/features.csv",
        "data/ohlcv.csv",
        "data/stock_data.csv",
    ]

    price_file = None

    for f in candidates:
        if os.path.exists(f):
            price_file = f
            break

    if price_file is None:
        raise FileNotFoundError(
            "Could not find price CSV."
        )

    df = pd.read_csv(price_file)

    if "Date" not in df.columns:
        raise ValueError(
            "Price file has no Date column."
        )

    if "Close" not in df.columns:
        raise ValueError(
            "Price file has no Close column."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df[
        ["Date", "Close"]
    ].dropna()

    df = df.sort_values(
        "Date"
    ).drop_duplicates(
        "Date"
    )

    print(
        f"Price rows: {len(df)}"
    )

    return df


def prepare(predictions, prices, model):
    df = predictions[
        predictions["Model"] == model
    ].copy()

    df = df.sort_values("Date")

    df = df.merge(
        prices,
        on="Date",
        how="left"
    )

    df["Next_Close"] = (
        df["Close"].shift(-1)
    )

    df["Market_Return"] = (
        df["Next_Close"]
        / df["Close"]
        - 1.0
    )

    df = df.dropna(
        subset=[
            "Close",
            "Next_Close",
            "Market_Return",
        ]
    )

    return df.reset_index(drop=True)


def run_threshold(df, threshold):
    capital = INITIAL_CAPITAL

    position = 0

    trade_count = 0
    winning_trades = 0

    gross_profit = 0.0
    gross_loss = 0.0

    entry_price = None
    entry_date = None

    equity = []

    for _, row in df.iterrows():

        date = row["Date"]
        close = float(row["Close"])
        probability = float(row["Probability"])
        market_return = float(
            row["Market_Return"]
        )

        # --------------------------------------------------------------
        # SIGNAL
        # --------------------------------------------------------------

        if position == 0:

            if probability >= threshold:
                new_position = 1
            else:
                new_position = 0

        else:

            if probability < threshold:
                new_position = 0
            else:
                new_position = 1

        # --------------------------------------------------------------
        # TURNOVER
        # --------------------------------------------------------------

        turnover = abs(
            new_position - position
        )

        cost = turnover * (
            TRANSACTION_COST
            + SLIPPAGE
        )

        # --------------------------------------------------------------
        # STRATEGY RETURN
        # --------------------------------------------------------------

        strategy_return = (
            new_position
            * market_return
        )

        net_return = (
            strategy_return
            - cost
        )

        capital *= (
            1.0 + net_return
        )

        # --------------------------------------------------------------
        # ENTRY
        # --------------------------------------------------------------

        if position == 0 and new_position == 1:

            entry_price = close
            entry_date = date

        # --------------------------------------------------------------
        # EXIT
        # --------------------------------------------------------------

        if position == 1 and new_position == 0:

            if entry_price is not None:

                trade_return = (
                    close / entry_price
                    - 1.0
                )

                trade_count += 1

                if trade_return > 0:
                    winning_trades += 1
                    gross_profit += trade_return
                elif trade_return < 0:
                    gross_loss += abs(
                        trade_return
                    )

            entry_price = None
            entry_date = None

        equity.append({
            "Date": date,
            "Equity": capital,
            "Position": new_position,
            "Probability": probability,
        })

        position = new_position

    equity = pd.DataFrame(equity)

    if len(equity) == 0:
        return None, None

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    final_equity = float(
        equity["Equity"].iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1.0
    )

    days = (
        equity["Date"].iloc[-1]
        - equity["Date"].iloc[0]
    ).days

    if days > 0:

        years = days / 365.25

        cagr = (
            final_equity
            / INITIAL_CAPITAL
        ) ** (
            1.0 / years
        ) - 1.0

    else:
        cagr = np.nan

    equity["Running_Max"] = (
        equity["Equity"].cummax()
    )

    equity["Drawdown"] = (
        equity["Equity"]
        / equity["Running_Max"]
        - 1.0
    )

    max_drawdown = float(
        equity["Drawdown"].min()
    )

    daily_returns = (
        equity["Equity"]
        .pct_change()
        .fillna(0.0)
    )

    if daily_returns.std() > 0:

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )

    else:
        sharpe = np.nan

    if trade_count > 0:

        win_rate = (
            winning_trades
            / trade_count
        )

    else:
        win_rate = np.nan

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    exposure = (
        equity["Position"].mean()
    )

    buy_hold = (
        df["Close"].iloc[-1]
        / df["Close"].iloc[0]
        - 1.0
    )

    excess = (
        total_return
        - buy_hold
    )

    metrics = {
        "Threshold": threshold,
        "Final_Equity": final_equity,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max_Drawdown": max_drawdown,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Trades": trade_count,
        "Exposure": exposure,
        "Buy_Hold_Return": buy_hold,
        "Excess_Return": excess,
    }

    return metrics, equity


def main():

    header(
        "ALPHALENS V8 THRESHOLD ROBUSTNESS TEST"
    )

    print(
        "V6 predictions remain completely untouched."
    )

    print()
    print(
        "Testing thresholds:"
    )

    print(
        THRESHOLDS
    )

    predictions = load_predictions()

    prices = load_prices()

    models = predictions[
        "Model"
    ].unique().tolist()

    all_results = []
    all_equity = []

    for model in models:

        header(
            f"MODEL: {model}"
        )

        df = prepare(
            predictions,
            prices,
            model
        )

        print(
            f"Usable rows: {len(df)}"
        )

        for threshold in THRESHOLDS:

            metrics, equity = run_threshold(
                df,
                threshold
            )

            if metrics is None:
                continue

            metrics["Model"] = model

            all_results.append(metrics)

            equity["Model"] = model
            equity["Threshold"] = threshold

            all_equity.append(equity)

            print(
                f"Threshold={threshold:.2f} | "
                f"Return={metrics['Total_Return']:.2%} | "
                f"CAGR={metrics['CAGR']:.2%} | "
                f"Sharpe={metrics['Sharpe']:.3f} | "
                f"MaxDD={metrics['Max_Drawdown']:.2%} | "
                f"Trades={metrics['Trades']}"
            )

    results = pd.DataFrame(
        all_results
    )

    if all_equity:

        equity_output = pd.concat(
            all_equity,
            ignore_index=True
        )

    else:

        equity_output = pd.DataFrame()

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    equity_output.to_csv(
        TRADE_FILE,
        index=False
    )

    header(
        "V8 THRESHOLD RESULTS"
    )

    display_cols = [
        "Model",
        "Threshold",
        "Total_Return",
        "CAGR",
        "Sharpe",
        "Max_Drawdown",
        "Win_Rate",
        "Profit_Factor",
        "Trades",
        "Exposure",
        "Buy_Hold_Return",
        "Excess_Return",
    ]

    print(
        results[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ------------------------------------------------------------------
    # BEST THRESHOLD BY MODEL
    # ------------------------------------------------------------------

    header(
        "BEST THRESHOLD BY MODEL"
    )

    for model in models:

        subset = results[
            results["Model"] == model
        ].copy()

        if len(subset) == 0:
            continue

        best = subset.loc[
            subset["Total_Return"].idxmax()
        ]

        print()
        print(
            f"{model}"
        )

        print(
            f"Best threshold : "
            f"{best['Threshold']:.2f}"
        )

        print(
            f"Return         : "
            f"{best['Total_Return']:.2%}"
        )

        print(
            f"Sharpe         : "
            f"{best['Sharpe']:.4f}"
        )

        print(
            f"Max Drawdown   : "
            f"{best['Max_Drawdown']:.2%}"
        )

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    header(
        "OUTPUT FILES"
    )

    print(
        f"Results : {RESULT_FILE}"
    )

    print(
        f"Equity  : {TRADE_FILE}"
    )

    print()
    print(
        "PASS: V8 threshold robustness completed."
    )


if __name__ == "__main__":
    main()