
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# ALPHALENS V11 REGIME-FILTERED OUT-OF-SAMPLE BACKTEST
# ======================================================================
#
# Purpose:
#   Test whether the V10 market-regime findings improve the V8/V9
#   XGBoost strategy when used as a trading filter.
#
# IMPORTANT:
#   - V6 predictions are NOT modified.
#   - No model is retrained.
#   - No future information is used to create the regime variables.
#   - The XGBoost threshold remains 0.70.
#   - Transaction cost and slippage remain identical to V9.
#
# V11 compares:
#   1. Baseline XGBoost 0.70
#   2. Exclude Strong_Downtrend_LowVol
#   3. Trade only Strong_Downtrend_HighVol
#   4. Exclude both problematic strong-downtrend/low-quality regimes
#   5. Confidence sizing + regime filter
#
# ======================================================================


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

PREDICTION_FILE = Path("data/walk_forward_v6_predictions.csv")
PRICE_FILE = Path("data/ml_ready_TCS_news_clean.csv")

RESULT_FILE = Path("data/v11_regime_filter_results.csv")
TRADE_FILE = Path("data/v11_regime_filter_trades.csv")
EQUITY_FILE = Path("data/v11_regime_filter_equity.csv")

MODEL_NAME = "XGBoost"

ENTRY_THRESHOLD = 0.70
EXIT_THRESHOLD = 0.50

TRANSACTION_COST = 0.0010
SLIPPAGE = 0.0005

INITIAL_CAPITAL = 100000.0


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def section(text):
    print()
    print("-" * 70)
    print(text)
    print("-" * 70)


# ======================================================================
# REGIME ENGINE
# ======================================================================
#
# This deliberately uses only information available at date t.
#
# Trend:
#   Close vs SMA50
#
# Volatility:
#   20-day realized volatility
#
# Strong trend:
#   absolute distance from SMA50 exceeds a volatility-scaled threshold
#
# The exact V10 labels are reproduced:
#
#   Strong_Downtrend_HighVol
#   Strong_Downtrend_LowVol
#   Strong_Uptrend_HighVol
#   Strong_Uptrend_LowVol
#   Weak_Downtrend
#   Weak_Uptrend
#
# ======================================================================

def build_regime_features(df):
    out = df.copy()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")

    # --------------------------------------------------------------
    # Moving average
    # --------------------------------------------------------------

    out["SMA_50"] = out["Close"].rolling(50, min_periods=50).mean()

    # --------------------------------------------------------------
    # Daily return
    # --------------------------------------------------------------

    out["Daily_Return"] = out["Close"].pct_change()

    # --------------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------------

    out["Volatility_20"] = (
        out["Daily_Return"]
        .rolling(20, min_periods=20)
        .std()
    )

    # --------------------------------------------------------------
    # Price distance from SMA50
    # --------------------------------------------------------------

    out["Trend_Distance"] = (
        out["Close"] / out["SMA_50"] - 1.0
    )

    # --------------------------------------------------------------
    # Volatility-scaled trend strength
    # --------------------------------------------------------------

    out["Trend_Strength"] = (
        out["Trend_Distance"].abs()
        / out["Volatility_20"]
    )

    # --------------------------------------------------------------
    # High / low volatility
    #
    # Compare current volatility with its expanding historical median.
    # This avoids using future observations.
    # --------------------------------------------------------------

    expanding_vol_median = (
        out["Volatility_20"]
        .expanding(min_periods=60)
        .median()
    )

    out["High_Vol"] = (
        out["Volatility_20"] > expanding_vol_median
    )

    # --------------------------------------------------------------
    # Strong trend
    # --------------------------------------------------------------

    out["Strong_Trend"] = (
        out["Trend_Strength"] >= 1.0
    )

    # --------------------------------------------------------------
    # Direction
    # --------------------------------------------------------------

    out["Uptrend"] = (
        out["Trend_Distance"] > 0
    )

    # --------------------------------------------------------------
    # Regime labels
    # --------------------------------------------------------------

    conditions = [
        out["Strong_Trend"]
        & (~out["Uptrend"])
        & out["High_Vol"],

        out["Strong_Trend"]
        & (~out["Uptrend"])
        & (~out["High_Vol"]),

        out["Strong_Trend"]
        & out["Uptrend"]
        & out["High_Vol"],

        out["Strong_Trend"]
        & out["Uptrend"]
        & (~out["High_Vol"]),

        (~out["Strong_Trend"])
        & (~out["Uptrend"]),

        (~out["Strong_Trend"])
        & out["Uptrend"],
    ]

    choices = [
        "Strong_Downtrend_HighVol",
        "Strong_Downtrend_LowVol",
        "Strong_Uptrend_HighVol",
        "Strong_Uptrend_LowVol",
        "Weak_Downtrend",
        "Weak_Uptrend",
    ]

    out["Regime"] = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    return out


# ======================================================================
# METRIC HELPERS
# ======================================================================

def calculate_max_drawdown(equity):
    equity = pd.Series(equity, dtype=float)

    if len(equity) == 0:
        return np.nan

    running_max = equity.cummax()

    drawdown = equity / running_max - 1.0

    return float(drawdown.min())


def calculate_sharpe(daily_returns):
    returns = pd.Series(daily_returns, dtype=float).dropna()

    if len(returns) < 2:
        return np.nan

    std = returns.std(ddof=1)

    if std == 0 or np.isnan(std):
        return np.nan

    return float(
        returns.mean()
        / std
        * np.sqrt(252)
    )


def calculate_sortino(daily_returns):
    returns = pd.Series(daily_returns, dtype=float).dropna()

    if len(returns) < 2:
        return np.nan

    downside = returns[returns < 0]

    if len(downside) == 0:
        return np.nan

    downside_std = np.sqrt(
        np.mean(downside ** 2)
    )

    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return float(
        returns.mean()
        / downside_std
        * np.sqrt(252)
    )


def calculate_profit_factor(trade_returns):
    returns = pd.Series(trade_returns, dtype=float).dropna()

    if len(returns) == 0:
        return np.nan

    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return float(
        gross_profit / gross_loss
    )


# ======================================================================
# BACKTEST
# ======================================================================

def run_backtest(
    df,
    strategy_name,
    signal_filter,
    sizing_mode="fixed",
):
    capital = INITIAL_CAPITAL

    position = 0.0
    entry_price = np.nan
    entry_date = None
    entry_regime = None

    equity_rows = []
    trades = []

    previous_equity = capital

    # --------------------------------------------------------------
    # Daily loop
    # --------------------------------------------------------------

    for i in range(len(df)):
        row = df.iloc[i]

        date = row["Date"]
        close = float(row["Close"])
        probability = float(row["Probability"])
        regime = row["Regime"]

        # ----------------------------------------------------------
        # Trading signal
        # ----------------------------------------------------------

        signal = probability >= ENTRY_THRESHOLD

        allowed = signal_filter(regime)

        desired_position = 0.0

        if signal and allowed:

            if sizing_mode == "confidence":
                # Confidence above threshold.
                #
                # At 0.70 => 0%
                # At 1.00 => 100%
                #
                confidence = (
                    probability - ENTRY_THRESHOLD
                ) / (1.0 - ENTRY_THRESHOLD)

                desired_position = float(
                    np.clip(confidence, 0.0, 1.0)
                )

            else:
                desired_position = 1.0

        # ----------------------------------------------------------
        # Existing position handling
        # ----------------------------------------------------------

        if position > 0:

            exit_signal = (
                probability < EXIT_THRESHOLD
                or not allowed
            )

            if exit_signal:

                exit_price = close

                gross_return = (
                    exit_price / entry_price - 1.0
                )

                round_trip_cost = (
                    TRANSACTION_COST
                    + SLIPPAGE
                )

                net_return = (
                    gross_return
                    - round_trip_cost
                )

                pnl = (
                    capital
                    * position
                    * net_return
                )

                capital += pnl

                trades.append(
                    {
                        "Strategy": strategy_name,
                        "Entry_Date": entry_date,
                        "Exit_Date": date,
                        "Entry_Price": entry_price,
                        "Exit_Price": exit_price,
                        "Position_Size": position,
                        "Entry_Regime": entry_regime,
                        "Exit_Regime": regime,
                        "Gross_Return": gross_return,
                        "Net_Return": net_return,
                        "PnL": pnl,
                    }
                )

                position = 0.0
                entry_price = np.nan
                entry_date = None
                entry_regime = None

        # ----------------------------------------------------------
        # Open position
        # ----------------------------------------------------------

        if position == 0 and desired_position > 0:

            position = desired_position

            entry_price = close * (
                1.0 + SLIPPAGE
            )

            entry_date = date
            entry_regime = regime

        # ----------------------------------------------------------
        # Mark-to-market equity
        # ----------------------------------------------------------

        if position > 0:

            unrealized_return = (
                close / entry_price - 1.0
            )

            equity = (
                capital
                + INITIAL_CAPITAL
                * position
                * unrealized_return
            )

        else:
            equity = capital

        daily_return = (
            equity / previous_equity - 1.0
            if previous_equity != 0
            else 0.0
        )

        equity_rows.append(
            {
                "Date": date,
                "Strategy": strategy_name,
                "Equity": equity,
                "Daily_Return": daily_return,
                "Position": position,
                "Probability": probability,
                "Regime": regime,
            }
        )

        previous_equity = equity

    # --------------------------------------------------------------
    # Force-close final position
    # --------------------------------------------------------------

    if position > 0:

        last = df.iloc[-1]

        exit_price = float(last["Close"])

        gross_return = (
            exit_price / entry_price - 1.0
        )

        round_trip_cost = (
            TRANSACTION_COST
            + SLIPPAGE
        )

        net_return = (
            gross_return
            - round_trip_cost
        )

        pnl = (
            capital
            * position
            * net_return
        )

        capital += pnl

        trades.append(
            {
                "Strategy": strategy_name,
                "Entry_Date": entry_date,
                "Exit_Date": last["Date"],
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Position_Size": position,
                "Entry_Regime": entry_regime,
                "Exit_Regime": last["Regime"],
                "Gross_Return": gross_return,
                "Net_Return": net_return,
                "PnL": pnl,
            }
        )

        if equity_rows:
            equity_rows[-1]["Equity"] = capital

    # --------------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------------

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    if len(equity_df) == 0:
        return None, trades_df, equity_df

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    final_equity = float(
        equity_df["Equity"].iloc[-1]
    )

    total_return = (
        final_equity / INITIAL_CAPITAL - 1.0
    )

    days = max(
        (
            equity_df["Date"].iloc[-1]
            - equity_df["Date"].iloc[0]
        ).days,
        1,
    )

    years = days / 365.25

    if final_equity > 0:
        cagr = (
            final_equity / INITIAL_CAPITAL
        ) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    daily_returns = equity_df["Daily_Return"]

    sharpe = calculate_sharpe(
        daily_returns
    )

    sortino = calculate_sortino(
        daily_returns
    )

    max_drawdown = calculate_max_drawdown(
        equity_df["Equity"]
    )

    if len(trades_df) > 0:

        win_rate = float(
            (
                trades_df["Net_Return"] > 0
            ).mean()
        )

        profit_factor = calculate_profit_factor(
            trades_df["Net_Return"]
        )

    else:

        win_rate = np.nan
        profit_factor = np.nan

    exposure = float(
        (
            equity_df["Position"] > 0
        ).mean()
    )

    # --------------------------------------------------------------
    # Buy & hold over exact backtest period
    # --------------------------------------------------------------

    first_price = float(
        df["Close"].iloc[0]
    )

    last_price = float(
        df["Close"].iloc[-1]
    )

    buy_hold_return = (
        last_price / first_price - 1.0
    )

    excess_return = (
        total_return - buy_hold_return
    )

    metrics = {
        "Strategy": strategy_name,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_drawdown,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Trades": len(trades_df),
        "Exposure": exposure,
        "Buy_Hold_Return": buy_hold_return,
        "Excess_Return": excess_return,
    }

    return metrics, trades_df, equity_df


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "ALPHALENS V11 REGIME-FILTERED OUT-OF-SAMPLE BACKTEST"
    )

    print(
        "Model           :",
        MODEL_NAME,
    )

    print(
        "Entry threshold :",
        f"{ENTRY_THRESHOLD:.2f}",
    )

    print(
        "Exit threshold  :",
        f"{EXIT_THRESHOLD:.2f}",
    )

    print(
        "Transaction cost:",
        f"{TRANSACTION_COST * 100:.3f}%",
    )

    print(
        "Slippage        :",
        f"{SLIPPAGE * 100:.3f}%",
    )

    # ==================================================================
    # LOAD PREDICTIONS
    # ==================================================================

    header("LOADING V6 OUT-OF-SAMPLE PREDICTIONS")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"],
        errors="coerce",
    )

    predictions = predictions[
        predictions["Model"] == MODEL_NAME
    ].copy()

    predictions = predictions.sort_values(
        "Date"
    ).reset_index(drop=True)

    required_prediction_columns = [
        "Date",
        "Actual",
        "Probability",
        "Prediction",
    ]

    missing = [
        c
        for c in required_prediction_columns
        if c not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "Missing prediction columns: "
            + str(missing)
        )

    predictions["Probability"] = pd.to_numeric(
        predictions["Probability"],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=[
            "Date",
            "Probability",
        ]
    )

    print(
        "Model            :",
        MODEL_NAME,
    )

    print(
        "Prediction rows  :",
        len(predictions),
    )

    if len(predictions) > 0:

        print(
            "Date range       :",
            predictions["Date"].min().date(),
            "to",
            predictions["Date"].max().date(),
        )

    # ==================================================================
    # LOAD PRICE DATA
    # ==================================================================

    header("LOADING PRICE DATA")

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Missing price/feature file: {PRICE_FILE}"
        )

    prices = pd.read_csv(
        PRICE_FILE
    )

    if "Date" not in prices.columns:
        raise ValueError(
            "Price file does not contain Date column."
        )

    if "Close" not in prices.columns:
        raise ValueError(
            "Price file does not contain Close column."
        )

    prices["Date"] = pd.to_datetime(
        prices["Date"],
        errors="coerce",
    )

    prices["Close"] = pd.to_numeric(
        prices["Close"],
        errors="coerce",
    )

    prices = prices.dropna(
        subset=[
            "Date",
            "Close",
        ]
    )

    prices = prices.sort_values(
        "Date"
    ).drop_duplicates(
        subset=["Date"],
        keep="last",
    ).reset_index(drop=True)

    print(
        "Price rows       :",
        len(prices),
    )

    print(
        "Price date range :",
        prices["Date"].min().date(),
        "to",
        prices["Date"].max().date(),
    )

    # ==================================================================
    # BUILD REGIME FEATURES
    # ==================================================================

    header("BUILDING MARKET REGIME FEATURES")

    prices = build_regime_features(
        prices
    )

    # ==================================================================
    # ALIGN
    # ==================================================================

    header("ALIGNING PREDICTIONS WITH PRICES")

    df = predictions.merge(
        prices[
            [
                "Date",
                "Close",
                "SMA_50",
                "Daily_Return",
                "Volatility_20",
                "Trend_Distance",
                "Trend_Strength",
                "High_Vol",
                "Strong_Trend",
                "Uptrend",
                "Regime",
            ]
        ],
        on="Date",
        how="inner",
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        "Merged rows:",
        len(df),
    )

    # Regime features need enough historical observations.
    df = df.dropna(
        subset=[
            "SMA_50",
            "Volatility_20",
            "Trend_Strength",
            "Regime",
        ]
    ).reset_index(drop=True)

    print(
        "Usable rows:",
        len(df),
    )

    if len(df) == 0:
        raise ValueError(
            "No usable rows remain after regime construction."
        )

    # ==================================================================
    # REGIME COUNTS
    # ==================================================================

    header("REGIME DISTRIBUTION")

    regime_counts = (
        df["Regime"]
        .value_counts()
        .sort_index()
    )

    print(
        regime_counts.to_string()
    )

    # ==================================================================
    # STRATEGIES
    # ==================================================================
    #
    # The filters intentionally test different hypotheses.
    #
    # Baseline:
    #   Every XGBoost 0.70 signal is traded.
    #
    # Filter 1:
    #   Remove the regime V10 identified as the worst:
    #   Strong_Downtrend_LowVol
    #
    # Filter 2:
    #   Trade only the best V10 regime:
    #   Strong_Downtrend_HighVol
    #
    # Filter 3:
    #   Exclude both strong downtrend regimes.
    #   This is a deliberately conservative test.
    #
    # Filter 4:
    #   Confidence sizing combined with the best-regime filter.
    #
    # ==================================================================

    strategies = [
        (
            "Baseline_XGB_0.70",
            lambda regime: True,
            "fixed",
        ),

        (
            "Exclude_Strong_Downtrend_LowVol",
            lambda regime:
                regime != "Strong_Downtrend_LowVol",
            "fixed",
        ),

        (
            "Only_Strong_Downtrend_HighVol",
            lambda regime:
                regime == "Strong_Downtrend_HighVol",
            "fixed",
        ),

        (
            "Exclude_Strong_Downtrend_Bad_Regimes",
            lambda regime:
                regime
                not in [
                    "Strong_Downtrend_LowVol",
                    "Weak_Downtrend",
                ],
            "fixed",
        ),

        (
            "Best_Regime_Confidence_Sizing",
            lambda regime:
                regime == "Strong_Downtrend_HighVol",
            "confidence",
        ),
    ]

    # ==================================================================
    # RUN
    # ==================================================================

    header("RUNNING V11 STRATEGIES")

    all_results = []
    all_trades = []
    all_equity = []

    for (
        strategy_name,
        signal_filter,
        sizing_mode,
    ) in strategies:

        section(
            f"STRATEGY: {strategy_name}"
        )

        metrics, trades, equity = run_backtest(
            df=df,
            strategy_name=strategy_name,
            signal_filter=signal_filter,
            sizing_mode=sizing_mode,
        )

        if metrics is None:
            print(
                "No usable result."
            )
            continue

        all_results.append(
            metrics
        )

        if len(trades) > 0:
            all_trades.append(
                trades
            )

        if len(equity) > 0:
            all_equity.append(
                equity
            )

        print(
            "Return         :",
            f"{metrics['Total_Return']:.2%}",
        )

        print(
            "CAGR           :",
            f"{metrics['CAGR']:.2%}",
        )

        print(
            "Sharpe         :",
            f"{metrics['Sharpe']:.4f}",
        )

        print(
            "Sortino        :",
            f"{metrics['Sortino']:.4f}",
        )

        print(
            "Max Drawdown   :",
            f"{metrics['Max_Drawdown']:.2%}",
        )

        print(
            "Win Rate       :",
            f"{metrics['Win_Rate']:.2%}",
        )

        print(
            "Profit Factor  :",
            f"{metrics['Profit_Factor']:.4f}",
        )

        print(
            "Trades         :",
            metrics["Trades"],
        )

        print(
            "Exposure       :",
            f"{metrics['Exposure']:.2%}",
        )

        print(
            "Excess Return  :",
            f"{metrics['Excess_Return']:.2%}",
        )

    # ==================================================================
    # RESULTS
    # ==================================================================

    header("V11 FINAL COMPARISON")

    results_df = pd.DataFrame(
        all_results
    )

    if len(results_df) == 0:
        raise ValueError(
            "No V11 strategies produced results."
        )

    display_columns = [
        "Strategy",
        "Total_Return",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Max_Drawdown",
        "Win_Rate",
        "Profit_Factor",
        "Trades",
        "Exposure",
        "Buy_Hold_Return",
        "Excess_Return",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ==================================================================
    # BEST STRATEGIES
    # ==================================================================

    header("V11 BEST CANDIDATES")

    best_return_index = (
        results_df["Total_Return"]
        .idxmax()
    )

    best_sharpe_index = (
        results_df["Sharpe"]
        .idxmax()
    )

    best_drawdown_index = (
        results_df["Max_Drawdown"]
        .idxmax()
    )

    best_return = results_df.loc[
        best_return_index
    ]

    best_sharpe = results_df.loc[
        best_sharpe_index
    ]

    best_drawdown = results_df.loc[
        best_drawdown_index
    ]

    print(
        "Best by total return :",
        best_return["Strategy"],
    )

    print(
        "Return               :",
        f"{best_return['Total_Return']:.2%}",
    )

    print()

    print(
        "Best by Sharpe       :",
        best_sharpe["Strategy"],
    )

    print(
        "Sharpe               :",
        f"{best_sharpe['Sharpe']:.4f}",
    )

    print()

    print(
        "Best by drawdown     :",
        best_drawdown["Strategy"],
    )

    print(
        "Max drawdown         :",
        f"{best_drawdown['Max_Drawdown']:.2%}",
    )

    # ==================================================================
    # COMPARE AGAINST BASELINE
    # ==================================================================

    header("V11 IMPROVEMENT VS BASELINE")

    baseline_rows = results_df[
        results_df["Strategy"]
        == "Baseline_XGB_0.70"
    ]

    if len(baseline_rows) > 0:

        baseline = baseline_rows.iloc[0]

        comparison_rows = []

        for _, row in results_df.iterrows():

            comparison_rows.append(
                {
                    "Strategy":
                        row["Strategy"],

                    "Return_Improvement":
                        row["Total_Return"]
                        - baseline["Total_Return"],

                    "Sharpe_Improvement":
                        row["Sharpe"]
                        - baseline["Sharpe"],

                    "Drawdown_Improvement":
                        row["Max_Drawdown"]
                        - baseline["Max_Drawdown"],

                    "Profit_Factor_Improvement":
                        row["Profit_Factor"]
                        - baseline["Profit_Factor"],
                }
            )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        print(
            comparison_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    header("SAVING V11 OUTPUTS")

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    if len(all_trades) > 0:

        trades_df = pd.concat(
            all_trades,
            ignore_index=True,
        )

    else:

        trades_df = pd.DataFrame()

    trades_df.to_csv(
        TRADE_FILE,
        index=False,
    )

    if len(all_equity) > 0:

        equity_df = pd.concat(
            all_equity,
            ignore_index=True,
        )

    else:

        equity_df = pd.DataFrame()

    equity_df.to_csv(
        EQUITY_FILE,
        index=False,
    )

    print()
    print(
        "Results :",
        RESULT_FILE,
    )

    print(
        "Trades  :",
        TRADE_FILE,
    )

    print(
        "Equity  :",
        EQUITY_FILE,
    )

    # ==================================================================
    # INTERPRETATION
    # ==================================================================

    header("V11 INTERPRETATION")

    print(
        "V11 tests whether the regime information discovered "
        "in V10 improves the historical out-of-sample XGBoost "
        "0.70 strategy."
    )

    print()

    print(
        "V6 predictions remain unchanged."
    )

    print(
        "No model retraining occurs in V11."
    )

    print(
        "Regime filters are evaluated only on the V6 "
        "out-of-sample prediction period."
    )

    print()

    if len(baseline_rows) > 0:

        baseline_return = (
            baseline["Total_Return"]
        )

        best_return_value = (
            best_return["Total_Return"]
        )

        if best_return_value > baseline_return:

            print(
                "RESULT: At least one regime filter "
                "improved historical total return "
                "relative to the baseline."
            )

        else:

            print(
                "RESULT: Regime filtering did not improve "
                "historical total return relative to baseline."
            )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "V11 is still historical out-of-sample research."
    )

    print(
        "A better historical regime-filtered result "
        "does not establish future profitability."
    )

    print(
        "Further validation should test parameter stability, "
        "walk-forward regime thresholds and realistic execution."
    )

    print()

    print(
        "PASS: V11 regime-filter backtest completed successfully."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
