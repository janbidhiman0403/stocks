# ================================================================
# ALPHALENS V12 ROBUSTNESS VALIDATION
# ================================================================
#
# Purpose:
#   Test robustness of the V6 out-of-sample XGBoost predictions
#   across:
#       - prediction thresholds
#       - holding periods
#       - market regimes
#
# Important:
#   - V6 predictions are NOT modified.
#   - No model retraining occurs.
#   - No future information is used to create the signal.
#   - Regime features are constructed only from current/past prices.
#   - All evaluator arrays remain aligned.
#
# Outputs:
#   data\v12_robustness_results.csv
#   data\v12_robustness_equity.csv
#   data\v12_robustness_trades.csv
#
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PREDICTIONS_FILE = DATA_DIR / "walk_forward_v6_predictions.csv"

OUTPUT_RESULTS = DATA_DIR / "v12_robustness_results.csv"
OUTPUT_EQUITY = DATA_DIR / "v12_robustness_equity.csv"
OUTPUT_TRADES = DATA_DIR / "v12_robustness_trades.csv"

MODEL_NAME = "XGBoost"

BASE_THRESHOLD = 0.70

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

HOLDING_PERIODS = [5, 10, 15, 20]

INITIAL_CAPITAL = 100000.0

TRANSACTION_COST = 0.0010
SLIPPAGE = 0.0005

TOTAL_ENTRY_COST = TRANSACTION_COST + SLIPPAGE
TOTAL_EXIT_COST = TRANSACTION_COST + SLIPPAGE

BEST_REGIME = "Strong_Downtrend_HighVol"

# Regime definitions
TREND_LOOKBACK_FAST = 50
TREND_LOOKBACK_SLOW = 200

VOL_LOOKBACK = 20
VOL_REGIME_LOOKBACK = 252

# Minimum amount of historical data needed before regime calculation
MIN_REGIME_ROWS = 200


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def fmt_pct(value):
    if pd.isna(value):
        return "nan"
    return f"{value * 100:.2f}%"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ================================================================
# FIND PRICE FILE
# ================================================================

def find_price_file():
    """
    Find a usable market price file.

    The original V12 script expected features_v3.csv, but that file
    is not required here. This function automatically searches the
    AlphaLens data directory for a file containing:
        - a date column
        - a close/price column
    """

    candidates = [
        DATA_DIR / "features_v3.csv",
        DATA_DIR / "ml_features_v4.csv",
        DATA_DIR / "ml_ready_TCS_v3.csv",
        DATA_DIR / "ml_ready_TCS_v2.csv",
        DATA_DIR / "ml_ready_TCS.csv",
        DATA_DIR / "ml_targets_v3.csv",
        DATA_DIR / "ml_targets_v2.csv",
        DATA_DIR / "processed_TCS_features.csv",
        DATA_DIR / "processed_TCS.csv",
    ]

    existing = [p for p in candidates if p.exists()]

    if not existing:
        raise FileNotFoundError(
            "Could not find a usable price/feature file in data directory."
        )

    for path in existing:
        try:
            sample = pd.read_csv(path, nrows=5)

            date_candidates = [
                "Date",
                "date",
                "Datetime",
                "datetime",
                "Timestamp",
                "timestamp",
            ]

            close_candidates = [
                "Close",
                "close",
                "Adj Close",
                "adj_close",
                "Adj_Close",
                "Price",
                "price",
            ]

            has_date = any(c in sample.columns for c in date_candidates)
            has_close = any(c in sample.columns for c in close_candidates)

            if has_date and has_close:
                return path

        except Exception:
            continue

    raise FileNotFoundError(
        "No usable price/feature file was found.\n"
        "The file must contain a Date column and a Close/Price column."
    )


# ================================================================
# LOAD PRICE DATA
# ================================================================

def load_price_data():
    print_header("LOADING PRICE DATA")

    price_file = find_price_file()

    print(f"Price source   : {price_file}")

    df = pd.read_csv(price_file)

    date_col = None
    for col in [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp",
    ]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("No date column found in price file.")

    close_col = None
    for col in [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
        "Adj_Close",
        "Price",
        "price",
    ]:
        if col in df.columns:
            close_col = col
            break

    if close_col is None:
        raise ValueError("No Close/Price column found in price file.")

    df = df[[date_col, close_col]].copy()

    df.columns = ["Date", "Close"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    df = df[df["Close"] > 0].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Price file contains no usable rows.")

    print(f"Price rows       : {len(df)}")
    print(
        f"Price date range : "
        f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    return df


# ================================================================
# LOAD V6 PREDICTIONS
# ================================================================

def load_predictions():
    print_header("LOADING V6 OUT-OF-SAMPLE PREDICTIONS")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "Model",
        "Fold",
        "Date",
        "Actual",
        "Probability",
        "Prediction",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Probability"] = pd.to_numeric(
        df["Probability"], errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Probability"]
    )

    df = df[
        df["Model"].astype(str).str.lower()
        == MODEL_NAME.lower()
    ].copy()

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise ValueError(
            f"No {MODEL_NAME} predictions found."
        )

    print(f"Model            : {MODEL_NAME}")
    print(f"Prediction rows  : {len(df)}")
    print(
        f"Date range       : "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    return df


# ================================================================
# BUILD MARKET REGIME FEATURES
# ================================================================

def build_regime_features(price_df):
    print_header("BUILDING MARKET REGIME FEATURES")

    df = price_df.copy()

    close = df["Close"]

    # ------------------------------------------------------------
    # Moving averages
    # ------------------------------------------------------------

    df["MA_50"] = close.rolling(
        TREND_LOOKBACK_FAST,
        min_periods=TREND_LOOKBACK_FAST,
    ).mean()

    df["MA_200"] = close.rolling(
        TREND_LOOKBACK_SLOW,
        min_periods=TREND_LOOKBACK_SLOW,
    ).mean()

    # ------------------------------------------------------------
    # Trend strength
    # ------------------------------------------------------------

    df["Trend_Strength"] = (
        df["MA_50"] / df["MA_200"] - 1.0
    )

    # ------------------------------------------------------------
    # Daily returns
    # ------------------------------------------------------------

    df["Daily_Return"] = close.pct_change()

    # ------------------------------------------------------------
    # Rolling annualized volatility
    # ------------------------------------------------------------

    df["Volatility_20D"] = (
        df["Daily_Return"]
        .rolling(
            VOL_LOOKBACK,
            min_periods=VOL_LOOKBACK,
        )
        .std()
        * np.sqrt(252)
    )

    # ------------------------------------------------------------
    # Historical volatility median
    #
    # IMPORTANT:
    # expanding median means no future information is used.
    # ------------------------------------------------------------

    df["Volatility_Median"] = (
        df["Volatility_20D"]
        .expanding(
            min_periods=VOL_REGIME_LOOKBACK
        )
        .median()
    )

    # ------------------------------------------------------------
    # Trend classification
    # ------------------------------------------------------------

    df["Trend_Regime"] = np.where(
        df["Trend_Strength"] > 0.02,
        "Strong_Uptrend",
        np.where(
            df["Trend_Strength"] < -0.02,
            "Strong_Downtrend",
            np.where(
                df["Trend_Strength"] > 0,
                "Weak_Uptrend",
                "Weak_Downtrend",
            ),
        ),
    )

    # ------------------------------------------------------------
    # Volatility classification
    # ------------------------------------------------------------

    df["Vol_Regime"] = np.where(
        df["Volatility_20D"]
        > df["Volatility_Median"],
        "HighVol",
        "LowVol",
    )

    # ------------------------------------------------------------
    # Combine regimes
    # ------------------------------------------------------------

    def combine_regime(row):
        trend = row["Trend_Regime"]
        vol = row["Vol_Regime"]

        if trend == "Strong_Downtrend":
            return f"Strong_Downtrend_{vol}"

        if trend == "Strong_Uptrend":
            return f"Strong_Uptrend_{vol}"

        return trend

    df["Regime"] = df.apply(
        combine_regime,
        axis=1,
    )

    return df


# ================================================================
# ALIGN PREDICTIONS AND PRICES
# ================================================================

def align_data(predictions, price_df):
    print_header("ALIGNING PREDICTIONS WITH PRICES")

    df = predictions.merge(
        price_df,
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    print(f"Merged rows : {len(df)}")

    # ------------------------------------------------------------
    # Do NOT drop rows independently from signal arrays later.
    # Everything used by evaluate_strategy stays inside this df.
    # ------------------------------------------------------------

    usable_columns = [
        "Date",
        "Probability",
        "Close",
        "Regime",
        "Volatility_20D",
        "Trend_Strength",
    ]

    df = df.dropna(
        subset=usable_columns
    ).reset_index(drop=True)

    print(f"Usable rows : {len(df)}")

    if len(df) < MIN_REGIME_ROWS:
        print(
            "WARNING: usable rows are below recommended "
            f"minimum of {MIN_REGIME_ROWS}."
        )

    return df


# ================================================================
# REGIME MASK
# ================================================================

def regime_mask(df, regime_filter):
    """
    Return a boolean Series indexed exactly like df.

    This is the central fix for the V12 crash.

    The old implementation could create a shorter signal array
    and then access it using the original DataFrame position.

    Here every mask is generated directly from the same df.
    """

    if regime_filter == "All_Regimes":
        return pd.Series(
            True,
            index=df.index,
            dtype=bool,
        )

    if regime_filter == "Only_Best_Regime":
        return (
            df["Regime"] == BEST_REGIME
        ).astype(bool)

    if regime_filter == "Exclude_Best_Regime":
        return (
            df["Regime"] != BEST_REGIME
        ).astype(bool)

    if regime_filter == "Only_Downtrend":
        return (
            df["Regime"].isin(
                [
                    "Strong_Downtrend_HighVol",
                    "Strong_Downtrend_LowVol",
                    "Weak_Downtrend",
                ]
            )
        ).astype(bool)

    if regime_filter == "Only_Uptrend":
        return (
            df["Regime"].isin(
                [
                    "Strong_Uptrend_HighVol",
                    "Strong_Uptrend_LowVol",
                    "Weak_Uptrend",
                ]
            )
        ).astype(bool)

    if regime_filter == "Exclude_Bad_Downtrend":
        return (
            df["Regime"]
            != "Strong_Downtrend_LowVol"
        ).astype(bool)

    raise ValueError(
        f"Unknown regime filter: {regime_filter}"
    )


# ================================================================
# BACKTEST ENGINE
# ================================================================

def evaluate_strategy(
    df,
    threshold,
    holding_period,
    regime_filter,
):
    """
    Run a non-overlapping long-only backtest.

    Signal:
        Probability >= threshold

    Entry:
        At current day's close.

    Holding:
        holding_period trading sessions.

    Exit:
        At close after holding_period sessions.

    Costs:
        Entry and exit transaction cost + slippage.

    No future information is used for the entry decision.
    """

    data = df.copy().reset_index(drop=True)

    if len(data) < 2:
        return (
            {
                "Total_Return": np.nan,
                "CAGR": np.nan,
                "Sharpe": np.nan,
                "Sortino": np.nan,
                "Max_Drawdown": np.nan,
                "Win_Rate": np.nan,
                "Profit_Factor": np.nan,
                "Trades": 0,
                "Exposure": 0.0,
                "Buy_Hold_Return": np.nan,
                "Excess_Return": np.nan,
            },
            [],
            pd.DataFrame(),
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    # signal and regime mask are created on EXACT SAME df.
    # ------------------------------------------------------------

    signal = (
        data["Probability"]
        .astype(float)
        >= float(threshold)
    )

    allowed_regime = regime_mask(
        data,
        regime_filter,
    ).reset_index(drop=True)

    signal = (
        signal.reset_index(drop=True)
        & allowed_regime
    )

    prices = (
        data["Close"]
        .astype(float)
        .to_numpy()
    )

    dates = data["Date"].to_numpy()

    n = len(data)

    # Daily portfolio returns
    portfolio_returns = np.zeros(n)

    equity = np.zeros(n)
    equity[0] = INITIAL_CAPITAL

    trades = []

    i = 0

    while i < n - 1:

        # --------------------------------------------------------
        # No signal -> stay in cash.
        # --------------------------------------------------------

        if not bool(signal.iloc[i]):
            portfolio_returns[i] = 0.0
            i += 1
            continue

        entry_idx = i

        # Need enough rows for a complete holding period.
        exit_idx = min(
            entry_idx + int(holding_period),
            n - 1,
        )

        if exit_idx <= entry_idx:
            break

        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]

        if (
            not np.isfinite(entry_price)
            or not np.isfinite(exit_price)
            or entry_price <= 0
            or exit_price <= 0
        ):
            i += 1
            continue

        # --------------------------------------------------------
        # Gross trade return
        # --------------------------------------------------------

        gross_return = (
            exit_price / entry_price
        ) - 1.0

        # --------------------------------------------------------
        # Net trade return after entry + exit costs.
        # Approximation:
        # capital * (1-entry_cost)
        # then exit after price movement and exit cost.
        # --------------------------------------------------------

        net_growth = (
            (1.0 + gross_return)
            * (1.0 - TOTAL_ENTRY_COST)
            * (1.0 - TOTAL_EXIT_COST)
        )

        net_return = net_growth - 1.0

        # --------------------------------------------------------
        # Distribute the holding-period return across daily
        # portfolio returns using actual daily price movement.
        # This gives a realistic equity curve.
        # --------------------------------------------------------

        entry_growth_factor = (
            1.0 - TOTAL_ENTRY_COST
        )

        exit_growth_factor = (
            1.0 - TOTAL_EXIT_COST
        )

        for j in range(
            entry_idx + 1,
            exit_idx + 1,
        ):
            daily_market_return = (
                prices[j] / prices[j - 1]
            ) - 1.0

            portfolio_returns[j] = (
                daily_market_return
            )

        # Entry cost is charged on entry day.
        portfolio_returns[entry_idx] -= (
            TOTAL_ENTRY_COST
        )

        # Exit cost is charged on exit day.
        portfolio_returns[exit_idx] = (
            (1.0 + portfolio_returns[exit_idx])
            * exit_growth_factor
            - 1.0
        )

        trades.append(
            {
                "Entry_Date": pd.Timestamp(
                    dates[entry_idx]
                ),
                "Exit_Date": pd.Timestamp(
                    dates[exit_idx]
                ),
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Gross_Return": gross_return,
                "Net_Return": net_return,
                "Threshold": threshold,
                "Holding_Period": holding_period,
                "Regime": data.loc[
                    entry_idx,
                    "Regime",
                ],
                "Probability": data.loc[
                    entry_idx,
                    "Probability",
                ],
            }
        )

        # --------------------------------------------------------
        # Non-overlapping positions.
        # --------------------------------------------------------

        i = exit_idx + 1

    # ============================================================
    # EQUITY CURVE
    # ============================================================

    equity[0] = INITIAL_CAPITAL

    for t in range(1, n):
        daily_return = portfolio_returns[t]

        if not np.isfinite(daily_return):
            daily_return = 0.0

        equity[t] = (
            equity[t - 1]
            * (1.0 + daily_return)
        )

    equity_series = pd.Series(
        equity,
        index=data.index,
    )

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================

    final_equity = float(equity[-1])

    total_return = (
        final_equity
        / INITIAL_CAPITAL
    ) - 1.0

    days = max(
        (
            data["Date"].iloc[-1]
            - data["Date"].iloc[0]
        ).days,
        1,
    )

    years = days / 365.25

    if years > 0 and final_equity > 0:
        cagr = (
            final_equity
            / INITIAL_CAPITAL
        ) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    daily_series = pd.Series(
        portfolio_returns
    )

    daily_std = daily_series.std(
        ddof=1
    )

    if (
        np.isfinite(daily_std)
        and daily_std > 0
    ):
        sharpe = (
            daily_series.mean()
            / daily_std
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    downside = daily_series[
        daily_series < 0
    ]

    downside_std = downside.std(
        ddof=1
    )

    if (
        len(downside) > 1
        and np.isfinite(downside_std)
        and downside_std > 0
    ):
        sortino = (
            daily_series.mean()
            / downside_std
            * np.sqrt(252)
        )
    else:
        sortino = np.nan

    running_max = equity_series.cummax()

    drawdown = (
        equity_series / running_max
    ) - 1.0

    max_drawdown = float(
        drawdown.min()
    )

    # ============================================================
    # TRADE METRICS
    # ============================================================

    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0:

        trade_returns = trade_df[
            "Net_Return"
        ].astype(float)

        win_rate = float(
            (trade_returns > 0).mean()
        )

        gross_profit = trade_returns[
            trade_returns > 0
        ].sum()

        gross_loss = -trade_returns[
            trade_returns < 0
        ].sum()

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = np.inf

    else:
        win_rate = np.nan
        profit_factor = np.nan

    # ============================================================
    # EXPOSURE
    # ============================================================

    invested_days = np.sum(
        portfolio_returns != 0
    )

    exposure = (
        invested_days / n
    )

    # ============================================================
    # BUY & HOLD
    # ============================================================

    buy_hold_return = (
        prices[-1] / prices[0]
    ) - 1.0

    excess_return = (
        total_return
        - buy_hold_return
    )

    metrics = {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_drawdown,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Trades": len(trade_df),
        "Exposure": exposure,
        "Buy_Hold_Return": buy_hold_return,
        "Excess_Return": excess_return,
    }

    # ============================================================
    # EQUITY DATA
    # ============================================================

    equity_df = pd.DataFrame(
        {
            "Date": data["Date"],
            "Equity": equity,
            "Daily_Return": portfolio_returns,
            "Probability": data[
                "Probability"
            ].to_numpy(),
            "Regime": data[
                "Regime"
            ].to_numpy(),
            "Signal": signal.to_numpy(),
            "Allowed_Regime": allowed_regime.to_numpy(),
            "Threshold": threshold,
            "Holding_Period": holding_period,
            "Regime_Filter": regime_filter,
        }
    )

    return metrics, trades, equity_df


# ================================================================
# PRINT STRATEGY RESULT
# ================================================================

def print_strategy_result(
    regime_filter,
    threshold,
    holding_period,
    metrics,
):
    print(
        f"Threshold={threshold:.2f} | "
        f"Hold={holding_period:2d}D | "
        f"Return={fmt_pct(metrics['Total_Return'])} | "
        f"CAGR={fmt_pct(metrics['CAGR'])} | "
        f"Sharpe={metrics['Sharpe']:.3f} | "
        f"MaxDD={fmt_pct(metrics['Max_Drawdown'])} | "
        f"Trades={metrics['Trades']}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "ALPHALENS V12 ROBUSTNESS VALIDATION"
    )

    print(
        f"Model                  : {MODEL_NAME}"
    )

    print(
        f"Base threshold         : {BASE_THRESHOLD}"
    )

    print(
        f"Thresholds tested      : {THRESHOLDS}"
    )

    print(
        f"Holding periods        : {HOLDING_PERIODS}"
    )

    print(
        f"Transaction cost       : "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print(
        f"Slippage               : "
        f"{SLIPPAGE * 100:.3f}%"
    )

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------

    predictions = load_predictions()

    prices = load_price_data()

    # ------------------------------------------------------------
    # REGIME FEATURES
    # ------------------------------------------------------------

    prices = build_regime_features(
        prices
    )

    # ------------------------------------------------------------
    # ALIGN
    # ------------------------------------------------------------

    df = align_data(
        predictions,
        prices,
    )

    if len(df) == 0:
        raise ValueError(
            "No usable rows after alignment."
        )

    # ------------------------------------------------------------
    # REGIME DISTRIBUTION
    # ------------------------------------------------------------

    print_header(
        "REGIME DISTRIBUTION"
    )

    print(
        df["Regime"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------------
    # REGIME FILTERS
    # ------------------------------------------------------------

    regime_filters = [
        "All_Regimes",
        "Only_Best_Regime",
        "Exclude_Best_Regime",
        "Only_Downtrend",
        "Only_Uptrend",
        "Exclude_Bad_Downtrend",
    ]

    # ------------------------------------------------------------
    # RUN TESTS
    # ------------------------------------------------------------

    print_header(
        "RUNNING V12 ROBUSTNESS TESTS"
    )

    results = []

    all_trades = []

    all_equity = []

    for regime_filter in regime_filters:

        print()
        print("-" * 70)
        print(
            f"REGIME FILTER: {regime_filter}"
        )
        print("-" * 70)

        for threshold in THRESHOLDS:

            for holding_period in HOLDING_PERIODS:

                metrics, trades, equity_df = (
                    evaluate_strategy(
                        df=df,
                        threshold=threshold,
                        holding_period=holding_period,
                        regime_filter=regime_filter,
                    )
                )

                print_strategy_result(
                    regime_filter,
                    threshold,
                    holding_period,
                    metrics,
                )

                row = {
                    "Regime_Filter": regime_filter,
                    "Threshold": threshold,
                    "Holding_Period": holding_period,
                    **metrics,
                }

                results.append(row)

                # ------------------------------------------------
                # Store trades
                # ------------------------------------------------

                for trade in trades:
                    trade_record = {
                        "Regime_Filter": regime_filter,
                        **trade,
                    }

                    all_trades.append(
                        trade_record
                    )

                # ------------------------------------------------
                # Store equity
                # ------------------------------------------------

                if len(equity_df) > 0:
                    all_equity.append(
                        equity_df
                    )

    # ============================================================
    # RESULTS DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) == 0:
        raise RuntimeError(
            "No robustness results were generated."
        )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print_header(
        "V12 ROBUSTNESS RESULTS"
    )

    display_columns = [
        "Regime_Filter",
        "Threshold",
        "Holding_Period",
        "Total_Return",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Max_Drawdown",
        "Trades",
        "Exposure",
        "Excess_Return",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    # ============================================================
    # BEST RESULTS
    # ============================================================

    print_header(
        "V12 BEST CANDIDATES"
    )

    # ------------------------------------------------------------
    # Best total return
    # ------------------------------------------------------------

    best_return_idx = (
        results_df[
            "Total_Return"
        ].idxmax()
    )

    best_return = results_df.loc[
        best_return_idx
    ]

    print(
        "Best by total return:"
    )

    print(
        f"Regime filter : "
        f"{best_return['Regime_Filter']}"
    )

    print(
        f"Threshold     : "
        f"{best_return['Threshold']:.2f}"
    )

    print(
        f"Holding       : "
        f"{int(best_return['Holding_Period'])}D"
    )

    print(
        f"Return        : "
        f"{fmt_pct(best_return['Total_Return'])}"
    )

    # ------------------------------------------------------------
    # Best Sharpe
    # ------------------------------------------------------------

    sharpe_df = results_df.dropna(
        subset=["Sharpe"]
    )

    if len(sharpe_df) > 0:

        best_sharpe_idx = (
            sharpe_df["Sharpe"].idxmax()
        )

        best_sharpe = (
            sharpe_df.loc[
                best_sharpe_idx
            ]
        )

        print()
        print(
            "Best by Sharpe:"
        )

        print(
            f"Regime filter : "
            f"{best_sharpe['Regime_Filter']}"
        )

        print(
            f"Threshold     : "
            f"{best_sharpe['Threshold']:.2f}"
        )

        print(
            f"Holding       : "
            f"{int(best_sharpe['Holding_Period'])}D"
        )

        print(
            f"Sharpe        : "
            f"{best_sharpe['Sharpe']:.4f}"
        )

    # ------------------------------------------------------------
    # Lowest drawdown
    # ------------------------------------------------------------

    drawdown_df = results_df.dropna(
        subset=["Max_Drawdown"]
    )

    if len(drawdown_df) > 0:

        best_dd_idx = (
            drawdown_df[
                "Max_Drawdown"
            ].idxmax()
        )

        best_dd = drawdown_df.loc[
            best_dd_idx
        ]

        print()
        print(
            "Best by drawdown:"
        )

        print(
            f"Regime filter : "
            f"{best_dd['Regime_Filter']}"
        )

        print(
            f"Threshold     : "
            f"{best_dd['Threshold']:.2f}"
        )

        print(
            f"Holding       : "
            f"{int(best_dd['Holding_Period'])}D"
        )

        print(
            f"Max drawdown  : "
            f"{fmt_pct(best_dd['Max_Drawdown'])}"
        )

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================

    print_header(
        "V12 BASELINE COMPARISON"
    )

    baseline = results_df[
        (
            results_df[
                "Regime_Filter"
            ] == "All_Regimes"
        )
        &
        (
            results_df[
                "Threshold"
            ] == BASE_THRESHOLD
        )
        &
        (
            results_df[
                "Holding_Period"
            ] == 10
        )
    ]

    if len(baseline) > 0:

        baseline_row = (
            baseline.iloc[0]
        )

        print(
            f"Baseline threshold : "
            f"{BASE_THRESHOLD:.2f}"
        )

        print(
            "Baseline holding   : 10D"
        )

        print(
            f"Baseline return    : "
            f"{fmt_pct(baseline_row['Total_Return'])}"
        )

        print(
            f"Baseline Sharpe    : "
            f"{baseline_row['Sharpe']:.4f}"
        )

        print(
            f"Baseline MaxDD     : "
            f"{fmt_pct(baseline_row['Max_Drawdown'])}"
        )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print_header(
        "SAVING V12 OUTPUTS"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    # ------------------------------------------------------------
    # Save trades
    # ------------------------------------------------------------

    if all_trades:

        trades_df = pd.DataFrame(
            all_trades
        )

        trades_df.to_csv(
            OUTPUT_TRADES,
            index=False,
        )

    else:

        trades_df = pd.DataFrame()

        trades_df.to_csv(
            OUTPUT_TRADES,
            index=False,
        )

    # ------------------------------------------------------------
    # Save equity
    # ------------------------------------------------------------

    if all_equity:

        equity_df = pd.concat(
            all_equity,
            ignore_index=True,
        )

        equity_df.to_csv(
            OUTPUT_EQUITY,
            index=False,
        )

    else:

        equity_df = pd.DataFrame()

        equity_df.to_csv(
            OUTPUT_EQUITY,
            index=False,
        )

    print(
        f"Results : {OUTPUT_RESULTS}"
    )

    print(
        f"Trades  : {OUTPUT_TRADES}"
    )

    print(
        f"Equity  : {OUTPUT_EQUITY}"
    )

    # ============================================================
    # INTERPRETATION
    # ============================================================

    print_header(
        "V12 INTERPRETATION"
    )

    print(
        "V12 tests threshold and holding-period robustness "
        "using the V6 out-of-sample XGBoost predictions."
    )

    print(
        "The V6 predictions remain completely untouched."
    )

    print(
        "No model retraining occurs in V12."
    )

    print(
        "Market regimes are constructed from historical/current "
        "price information only."
    )

    print(
        "Regime-filtered signals are evaluated on the same "
        "aligned rows as prices and dates."
    )

    # ------------------------------------------------------------
    # Determine whether any positive-return result exists
    # ------------------------------------------------------------

    positive_results = results_df[
        results_df["Total_Return"] > 0
    ]

    positive_sharpe = results_df[
        results_df["Sharpe"] > 0
    ]

    print()

    if len(positive_results) > 0:

        print(
            f"RESULT: {len(positive_results)} "
            "parameter combinations produced positive "
            "historical total return."
        )

    else:

        print(
            "RESULT: No tested parameter combination "
            "produced positive historical total return."
        )

    if len(positive_sharpe) > 0:

        print(
            f"RESULT: {len(positive_sharpe)} "
            "parameter combinations produced positive Sharpe."
        )

    else:

        print(
            "RESULT: No tested parameter combination "
            "produced positive Sharpe."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A positive historical result does not establish "
        "future profitability."
    )

    print(
        "Threshold and regime selection can introduce "
        "research-selection bias."
    )

    print(
        "Further validation should test unseen periods, "
        "parameter stability and realistic execution."
    )

    print()
    print(
        "PASS: V12 robustness validation completed successfully."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()