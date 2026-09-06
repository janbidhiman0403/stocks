# ================================================================
# ALPHALENS V12 ROBUSTNESS VALIDATION
# ================================================================
#
# Purpose:
#   Validate robustness of the existing V6 XGBoost predictions
#   across:
#       - prediction thresholds
#       - holding periods
#       - market regimes
#
# IMPORTANT:
#   - V6 predictions are NOT modified.
#   - No model retraining occurs.
#   - This script uses the existing:
#         data/model_v6_predictions.csv
#   - Only rows with Model == XGBoost are evaluated.
#   - Regime features use historical/current price information.
#   - Signal arrays remain aligned with the evaluation DataFrame.
#   - Entry is made on the NEXT available trading row after
#     the prediction date to avoid same-close execution bias.
#
# INPUT:
#   data/model_v6_predictions.csv
#   data/ml_features_v4.csv
#
# OUTPUT:
#   data/v12_robustness_results.csv
#   data/v12_robustness_equity.csv
#   data/v12_robustness_trades.csv
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

# Existing repository files
PREDICTIONS_FILE = DATA_DIR / "model_v6_predictions.csv"
PRICE_FILE = DATA_DIR / "ml_features_v4.csv"

# V12 outputs
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

ENTRY_COST = TRANSACTION_COST + SLIPPAGE
EXIT_COST = TRANSACTION_COST + SLIPPAGE

BEST_REGIME = "Strong_Downtrend_HighVol"

TREND_LOOKBACK_FAST = 50
TREND_LOOKBACK_SLOW = 200

VOL_LOOKBACK = 20
VOL_REGIME_LOOKBACK = 252

MIN_REGIME_ROWS = 200


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_header(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_pct(value):
    if pd.isna(value):
        return "nan"
    return f"{value * 100:.2f}%"


# ================================================================
# LOAD V6 PREDICTIONS
# ================================================================

def load_predictions():

    print_header("LOADING V6 XGBOOST PREDICTIONS")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing prediction file:\n{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "Date",
        "Actual",
        "Probability_Up",
        "Prediction_0.50",
        "Model",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "model_v6_predictions.csv is missing columns: "
            f"{missing}"
        )

    # Keep only the XGBoost predictions
    df = df[
        df["Model"].astype(str).str.strip().str.lower()
        == MODEL_NAME.lower()
    ].copy()

    if df.empty:
        raise ValueError(
            "No XGBoost rows were found in "
            "model_v6_predictions.csv."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Probability"] = pd.to_numeric(
        df["Probability_Up"],
        errors="coerce",
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Probability",
        ]
    )

    # Multiple model records on the same date are not allowed
    # to create duplicate strategy signals.
    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No usable XGBoost prediction rows remain."
        )

    print(f"Prediction file : {PREDICTIONS_FILE}")
    print(f"Model           : {MODEL_NAME}")
    print(f"Prediction rows : {len(df)}")
    print(
        "Prediction range: "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    return df


# ================================================================
# LOAD PRICE DATA
# ================================================================

def load_price_data():

    print_header("LOADING PRICE DATA")

    if not PRICE_FILE.exists():
        raise FileNotFoundError(
            f"Missing price file:\n{PRICE_FILE}"
        )

    df = pd.read_csv(PRICE_FILE)

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

    if date_col is None:
        raise ValueError(
            "No Date column found in price file."
        )

    if close_col is None:
        raise ValueError(
            "No Close/Price column found in price file."
        )

    df = df[
        [date_col, close_col]
    ].copy()

    df.columns = [
        "Date",
        "Close",
    ]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Close",
        ]
    )

    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df = df[
        df["Close"] > 0
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No usable price rows found."
        )

    print(f"Price file      : {PRICE_FILE}")
    print(f"Price rows      : {len(df)}")
    print(
        "Price range     : "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    return df


# ================================================================
# BUILD MARKET REGIMES
# ================================================================

def build_regime_features(price_df):

    print_header("BUILDING MARKET REGIME FEATURES")

    df = price_df.copy()

    close = df["Close"]

    # ------------------------------------------------------------
    # Trend indicators
    # ------------------------------------------------------------

    df["MA_50"] = close.rolling(
        TREND_LOOKBACK_FAST,
        min_periods=TREND_LOOKBACK_FAST,
    ).mean()

    df["MA_200"] = close.rolling(
        TREND_LOOKBACK_SLOW,
        min_periods=TREND_LOOKBACK_SLOW,
    ).mean()

    df["Trend_Strength"] = (
        df["MA_50"] / df["MA_200"]
    ) - 1.0

    # ------------------------------------------------------------
    # Daily return
    # ------------------------------------------------------------

    df["Daily_Return"] = close.pct_change()

    # ------------------------------------------------------------
    # 20-day annualized volatility
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
    # Expanding historical volatility median
    #
    # This uses only information available up to the current row.
    # ------------------------------------------------------------

    df["Volatility_Median"] = (
        df["Volatility_20D"]
        .expanding(
            min_periods=VOL_REGIME_LOOKBACK
        )
        .median()
    )

    # ------------------------------------------------------------
    # Trend regime
    # ------------------------------------------------------------

    df["Trend_Regime"] = np.select(
        [
            df["Trend_Strength"] > 0.02,
            df["Trend_Strength"] < -0.02,
            df["Trend_Strength"] > 0,
        ],
        [
            "Strong_Uptrend",
            "Strong_Downtrend",
            "Weak_Uptrend",
        ],
        default="Weak_Downtrend",
    )

    # ------------------------------------------------------------
    # Volatility regime
    # ------------------------------------------------------------

    df["Vol_Regime"] = np.where(
        df["Volatility_20D"]
        > df["Volatility_Median"],
        "HighVol",
        "LowVol",
    )

    # ------------------------------------------------------------
    # Combined regime
    # ------------------------------------------------------------

    df["Regime"] = np.where(
        df["Trend_Regime"] == "Strong_Downtrend",
        "Strong_Downtrend_" + df["Vol_Regime"],
        np.where(
            df["Trend_Regime"] == "Strong_Uptrend",
            "Strong_Uptrend_" + df["Vol_Regime"],
            df["Trend_Regime"],
        ),
    )

    return df


# ================================================================
# ALIGN DATA
# ================================================================

def align_data(predictions, price_df):

    print_header("ALIGNING PREDICTIONS AND PRICES")

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

    print(f"Prediction rows : {len(predictions)}")
    print(f"Aligned rows    : {len(df)}")

    required = [
        "Date",
        "Probability",
        "Close",
        "Regime",
        "Volatility_20D",
        "Trend_Strength",
    ]

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    print(f"Usable rows     : {len(df)}")

    if len(df) < MIN_REGIME_ROWS:
        print(
            "WARNING: fewer than "
            f"{MIN_REGIME_ROWS} usable rows are available "
            "for regime analysis."
        )

    if df.empty:
        raise ValueError(
            "No usable rows remain after alignment."
        )

    return df


# ================================================================
# REGIME FILTER
# ================================================================

def regime_mask(df, regime_filter):

    if regime_filter == "All_Regimes":

        return pd.Series(
            True,
            index=df.index,
            dtype=bool,
        )

    if regime_filter == "Only_Best_Regime":

        return (
            df["Regime"]
            == BEST_REGIME
        )

    if regime_filter == "Exclude_Best_Regime":

        return (
            df["Regime"]
            != BEST_REGIME
        )

    if regime_filter == "Only_Downtrend":

        return df["Regime"].isin(
            [
                "Strong_Downtrend_HighVol",
                "Strong_Downtrend_LowVol",
                "Weak_Downtrend",
            ]
        )

    if regime_filter == "Only_Uptrend":

        return df["Regime"].isin(
            [
                "Strong_Uptrend_HighVol",
                "Strong_Uptrend_LowVol",
                "Weak_Uptrend",
            ]
        )

    if regime_filter == "Exclude_Bad_Downtrend":

        return (
            df["Regime"]
            != "Strong_Downtrend_LowVol"
        )

    raise ValueError(
        f"Unknown regime filter: {regime_filter}"
    )


# ================================================================
# PERFORMANCE METRICS
# ================================================================

def calculate_metrics(
    data,
    equity,
    daily_returns,
    trades,
    position_mask,
):

    initial = INITIAL_CAPITAL
    final_equity = float(equity[-1])

    total_return = (
        final_equity / initial
    ) - 1.0

    days = max(
        (
            data["Date"].iloc[-1]
            - data["Date"].iloc[0]
        ).days,
        1,
    )

    years = days / 365.25

    if (
        years > 0
        and final_equity > 0
    ):
        cagr = (
            final_equity / initial
        ) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    daily = pd.Series(
        daily_returns,
        dtype=float,
    )

    daily_std = daily.std(
        ddof=1
    )

    if (
        np.isfinite(daily_std)
        and daily_std > 0
    ):
        sharpe = (
            daily.mean()
            / daily_std
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    negative_returns = daily[
        daily < 0
    ]

    downside_std = negative_returns.std(
        ddof=1
    )

    if (
        len(negative_returns) > 1
        and np.isfinite(downside_std)
        and downside_std > 0
    ):
        sortino = (
            daily.mean()
            / downside_std
            * np.sqrt(252)
        )
    else:
        sortino = np.nan

    equity_series = pd.Series(
        equity
    )

    running_max = (
        equity_series.cummax()
    )

    drawdown = (
        equity_series
        / running_max
    ) - 1.0

    max_drawdown = float(
        drawdown.min()
    )

    trade_df = pd.DataFrame(
        trades
    )

    if not trade_df.empty:

        trade_returns = (
            trade_df["Net_Return"]
            .astype(float)
        )

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
        elif gross_profit > 0:
            profit_factor = np.inf
        else:
            profit_factor = np.nan

    else:

        win_rate = np.nan
        profit_factor = np.nan

    exposure = (
        np.mean(position_mask)
        if len(position_mask) > 0
        else 0.0
    )

    prices = (
        data["Close"]
        .astype(float)
        .to_numpy()
    )

    buy_hold_return = (
        prices[-1] / prices[0]
    ) - 1.0

    excess_return = (
        total_return
        - buy_hold_return
    )

    return {
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


# ================================================================
# BACKTEST ENGINE
# ================================================================

def evaluate_strategy(
    df,
    threshold,
    holding_period,
    regime_filter,
):

    data = (
        df.copy()
        .reset_index(drop=True)
    )

    n = len(data)

    if n < 2:
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
    # Signal is created from the prediction date.
    #
    # Entry happens on the NEXT available price row.
    # This avoids assuming that a prediction based on today's
    # information can be executed at today's closing price.
    # ------------------------------------------------------------

    signal = (
        data["Probability"]
        .astype(float)
        >= float(threshold)
    )

    allowed_regime = (
        regime_mask(
            data,
            regime_filter,
        )
        .reset_index(drop=True)
        .astype(bool)
    )

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

    # ------------------------------------------------------------
    # Equity and return arrays
    # ------------------------------------------------------------

    equity = np.zeros(n)
    equity[0] = INITIAL_CAPITAL

    daily_returns = np.zeros(n)

    position_mask = np.zeros(
        n,
        dtype=bool,
    )

    trades = []

    i = 0

    while i < n - 1:

        # --------------------------------------------------------
        # No signal -> remain in cash.
        # --------------------------------------------------------

        if not bool(signal.iloc[i]):
            i += 1
            continue

        # --------------------------------------------------------
        # Prediction date = i
        # Entry = next available trading row
        # --------------------------------------------------------

        entry_idx = i + 1

        if entry_idx >= n:
            break

        exit_idx = (
            entry_idx
            + int(holding_period)
        )

        if exit_idx >= n:
            # Incomplete final trade is not opened.
            break

        entry_price = prices[
            entry_idx
        ]

        exit_price = prices[
            exit_idx
        ]

        if (
            not np.isfinite(entry_price)
            or not np.isfinite(exit_price)
            or entry_price <= 0
            or exit_price <= 0
        ):
            i += 1
            continue

        # --------------------------------------------------------
        # Mark position exposure.
        # --------------------------------------------------------

        position_mask[
            entry_idx:
            exit_idx + 1
        ] = True

        # --------------------------------------------------------
        # Gross trade return.
        # --------------------------------------------------------

        gross_return = (
            exit_price / entry_price
        ) - 1.0

        # --------------------------------------------------------
        # Net trade return after costs.
        # --------------------------------------------------------

        net_growth = (
            (1.0 + gross_return)
            * (1.0 - ENTRY_COST)
            * (1.0 - EXIT_COST)
        )

        net_return = (
            net_growth - 1.0
        )

        # --------------------------------------------------------
        # Store trade.
        # --------------------------------------------------------

        trades.append(
            {
                "Prediction_Date": pd.Timestamp(
                    dates[i]
                ),
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
                    i,
                    "Regime",
                ],
                "Probability": data.loc[
                    i,
                    "Probability",
                ],
            }
        )

        # --------------------------------------------------------
        # Rebuild daily portfolio returns.
        #
        # Entry cost is applied on entry.
        # Market return is applied on each held day.
        # Exit cost is applied on exit.
        # --------------------------------------------------------

        daily_returns[
            entry_idx
        ] -= ENTRY_COST

        for j in range(
            entry_idx + 1,
            exit_idx + 1,
        ):

            market_return = (
                prices[j]
                / prices[j - 1]
            ) - 1.0

            daily_returns[j] = (
                (1.0 + daily_returns[j])
                * (1.0 + market_return)
                - 1.0
            )

        daily_returns[
            exit_idx
        ] = (
            (1.0 + daily_returns[exit_idx])
            * (1.0 - EXIT_COST)
            - 1.0
        )

        # --------------------------------------------------------
        # Non-overlapping trades.
        # --------------------------------------------------------

        i = exit_idx + 1

    # ============================================================
    # EQUITY CURVE
    # ============================================================

    for t in range(1, n):

        r = daily_returns[t]

        if not np.isfinite(r):
            r = 0.0

        equity[t] = (
            equity[t - 1]
            * (1.0 + r)
        )

    # ============================================================
    # METRICS
    # ============================================================

    metrics = calculate_metrics(
        data=data,
        equity=equity,
        daily_returns=daily_returns,
        trades=trades,
        position_mask=position_mask,
    )

    # ============================================================
    # EQUITY OUTPUT
    # ============================================================

    equity_df = pd.DataFrame(
        {
            "Date": data["Date"],
            "Equity": equity,
            "Daily_Return": daily_returns,
            "Probability": data[
                "Probability"
            ].to_numpy(),
            "Regime": data[
                "Regime"
            ].to_numpy(),
            "Signal": signal.to_numpy(),
            "Allowed_Regime": allowed_regime.to_numpy(),
            "Position": position_mask,
            "Threshold": threshold,
            "Holding_Period": holding_period,
            "Regime_Filter": regime_filter,
        }
    )

    return (
        metrics,
        trades,
        equity_df,
    )


# ================================================================
# PRINT RESULT
# ================================================================

def print_strategy_result(
    regime_filter,
    threshold,
    holding_period,
    metrics,
):

    sharpe = metrics["Sharpe"]

    sharpe_text = (
        f"{sharpe:.3f}"
        if np.isfinite(sharpe)
        else "nan"
    )

    print(
        f"{regime_filter:26s} | "
        f"Threshold={threshold:.2f} | "
        f"Hold={holding_period:2d}D | "
        f"Return={fmt_pct(metrics['Total_Return']):>9s} | "
        f"CAGR={fmt_pct(metrics['CAGR']):>9s} | "
        f"Sharpe={sharpe_text:>7s} | "
        f"MaxDD={fmt_pct(metrics['Max_Drawdown']):>9s} | "
        f"Trades={metrics['Trades']:3d}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "ALPHALENS V12 ROBUSTNESS VALIDATION"
    )

    print(
        "This version uses the existing repository V6 "
        "prediction file."
    )

    print(
        f"Prediction source : {PREDICTIONS_FILE}"
    )

    print(
        f"Price source      : {PRICE_FILE}"
    )

    print(
        f"Model             : {MODEL_NAME}"
    )

    print(
        f"Base threshold    : {BASE_THRESHOLD:.2f}"
    )

    print(
        f"Thresholds tested : {THRESHOLDS}"
    )

    print(
        f"Holding periods   : {HOLDING_PERIODS}"
    )

    print(
        f"Entry cost        : {ENTRY_COST * 100:.3f}%"
    )

    print(
        f"Exit cost         : {EXIT_COST * 100:.3f}%"
    )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    predictions = load_predictions()

    prices = load_price_data()

    # ------------------------------------------------------------
    # Regimes
    # ------------------------------------------------------------

    prices = build_regime_features(
        prices
    )

    # ------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------

    df = align_data(
        predictions,
        prices,
    )

    # ------------------------------------------------------------
    # Regime distribution
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
    # Filters
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
    # Run tests
    # ------------------------------------------------------------

    print_header(
        "RUNNING V12 ROBUSTNESS TESTS"
    )

    results = []
    all_trades = []
    all_equity = []

    for regime_filter in regime_filters:

        print()
        print("-" * 72)
        print(
            f"REGIME FILTER: {regime_filter}"
        )
        print("-" * 72)

        for threshold in THRESHOLDS:

            for holding_period in HOLDING_PERIODS:

                (
                    metrics,
                    trades,
                    equity_df,
                ) = evaluate_strategy(
                    df=df,
                    threshold=threshold,
                    holding_period=holding_period,
                    regime_filter=regime_filter,
                )

                print_strategy_result(
                    regime_filter,
                    threshold,
                    holding_period,
                    metrics,
                )

                results.append(
                    {
                        "Regime_Filter": regime_filter,
                        "Threshold": threshold,
                        "Holding_Period": holding_period,
                        **metrics,
                    }
                )

                for trade in trades:

                    all_trades.append(
                        {
                            "Regime_Filter":
                                regime_filter,
                            **trade,
                        }
                    )

                if not equity_df.empty:

                    all_equity.append(
                        equity_df
                    )

    # ============================================================
    # RESULTS DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No robustness results were generated."
        )

    # ============================================================
    # PRINT SUMMARY
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
    # BEST CANDIDATES
    # ============================================================

    print_header(
        "V12 BEST CANDIDATES"
    )

    # ------------------------------------------------------------
    # Best total return
    # ------------------------------------------------------------

    valid_return = results_df[
        results_df["Total_Return"].notna()
    ]

    if not valid_return.empty:

        best = valid_return.loc[
            valid_return[
                "Total_Return"
            ].idxmax()
        ]

        print(
            "Best by total return:"
        )

        print(
            f"  Regime filter : "
            f"{best['Regime_Filter']}"
        )

        print(
            f"  Threshold     : "
            f"{best['Threshold']:.2f}"
        )

        print(
            f"  Holding       : "
            f"{int(best['Holding_Period'])}D"
        )

        print(
            f"  Return        : "
            f"{fmt_pct(best['Total_Return'])}"
        )

    # ------------------------------------------------------------
    # Best Sharpe
    # ------------------------------------------------------------

    valid_sharpe = results_df[
        results_df["Sharpe"].notna()
    ]

    if not valid_sharpe.empty:

        best = valid_sharpe.loc[
            valid_sharpe[
                "Sharpe"
            ].idxmax()
        ]

        print()
        print(
            "Best by Sharpe:"
        )

        print(
            f"  Regime filter : "
            f"{best['Regime_Filter']}"
        )

        print(
            f"  Threshold     : "
            f"{best['Threshold']:.2f}"
        )

        print(
            f"  Holding       : "
            f"{int(best['Holding_Period'])}D"
        )

        print(
            f"  Sharpe        : "
            f"{best['Sharpe']:.4f}"
        )

    # ------------------------------------------------------------
    # Best drawdown
    #
    # Max_Drawdown is negative.
    # Therefore the largest value is the least negative drawdown.
    # ------------------------------------------------------------

    valid_dd = results_df[
        results_df["Max_Drawdown"].notna()
    ]

    if not valid_dd.empty:

        best = valid_dd.loc[
            valid_dd[
                "Max_Drawdown"
            ].idxmax()
        ]

        print()
        print(
            "Best by drawdown:"
        )

        print(
            f"  Regime filter : "
            f"{best['Regime_Filter']}"
        )

        print(
            f"  Threshold     : "
            f"{best['Threshold']:.2f}"
        )

        print(
            f"  Holding       : "
            f"{int(best['Holding_Period'])}D"
        )

        print(
            f"  Max drawdown  : "
            f"{fmt_pct(best['Max_Drawdown'])}"
        )

    # ============================================================
    # BASELINE
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

    if not baseline.empty:

        baseline_row = baseline.iloc[0]

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
    # SAVE OUTPUTS
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

    trades_df = pd.DataFrame(
        all_trades
    )

    trades_df.to_csv(
        OUTPUT_TRADES,
        index=False,
    )

    if all_equity:

        equity_df = pd.concat(
            all_equity,
            ignore_index=True,
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
        "V12 evaluates the existing V6 XGBoost predictions "
        "under multiple thresholds, holding periods and "
        "market-regime filters."
    )

    print(
        "The original V6 prediction probabilities are not "
        "modified or retrained by V12."
    )

    print(
        "Regime features are constructed from the historical "
        "price series."
    )

    print(
        "Signals are aligned to the same rows used for "
        "price and regime evaluation."
    )

    print(
        "A prediction is acted upon on the next available "
        "trading row rather than assuming same-close execution."
    )

    positive_results = results_df[
        results_df["Total_Return"] > 0
    ]

    positive_sharpe = results_df[
        results_df["Sharpe"] > 0
    ]

    print()

    print(
        f"Positive-return combinations : "
        f"{len(positive_results)} / {len(results_df)}"
    )

    print(
        f"Positive-Sharpe combinations : "
        f"{len(positive_sharpe)} / {len(results_df)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The best-performing parameter combination is "
        "descriptive rather than an independently validated "
        "future strategy."
    )

    print(
        "Testing many thresholds, holding periods and regimes "
        "can introduce research-selection bias."
    )

    print(
        "V12 should therefore be presented as a robustness and "
        "sensitivity analysis, not as proof of future profitability."
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
