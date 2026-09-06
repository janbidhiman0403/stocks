import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ======================================================================
# ALPHALENS V9 RISK & POSITION-SIZING BACKTEST
# ======================================================================

DATA_DIR = "data"

PRED_FILE = os.path.join(
    DATA_DIR,
    "walk_forward_v6_predictions.csv"
)

RESULT_FILE = os.path.join(
    DATA_DIR,
    "v9_risk_results.csv"
)

TRADE_FILE = os.path.join(
    DATA_DIR,
    "v9_risk_trades.csv"
)

EQUITY_FILE = os.path.join(
    DATA_DIR,
    "v9_risk_equity.csv"
)

INITIAL_CAPITAL = 100000.0

TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

# Selected from V8.
MODEL = "XGBoost"
THRESHOLD = 0.70

# Position sizing tests.
FIXED_SIZES = [
    1.00,
    0.75,
    0.50,
    0.25,
]

# Confidence-based sizing.
CONFIDENCE_FLOOR = 0.70
CONFIDENCE_CEILING = 1.00
MAX_CONFIDENCE_POSITION = 1.00

# Volatility targeting.
VOL_TARGET = 0.15
VOL_LOOKBACK = 20
MAX_VOL_POSITION = 1.00


def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def load_predictions():
    header("LOADING V6 OUT-OF-SAMPLE PREDICTIONS")

    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"Missing prediction file: {PRED_FILE}"
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
            f"Missing prediction columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Probability"] = pd.to_numeric(
        df["Probability"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Date",
            "Probability",
        ]
    )

    df = df[
        df["Model"] == MODEL
    ].copy()

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        f"Model           : {MODEL}"
    )
    print(
        f"Prediction rows  : {len(df)}"
    )
    print(
        f"Date range       : "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
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

    for path in candidates:
        if os.path.exists(path):
            price_file = path
            break

    if price_file is None:
        raise FileNotFoundError(
            "Could not locate a price CSV."
        )

    df = pd.read_csv(price_file)

    required = [
        "Date",
        "Close",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Price file missing columns: {missing}"
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
    )

    df = df.drop_duplicates(
        "Date"
    )

    print(
        f"Price rows      : {len(df)}"
    )
    print(
        f"Price date range: "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    return df


def prepare_dataset(predictions, prices):
    header("ALIGNING PREDICTIONS WITH PRICES")

    df = predictions.merge(
        prices,
        on="Date",
        how="left"
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    df["Market_Return"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1.0
    )

    # Historical volatility used only for sizing.
    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(VOL_LOOKBACK)
        .std()
        * np.sqrt(252)
    )

    df = df.dropna(
        subset=[
            "Close",
            "Market_Return",
        ]
    ).reset_index(drop=True)

    print(
        f"Usable rows: {len(df)}"
    )

    return df


def confidence_position(probability):
    if probability < CONFIDENCE_FLOOR:
        return 0.0

    if probability >= CONFIDENCE_CEILING:
        return MAX_CONFIDENCE_POSITION

    position = (
        probability - CONFIDENCE_FLOOR
    ) / (
        CONFIDENCE_CEILING
        - CONFIDENCE_FLOOR
    )

    return min(
        max(position, 0.0),
        MAX_CONFIDENCE_POSITION
    )


def volatility_position(volatility):
    if (
        not np.isfinite(volatility)
        or volatility <= 0
    ):
        return MAX_VOL_POSITION

    position = (
        VOL_TARGET / volatility
    )

    return min(
        max(position, 0.0),
        MAX_VOL_POSITION
    )


def run_strategy(
    df,
    strategy_name,
    sizing_function,
):
    capital = INITIAL_CAPITAL

    previous_position = 0.0

    entry_date = None
    entry_price = None

    trade_records = []

    equity_records = []

    for _, row in df.iterrows():

        date = row["Date"]
        close = float(row["Close"])
        probability = float(row["Probability"])
        market_return = float(
            row["Market_Return"]
        )

        position = sizing_function(row)

        # Only take long positions when probability
        # is above the V8-selected threshold.
        if probability < THRESHOLD:
            position = 0.0

        position = min(
            max(float(position), 0.0),
            1.0
        )

        turnover = abs(
            position - previous_position
        )

        trading_cost = turnover * (
            TRANSACTION_COST
            + SLIPPAGE
        )

        strategy_return = (
            position * market_return
        )

        net_return = (
            strategy_return
            - trading_cost
        )

        capital *= (
            1.0 + net_return
        )

        # --------------------------------------------------------------
        # ENTRY
        # --------------------------------------------------------------

        if (
            previous_position == 0.0
            and position > 0.0
        ):
            entry_date = date
            entry_price = close

        # --------------------------------------------------------------
        # EXIT
        # --------------------------------------------------------------

        if (
            previous_position > 0.0
            and position == 0.0
            and entry_price is not None
        ):
            trade_return = (
                close / entry_price
                - 1.0
            )

            trade_records.append({
                "Strategy": strategy_name,
                "Entry_Date": entry_date,
                "Exit_Date": date,
                "Entry_Price": entry_price,
                "Exit_Price": close,
                "Trade_Return": trade_return,
            })

            entry_date = None
            entry_price = None

        equity_records.append({
            "Strategy": strategy_name,
            "Date": date,
            "Equity": capital,
            "Position": position,
            "Probability": probability,
            "Market_Return": market_return,
            "Trading_Cost": trading_cost,
        })

        previous_position = position

    equity = pd.DataFrame(
        equity_records
    )

    trades = pd.DataFrame(
        trade_records
    )

    if equity.empty:
        return None, trades

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

    downside = daily_returns[
        daily_returns < 0
    ]

    if len(downside) > 0:
        downside_std = downside.std()

        if downside_std > 0:
            sortino = (
                daily_returns.mean()
                / downside_std
                * np.sqrt(252)
            )
        else:
            sortino = np.nan
    else:
        sortino = np.nan

    if len(trades) > 0:

        wins = trades[
            trades["Trade_Return"] > 0
        ]

        losses = trades[
            trades["Trade_Return"] < 0
        ]

        win_rate = (
            len(wins)
            / len(trades)
        )

        gross_profit = wins[
            "Trade_Return"
        ].sum()

        gross_loss = abs(
            losses[
                "Trade_Return"
            ].sum()
        )

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

    exposure = float(
        equity["Position"].mean()
    )

    buy_hold = (
        df["Close"].iloc[-1]
        / df["Close"].iloc[0]
        - 1.0
    )

    excess_return = (
        total_return
        - buy_hold
    )

    metrics = {
        "Strategy": strategy_name,
        "Threshold": THRESHOLD,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_drawdown,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Trades": len(trades),
        "Exposure": exposure,
        "Buy_Hold_Return": buy_hold,
        "Excess_Return": excess_return,
        "Final_Equity": final_equity,
    }

    return metrics, trades, equity


def main():

    header(
        "ALPHALENS V9 RISK & POSITION-SIZING BACKTEST"
    )

    print(
        "Model           : XGBoost"
    )

    print(
        f"Entry threshold : {THRESHOLD:.2f}"
    )

    print(
        f"Transaction cost: {TRANSACTION_COST:.3%}"
    )

    print(
        f"Slippage        : {SLIPPAGE:.3%}"
    )

    predictions = load_predictions()

    prices = load_prices()

    df = prepare_dataset(
        predictions,
        prices
    )

    strategies = []

    for size in FIXED_SIZES:

        strategies.append((
            f"Fixed_{int(size * 100)}pct",
            lambda row, s=size: s
        ))

    strategies.append((
        "Confidence_Sizing",
        lambda row: confidence_position(
            float(row["Probability"])
        )
    ))

    strategies.append((
        "Volatility_Sizing",
        lambda row: volatility_position(
            float(row["Rolling_Volatility"])
        )
    ))

    all_metrics = []
    all_trades = []
    all_equity = []

    header("RUNNING V9 STRATEGIES")

    for name, sizing_function in strategies:

        result = run_strategy(
            df,
            name,
            sizing_function
        )

        metrics, trades, equity = result

        all_metrics.append(metrics)

        if not trades.empty:
            all_trades.append(trades)

        if not equity.empty:
            all_equity.append(equity)

        print()
        print(
            f"Strategy       : {name}"
        )
        print(
            f"Return         : "
            f"{metrics['Total_Return']:.2%}"
        )
        print(
            f"CAGR           : "
            f"{metrics['CAGR']:.2%}"
        )
        print(
            f"Sharpe         : "
            f"{metrics['Sharpe']:.4f}"
        )
        print(
            f"Sortino        : "
            f"{metrics['Sortino']:.4f}"
        )
        print(
            f"Max Drawdown   : "
            f"{metrics['Max_Drawdown']:.2%}"
        )
        print(
            f"Win Rate       : "
            f"{metrics['Win_Rate']:.2%}"
        )
        print(
            f"Profit Factor  : "
            f"{metrics['Profit_Factor']:.4f}"
        )
        print(
            f"Trades         : "
            f"{metrics['Trades']}"
        )
        print(
            f"Exposure       : "
            f"{metrics['Exposure']:.2%}"
        )
        print(
            f"Excess Return  : "
            f"{metrics['Excess_Return']:.2%}"
        )

    results = pd.DataFrame(
        all_metrics
    )

    if all_trades:
        trades_output = pd.concat(
            all_trades,
            ignore_index=True
        )
    else:
        trades_output = pd.DataFrame()

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

    trades_output.to_csv(
        TRADE_FILE,
        index=False
    )

    equity_output.to_csv(
        EQUITY_FILE,
        index=False
    )

    header("V9 FINAL COMPARISON")

    columns = [
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
        results[columns].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # Best by Sharpe, but only among strategies
    # that have positive excess return.
    candidates = results[
        results["Excess_Return"] > 0
    ].copy()

    if len(candidates) > 0:

        best = candidates.loc[
            candidates["Sharpe"].idxmax()
        ]

        print()
        print(
            "Best risk-adjusted candidate:"
        )

        print(
            f"Strategy : {best['Strategy']}"
        )

        print(
            f"Sharpe   : {best['Sharpe']:.4f}"
        )

        print(
            f"Return   : {best['Total_Return']:.2%}"
        )

        print(
            f"Max DD   : {best['Max_Drawdown']:.2%}"
        )

    header("V9 INTERPRETATION")

    print(
        "V9 uses the XGBoost 0.70 threshold selected "
        "from the V8 robustness test."
    )

    print(
        "The purpose is to test risk and position sizing, "
        "not to retrain the model."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A lower drawdown or higher Sharpe does not "
        "automatically establish future profitability."
    )

    print(
        "The V9 results must still be interpreted as "
        "historical out-of-sample research."
    )

    header("OUTPUT FILES")

    print(
        f"Results : {RESULT_FILE}"
    )

    print(
        f"Trades  : {TRADE_FILE}"
    )

    print(
        f"Equity  : {EQUITY_FILE}"
    )

    print()
    print(
        "PASS: V9 risk backtest completed successfully."
    )


if __name__ == "__main__":
    main()