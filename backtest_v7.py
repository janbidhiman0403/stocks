import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ======================================================================
# ALPHALENS V7 BACKTEST
# ======================================================================

DATA = "data"

PRED_FILE = os.path.join(
    DATA,
    "walk_forward_v6_predictions.csv"
)

RESULT_FILE = os.path.join(
    DATA,
    "backtest_v7_results.csv"
)

TRADE_FILE = os.path.join(
    DATA,
    "backtest_v7_trades.csv"
)

EQUITY_FILE = os.path.join(
    DATA,
    "backtest_v7_equity.csv"
)


# ======================================================================
# BACKTEST CONFIGURATION
# ======================================================================

INITIAL_CAPITAL = 100000.0

# Probability required to enter a long position.
ENTRY_THRESHOLD = 0.50

# Exit when probability falls below this level.
EXIT_THRESHOLD = 0.50

# Approximate one-way trading cost.
TRANSACTION_COST = 0.001

# Slippage per position change.
SLIPPAGE = 0.0005

# Position size.
POSITION_SIZE = 1.0


# ======================================================================
# HELPERS
# ======================================================================

def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ======================================================================
# LOAD PREDICTIONS
# ======================================================================

def load_predictions():
    header("LOADING V6 PREDICTIONS")

    if not os.path.exists(PRED_FILE):
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_FILE}"
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
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
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
    ).copy()

    df = df.sort_values(
        ["Model", "Date"]
    ).reset_index(drop=True)

    print(f"Prediction rows : {len(df)}")
    print(
        f"Date range      : "
        f"{df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    print()
    print("Models:")
    print(df["Model"].value_counts().to_string())

    return df


# ======================================================================
# LOAD PRICE DATA
# ======================================================================

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

    for candidate in candidates:
        if os.path.exists(candidate):
            price_file = candidate
            break

    if price_file is None:
        raise FileNotFoundError(
            "Could not find a price/features CSV containing Close."
        )

    prices = pd.read_csv(price_file)

    if "Date" not in prices.columns:
        raise ValueError(
            f"{price_file} does not contain Date."
        )

    if "Close" not in prices.columns:
        raise ValueError(
            f"{price_file} does not contain Close."
        )

    prices["Date"] = pd.to_datetime(
        prices["Date"],
        errors="coerce"
    )

    prices["Close"] = pd.to_numeric(
        prices["Close"],
        errors="coerce"
    )

    prices = prices[
        ["Date", "Close"]
    ].dropna().copy()

    prices = prices.sort_values(
        "Date"
    ).drop_duplicates(
        "Date"
    )

    print(f"Price rows      : {len(prices)}")
    print(
        f"Price date range: "
        f"{prices['Date'].min().date()} to "
        f"{prices['Date'].max().date()}"
    )

    return prices


# ======================================================================
# PREPARE MODEL DATA
# ======================================================================

def prepare_model_data(predictions, prices, model):
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
        df["Next_Close"] / df["Close"]
    ) - 1.0

    df = df.dropna(
        subset=[
            "Close",
            "Next_Close",
            "Market_Return",
        ]
    ).copy()

    return df.reset_index(drop=True)


# ======================================================================
# RUN BACKTEST
# ======================================================================

def run_backtest(df, model):
    header(
        f"BACKTEST: {model}"
    )

    capital = INITIAL_CAPITAL

    position = 0

    previous_position = 0

    equity_rows = []
    trade_rows = []

    entry_date = None
    entry_price = None
    entry_probability = None

    for i, row in df.iterrows():

        date = row["Date"]
        close = float(row["Close"])
        probability = float(row["Probability"])
        market_return = float(row["Market_Return"])

        # --------------------------------------------------------------
        # SIGNAL
        # --------------------------------------------------------------

        if position == 0:

            if probability >= ENTRY_THRESHOLD:
                new_position = 1
            else:
                new_position = 0

        else:

            if probability < EXIT_THRESHOLD:
                new_position = 0
            else:
                new_position = 1

        # --------------------------------------------------------------
        # TRANSACTION COST
        # --------------------------------------------------------------

        turnover = abs(
            new_position - previous_position
        )

        trading_cost = (
            turnover
            * (TRANSACTION_COST + SLIPPAGE)
        )

        # --------------------------------------------------------------
        # RETURN
        # --------------------------------------------------------------

        strategy_return = (
            new_position
            * market_return
            * POSITION_SIZE
        )

        net_return = (
            strategy_return
            - trading_cost
        )

        capital *= (
            1.0 + net_return
        )

        # --------------------------------------------------------------
        # TRADE ENTRY
        # --------------------------------------------------------------

        if previous_position == 0 and new_position == 1:

            entry_date = date
            entry_price = close
            entry_probability = probability

        # --------------------------------------------------------------
        # TRADE EXIT
        # --------------------------------------------------------------

        if previous_position == 1 and new_position == 0:

            if entry_price is not None:

                gross_trade_return = (
                    close / entry_price
                ) - 1.0

                trade_rows.append({
                    "Model": model,
                    "Entry_Date": entry_date,
                    "Exit_Date": date,
                    "Entry_Price": entry_price,
                    "Exit_Price": close,
                    "Entry_Probability": entry_probability,
                    "Exit_Probability": probability,
                    "Gross_Return": gross_trade_return,
                    "Winning_Trade":
                        int(gross_trade_return > 0),
                })

            entry_date = None
            entry_price = None
            entry_probability = None

        # --------------------------------------------------------------
        # EQUITY
        # --------------------------------------------------------------

        equity_rows.append({
            "Model": model,
            "Date": date,
            "Close": close,
            "Probability": probability,
            "Position": new_position,
            "Market_Return": market_return,
            "Strategy_Return": strategy_return,
            "Trading_Cost": trading_cost,
            "Net_Return": net_return,
            "Equity": capital,
        })

        previous_position = new_position
        position = new_position

    equity = pd.DataFrame(equity_rows)

    trades = pd.DataFrame(trade_rows)

    return equity, trades


# ======================================================================
# PERFORMANCE METRICS
# ======================================================================

def calculate_metrics(equity, trades):
    if len(equity) == 0:
        return {}

    start_equity = float(
        equity["Equity"].iloc[0]
    )

    end_equity = float(
        equity["Equity"].iloc[-1]
    )

    total_return = (
        end_equity / INITIAL_CAPITAL
    ) - 1.0

    days = (
        equity["Date"].iloc[-1]
        - equity["Date"].iloc[0]
    ).days

    if days > 0:
        years = days / 365.25

        cagr = (
            end_equity / INITIAL_CAPITAL
        ) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    returns = equity["Net_Return"]

    volatility = returns.std()

    if volatility > 0:
        sharpe = (
            returns.mean()
            / volatility
            * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    equity["Running_Max"] = (
        equity["Equity"].cummax()
    )

    equity["Drawdown"] = (
        equity["Equity"]
        / equity["Running_Max"]
    ) - 1.0

    max_drawdown = equity["Drawdown"].min()

    position_rate = (
        equity["Position"].mean()
    )

    if len(trades) > 0:

        win_rate = (
            trades["Winning_Trade"].mean()
        )

        gross_profit = trades.loc[
            trades["Gross_Return"] > 0,
            "Gross_Return"
        ].sum()

        gross_loss = abs(
            trades.loc[
                trades["Gross_Return"] < 0,
                "Gross_Return"
            ].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = np.inf

    else:

        win_rate = np.nan
        profit_factor = np.nan

    benchmark_return = (
        equity["Close"].iloc[-1]
        / equity["Close"].iloc[0]
    ) - 1.0

    return {
        "Initial_Capital":
            INITIAL_CAPITAL,

        "Final_Equity":
            end_equity,

        "Total_Return":
            total_return,

        "CAGR":
            cagr,

        "Sharpe":
            sharpe,

        "Max_Drawdown":
            max_drawdown,

        "Win_Rate":
            win_rate,

        "Profit_Factor":
            profit_factor,

        "Number_of_Trades":
            len(trades),

        "Market_Exposure":
            position_rate,

        "Buy_Hold_Return":
            benchmark_return,

        "Excess_Return_vs_Buy_Hold":
            total_return - benchmark_return,
    }


# ======================================================================
# PRINT METRICS
# ======================================================================

def print_metrics(model, metrics):
    print()
    print("-" * 70)
    print(f"MODEL: {model}")
    print("-" * 70)

    print(
        f"Initial Capital        : "
        f"{metrics['Initial_Capital']:,.2f}"
    )

    print(
        f"Final Equity           : "
        f"{metrics['Final_Equity']:,.2f}"
    )

    print(
        f"Total Return           : "
        f"{metrics['Total_Return']:.2%}"
    )

    print(
        f"CAGR                   : "
        f"{metrics['CAGR']:.2%}"
    )

    print(
        f"Sharpe                 : "
        f"{metrics['Sharpe']:.4f}"
    )

    print(
        f"Max Drawdown           : "
        f"{metrics['Max_Drawdown']:.2%}"
    )

    print(
        f"Win Rate               : "
        f"{metrics['Win_Rate']:.2%}"
    )

    print(
        f"Profit Factor          : "
        f"{metrics['Profit_Factor']:.4f}"
    )

    print(
        f"Number of Trades       : "
        f"{metrics['Number_of_Trades']}"
    )

    print(
        f"Market Exposure        : "
        f"{metrics['Market_Exposure']:.2%}"
    )

    print(
        f"Buy & Hold Return      : "
        f"{metrics['Buy_Hold_Return']:.2%}"
    )

    print(
        f"Excess vs Buy & Hold   : "
        f"{metrics['Excess_Return_vs_Buy_Hold']:.2%}"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "ALPHALENS V7 OUT-OF-SAMPLE BACKTEST"
    )

    print(
        "Using V6 chronological predictions only."
    )

    print(
        f"Entry threshold : {ENTRY_THRESHOLD:.2f}"
    )

    print(
        f"Exit threshold  : {EXIT_THRESHOLD:.2f}"
    )

    print(
        f"Transaction cost: {TRANSACTION_COST:.3%}"
    )

    print(
        f"Slippage        : {SLIPPAGE:.3%}"
    )

    predictions = load_predictions()

    prices = load_prices()

    models = predictions[
        "Model"
    ].dropna().unique().tolist()

    all_results = []
    all_trades = []
    all_equity = []

    for model in models:

        df = prepare_model_data(
            predictions,
            prices,
            model
        )

        if len(df) == 0:
            print(
                f"No usable price data for {model}."
            )
            continue

        equity, trades = run_backtest(
            df,
            model
        )

        metrics = calculate_metrics(
            equity,
            trades
        )

        metrics["Model"] = model

        all_results.append(metrics)

        if len(trades) > 0:
            all_trades.append(trades)

        all_equity.append(equity)

        print_metrics(
            model,
            metrics
        )

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------

    header("SAVING V7 OUTPUTS")

    results = pd.DataFrame(
        all_results
    )

    if len(all_trades) > 0:
        trades_output = pd.concat(
            all_trades,
            ignore_index=True
        )
    else:
        trades_output = pd.DataFrame()

    if len(all_equity) > 0:
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

    print(
        f"Results     : {RESULT_FILE}"
    )

    print(
        f"Trades      : {TRADE_FILE}"
    )

    print(
        f"Equity      : {EQUITY_FILE}"
    )

    # ------------------------------------------------------------------
    # FINAL COMPARISON
    # ------------------------------------------------------------------

    header("V7 FINAL COMPARISON")

    if len(results) > 0:

        display_cols = [
            "Model",
            "Total_Return",
            "CAGR",
            "Sharpe",
            "Max_Drawdown",
            "Win_Rate",
            "Profit_Factor",
            "Number_of_Trades",
            "Buy_Hold_Return",
            "Excess_Return_vs_Buy_Hold",
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

        best_idx = results[
            "Total_Return"
        ].idxmax()

        best_model = results.loc[
            best_idx,
            "Model"
        ]

        print()
        print(
            f"Best model by total return: "
            f"{best_model}"
        )

    # ------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------

    header("V7 INTERPRETATION")

    print(
        "This backtest uses only V6 out-of-sample "
        "walk-forward predictions."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A profitable backtest does not automatically "
        "prove future profitability."
    )

    print(
        "Costs, slippage, regime changes and "
        "execution assumptions remain important."
    )

    print()
    print(
        "PASS: V7 backtest completed successfully."
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()