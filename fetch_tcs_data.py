import os
import sys
import time
import pandas as pd
import yfinance as yf


# ============================================================
# TCS HISTORICAL PRICE DATA FETCHER
# ============================================================

TICKER = "TCS.NS"
OUTPUT_FILE = "data/raw/TCS.csv"

START_DATE = "2018-01-01"

# yfinance generally cannot reliably return today's/future data
# beyond the latest available market session.
END_DATE = None


def main():
    print("=" * 60)
    print("TCS HISTORICAL DATA FETCH")
    print("=" * 60)

    print(f"Ticker      : {TICKER}")
    print(f"Start date  : {START_DATE}")
    print("Downloading latest available historical data...")
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    output_dir = os.path.dirname(OUTPUT_FILE)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------
    try:
        if END_DATE:
            df = yf.download(
                TICKER,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=True,
                actions=False,
            )
        else:
            df = yf.download(
                TICKER,
                start=START_DATE,
                auto_adjust=False,
                progress=True,
                actions=False,
            )

    except Exception as e:
        print()
        print("ERROR: Failed to download TCS data.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # Validate download
    # --------------------------------------------------------
    if df is None or df.empty:
        print()
        print("ERROR: No data was returned by Yahoo Finance.")
        print("Check your internet connection and try again.")
        sys.exit(1)

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --------------------------------------------------------
    # Remove unnecessary columns
    # --------------------------------------------------------
    required_columns = ["Close", "High", "Low", "Open", "Volume"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print()
        print("ERROR: Missing expected columns:")
        print(missing_columns)
        print()
        print("Columns returned:")
        print(df.columns.tolist())
        sys.exit(1)

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # Reset index so Date becomes a normal column
    # --------------------------------------------------------
    df.reset_index(inplace=True)

    # --------------------------------------------------------
    # Normalize Date column
    # --------------------------------------------------------
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        else:
            print("ERROR: Could not find Date column.")
            sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=["Date"]).copy()

    # Keep only date, not time
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where essential price data is missing
    df = df.dropna(
        subset=["Date", "Close", "High", "Low", "Open", "Volume"]
    ).copy()

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------
    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    ).copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)

    # --------------------------------------------------------
    # Keep project-compatible column order
    #
    # Price = Date
    # This matches the existing AlphaLens raw-data structure.
    # --------------------------------------------------------
    df = df[
        [
            "Date",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
        ]
    ].copy()

    df.rename(
        columns={"Date": "Price"},
        inplace=True
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    try:
        df.to_csv(
            OUTPUT_FILE,
            index=False
        )
    except Exception as e:
        print()
        print("ERROR: Could not save CSV.")
        print(f"Details: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("DOWNLOAD SUCCESSFUL")
    print("=" * 60)

    print(f"Saved file : {OUTPUT_FILE}")
    print(f"Rows       : {len(df):,}")
    print(f"First date : {df['Price'].iloc[0]}")
    print(f"Last date  : {df['Price'].iloc[-1]}")

    print()
    print("Columns:")
    print(df.columns.tolist())

    print()
    print("Missing values:")
    print(df.isna().sum().to_string())

    print()
    print("First 5 rows:")
    print(df.head().to_string(index=False))

    print()
    print("Last 5 rows:")
    print(df.tail().to_string(index=False))

    print()
    print("=" * 60)
    print("TCS DATA READY")
    print("=" * 60)


if __name__ == "__main__":
    main()