import pandas as pd
import yfinance as yf

def get_data(stocks, start, end):
    tickers = [str(t).strip().upper() for t in stocks if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True
    )

    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance. Check tickers/date range.")

    # Extract prices as DataFrame with columns = tickers
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        if "Adj Close" in lvl0:
            prices = df["Adj Close"]
        elif "Close" in lvl0:
            prices = df["Close"]
        elif "Adj Close" in lvl1:
            prices = df.xs("Adj Close", axis=1, level=1)
        elif "Close" in lvl1:
            prices = df.xs("Close", axis=1, level=1)
        else:
            raise KeyError(f"Could not find 'Adj Close' or 'Close' in returned columns: {df.columns}")
    else:
        # Single ticker case
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            prices = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError(f"Could not find 'Adj Close' or 'Close' in

