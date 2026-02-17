import pandas as pd
import yfinance as yf

def get_data(stocks, start, end):
    """
    Download historical prices and return:
    returns (DataFrame), mean_returns (Series), cov_matrix (DataFrame)
    Robust to yfinance MultiIndex and missing Adj Close.
    """

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
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError("No data returned from Yahoo Finance. Check tickers/date range.")

    # df can be MultiIndex; extract Adj Close or Close safely
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
            raise KeyError(
                "Could not find 'Adj Close' or 'Close' in downloaded data. "
                f"Level0: {sorted(set(lvl0))}, Level1: {sorted(set(lvl1))}"
            )
    else:
        # Single-ticker case
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in df.columns:
            prices = df[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise KeyError(
                "Could not find 'Adj Close' or 'Close' in downloaded data. "
                f"Columns: {list(df.columns)}"
            )

    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Prices are empty after dropping missing rows.")

    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough data to compute returns. Try a longer date range.")

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix
