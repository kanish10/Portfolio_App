import yfinance as yf
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=8)
def fetch_prices(tickers: str, start="2015-01-01") -> pd.DataFrame:
    """Return daily adjusted close prices for comma/space-separated tickers."""
    df = (
        yf.download(tickers=tickers.replace(",", " "),
                    start=start,
                    progress=False,
                    auto_adjust=True)["Close"]
        .dropna(how="all")
    )
    # If single ticker, df is Series → make DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers.strip().upper())
    return df
