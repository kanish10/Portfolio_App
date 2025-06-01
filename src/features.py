import pandas as pd
import numpy as np

def pct_change(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.pct_change(window)

def rolling_vol(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return prices.pct_change().rolling(window).std() * np.sqrt(252)

def make_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return MultiIndex DataFrame: (date, ticker) × features.
    """
    feats = {
        "mom_12m": pct_change(prices, 252),
        "mom_3m" : pct_change(prices, 63),
        "vol_60d": rolling_vol(prices, 60),
    }
    panel = (
        pd.concat(feats, axis=1)            # wide columns (feature, ticker)
          .stack(level=1)                  # → index (date, ticker)
          .dropna()
    )
    panel.index.names = ["Date", "Ticker"]
    return panel
