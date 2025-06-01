import pandas as pd
import numpy as np

def weight_long_only(alpha: pd.Series, top_cut: float = 0.2) -> pd.DataFrame:
    """
    alpha: MultiIndex (date, ticker) series.
    Returns daily weight DataFrame (date × ticker).
    """
    alpha_df = alpha.unstack()            # wide
    ranks = alpha_df.rank(axis=1, pct=True, ascending=False)
    long_mask = ranks > (1 - top_cut)
    w = long_mask.div(long_mask.sum(axis=1), axis=0).fillna(0)
    return w

def backtest(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    daily_ret = prices.pct_change().shift(-1)
    port = (weights.shift() * daily_ret).sum(axis=1)
    return port.rename("portfolio")
