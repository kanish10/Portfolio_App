import pandas as pd

def performance_stats(series: pd.Series) -> pd.Series:
    ann_ret   = (1 + series).prod() ** (252/len(series)) - 1
    ann_vol   = series.std() * (252 ** 0.5)
    sharpe    = ann_ret / ann_vol if ann_vol else 0
    max_dd    = (series.cumsum()+1).cummax() - (series.cumsum()+1)
    return pd.Series(dict(CAGR=ann_ret, Vol=ann_vol, Sharpe=sharpe,
                          MaxDD=max_dd.max()))
