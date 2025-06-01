import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def zscore(df: pd.Series) -> pd.Series:
    return (df - df.mean()) / df.std()

def composite_alpha(panel: pd.DataFrame) -> pd.Series:
    # simple equal-weighted z-scores of 3 factors
    panel = panel.copy()
    for col in panel.columns:
        panel[col] = zscore(panel[col])
    alpha = panel.mean(axis=1)            # average across factors
    return alpha.rename("alpha")

def ml_alpha(panel: pd.DataFrame) -> pd.Series:
    """Cross-sectional GBM each month; walk-forward 1-step."""
    alpha_ml = []
    scaler = StandardScaler()
    for date, group in panel.groupby(level=0):
        X = scaler.fit_transform(group.values)
        y = group["mom_1m"] if "mom_1m" in group else group.iloc[:, 0]
        mdl = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        mdl.fit(X, y)
        preds = pd.Series(mdl.predict(X), index=group.index)
        alpha_ml.append(preds.rename(date))
    return pd.concat(alpha_ml).sort_index().rename("alpha_ml")
