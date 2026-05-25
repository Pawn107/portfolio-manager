from __future__ import annotations
"""Fama-French 三因素回归模块。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import TRADING_DAYS

MIN_OBS = 60


def run_ff3_single(excess_returns: pd.Series, ff3_factors: pd.DataFrame,
                    trading_days: int = TRADING_DAYS) -> dict | None:
    """对单只股票做 FF3 回归。"""
    y = excess_returns.dropna()
    X = sm.add_constant(ff3_factors[["Mkt-RF", "SMB", "HML"]])
    common = y.index.intersection(X.dropna().index)
    if len(common) < MIN_OBS:
        return None

    model = sm.OLS(y.loc[common], X.loc[common]).fit()

    return {
        "alpha_daily": model.params.get("const", np.nan),
        "alpha_annual": model.params.get("const", np.nan) * trading_days,
        "beta_mkt": model.params.get("Mkt-RF", np.nan),
        "beta_smb": model.params.get("SMB", np.nan),
        "beta_hml": model.params.get("HML", np.nan),
        "t_mkt": model.tvalues.get("Mkt-RF", np.nan),
        "t_smb": model.tvalues.get("SMB", np.nan),
        "t_hml": model.tvalues.get("HML", np.nan),
        "r_squared": model.rsquared,
        "r_squared_adj": model.rsquared_adj,
    }


def run_ff3_batch(excess_returns: pd.DataFrame, ff3_factors: pd.DataFrame,
                   trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    """批量 FF3 回归。"""
    results = {}
    for ticker in excess_returns.columns:
        y = excess_returns[ticker]
        r = run_ff3_single(y, ff3_factors, trading_days)
        if r is not None:
            results[ticker] = r

    df = pd.DataFrame(results).T
    df.index.name = "ticker"
    return df
