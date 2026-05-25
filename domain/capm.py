from __future__ import annotations
"""CAPM 回归模块 — 支持双市场基准 (美股→S&P500, A股→沪深300)。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import TRADING_DAYS

MIN_OBS = 60


def run_capm_single(excess_returns: pd.Series, market_excess: pd.Series,
                     trading_days: int = TRADING_DAYS) -> dict | None:
    """对单只股票做 CAPM 回归。"""
    y = excess_returns.dropna()
    common = y.index.intersection(market_excess.dropna().index)
    if len(common) < MIN_OBS:
        return None

    X = sm.add_constant(market_excess.loc[common])
    y_reg = y.loc[common]
    model = sm.OLS(y_reg, X).fit()

    return {
        "alpha_daily": model.params.iloc[0],
        "alpha_annual": model.params.iloc[0] * trading_days,
        "beta": model.params.iloc[1],
        "t_alpha": model.tvalues.iloc[0],
        "t_beta": model.tvalues.iloc[1],
        "r_squared": model.rsquared,
        "resid_std": np.std(model.resid),
    }


def run_capm_batch(excess_returns: pd.DataFrame,
                    us_market_excess: pd.Series | None,
                    cn_market_excess: pd.Series | None,
                    cn_tickers: list[str] | None = None,
                    trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    """批量 CAPM 回归。美股用 S&P500，A股用沪深300。"""
    if cn_tickers is None:
        cn_tickers = []

    results = {}
    for ticker in excess_returns.columns:
        y = excess_returns[ticker]

        if ticker in cn_tickers and cn_market_excess is not None:
            mkt = cn_market_excess
            mkt_name = "沪深300"
        else:
            mkt = us_market_excess
            mkt_name = "S&P500"

        if mkt is None:
            continue

        r = run_capm_single(y, mkt, trading_days)
        if r is not None:
            r["market"] = mkt_name
            results[ticker] = r

    df = pd.DataFrame(results).T
    df.index.name = "ticker"
    return df
