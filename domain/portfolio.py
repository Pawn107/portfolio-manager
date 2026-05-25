"""Markowitz 均值-方差投资组合优化。"""
import numpy as np
from scipy.optimize import minimize
from config import TRADING_DAYS


def annualize(returns, rf_daily=None, trading_days=TRADING_DAYS):
    """年化收益率、协方差矩阵、无风险利率。"""
    mu = returns.mean() * trading_days
    cov = returns.cov() * trading_days
    rf = rf_daily.mean() * trading_days if rf_daily is not None else 0.0
    return mu.values, cov.values, rf


def _portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)


def min_variance(cov, bounds=(0.0, 1.0)):
    """最小方差组合权重。"""
    n = cov.shape[0]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bnds = [bounds] * n
    w0 = np.ones(n) / n
    res = minimize(_portfolio_vol, w0, args=(cov,), bounds=bnds,
                   constraints=cons, method="SLSQP")
    return res.x


def max_sharpe(mu, cov, rf, bounds=(0.0, 1.0)):
    """最大 Sharpe 组合权重。"""
    n = cov.shape[0]

    def _neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bnds = [bounds] * n
    w0 = np.ones(n) / n
    res = minimize(_neg_sharpe, w0, bounds=bnds, constraints=cons, method="SLSQP")
    return res.x


def portfolio_stats(w, mu, cov, rf):
    """计算组合指标。"""
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return {"ret": ret, "vol": vol, "sharpe": sharpe}


def efficient_frontier(mu, cov, bounds=(0.0, 1.0), n_points=50):
    """生成有效前沿 (target_rets, min_vols)。"""
    n = cov.shape[0]
    w_mv = min_variance(cov, bounds)
    min_ret = w_mv @ mu
    max_ret = max(mu)

    target_rets = np.linspace(min_ret, max_ret, n_points)
    frontier_vols = []

    for tr in target_rets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, tr=tr: w @ mu - tr},
        )
        bnds = [bounds] * n
        w0 = np.ones(n) / n
        res = minimize(_portfolio_vol, w0, args=(cov,), bounds=bnds,
                       constraints=cons, method="SLSQP")
        if res.success:
            frontier_vols.append(np.sqrt(res.x @ cov @ res.x))
        else:
            frontier_vols.append(np.nan)

    frontier_vols = np.array(frontier_vols)
    valid = ~np.isnan(frontier_vols)
    return target_rets[valid], frontier_vols[valid]


def monte_carlo(mu, cov, rf, n_portfolios=10000, seed=42):
    """蒙特卡洛随机组合生成。"""
    np.random.seed(seed)
    n = len(mu)
    weights = np.random.dirichlet(np.ones(n), n_portfolios)
    rets = weights @ mu
    vols = np.array([np.sqrt(w @ cov @ w) for w in weights])
    sharpes = (rets - rf) / vols
    return rets, vols, sharpes
