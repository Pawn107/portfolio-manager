"""CH-3 中国版三因子模型 (Liu, Stambaugh & Yuan, 2019).

核心改进 vs US FF3:
- 市值因子: 剔除底部 30% 最小市值股票 (壳价值污染)
- 价值因子: 用 EP (1/PE) 替代 B/M (会计准则差异)
- 市场因子: A 股本地市场

因子构建频率: 月频 (每月末重新分组)
因子收益频率: 日频 (组内等权)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import TRADING_DAYS, CH3_UNIVERSE

MIN_OBS = 60


def build_ch3_factors(daily_data, valuation, rf_daily=None):
    """从个股数据构建 CH-3 日频因子。

    每月末按以下规则分组:
    1. 估算个股市值 = 当日价格 × (当前总市值 / 当前价格)
    2. 按市值排序，剔除末 30% (壳价值过滤)
    3. 剩余股票: 市值中位数分 B(大) / S(小)
    4. EP = 1/PE_ttm，分 H(高EP/价值, top 30%) / L(低EP/成长, bottom 30%)
    5. 形成 4 组合: SH, SL, BH, BL
    6. SMB = (SH+SL)/2 - (BH+BL)/2
    7. HML = (SH+BH)/2 - (SL+BL)/2

    Args:
        daily_data: {code: DataFrame with OHLCV (date index)}
        valuation: {code: {mcap_yi, pe_ttm, price}}
        rf_daily: Series of daily risk-free rates (optional, used for MKT excess)

    Returns:
        DataFrame with columns: MKT, SMB, HML (daily frequency)
    """
    prices = {}
    for code, df in daily_data.items():
        if df is not None and "close" in df.columns and len(df) > 60:
            prices[code] = df["close"]

    if len(prices) < 8:
        return None

    price_df = pd.DataFrame(prices).dropna()
    returns = price_df.pct_change().dropna()

    if len(returns) < MIN_OBS:
        return None

    # 市场因子: 等权平均收益
    mkt = returns.mean(axis=1)

    # 估算每只股票的隐含股数 (当前市值 / 当前价格)
    implied_shares = {}
    for code in price_df.columns:
        val = valuation.get(code, {})
        mcap = val.get("mcap_yi", 0)
        cur_price = val.get("price", 0)
        if mcap > 0 and cur_price > 0:
            implied_shares[code] = mcap / cur_price  # 亿股

    # 每月末分组
    monthly_groups = price_df.resample("ME").last()
    monthly_dates = monthly_groups.index

    smb_series = pd.Series(0.0, index=returns.index)
    hml_series = pd.Series(0.0, index=returns.index)

    for i, month_end in enumerate(monthly_dates):
        # 确定下一月区间
        if i < len(monthly_dates) - 1:
            next_end = monthly_dates[i + 1]
        else:
            next_end = returns.index[-1]

        mask = (returns.index > month_end) & (returns.index <= next_end)
        month_ret = returns.loc[mask]
        if len(month_ret) < 5:
            continue

        # 月末价格
        mp = monthly_groups.loc[month_end]

        # 估算月末市值
        est_mcaps = {}
        est_eps = {}
        for code in mp.index:
            if pd.isna(mp[code]) or mp[code] <= 0:
                continue
            p = mp[code]
            if code in implied_shares:
                est_mcaps[code] = p * implied_shares[code]
            else:
                est_mcaps[code] = p * 10  # fallback

            val = valuation.get(code, {})
            pe = val.get("pe_ttm", 0)
            if pe > 0:
                est_eps[code] = 1.0 / pe

        if len(est_mcaps) < 8:
            continue

        # 按市值排序，剔除底部 30%
        sorted_size = sorted(est_mcaps.items(), key=lambda x: x[1])
        n = len(sorted_size)
        cutoff = max(int(n * 0.3), 1)
        eligible = dict(sorted_size[cutoff:])

        if len(eligible) < 8:
            continue

        # B / S 分组 (市值中位数)
        sorted_eligible = sorted(eligible.items(), key=lambda x: x[1])
        mid = len(sorted_eligible) // 2
        S_codes = {c for c, _ in sorted_eligible[:mid]}
        B_codes = {c for c, _ in sorted_eligible[mid:]}

        # H / L 分组 (EP 前30% / 后30%)
        eligible_ep = {c: est_eps.get(c, 0.05) for c in eligible}
        sorted_ep = sorted(eligible_ep.items(), key=lambda x: x[1], reverse=True)
        n_ep = len(sorted_ep)
        h_cutoff = max(int(n_ep * 0.3), 1)
        l_cutoff = min(int(n_ep * 0.7), n_ep - 1)
        H_codes = {c for c, _ in sorted_ep[:h_cutoff]}
        L_codes = {c for c, _ in sorted_ep[l_cutoff:]}

        # 形成 4 组合
        SH = list(S_codes & H_codes)
        SL = list(S_codes & L_codes)
        BH = list(B_codes & H_codes)
        BL = list(B_codes & L_codes)

        for date in month_ret.index:
            dr = month_ret.loc[date]
            sh_r = dr[SH].mean() if SH else 0
            sl_r = dr[SL].mean() if SL else 0
            bh_r = dr[BH].mean() if BH else 0
            bl_r = dr[BL].mean() if BL else 0

            smb_series[date] = (sh_r + sl_r) / 2 - (bh_r + bl_r) / 2
            hml_series[date] = (sh_r + bh_r) / 2 - (sl_r + bl_r) / 2

    factors = pd.DataFrame({
        "MKT": mkt,
        "SMB": smb_series,
        "HML": hml_series,
    }, index=returns.index)

    factors = factors.replace([np.inf, -np.inf], np.nan).dropna()
    return factors


def run_ch3_single(excess_returns, ch3_factors, trading_days=TRADING_DAYS):
    """单只股票 CH-3 回归。"""
    y = excess_returns.dropna()
    X = sm.add_constant(ch3_factors[["MKT", "SMB", "HML"]])
    common = y.index.intersection(X.dropna().index)
    if len(common) < MIN_OBS:
        return None

    model = sm.OLS(y.loc[common], X.loc[common]).fit()

    return {
        "alpha_daily": model.params.get("const", np.nan),
        "alpha_annual": model.params.get("const", np.nan) * trading_days,
        "beta_mkt": model.params.get("MKT", np.nan),
        "beta_smb": model.params.get("SMB", np.nan),
        "beta_hml": model.params.get("HML", np.nan),
        "t_mkt": model.tvalues.get("MKT", np.nan),
        "t_smb": model.tvalues.get("SMB", np.nan),
        "t_hml": model.tvalues.get("HML", np.nan),
        "r_squared": model.rsquared,
        "r_squared_adj": model.rsquared_adj,
    }


def run_ch3_batch(excess_returns, ch3_factors, trading_days=TRADING_DAYS):
    """批量 CH-3 回归。"""
    results = {}
    for ticker in excess_returns.columns:
        r = run_ch3_single(excess_returns[ticker], ch3_factors, trading_days)
        if r is not None:
            results[ticker] = r

    df = pd.DataFrame(results).T
    df.index.name = "ticker"
    return df
