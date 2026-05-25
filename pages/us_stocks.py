"""美股分析 — CAPM + Fama-French + Markowitz 最优组合。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from config import US_POOL, TICKER_NAMES, START_DATE, END_DATE
from shared import inject_css, page_header
from data.us_fetcher import fetch_prices
from data.ff3_fetcher import fetch_ff3_daily, get_rf_daily, get_mkt_excess
from domain.capm import run_capm_batch
from domain.fama_french import run_ff3_batch
from domain.portfolio import (
    annualize, min_variance, max_sharpe, portfolio_stats,
    efficient_frontier, monte_carlo,
)
from viz.charts import (
    capm_beta_alpha, ff3_factor_loadings, model_r2_comparison,
    efficient_frontier_chart, portfolio_weights_chart,
)

inject_css()
page_header("美股分析", "CAPM · Fama-French 三因素 · Markowitz 最优投资组合")

# ── 侧边栏 ──
with st.sidebar:
    st.header("配置")

    st.subheader("美股代码")
    us_default = "\n".join(["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"])
    us_text = st.text_area(
        "输入 yfinance 代码，一行一个", us_default,
        height=120,
        help="提示：" + ", ".join(list(US_POOL.keys())[:10]) + " …",
    )
    selected = [t.strip().upper() for t in us_text.split("\n") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", pd.to_datetime(START_DATE))
    with col2:
        end_date = st.date_input("结束日期", pd.to_datetime(END_DATE))

    st.subheader("分析模型")
    run_capm = st.checkbox("CAPM", value=True)
    run_ff3 = st.checkbox("Fama-French 三因素", value=True)

    st.subheader("组合约束")
    allow_short = st.checkbox("允许做空", value=False)
    max_weight = st.slider("单票最大权重", 0.2, 1.0, 1.0, 0.05)
    n_mc = st.slider("MC 模拟次数", 1000, 50000, 10000, 1000)

    st.divider()
    st.caption(f"已选 {len(selected)} 只美股")
    st.caption("数据源: yfinance + Kenneth French Data Library")

# ── 数据加载 ──
@st.cache_data(ttl=3600, show_spinner="下载美股数据...")
def load_us_data(tickers, start_s, end_s):
    return fetch_prices(list(tickers), start_s, end_s, verbose=False)


if not selected:
    st.warning("请在侧边栏至少选择一只股票。")
    st.stop()

start_s = start_date.strftime("%Y-%m-%d")
end_s = end_date.strftime("%Y-%m-%d")
prices, returns = load_us_data(tuple(selected), start_s, end_s)

if returns.empty:
    st.error("未能加载任何股票数据。")
    st.stop()

ff3 = fetch_ff3_daily()
rf_daily = get_rf_daily(ff3)
mkt_excess = get_mkt_excess(ff3)
excess_returns = returns.sub(rf_daily, axis=0)

# ── 概览 ──
st.header("概览")

mu, cov, rf_annual = annualize(returns, rf_daily)
n_assets = len(returns.columns)
bounds = (-1.0, 1.0) if allow_short else (0.0, min(max_weight, 1.0))

w_eq = np.ones(n_assets) / n_assets
eq_stats = portfolio_stats(w_eq, mu, cov, rf_annual)
w_mv = min_variance(cov, bounds)
mv_stats = portfolio_stats(w_mv, mu, cov, rf_annual)
w_ms = max_sharpe(mu, cov, rf_annual, bounds)
ms_stats = portfolio_stats(w_ms, mu, cov, rf_annual)

cols = st.columns(5)
cols[0].metric("加载股票", n_assets)
cols[1].metric("交易日数", len(returns))
cols[2].metric("最大 Sharpe", f"{ms_stats['sharpe']:.2f}")
cols[3].metric("最小波动", f"{mv_stats['vol']:.1%}")
cols[4].metric("无风险利率", f"{rf_annual:.2%}")

# ── CAPM ──
if run_capm:
    st.header("CAPM 回归结果")
    st.caption("基准: S&P 500 (FF Mkt-RF)")

    capm_df = run_capm_batch(excess_returns, mkt_excess, None)

    if not capm_df.empty:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig_capm = capm_beta_alpha(capm_df)
            st.plotly_chart(fig_capm, width="stretch", config={
                "displayModeBar": True, "displaylogo": False,
            })
        with col_r:
            st.subheader("回归结果表")
            display_capm = capm_df[["beta", "alpha_annual", "r_squared", "market"]].copy()
            display_capm["alpha_annual"] = display_capm["alpha_annual"].apply(
                lambda x: f"{x*100:+.2f}%")
            display_capm = display_capm.rename(columns={
                "beta": "Beta", "alpha_annual": "Alpha(年化)",
                "r_squared": "R²", "market": "基准",
            })
            st.dataframe(display_capm, width="stretch")
    else:
        st.warning("CAPM 回归失败：数据不足。")

# ── FF3 ──
if run_ff3:
    st.header("Fama-French 三因素回归")
    st.caption("因子来源: Kenneth French Data Library")

    ff3_df = run_ff3_batch(excess_returns, ff3)

    if not ff3_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig_ff3 = ff3_factor_loadings(ff3_df)
            st.plotly_chart(fig_ff3, width="stretch", config={
                "displayModeBar": True, "displaylogo": False,
            })
        with c2:
            if run_capm and not capm_df.empty:
                common = capm_df.index.intersection(ff3_df.index)
                fig_r2 = model_r2_comparison(capm_df.loc[common], ff3_df.loc[common])
                st.plotly_chart(fig_r2, width="stretch", config={
                    "displayModeBar": True, "displaylogo": False,
                })

        st.subheader("FF3 回归结果表")
        display_ff3 = ff3_df[["beta_mkt", "beta_smb", "beta_hml",
                               "alpha_annual", "r_squared"]].copy()
        display_ff3["alpha_annual"] = display_ff3["alpha_annual"].apply(
            lambda x: f"{x*100:+.2f}%")
        display_ff3 = display_ff3.rename(columns={
            "beta_mkt": "β_Mkt", "beta_smb": "β_SMB", "beta_hml": "β_HML",
            "alpha_annual": "Alpha(年化)", "r_squared": "R²",
        })
        st.dataframe(display_ff3, width="stretch")
    else:
        st.warning("FF3 回归失败：数据不足。")

# ── 投资组合优化 ──
st.header("最优投资组合")

c1, c2, c3 = st.columns(3)
c1.metric("最大 Sharpe 组合",
          f"Sharpe: {ms_stats['sharpe']:.2f}",
          f"收益 {ms_stats['ret']:.1%} | 波动 {ms_stats['vol']:.1%}")
c2.metric("最小方差组合",
          f"波动: {mv_stats['vol']:.1%}",
          f"收益 {mv_stats['ret']:.1%} | Sharpe {mv_stats['sharpe']:.2f}")
c3.metric("等权重组合",
          f"Sharpe: {eq_stats['sharpe']:.2f}",
          f"收益 {eq_stats['ret']:.1%} | 波动 {eq_stats['vol']:.1%}")

mc_rets, mc_vols, mc_sharpes = monte_carlo(mu, cov, rf_annual, n_portfolios=n_mc)
f_rets, f_vols = efficient_frontier(mu, cov, bounds)

col_ef, col_wt = st.columns([2, 1])

with col_ef:
    ticker_names = returns.columns.tolist()
    fig_ef = efficient_frontier_chart(
        mu, cov, rf_annual, ticker_names,
        mc_rets, mc_vols, mc_sharpes,
        f_rets, f_vols, eq_stats, mv_stats, ms_stats)
    st.plotly_chart(fig_ef, width="stretch", config={
        "displayModeBar": True, "displaylogo": False,
    })

with col_wt:
    fig_wt = portfolio_weights_chart(ticker_names, w_mv, w_ms, w_eq)
    st.plotly_chart(fig_wt, width="stretch", config={
        "displayModeBar": True, "displaylogo": False,
    })

with st.expander("最大 Sharpe 组合权重明细"):
    wt_df = pd.DataFrame({
        "股票": ticker_names,
        "名称": [TICKER_NAMES.get(t, t) for t in ticker_names],
        "最大Sharpe": w_ms,
        "最小方差": w_mv,
        "等权重": w_eq,
    }).sort_values("最大Sharpe", ascending=False)
    for col in ["最大Sharpe", "最小方差", "等权重"]:
        wt_df[col] = wt_df[col].apply(lambda x: f"{x:.1%}")
    st.dataframe(wt_df, width="stretch", hide_index=True)

st.divider()
st.caption("数据来源: yfinance + Kenneth French Data Library | 基准: S&P 500 (FF Mkt-RF)")
