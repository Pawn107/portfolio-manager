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

# ── URL 参数记忆 ──
_default_codes = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META", "TSLA"]
try:
    url_codes = st.query_params.get("codes", "")
    if url_codes:
        _parsed = [c.strip().upper() for c in url_codes.split(",") if c.strip()]
        if _parsed:
            _default_codes = _parsed
except Exception:
    pass

us_default = "\n".join(_default_codes)

# ── 侧边栏 ──
with st.sidebar:
    st.header("配置")

    st.subheader("美股代码")
    us_text = st.text_area(
        "输入 yfinance 代码，一行一个", us_default,
        height=120,
        help="提示：" + ", ".join(list(US_POOL.keys())[:10]) + " …",
    )
    selected = [t.strip().upper() for t in us_text.split("\n") if t.strip()]

    if selected and selected != _default_codes:
        st.query_params["codes"] = ",".join(selected)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", pd.to_datetime(START_DATE))
    with col2:
        end_date = st.date_input("结束日期", pd.to_datetime(END_DATE))

    st.subheader("组合约束")
    allow_short = st.checkbox("允许做空", value=False)
    max_weight = st.slider("单票最大权重", 0.2, 1.0, 1.0, 0.05)
    n_mc = st.slider("MC 模拟次数", 1000, 50000, 10000, 1000)

    st.divider()
    if st.button("清除缓存并刷新"):
        st.cache_data.clear()
        st.rerun()

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

mu, cov, rf_annual = annualize(returns, rf_daily)
n_assets = len(returns.columns)
ticker_names = returns.columns.tolist()
bounds = (-1.0, 1.0) if allow_short else (0.0, min(max_weight, 1.0))
latest_prices = {t: float(prices[t].dropna().iloc[-1]) for t in ticker_names}

# ── 公共：最优权重计算 ──
w_eq = np.ones(n_assets) / n_assets
eq_stats = portfolio_stats(w_eq, mu, cov, rf_annual)

w_mv = min_variance(cov, bounds)
mv_stats = portfolio_stats(w_mv, mu, cov, rf_annual)

w_ms = max_sharpe(mu, cov, rf_annual, bounds)
ms_stats = portfolio_stats(w_ms, mu, cov, rf_annual)

mc_rets, mc_vols, mc_sharpes = monte_carlo(mu, cov, rf_annual, n_portfolios=n_mc)
f_rets, f_vols = efficient_frontier(mu, cov, bounds)

# ═══════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["因子分析", "持仓权重"])

# ── Tab 1: 因子分析 ──
with tab1:
    st.subheader("CAPM 回归结果")
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

    st.divider()

    st.subheader("Fama-French 三因素回归")
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
            if not capm_df.empty:
                common = capm_df.index.intersection(ff3_df.index)
                fig_r2 = model_r2_comparison(capm_df.loc[common], ff3_df.loc[common], multi_label="FF3 R²")
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


# ── Tab 2: 持仓权重 ──
with tab2:
    st.caption(f"基于 {len(returns)} 个交易日 | 无风险利率 {rf_annual:.2%}")

    # ═══════════════════════════════════════
    # 区域 1: 手动调仓
    # ═══════════════════════════════════════
    st.subheader("手动调仓")

    tab_weight_mode = st.radio(
        "输入方式", ["百分比权重", "持仓股数"],
        horizontal=True, key="us_weight_mode"
    )

    manual_weights = np.zeros(n_assets)
    manual_quantities = np.zeros(n_assets, dtype=int)
    total_value = 100000.0

    if tab_weight_mode == "百分比权重":
        cols_wt = st.columns(min(n_assets, 6))
        for i, ticker in enumerate(ticker_names):
            name = TICKER_NAMES.get(ticker, ticker)
            with cols_wt[i % len(cols_wt)]:
                manual_weights[i] = st.number_input(
                    f"{ticker}",
                    min_value=0.0, max_value=100.0,
                    value=round(float(w_ms[i]) * 100, 1),
                    step=0.5, key=f"us_wt_{ticker}"
                ) / 100.0
    else:
        st.caption("输入每只股票的持仓股数，系统会根据最新价格计算权重")
        total_value = st.number_input("总投入资金 ($)", 1000.0, 10000000.0, 100000.0, 1000.0,
                                       key="us_total_capital")
        cols_qt = st.columns(min(n_assets, 6))
        for i, ticker in enumerate(ticker_names):
            price = latest_prices.get(ticker, 0)
            name = TICKER_NAMES.get(ticker, ticker)
            with cols_qt[i % len(cols_qt)]:
                manual_quantities[i] = st.number_input(
                    f"{ticker} (@${price:.1f})",
                    min_value=0, value=100, step=10, key=f"us_qt_{ticker}"
                )

    # 计算手动组合
    if tab_weight_mode == "百分比权重":
        w_manual = manual_weights
        w_manual = w_manual / w_manual.sum() if w_manual.sum() > 0 else w_manual
    else:
        pos_values = manual_quantities.astype(float) * np.array([latest_prices[t] for t in ticker_names])
        w_manual = pos_values / pos_values.sum() if pos_values.sum() > 0 else np.ones(n_assets) / n_assets

    manual_stats = portfolio_stats(w_manual, mu, cov, rf_annual)

    # 手动组合指标卡片
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("年化收益", f"{manual_stats['ret']:.2%}",
               delta=f"{manual_stats['ret'] - ms_stats['ret']:+.2%} vs 最优")
    mc2.metric("年化波动", f"{manual_stats['vol']:.2%}",
               delta=f"{manual_stats['vol'] - mv_stats['vol']:+.2%} vs 最小")
    mc3.metric("Sharpe 比率", f"{manual_stats['sharpe']:.2f}",
               delta=f"{manual_stats['sharpe'] - ms_stats['sharpe']:+.2f} vs 最大")
    mc4.metric("总仓位", f"{w_manual.sum():.0%}",
               delta=f"{w_manual.sum() - 1.0:+.0%}" if abs(w_manual.sum() - 1.0) > 0.001 else None)

    # 手动组合权重明细
    with st.expander("手动调仓权重明细"):
        mw_rows = []
        for i, ticker in enumerate(ticker_names):
            name = TICKER_NAMES.get(ticker, ticker)
            price = latest_prices.get(ticker, 0)
            mw_rows.append({
                "代码": ticker,
                "名称": name,
                "最新价": f"${price:.2f}",
                "权重": f"{w_manual[i]:.1%}",
                "股数": int(manual_quantities[i]) if tab_weight_mode == "持仓股数"
                        else int(w_manual[i] * total_value / price) if price > 0 else 0,
            })
        st.dataframe(pd.DataFrame(mw_rows), width="stretch", hide_index=True)

    st.divider()

    # ═══════════════════════════════════════
    # 区域 2: Markowitz 最优权重 (参考)
    # ═══════════════════════════════════════
    st.subheader("Markowitz 最优权重 (参考)")

    c1, c2, c3 = st.columns(3)
    c1.metric("最大 Sharpe", f"{ms_stats['sharpe']:.2f}",
              f"收益 {ms_stats['ret']:.1%} | 波动 {ms_stats['vol']:.1%}")
    c2.metric("最小方差", f"{mv_stats['vol']:.1%}",
              f"收益 {mv_stats['ret']:.1%} | Sharpe {mv_stats['sharpe']:.2f}")
    c3.metric("等权重", f"{eq_stats['sharpe']:.2f}",
              f"收益 {eq_stats['ret']:.1%} | 波动 {eq_stats['vol']:.1%}")

    col_ef, col_wt = st.columns([2, 1])

    with col_ef:
        fig_ef = efficient_frontier_chart(
            mu, cov, rf_annual, ticker_names,
            mc_rets, mc_vols, mc_sharpes,
            f_rets, f_vols, eq_stats, mv_stats, ms_stats,
            n_mc=n_mc)

        import plotly.graph_objects as go
        from viz.theme import COLORS
        fig_ef.add_trace(go.Scatter(
            x=[manual_stats["vol"]], y=[manual_stats["ret"]],
            mode="markers",
            marker=dict(color=COLORS["orange"], size=18, symbol="x-thin", line=dict(width=3)),
            name=f"手动调仓 (SR={manual_stats['sharpe']:.2f})",
        ))

        st.plotly_chart(fig_ef, width="stretch", config={
            "displayModeBar": True, "displaylogo": False,
        })

    with col_wt:
        fig_wt = portfolio_weights_chart(ticker_names, w_mv, w_ms, w_eq)
        st.plotly_chart(fig_wt, width="stretch", config={
            "displayModeBar": True, "displaylogo": False,
        })

    # 权重对比表
    with st.expander("最优权重 vs 手动权重对比"):
        comp_rows = []
        for i, ticker in enumerate(ticker_names):
            name = TICKER_NAMES.get(ticker, ticker)
            comp_rows.append({
                "代码": ticker,
                "名称": name,
                "最大Sharpe": w_ms[i],
                "最小方差": w_mv[i],
                "等权重": w_eq[i],
                "手动调仓": w_manual[i],
            })
        comp_df = pd.DataFrame(comp_rows).sort_values("手动调仓", ascending=False)
        for col in ["最大Sharpe", "最小方差", "等权重", "手动调仓"]:
            comp_df[col] = comp_df[col].apply(lambda x: f"{x:.1%}")
        st.dataframe(comp_df, width="stretch", hide_index=True)

st.divider()
st.caption("数据源: yfinance + Kenneth French Data Library | 基准: S&P 500 (FF Mkt-RF)")
