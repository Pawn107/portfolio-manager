"""A股分析 — 信号扫描 · 持仓权重 · 因子分析。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from config import CN_POOL, CH3_UNIVERSE
from shared import inject_css, page_header
from data.cn_fetcher import (
    fetch_daily_kline, tencent_quote,
    fetch_csi300_returns, fetch_finance,
)
from data.ff3_fetcher import fetch_ff3_daily, get_rf_daily
from signals.scoring import compute_signals
from domain.capm import run_capm_batch
from domain.ch3 import build_ch3_factors, run_ch3_batch
from domain.portfolio import (
    annualize, min_variance, max_sharpe, portfolio_stats,
    efficient_frontier, monte_carlo,
)
from viz.charts import (
    capm_beta_alpha, ff3_factor_loadings, model_r2_comparison,
    efficient_frontier_chart, portfolio_weights_chart,
)

inject_css()
page_header("A股分析", "信号扫描 · 持仓权重 · 因子分析")

# ── URL 参数记忆：从 ?codes=600519,000333 读取默认股票 ──
_default_codes = ["600519", "000333", "300750"]
try:
    url_codes = st.query_params.get("codes", "")
    if url_codes:
        _parsed = [c.strip() for c in url_codes.split(",") if len(c.strip()) == 6 and c.strip().isdigit()]
        if _parsed:
            _default_codes = _parsed
except Exception:
    pass

cn_default = "\n".join(_default_codes)

# ── 侧边栏 ──
with st.sidebar:
    st.header("配置")

    st.subheader("A股代码")
    cn_text = st.text_area(
        "输入6位代码，一行一个", cn_default,
        height=90,
        help="提示：" + ", ".join(list(CN_POOL.keys())[:8]) + " …",
    )
    raw_codes = [t.strip() for t in cn_text.split("\n") if t.strip()]
    selected = [c for c in raw_codes if len(c) == 6 and c.isdigit()]

    # 股票列表变化时更新 URL
    if selected and selected != _default_codes:
        st.query_params["codes"] = ",".join(selected)

    st.divider()
    if st.button("清除缓存并刷新"):
        from data.cache import clear
        clear()
        st.cache_data.clear()
        st.rerun()

    st.caption(f"已选 {len(selected)} 只A股")
    st.caption("数据源: 东财 + 腾讯财经 + 新浪")

if not selected:
    st.warning("请在侧边栏至少输入一个6位A股代码。")
    st.stop()

# ── 数据加载 ──
@st.cache_data(ttl=1800, show_spinner="下载A股数据...")
def load_cn_data(codes):
    """加载所有选中股票的日K和估值数据。"""
    daily = {}
    valuation = {}
    for code in codes:
        d = fetch_daily_kline(code)
        if d is not None:
            daily[code] = d
    # 批量获取估值
    val = tencent_quote(list(codes))
    for code in codes:
        if code in val:
            valuation[code] = val[code]
    return daily, valuation


@st.cache_data(ttl=3600, show_spinner="加载CH-3因子构建数据...")
def load_ch3_universe():
    """加载 CH-3 因子构建所需的全部股票日K和估值。"""
    all_codes = list(set(CH3_UNIVERSE))
    daily = {}
    for code in all_codes:
        d = fetch_daily_kline(code)
        if d is not None:
            daily[code] = d
    val = tencent_quote(all_codes)
    return daily, val


daily_data, valuation = load_cn_data(tuple(selected))

if not daily_data:
    st.error("未能加载任何A股数据，请检查网络连接。")
    st.stop()

# 主股票代码（单只分析时用）
main_code = selected[0]


# ═══════════════════════════════════════════════════════════
# Tab 1: 信号扫描
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["信号扫描", "持仓权重", "因子分析"])

with tab1:
    st.subheader("信号扫描")

    # 选股方式：自由输入 + 已选列表快捷切换
    c1, c2 = st.columns([2, 1])
    with c1:
        scan_input = st.text_input(
            "输入6位A股代码", value=selected[0],
            max_chars=6, placeholder="例如 600519",
            key="scan_input",
            help="直接输入任意A股代码即可扫描"
        )
    with c2:
        if len(selected) > 1:
            quick_pick = st.selectbox("或从已选列表切换", selected, key="scan_quick")
            if st.button("切换", key="scan_switch"):
                scan_input = quick_pick

    scan_code = scan_input.strip()
    if not (len(scan_code) == 6 and scan_code.isdigit()):
        st.warning("请输入有效的6位数字代码。")
    else:
        # 优先用已加载数据，否则实时拉取
        df_daily = daily_data.get(scan_code)
        val = valuation.get(scan_code)

        if df_daily is None:
            with st.spinner(f"正在拉取 {scan_code} 数据..."):
                df_daily = fetch_daily_kline(scan_code)
                if df_daily is not None and scan_code not in daily_data:
                    daily_data[scan_code] = df_daily
            if df_daily is None:
                st.error(f"无法获取 {scan_code} 的K线数据，请检查代码是否正确或网络连接。")

        if val is None:
            new_val = tencent_quote([scan_code])
            val = new_val.get(scan_code, {})
            if val and scan_code not in valuation:
                valuation[scan_code] = val

        # 获取财务数据
        fin = fetch_finance(scan_code)

        # 计算信号（基本面 + 风控）
        with st.spinner("分析基本面和技术风控..."):
            signal_result = compute_signals(df_daily, valuation=val, finance=fin)

        val_name = val.get("name", scan_code) if val else scan_code

        # ── 信号卡片 ──
        st.subheader(f"{val_name} ({scan_code})")

        sig = signal_result["signal"]
        sig_color_map = {"BUY": "#22c55e", "SELL": "#ef4444", "HOLD": "#f59e0b", "AVOID": "#6b7280"}
        sig_icon_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "AVOID": "⚫"}

        cols = st.columns(5)
        cols[0].metric("综合信号", f"{sig_icon_map.get(sig, '')} {sig}")
        cols[1].metric("最新收盘价", f"{signal_result['close']:.2f}")
        if val:
            cols[2].metric("PE(TTM)", f"{val['pe_ttm']:.1f}" if val.get("pe_ttm") else "N/A")
            cols[3].metric("PB", f"{val['pb']:.2f}" if val.get("pb") else "N/A")
            cols[4].metric("总市值(亿)", f"{val['mcap_yi']:.0f}" if val.get("mcap_yi") else "N/A")

        st.caption(f"判断依据: {signal_result['reason']}")

        # ── 基本面得分明细 ──
        with st.expander("基本面得分明细"):
            fs = signal_result["fundamental"]
            fd_cols = st.columns(4)
            fd_cols[0].metric("PE得分", f"{fs['pe_score']}/35", help=fs.get("details", [""])[0] if len(fs.get("details", [])) > 0 else "")
            fd_cols[1].metric("PB得分", f"{fs['pb_score']}/25", help=fs.get("details", [""])[1] if len(fs.get("details", [])) > 1 else "")
            fd_cols[2].metric("ROE得分", f"{fs['roe_score']}/25", help=fs.get("details", [""])[2] if len(fs.get("details", [])) > 2 else "")
            fd_cols[3].metric("市值得分", f"{fs['mcap_score']}/15", help=fs.get("details", [""])[3] if len(fs.get("details", [])) > 3 else "")

            st.progress(fs["total_score"] / 100, text=f"基本面总分: {fs['total_score']}/100")
            for d in fs.get("details", []):
                st.caption(f"• {d}")

        # ── 技术风控状态 ──
        with st.expander("技术风控状态"):
            risk = signal_result["risk"]
            if risk["signal"] == "SELL":
                st.error(f"⚠️ 风控触发: {risk['reason']}")
            else:
                st.success(f"✅ {risk['reason']}")

            if risk.get("risk_flags"):
                for flag in risk["risk_flags"]:
                    st.warning(f"• {flag}")
            else:
                st.caption("无风控警报")

        # ── 市场环境 ──
        with st.expander("市场环境 & 仓位建议"):
            mkt = signal_result["market"]
            mode_names = {"normal": "正常", "panic": "恐慌", "euphoria": "狂热"}
            st.metric("市场状态", mode_names.get(mkt["mode"], mkt["mode"]))
            st.metric("建议仓位", f"{mkt['suggested_position']:.0%}")
            st.caption("数据源: 全市场涨跌家数（敬请期待）→ 目前默认正常模式")

        # ── K线图 (含HMA趋势线) ──
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from viz.theme import TEMPLATE, DARK_LAYOUT, COLORS
        from signals import indicators as ind

        kline_days = st.slider("显示交易日数", 60, 500, 120, 20, key="kline_days",
                               help="拖动调整K线图显示的交易日数量")
        df_plot = df_daily.iloc[-kline_days:].copy()
        hma20 = ind.hma(df_plot["close"], 20)
        hma50 = ind.hma(df_plot["close"], 50)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"],
            name="K线", increasing_line_color=COLORS["red"],
            decreasing_line_color=COLORS["green"],
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df_plot.index, y=hma20, mode="lines",
            line=dict(color=COLORS["blue"], width=1.5), name="HMA20",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=hma50, mode="lines",
            line=dict(color=COLORS["orange"], width=1.5), name="HMA50",
        ), row=1, col=1)

        # 标注金叉点 (买入信号)
        golden = (hma20 > hma50) & (hma20.shift(1) <= hma50.shift(1))
        golden_dates = df_plot.index[golden.values]
        if len(golden_dates) > 0:
            fig.add_trace(go.Scatter(
                x=golden_dates, y=df_plot.loc[golden_dates, "low"] * 0.97,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=COLORS["red"]),
                name="HMA金叉 (买入)",
            ), row=1, col=1)

        # 标注死叉点 (卖出信号)
        death = (hma20 < hma50) & (hma20.shift(1) >= hma50.shift(1))
        death_dates = df_plot.index[death.values]
        if len(death_dates) > 0:
            fig.add_trace(go.Scatter(
                x=death_dates, y=df_plot.loc[death_dates, "high"] * 1.03,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=COLORS["green"]),
                name="HMA死叉 (卖出)",
            ), row=1, col=1)

        # 成交量
        vol_colors = [COLORS["green"] if df_plot["close"].iloc[i] < df_plot["open"].iloc[i]
                      else COLORS["red"] for i in range(len(df_plot))]
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot["volume"], marker_color=vol_colors,
            name="成交量", showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            template=TEMPLATE, **DARK_LAYOUT,
            title=f"{val_name} ({scan_code}) — K线 + HMA趋势线",
            height=550, xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        # 隐藏周末/非交易日空白
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, width="stretch", config={
            "displayModeBar": True, "displaylogo": False,
        })

        # ── 行业资金动向（敬请期待）──
        st.divider()
        st.caption("📊 **行业资金动向** — 敬请期待")
        st.caption("追踪行业板块资金流入/流出，提前判断热点轮动方向。")
        st.caption("数据就绪: 东财行业板块涨跌排名 · 概念板块排名 · 同花顺题材归因")
        st.caption("等待后续迭代接入。")


# ═══════════════════════════════════════════════════════════
# Tab 2: 持仓权重
# ═══════════════════════════════════════════════════════════
with tab2:
    # 准备收益率矩阵 (用日线价格计算)
    price_data = {}
    for code in selected:
        d = daily_data.get(code)
        if d is not None and "close" in d.columns:
            price_data[code] = d["close"]

    if len(price_data) < 1:
        st.warning("至少需要1只有效数据的股票。")
    else:
        prices_df = pd.DataFrame(price_data).dropna()
        returns_df = prices_df.pct_change().dropna().clip(-0.5, 0.5)

        if len(returns_df) < 60:
            st.warning(f"有效交易日不足 ({len(returns_df)} < 60)，结果仅供参考。")

        ff3 = fetch_ff3_daily()
        rf_daily = get_rf_daily(ff3)

        mu, cov, rf_annual = annualize(returns_df, rf_daily)
        n_assets = len(returns_df.columns)
        ticker_names = list(price_data.keys())
        bounds = (0.0, 1.0)
        latest_prices = {c: float(price_data[c].iloc[-1]) for c in ticker_names}

        st.caption(f"基于 {len(returns_df)} 个交易日 | 无风险利率 {rf_annual:.2%}")

        # ── 最优权重计算 ──
        w_eq = np.ones(n_assets) / n_assets
        eq_stats = portfolio_stats(w_eq, mu, cov, rf_annual)

        try:
            w_mv = min_variance(cov, bounds)
            mv_stats = portfolio_stats(w_mv, mu, cov, rf_annual)
            w_ms = max_sharpe(mu, cov, rf_annual, bounds)
            ms_stats = portfolio_stats(w_ms, mu, cov, rf_annual)
        except Exception:
            st.error("优化求解失败，请尝试调整股票组合。")
            st.stop()

        mc_rets, mc_vols, mc_sharpes = monte_carlo(mu, cov, rf_annual, n_portfolios=5000)
        f_rets, f_vols = efficient_frontier(mu, cov, bounds)

        # ═══════════════════════════════════════
        # 区域 1: 手动调仓
        # ═══════════════════════════════════════
        st.subheader("手动调仓")

        tab_weight_mode = st.radio(
            "输入方式", ["百分比权重", "持仓股数"],
            horizontal=True, key="weight_mode"
        )

        manual_weights = np.zeros(n_assets)
        manual_quantities = np.zeros(n_assets, dtype=int)
        total_value = 100000.0

        if tab_weight_mode == "百分比权重":
            cols_wt = st.columns(min(n_assets, 6))
            for i, code in enumerate(ticker_names):
                cname = valuation.get(code, {}).get("name", code)
                with cols_wt[i % len(cols_wt)]:
                    manual_weights[i] = st.number_input(
                        f"{cname}",
                        min_value=0.0, max_value=100.0,
                        value=max(0.0, round(float(w_ms[i]) * 100, 1)),
                        step=0.5, key=f"wt_{code}"
                    ) / 100.0
        else:
            st.caption("输入每只股票的持仓股数，系统会根据最新价格计算权重")
            total_value = st.number_input("总投入资金", 10000.0, 100000000.0, 100000.0, 10000.0,
                                           key="total_capital")
            cols_qt = st.columns(min(n_assets, 6))
            for i, code in enumerate(ticker_names):
                cname = valuation.get(code, {}).get("name", code)
                price = latest_prices.get(code, 0)
                with cols_qt[i % len(cols_qt)]:
                    manual_quantities[i] = st.number_input(
                        f"{cname} (@{price:.0f})",
                        min_value=0, value=100, step=100, key=f"qt_{code}"
                    )

        # ── 计算手动组合指标 ──
        if tab_weight_mode == "百分比权重":
            w_manual = manual_weights
            w_manual = w_manual / w_manual.sum() if w_manual.sum() > 0 else w_manual
        else:
            pos_values = manual_quantities.astype(float) * np.array([latest_prices[c] for c in ticker_names])
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
            for i, code in enumerate(ticker_names):
                cname = valuation.get(code, {}).get("name", code)
                price = latest_prices.get(code, 0)
                mw_rows.append({
                    "代码": code,
                    "名称": cname,
                    "最新价": f"{price:.2f}",
                    "权重": f"{w_manual[i]:.1%}",
                    "股数": int(manual_quantities[i]) if tab_weight_mode == "持仓股数" else int(w_manual[i] * total_value / price) if price > 0 else 0,
                })
            st.dataframe(pd.DataFrame(mw_rows), width="stretch", hide_index=True)

        st.divider()

        # ═══════════════════════════════════════
        # 区域 2: Markowitz 最优权重
        # ═══════════════════════════════════════
        st.subheader("Markowitz 最优权重 (参考)")

        c1, c2, c3 = st.columns(3)
        c1.metric("最大 Sharpe", f"{ms_stats['sharpe']:.2f}",
                  f"收益 {ms_stats['ret']:.1%} | 波动 {ms_stats['vol']:.1%}")
        c2.metric("最小方差", f"{mv_stats['vol']:.1%}",
                  f"收益 {mv_stats['ret']:.1%} | Sharpe {mv_stats['sharpe']:.2f}")
        c3.metric("等权重", f"{eq_stats['sharpe']:.2f}",
                  f"收益 {eq_stats['ret']:.1%} | 波动 {eq_stats['vol']:.1%}")

        # 有效前沿
        col_ef, col_wt = st.columns([2, 1])

        with col_ef:
            fig_ef = efficient_frontier_chart(
                mu, cov, rf_annual, ticker_names,
                mc_rets, mc_vols, mc_sharpes,
                f_rets, f_vols, eq_stats, mv_stats, ms_stats,
                n_mc=5000)

            # 加上手动组合点
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

        # 权重对比
        with st.expander("最优权重 vs 手动权重对比"):
            comp_rows = []
            for i, code in enumerate(ticker_names):
                cname = valuation.get(code, {}).get("name", code)
                comp_rows.append({
                    "代码": code,
                    "名称": cname,
                    "最大Sharpe": w_ms[i],
                    "最小方差": w_mv[i],
                    "等权重": w_eq[i],
                    "手动调仓": w_manual[i],
                })
            comp_df = pd.DataFrame(comp_rows).sort_values("手动调仓", ascending=False)
            for col in ["最大Sharpe", "最小方差", "等权重", "手动调仓"]:
                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.1%}")
            st.dataframe(comp_df, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════
# Tab 3: 因子分析 (CH-3 中国版三因子)
# ═══════════════════════════════════════════════════════════
with tab3:
    st.subheader("因子分析")

    # 准备收益数据
    price_data = {}
    for code in selected:
        d = daily_data.get(code)
        if d is not None and "close" in d.columns:
            price_data[code] = d["close"]

    if not price_data:
        st.warning("无有效价格数据。")
    else:
        prices_df = pd.DataFrame(price_data).dropna()
        returns_df = prices_df.pct_change().dropna().clip(-0.5, 0.5)

        ff3 = fetch_ff3_daily()
        rf_daily = get_rf_daily(ff3)

        # 超额收益
        excess_returns = returns_df.sub(rf_daily, axis=0)

        # 沪深300 基准
        csi300_ret = fetch_csi300_returns()
        cn_mkt_excess = None
        if csi300_ret is not None:
            cn_idx = csi300_ret.index.intersection(rf_daily.index)
            cn_mkt_excess = csi300_ret.loc[cn_idx].astype(float) - rf_daily.loc[cn_idx].astype(float)
            cn_mkt_excess = cn_mkt_excess.dropna()

        # ── CAPM (单因子: 沪深300) ──
        st.subheader("CAPM 回归 (沪深300基准)")
        capm_df = run_capm_batch(excess_returns, None, cn_mkt_excess,
                                  cn_tickers=list(returns_df.columns))

        if not capm_df.empty:
            col_l, col_r = st.columns([2, 1])
            with col_l:
                fig_capm = capm_beta_alpha(capm_df)
                st.plotly_chart(fig_capm, width="stretch", config={
                    "displayModeBar": True, "displaylogo": False,
                })
            with col_r:
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

        # ── CH-3 (三因子: A股本地市值+价值) ──
        st.subheader("CH-3 中国版三因素回归")
        st.caption("Liu, Stambaugh & Yuan (2019) — 剔除壳污染 + EP价值因子，本地化A股因子")

        with st.spinner("构建CH-3因子中 (拉取约25只成分股数据)..."):
            ch3_daily, ch3_val = load_ch3_universe()
            ch3_factors = build_ch3_factors(ch3_daily, ch3_val)

        if ch3_factors is not None and len(ch3_factors) > 60:
            ch3_df = run_ch3_batch(excess_returns, ch3_factors)

            if not ch3_df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    fig_ch3 = ff3_factor_loadings(ch3_df)
                    fig_ch3.update_layout(title="CH-3 三因素因子载荷 (A股本地因子)")
                    st.plotly_chart(fig_ch3, width="stretch", config={
                        "displayModeBar": True, "displaylogo": False,
                    })
                with c2:
                    if not capm_df.empty:
                        common_ch3 = capm_df.index.intersection(ch3_df.index)
                        fig_r2 = model_r2_comparison(
                            capm_df.loc[common_ch3], ch3_df.loc[common_ch3],
                            multi_label="CH-3 R²")
                        fig_r2.update_layout(title="CAPM vs CH-3 拟合优度对比")
                        st.plotly_chart(fig_r2, width="stretch", config={
                            "displayModeBar": True, "displaylogo": False,
                        })

                display_ch3 = ch3_df[["beta_mkt", "beta_smb", "beta_hml",
                                       "alpha_annual", "r_squared"]].copy()
                display_ch3["alpha_annual"] = display_ch3["alpha_annual"].apply(
                    lambda x: f"{x*100:+.2f}%")
                display_ch3 = display_ch3.rename(columns={
                    "beta_mkt": "β_Mkt", "beta_smb": "β_SMB(规模)",
                    "beta_hml": "β_HML(价值)", "alpha_annual": "Alpha(年化)",
                    "r_squared": "R²",
                })
                st.dataframe(display_ch3, width="stretch")

                # CH-3 因子统计
                with st.expander("CH-3 因子统计信息"):
                    fstats = ch3_factors.describe()
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("因子构建股票数", len(ch3_daily))
                    fc2.metric("因子观测天数", len(ch3_factors))
                    fc3.metric("MKT日均收益", f"{ch3_factors['MKT'].mean()*100:.3f}%")

                    st.caption(
                        "**因子构建方法** (月频调仓):\n"
                        "- MKT: 成分股等权平均日收益\n"
                        "- SMB: 小盘组合 - 大盘组合 (剔除末30%壳污染, 市值中位数分界)\n"
                        "- HML: 高EP组合 - 低EP组合 (EP=1/PE, 前30% vs 后30%)\n"
                        "- 细分组合: SH(小盘价值), SL(小盘成长), BH(大盘价值), BL(大盘成长)"
                    )
            else:
                st.warning("CH-3 回归失败：有效样本不足。")
        else:
            st.warning("CH-3 因子构建失败：成分股数据不足。")

st.divider()
st.caption("数据源: 东财 (K线) + 腾讯财经 (估值) + CH-3 A股本地因子")
