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
    fetch_csi300_returns,
)
from data.ff3_fetcher import fetch_ff3_daily, get_rf_daily
from signals.scoring import compute_scores, latest_signal, signal_summary
from signals.risk import check_entry
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
from viz.signal_charts import kline_with_signals, signal_score_chart

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
    st.caption("数据源: mootdx + 腾讯财经")

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
    st.error("未能加载任何A股数据，请检查网络或 mootdx 连接。")
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
                st.error(f"无法获取 {scan_code} 的K线数据，请检查代码是否正确或 mootdx 连接。")

        if val is None:
            new_val = tencent_quote([scan_code])
            val = new_val.get(scan_code, {})
            if val and scan_code not in valuation:
                valuation[scan_code] = val

        # 计算信号
        with st.spinner("计算技术指标和信号..."):
            scores = compute_scores(df_daily)
            sig = latest_signal(scores)
            summary = signal_summary(scores)

        val_name = val.get("name", scan_code) if val else scan_code

        # 最新信号卡片
        st.subheader(f"{val_name} ({scan_code})")

        cols = st.columns(5)
        signal_color = {
            "BUY": "green", "SELL": "red", "HOLD": "gray",
        }
        color = signal_color.get(sig["signal"], "gray")
        cols[0].metric("最新信号", sig["signal"])
        cols[1].metric("综合得分", f"{sig['total_score']:.1f} / 100")
        cols[2].metric("最新收盘价", f"{sig['close']:.2f}")
        if val:
            cols[3].metric("PE(TTM)", f"{val['pe_ttm']:.1f}" if val.get("pe_ttm") else "N/A")
            cols[4].metric("总市值(亿)", f"{val['mcap_yi']:.0f}" if val.get("mcap_yi") else "N/A")

        # 信号统计
        st.caption(
            f"统计: BUY {summary['buy_count']}天 ({summary['buy_pct']}%) | "
            f"HOLD {summary['hold_count']}天 | "
            f"SELL {summary['sell_count']}天 ({summary['sell_pct']}%)"
        )

        # K线 + 信号图
        fig_kline = kline_with_signals(df_daily, scores,
                                        title=f"{val_name} ({scan_code}) — K线与买卖信号")
        st.plotly_chart(fig_kline, width="stretch", config={
            "displayModeBar": True, "displaylogo": False,
        })

        # 信号得分走势
        fig_score = signal_score_chart(scores, title=f"{val_name} — 信号得分走势")
        st.plotly_chart(fig_score, width="stretch", config={
            "displayModeBar": True, "displaylogo": False,
        })

        # 因子明细
        with st.expander("最新因子得分明细"):
            detail = sig.get("detail", {})
            factor_names = {
                "trend": "HMA趋势", "rsi": "RSI位置", "macd": "MACD交叉",
                "volume": "量价关系", "volatility": "波动率情绪",
                "amplitude": "振幅情绪", "volume_emotion": "量能情绪",
            }
            detail_rows = []
            for k, v in detail.items():
                max_score = {"trend": 25, "rsi": 15, "macd": 20, "volume": 15,
                             "volatility": 10, "amplitude": 5, "volume_emotion": 10}.get(k, 0)
                detail_rows.append({
                    "因子": factor_names.get(k, k),
                    "得分": f"{v:.1f} / {max_score}",
                    "占比": f"{v/max_score*100:.0f}%" if max_score > 0 else "-",
                })
            st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)

        # 入场检查
        with st.expander("入场风控检查"):
            pe = val.get("pe_ttm") if val else None
            ok, reason = check_entry(df_daily, pe)
            if ok:
                st.success("✅ 入场条件满足")
            else:
                st.error(f"❌ {reason}")


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
                        value=round(float(w_ms[i]) * 100, 1),
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
                f_rets, f_vols, eq_stats, mv_stats, ms_stats)

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
st.caption("数据源: mootdx (K线) + 腾讯财经 (估值) + CH-3 A股本地因子")
