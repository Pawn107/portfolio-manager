"""Plotly 图表工厂：CAPM、FF3、有效前沿、组合权重。"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from viz.theme import TEMPLATE, DARK_LAYOUT, COLORS, CHART_HEIGHT


def _dark_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=TEMPLATE, **DARK_LAYOUT)
    return fig


def capm_beta_alpha(capm_df):
    """CAPM Beta + Alpha 双柱状图。"""
    tickers = capm_df.index.tolist()
    betas = capm_df["beta"].values
    alphas = capm_df["alpha_annual"].values * 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Beta 系数", "年化 Alpha (%)"),
                         horizontal_spacing=0.15)

    beta_colors = [COLORS["red"] if b > 1 else COLORS["blue"] for b in betas]
    fig.add_trace(go.Bar(
        y=tickers, x=betas, orientation="h", marker_color=beta_colors,
        text=[f"{b:.2f}" for b in betas], textposition="outside",
        name="Beta", showlegend=False,
    ), row=1, col=1)
    fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    alpha_colors = [COLORS["green"] if a > 0 else COLORS["red"] for a in alphas]
    fig.add_trace(go.Bar(
        y=tickers, x=alphas, orientation="h", marker_color=alpha_colors,
        text=[f"{a:+.1f}%" for a in alphas],
        textposition=["auto" if abs(a) < max(abs(alphas)) * 0.8 else "outside"
                       for a in alphas],
        name="Alpha", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=0, line_color="gray", opacity=0.5, row=1, col=2)

    fig.update_layout(title="CAPM 回归结果", height=CHART_HEIGHT)
    fig.update_yaxes(autorange="reversed")
    return _dark_fig(fig)


def ff3_factor_loadings(ff3_df):
    """FF3 因子载荷图。"""
    tickers = ff3_df.index.tolist()
    x = np.arange(len(tickers))
    w = 0.25

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x - w, y=ff3_df["beta_mkt"], name="β_Mkt (市场)",
                         marker_color=COLORS["blue"], width=w * 2))
    fig.add_trace(go.Bar(x=x, y=ff3_df["beta_smb"], name="β_SMB (规模)",
                         marker_color=COLORS["orange"], width=w * 2))
    fig.add_trace(go.Bar(x=x + w, y=ff3_df["beta_hml"], name="β_HML (价值)",
                         marker_color=COLORS["green"], width=w * 2))

    fig.update_xaxes(tickvals=x.tolist(), ticktext=tickers)
    fig.add_hline(y=0, line_color="gray", line_width=0.5)

    fig.update_layout(title="Fama-French 三因素因子载荷", height=CHART_HEIGHT,
                       yaxis_title="因子载荷 (Factor Loading)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _dark_fig(fig)


def model_r2_comparison(capm_df, multi_df, multi_label: str = "多因子 R²"):
    """CAPM vs 多因子模型 R² 对比图。"""
    tickers = capm_df.index.tolist()
    x = np.arange(len(tickers))
    w = 0.35

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x - w/2, y=capm_df["r_squared"], name="CAPM R²",
                         marker_color=COLORS["blue"], width=w))
    fig.add_trace(go.Bar(x=x + w/2, y=multi_df["r_squared"], name=multi_label,
                         marker_color=COLORS["orange"], width=w))

    fig.update_xaxes(tickvals=x.tolist(), ticktext=tickers)

    fig.update_layout(title="模型拟合优度对比", height=CHART_HEIGHT,
                       yaxis_title="R²",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _dark_fig(fig)


def efficient_frontier_chart(mu, cov, rf, ticker_names,
                              mc_rets, mc_vols, mc_sharpes,
                              frontier_rets, frontier_vols,
                              eq_stats, mv_stats, ms_stats):
    """有效前沿 + CML + 蒙特卡洛散点。"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets, mode="markers",
        marker=dict(color=mc_sharpes, colorscale="RdYlGn", size=3, opacity=0.3,
                    showscale=True, colorbar=dict(title="Sharpe")),
        name="随机组合", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets, mode="lines",
        line=dict(color=COLORS["blue"], width=3),
        name="有效前沿",
    ))

    vols_individual = np.sqrt(np.diag(cov))
    fig.add_trace(go.Scatter(
        x=vols_individual, y=mu, mode="markers+text",
        marker=dict(color=COLORS["gray"], size=8, symbol="square"),
        text=ticker_names, textposition="top center", textfont=dict(size=9),
        name="个股", showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=[ms_stats["vol"]], y=[ms_stats["ret"]], mode="markers",
        marker=dict(color=COLORS["red"], size=16, symbol="star"),
        name=f"最大Sharpe (SR={ms_stats['sharpe']:.2f})",
    ))
    fig.add_trace(go.Scatter(
        x=[mv_stats["vol"]], y=[mv_stats["ret"]], mode="markers",
        marker=dict(color=COLORS["green"], size=12, symbol="triangle-up"),
        name="最小方差",
    ))
    fig.add_trace(go.Scatter(
        x=[eq_stats["vol"]], y=[eq_stats["ret"]], mode="markers",
        marker=dict(color=COLORS["blue"], size=12, symbol="diamond"),
        name="等权重",
    ))

    max_vol = max(mc_vols.max(), ms_stats["vol"] * 1.5)
    cml_x = np.array([0, max_vol])
    cml_y = rf + (ms_stats["ret"] - rf) / ms_stats["vol"] * cml_x
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines",
        line=dict(color=COLORS["red"], dash="dash", width=1.5),
        name="资本市场线 (CML)",
    ))

    fig.update_layout(
        title="Markowitz 有效前沿", height=600,
        xaxis_title="年化波动率", yaxis_title="年化期望收益",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        legend=dict(font=dict(size=10)),
    )
    return _dark_fig(fig)


def portfolio_weights_chart(ticker_names, w_mv, w_ms, w_eq):
    """最优组合权重分配图。"""
    x = np.arange(len(ticker_names))
    w_bar = 0.25

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x - w_bar, y=w_ms, name="最大Sharpe组合",
                         marker_color=COLORS["red"], width=w_bar))
    fig.add_trace(go.Bar(x=x, y=w_mv, name="最小方差组合",
                         marker_color=COLORS["green"], width=w_bar))
    fig.add_trace(go.Bar(x=x + w_bar, y=w_eq, name="等权重组合",
                         marker_color=COLORS["blue"], width=w_bar))

    fig.update_xaxes(tickvals=x.tolist(), ticktext=ticker_names)

    fig.update_layout(
        title="最优投资组合权重分配", height=CHART_HEIGHT,
        yaxis_title="权重", yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        bargap=0.15,
    )
    return _dark_fig(fig)
