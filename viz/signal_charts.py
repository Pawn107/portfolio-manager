"""信号图表：K线 + 信号标注 + 资金曲线。"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from viz.theme import TEMPLATE, DARK_LAYOUT, COLORS


def _dark_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=TEMPLATE, **DARK_LAYOUT)
    return fig


def kline_with_signals(df: pd.DataFrame, scores: pd.DataFrame,
                        title: str = "K线图与买卖信号") -> go.Figure:
    """K线图 + 信号标注 (BUY 绿色箭头, SELL 红色箭头)。

    Args:
        df: 日K DataFrame (date index, 含 open/high/low/close/volume)
        scores: 日频信号 DataFrame (含 signal 列)
    """
    # 取最近 120 个交易日
    df_plot = df.iloc[-120:].copy()
    scores_plot = scores.iloc[-120:].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"],
        name="K线",
        increasing_line_color=COLORS["red"],
        decreasing_line_color=COLORS["green"],
    ), row=1, col=1)

    # BUY 标记
    buy_dates = scores_plot[scores_plot["signal"] == "BUY"].index
    if len(buy_dates) > 0:
        buy_prices = df_plot.loc[buy_dates, "low"] * 0.98 if all(
            d in df_plot.index for d in buy_dates
        ) else df_plot["low"].iloc[-1] * 0.98
        fig.add_trace(go.Scatter(
            x=buy_dates, y=df_plot.loc[buy_dates, "low"] * 0.98,
            mode="markers", marker=dict(symbol="triangle-up", size=12, color=COLORS["red"]),
            name="BUY", showlegend=True,
        ), row=1, col=1)

    # SELL 标记
    sell_dates = scores_plot[scores_plot["signal"] == "SELL"].index
    if len(sell_dates) > 0:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=df_plot.loc[sell_dates, "high"] * 1.02,
            mode="markers", marker=dict(symbol="triangle-down", size=12, color=COLORS["green"]),
            name="SELL", showlegend=True,
        ), row=1, col=1)

    # 成交量
    colors_vol = [
        COLORS["red"] if df_plot["close"].iloc[i] >= df_plot["open"].iloc[i]
        else COLORS["green"]
        for i in range(len(df_plot))
    ]
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot["volume"],
        marker_color=colors_vol, name="成交量",
        opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        title=title, height=600,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    # 隐藏周末/非交易日空白
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return _dark_fig(fig)


def equity_curve_chart(result, benchmark_ret: float = None,
                        title: str = "资金曲线") -> go.Figure:
    """资金曲线 + 回撤图。

    Args:
        result: BacktestResult
        benchmark_ret: 基准收益 (buy & hold)
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
    )

    eq = result.equity_curve
    dates = result.dates

    # 确保日期和资金曲线长度一致
    min_len = min(len(dates), len(eq))
    dates = dates[:min_len]
    eq = eq[:min_len]

    # 资金曲线
    fig.add_trace(go.Scatter(
        x=dates, y=eq, mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        name="策略净值",
    ), row=1, col=1)

    # 初始资金线
    if eq:
        fig.add_hline(y=eq[0], line_dash="dash", line_color="gray",
                       opacity=0.5, row=1, col=1)

    # 回撤
    eq_arr = pd.Series(eq, index=dates)
    peak = eq_arr.expanding().max()
    drawdown = (eq_arr - peak) / peak * 100

    fig.add_trace(go.Scatter(
        x=dates, y=drawdown.values, mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["red"], width=1),
        fillcolor="rgba(239,68,68,0.15)",
        name="回撤 %",
    ), row=2, col=1)

    fig.update_layout(title=title, height=500)
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", row=2, col=1)
    # 隐藏周末/非交易日空白
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return _dark_fig(fig)


def signal_score_chart(scores: pd.DataFrame, title: str = "信号得分走势") -> go.Figure:
    """信号得分时序图。"""
    scores_plot = scores.iloc[-120:].copy()

    fig = go.Figure()

    colors = []
    for s in scores_plot["signal"]:
        if s == "BUY":
            colors.append(COLORS["red"])
        elif s == "SELL":
            colors.append(COLORS["green"])
        else:
            colors.append(COLORS["gray"])

    fig.add_trace(go.Scatter(
        x=scores_plot.index, y=scores_plot["total_score"],
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=1.5),
        marker=dict(color=colors, size=6),
        name="总分",
    ))

    # 阈值线
    fig.add_hline(y=60, line_dash="dash", line_color=COLORS["green"],
                   annotation_text="BUY ≥ 60", opacity=0.6)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS["red"],
                   annotation_text="SELL < 30", opacity=0.6)

    fig.update_layout(
        title=title, height=400,
        yaxis_title="得分 (0-100)", yaxis_range=[0, 100],
    )
    # 隐藏周末/非交易日空白
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return _dark_fig(fig)
