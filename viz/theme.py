"""暗色 Plotly 主题常量。"""
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"

DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", size=12, family="Arial Unicode MS, Noto Sans SC, SimHei, sans-serif"),
    title_font=dict(color="#e6edf3", family="Arial Unicode MS, Noto Sans SC, SimHei, sans-serif"),
    modebar=dict(orientation="v", bgcolor="rgba(13,17,23,0.7)"),
)

COLORS = {
    "blue": "#3b82f6",
    "green": "#22c55e",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "purple": "#8b5cf6",
    "gray": "#8b949e",
    "pink": "#ec4899",
    "cyan": "#06b6d4",
}

CHART_HEIGHT = 500
MODE_BAR_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "sendDataToCloud"],
}
