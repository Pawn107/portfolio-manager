"""portfolio-manager 公共组件：暗色 CSS + 工具函数。"""
import streamlit as st


def inject_css():
    """注入暗色主题 CSS，所有页面统一调用。"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
        .stApp { background-color: #0d1117; }
        .main .block-container { padding-top: 1.5rem; }
        h1, h2, h3 { color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace; }
        p, li, label, span { color: #c9d1d9; }
        [data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 12px 16px; }
        [data-testid="stMetric"] label { color: #8b949e !important; font-size: 0.75rem; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3 !important;
            font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; }
        .stDataFrame { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
        section[data-testid="stSidebar"] { background: #161b22; }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
            color: #c9d1d9 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] {
            color: #8b949e; background: #161b22; border: 1px solid #30363d;
            border-radius: 6px 6px 0 0; padding: 8px 20px;
        }
        .stTabs [aria-selected="true"] {
            color: #e6edf3 !important; background: #0d1117 !important;
            border-bottom: 2px solid #3b82f6 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, caption: str = ""):
    """统一页面标题。"""
    st.title(title)
    if caption:
        st.caption(caption)


def fmt_pct(x: float) -> str:
    """格式化百分比。"""
    return f"{x*100:+.2f}%" if abs(x) < 1 else f"{x:.1%}"


def fmt_num(x: float, decimals: int = 2) -> str:
    """格式化数字。"""
    return f"{x:.{decimals}f}"
