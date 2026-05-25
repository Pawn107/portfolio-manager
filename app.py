"""portfolio-manager — 持仓管理系统入口。
启动: streamlit run app.py --server.port 8502
"""
import streamlit as st
from shared import inject_css

st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# ── 标题 ──
st.title("Portfolio Manager")
st.caption("持仓管理系统 — 选股 · 信号 · 配权 · 回测 · 评估")

st.divider()

# ── 导航入口 ──
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    ### 美股分析
    CAPM · Fama-French 三因素 · Markowitz 最优投资组合

    - yfinance 数据源
    - S&P 500 市场基准
    - FF3 因子回归
    - 有效前沿 + 蒙特卡洛
    """)
    if st.button("进入美股分析", type="primary", use_container_width=True):
        st.switch_page("pages/us_stocks.py")

with col2:
    st.markdown("""
    ### A股分析
    信号扫描 · 持仓权重 · 回测 · 因子分析

    - mootdx + 腾讯财经数据
    - 多因子买卖信号
    - 周频回测 (含手续费)
    - 沪深300 基准
    """)
    if st.button("进入A股分析", type="primary", use_container_width=True):
        st.switch_page("pages/cn_stocks.py")

st.divider()
st.caption("数据源: yfinance (美股) | mootdx + 腾讯财经 (A股) | Kenneth French Data Library (FF3因子)")
