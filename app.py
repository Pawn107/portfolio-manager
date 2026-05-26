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

    - 东财 + 腾讯财经 + 新浪 + yfinance 数据
    - 多因子买卖信号
    - 周频回测 (含手续费)
    - 沪深300 基准
    """)
    if st.button("进入A股分析", type="primary", use_container_width=True):
        st.switch_page("pages/cn_stocks.py")

st.divider()

# ── 使用说明 ──
with st.expander("使用说明"):
    st.markdown("""
    ### 基本操作
    - **更换组合**：在侧边栏修改股票代码（一行一个），系统会自动更新组合和图表
    - **保存/分享组合**：修改代码后浏览器地址栏 URL 会自动更新（如 `?codes=600519,000333`），收藏链接或复制发送即可保存和分享当前组合
    - **调整K线时间范围**：在"信号扫描"页面可以拖动滑块调整显示的K线天数（60/120/250/500个交易日）
    - **手动调仓**：在"持仓权重"页面可以输入百分比权重或具体股数，实时查看组合指标

    ### 重要提示：非实时数据
    本系统是**分析工具**，不是看盘软件，所有数据不会自动刷新：
    - **A股日K**：每日收盘后更新（约 15:30 后）
    - **美股日K**：美股收盘后更新（约次日凌晨 5:00 后）
    - **估值数据**（PE/PB/市值）：基于最近交易日
    - **财务数据**：基于最新季报，季度更新
    - 如需查看最新数据，请手动刷新页面（`F5` 或浏览器刷新按钮）

    ### 交易时段
    - **A股**：工作日 9:30–11:30, 13:00–15:00
    - **美股**：夏令时 21:30–次日 4:00，冬令时 22:30–次日 5:00
    - **K线图**：已自动隐藏周末和非交易日空白，图表连续显示

    ### 功能概览
    - **美股分析**：CAPM 回归 + Fama-French 三因子 + Markowitz 最优组合 + 手动调仓
    - **A股分析**：
      - 信号扫描：基本面四维打分（PE/PB/ROE/市值）+ 技术风控 → 买入/持有/卖出信号
      - 持仓权重：手动调仓 + Markowitz 最优权重参考 + 有效前沿
      - 因子分析：CAPM（沪深300基准）+ CH-3 中国版三因子
    """)

st.caption("数据源: yfinance (美股) | 东财·腾讯·新浪·yfinance (A股) | Kenneth French Data Library (FF3因子)")
