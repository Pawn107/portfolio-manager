# portfolio-manager 项目规范

## 项目
A股+美股持仓管理系统，覆盖"信号扫描 → 持仓权重 → 因子分析"全流程。
- **看板**: `:8502` (Streamlit) | **Python**: 3.12
- **数据源**: yfinance (美股) | 东财 (K线) + 腾讯财经 (估值) + 新浪 (财务) | CH-3 A股本地因子

## 协作规则
- **不要每次改完代码就用 Playwright 打开浏览器验证**，除非用户明确要求
- 改完代码 → 提交 git → push 即可，Streamlit Cloud 会自动部署
- 本地开发时运行 `/opt/homebrew/Cellar/python@3.12/3.12.13_2/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /Users/lyu/Projects/quant/legend-timing/.venv/bin/streamlit run app.py --server.port 8502 --server.headless true`

## 架构
```
app.py              → 入口导航页 (两个按钮: 美股/A股)
pages/
  us_stocks.py      → 美股: CAPM + FF3 + Markowitz
  cn_stocks.py      → A股: 信号扫描 + 持仓权重 + 因子分析 (3 tabs)
data/
  cache.py          → CSV 缓存, TTL 24h
  us_fetcher.py     → yfinance 美股下载
  cn_fetcher.py     → 东财 K线 + 腾讯财经估值 + 新浪财务 + 沪深300
  ff3_fetcher.py    → FF3 因子 (French Data Library)
domain/
  capm.py           → CAPM 回归 (双市场: S&P500 / 沪深300)
  fama_french.py    → FF3 回归 (美股用)
  ch3.py            → CH-3 中国版三因子 (A股用: 剔除壳污染 + EP价值因子)
  portfolio.py      → Markowitz 均值-方差优化
  backtest.py       → 周频回测引擎 (当前未使用)
signals/
  indicators.py     → HMA, RSI, MACD, ATR, 量价, 振幅, 波动率情绪
  scoring.py        → 多因子打分 (0-100) → BUY/HOLD/SELL
  risk.py           → 入场过滤 + 持仓风控
viz/
  theme.py          → Plotly 暗色主题常量
  charts.py         → CAPM/FF3/有效前沿/权重图
  signal_charts.py  → K线+信号标注 + 资金曲线 + 得分走势
shared.py           → 暗色CSS + 公共组件
config.py           → 全局配置 + CH3_UNIVERSE
```

## A股数据方案
| 数据 | 来源 | 方式 |
|------|------|------|
| 日K/周K | 东财 push2his | HTTP, 前复权 |
| PE/PB/市值/换手率 | 腾讯财经 qt.gtimg.cn | HTTP GET, GBK解码 |
| 财务快照 | 新浪 quotes.sina.cn | HTTP, 利润表+资产负债表 |
| 沪深300 | yfinance `000300.SS` | 日线close |
| 无风险利率 | FF3 RF列 | French Data Library |

## 信号系统
- 7因子打分: HMA趋势(25) + RSI(15) + MACD(20) + 量价(15) + 波动率(10) + 振幅(5) + 量能情绪(10) = 满分100
- BUY ≥ 60, SELL < 30, 其余 HOLD
- 风控: 止损 -8%, 止盈 +25%, 移动止盈 -5%, 最大持有12周

## CH-3 因子模型 (Liu, Stambaugh & Yuan 2019)
- MKT: 成分股等权平均日收益
- SMB: 小盘−大盘 (市值中位数分界, 剔除底部30%壳污染)
- HML: 高EP−低EP (EP=1/PE, 前30% vs 后30%)
- 月频调仓, 约25只成分股 (CH3_UNIVERSE)

## URL 参数记忆
- 侧边栏改股票代码时自动更新 URL: `?codes=600519,000333`
- 页面加载时从 URL 读取默认代码，用户可收藏链接保留组合
- A股和美股页面独立记忆

## Plotly 暗色规范
- paper_bgcolor=#0d1117, plot_bgcolor=#0d1117
- modebar: orientation="v", bgcolor="rgba(13,17,23,0.7)"
- 字体: Arial Unicode MS, Noto Sans SC, SimHei, sans-serif
- 色板: blue=#3b82f6, green=#22c55e, red=#ef4444, orange=#f59e0b

## 启动
```bash
cd /Users/lyu/Projects/quant/portfolio-manager
streamlit run app.py --server.port 8502
```
