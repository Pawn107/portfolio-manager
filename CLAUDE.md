# portfolio-manager 项目规范

## 项目
A股+美股持仓管理系统，覆盖"选股 → 信号 → 配权 → 回测 → 评估"全流程。
- **看板**: `:8502` (Streamlit) | **Python**: 3.12
- **数据源**: yfinance (美股) | mootdx + 腾讯财经 (A股) | Kenneth French Data Library (FF3因子)

## 架构
```
app.py              → 入口导航页 (两个按钮: 美股/A股)
pages/
  us_stocks.py      → 美股: CAPM + FF3 + Markowitz
  cn_stocks.py      → A股: 信号扫描 + 权重 + 回测 + 因子分析 (4 tabs)
data/
  cache.py          → CSV 缓存, TTL 24h
  us_fetcher.py     → yfinance 美股下载
  cn_fetcher.py     → mootdx K线 + 腾讯财经估值 + 沪深300
  ff3_fetcher.py    → FF3 因子 (French Data Library)
domain/
  capm.py           → CAPM 回归 (双市场: S&P500 / 沪深300)
  fama_french.py    → FF3 回归
  portfolio.py      → Markowitz 均值-方差优化
  backtest.py       → 周频回测引擎 (含手续费/滑点/止损止盈)
signals/
  indicators.py     → HMA, RSI, MACD, ATR, 量价, 振幅, 波动率情绪
  scoring.py        → 多因子打分 (0-100) → BUY/HOLD/SELL
  risk.py           → 入场过滤 + 持仓风控
viz/
  theme.py          → Plotly 暗色主题常量
  charts.py         → CAPM/FF3/有效前沿/权重图
  signal_charts.py  → K线+信号标注 + 资金曲线 + 得分走势
shared.py           → 暗色CSS + 公共组件
config.py           → 全局配置
```

## A股数据方案
| 数据 | 来源 | 方式 |
|------|------|------|
| 日K/周K | mootdx | TCP 7709, `client.bars(category=4/5)` |
| PE/PB/市值/换手率 | 腾讯财经 qt.gtimg.cn | HTTP GET, GBK解码 |
| 财务快照 | mootdx finance | `client.finance(symbol)` |
| 沪深300 | 腾讯财经 K线 | HTTP JSON |
| FF3因子 | French Data Library | HTTP ZIP |

## 信号系统
- 7因子打分: HMA趋势(25) + RSI(15) + MACD(20) + 量价(15) + 波动率(10) + 振幅(5) + 量能情绪(10) = 满分100
- BUY ≥ 70, HOLD 40-69, SELL < 40
- 风控: 止损 -8%, 止盈 +25%, 移动止盈 -5%, 最大持有12周

## 回测引擎
- 频率: 周频 (周五收盘)
- 手续费: 买入 0.03%, 卖出 0.13% (万三+千一印花税)
- 滑点: 0.1%
- 输出: 胜率, 盈亏比, 最大回撤, Sharpe, 年化收益

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
