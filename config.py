"""portfolio-manager 全局配置常量。"""
import os

# ── 股票宇宙 (参考池，非硬限制) ──
US_POOL = {
    "GOOGL": "Alphabet", "TSLA": "Tesla", "BRK-B": "Berkshire Hathaway",
    "XOM": "Exxon Mobil", "PG": "Procter & Gamble", "WMT": "Walmart",
    "UNH": "UnitedHealth", "AAPL": "Apple", "MSFT": "Microsoft",
    "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta",
    "JPM": "JPMorgan Chase", "JNJ": "Johnson & Johnson", "V": "Visa",
    "MA": "Mastercard", "DIS": "Disney", "NFLX": "Netflix",
    "ADBE": "Adobe", "CRM": "Salesforce",
}

CN_POOL = {
    "600519": "贵州茅台", "000333": "美的集团", "300750": "宁德时代",
    "000858": "五粮液", "601318": "中国平安", "600036": "招商银行",
    "000568": "泸州老窖", "601166": "兴业银行", "600276": "恒瑞医药",
    "002415": "海康威视", "300124": "汇川技术", "600900": "长江电力",
    "601899": "紫金矿业", "002594": "比亚迪", "601398": "工商银行",
    "600030": "中信证券",
}


def cn_to_yf(ticker: str) -> str:
    """A股代码 → yfinance ticker。6开头→.SS, 其他→.SZ。已有后缀则不重复。"""
    if ticker.endswith((".SS", ".SZ")):
        return ticker
    return f"{ticker}.SS" if ticker.startswith("6") else f"{ticker}.SZ"


TICKER_NAMES = {
    **{k: f"{v}(美股)" for k, v in US_POOL.items()},
    **{k: f"{v}(A股)" for k, v in CN_POOL.items()},
}

# ── 市场指数 ──
US_MARKET = "^GSPC"
CN_MARKET = "000300.SS"

# ── 默认参数 ──
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
RISK_FREE_RATE = 0.045
TRADING_DAYS = 252

# ── 路径 ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CACHE_DIR = os.path.join(PROJECT_DIR, "data", ".cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 中文字体 (跨平台) ──
CN_FONT = "Arial Unicode MS, Noto Sans SC, SimHei, sans-serif"

# ── mootdx 服务器 ──
MOTDX_SERVER = ("110.41.147.114", 7709)

# ── 回测参数 ──
BACKTEST_CONFIG = {
    "frequency": "weekly",        # 周频
    "commission_buy": 0.0003,     # 万三佣金
    "commission_sell": 0.0013,    # 万三 + 千一印花税
    "slippage": 0.001,            # 0.1% 滑点
    "stop_loss": -0.08,           # 止损 -8%
    "take_profit": 0.25,          # 止盈 +25%
    "max_hold_weeks": 12,         # 最大持有周数
}
