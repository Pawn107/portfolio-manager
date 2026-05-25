"""A股数据下载：mootdx (K线/财务) + 腾讯财经 (PE/PB/市值/换手率)。"""
import urllib.request
import time
import numpy as np
import pandas as pd
from mootdx.quotes import Quotes
from config import MOTDX_SERVER
from data.cache import get as cache_get, put as cache_put


def _get_client():
    """获取 mootdx 客户端。"""
    return Quotes.factory(market="std", server=MOTDX_SERVER)


# ════════════════════════════════════════════════════════════
#  mootdx K线
# ════════════════════════════════════════════════════════════

def fetch_kline(symbol: str, category: int = 4, offset: int = 500) -> pd.DataFrame | None:
    """下载K线数据。

    Args:
        symbol: 6位代码
        category: 4=日线, 5=周线
        offset: 获取最近多少根K线
    """
    try:
        client = _get_client()
        df = client.bars(symbol=symbol, category=category, offset=offset)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"datetime": "date"}).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return None
        return df[required].astype(float)
    except Exception:
        return None


def fetch_daily_kline(symbol: str) -> pd.DataFrame | None:
    """获取日K线（缓存1小时）。"""
    cache_key = f"cn_daily_{symbol}"
    cached = cache_get(cache_key, ttl_hours=1)
    if cached is not None and not cached.empty:
        return cached

    df = fetch_kline(symbol, category=4, offset=500)
    if df is not None:
        cache_put(cache_key, df)
    return df


def fetch_weekly_kline(symbol: str) -> pd.DataFrame | None:
    """获取周K线（缓存1小时）。"""
    cache_key = f"cn_weekly_{symbol}"
    cached = cache_get(cache_key, ttl_hours=1)
    if cached is not None and not cached.empty:
        return cached

    df = fetch_kline(symbol, category=5, offset=200)
    if df is not None:
        cache_put(cache_key, df)
    return df


# ════════════════════════════════════════════════════════════
#  腾讯财经 — PE/PB/市值/换手率
# ════════════════════════════════════════════════════════════

def tencent_quote(codes: list[str]) -> dict[str, dict]:
    """批量拉取腾讯财经实时行情。返回 {code: {name, price, pe_ttm, pb, mcap, ...}}。

    也支持指数 (000001, 000300, 399006) 和 ETF (510050, 510300)。
    """
    prefixed = []
    for c in codes:
        if c.startswith(("6", "9")):
            prefixed.append(f"sh{c}")
        elif c.startswith("8"):
            prefixed.append(f"bj{c}")
        else:
            prefixed.append(f"sz{c}")

    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("gbk")
    except Exception:
        return {}

    result = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]
        result[code] = {
            "name": vals[1],
            "price": float(vals[3]) if vals[3] else 0,
            "last_close": float(vals[4]) if vals[4] else 0,
            "open": float(vals[5]) if vals[5] else 0,
            "change_pct": float(vals[32]) if vals[32] else 0,
            "high": float(vals[33]) if vals[33] else 0,
            "low": float(vals[34]) if vals[34] else 0,
            "amount_wan": float(vals[37]) if vals[37] else 0,
            "turnover_pct": float(vals[38]) if vals[38] else 0,
            "pe_ttm": float(vals[39]) if vals[39] else 0,
            "mcap_yi": float(vals[44]) if vals[44] else 0,
            "float_mcap_yi": float(vals[45]) if vals[45] else 0,
            "pb": float(vals[46]) if vals[46] else 0,
            "limit_up": float(vals[47]) if vals[47] else 0,
            "limit_down": float(vals[48]) if vals[48] else 0,
            "vol_ratio": float(vals[49]) if vals[49] else 0,
            "pe_static": float(vals[52]) if vals[52] else 0,
        }
    return result


# ════════════════════════════════════════════════════════════
#  mootdx 财务快照
# ════════════════════════════════════════════════════════════

def fetch_finance(symbol: str) -> dict | None:
    """获取最新财务快照 (EPS, ROE, 净利等)。mootdx 字段为拼音缩写。"""
    try:
        client = _get_client()
        fin = client.finance(symbol=symbol)
        if fin is None or fin.empty:
            return None
        row = fin.iloc[0]

        jinglirun = float(row.get("jinglirun", 0) or 0)        # 净利润
        jingzichan = float(row.get("jingzichan", 0) or 0)      # 净资产
        zongguben = float(row.get("zongguben", 0) or 0)        # 总股本
        zhuyingshouru = float(row.get("zhuyingshouru", 0) or 0)  # 主营收入

        roe = (jinglirun / jingzichan * 100) if jingzichan > 0 else 0
        eps = (jinglirun / zongguben) if zongguben > 0 else 0
        bvps = (jingzichan / zongguben) if zongguben > 0 else 0

        return {
            "eps": eps,
            "roe": roe,
            "net_profit": jinglirun,
            "revenue": zhuyingshouru,
            "bvps": bvps,
            "net_assets": jingzichan,
            "industry": str(row.get("industry", "") or ""),
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
#  沪深300 指数 (用于CAPM基准)
# ════════════════════════════════════════════════════════════

def fetch_csi300_returns(start_date: str = "2020-01-01") -> pd.Series | None:
    """获取沪深300日收益率序列（yfinance）。"""
    import yfinance as yf
    cache_key = "cn_csi300_returns"
    cached = cache_get(cache_key, ttl_hours=6)
    if cached is not None and not cached.empty:
        return cached.iloc[:, 0].astype(float)

    try:
        df = yf.download("000300.SS", start=start_date, auto_adjust=True, progress=False)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ret = close.pct_change().dropna()
        ret.name = "csi300"
        cache_put(cache_key, ret.to_frame())
        return ret.astype(float)
    except Exception:
        return None
