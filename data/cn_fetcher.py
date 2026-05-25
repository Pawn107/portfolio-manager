"""A股数据下载：东财 (K线) + 腾讯财经 (PE/PB/市值) + 新浪 (财务) + yfinance (沪深300)。

全部 HTTP 协议，Streamlit Cloud 可用。
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import json
import numpy as np
import pandas as pd
from data.cache import get as cache_get, put as cache_put

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def _http_get_json(url: str, referer: str = "") -> dict | None:
    """通用 HTTP GET → JSON。"""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    if referer:
        req.add_header("Referer", referer)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except Exception:
        return None


def _em_market(symbol: str) -> str:
    """6位代码 → 东财市场代码 (1=沪, 0=深)。"""
    return "1" if symbol.startswith(("6", "9")) else "0"


# ════════════════════════════════════════════════════════════
#  东财 K线 (HTTP, 替代 mootdx TCP)
# ════════════════════════════════════════════════════════════

def _eastmoney_kline(symbol: str, klt: str = "101", count: int = 500) -> pd.DataFrame | None:
    """东财 K 线 HTTP API。

    Args:
        symbol: 6位代码
        klt: 101=日线, 102=周线, 103=月线
        count: 获取多少根K线
    """
    secid = f"{_em_market(symbol)}.{symbol}"
    params = {
        "secid": secid, "klt": klt, "fqt": "1",
        "end": "20500101", "lmt": str(count),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)

    d = _http_get_json(url, referer="https://quote.eastmoney.com/")
    if d is None:
        return None

    klines = d.get("data", {}).get("klines", [])
    if not klines:
        return None

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_daily_kline(symbol: str) -> pd.DataFrame | None:
    """获取日K线（缓存1小时）。"""
    cache_key = f"cn_daily_{symbol}"
    cached = cache_get(cache_key, ttl_hours=1)
    if cached is not None and not cached.empty:
        return cached

    df = _eastmoney_kline(symbol, klt="101", count=500)
    if df is not None:
        cache_put(cache_key, df)
    return df


def fetch_weekly_kline(symbol: str) -> pd.DataFrame | None:
    """获取周K线（缓存1小时）。"""
    cache_key = f"cn_weekly_{symbol}"
    cached = cache_get(cache_key, ttl_hours=1)
    if cached is not None and not cached.empty:
        return cached

    df = _eastmoney_kline(symbol, klt="102", count=200)
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
#  新浪财报 — EPS/ROE/净利 (替代 mootdx finance)
# ════════════════════════════════════════════════════════════

_SINA_FINANCE_URL = (
    "https://quotes.sina.cn/cn/api/openapi.php/"
    "CompanyFinanceService.getFinanceReport2022"
)


def fetch_finance(symbol: str) -> dict | None:
    """获取最新财务快照 (EPS, ROE, 净利等)。使用新浪财报 HTTP API。"""
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    paper_code = f"{prefix}{symbol}"

    try:
        # 利润表 → 净利润, 营收
        lrb_params = urllib.parse.urlencode({
            "paperCode": paper_code, "source": "lrb",
            "type": "0", "page": "1", "num": "2",
        })
        lrb_d = _http_get_json(f"{_SINA_FINANCE_URL}?{lrb_params}")
        if lrb_d is None:
            return None
        lrb_list = lrb_d.get("result", {}).get("data", {}).get("lrb", [])
        if not lrb_list:
            return None

        latest_lrb = lrb_list[0]
        jinglirun = float(latest_lrb.get("净利润", 0) or 0)
        zhuyingshouru = float(latest_lrb.get("营业总收入", 0) or 0)

        # 资产负债表 → 净资产, 总股本
        fzb_params = urllib.parse.urlencode({
            "paperCode": paper_code, "source": "fzb",
            "type": "0", "page": "1", "num": "2",
        })
        fzb_d = _http_get_json(f"{_SINA_FINANCE_URL}?{fzb_params}")
        if fzb_d is None:
            return None
        fzb_list = fzb_d.get("result", {}).get("data", {}).get("fzb", [])
        if not fzb_list:
            return None

        latest_fzb = fzb_list[0]
        jingzichan = float(latest_fzb.get("归属于母公司股东权益合计", 0) or 0)
        zongguben = float(latest_fzb.get("实收资本（或股本）", 0) or 0)

        if jingzichan <= 0 or zongguben <= 0:
            return None

        roe = (jinglirun / jingzichan * 100)
        eps = (jinglirun / zongguben)
        bvps = (jingzichan / zongguben)

        return {
            "eps": eps,
            "roe": roe,
            "net_profit": jinglirun,
            "revenue": zhuyingshouru,
            "bvps": bvps,
            "net_assets": jingzichan,
            "industry": "",
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
