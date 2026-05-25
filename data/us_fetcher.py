"""美股数据下载：yfinance。"""
import time
import pandas as pd
import yfinance as yf
from data.cache import get as cache_get, put as cache_put

MAX_RETRIES = 3
RETRY_DELAY = 2.0


def download_with_retry(ticker: str, start: str, end: str,
                         max_retries: int = MAX_RETRIES,
                         retry_delay: float = RETRY_DELAY) -> pd.DataFrame | None:
    """带重试的 yfinance 下载。"""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                return df
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None


def _extract_close(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """从 yfinance 返回的 DataFrame 提取收盘价 Series。"""
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ("Close", ticker)
        if close_col in df.columns:
            return df[close_col]
        return None
    if "Close" in df.columns:
        return df["Close"]
    return None


def fetch_prices(tickers: list[str], start: str, end: str,
                 labels: list[str] | None = None,
                 verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """下载多只美股价格，返回 (prices_df, returns_df)。"""
    if labels is None:
        labels = list(tickers)

    all_prices = {}
    for t, label in zip(tickers, labels):
        cache_key = f"us_price_{label}"
        cached = cache_get(cache_key)
        if cached is not None and not cached.empty:
            s_cached = cached.iloc[:, 0].astype(float) if cached.shape[1] == 1 else cached.astype(float)
            all_prices[label] = s_cached
            if verbose:
                print(f"    ✓ {label} (缓存)")
            continue

        df = download_with_retry(t, start, end)
        if df is not None:
            s = _extract_close(df, t)
            if s is not None and len(s) > 1:
                s.name = "close"
                cache_put(cache_key, s.to_frame())
                all_prices[label] = s
                if verbose:
                    print(f"    ✓ {label}")
            else:
                if verbose:
                    print(f"    ✗ {label}: 无 Close 数据")
        else:
            if verbose:
                print(f"    ✗ {label}: 下载失败")

    prices = pd.DataFrame(all_prices).dropna(axis=1, how="all")
    returns = prices.pct_change().clip(-0.5, 0.5)
    return prices, returns
