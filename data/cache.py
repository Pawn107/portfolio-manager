"""CSV 文件缓存，TTL 24 小时。"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from config import CACHE_DIR


def _cache_path(key: str) -> Path:
    return Path(CACHE_DIR) / f"{key}.csv"


def get(key: str, ttl_hours: int = 24) -> pd.DataFrame | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    if datetime.now() - mtime > timedelta(hours=ttl_hours):
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def put(key: str, df: pd.DataFrame):
    p = _cache_path(key)
    df.to_csv(p)


def clear():
    for p in Path(CACHE_DIR).glob("*.csv"):
        p.unlink()
