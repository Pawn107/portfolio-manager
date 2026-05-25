"""Kenneth French Data Library: Fama-French 因子下载与解析。"""
import io
import zipfile
import pandas as pd
import requests
from data.cache import get as cache_get, put as cache_put

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
CACHE_KEY = "ff3_daily"


def fetch_ff3_daily(use_cache: bool = True) -> pd.DataFrame:
    """下载 FF3 日频因子，返回 DataFrame (date index, 列为小数)。"""
    if use_cache:
        cached = cache_get(CACHE_KEY)
        if cached is not None:
            for col in ["Mkt-RF", "SMB", "HML", "RF"]:
                if col in cached.columns:
                    cached[col] = cached[col].astype(float)
            return cached

    resp = requests.get(FF3_URL, timeout=30)
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [n for n in zf.namelist() if n.endswith((".CSV", ".csv"))][0]
    raw = zf.read(csv_name).decode("utf-8")

    lines = raw.splitlines()
    data_start = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "HML" in line:
            data_start = i
            break

    if data_start is None:
        raise ValueError("无法解析 FF3 CSV 格式")

    col_names = lines[data_start].strip().split(",")
    records = []
    for line in lines[data_start + 1:]:
        line = line.strip()
        if not line or "Copyright" in line or "Annual Factors" in line:
            break
        parts = line.split(",")
        if len(parts) >= 5:
            try:
                records.append([float(p) for p in parts])
            except ValueError:
                continue

    df = pd.DataFrame(records, columns=col_names)
    df = df.rename(columns={col_names[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("date")
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        if col in df.columns:
            df[col] = df[col] / 100.0

    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.dropna()
    if use_cache:
        cache_put(CACHE_KEY, df)
    return df


def get_rf_daily(ff3: pd.DataFrame) -> pd.Series:
    """从 FF3 数据提取日度无风险利率。"""
    return ff3["RF"].astype(float)


def get_mkt_excess(ff3: pd.DataFrame) -> pd.Series:
    """从 FF3 数据提取市场超额收益 (Mkt-RF)。"""
    return ff3["Mkt-RF"].astype(float)
