"""技术指标计算模块 — HMA, RSI, MACD, ATR, 支撑阻力, 量价情绪。"""
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
#  基础工具
# ════════════════════════════════════════════════════════════

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder平滑 (RSI原版)。"""
    smoothed = series.copy().astype(float)
    smoothed.iloc[:period] = np.nan
    if len(series) <= period:
        return smoothed
    smoothed.iloc[period] = series.iloc[1:period + 1].mean()
    for i in range(period + 1, len(series)):
        smoothed.iloc[i] = (smoothed.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
    return smoothed


def _wma(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均。"""
    if period <= 0:
        return series.copy()
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# ════════════════════════════════════════════════════════════
#  HMA — Hull Moving Average
# ════════════════════════════════════════════════════════════

def hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average。"""
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma_half = _wma(series, half)
    wma_full = _wma(series, period)
    raw = 2 * wma_half - wma_full
    return _wma(raw, sqrt)


# ════════════════════════════════════════════════════════════
#  RSI
# ════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ════════════════════════════════════════════════════════════
#  MACD
# ════════════════════════════════════════════════════════════

def macd(df: pd.DataFrame, fast: int = 8, slow: int = 17,
         signal: int = 5) -> pd.DataFrame:
    """加速版MACD。"""
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    histogram = 2 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "histogram": histogram}, index=df.index)


def macd_cross(df: pd.DataFrame, fast: int = 8, slow: int = 17,
               signal: int = 5) -> pd.Series:
    """MACD金叉/死叉: 1=金叉, -1=死叉, 0=无交叉。"""
    m = macd(df, fast, slow, signal)
    cross = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        if m["dif"].iloc[i] > m["dea"].iloc[i] and m["dif"].iloc[i - 1] <= m["dea"].iloc[i - 1]:
            cross[i] = 1
        elif m["dif"].iloc[i] < m["dea"].iloc[i] and m["dif"].iloc[i - 1] >= m["dea"].iloc[i - 1]:
            cross[i] = -1
    return pd.Series(cross, index=df.index)


# ════════════════════════════════════════════════════════════
#  ATR
# ════════════════════════════════════════════════════════════

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range。"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ════════════════════════════════════════════════════════════
#  量价关系
# ════════════════════════════════════════════════════════════

def volume_price_signal(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """量价关系: 1=放量上涨, -1=放量下跌, 0.5=缩量下跌, -0.5=缩量上涨。"""
    vol_series = df["volume"]
    if isinstance(vol_series, pd.DataFrame):
        vol_series = vol_series.iloc[:, 0]
    close_series = df["close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    avg_vol = vol_series.rolling(window=period).mean()
    vr = (vol_series / avg_vol.replace(0, np.nan)).to_numpy(dtype=float)
    ret = close_series.pct_change().to_numpy(dtype=float)
    result = np.zeros(len(df))

    for i in range(1, len(result)):
        v, r = vr[i], ret[i]
        if np.isnan(v) or np.isnan(r):
            continue
        if v > 1.5 and r > 0:
            result[i] = 1
        elif v > 1.5 and r < 0:
            result[i] = -1
        elif 0 < v < 0.6 and r < 0:
            result[i] = 0.5
        elif 0 < v < 0.6 and r > 0:
            result[i] = -0.5

    return pd.Series(result, index=df.index)


# ════════════════════════════════════════════════════════════
#  支撑阻力
# ════════════════════════════════════════════════════════════

def support_resistance(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """基于滚动窗口识别支撑/阻力位。"""
    close = df["close"].values
    support = np.full(len(df), np.nan)
    resistance = np.full(len(df), np.nan)
    sup_dist = np.full(len(df), np.nan)
    res_dist = np.full(len(df), np.nan)

    for i in range(window, len(df)):
        w_high = df["high"].iloc[i - window:i].max()
        w_low = df["low"].iloc[i - window:i].min()
        resistance[i] = w_high
        support[i] = w_low
        sup_dist[i] = (close[i] - w_low) / w_low * 100
        res_dist[i] = (w_high - close[i]) / w_high * 100

    return pd.DataFrame({
        "support": support, "resistance": resistance,
        "support_dist_pct": sup_dist, "resistance_dist_pct": res_dist,
    }, index=df.index)


# ════════════════════════════════════════════════════════════
#  振幅 & 波动率情绪
# ════════════════════════════════════════════════════════════

def amplitude_signal(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """振幅情绪: 振幅飙升→-2, 回落→+2。"""
    amp = (df["high"] - df["low"]) / df["close"] * 100
    amp_high = amp.rolling(window, min_periods=window // 2).quantile(0.80)

    spike = (amp > amp_high).fillna(False)
    retreat = (~spike) & spike.shift(1).fillna(False)
    spike_new = spike & (~spike.shift(1).fillna(False))

    result = np.zeros(len(df))
    result[retreat.values] = 2.0
    result[spike_new.values] = -2.0
    result[spike.values & ~spike_new.values] = -1.0

    return pd.Series(result, index=df.index)


def volume_emotion_signal(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """量能情绪: 放量下跌→恐慌买入+2, 放量上涨→狂热卖出-2。"""
    vol = df["volume"]
    close = df["close"]
    ret = close.pct_change().fillna(0)
    vol_ma = vol.rolling(window, min_periods=window // 2).mean()
    vol_ratio = vol / vol_ma.replace(0, np.nan)

    result = np.zeros(len(df))
    for i in range(window, len(df)):
        vr = vol_ratio.iloc[i]
        r = ret.iloc[i]
        if pd.isna(vr) or pd.isna(r):
            continue
        if vr > 2.0 and r < 0:
            result[i] = 2.0
        elif vr > 2.0 and r > 0:
            result[i] = -2.0
        elif vr < 0.5:
            result[i] = 1.0
        elif vr > 2.0:
            result[i] = -1.0

    return pd.Series(result, index=df.index)


def volatility_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """波动率情绪: 波动率飙升→恐慌-2, 回落→+1。"""
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252) * 100

    lookback = max(window * 5, 60)
    rv_high = rv.rolling(lookback).quantile(0.80)
    spike = (rv > rv_high).fillna(False)
    retreat = (~spike) & spike.shift(1).fillna(False)
    spike_new = spike & (~spike.shift(1).fillna(False))

    result = np.zeros(len(df))
    result[retreat.values] = 1.0
    result[spike_new.values] = -2.0
    result[spike.values & ~spike_new.values] = -1.0

    return pd.Series(result, index=df.index)
