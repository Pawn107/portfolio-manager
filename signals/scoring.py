"""多因子打分引擎 → buy/sell/hold 判定。

基于日K和周K数据，计算技术面 + 情绪面因子，输出 0-100 综合得分。
"""
import numpy as np
import pandas as pd
from signals import indicators as ind

# ── 因子权重 (总和归一化为 100) ──
FACTOR_WEIGHTS = {
    "trend": 25,         # HMA 趋势方向
    "rsi": 15,           # RSI 超买超卖位置
    "macd": 20,          # MACD 交叉信号
    "volume": 15,        # 量价关系
    "volatility": 10,    # 波动率情绪
    "amplitude": 5,      # 振幅情绪
    "volume_emotion": 10, # 量能情绪
}

# ── 信号阈值 ──
BUY_THRESHOLD = 60
SELL_THRESHOLD = 30


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """计算综合得分 (0-100)。

    Args:
        df: OHLCV DataFrame (日K), 需要 date, open, high, low, close, volume 列

    Returns:
        DataFrame 含各因子得分、综合得分、信号
    """
    close = df["close"]
    scores = pd.DataFrame(index=df.index)
    scores["close"] = close

    # 1. HMA 趋势 (0-25)
    hma20 = ind.hma(close, 20)
    hma50 = ind.hma(close, 50)
    trend = np.zeros(len(df))
    # 价>HMA20 且 HMA20>HMA50 → 强势多头 25
    mask_strong = (close > hma20) & (hma20 > hma50)
    trend[mask_strong.values] = 25
    # 价在HMA20和HMA50之间 → 中性偏多 15
    mask_neutral = (
        ((close > hma20) & (hma20 <= hma50)) |
        ((close <= hma20) & (close > hma50))
    )
    trend[mask_neutral.values] = 15
    # 价<HMA20 但 HMA20>HMA50 → 回调中 10
    mask_pullback = (close < hma20) & (hma20 > hma50)
    trend[mask_pullback.values] = 10
    # 价<HMA20 且 HMA20<HMA50 → 空头 0
    scores["trend"] = trend

    # 2. RSI 位置 (0-15)
    rsi7 = ind.rsi(close, 7)
    rsi_score = np.zeros(len(df))
    # RSI 30-70 中性区 → 10
    rsi_score[(rsi7 >= 30) & (rsi7 <= 70)] = 10
    # RSI 25-30 偏超卖 → 12
    rsi_score[(rsi7 >= 25) & (rsi7 < 30)] = 12
    # RSI < 25 超卖 → 15
    rsi_score[rsi7 < 25] = 15
    # RSI 70-75 偏超买 → 8
    rsi_score[(rsi7 > 70) & (rsi7 <= 75)] = 8
    # RSI > 75 超买 → 3
    rsi_score[rsi7 > 75] = 3
    scores["rsi"] = rsi_score

    # 3. MACD 交叉 (0-20)
    macd_cross = ind.macd_cross(df, fast=8, slow=17, signal=5)
    m = ind.macd(df, fast=8, slow=17, signal=5)
    macd_score = np.zeros(len(df))
    macd_score[macd_cross == 1] = 20  # 金叉
    macd_score[macd_cross == -1] = 0  # 死叉
    # 未交叉但 dif > dea (金叉后延续) → 15
    cont_bull = (macd_cross == 0) & (m["dif"] > m["dea"]) & (m["histogram"] > 0)
    macd_score[cont_bull.values] = 15
    # 未交叉但 dif < dea (死叉后延续) → 5
    cont_bear = (macd_cross == 0) & (m["dif"] < m["dea"]) & (m["histogram"] < 0)
    macd_score[cont_bear.values] = 5
    # 其余 → 10
    macd_score[(macd_cross == 0) & (macd_score == 0)] = 10
    scores["macd"] = macd_score

    # 4. 量价关系 (0-15)
    vp = ind.volume_price_signal(df, period=20)
    vp_score = np.zeros(len(df))
    vp_score[vp == 1] = 15     # 放量上涨
    vp_score[vp == 0.5] = 12   # 缩量下跌
    vp_score[vp == 0] = 8      # 正常
    vp_score[vp == -0.5] = 5   # 缩量上涨
    vp_score[vp == -1] = 0     # 放量下跌
    scores["volume"] = vp_score

    # 5. 波动率情绪 (0-10)
    vol_sig = ind.volatility_signal(df)
    vol_score = np.zeros(len(df))
    vol_score[vol_sig == 1.0] = 10
    vol_score[vol_sig == -2.0] = 0
    vol_score[vol_sig == -1.0] = 4
    vol_score[vol_sig == 0] = 7
    scores["volatility"] = vol_score

    # 6. 振幅情绪 (0-5)
    amp_sig = ind.amplitude_signal(df)
    amp_score = np.zeros(len(df))
    amp_score[amp_sig == 2.0] = 5
    amp_score[amp_sig == -2.0] = 0
    amp_score[amp_sig == -1.0] = 2
    amp_score[amp_sig == 0] = 3
    scores["amplitude"] = amp_score

    # 7. 量能情绪 (0-10)
    ve_sig = ind.volume_emotion_signal(df)
    ve_score = np.zeros(len(df))
    ve_score[ve_sig == 2.0] = 10   # 放量下跌→恐慌买
    ve_score[ve_sig == -2.0] = 0   # 放量上涨→狂热卖
    ve_score[ve_sig == 1.0] = 8
    ve_score[ve_sig == -1.0] = 4
    ve_score[ve_sig == 0] = 5
    scores["volume_emotion"] = ve_score

    # ── 综合得分 (各因子得分直接相加，满分100) ──
    scores["total_score"] = 0.0
    for factor in FACTOR_WEIGHTS:
        if factor in scores.columns:
            scores["total_score"] += scores[factor].fillna(0)

    # ── 信号判定 ──
    scores["signal"] = "HOLD"
    scores.loc[scores["total_score"] >= BUY_THRESHOLD, "signal"] = "BUY"
    scores.loc[scores["total_score"] < SELL_THRESHOLD, "signal"] = "SELL"

    return scores.round(2)


def latest_signal(scores: pd.DataFrame) -> dict:
    """最新交易日的信号摘要。"""
    last = scores.iloc[-1]
    detail = {}
    for f in FACTOR_WEIGHTS:
        if f in scores.columns:
            detail[f] = round(float(last[f]), 1)
    return {
        "date": str(last.name)[:10] if hasattr(last, 'name') else "",
        "close": float(last["close"]),
        "total_score": round(float(last["total_score"]), 1),
        "signal": str(last["signal"]),
        "detail": detail,
    }


def signal_summary(scores: pd.DataFrame) -> dict:
    """信号统计摘要。"""
    signals = scores["signal"]
    total = len(signals)
    return {
        "buy_count": int((signals == "BUY").sum()),
        "hold_count": int((signals == "HOLD").sum()),
        "sell_count": int((signals == "SELL").sum()),
        "buy_pct": round((signals == "BUY").sum() / total * 100, 1),
        "sell_pct": round((signals == "SELL").sum() / total * 100, 1),
        "latest": str(signals.iloc[-1]),
    }
