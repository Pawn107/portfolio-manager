"""信号系统 V3 — 基本面选股 + 技术面风控。

BUY = 估值合理 + 盈利质量过关（纯基本面判断）
SELL = 止损/移动止盈/趋势破坏（纯风控判断）
HOLD = 其余时间持有

不再用技术指标做买入择时。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from signals import indicators as ind

MIN_OBS = 60

# ── 基本面阈值 ──
PE_LOW = 0         # PE < 此值 = 亏损，无效
PE_IDEAL = (5, 50)   # 理想 PE 区间
PE_OK = (50, 100)    # 可接受区间
PE_HIGH = 100        # PE > 此值 = 过高

PB_IDEAL = (0.5, 5)  # 理想 PB 区间
PB_OK = (5, 10)       # 可接受

ROE_HIGH = 15        # ROE > 此值 = 优质
ROE_OK = 5           # ROE > 此值 = 及格

MCAP_LARGE = 1000    # 大盘股 (亿)
MCAP_MID = 100       # 中盘股 (亿)

FUNDAMENTAL_BUY = 60  # 基本面总分 >= 此值 → BUY

# ── 风控参数 ──
STOP_LOSS = -0.08        # 固定止损 -8%
TRAILING_STOP = -0.08    # 移动止盈 -8% from peak（A股波动大，5%太紧）
MAX_HOLD_DAYS = 60       # 最大持有 60 个交易日


# ═══════════════════════════════════════════════════════════
#  Layer 1: 基本面打分（决定是否买入）
# ═══════════════════════════════════════════════════════════

def fundamental_score(valuation: dict, finance: dict | None = None) -> dict:
    """对单只股票的基本面打分 (0-100)。

    Args:
        valuation: 腾讯财经数据 {pe_ttm, pb, mcap_yi, ...}
        finance: 财务快照 {eps, roe, net_profit, ...}

    Returns:
        {total_score, pe_score, pb_score, roe_score, mcap_score, details: [...]}
    """
    pe = valuation.get("pe_ttm", 0) or 0
    pb = valuation.get("pb", 0) or 0
    mcap = valuation.get("mcap_yi", 0) or 0

    roe = 0
    if finance:
        roe = finance.get("roe", 0) or 0

    # PE 得分 (0-35)
    if PE_IDEAL[0] < pe <= PE_IDEAL[1]:
        pe_score = 35
        pe_detail = f"PE {pe:.1f} → 合理区间"
    elif pe <= PE_LOW:
        pe_score = 0
        pe_detail = f"PE {pe:.1f} → 亏损"
    elif 0 < pe <= PE_IDEAL[0]:
        pe_score = 20
        pe_detail = f"PE {pe:.1f} → 偏低(可能有风险)"
    elif PE_IDEAL[1] < pe <= PE_OK[1]:
        pe_score = 20
        pe_detail = f"PE {pe:.1f} → 偏高但可接受"
    elif PE_OK[1] < pe <= PE_HIGH * 2:
        pe_score = 10
        pe_detail = f"PE {pe:.1f} → 偏高"
    else:
        pe_score = 0
        pe_detail = f"PE {pe:.1f} → 极高或无效"

    # PB 得分 (0-25)
    if PB_IDEAL[0] < pb <= PB_IDEAL[1]:
        pb_score = 25
        pb_detail = f"PB {pb:.2f} → 合理"
    elif PB_OK[0] < pb <= PB_OK[1]:
        pb_score = 15
        pb_detail = f"PB {pb:.2f} → 偏高"
    elif 0 < pb <= PB_IDEAL[0]:
        pb_score = 10
        pb_detail = f"PB {pb:.2f} → 破净(需关注)"
    else:
        pb_score = 5
        pb_detail = f"PB {pb:.2f} → 极高或无效"

    # ROE 得分 (0-25)
    if roe >= ROE_HIGH:
        roe_score = 25
        roe_detail = f"ROE {roe:.1f}% → 优质"
    elif roe >= ROE_OK:
        roe_score = 15
        roe_detail = f"ROE {roe:.1f}% → 及格"
    elif roe > 0:
        roe_score = 8
        roe_detail = f"ROE {roe:.1f}% → 偏低"
    else:
        roe_score = 0
        roe_detail = "ROE 缺失或为负"

    # 市值得分 (0-15)
    if mcap >= MCAP_LARGE:
        mcap_score = 15
        mcap_detail = f"市值 {mcap:.0f}亿 → 大盘"
    elif mcap >= MCAP_MID:
        mcap_score = 10
        mcap_detail = f"市值 {mcap:.0f}亿 → 中盘"
    elif mcap > 0:
        mcap_score = 5
        mcap_detail = f"市值 {mcap:.0f}亿 → 小盘"
    else:
        mcap_score = 0
        mcap_detail = "市值缺失"

    total = pe_score + pb_score + roe_score + mcap_score

    return {
        "total_score": total,
        "pe_score": pe_score,
        "pb_score": pb_score,
        "roe_score": roe_score,
        "mcap_score": mcap_score,
        "details": [pe_detail, pb_detail, roe_detail, mcap_detail],
    }


def fundamental_signal(valuation: dict, finance: dict | None = None) -> tuple[str, dict]:
    """基本面 → 买入判断。

    Returns:
        (signal, score_dict) — signal ∈ {"BUY", "HOLD", "AVOID"}
    """
    score = fundamental_score(valuation, finance)
    total = score["total_score"]

    pe = valuation.get("pe_ttm", 0) or 0

    if pe <= 0:
        return "AVOID", score
    if total >= FUNDAMENTAL_BUY:
        return "BUY", score
    if total >= 40:
        return "HOLD", score
    return "AVOID", score


# ═══════════════════════════════════════════════════════════
#  Layer 2: 技术面风控（决定是否卖出）
# ═══════════════════════════════════════════════════════════

def risk_check(df: pd.DataFrame, entry_price: float | None = None,
               entry_date=None) -> dict:
    """技术面风控检查（基于历史K线 + 持仓状态）。

    Args:
        df: 日K线 DataFrame
        entry_price: 入场价格 (None = 未持仓)
        entry_date: 入场日期

    Returns:
        {"signal": "SELL"|"OK", "reason": str, "risk_flags": [...]}
    """
    close = df["close"]
    high = df["high"]
    latest = close.iloc[-1]
    latest_idx = df.index[-1]
    risk_flags = []

    # ── 无持仓：只做波动率过大提醒 ──
    if entry_price is None:
        atr14 = ind.atr(df, 14)
        if len(atr14.dropna()) > 0:
            latest_atr = atr14.iloc[-1]
            if latest_atr / latest > 0.04:
                risk_flags.append(f"波动率偏高 (ATR/Price={latest_atr/latest*100:.1f}%)")
        return {"signal": "OK", "reason": "无持仓", "risk_flags": risk_flags}

    # ── 有持仓：风控检查 ──

    # 1. 固定止损
    loss_pct = (latest - entry_price) / entry_price
    if loss_pct <= STOP_LOSS:
        risk_flags.append(f"止损触发: {loss_pct*100:.1f}% (≤ -8%)")
        return {"signal": "SELL", "reason": f"止损 {loss_pct*100:.1f}%", "risk_flags": risk_flags}

    # 2. 移动止盈 (从持仓期间最高点回撤 > 5%)
    if entry_date is not None:
        try:
            mask = (df.index >= entry_date) & (df.index <= latest_idx)
            highest_since_entry = high[mask].max()
        except Exception:
            highest_since_entry = entry_price
    else:
        highest_since_entry = max(entry_price, latest)

    if highest_since_entry > entry_price:
        drawdown = (latest - highest_since_entry) / highest_since_entry
        if drawdown <= TRAILING_STOP:
            risk_flags.append(f"移动止盈触发: 从最高 {highest_since_entry:.2f} 回撤 {drawdown*100:.1f}%")
            return {"signal": "SELL", "reason": f"移动止盈 (回撤{drawdown*100:.1f}%)",
                    "risk_flags": risk_flags}

    # 3. 止盈: +25%
    if loss_pct >= 0.25:
        risk_flags.append(f"止盈触发: +{loss_pct*100:.1f}%")
        return {"signal": "SELL", "reason": f"止盈 +{loss_pct*100:.1f}%", "risk_flags": risk_flags}

    # 4. 趋势破坏: HMA 死叉
    hma20 = ind.hma(close, 20)
    hma50 = ind.hma(close, 50)
    if len(hma20.dropna()) >= 2 and len(hma50.dropna()) >= 2:
        death_cross = (hma20.iloc[-1] < hma50.iloc[-1]) and (hma20.iloc[-2] >= hma50.iloc[-2])
        if death_cross:
            risk_flags.append("HMA 死叉 (20/50) → 趋势转弱")
            return {"signal": "SELL", "reason": "HMA死叉 → 趋势破坏", "risk_flags": risk_flags}

    # 5. 最大持有期
    if entry_date is not None:
        try:
            days_held = (df.index.get_loc(latest_idx) - df.index.get_loc(entry_date))
            if days_held >= MAX_HOLD_DAYS:
                risk_flags.append(f"最大持有期 {MAX_HOLD_DAYS} 天")
                return {"signal": "SELL", "reason": f"持有超{MAX_HOLD_DAYS}天", "risk_flags": risk_flags}
        except Exception:
            pass

    return {"signal": "OK", "reason": "风控通过", "risk_flags": risk_flags}


# ═══════════════════════════════════════════════════════════
#  Layer 3: 市场环境（极端情绪 → 仓位建议）
# ═══════════════════════════════════════════════════════════

def market_env_signal(market_breadth: dict | None = None) -> dict:
    """市场环境 → 仓位建议。

    Returns:
        {"mode": "panic"|"euphoria"|"normal", "suggested_position": 0.3-1.0}
    """
    if market_breadth is None:
        return {"mode": "normal", "suggested_position": 1.0}

    up = market_breadth.get("up_count", 0)
    down = market_breadth.get("down_count", 0)
    total = up + down
    if total == 0:
        return {"mode": "normal", "suggested_position": 1.0}

    up_pct = up / total

    if up_pct < 0.30:
        return {"mode": "panic", "suggested_position": 0.3}
    elif up_pct > 0.80:
        return {"mode": "euphoria", "suggested_position": 0.5}
    else:
        return {"mode": "normal", "suggested_position": 1.0}


# ═══════════════════════════════════════════════════════════
#  综合信号输出
# ═══════════════════════════════════════════════════════════

def compute_signals(df: pd.DataFrame,
                    valuation: dict | None = None,
                    finance: dict | None = None,
                    market_breadth: dict | None = None,
                    entry_price: float | None = None,
                    entry_date=None) -> dict:
    """综合信号引擎。

    Args:
        df: 日K线 DataFrame
        valuation: 腾讯财经估值数据
        finance: 新浪财务快照
        market_breadth: 全市场涨跌家数
        entry_price: 持仓入场价 (None = 空仓)
        entry_date: 持仓入场日期

    Returns:
        {
            signal: "BUY"|"HOLD"|"SELL"|"AVOID",
            fundamental: {...},
            risk: {...},
            market: {...},
            suggested_position: float,
        }
    """
    # Layer 1: 基本面
    if valuation:
        f_signal, f_score = fundamental_signal(valuation, finance)
    else:
        f_signal, f_score = "HOLD", {"total_score": 0, "details": ["无估值数据"]}

    # Layer 2: 风控
    r_check = risk_check(df, entry_price, entry_date)

    # Layer 3: 环境
    m_env = market_env_signal(market_breadth)

    # ── 综合判断 ──
    if r_check["signal"] == "SELL":
        final_signal = "SELL"
        reason = r_check["reason"]
    elif f_signal == "BUY":
        final_signal = "BUY"
        reason = f"基本面达标 (得分{f_score['total_score']})"
    elif f_signal == "AVOID":
        final_signal = "AVOID"
        reason = f"基本面不达标 (得分{f_score['total_score']})"
    else:
        final_signal = "HOLD"
        reason = f"基本面一般 (得分{f_score['total_score']})"

    return {
        "signal": final_signal,
        "reason": reason,
        "fundamental": f_score,
        "risk": r_check,
        "market": m_env,
        "suggested_position": m_env["suggested_position"],
        "close": float(df["close"].iloc[-1]),
        "date": str(df.index[-1])[:10],
    }
