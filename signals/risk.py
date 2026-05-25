"""风控模块：入场过滤 + 持仓止损止盈。"""
import numpy as np
import pandas as pd
from signals import indicators as ind


def check_entry(df: pd.DataFrame, pe_ttm: float = None) -> tuple[bool, str]:
    """检查是否允许入场。

    Args:
        df: 个股日K DataFrame
        pe_ttm: 当前 PE_TTM (可选)

    Returns:
        (是否允许入场, 拒绝原因)
    """
    close = df["close"].values

    # 1. 波动率过高
    atr14 = ind.atr(df, 14)
    if len(atr14.dropna()) > 0 and atr14.iloc[-1] / close[-1] > 0.05:
        return False, "波动率过高 (ATR/Price > 5%)"

    # 2. 极度缩量
    vol = df["volume"].values
    if len(vol) >= 20:
        vol_ma20 = np.mean(vol[-20:])
        if vol[-1] < vol_ma20 * 0.5:
            return False, "极度缩量 (量比 < 0.5)"

    # 3. PE 无效
    if pe_ttm is not None and pe_ttm <= 0:
        return False, "PE_TTM 无效"

    return True, "OK"


def check_exit(entry_price: float, current_price: float,
               highest_since_entry: float,
               current_signal: str,
               weeks_held: int = 0) -> tuple[bool, str]:
    """检查是否需要强制离场。

    Args:
        entry_price: 入场价格
        current_price: 当前价格
        highest_since_entry: 持仓期间最高价
        current_signal: 当前信号 (BUY/SELL/HOLD)
        weeks_held: 已持有周数

    Returns:
        (是否离场, 原因)
    """
    # 1. 止损: 亏损 <= -8%
    loss_pct = (current_price - entry_price) / entry_price
    if loss_pct <= -0.08:
        return True, f"止损 ({loss_pct*100:.1f}%)"

    # 2. 移动止盈: 从最高点回撤 >= 5%
    if highest_since_entry > entry_price:
        drawdown = (current_price - highest_since_entry) / highest_since_entry
        if drawdown <= -0.05:
            return True, f"移动止盈 (回撤 {drawdown*100:.1f}%)"

    # 3. 信号反转 → SELL
    if current_signal == "SELL":
        return True, "信号反转 → SELL"

    # 4. 最大持有期 12 周
    if weeks_held >= 12:
        return True, f"最大持有期 ({weeks_held}周)"

    # 5. 止盈: 盈利 >= 25%
    if loss_pct >= 0.25:
        return True, f"止盈 (+{loss_pct*100:.1f}%)"

    return False, "OK"
