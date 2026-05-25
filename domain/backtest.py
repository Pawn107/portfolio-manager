"""周频回测引擎 — 基于信号系统的买卖模拟。

频率: 周频 (周五收盘)
手续费: 买入 0.03%, 卖出 0.13% (含千一印花税)
滑点: 0.1%
"""
import numpy as np
import pandas as pd
from config import BACKTEST_CONFIG


class BacktestResult:
    """回测结果。"""

    def __init__(self):
        self.trades: list[dict] = []
        self.equity_curve: list[float] = []
        self.dates: list = []
        self.signals: list[str] = []

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t["pnl_pct"] is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t["pnl_pct"] > 0)
        return wins / len(closed)

    @property
    def avg_return(self) -> float:
        closed = [t for t in self.trades if t["pnl_pct"] is not None]
        if not closed:
            return 0.0
        return np.mean([t["pnl_pct"] for t in closed])

    @property
    def profit_factor(self) -> float:
        closed = [t for t in self.trades if t["pnl_pct"] is not None]
        wins = [t["pnl_pct"] for t in closed if t["pnl_pct"] > 0]
        losses = [abs(t["pnl_pct"]) for t in closed if t["pnl_pct"] < 0]
        if not losses or sum(losses) == 0:
            return float("inf") if wins else 0.0
        return sum(wins) / sum(losses)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return float(np.min(dd))

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        eq = np.array(self.equity_curve)
        returns = eq[1:] / eq[:-1] - 1
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(np.mean(returns) / returns.std() * np.sqrt(52))

    @property
    def annual_return(self) -> float:
        if len(self.equity_curve) < 2 or self.equity_curve[0] == 0:
            return 0.0
        total_ret = self.equity_curve[-1] / self.equity_curve[0] - 1
        years = len(self.equity_curve) / 52
        if years == 0:
            return 0.0
        return float((1 + total_ret) ** (1 / years) - 1)

    def summary(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "annual_return": self.annual_return,
        }


def run_backtest(df_weekly: pd.DataFrame, scores: pd.DataFrame,
                 initial_capital: float = 100000) -> BacktestResult:
    """运行周频回测。

    Args:
        df_weekly: 周K DataFrame (date index, 含 open/high/low/close/volume)
        scores: 日频信号 DataFrame (含 signal 列)
        initial_capital: 初始资金

    Returns:
        BacktestResult
    """
    commission_buy = BACKTEST_CONFIG["commission_buy"]
    commission_sell = BACKTEST_CONFIG["commission_sell"]
    slippage = BACKTEST_CONFIG["slippage"]
    stop_loss = BACKTEST_CONFIG["stop_loss"]
    take_profit = BACKTEST_CONFIG["take_profit"]
    max_hold = BACKTEST_CONFIG["max_hold_weeks"]

    result = BacktestResult()
    result.equity_curve = [initial_capital]
    result.signals = []

    # 将信号对齐到周频：取每周最后一个交易日的信号
    weekly_signals = _align_signals_to_weekly(df_weekly, scores)

    capital = initial_capital
    position = 0           # 持仓股数
    entry_price = 0.0
    highest_since_entry = 0.0
    weeks_held = 0

    for i, (date, row) in enumerate(df_weekly.iterrows()):
        week_close = float(row["close"])
        signal = weekly_signals.get(date, "HOLD")
        result.signals.append(signal)
        result.dates.append(date)

        # 无持仓 → 等待买入信号
        if position == 0:
            if signal == "BUY":
                buy_price = week_close * (1 + slippage)
                cost = buy_price * commission_buy
                position = int(capital / (buy_price * (1 + commission_buy)))
                if position > 0:
                    capital -= position * buy_price * (1 + commission_buy)
                    entry_price = buy_price
                    highest_since_entry = buy_price
                    weeks_held = 0
                    result.trades.append({
                        "entry_date": date,
                        "entry_price": buy_price,
                        "exit_date": None,
                        "exit_price": None,
                        "pnl_pct": None,
                        "exit_reason": None,
                    })
            result.equity_curve.append(capital + position * week_close)
            continue

        # 有持仓 → 更新 + 检查离场
        weeks_held += 1
        highest_since_entry = max(highest_since_entry, week_close)

        exit_now = False
        exit_reason = ""

        # 止损
        loss_pct = (week_close - entry_price) / entry_price
        if loss_pct <= stop_loss:
            exit_now = True
            exit_reason = f"止损 {loss_pct*100:.1f}%"
        # 止盈
        elif loss_pct >= take_profit:
            exit_now = True
            exit_reason = f"止盈 +{loss_pct*100:.1f}%"
        # 移动止盈
        elif highest_since_entry > entry_price:
            dd_from_peak = (week_close - highest_since_entry) / highest_since_entry
            if dd_from_peak <= -0.05:
                exit_now = True
                exit_reason = f"移动止盈 (回撤{dd_from_peak*100:.1f}%)"
        # 最大持有期
        elif weeks_held >= max_hold:
            exit_now = True
            exit_reason = f"最大持有期({weeks_held}周)"
        # 信号反转
        elif signal == "SELL":
            exit_now = True
            exit_reason = "信号 → SELL"

        if exit_now:
            sell_price = week_close * (1 - slippage)
            cost = sell_price * commission_sell
            proceeds = position * sell_price * (1 - commission_sell)
            pnl_pct = (sell_price * (1 - commission_sell)) / entry_price - 1
            capital += proceeds
            result.trades[-1].update({
                "exit_date": date,
                "exit_price": sell_price,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
            })
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            weeks_held = 0

        result.equity_curve.append(capital + position * week_close)

    return result


def _align_signals_to_weekly(df_weekly: pd.DataFrame,
                              scores: pd.DataFrame) -> dict:
    """将日频信号对齐到周频。

    对每周，取该周最后一个有信号的交易日的信号。
    """
    weekly_signals = {}
    for week_date in df_weekly.index:
        # 找到该周之前最近的日频信号
        mask = scores.index <= week_date
        if mask.any():
            week_signal = scores.loc[mask, "signal"].iloc[-1]
            weekly_signals[week_date] = week_signal
        else:
            weekly_signals[week_date] = "HOLD"
    return weekly_signals


def benchmark_return(df_weekly: pd.DataFrame) -> float:
    """计算 buy & hold 基准收益。"""
    if len(df_weekly) < 2:
        return 0.0
    return float(df_weekly["close"].iloc[-1] / df_weekly["close"].iloc[0] - 1)
