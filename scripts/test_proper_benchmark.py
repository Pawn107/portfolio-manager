"""正确评测：信号收益 vs 买入持有 vs 沪深300基准。

分两组：科技成长股 vs 蓝筹价值股
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from data.cn_fetcher import fetch_daily_kline, fetch_csi300_returns
from signals.scoring import compute_signals
from signals.scoring_old import compute_scores as compute_scores_old

# ── 分组 ──
TECH = ["300750", "002475", "601012", "300059", "002594", "002714", "600276"]
BLUE = ["600519", "000333", "600036", "600585", "000858", "601318", "600900"]

def signal_returns(df, result, hold_days=20):
    """计算按BUY信号买入并持有hold_days的收益。
    每次BUY信号买入，持有到卖出或expire。
    同时计算买入持有基准和超额收益。
    """
    close = df["close"]
    ret = close.pct_change()
    csi300 = fetch_csi300_returns()

    # 策略收益：只在BUY日入场，持有hold_days
    buy_dates = result[result["signal"] == "BUY"].index
    strategy_rets = []
    bh_rets = []
    excess_rets = []
    mkt_rets = []

    for d in buy_dates:
        try:
            idx = close.index.get_loc(d)
            end_idx = min(idx + hold_days, len(close) - 1)
            if end_idx <= idx + 1:
                continue

            entry = close.iloc[idx]
            exit_p = close.iloc[end_idx]

            # 策略持有期收益
            r = (exit_p - entry) / entry
            strategy_rets.append(r)

            # 买入持有同期收益（从该点开始一直持有）
            bh_r = ret.iloc[idx+1:end_idx+1].sum()  # 近似
            bh_rets.append(bh_r)

            # 超额收益
            excess_rets.append(r - bh_r)

            # 同期沪深300收益
            if csi300 is not None:
                mkt_r = csi300.loc[d:close.index[end_idx]].sum() if d in csi300.index else 0
                mkt_rets.append(mkt_r)
        except (IndexError, KeyError):
            continue

    if not strategy_rets:
        return None

    return {
        "count": len(strategy_rets),
        "win_rate": sum(1 for r in strategy_rets if r > 0) / len(strategy_rets) * 100,
        "avg_ret": np.mean(strategy_rets) * 100,
        "total_ret": (np.prod([1+r for r in strategy_rets]) - 1) * 100,
        "avg_bh": np.mean(bh_rets) * 100 if bh_rets else 0,
        "avg_excess": np.mean(excess_rets) * 100,
        "avg_mkt": np.mean(mkt_rets) * 100 if mkt_rets else 0,
        "excess_vs_mkt": np.mean([e - m for e, m in zip(excess_rets, mkt_rets)]) * 100 if mkt_rets else 0,
    }


print("=" * 90)
print("正确评测：信号策略 vs 买入持有 vs 沪深300")
print("=" * 90)

for hold_days in [5, 10, 20]:
    print(f"\n{'='*90}")
    print(f"持有期 = {hold_days} 个交易日")
    print(f"{'='*90}")

    for group_name, codes in [("科技成长", TECH), ("蓝筹价值", BLUE)]:
        print(f"\n--- {group_name} ---")
        print(f"{'股票':<8} {'系统':<6} {'信号数':>6} {'胜率':>7} {'均收益':>8} {'策略累计':>9} {'持仓均收':>9} {'超额vs持仓':>10} {'超额vs300':>10}")

        for code in codes:
            df = fetch_daily_kline(code)
            if df is None or len(df) < 120:
                continue

            for sys_name, compute_fn in [("新三层", compute_signals), ("旧七因子", compute_scores_old)]:
                result = compute_fn(df)
                r = signal_returns(df, result, hold_days)
                if r is None or r["count"] < 3:
                    continue

                print(f"  {code:<6} {sys_name:<6} {r['count']:>5}  {r['win_rate']:>5.1f}% {r['avg_ret']:>7.2f}% {r['total_ret']:>8.2f}% {r['avg_bh']:>8.2f}% {r['avg_excess']:>9.2f}% {r['excess_vs_mkt']:>9.2f}%")

    # 汇总
    print(f"\n{'='*40}")
    print(f"持有期 {hold_days}天 — 两组新系统对比")
    print(f"{'':<15} {'科技成长':>10} {'蓝筹价值':>10}")
    for metric_name, key in [("胜率%", "win_rate"), ("均收益%", "avg_ret"), ("超额vs持仓%", "avg_excess"), ("超额vs300%", "excess_vs_mkt")]:
        tech_vals = []
        blue_vals = []
        for code in TECH:
            df = fetch_daily_kline(code)
            if df is None: continue
            r = signal_returns(df, compute_signals(df), hold_days)
            if r and r["count"] >= 3:
                tech_vals.append(r[key])
        for code in BLUE:
            df = fetch_daily_kline(code)
            if df is None: continue
            r = signal_returns(df, compute_signals(df), hold_days)
            if r and r["count"] >= 3:
                blue_vals.append(r[key])
        if tech_vals and blue_vals:
            print(f"  {metric_name:<13} {np.mean(tech_vals):>9.2f}  {np.mean(blue_vals):>9.2f}")
