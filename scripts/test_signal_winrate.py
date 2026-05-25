"""测试当前信号系统的胜率 — BUY/SELL 信号与未来 N 日收益的一致性。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from data.cn_fetcher import fetch_daily_kline
from signals.scoring import compute_scores

# 测试股票池
TEST_CODES = [
    "600519",  # 贵州茅台
    "000333",  # 美的集团
    "300750",  # 宁德时代
    "600036",  # 招商银行
    "000858",  # 五粮液
    "601318",  # 中国平安
    "002594",  # 比亚迪
    "600276",  # 恒瑞医药
]

FORWARD_DAYS = [1, 3, 5, 10, 20]

print("=" * 80)
print("信号胜率测试 — 当前逻辑")
print("=" * 80)
print(f"因子: HMA趋势(25)+RSI(15)+MACD(20)+量价(15)+波动率(10)+振幅(5)+量能情绪(10)")
print(f"阈值: BUY ≥ 60 | SELL < 30 | HOLD 其余")
print(f"测试股票: {len(TEST_CODES)} 只")
print(f"前向窗口: {FORWARD_DAYS} 天")
print()

all_results = []

for code in TEST_CODES:
    print(f"--- {code} ---")
    df = fetch_daily_kline(code)
    if df is None or len(df) < 120:
        print(f"  数据不足，跳过")
        continue

    scores = compute_scores(df)

    for horizon in FORWARD_DAYS:
        # 前向收益
        fwd_ret = df["close"].pct_change(horizon).shift(-horizon)

        # BUY 信号的胜率（买入后涨了）
        buy_mask = scores["signal"] == "BUY"
        buy_fwd = fwd_ret[buy_mask].dropna()
        buy_win = (buy_fwd > 0).sum()
        buy_total = len(buy_fwd)

        # SELL 信号的胜率（卖出后跌了）
        sell_mask = scores["signal"] == "SELL"
        sell_fwd = fwd_ret[sell_mask].dropna()
        sell_win = (sell_fwd < 0).sum()
        sell_total = len(sell_fwd)

        # HOLD 信号
        hold_mask = scores["signal"] == "HOLD"
        hold_fwd = fwd_ret[hold_mask].dropna()

        all_results.append({
            "code": code,
            "horizon": horizon,
            "buy_count": buy_total,
            "buy_win": buy_win,
            "buy_winrate": round(buy_win / buy_total * 100, 1) if buy_total > 0 else 0,
            "buy_avg_ret": round(buy_fwd.mean() * 100, 2) if buy_total > 0 else 0,
            "sell_count": sell_total,
            "sell_win": sell_win,
            "sell_winrate": round(sell_win / sell_total * 100, 1) if sell_total > 0 else 0,
            "sell_avg_ret": round(sell_fwd.mean() * 100, 2) if sell_total > 0 else 0,
            "hold_count": len(hold_fwd.dropna()),
            "total_days": len(fwd_ret.dropna()),
        })

    # 打印该股票的汇总
    for horizon in FORWARD_DAYS:
        r = [x for x in all_results if x["code"] == code and x["horizon"] == horizon][0]
        print(f"  T+{horizon:2d}: BUY {r['buy_count']:3d}次 胜率 {r['buy_winrate']:5.1f}% 均值 {r['buy_avg_ret']:+6.2f}% | "
              f"SELL {r['sell_count']:3d}次 胜率 {r['sell_winrate']:5.1f}% 均值 {r['sell_avg_ret']:+6.2f}%")

print()
print("=" * 80)
print("汇总统计")
print("=" * 80)

df_all = pd.DataFrame(all_results)

for horizon in FORWARD_DAYS:
    sub = df_all[df_all["horizon"] == horizon]
    total_buy = sub["buy_count"].sum()
    total_buy_win = sub["buy_win"].sum()
    total_sell = sub["sell_count"].sum()
    total_sell_win = sub["sell_win"].sum()

    buy_wr = total_buy_win / total_buy * 100 if total_buy > 0 else 0
    sell_wr = total_sell_win / total_sell * 100 if total_sell > 0 else 0

    print(f"\nT+{horizon}:")
    print(f"  BUY  总计 {total_buy} 次 → 上涨 {total_buy_win} 次 → 胜率 {buy_wr:.1f}%")
    print(f"  SELL 总计 {total_sell} 次 → 下跌 {total_sell_win} 次 → 胜率 {sell_wr:.1f}%")

    # 每只股票分别显示
    for _, row in sub.iterrows():
        print(f"  {row['code']}: BUY胜率 {row['buy_winrate']:5.1f}% ({row['buy_count']}次) | "
              f"SELL胜率 {row['sell_winrate']:5.1f}% ({row['sell_count']}次)")

# 随机基准：如果随��买，胜率是多少
print()
print("=" * 80)
print("随机基准 (买入后 N 日涨的概率)")
for code in TEST_CODES:
    df = fetch_daily_kline(code)
    if df is None:
        continue
    for horizon in [1, 5, 10, 20]:
        fwd = df["close"].pct_change(horizon).shift(-horizon).dropna()
        up_pct = (fwd > 0).sum() / len(fwd) * 100
        print(f"  {code} T+{horizon}: 随机上涨概率 {up_pct:.1f}%")
    break  # 只打一只就够了
