"""新三层信号引擎胜率测试 — vs 旧七因子系统。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from data.cn_fetcher import fetch_daily_kline
from signals.scoring import compute_signals
from signals.scoring_old import compute_scores as compute_scores_old  # 旧系统

TEST_CODES = [
    "600519", "000333", "300750", "600036", "000858",
    "601318", "002594", "600276", "600900", "002475",
    "601012", "300059", "600585", "000568", "002714",
]
FORWARD = [1, 3, 5, 10, 20]

# ── 加载数据 ──
print("加载数据...")
all_data = {}
for code in TEST_CODES:
    df = fetch_daily_kline(code)
    if df is not None and len(df) > 120:
        all_data[code] = df
        print(f"  {code}: {len(df)} 天")
print(f"\n有效: {len(all_data)} 只\n")

# ── 新系统测试 ──
print("=" * 90)
print("新三层信号系统")
print("=" * 90)

new_results = []

for code, df in all_data.items():
    result = compute_signals(df, market_breadth=None, turnover_series=None)

    for h in FORWARD:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        common = result.index.intersection(fwd_ret.dropna().index)

        buy_mask = result.loc[common, "signal"] == "BUY"
        sell_mask = result.loc[common, "signal"] == "SELL"
        hold_mask = result.loc[common, "signal"] == "HOLD"
        high_conf = result.loc[common, "confidence"] == "high"

        buy_fwd = fwd_ret.loc[common][buy_mask]
        sell_fwd = fwd_ret.loc[common][sell_mask]
        high_fwd = fwd_ret.loc[common][high_conf & buy_mask]

        new_results.append({
            "code": code, "horizon": h,
            "total": len(common),
            "buy_count": len(buy_fwd.dropna()),
            "buy_win": (buy_fwd > 0).sum(),
            "buy_avg_ret": buy_fwd.mean(),
            "sell_count": len(sell_fwd.dropna()),
            "sell_win": (sell_fwd < 0).sum(),
            "sell_avg_ret": sell_fwd.mean(),
            "hold_count": hold_mask.sum(),
            "high_conf_buy": len(high_fwd.dropna()),
            "high_conf_win": (high_fwd > 0).sum(),
        })

df_new = pd.DataFrame(new_results)

# 打印每只股票
for code in all_data:
    sub = df_new[df_new["code"] == code]
    print(f"\n--- {code} ---")
    for _, r in sub.iterrows():
        bw = r["buy_win"]/r["buy_count"]*100 if r["buy_count"] > 5 else 0
        sw = r["sell_win"]/r["sell_count"]*100 if r["sell_count"] > 5 else 0
        hw = r["high_conf_win"]/r["high_conf_buy"]*100 if r["high_conf_buy"] > 5 else 0
        print(f"  T+{r['horizon']:2d}: BUY {r['buy_count']:3d}次 胜率 {bw:5.1f}% 均值 {r['buy_avg_ret']*100:+6.2f}% | "
              f"SELL {r['sell_count']:3d}次 胜率 {sw:5.1f}% | "
              f"高置信BUY {r['high_conf_buy']:3d}次 胜率 {hw:5.1f}%")

# ── 汇总 ──
print("\n" + "=" * 90)
print("新系统汇总")
print("=" * 90)
for h in FORWARD:
    sub = df_new[df_new["horizon"] == h]
    tb = sub["buy_count"].sum()
    tw = sub["buy_win"].sum()
    ts = sub["sell_count"].sum()
    sw = sub["sell_win"].sum()
    hb = sub["high_conf_buy"].sum()
    hw = sub["high_conf_win"].sum()

    print(f"T+{h:2d}: BUY {tb}次 胜率 {tw/tb*100:.1f}% | "
          f"SELL {ts}次 胜率 {sw/ts*100:.1f}% | "
          f"高置信BUY {hb}次 胜率 {hw/hb*100:.1f}% | "
          f"BUY:SELL = {tb}:{ts} = {tb/ts:.1f}:1" if ts > 0 else f"BUY:SELL = {tb}:0")

# ── 旧系统对比 ──
print("\n" + "=" * 90)
print("旧七因子系统 (对比)")
print("=" * 90)

old_results = []
for code, df in all_data.items():
    scores = compute_scores_old(df)
    for h in FORWARD:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        common = scores.index.intersection(fwd_ret.dropna().index)
        buy_mask = scores.loc[common, "signal"] == "BUY"
        sell_mask = scores.loc[common, "signal"] == "SELL"
        buy_fwd = fwd_ret.loc[common][buy_mask]
        sell_fwd = fwd_ret.loc[common][sell_mask]
        old_results.append({
            "code": code, "horizon": h,
            "buy_count": len(buy_fwd.dropna()),
            "buy_win": (buy_fwd > 0).sum() if len(buy_fwd.dropna()) > 0 else 0,
            "sell_count": len(sell_fwd.dropna()),
            "sell_win": (sell_fwd < 0).sum() if len(sell_fwd.dropna()) > 0 else 0,
        })

df_old = pd.DataFrame(old_results)
for h in FORWARD:
    sub = df_old[df_old["horizon"] == h]
    tb = sub["buy_count"].sum()
    tw = sub["buy_win"].sum()
    ts = sub["sell_count"].sum()
    sw = sub["sell_win"].sum()
    print(f"T+{h:2d}: BUY {tb}次 胜率 {tw/tb*100:.1f}% | "
          f"SELL {ts}次 胜率 {sw/ts*100:.1f}% | BUY:SELL = {tb}:{ts}")

# ── 对比表格 ──
print("\n" + "=" * 90)
print("新旧对比")
print("=" * 90)
print(f"{'指标':<25} {'旧系统':>12} {'新系统':>12} {'变化':>12}")
print("-" * 61)
for h in [5, 10]:
    no = df_old[df_old["horizon"] == h]
    nn = df_new[df_new["horizon"] == h]

    old_wr = no["buy_win"].sum() / no["buy_count"].sum() * 100
    new_wr = nn["buy_win"].sum() / nn["buy_count"].sum() * 100

    print(f"{'T+'+str(h)+' BUY胜率':<25} {old_wr:>11.1f}% {new_wr:>11.1f}% {new_wr-old_wr:>+11.1f}%")

    old_sr = no["sell_win"].sum() / no["sell_count"].sum() * 100 if no["sell_count"].sum() > 0 else 0
    new_sr = nn["sell_win"].sum() / nn["sell_count"].sum() * 100 if nn["sell_count"].sum() > 0 else 0
    print(f"{'T+'+str(h)+' SELL胜率':<25} {old_sr:>11.1f}% {new_sr:>11.1f}% {new_sr-old_sr:>+11.1f}%")

    old_ratio = no["buy_count"].sum() / no["sell_count"].sum() if no["sell_count"].sum() > 0 else 0
    new_ratio = nn["buy_count"].sum() / nn["sell_count"].sum() if nn["sell_count"].sum() > 0 else 0
    print(f"{'T+'+str(h)+' BUY:SELL':<25} {old_ratio:>11.1f} {new_ratio:>11.1f}")

    # 信号分布
    print(f"{'T+'+str(h)+' 总BUY次数':<25} {no['buy_count'].sum():>11.0f} {nn['buy_count'].sum():>11.0f}")
    print(f"{'T+'+str(h)+' 总SELL次数':<25} {no['sell_count'].sum():>11.0f} {nn['sell_count'].sum():>11.0f}")
    print()

# ── 高置信度信号分析 ──
print("=" * 90)
print("高置信度 BUY 信号分析 (新系统)")
print("=" * 90)
for h in FORWARD:
    sub = df_new[df_new["horizon"] == h]
    hb = sub["high_conf_buy"].sum()
    hw = sub["high_conf_win"].sum()
    all_buy = sub["buy_count"].sum()
    ratio = hb / all_buy * 100 if all_buy > 0 else 0
    print(f"T+{h}: 高置信BUY {hb}次 / 全部BUY {all_buy}次 = {ratio:.1f}% | "
          f"高置信胜率 {hw/hb*100:.1f}%" if hb > 0 else f"T+{h}: 无高置信信号")
