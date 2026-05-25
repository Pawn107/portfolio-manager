"""统计验证：个股信号准确率与市场环境的相关性。

检验两个假设：
H1: 个股信号的准确率在牛/熊市有显著差异
H2: 市场环境（涨跌家数比、行业板块资金方向）能预测信号质量
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from data.cn_fetcher import fetch_daily_kline, fetch_csi300_returns
from signals.scoring import compute_scores

TEST_CODES = [
    "600519", "000333", "300750", "600036", "000858",
    "601318", "002594", "600276", "600900", "002475",
    "601012", "300059", "600585", "000568", "002714",
]

# ── 1. 加载所有股票数据 ──
print("加载股票数据...")
all_scores = {}
all_returns = {}
for code in TEST_CODES:
    df = fetch_daily_kline(code)
    if df is not None and len(df) > 200:
        scores = compute_scores(df)
        all_scores[code] = scores
        all_returns[code] = df["close"].pct_change()
        print(f"  {code}: {len(df)} 天")

print(f"\n有效股票: {len(all_scores)} 只")

# ── 2. 加载市场数据 ──
print("\n加载市场数据...")
csi300 = fetch_csi300_returns()
print(f"  沪深300: {len(csi300) if csi300 is not None else 0} 天")

# ── 3. 按市场状态分层测试 ──
FORWARD = [5, 10, 20]

def market_regime(day_ret):
    """单日市场状态"""
    if day_ret > 0.01:  return "强势上涨 >1%"
    elif day_ret > 0:   return "小幅上涨 0-1%"
    elif day_ret > -0.01: return "小幅下跌 0-1%"
    else:               return "弱势下跌 <-1%"

print("\n" + "=" * 90)
print("H1 检验：市场状态分层 vs BUY 信号胜率")
print("=" * 90)

# 筛选大涨/大跌日
all_rows = []

for code, scores in all_scores.items():
    ret = all_returns[code]

    # 对齐 csi300
    common_idx = scores.index.intersection(csi300.index)
    if len(common_idx) < 100:
        continue

    for horizon in FORWARD:
        fwd = ret.shift(-horizon)
        fwd.name = "fwd_ret"

        merged = pd.concat([scores[["signal", "total_score"]], csi300, fwd], axis=1).dropna()
        merged.columns = ["signal", "score", "csi300", "fwd_ret"]

        # 按市场状态分组
        merged["regime"] = merged["csi300"].apply(market_regime)

        for regime in ["强势上涨 >1%", "小幅上涨 0-1%", "小幅下跌 0-1%", "弱势下跌 <-1%"]:
            sub = merged[merged["regime"] == regime]
            buy_sub = sub[sub["signal"] == "BUY"]
            sell_sub = sub[sub["signal"] == "SELL"]

            buy_win = (buy_sub["fwd_ret"] > 0).sum() if len(buy_sub) > 0 else 0
            buy_total = len(buy_sub)
            sell_win = (sell_sub["fwd_ret"] < 0).sum() if len(sell_sub) > 0 else 0
            sell_total = len(sell_sub)

            all_rows.append({
                "code": code, "horizon": horizon, "regime": regime,
                "buy_winrate": round(buy_win / buy_total * 100, 1) if buy_total > 5 else None,
                "buy_count": buy_total,
                "sell_winrate": round(sell_win / sell_total * 100, 1) if sell_total > 5 else None,
                "sell_count": sell_total,
                "market_days": len(sub),
            })

df_regime = pd.DataFrame(all_rows)

# 打印汇总：每个 regime 跨股票平均
for horizon in FORWARD:
    print(f"\n--- T+{horizon} ---")
    for regime in ["强势上涨 >1%", "小幅上涨 0-1%", "小幅下跌 0-1%", "弱势下跌 <-1%"]:
        sub = df_regime[(df_regime["horizon"] == horizon) & (df_regime["regime"] == regime)]
        buy_wr = sub["buy_winrate"].dropna()
        sell_wr = sub["sell_winrate"].dropna()
        total_buy = sub["buy_count"].sum()

        if len(buy_wr) > 0:
            print(f"  {regime:12s}: BUY胜率 {buy_wr.mean():5.1f}% (std={buy_wr.std():.1f}, n={len(buy_wr)}只, 共{total_buy}次) | "
                  f"SELL胜率 {sell_wr.mean():5.1f}%" if len(sell_wr) > 0 else f"  {regime:12s}: BUY胜率 {buy_wr.mean():5.1f}% (共{total_buy}次)")

# ── 4. H2: 信号质量与市场环境的相关性 ──
print("\n" + "=" * 90)
print("H2 检验：信号准确率是否与市场环境相关？")
print("=" * 90)

# 对每只股票的每一天：BUY信号是否正确 + 当天市场状态
for horizon in [5, 10]:
    print(f"\n--- T+{horizon} 逻辑回归近似 ---")

    buy_correct_list = []
    mkt_ret_list = []

    for code, scores in all_scores.items():
        ret = all_returns[code]
        common = scores.index.intersection(csi300.index)
        fwd = ret.shift(-horizon)

        merged = pd.concat([scores["signal"], csi300, fwd], axis=1).dropna()
        merged.columns = ["signal", "csi300", "fwd_ret"]

        buy_mask = merged["signal"] == "BUY"
        buy_correct = (merged.loc[buy_mask, "fwd_ret"] > 0).astype(int)
        buy_mkt = merged.loc[buy_mask, "csi300"]

        buy_correct_list.extend(buy_correct.tolist())
        mkt_ret_list.extend(buy_mkt.tolist())

    # 按市场分位数分5组，看胜率梯度
    mkt_arr = np.array(mkt_ret_list)
    correct_arr = np.array(buy_correct_list)

    if len(mkt_arr) > 100:
        quintiles = np.percentile(mkt_arr, [0, 20, 40, 60, 80, 100])
        print(f"  市场收益分位数: {[f'{q:.2%}' for q in quintiles]}")

        for i in range(5):
            mask = (mkt_arr >= quintiles[i]) & (mkt_arr < quintiles[i+1])
            if i == 4:
                mask = mkt_arr >= quintiles[i]
            wr = correct_arr[mask].mean() * 100 if mask.sum() > 10 else float('nan')
            print(f"  Q{i+1} (市场 {quintiles[i]:.2%} ~ {quintiles[i+1]:.2%}): "
                  f"BUY胜率 {wr:.1f}%, 样本 {mask.sum()}")

        # 相关性
        from scipy import stats
        if len(mkt_arr) > 1000:
            # 对大数据采样计算
            sample_n = min(5000, len(mkt_arr))
            idx = np.random.choice(len(mkt_arr), sample_n, replace=False)

            # 点二列相关
            r, p = stats.pointbiserialr(correct_arr[idx], mkt_arr[idx])
            print(f"\n  点二列相关系数 r={r:.4f}, p={p:.4f}")
            if p < 0.01:
                print(f"  ** 显著相关 ** → 市场环境影响信号质量")
            elif p < 0.05:
                print(f"  * 弱显著 * → 市场环境有一定影响")
            else:
                print(f"  不显著 → 市场环境与信号质量无统计关系")

# ── 5. 结论 ──
print("\n" + "=" * 90)
print("结论")
print("=" * 90)

# 计算大涨日 vs 大跌日的胜率差
for horizon in FORWARD:
    sub = df_regime[df_regime["horizon"] == horizon]
    bull = sub[sub["regime"] == "强势上涨 >1%"]["buy_winrate"].dropna()
    bear = sub[sub["regime"] == "弱势下跌 <-1%"]["buy_winrate"].dropna()
    neutral_up = sub[sub["regime"] == "小幅上涨 0-1%"]["buy_winrate"].dropna()
    neutral_down = sub[sub["regime"] == "小幅下跌 0-1%"]["buy_winrate"].dropna()

    print(f"\nT+{horizon}:")
    print(f"  强势上涨日 BUY 胜率: {bull.mean():.1f}% (n={len(bull)}只)")
    print(f"  小幅上涨日 BUY 胜率: {neutral_up.mean():.1f}% (n={len(neutral_up)}只)")
    print(f"  小幅下跌日 BUY 胜率: {neutral_down.mean():.1f}% (n={len(neutral_down)}只)")
    print(f"  弱势下跌日 BUY 胜率: {bear.mean():.1f}% (n={len(bear)}只)")
    if len(bull) > 0 and len(bear) > 0:
        print(f"  极差 (牛-熊): {bull.mean() - bear.mean():.1f}%")
