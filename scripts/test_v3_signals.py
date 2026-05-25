"""测试 V3 信号系统：基本面BUY + 技术风控SELL。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from data.cn_fetcher import fetch_daily_kline, tencent_quote, fetch_finance
from signals.scoring import compute_signals, risk_check, fundamental_score
from signals import indicators as ind

MAX_HOLD_DAYS = 60

TEST_CODES = [
    "600519", "000333", "300750", "600036", "000858",
    "601318", "002594", "600276", "600900", "002475",
    "601012", "300059", "600585", "000568", "002714",
]

# ═══════════════════════════════════════════════════════════
# Part 1: 当前基本面评分
# ═══════════════════════════════════════════════════════════
print("=" * 90)
print("Part 1: 当前基本面评分")
print("=" * 90)

vals = tencent_quote(TEST_CODES)

print(f"\n{'代码':<8} {'名称':<8} {'PE':>7} {'PB':>6} {'市值(亿)':>9} {'ROE':>7} {'PE分':>5} {'PB分':>5} {'ROE分':>6} {'市值分':>6} {'总分':>5} {'信号':>6}")
print("-" * 90)

for code in TEST_CODES:
    v = vals.get(code, {})
    fin = fetch_finance(code)

    score = fundamental_score(v, fin)
    s = score

    name = v.get("name", code)[:6]
    pe = v.get("pe_ttm", 0) or 0
    pb = v.get("pb", 0) or 0
    mcap = v.get("mcap_yi", 0) or 0
    roe = fin.get("roe", 0) if fin else 0

    # 判断信号
    if pe <= 0:
        sig = "AVOID"
    elif s["total_score"] >= 60:
        sig = "BUY"
    elif s["total_score"] >= 40:
        sig = "HOLD"
    else:
        sig = "AVOID"

    print(f"  {code:<6} {name:<8} {pe:>6.1f} {pb:>5.2f} {mcap:>8.0f} {roe:>6.1f}% "
          f"{s['pe_score']:>4} {s['pb_score']:>4} {s['roe_score']:>5} {s['mcap_score']:>5} "
          f"{s['total_score']:>4}  {sig:>6}")

# ═══════════════════════════════════════════════════════════
# Part 2: 技术风控回测（从任意入场点，风控能否控制亏损）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("Part 2: 技术风控回测")
print("规则: 止损-8% | 移动止盈-5% | HMA死叉 | 最大持有60天")
print("=" * 90)

results = []

for code in TEST_CODES:
    df = fetch_daily_kline(code)
    if df is None or len(df) < 120:
        continue

    close = df["close"]

    # 随机采样入场点 (每隔30天取一个入场点，模拟不同市场环境)
    entry_indices = list(range(60, len(df) - 30, 30))

    exits = []
    for ei in entry_indices:
        entry_price = close.iloc[ei]
        entry_date = df.index[ei]

        # 模拟持仓：每天检查风控
        exited = False
        exit_idx = ei
        exit_reason = "到期"

        for j in range(ei + 1, min(ei + MAX_HOLD_DAYS + 5, len(df))):
            sub_df = df.iloc[:j+1]
            sub_df = sub_df.copy()  # avoid view issues
            check = risk_check(sub_df, entry_price, entry_date)
            if check["signal"] == "SELL":
                exit_idx = j
                exit_reason = check["reason"]
                exited = True
                break

        exit_price = close.iloc[exit_idx]
        ret = (exit_price - entry_price) / entry_price

        # 买入持有同期
        bh_exit = close.iloc[min(ei + MAX_HOLD_DAYS, len(df) - 1)]
        bh_ret = (bh_exit - entry_price) / entry_price

        exits.append({
            "entry": entry_date,
            "entry_price": entry_price,
            "exit_idx": exit_idx,
            "exit_price": exit_price,
            "ret": ret,
            "bh_ret": bh_ret,
            "excess": ret - bh_ret,
            "reason": exit_reason,
            "triggered": exited,
        })

    if not exits:
        continue

    # 汇总该股票的风控效果
    triggered = [e for e in exits if e["triggered"]]
    avg_ret = np.mean([e["ret"] for e in exits])
    avg_bh = np.mean([e["bh_ret"] for e in exits])
    avg_excess = np.mean([e["excess"] for e in exits])

    # 风控是否减少了亏损？
    # 比较: 被风控触发卖出 vs 如果继续持有60天
    loss_saved = []
    for e in triggered:
        saved = e["ret"] - e["bh_ret"]
        loss_saved.append(saved)

    reasons = {}
    for e in triggered:
        r = e["reason"].split("→")[0].strip()[:10]
        reasons[r] = reasons.get(r, 0) + 1

    print(f"\n--- {code} ---")
    print(f"  模拟入场 {len(exits)} 次 | 风控触发 {len(triggered)} 次 ({len(triggered)/len(exits)*100:.0f}%)")
    print(f"  平均收益: {avg_ret*100:.1f}% | 买入持有: {avg_bh*100:.1f}% | 超额: {avg_excess*100:.1f}%")
    if loss_saved:
        avg_saved = np.mean(loss_saved) * 100
        print(f"  风控减少亏损: {avg_saved:.1f}% (风控卖出 vs 继续持有)")
    print(f"  触发原因: {reasons}")

# ═══════════════════════════════════════════════════════════
# Part 3: 死叉风控有效性
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("Part 3: HMA 死叉卖出有效性（所有死叉点后20天收益）")
print("=" * 90)

all_death_rets = []
for code in TEST_CODES:
    df = fetch_daily_kline(code)
    if df is None: continue

    close = df["close"]
    hma20 = ind.hma(close, 20)
    hma50 = ind.hma(close, 50)

    death = (hma20 < hma50) & (hma20.shift(1) >= hma50.shift(1))
    death_dates = df.index[death.values]

    for d in death_dates:
        try:
            idx = close.index.get_loc(d)
            end = min(idx + 20, len(close) - 1)
            ret = (close.iloc[end] - close.iloc[idx]) / close.iloc[idx]
            all_death_rets.append({"code": code, "date": d, "ret_20d": ret})
        except Exception:
            pass

if all_death_rets:
    dead_df = pd.DataFrame(all_death_rets)
    dead_neg = (dead_df["ret_20d"] < 0).sum()
    dead_pos = (dead_df["ret_20d"] > 0).sum()
    print(f"  死叉后20天: 共 {len(dead_df)} 次 | 下跌 {dead_neg} 次 ({dead_neg/len(dead_df)*100:.1f}%) | "
          f"上涨 {dead_pos} 次 | 均收益 {dead_df['ret_20d'].mean()*100:.2f}%")
    for code in TEST_CODES:
        sub = dead_df[dead_df["code"] == code]
        if len(sub) > 0:
            neg_pct = (sub["ret_20d"] < 0).sum() / len(sub) * 100
            print(f"  {code}: {len(sub)}次死叉, 后20天下跌 {neg_pct:.0f}%")
