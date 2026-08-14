"""
Live logged 31 Supertrend flips. Where do they fall in the session, and does
the backtest's flip distribution match?

Live flips (from dynatrail_paper.log, 2026-07-07 -> 2026-08-14).
The bot's entry window is 09:30-14:45, so flips outside it are discarded.
"""
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")
from data_loader import load_nifty
from strategies.dynatrail_nifty_option import compute_supertrend

LIVE = """2026-07-07 15:00|2026-07-09 09:15|2026-07-09 15:00|2026-07-10 09:00
2026-07-13 09:00|2026-07-13 12:00|2026-07-14 09:00|2026-07-15 09:15
2026-07-15 12:45|2026-07-16 10:45|2026-07-16 13:30|2026-07-17 09:15
2026-07-20 09:00|2026-07-20 13:45|2026-07-21 09:00|2026-07-24 12:15
2026-07-31 09:15|2026-07-31 15:15|2026-08-04 09:15|2026-08-04 15:15
2026-08-05 12:45|2026-08-05 15:15|2026-08-06 09:00|2026-08-06 15:15
2026-08-07 09:00|2026-08-10 12:15|2026-08-11 09:00|2026-08-12 14:45
2026-08-13 09:15|2026-08-13 15:15|2026-08-14 09:15"""
live = [pd.Timestamp(x.strip()) for x in LIVE.replace("\n", "|").split("|") if x.strip()]

ENTRY_OPEN, ENTRY_CLOSE = pd.Timestamp("09:30").time(), pd.Timestamp("14:45").time()


def bucket(t):
    if t < ENTRY_OPEN:
        return "BEFORE 09:30 (discarded)"
    if t > ENTRY_CLOSE:
        return "AFTER 14:45 (discarded)"
    return "tradeable 09:30-14:45"


print(f"LIVE — {len(live)} flips logged, {live[0].date()} -> {live[-1].date()}")
lc = Counter(bucket(t.time()) for t in live)
for k in ("BEFORE 09:30 (discarded)", "tradeable 09:30-14:45", "AFTER 14:45 (discarded)"):
    print(f"  {k:<28}{lc[k]:>4}  ({lc[k]/len(live):>5.1%})")
print("  live flip times: " + ", ".join(
    f"{k}×{v}" for k, v in sorted(Counter(t.strftime('%H:%M') for t in live).items())))

# ── backtest flip distribution, same config, same instrument family ──
df = load_nifty("1min", start="2022-01-03", end="2026-07-24")
bars = df.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last"}).dropna()
st = compute_supertrend(bars, 10, 2.5)
flips = st[st["st_flip"] != 0]
print(f"\nBACKTEST — {len(flips):,} flips, {bars.index[0].date()} -> {bars.index[-1].date()}"
      f"  ({bars.index.normalize().nunique()} sessions)")
bc = Counter(bucket(t.time()) for t in flips.index)
for k in ("BEFORE 09:30 (discarded)", "tradeable 09:30-14:45", "AFTER 14:45 (discarded)"):
    print(f"  {k:<28}{bc[k]:>4}  ({bc[k]/len(flips):>5.1%})")

print("\n  backtest flips by bar time:")
for t, n in sorted(Counter(x.strftime("%H:%M") for x in flips.index).items()):
    print(f"    {t}  {n:>4}  ({n/len(flips):>5.1%})")

print(f"\nFirst-bar share — live {lc['BEFORE 09:30 (discarded)']/len(live):.1%} "
      f"vs backtest {bc['BEFORE 09:30 (discarded)']/len(flips):.1%}")
print(f"Flips per session — live {len(live)/28:.2f} (28 sessions), "
      f"backtest {len(flips)/bars.index.normalize().nunique():.2f}")
