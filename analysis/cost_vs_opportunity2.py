"""
Cost efficiency, corrected: everything in PERCENTAGE terms.

v1 measured NIFTY in absolute index points over 2005-2026, when the index went
from ~2,000 to ~23,800 -- that shrinks the early-year medians and made the
long-horizon numbers look worse than they are. Percentages are scale-free.

Cost is fixed per round trip; the move grows with holding period. The ratio is
what decides cost effectiveness.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")
from data_loader import load_nifty

NIFTY_LOT, CRUDE_LOT = 65, 100
NIFTY_SPOT, CRUDE_INR = 23_800.0, 6_970.0

# break-even round-trip cost, as a FRACTION of position/notional value
BE_PCT = {
    "NIFTY futures":   918.0 / (NIFTY_SPOT * NIFTY_LOT),   # 0.059%
    "MCX CrudeOil":    175.0 / (CRUDE_INR * CRUDE_LOT),    # 0.025%
    "Equity intraday": 0.00082,
    "Equity delivery": 0.00241,
}

nifty_1m = load_nifty("1min", start="2022-01-03", end="2026-07-24")
bars15 = nifty_1m["close"].resample("15min").last().dropna()

nd_raw = pd.read_csv(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv",
                     parse_dates=["timestamp"])
nd = pd.Series(nd_raw["close"].to_numpy(dtype=float),
               index=pd.to_datetime(nd_raw["timestamp"], utc=True)
               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)).sort_index()
nd = nd[~nd.index.duplicated(keep="last")]

wti = pd.read_csv(r"D:\MyPython\Download_1min_History\data\wti\WTI_FRONT_daily.csv",
                  parse_dates=["date"])
wd = pd.Series(wti["close"].to_numpy(dtype=float), index=wti["date"]).sort_index()
wd[wd <= 0] = np.nan
wd = wd.dropna()

# a real single stock, not the index, for the equity rows
stk_raw = pd.read_csv(r"D:\MyPython\Download_1min_History\data\nifty50\RELIANCE_daily.csv",
                      parse_dates=["timestamp"])
stk = pd.Series(stk_raw["close"].to_numpy(dtype=float),
                index=pd.to_datetime(stk_raw["timestamp"], utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)).sort_index()
stk = stk[~stk.index.duplicated(keep="last")]


def med_pct(series, periods):
    return float((series.shift(-periods) / series - 1.0).abs().dropna().median())


rows = []
print("=" * 94)
print("COST AS A SHARE OF THE TYPICAL MOVE — all in % terms (lower = more cost effective)")
print("=" * 94)
print(f"{'segment / holding period':<40}{'typical move':>15}{'break-even':>13}{'cost share':>13}")
print("-" * 94)

specs = [
    ("NIFTY futures   ~1 hour",    bars15, 4,   "NIFTY futures"),
    ("NIFTY futures   ~3 hours",   bars15, 12,  "NIFTY futures"),
    ("NIFTY futures   1 day",      nd,     1,   "NIFTY futures"),
    ("NIFTY futures   5 days",     nd,     5,   "NIFTY futures"),
    ("NIFTY futures   20 days",    nd,     20,  "NIFTY futures"),
    ("NIFTY futures   60 days",    nd,     60,  "NIFTY futures"),
    ("MCX CrudeOil    1 day",      wd,     1,   "MCX CrudeOil"),
    ("MCX CrudeOil    5 days",     wd,     5,   "MCX CrudeOil"),
    ("MCX CrudeOil    20 days",    wd,     20,  "MCX CrudeOil"),
    ("Equity intraday 1 day",      stk,    1,   "Equity intraday"),
    ("Equity delivery 5 days",     stk,    5,   "Equity delivery"),
    ("Equity delivery 20 days",    stk,    20,  "Equity delivery"),
    ("Equity delivery 60 days",    stk,    60,  "Equity delivery"),
    ("Equity delivery 250 days",   stk,    250, "Equity delivery"),
]

for label, series, periods, key in specs:
    mv = med_pct(series, periods)
    be = BE_PCT[key]
    share = be / mv
    rows.append((label, share))
    print(f"{label:<40}{mv:>14.2%}{be:>12.3%}{share:>13.1%}")

# options: cost + theta against a delta-scaled index move
PREMIUM, DELTA = 175.0, 0.5
opt_cost = 230.0 / (PREMIUM * NIFTY_LOT)
print("-" * 94)
for label, series, periods, hours in [
    ("NIFTY option buy ~3 hours", bars15, 12, 3.0),
    ("NIFTY option buy 1 day",    nd,     1,  6.25),
]:
    idx_move = med_pct(series, periods)
    opt_move = idx_move * NIFTY_SPOT * DELTA / PREMIUM
    theta = 0.05 * (hours / 6.25)
    share = (opt_cost + theta) / opt_move
    rows.append((label, share))
    print(f"{label:<40}{opt_move:>14.1%}{opt_cost + theta:>12.1%}{share:>13.1%}"
          f"   (cost {opt_cost:.1%} + theta {theta:.1%})")

print("\n" + "=" * 94)
print("RANKING — share of the typical move handed over to costs")
print("=" * 94)
for label, share in sorted(rows, key=lambda x: x[1]):
    print(f"  {label:<40}{share:>8.1%}  {'#' * max(int(share * 120), 1)}")
