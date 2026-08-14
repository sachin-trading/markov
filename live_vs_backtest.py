# =============================================================================
# live_vs_backtest.py — is the 8/8 losing paper window bad luck, or the known
# unmodelled-theta gap?
#
# Note 34 flags a chassis-level gap: dte_years() defaults to whole-day
# granularity, so with intraday_theta=False NO DynaTrail trade ever pays any
# time decay -- every trade closes intraday, so T never changes within a trade.
# The live bot prices from real option-chain quotes and therefore pays REAL
# theta. Precedent: ADX1 passed both pre-registered gates at PF 1.46 and
# collapsed to 0.80 once theta was charged, and was retired.
#
# The live trades are afternoon entries held to the 15:15 EOD exit -- exactly
# the population where afternoon theta on a near-expiry ATM option bites
# hardest, and exactly what the backtest cannot see.
#
# This compares the SAME config with theta off vs on, on the live-like subgroup.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")

from data_loader import load_nifty
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption
from filter_dynatrail import PRODUCTION_CONFIG, VIX_CSV, START, END

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Live paper trades as reported from dynatrail_paper.sqlite3 on 2026-08-14
LIVE = [
    ("PE", 24050.0, "2026-07-15T13:00:03", "2026-07-15T15:15:02", -2265.25, "eod"),
    ("CE", 24150.0, "2026-07-16T11:00:03", "2026-07-16T13:37:48", -4355.00, "premium_sl"),
    ("PE", 24050.0, "2026-07-16T13:45:03", "2026-07-16T15:15:12", -2102.75, "eod"),
    ("CE", 24250.0, "2026-07-20T14:00:03", "2026-07-20T15:15:07", -767.00, "eod"),
    ("CE", 23800.0, "2026-07-24T12:30:04", "2026-07-24T15:15:07", -978.25, "eod"),
    ("PE", 24500.0, "2026-08-05T13:00:03", "2026-08-05T15:15:02", -3009.50, "eod"),
    ("CE", 24600.0, "2026-08-10T12:30:04", "2026-08-10T15:15:08", -1400.75, "eod"),
    ("CE", 24350.0, "2026-08-12T15:00:04", "2026-08-12T15:15:13", -71.50, "eod"),
]


def pf(p):
    w, l = p[p > 0].sum(), -p[p <= 0].sum()
    return float(w / l) if l > 0 else (float("inf") if w > 0 else 0.0)


def run(theta: bool):
    df = load_nifty("1min", start=START, end=END)
    vix = pd.read_csv(VIX_CSV)
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    cfg = {**PRODUCTION_CONFIG, "vix_by_date": dict(zip(vix["date"], vix["close"])),
           "vix_scale": 1.0, "intraday_theta": theta}
    res = DynaTrailNiftyOption(df, cfg).run()
    t = pd.DataFrame([{"entry_time": x.entry_time, "exit_time": x.exit_time,
                       "pnl": x.pnl, "exit_reason": x.exit_reason} for x in res.trades])
    t["hour"] = t["entry_time"].dt.hour
    t["hold_min"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60
    return t


def main():
    live = pd.DataFrame(LIVE, columns=["side", "strike", "entry", "exit", "pnl", "reason"])
    live["entry"] = pd.to_datetime(live["entry"])
    live["hour"] = live["entry"].dt.hour
    live["hold_min"] = (pd.to_datetime(live["exit"]) - live["entry"]).dt.total_seconds() / 60

    print("LIVE PAPER WINDOW (8 trades, dynatrail_paper.sqlite3)")
    print(f"  net ₹{live['pnl'].sum():,.2f}   wins {int((live['pnl'] > 0).sum())}/{len(live)}"
          f"   avg ₹{live['pnl'].mean():,.0f}   median hold {live['hold_min'].median():.0f} min")
    print(f"  exit reasons: {dict(live['reason'].value_counts())}")
    print(f"  entry hours : {dict(live['hour'].value_counts().sort_index())}")

    off, on = run(False), run(True)

    print("\n" + "=" * 74)
    print("EXIT-REASON MIX — live vs backtest (theta off = the current benchmark)")
    print("=" * 74)
    mix_o = off["exit_reason"].str.lower().value_counts(normalize=True)
    mix_l = live["reason"].str.lower().value_counts(normalize=True)
    print(f"  {'reason':<16}{'backtest':>10}{'live':>10}")
    for r in ["eod", "max_hold", "premium_sl", "expiry_close", "st_reversal"]:
        print(f"  {r:<16}{mix_o.get(r, 0):>9.0%}{mix_l.get(r, 0):>10.0%}")
    print(f"\n  Backtest MAX_HOLD trades carry PF "
          f"{pf(off.loc[off['exit_reason']=='MAX_HOLD','pnl']):.2f} and "
          f"₹{off.loc[off['exit_reason']=='MAX_HOLD','pnl'].sum():,.0f} of profit — "
          f"live has produced ZERO of them.")

    print("\n" + "=" * 74)
    print("THETA: same config, same window, decay off vs on")
    print("=" * 74)
    print(f"  {'':<26}{'theta OFF':>14}{'theta ON':>14}")
    rows = [("all trades", lambda d: len(d)),
            ("profit factor", lambda d: pf(d["pnl"])),
            ("win rate", lambda d: (d["pnl"] > 0).mean()),
            ("net ₹", lambda d: d["pnl"].sum()),
            ("avg ₹/trade", lambda d: d["pnl"].mean())]
    for name, fn in rows:
        a, b = fn(off), fn(on)
        fmt = "{:.0f}" if name == "all trades" else ("{:.1%}" if name == "win rate"
                                                     else ("{:.2f}" if name == "profit factor"
                                                           else "{:,.0f}"))
        print(f"  {name:<26}{fmt.format(a):>14}{fmt.format(b):>14}")

    # live-like population: afternoon entries that run to the EOD exit
    print("\n  LIVE-LIKE subgroup (entry >= 12:00 AND exit reason EOD):")
    print(f"  {'':<26}{'theta OFF':>14}{'theta ON':>14}")
    for name, fn in rows:
        sa = off[(off["hour"] >= 12) & (off["exit_reason"] == "EOD")]
        sb = on[(on["hour"] >= 12) & (on["exit_reason"] == "EOD")]
        a, b = fn(sa), fn(sb)
        fmt = "{:.0f}" if name == "all trades" else ("{:.1%}" if name == "win rate"
                                                     else ("{:.2f}" if name == "profit factor"
                                                           else "{:,.0f}"))
        print(f"  {name:<26}{fmt.format(a):>14}{fmt.format(b):>14}")

    # how surprising is 8/8 losses under each benchmark?
    print("\n" + "=" * 74)
    print("IS 8/8 LOSSES SURPRISING?")
    print("=" * 74)
    for label, d in (("theta OFF (current benchmark)", off), ("theta ON", on)):
        sub = d[(d["hour"] >= 12) & (d["exit_reason"] == "EOD")]
        for scope, s in (("all trades", d), ("live-like subgroup", sub)):
            wr = (s["pnl"] > 0).mean()
            p = (1 - wr) ** len(live)
            print(f"  {label:<32}{scope:<22} WR {wr:.1%}  "
                  f"P(8 straight losses) = {p:.4%}  ~1 in {1/p:,.0f}" if p > 0 else "")

    pd.concat([off.assign(theta="off"), on.assign(theta="on")]).to_csv(
        OUT / "theta_comparison.csv", index=False)
    print(f"\nSaved: {OUT / 'theta_comparison.csv'}")


if __name__ == "__main__":
    main()
