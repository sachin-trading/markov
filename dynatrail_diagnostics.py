# =============================================================================
# dynatrail_diagnostics.py — the per-trade dump note 34 says is missing
#
# Note 34 (2026-07-26) flags three questions as UNRESOLVED and blocked by the
# same thing: walkforward_validate_dynatrail.py reduces every trade to
# {"net_pnl": t.pnl} and writes a 3-row fold summary, discarding the entry/exit
# times, exit reasons and strikes that TradeRecord already carries.
#
#   Q1 chop-sampler — entry needs a Supertrend FLIP, so chop yields many entries
#      and a sustained trend yields one. If the walk-forward sampled a
#      trend-richer period than live, PF 1.08 is itself flattering.
#   Q2 entry-time truncation — all 5 live entries were at/after 11:00, four at/
#      after 12:30, so the 12-bar (3h) hold was cut short by the 15:15 EOD exit
#      on 4 of 5. A trade that cannot complete its designed hold is not the
#      strategy that was tested.
#   Q3 exit-reason mix.
#
# READ-ONLY diagnostic. No strategy change, no parameter tuning, no P&L claim.
# Regime is classified with the platform's OWN classify_trend_range() from vault
# note 10 — not Markov labels — so the answer does not depend on a method this
# project has just closed.
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

EOD = pd.Timestamp("15:15").time()
BAR_MIN = 15


def classify_trend_range(df, fast=20, slow=50, slope_lookback=10):
    """Verbatim from vault note 10 — the platform's own classifier."""
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()
    df["fast_slope"] = df["ma_fast"].diff(slope_lookback)
    df["slow_slope"] = df["ma_slow"].diff(slope_lookback)
    cond_up = ((df["fast_slope"] > 0) & (df["slow_slope"] > 0)
               & (df["close"] > df["ma_fast"]) & (df["close"] > df["ma_slow"])
               & (df["ma_fast"] > df["ma_slow"]))
    cond_down = ((df["fast_slope"] < 0) & (df["slow_slope"] < 0)
                 & (df["close"] < df["ma_fast"]) & (df["close"] < df["ma_slow"])
                 & (df["ma_fast"] < df["ma_slow"]))
    df["regime_basic"] = "range"
    df.loc[cond_up, "regime_basic"] = "uptrend"
    df.loc[cond_down, "regime_basic"] = "downtrend"
    return df


def pf(pnl: pd.Series) -> float:
    w, l = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return float(w / l) if l > 0 else float("inf")


def block(label):
    print(f"\n{'='*74}\n{label}\n{'='*74}")


def main():
    df = load_nifty("1min", start=START, end=END)
    vix = pd.read_csv(VIX_CSV)
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    cfg = {**PRODUCTION_CONFIG, "vix_by_date": dict(zip(vix["date"], vix["close"])),
           "vix_scale": 1.0}
    res = DynaTrailNiftyOption(df, cfg).run()

    t = pd.DataFrame([{
        "entry_time": x.entry_time, "exit_time": x.exit_time, "direction": x.direction,
        "entry_spot": x.entry_spot, "strike": x.strike, "entry_prem": x.entry_prem,
        "exit_prem": x.exit_prem, "pnl": x.pnl, "exit_reason": x.exit_reason,
    } for x in res.trades])
    t["date"] = t["entry_time"].dt.normalize()
    t["entry_hhmm"] = t["entry_time"].dt.strftime("%H:%M")
    t["hold_min"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60

    # bars of session left at entry vs the 12-bar design hold
    mins_to_eod = t["entry_time"].apply(
        lambda ts: (pd.Timestamp.combine(ts.date(), EOD) - ts).total_seconds() / 60)
    t["bars_available"] = (mins_to_eod // BAR_MIN).clip(lower=0).astype(int)
    t["full_hold_possible"] = t["bars_available"] >= cfg["max_hold_bars"]

    t.to_csv(OUT / "dynatrail_trades.csv", index=False)
    print(f"Per-trade dump: {len(t)} trades -> {OUT / 'dynatrail_trades.csv'}")
    print(f"Window {START} -> {END}, pinned production config, "
          f"costs ₹50/order + 1pt/leg. Overall PF {pf(t['pnl']):.3f}, "
          f"net ₹{t['pnl'].sum():,.0f}")

    # ── Q3 exit reasons ──
    block("Q3 — exit-reason mix (what actually closes these trades)")
    ex = t.groupby("exit_reason").agg(trades=("pnl", "size"), pf=("pnl", pf),
                                      net=("pnl", "sum"), avg=("pnl", "mean"))
    ex["share"] = ex["trades"] / len(t)
    print(ex.sort_values("trades", ascending=False).to_string(
        formatters={"pf": "{:.2f}".format, "net": "{:,.0f}".format,
                    "avg": "{:,.0f}".format, "share": "{:.0%}".format}))
    clock = t["exit_reason"].isin(["EOD", "MAX_HOLD", "EXPIRY_CLOSE"]).mean()
    print(f"\n  {clock:.0%} of trades are closed by a CLOCK (EOD / MAX_HOLD / EXPIRY), "
          f"not by a signal.")
    print(f"  Only {(t['exit_reason']=='ST_REVERSAL').mean():.0%} exit on the Supertrend "
          f"reversal the thesis is built on.")

    # ── Q2 entry time and truncation ──
    block("Q2 — entry time vs the 12-bar (3h) design hold")
    t["entry_hour"] = t["entry_time"].dt.hour
    by_hour = t.groupby("entry_hour").agg(trades=("pnl", "size"), pf=("pnl", pf),
                                          avg=("pnl", "mean"),
                                          full_hold=("full_hold_possible", "mean"))
    print(by_hour.to_string(formatters={"pf": "{:.2f}".format, "avg": "{:,.0f}".format,
                                        "full_hold": "{:.0%}".format}))
    print(f"\n  {'group':<34}{'trades':>8}{'PF':>8}{'avg ₹':>10}{'net ₹':>12}")
    for label, mask in (("full 12-bar hold possible", t["full_hold_possible"]),
                        ("TRUNCATED by 15:15 EOD", ~t["full_hold_possible"])):
        s = t.loc[mask, "pnl"]
        print(f"  {label:<34}{len(s):>8}{pf(s):>8.2f}{s.mean():>10,.0f}{s.sum():>12,.0f}")
    print(f"\n  Truncated trades are {(~t['full_hold_possible']).mean():.0%} of all trades.")

    # ── Q1 chop-sampler ──
    block("Q1 — chop-sampler: which regime do the trades actually come from?")
    daily = df["close"].resample("1D").last().dropna().to_frame()
    daily = classify_trend_range(daily)
    reg = daily["regime_basic"]
    reg.index = reg.index.normalize()

    sessions = pd.Series(df.index.normalize().unique()).sort_values()
    sess_reg = reg.reindex(sessions).dropna()
    avail = sess_reg.value_counts(normalize=True)

    t["regime"] = t["date"].map(reg)
    tr = t.dropna(subset=["regime"])
    traded = tr["regime"].value_counts(normalize=True)

    print(f"  {'regime':<12}{'% of sessions':>15}{'% of trades':>14}{'over/under':>13}"
          f"{'PF':>8}{'avg ₹':>10}")
    for r in ["uptrend", "downtrend", "range"]:
        a, td = avail.get(r, 0.0), traded.get(r, 0.0)
        sub = tr.loc[tr["regime"] == r, "pnl"]
        ratio = (td / a) if a > 0 else float("nan")
        print(f"  {r:<12}{a:>14.1%}{td:>14.1%}{ratio:>12.2f}x"
              f"{(pf(sub) if len(sub) else float('nan')):>8.2f}"
              f"{(sub.mean() if len(sub) else float('nan')):>10,.0f}")

    tpd = tr.groupby("regime").size() / sess_reg.value_counts()
    print(f"\n  Trades per session by regime: "
          + ", ".join(f"{k} {v:.2f}" for k, v in tpd.sort_values(ascending=False).items()))
    print(f"  -> the chop-sampler claim is {'CONFIRMED' if tpd.get('range', 0) > max(tpd.get('uptrend', 0), tpd.get('downtrend', 0)) else 'NOT confirmed'}: "
          f"'range' sessions generate "
          f"{tpd.get('range', float('nan')) / max(tpd.drop('range').max(), 1e-9):.2f}x "
          f"the trades per session of the best trending regime.")

    pd.concat([ex, by_hour], axis=0, keys=["exit_reason", "entry_hour"]).to_csv(
        OUT / "dynatrail_diagnostics.csv")
    print(f"\nSaved: {OUT / 'dynatrail_diagnostics.csv'}")
    print("\nREAD-ONLY diagnostic. Any rule change suggested by these numbers is a NEW "
          "registry entry in note 40, tested out of sample — these are in-sample "
          "observations and subgroup PFs are exactly the kind of thing that overfits.")


if __name__ == "__main__":
    main()
