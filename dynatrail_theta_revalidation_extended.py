"""
DynaTrail theta re-validation — EXTENDED WINDOW (2026-08-14).

The existing dynatrail_theta_revalidation.py runs 2025-06-23 -> 2026-06-22
(1 year, 3 folds). The NIFTY Futures file on disk actually starts 2024-10-03,
so roughly 21 months are available — nearly double. This re-runs the same
theta OFF/ON comparison over the full history with 6 expanding folds.

Prompted by the live paper window: 8 trades, 8 losses, -Rs.14,950. Under the
theta-OFF benchmark that run of losses is a ~1-in-176 event; under a
theta-charged model it is ordinary. The question is which benchmark the
paper window should be judged against.

FOUR PARTS
  1. Data-integrity audit. The futures file has TWO multi-week holes, not one:
     2024-11-28 -> 2024-12-27 and 2025-03-27 -> 2025-05-02. Supertrend runs
     continuously across them, so the bars just after each gap carry a stale
     indicator plus a large price jump. Trades opened within GAP_QUARANTINE
     sessions of a gap are reported separately so their contribution is visible.
  2. Pinned production config over the full window, theta OFF vs ON. This is
     the honest benchmark for the live bot, which prices from real quotes and
     therefore already pays real theta.
  3. Walk-forward, config re-selected per fold from prior data only, OFF vs ON.
  4. FOLD-BOUNDARY SENSITIVITY. Fold dates are an arbitrary partition choice.
     This platform documented on 2026-08-14 that an arbitrary sampling-grid
     choice swung a headline number from 20% to 55% (see note 34, FIX 4).
     The same question applies here: shift every boundary by +/-3 weeks and
     see whether the pooled verdict moves. If it does, the verdict is noise.

Usage:
    py dynatrail_theta_revalidation_extended.py --nifty-data "D:\\...\\NIFTY_FUT_1min.csv"
"""

import argparse
import datetime as dt_mod
import itertools
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_loader import load_ohlcv
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption

VIX_CSV = r"D:\MyPython\Download_1min_History\data\vix\INDIAVIX_daily.csv"

FULL_START = "2024-10-03"          # earliest bar in the futures file
FULL_END = "2026-06-30"

FOLDS = [
    {"train_end": "2025-05-31", "oos_start": "2025-06-01", "oos_end": "2025-07-31"},
    {"train_end": "2025-07-31", "oos_start": "2025-08-01", "oos_end": "2025-09-30"},
    {"train_end": "2025-09-30", "oos_start": "2025-10-01", "oos_end": "2025-11-30"},
    {"train_end": "2025-11-30", "oos_start": "2025-12-01", "oos_end": "2026-01-31"},
    {"train_end": "2026-01-31", "oos_start": "2026-02-01", "oos_end": "2026-03-31"},
    {"train_end": "2026-03-31", "oos_start": "2026-04-01", "oos_end": "2026-06-30"},
]

PRODUCTION_CONFIG = {"st_period": 10, "st_mult": 2.5, "atr_min": 25,
                     "premium_sl_pct": 0.40, "max_hold_bars": 12}

ST_PERIODS = [7, 10]
ST_MULTS = [2.0, 2.5, 3.0]
ATR_MINS = [25, 30]
PREMIUM_SLS = [0.30, 0.40, 0.50]
MAX_HOLDS = [8, 12]
MIN_TRADES_FOR_SELECTION = 15

GAP_QUARANTINE = 2                 # sessions after a data gap to flag

VIX_MAP = None
COST_PER_ORDER = 50.0
SLIPPAGE_PTS = 1.0
CAPITAL = 500_000


def run_bt(config, df, theta, want_trades=False):
    if len(df) < 100:
        return None, None
    cfg = {**config, "capital": CAPITAL, "iv": 0.13, "cb_consec_losses": 0,
           "vix_by_date": VIX_MAP, "vix_scale": 1.0,
           "cost_per_order": COST_PER_ORDER, "slippage_pts_per_leg": SLIPPAGE_PTS,
           "intraday_theta": theta}
    res = DynaTrailNiftyOption(df, cfg).run()
    if want_trades:
        t = pd.DataFrame([{"entry_time": x.entry_time, "net_pnl": x.pnl,
                           "exit_reason": x.exit_reason} for x in res.trades])
    else:
        t = pd.DataFrame([{"net_pnl": x.pnl} for x in res.trades])
    return t, res.summary()


def grid_best(df_train, theta):
    rows = []
    for stp, stm, amin, psl, hold in itertools.product(
            ST_PERIODS, ST_MULTS, ATR_MINS, PREMIUM_SLS, MAX_HOLDS):
        c = {"st_period": stp, "st_mult": stm, "atr_min": amin,
             "premium_sl_pct": psl, "max_hold_bars": hold}
        _, m = run_bt(c, df_train, theta)
        if m is None:
            continue
        rows.append({**c, "trades": m["num_trades"], "profit_factor": m["profit_factor"]})
    rdf = pd.DataFrame(rows)
    robust = rdf[rdf["trades"] >= MIN_TRADES_FOR_SELECTION]
    if robust.empty:
        robust = rdf
    b = robust.sort_values("profit_factor", ascending=False).iloc[0]
    return {"st_period": int(b["st_period"]), "st_mult": float(b["st_mult"]),
            "atr_min": int(b["atr_min"]), "premium_sl_pct": float(b["premium_sl_pct"]),
            "max_hold_bars": int(b["max_hold_bars"])}, float(b["profit_factor"])


def pooled(tl):
    valid = [t for t in tl if t is not None and len(t)]
    if not valid:
        return {"trades": 0, "profit_factor": 0.0, "net_pnl": 0.0, "win_rate": 0.0}
    a = pd.concat(valid, ignore_index=True)
    w = a[a["net_pnl"] > 0]["net_pnl"].sum()
    l = -a[a["net_pnl"] < 0]["net_pnl"].sum()
    return {"trades": len(a), "profit_factor": (w / l) if l > 0 else float("inf"),
            "net_pnl": a["net_pnl"].sum(), "win_rate": (a["net_pnl"] > 0).mean() * 100}


def shift_folds(folds, weeks):
    d = pd.Timedelta(weeks=weeks)
    out = []
    for f in folds:
        out.append({k: (pd.Timestamp(v) + d).strftime("%Y-%m-%d") for k, v in f.items()})
    return out


def main():
    global VIX_MAP, COST_PER_ORDER, SLIPPAGE_PTS
    p = argparse.ArgumentParser()
    p.add_argument("--nifty-data", required=True)
    p.add_argument("--cost-per-order", type=float, default=50.0)
    p.add_argument("--slippage-pts", type=float, default=1.0)
    p.add_argument("--skip-grid", action="store_true",
                   help="skip part 3 (the 864-run grid walk-forward)")
    args = p.parse_args()

    COST_PER_ORDER, SLIPPAGE_PTS = args.cost_per_order, args.slippage_pts
    v = pd.read_csv(VIX_CSV)
    v["date"] = pd.to_datetime(v["date"]).dt.date
    VIX_MAP = dict(zip(v["date"], v["close"]))

    df = load_ohlcv(args.nifty_data, start=FULL_START, end=FULL_END,
                    session_start="09:15", session_end="15:30")
    print(f"Cost: Rs.{COST_PER_ORDER:g}/order + {SLIPPAGE_PTS:g} pt/leg | "
          f"IV: VIX-proxy prev-day close")
    print(f"Loaded {len(df):,} bars, {df.index.min()} -> {df.index.max()}")

    # ── 1. data integrity ──
    print("\n" + "=" * 78)
    print("  1. DATA INTEGRITY — the extended window is not continuous")
    print("=" * 78)
    sess = pd.Series(sorted(set(df.index.normalize())))
    gaps = sess.diff().dt.days
    holes = [(sess[i - 1], sess[i], int(gaps[i])) for i in range(1, len(sess)) if gaps[i] > 5]
    print(f"  {len(sess)} sessions. Multi-week holes: {len(holes)}")
    quarantined = set()
    for a, b, n in holes:
        print(f"    {a.date()} -> {b.date()}  ({n} calendar days missing)")
        after = sess[sess >= b].head(GAP_QUARANTINE)
        quarantined.update(after.dt.normalize())
    print(f"  Supertrend runs across these holes with a stale band and a price jump.")
    print(f"  Trades opened in the first {GAP_QUARANTINE} sessions after a hole are "
          f"flagged below ({len(quarantined)} sessions quarantined).")

    # ── 2. pinned config, full window ──
    print("\n" + "=" * 78)
    print("  2. PINNED PRODUCTION CONFIG, FULL EXTENDED WINDOW — theta OFF vs ON")
    print("=" * 78)
    res = {}
    for theta in (False, True):
        t, m = run_bt(PRODUCTION_CONFIG, df, theta, want_trades=True)
        res[theta] = (t, m)
    print(f"  {'':<24}{'theta OFF':>14}{'theta ON':>14}")
    for k, lbl, f in [("profit_factor", "Profit factor", "{:.3f}"),
                      ("num_trades", "Trades", "{:d}"),
                      ("win_rate", "Win rate", "{:.1%}"),
                      ("net_pnl", "Net P&L Rs.", "{:,.0f}"),
                      ("max_drawdown", "Max drawdown", "{:.1%}"),
                      ("sharpe_ratio", "Sharpe", "{:.2f}")]:
        print(f"  {lbl:<24}{f.format(res[False][1][k]):>14}{f.format(res[True][1][k]):>14}")
    pf_off, pf_on = res[False][1]["profit_factor"], res[True][1]["profit_factor"]
    print(f"\n  PF change from charging real time decay: "
          f"{(pf_on/pf_off - 1)*100:+.1f}%   (ADX1's was -45% and it was retired)")

    for theta in (False, True):
        t = res[theta][0]
        t["date"] = pd.to_datetime(t["entry_time"]).dt.normalize()
        q = t[t["date"].isin(quarantined)]
        clean = t[~t["date"].isin(quarantined)]
        if len(q):
            cw = clean[clean["net_pnl"] > 0]["net_pnl"].sum()
            cl = -clean[clean["net_pnl"] <= 0]["net_pnl"].sum()
            print(f"  [theta {'ON ' if theta else 'OFF'}] post-gap trades: {len(q)} "
                  f"(net Rs.{q['net_pnl'].sum():,.0f}) | excluding them: {len(clean)} trades, "
                  f"PF {(cw/cl if cl else 0):.3f}, net Rs.{clean['net_pnl'].sum():,.0f}")

    # ── 3. walk-forward with per-fold grid selection ──
    summary = {}
    if not args.skip_grid:
        print("\n" + "=" * 78)
        print("  3. WALK-FORWARD, config re-selected per fold — theta OFF vs ON")
        print("=" * 78)
        for theta in (False, True):
            tag = "ON " if theta else "OFF"
            oos_list, per_fold = [], []
            for i, f in enumerate(FOLDS, 1):
                dtr = df[df.index < f["train_end"]]
                doo = df[(df.index >= f["oos_start"]) & (df.index <= f["oos_end"])]
                if len(dtr) < 100 or len(doo) < 100:
                    print(f"  [theta {tag}] Fold {i}: SKIPPED (insufficient bars)")
                    continue
                cfg, tr_pf = grid_best(dtr, theta)
                t, m = run_bt(cfg, doo, theta)
                oos_list.append(t)
                per_fold.append((i, cfg, tr_pf, m))
                print(f"  [theta {tag}] Fold {i} {f['oos_start']}->{f['oos_end']}: "
                      f"ST{cfg['st_period']}/{cfg['st_mult']}/atr{cfg['atr_min']}/"
                      f"psl{cfg['premium_sl_pct']:.0%}/hold{cfg['max_hold_bars']}  "
                      f"train_PF={tr_pf:.2f} -> OOS_PF={m['profit_factor']:.2f} "
                      f"({m['num_trades']}tr) net=Rs.{m['net_pnl']:,.0f}", flush=True)
            pl = pooled(oos_list)
            summary[tag.strip()] = (pl, per_fold)
            print(f"  [theta {tag}] POOLED: PF={pl['profit_factor']:.2f}  "
                  f"{pl['trades']} trades  WR={pl['win_rate']:.1f}%  "
                  f"net=Rs.{pl['net_pnl']:,.0f}\n", flush=True)

    # ── 4. fold-boundary sensitivity (the FIX 4 analogue) ──
    print("\n" + "=" * 78)
    print("  4. FOLD-BOUNDARY SENSITIVITY — pinned config, boundaries shifted")
    print("=" * 78)
    print("  Fold dates are arbitrary. If the verdict moves when they move, it is noise.")
    print(f"  {'shift':<12}{'theta OFF pooled PF':>22}{'theta ON pooled PF':>22}{'ON net Rs.':>14}")
    sens = []
    for weeks in (-3, -1, 0, 1, 3):
        row = {"shift_weeks": weeks}
        for theta in (False, True):
            oos = []
            for f in shift_folds(FOLDS, weeks):
                doo = df[(df.index >= f["oos_start"]) & (df.index <= f["oos_end"])]
                if len(doo) < 100:
                    continue
                t, _ = run_bt(PRODUCTION_CONFIG, doo, theta)
                oos.append(t)
            pl = pooled(oos)
            row[f"pf_{'on' if theta else 'off'}"] = pl["profit_factor"]
            row[f"net_{'on' if theta else 'off'}"] = pl["net_pnl"]
            row[f"trades_{'on' if theta else 'off'}"] = pl["trades"]
        sens.append(row)
        print(f"  {weeks:+d} weeks{'':<4}{row['pf_off']:>22.3f}{row['pf_on']:>22.3f}"
              f"{row['net_on']:>14,.0f}", flush=True)
    sdf = pd.DataFrame(sens)
    print(f"\n  theta-ON pooled PF across boundary shifts: "
          f"{sdf['pf_on'].min():.3f} to {sdf['pf_on'].max():.3f}  "
          f"(spread {sdf['pf_on'].max()-sdf['pf_on'].min():.3f})")
    crosses = (sdf["pf_on"] < 1.0).any() and (sdf["pf_on"] >= 1.0).any()
    print(f"  Does the PF>1.0 verdict flip with an arbitrary boundary shift? "
          f"{'YES — the verdict is not robust' if crosses else 'No — verdict stable'}")

    # ── verdict ──
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  Pinned config, full 21-month window: PF {pf_off:.3f} (OFF) -> {pf_on:.3f} (ON), "
          f"net Rs.{res[False][1]['net_pnl']:,.0f} -> Rs.{res[True][1]['net_pnl']:,.0f}")
    if summary:
        off, on = summary["OFF"][0], summary["ON"][0]
        folds_pos = sum(1 for _, _, _, m in summary["ON"][1] if m["profit_factor"] > 1.0)
        n_folds = len(summary["ON"][1])
        print(f"  Walk-forward pooled OOS:  PF {off['profit_factor']:.2f} (OFF) -> "
              f"{on['profit_factor']:.2f} (ON), net Rs.{off['net_pnl']:,.0f} -> "
              f"Rs.{on['net_pnl']:,.0f}")
        print(f"  Folds with PF>1.0, theta ON: {folds_pos}/{n_folds}")
        if on["profit_factor"] >= 1.2 and on["net_pnl"] > 0 and folds_pos >= n_folds * 0.6:
            print("\n  [SURVIVES] Edge holds with real time decay charged.")
        elif on["profit_factor"] >= 1.0 and on["net_pnl"] > 0:
            print("\n  [WEAKENED BUT POSITIVE] Edge survives but is materially smaller than "
                  "the recorded benchmark.")
        else:
            print("\n  [DOES NOT SURVIVE] Once real time decay is charged, the backtested "
                  "edge does not hold over the full available history.")
    print("=" * 78)

    ts = dt_mod.datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    out = os.path.join(BASE_DIR, "results", f"dynatrail_theta_extended_{ts}.csv")
    rows = []
    for theta_tag, (t, m) in (("OFF", res[False]), ("ON", res[True])):
        rows.append({"part": "pinned_full", "theta": theta_tag, **{
            k: m[k] for k in ("profit_factor", "num_trades", "win_rate", "net_pnl",
                              "max_drawdown", "sharpe_ratio")}})
    for tag in summary:
        pl, pf_list = summary[tag]
        for i, cfg, tr_pf, m in pf_list:
            rows.append({"part": "walkforward", "theta": tag, "fold": i, **cfg,
                         "train_pf": tr_pf, "oos_pf": m["profit_factor"],
                         "oos_trades": m["num_trades"], "oos_net": m["net_pnl"]})
        rows.append({"part": "walkforward", "theta": tag, "fold": "POOLED",
                     "oos_pf": pl["profit_factor"], "oos_trades": pl["trades"],
                     "oos_net": pl["net_pnl"]})
    for r in sens:
        rows.append({"part": "fold_sensitivity", **r})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
