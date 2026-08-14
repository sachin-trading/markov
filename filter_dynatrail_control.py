# =============================================================================
# filter_dynatrail_control.py — the control the FILTER result demands
#
# The gated arm took 41 CE trades and 0 PE trades, because the stride matrix's
# signal is positive from ALL THREE states on NIFTY (upward drift): BEAR +0.48,
# BULL +0.13, SIDEWAYS +0.04. So the gate is really two rules at once:
#   (a) never buy puts            <- a long-only bias, nothing to do with regime
#   (b) only trade in BULL/BEAR windows, never SIDEWAYS  <- the actual regime call
#
# If (a) explains the whole improvement, the gate adds nothing. This script
# isolates it: compare the gated CE trades against the UNGATED CE-only trades
# over the same window. Any edge left after that is attributable to (b).
# =============================================================================

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")

import markov_regime as mk
from data_loader import load_nifty
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption
from filter_dynatrail import (DAILY_CSV, VIX_CSV, PRODUCTION_CONFIG, START, END,
                              WINDOW, BULL_THR, BEAR_THR, SIGNAL_THR, MIN_TRAIN,
                              load_daily_close, build_gate, OUT)


def metrics(trades) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "net_pnl": 0.0, "avg_pnl": 0.0}
    pnl = pd.Series([t.pnl for t in trades])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {
        "trades": len(pnl),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "net_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
    }


def main():
    close_d = load_daily_close()
    mk.assert_labels_verified(close_d, WINDOW, BULL_THR, BEAR_THR)
    wf = mk.walk_forward(close_d, WINDOW, BULL_THR, BEAR_THR, matrix_mode="stride",
                         min_train_bars=MIN_TRAIN, signal_threshold=SIGNAL_THR,
                         apply_costs=False)

    df = load_nifty("1min", start=START, end=END)
    vix = pd.read_csv(VIX_CSV)
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    vix_map = dict(zip(vix["date"], vix["close"]))

    gate = build_gate(wf["signals"], sorted(set(df.index.date)), SIGNAL_THR)

    base = {**PRODUCTION_CONFIG, "vix_by_date": vix_map, "vix_scale": 1.0}
    ungated = DynaTrailNiftyOption(df, base).run()
    gated = DynaTrailNiftyOption(df, {**base, "side_gate_by_date": gate}).run()

    ce_ungated = [t for t in ungated.trades if t.direction == "CE"]
    pe_ungated = [t for t in ungated.trades if t.direction == "PE"]
    ce_gated = [t for t in gated.trades if t.direction == "CE"]

    rows = {
        "UNGATED all": metrics(ungated.trades),
        "UNGATED CE-only": metrics(ce_ungated),
        "UNGATED PE-only": metrics(pe_ungated),
        "GATED (CE only)": metrics(ce_gated),
    }

    print("\nCONTROL — is the gate's edge just 'never buy puts'?")
    print(f"{'':<20}{'trades':>8}{'win%':>8}{'PF':>8}{'net ₹':>12}{'avg ₹/trade':>14}")
    for k, m in rows.items():
        print(f"{k:<20}{m['trades']:>8}{m['win_rate']:>8.1%}{m['profit_factor']:>8.2f}"
              f"{m['net_pnl']:>12,.0f}{m['avg_pnl']:>14,.0f}")

    ce_u, ce_g = rows["UNGATED CE-only"], rows["GATED (CE only)"]
    print(f"\nLike-for-like (CE trades only, identical engine and costs):")
    print(f"  ungated CE : PF {ce_u['profit_factor']:.2f} over {ce_u['trades']} trades, "
          f"avg ₹{ce_u['avg_pnl']:,.0f}/trade")
    print(f"  gated   CE : PF {ce_g['profit_factor']:.2f} over {ce_g['trades']} trades, "
          f"avg ₹{ce_g['avg_pnl']:,.0f}/trade")
    lift = ce_g["avg_pnl"] - ce_u["avg_pnl"]
    print(f"  regime lift on the SAME side: {lift:+,.0f} ₹/trade "
          f"({'gate helps' if lift > 0 else 'gate does NOT help'} beyond the long-only bias)")

    # How much of the ungated damage was the puts?
    pe = rows["UNGATED PE-only"]
    print(f"\n  For reference, the PE trades the gate silently removed: {pe['trades']} trades, "
          f"PF {pe['profit_factor']:.2f}, net ₹{pe['net_pnl']:,.0f} "
          f"({'profitable — the gate threw away money' if pe['net_pnl'] > 0 else 'loss-making'})")

    pd.DataFrame(rows).T.to_csv(OUT / "filter_dynatrail_control.csv")
    print(f"\nSaved: {OUT / 'filter_dynatrail_control.csv'}")


if __name__ == "__main__":
    main()
