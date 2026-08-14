# =============================================================================
# filter_significance.py — is the gated CE edge real, or 41 lucky trades?
#
# The control showed gated CE trades average ₹956 vs ₹82 for ungated CE.
# 41 trades is thin. Two checks:
#   1. Are the gated CE trades a strict subset of the ungated CE trades?
#      (If yes, the comparison is a clean selection test. If not, skipping
#      entries changed which later trades were available and the arms are
#      not directly comparable trade-for-trade.)
#   2. Permutation test: if the regime label carried no information, how often
#      would 41 CE trades drawn at random from the ungated CE population beat
#      the observed mean? That p-value is the honest read on 41 trades.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")

import markov_regime as mk
from data_loader import load_nifty
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption
from filter_dynatrail import (VIX_CSV, PRODUCTION_CONFIG, START, END,
                              WINDOW, BULL_THR, BEAR_THR, SIGNAL_THR, MIN_TRAIN,
                              load_daily_close, build_gate, OUT)

N_PERM = 20000
RNG = np.random.default_rng(20260814)


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

    ce_u = [t for t in ungated.trades if t.direction == "CE"]
    ce_g = [t for t in gated.trades if t.direction == "CE"]

    # ── 1. subset check ──
    u_keys = {(t.entry_time, t.direction) for t in ce_u}
    g_keys = {(t.entry_time, t.direction) for t in ce_g}
    shared = g_keys & u_keys
    print(f"Subset check: {len(shared)}/{len(g_keys)} gated CE trades have an identical "
          f"entry timestamp in the ungated run.")
    if len(shared) == len(g_keys):
        print("  -> gated CE trades are a strict SUBSET of ungated CE trades; the gate purely "
              "selects, it never creates new trades. Clean selection test.")
    else:
        print(f"  -> {len(g_keys) - len(shared)} gated trades have no ungated twin: skipping an "
              f"entry left the strategy flat and it caught a later flip the ungated arm missed. "
              f"Permutation test below is therefore approximate.")

    # ── 2. permutation test on the shared population ──
    pnl_u = np.array([t.pnl for t in ce_u], dtype=float)
    pnl_g = np.array([t.pnl for t in ce_g], dtype=float)
    n, obs = len(pnl_g), pnl_g.mean()

    # each permutation draws n distinct trades from the ungated CE population
    idx = RNG.random((N_PERM, len(pnl_u))).argsort(axis=1)[:, :n]
    null_means = pnl_u[idx].mean(axis=1)
    p = float((null_means >= obs).mean())

    print(f"\nPermutation test ({N_PERM:,} random {n}-trade draws from the {len(pnl_u)} "
          f"ungated CE trades):")
    print(f"  observed gated mean : ₹{obs:,.0f}/trade")
    print(f"  null mean           : ₹{null_means.mean():,.0f}/trade "
          f"(5th–95th pct: ₹{np.percentile(null_means,5):,.0f} to ₹{np.percentile(null_means,95):,.0f})")
    print(f"  p-value             : {p:.4f}  "
          f"({'significant at 5%' if p < 0.05 else 'NOT significant at 5% — consistent with luck'})")

    # ── context: what the gate gave up ──
    pe_u = np.array([t.pnl for t in ungated.trades if t.direction == "PE"], dtype=float)
    print(f"\nOpportunity cost: gate blocked {len(pe_u)} PE trades worth ₹{pe_u.sum():,.0f} net.")
    print(f"  Gated total net ₹{sum(t.pnl for t in gated.trades):,.0f} vs "
          f"ungated ₹{sum(t.pnl for t in ungated.trades):,.0f}.")

    pd.DataFrame({
        "metric": ["gated_ce_trades", "gated_ce_mean_pnl", "ungated_ce_trades",
                   "ungated_ce_mean_pnl", "permutation_p", "blocked_pe_trades",
                   "blocked_pe_net_pnl"],
        "value": [n, obs, len(pnl_u), pnl_u.mean(), p, len(pe_u), pe_u.sum()],
    }).to_csv(OUT / "filter_significance.csv", index=False)
    print(f"\nSaved: {OUT / 'filter_significance.csv'}")


if __name__ == "__main__":
    main()
