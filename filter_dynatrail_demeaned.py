# =============================================================================
# filter_dynatrail_demeaned.py — Registry Entry #016
#
# Entry #015's gate compared the raw signal against ±0.10. On NIFTY the raw
# signal is positive from all three states (upward drift), so "PE when
# signal < -0.10" could never fire and the filter silently became long-only.
#
# Entry #016 changes exactly one thing: the signal is compared against its own
# EXPANDING-WINDOW MEAN instead of against zero.
#     m(t)  = mean of every signal value observed strictly before day t
#     CE if signal(t) - m(t) > +0.10
#     PE if signal(t) - m(t) < -0.10
#     else no entry
# Everything else — DynaTrail pinned production config, exits ungated, data
# window, IV policy, costs, capital, lot — is identical to Entry #015, and the
# same ungated arm is the control.
#
# Pass criteria (fixed before this ran, vault note 40 Entry #016):
#   beat the UNGATED control on BOTH profit factor and max drawdown,
#   AND permit >= 20 PE trades. <20 trades on either side => INCONCLUSIVE.
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")

import markov_regime as mk
from data_loader import load_nifty
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption, print_result
from filter_dynatrail import (VIX_CSV, PRODUCTION_CONFIG, START, END,
                              WINDOW, BULL_THR, BEAR_THR, SIGNAL_THR, MIN_TRAIN,
                              load_daily_close, build_gate, OUT)

N_PERM = 20000
RNG = np.random.default_rng(20260814)


def demean_expanding(signals: pd.Series) -> pd.Series:
    """signal(t) - mean(signal values strictly before t). No lookahead:
    .shift(1) excludes t itself from its own reference mean."""
    sig = signals.dropna()
    running_mean = sig.expanding().mean().shift(1)
    return (sig - running_mean).dropna()


def side_metrics(trades, side=None) -> dict:
    sel = [t for t in trades if side is None or t.direction == side]
    if not sel:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "net_pnl": 0.0, "avg_pnl": 0.0}
    pnl = pd.Series([t.pnl for t in sel])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {"trades": len(pnl), "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "net_pnl": float(pnl.sum()), "avg_pnl": float(pnl.mean())}


def permutation_p(gated_pnl: np.ndarray, ungated_pnl: np.ndarray) -> float:
    """P(random same-side subsample of the ungated trades >= observed mean)."""
    n = len(gated_pnl)
    if n == 0 or n > len(ungated_pnl):
        return float("nan")
    idx = RNG.random((N_PERM, len(ungated_pnl))).argsort(axis=1)[:, :n]
    return float((ungated_pnl[idx].mean(axis=1) >= gated_pnl.mean()).mean())


def main():
    # ── signal, de-meaned (FIX 2 gate first) ──
    close_d = load_daily_close()
    mk.assert_labels_verified(close_d, WINDOW, BULL_THR, BEAR_THR)
    wf = mk.walk_forward(close_d, WINDOW, BULL_THR, BEAR_THR, matrix_mode="stride",
                         min_train_bars=MIN_TRAIN, signal_threshold=SIGNAL_THR,
                         apply_costs=False)
    raw = wf["signals"].dropna()
    dem = demean_expanding(wf["signals"])
    print(f"Signal: {len(raw)} daily values, {raw.index[0].date()} -> {raw.index[-1].date()}")
    print(f"  raw      : min {raw.min():+.3f}  mean {raw.mean():+.3f}  max {raw.max():+.3f}  "
          f"-> {(raw < -SIGNAL_THR).sum()} days below -{SIGNAL_THR} (the unreachable short leg)")
    print(f"  de-meaned: min {dem.min():+.3f}  mean {dem.mean():+.3f}  max {dem.max():+.3f}  "
          f"-> {(dem < -SIGNAL_THR).sum()} days below -{SIGNAL_THR}, "
          f"{(dem > SIGNAL_THR).sum()} days above +{SIGNAL_THR}")

    # ── data, identical to Entry #015 ──
    df = load_nifty("1min", start=START, end=END)
    vix = pd.read_csv(VIX_CSV)
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    vix_map = dict(zip(vix["date"], vix["close"]))

    session_dates = sorted(set(df.index.date))
    gate = build_gate(dem, session_dates, SIGNAL_THR)
    n_ce = sum(1 for a in gate.values() if "CE" in a)
    n_pe = sum(1 for a in gate.values() if "PE" in a)
    n_block = sum(1 for a in gate.values() if not a)
    print(f"\nGate exposure over {len(gate)} sessions: CE allowed {n_ce} ({n_ce/len(gate):.0%}), "
          f"PE allowed {n_pe} ({n_pe/len(gate):.0%}), fully blocked {n_block} ({n_block/len(gate):.0%})")

    # ── both arms ──
    base = {**PRODUCTION_CONFIG, "vix_by_date": vix_map, "vix_scale": 1.0}
    ungated = DynaTrailNiftyOption(df, base).run()
    gated = DynaTrailNiftyOption(df, {**base, "side_gate_by_date": gate}).run()
    print_result(gated, f"DynaTrail GATED (de-meaned, Entry #016) — {START}->{END}")

    mu, mg = ungated.summary(), gated.summary()
    print("\nEntry #016 vs control (identical engine/config/costs; gate only):")
    print(f"{'':<18}{'UNGATED':>14}{'GATED #016':>14}")
    for k, fmt in (("num_trades", "{:d}"), ("win_rate", "{:.1%}"),
                   ("profit_factor", "{:.3f}"), ("net_pnl", "₹{:,.0f}"),
                   ("max_drawdown", "{:.1%}"), ("sharpe_ratio", "{:.2f}"),
                   ("long_trades", "{:d}"), ("short_trades", "{:d}")):
        print(f"{k:<18}{fmt.format(mu[k]):>14}{fmt.format(mg[k]):>14}")

    # ── same-side controls: does the regime beat a plain side bias? ──
    print("\nSame-side control (the check Entry #015 taught us to always run):")
    print(f"{'':<22}{'trades':>8}{'PF':>8}{'avg ₹/trade':>14}{'perm p':>10}")
    for side in ("CE", "PE"):
        u, g = side_metrics(ungated.trades, side), side_metrics(gated.trades, side)
        pu = np.array([t.pnl for t in ungated.trades if t.direction == side], dtype=float)
        pg = np.array([t.pnl for t in gated.trades if t.direction == side], dtype=float)
        p = permutation_p(pg, pu)
        print(f"  ungated {side:<13}{u['trades']:>8}{u['profit_factor']:>8.2f}{u['avg_pnl']:>14,.0f}{'':>10}")
        print(f"  gated   {side:<13}{g['trades']:>8}{g['profit_factor']:>8.2f}{g['avg_pnl']:>14,.0f}"
              f"{(f'{p:.4f}' if p == p else 'n/a'):>10}")

    # ── verdict against the pre-registered criteria ──
    pf_pass = mg["profit_factor"] > mu["profit_factor"]
    dd_pass = mg["max_drawdown"] > mu["max_drawdown"]  # less negative = shallower
    pe_pass = mg["short_trades"] >= 20
    thin = mg["long_trades"] < 20 or mg["short_trades"] < 20

    print("\nVERDICT vs pre-registered Entry #016 criteria:")
    print(f"  beat control on profit factor : {'PASS' if pf_pass else 'FAIL'} "
          f"({mg['profit_factor']:.3f} vs {mu['profit_factor']:.3f})")
    print(f"  beat control on max drawdown  : {'PASS' if dd_pass else 'FAIL'} "
          f"({mg['max_drawdown']:.1%} vs {mu['max_drawdown']:.1%})")
    print(f"  >= 20 PE trades permitted     : {'PASS' if pe_pass else 'FAIL'} "
          f"({mg['short_trades']} PE trades)")
    if thin:
        print(f"  [INCONCLUSIVE] fewer than 20 trades on one side "
              f"(CE {mg['long_trades']}, PE {mg['short_trades']}) — "
              f"sample too thin to judge either way.")
    elif pf_pass and dd_pass and pe_pass:
        print("  [PASS] all three conditions met.")
    else:
        print("  [FAIL] at least one pre-registered condition not met.")

    # ── chart: all three arms ──
    fig, ax = plt.subplots(figsize=(11, 6))
    for res, color, label in ((ungated, "#7f7f7f", "DynaTrail UNGATED (control)"),
                              (gated, "#2ca02c", "GATED — de-meaned signal (Entry #016)")):
        ax.plot(res.equity.index, res.equity, color=color, lw=1.4, label=label)
    raw_gate = build_gate(raw, session_dates, SIGNAL_THR)
    raw_gated = DynaTrailNiftyOption(df, {**base, "side_gate_by_date": raw_gate}).run()
    ax.plot(raw_gated.equity.index, raw_gated.equity, color="#1f77b4", lw=1.2, ls="--",
            label="GATED — raw signal (Entry #015, long-only artifact)")
    ax.set_title(f"Markov 2.0 FILTER — raw vs de-meaned gate on DynaTrail, {START} -> {END}")
    ax.set_ylabel("Equity (₹, capital 500,000)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / "markov2_filter_demeaned.png"
    fig.savefig(png, dpi=130)
    print(f"\nChart saved: {png}")

    pd.DataFrame({"UNGATED": mu, "GATED_demeaned": mg,
                  "GATED_raw_entry015": raw_gated.summary()}).to_csv(
        OUT / "filter_demeaned_summary.csv")
    print(f"Summary saved: {OUT / 'filter_demeaned_summary.csv'}")


if __name__ == "__main__":
    main()
