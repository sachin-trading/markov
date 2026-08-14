# =============================================================================
# phase_robustness.py — FIX 4: is the stride matrix an artifact of grid phase?
#
# FIX 1 says: never build the matrix from overlapping windows, use a
# non-overlapping stride instead. True, but incomplete — a stride grid must
# START somewhere, and there are `stride` equally valid starting bars. Nothing
# in the method says to check whether the answer depends on that arbitrary
# choice. This script checks.
#
# Run on both instruments, because the answer revises earlier NIFTY reporting.
# =============================================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import markov_regime as mk
from filter_dynatrail import load_daily_close
from crude_regime_report import load_contract, CONTRACT, BAR, WINDOW, PCT

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def analyse(states: pd.Series, label: str, stride: int):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    single = mk.to_probabilities(mk.transition_counts(states, stride=stride))
    print(f"Single-phase matrix (offset 0 — what the method as written produces):")
    print("  signal by state: " + "  ".join(
        f"{mk.STATE_NAMES[s]} {mk.signal(single, s):+.3f}" for s in range(3)))

    pr = mk.phase_report(states, stride)
    print(f"\nAcross all {pr['n_phases']} equally valid grid phases "
          f"(~{pr['mean_transitions']:.0f} transitions each):")
    print(mk.format_phase_report(pr))

    unstable = [mk.STATE_NAMES[s] for s in range(3) if not pr["sign_stable"][s]]
    if unstable:
        print(f"\n  -> the signal SIGN flips across phases for: {', '.join(unstable)}.")
        print("     Those signals are grid artifacts, not market structure.")
    else:
        print("\n  -> all signal signs are stable across phases.")

    print(f"\n  P(BULL next | BEAR now): single-phase {single[mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"phase mean {pr['P_mean'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"range [{pr['P_min'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"{pr['P_max'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}]")
    return pr, single


def main():
    # NIFTY daily, as originally reported
    nifty = load_daily_close()
    n_states = mk.label_states(nifty, WINDOW, 0.05, -0.05)
    n_pr, n_single = analyse(n_states, "NIFTY 50 daily — 20-day window, ±5% thresholds", WINDOW)

    # CrudeOil MCX intraday
    bars = load_contract(CONTRACT)["close"].resample(BAR).last().dropna()
    bull, bear = mk.percentile_thresholds(bars, WINDOW, PCT)
    c_states = mk.label_states(bars, WINDOW, bull, bear)
    c_pr, c_single = analyse(c_states,
                             f"MCX CrudeOil {CONTRACT} — 20 x {BAR} bars, percentile thresholds",
                             WINDOW)

    # ── chart ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (pr, single, name) in zip(axes, ((n_pr, n_single, "NIFTY 50 daily"),
                                             (c_pr, c_single, f"CrudeOil {CONTRACT} 15-min"))):
        x = np.arange(3)
        ax.bar(x, pr["signal_mean"], color="#1f77b4", alpha=0.85, label="mean across phases")
        ax.errorbar(x, pr["signal_mean"],
                    yerr=[pr["signal_mean"] - pr["signal_min"],
                          pr["signal_max"] - pr["signal_mean"]],
                    fmt="none", ecolor="black", capsize=6, lw=1.4,
                    label="range across all 20 phases")
        ax.scatter(x, [mk.signal(single, s) for s in range(3)], color="#d62728", zorder=5,
                   s=60, label="single phase (as reported)")
        ax.axhline(0, color="black", lw=0.9)
        ax.axhline(0.10, color="grey", ls=":", lw=1)
        ax.axhline(-0.10, color="grey", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([mk.STATE_NAMES[s] for s in range(3)])
        ax.set_ylabel("signal  P(bull) − P(bear)")
        ax.set_title(f"{name}\nsignal vs sampling-grid phase")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png = OUT / "phase_robustness.png"
    fig.savefig(png, dpi=130)
    print(f"\nChart saved: {png}")

    pd.DataFrame({
        "instrument": ["NIFTY"] * 3 + [f"CRUDE_{CONTRACT}"] * 3,
        "state": [mk.STATE_NAMES[s] for s in range(3)] * 2,
        "signal_single_phase": [mk.signal(n_single, s) for s in range(3)]
                               + [mk.signal(c_single, s) for s in range(3)],
        "signal_phase_mean": list(n_pr["signal_mean"]) + list(c_pr["signal_mean"]),
        "signal_phase_min": list(n_pr["signal_min"]) + list(c_pr["signal_min"]),
        "signal_phase_max": list(n_pr["signal_max"]) + list(c_pr["signal_max"]),
        "sign_stable": list(n_pr["sign_stable"]) + list(c_pr["sign_stable"]),
    }).to_csv(OUT / "phase_robustness.csv", index=False)
    print(f"Saved: {OUT / 'phase_robustness.csv'}")


if __name__ == "__main__":
    main()
