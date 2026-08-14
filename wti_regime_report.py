# =============================================================================
# wti_regime_report.py — Markov 2.0 on WTI as the long-history proxy for MCX
#
# MCX CrudeOil is cash-settled against NYMEX WTI and the proxy is verified in
# prepare_wti.py (MCX vs WTI 20-day return correlation 0.93-0.99 on the 73-54
# overlapping sessions of real MCX data). WTI gives 40 years where MCX gives 4
# months, so this is the first properly-powered look at crude regimes.
#
# All four fixes applied:
#   FIX 1  stride-sampled matrix shown against the overlapping one
#   FIX 2  label verification on generic anchors (COVID/election are NSE-only)
#   FIX 3  n/a here - descriptive, no mode chosen, no P&L
#   FIX 4  the matrix across ALL sampling-grid phases - the test that has
#          dissolved every apparent signal in this project so far
#
# Plus the conditional forward-return study with non-overlapping block t-stats:
# the same test that returned t=0.93 on NIFTY, now on a much larger sample.
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

WTI_CSV = Path(r"D:\MyPython\Download_1min_History\data\wti\WTI_FRONT_daily.csv")
NIFTY_CSV = Path(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

WINDOW = 20
PCT = 0.25


def load_wti() -> pd.Series:
    df = pd.read_csv(WTI_CSV, parse_dates=["date"])
    s = pd.Series(df["close"].to_numpy(dtype=float), index=df["date"]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    # 2020-04-20 settled at -$36.98. A percentage return across a non-positive
    # price is meaningless, so blank it: every window using it as an endpoint
    # becomes NaN and drops out, instead of silently producing nonsense.
    bad = (s <= 0).sum()
    s[s <= 0] = np.nan
    if bad:
        print(f"  blanked {bad} non-positive price(s); windows using them as an "
              f"endpoint are dropped")
    return s


def block_tstat(x: pd.Series, block: int = WINDOW) -> float:
    b = x.iloc[::block]
    return float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))) if len(b) > 2 and b.std(ddof=1) > 0 else float("nan")


def main():
    print("WTI daily (EIA Cushing spot, proxy for MCX CrudeOil — see prepare_wti.py)")
    wti = load_wti()
    print(f"  {wti.notna().sum():,} usable days, {wti.index[0].date()} -> {wti.index[-1].date()} "
          f"({(wti.index[-1]-wti.index[0]).days/365.25:.0f} years)")

    # ── calibration ──
    ret = mk.window_returns(wti, WINDOW).dropna()
    nif = pd.read_csv(NIFTY_CSV, parse_dates=["timestamp"])
    nret = mk.window_returns(pd.Series(nif["close"].to_numpy(dtype=float)), WINDOW)
    print(f"\nCalibration:")
    print(f"  WTI   20-day return sd {ret.std():.2%}  range {ret.min():+.1%} to {ret.max():+.1%}")
    print(f"  NIFTY 20-day return sd {nret.std():.2%}  range {nret.min():+.1%} to {nret.max():+.1%}")
    base = mk.label_states(wti, WINDOW, 0.05, -0.05)
    d0 = mk.state_distribution(base)
    print(f"  with default ±5%: " + ", ".join(f"{k} {v:.1%}" for k, v in d0.items()))

    if min(d0.values()) < 0.10:
        bull_thr, bear_thr = mk.percentile_thresholds(wti, WINDOW, PCT)
        print(f"  -> a state is under 10%; percentile fallback applies: "
              f"BULL >= {bull_thr:+.2%}, BEAR <= {bear_thr:+.2%}")
    else:
        bull_thr, bear_thr = 0.05, -0.05
        print(f"  -> all states above 10%, so the default ±5% thresholds are kept "
              f"(crude is volatile enough that they fit here, unlike intraday MCX)")

    # ── FIX 2 ──
    checks = mk.assert_labels_verified_generic(wti, WINDOW, bull_thr, bear_thr)
    print("\nLabel verification (FIX 2, generic anchors):")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    states = mk.label_states(wti, WINDOW, bull_thr, bear_thr)
    print("\nState distribution: "
          + ", ".join(f"{k} {v:.1%}" for k, v in mk.state_distribution(states).items()))

    # ── FIX 1 ──
    rep = mk.matrix_report(states, WINDOW)
    print(f"\nOVERLAPPING matrix (legacy — shares {WINDOW-1} of {WINDOW} days, NOT honest):")
    print(mk.format_matrix(rep["overlap"]["P"], rep["overlap"]["counts"]))
    print(f"\nSTRIDE-SAMPLED matrix (phase 0):")
    print(mk.format_matrix(rep["stride"]["P"], rep["stride"]["counts"]))
    print(f"  transitions: {rep['stride']['n_transitions']} "
          f"(vs 266 for NIFTY over 21 years)")
    if rep["stride"]["unreliable"]:
        print(f"  ⚠ cells with <{mk.MIN_CELL_OBS} obs: " + ", ".join(rep["stride"]["unreliable"]))
    print("\nStickiness:")
    print(f"  overlap: { {k: round(v,3) for k,v in mk.stickiness(rep['overlap']['P']).items()} }")
    print(f"  stride : { {k: round(v,3) for k,v in mk.stickiness(rep['stride']['P']).items()} }")

    # ── FIX 4: the decisive test ──
    pr = mk.phase_report(states, WINDOW)
    print(f"\nFIX 4 — all {pr['n_phases']} sampling-grid phases "
          f"(~{pr['mean_transitions']:.0f} transitions each):")
    print(mk.format_phase_report(pr))
    unstable = [mk.STATE_NAMES[s] for s in range(3) if not pr["sign_stable"][s]]
    if unstable:
        print(f"  -> signal sign FLIPS across phases for: {', '.join(unstable)} — artifact.")
    else:
        print("  -> every signal sign is STABLE across all phases. First time in this "
              "project that a signal has survived FIX 4.")
    print(f"  P(BULL|BEAR): phase-0 {rep['stride']['P'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"phase mean {pr['P_mean'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"range [{pr['P_min'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}, "
          f"{pr['P_max'][mk.STATE_BEAR, mk.STATE_BULL]:.1%}]")

    # ── conditional forward returns, block t-stats (phase-independent) ──
    fwd = (wti.shift(-WINDOW) / wti - 1.0).reindex(states.index)
    d = pd.DataFrame({"state": states, "fwd": fwd}).dropna()
    print(f"\nForward {WINDOW}-day return conditional on state "
          f"(non-overlapping blocks, so t-stats are honest):")
    print(f"  {'state':<10}{'days':>7}{'indep':>7}{'mean fwd':>11}{'median':>10}{'win%':>7}{'t-stat':>9}")
    for s in (mk.STATE_BEAR, mk.STATE_SIDEWAYS, mk.STATE_BULL):
        sub = d[d["state"] == s]["fwd"]
        print(f"  {mk.STATE_NAMES[s]:<10}{len(sub):>7}{len(sub[::WINDOW]):>7}"
              f"{sub.mean():>10.2%}{sub.median():>10.2%}{(sub>0).mean():>7.0%}"
              f"{block_tstat(sub):>9.2f}")
    allf = d["fwd"]
    print(f"  {'ALL':<10}{len(allf):>7}{len(allf[::WINDOW]):>7}"
          f"{allf.mean():>10.2%}{allf.median():>10.2%}{(allf>0).mean():>7.0%}"
          f"{block_tstat(allf):>9.2f}   <- unconditional baseline")

    print("\nEdge over baseline by state, and stability by decade:")
    for s in (mk.STATE_BEAR, mk.STATE_BULL):
        sub = d[d["state"] == s]["fwd"]
        print(f"  {mk.STATE_NAMES[s]}: {sub.mean()-allf.mean():+.2%} overall")
        for lo, hi in (("1986", "1995"), ("1996", "2005"), ("2006", "2015"), ("2016", "2026")):
            w = d.loc[lo:hi]
            ws = w[w["state"] == s]["fwd"]
            if len(ws) > WINDOW:
                print(f"      {lo}-{hi}: {ws.mean()-w['fwd'].mean():+7.2%} "
                      f"({len(ws):>4} days, {len(ws[::WINDOW])} indep)")

    # ── chart ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3)
    axes[0].bar(x, pr["signal_mean"], color="#1f77b4", alpha=0.85, label="mean across phases")
    axes[0].errorbar(x, pr["signal_mean"],
                     yerr=[pr["signal_mean"] - pr["signal_min"],
                           pr["signal_max"] - pr["signal_mean"]],
                     fmt="none", ecolor="black", capsize=6, lw=1.4, label="range over 20 phases")
    axes[0].axhline(0, color="black", lw=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([mk.STATE_NAMES[s] for s in range(3)])
    axes[0].set_ylabel("signal  P(bull) − P(bear)")
    axes[0].set_title("WTI 1986–2026: signal vs grid phase (FIX 4)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

    means = [d[d["state"] == s]["fwd"].mean() * 100 for s in (mk.STATE_BEAR, mk.STATE_SIDEWAYS, mk.STATE_BULL)]
    axes[1].bar(range(3), means, color=["#d62728", "#7f7f7f", "#2ca02c"], alpha=0.85)
    axes[1].axhline(allf.mean() * 100, color="black", ls="--", lw=1.2,
                    label=f"unconditional {allf.mean():.2%}")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["BEAR", "SIDEWAYS", "BULL"])
    axes[1].set_ylabel("mean forward 20-day return (%)")
    axes[1].set_title("Forward return by state")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png = OUT / "wti_regime.png"
    fig.savefig(png, dpi=130)
    print(f"\nChart saved: {png}")

    pd.DataFrame({
        "state": [mk.STATE_NAMES[s] for s in range(3)],
        "signal_phase_mean": pr["signal_mean"],
        "signal_phase_min": pr["signal_min"],
        "signal_phase_max": pr["signal_max"],
        "sign_stable": pr["sign_stable"],
        "stickiness_stride_phase0": [rep["stride"]["P"][s, s] for s in range(3)],
    }).to_csv(OUT / "wti_regime_summary.csv", index=False)
    print(f"Saved: {OUT / 'wti_regime_summary.csv'}")
    print("\nDESCRIPTIVE ONLY — no P&L, no tradeable claim. A tradeable rule needs a new "
          "registry entry in vault note 40 first.")


if __name__ == "__main__":
    main()
