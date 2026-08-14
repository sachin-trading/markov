# =============================================================================
# run_demo.py — Markov 2.0 proof run: NIFTY 50 daily, walk-forward
#
# 1. Load the longest NIFTY daily history on disk
# 2. FIX 2 gate: verify state labels against known periods (abort on failure)
# 3. Calibration check: state distribution; offer percentile thresholds if skewed
# 4. FIX 1: show overlapping (legacy) vs stride-sampled (true) matrices side by side
# 5. Matrix-power forecasts -> convergence to the stationary distribution
# 6. Walk-forward backtest, before-fix (overlap) vs after-fix (stride),
#    STANDALONE differential on NIFTY futures, 1 lot, Indian F&O costs
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

DATA = Path(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05
SIGNAL_THR = 0.10
LOT_SIZE, LOTS, SLIPPAGE_PTS = 65, 1, 1.0
MIN_TRAIN = 1250  # ~5 trading years before the first out-of-sample decision


def load_close() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    s = pd.Series(df["close"].to_numpy(dtype=float), index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def main():
    close = load_close()
    print(f"NIFTY daily: {len(close)} bars, {close.index[0].date()} -> {close.index[-1].date()}")

    # ── FIX 2 gate ──
    checks = mk.assert_labels_verified(close, WINDOW, BULL_THR, BEAR_THR)
    print("\nLabel verification (FIX 2):")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    states = mk.label_states(close, WINDOW, BULL_THR, BEAR_THR)

    # ── calibration ──
    dist = mk.state_distribution(states)
    print("\nState distribution (±5% / 20d thresholds):")
    for k, v in dist.items():
        print(f"  {k:<9} {v:.1%}")
    p_bull, p_bear = mk.percentile_thresholds(close, WINDOW, 0.25)
    if min(dist.values()) < 0.10:
        print(f"  ⚠ a state is under 10% — percentile alternative (25/75): "
              f"bull >= {p_bull:+.2%}, bear <= {p_bear:+.2%}")
    else:
        print(f"  (calibration OK; percentile 25/75 alternative would be "
              f"bull >= {p_bull:+.2%}, bear <= {p_bear:+.2%})")

    # ── FIX 1: both matrices ──
    rep = mk.matrix_report(states, WINDOW)
    print(f"\nOVERLAPPING matrix (legacy — windows share {WINDOW-1} of {WINDOW} days; "
          f"diagonal persistence is an artifact, NOT statistically honest):")
    print(mk.format_matrix(rep["overlap"]["P"], rep["overlap"]["counts"]))
    print(f"  transitions: {rep['overlap']['n_transitions']}")
    print(f"\nSTRIDE-SAMPLED matrix (true — non-overlapping {WINDOW}-day windows; "
          f"the only statistically honest one):")
    print(mk.format_matrix(rep["stride"]["P"], rep["stride"]["counts"]))
    print(f"  transitions: {rep['stride']['n_transitions']} "
          f"(~{rep['stride']['n_transitions'] / (len(states)/250):.1f}/year — small-sample cost of honesty)")
    if rep["stride"]["unreliable"]:
        print(f"  ⚠ cells with <{mk.MIN_CELL_OBS} obs (unreliable): "
              + ", ".join(rep["stride"]["unreliable"]))

    print("\nStickiness (diagonal):")
    print(f"  overlap: { {k: round(v,3) for k,v in mk.stickiness(rep['overlap']['P']).items()} }")
    print(f"  stride : { {k: round(v,3) for k,v in mk.stickiness(rep['stride']['P']).items()} }")

    # ── forecasts & convergence ──
    P = rep["stride"]["P"]
    cur = int(states.iloc[-1])
    print(f"\nCurrent state: {mk.STATE_NAMES[cur]}  "
          f"(signal = P(bull)-P(bear) = {mk.signal(P, cur):+.3f} per {WINDOW}-day step)")
    stat = mk.stationary_distribution(P)
    print("Forecast convergence (stride matrix, k steps of 20 trading days):")
    for k in (1, 2, 3, 5, 10):
        f = mk.forecast(P, cur, k)
        print(f"  k={k:>2}: " + "  ".join(f"{mk.STATE_NAMES[i]} {f[i]:.1%}" for i in range(3)))
    print("  stationary: " + "  ".join(f"{mk.STATE_NAMES[i]} {stat[i]:.1%}" for i in range(3))
          + "   <- long-horizon forecasts converge here and carry no signal")

    # ── walk-forward: before-fix vs after-fix ──
    results = {}
    for mode in ("overlap", "stride"):
        results[mode] = mk.walk_forward(
            close, WINDOW, BULL_THR, BEAR_THR, matrix_mode=mode,
            min_train_bars=MIN_TRAIN, signal_threshold=SIGNAL_THR,
            lot_size=LOT_SIZE, lots=LOTS, slippage_pts=SLIPPAGE_PTS)

    print(f"\nWalk-forward (expanding window, first {MIN_TRAIN} bars train-only, "
          f"matrix rebuilt as it walks, costs: Upstox F&O + {SLIPPAGE_PTS} pt slippage/leg, "
          f"1 lot = {LOT_SIZE}):")
    hdr = f"{'':<22}{'BEFORE fix (overlap)':>22}{'AFTER fix (stride)':>22}"
    print(hdr)
    rows = [
        ("test window", lambda r: f"{r['test_start']}->{r['test_end']}"),
        ("trades", lambda r: f"{r['n_trades']}"),
        ("win rate", lambda r: f"{r['win_rate']:.1%}"),
        ("profit factor", lambda r: f"{r['profit_factor']:.2f}"),
        ("total return", lambda r: f"{r['total_return']:+.1%}"),
        ("CAGR", lambda r: f"{r['cagr']:+.2%}"),
        ("max drawdown", lambda r: f"{r['max_drawdown']:.1%}"),
        ("total costs ₹", lambda r: f"{r['total_costs']:,.0f}"),
        ("net P&L ₹", lambda r: f"{r['net_pnl']:+,.0f}"),
    ]
    for name, fn in rows:
        print(f"{name:<22}{fn(results['overlap']):>22}{fn(results['stride']):>22}")
    r = results["stride"]
    print(f"{'buy & hold return':<22}{r['buyhold_return']:>+21.1%} (same window)   "
          f"maxDD {r['buyhold_maxdd']:.1%}")

    # ── equity curve ──
    fig, ax = plt.subplots(figsize=(11, 6))
    for mode, color, label in (("overlap", "#d62728", "BEFORE fix — overlapping matrix (flawed)"),
                               ("stride", "#1f77b4", "AFTER fix — stride-sampled matrix (honest)")):
        eq = results[mode]["equity"]
        ax.plot(eq.index, eq / eq.iloc[0], color=color, label=label, lw=1.4)
    bh = close.loc[results["stride"]["equity"].index]
    ax.plot(bh.index, bh / bh.iloc[0], color="#7f7f7f", ls="--", lw=1.1,
            label="NIFTY buy & hold")
    ax.set_title("Markov 2.0 walk-forward — NIFTY 50 daily, standalone differential, "
                 "1 lot futures, net of costs")
    ax.set_ylabel("Growth of initial capital (×)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / "markov2_nifty_walkforward.png"
    fig.savefig(png, dpi=130)
    print(f"\nEquity curve saved: {png}")

    # persist metrics for the report
    summary = pd.DataFrame({m: {k: v for k, v in res.items()
                                if not isinstance(v, (pd.Series,))}
                            for m, res in results.items()})
    summary.to_csv(OUT / "walkforward_summary.csv")
    print(f"Summary saved: {OUT / 'walkforward_summary.csv'}")


if __name__ == "__main__":
    main()
