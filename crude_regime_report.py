# =============================================================================
# crude_regime_report.py — Markov 2.0 applied to MCX CrudeOil
#
# DESCRIPTIVE ONLY. No P&L, no strategy, no pass/fail — this is the regime
# monitor use that vault note 34 §S15 retained the method for after the FILTER
# line was closed. A tradeable claim on this instrument would need a new
# registry entry in note 40 first, and (see the sample warnings printed below)
# the data cannot currently support one.
#
# Why not the daily horizon: expired MCX contracts require Upstox Plus
# (this token has isPlusPlan=false), so reachable history is only what the
# listed contracts carry — about 4 months. At a 20-DAY window that is ~4
# non-overlapping transitions for a 9-cell matrix. Unusable. This report
# therefore uses a 20-BAR window on intraday bars, which is a different
# horizon answering a different question, and says so.
#
# Single contract, never a stitched continuous series: rolling injects a basis
# jump at each roll that is indistinguishable from a real return.
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import markov_regime as mk

DATA = Path(r"D:\MyPython\Download_1min_History\data\crudeoil")
NIFTY_CSV = Path(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

CONTRACT = "AUG26"          # deepest clean contract: 78 sessions
BAR = "15min"
WINDOW = 20                 # 20 bars = 5 hours ~= 1/3 of an MCX session
PCT = 0.25                  # percentile calibration (top/bottom quartile)
SESSION_START, SESSION_END = "09:00", "23:30"


def load_contract(label: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"CRUDEOIL_{label}_1min.csv", parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.set_index(ts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.between_time(SESSION_START, SESSION_END)


def main():
    raw = load_contract(CONTRACT)
    bars = raw["close"].resample(BAR).last().dropna()
    sessions = bars.index.normalize()
    print(f"MCX CrudeOil {CONTRACT}, {BAR} bars from 1-min "
          f"(MCX evening session {SESSION_START}-{SESSION_END} IST)")
    print(f"  {len(bars):,} bars over {sessions.nunique()} sessions, "
          f"{bars.index[0].date()} -> {bars.index[-1].date()}")

    # ── calibration: ±5% was tuned on SPY/NIFTY DAILY windows ──
    ret = mk.window_returns(bars, WINDOW)
    nifty = pd.read_csv(NIFTY_CSV, parse_dates=["timestamp"])
    nret = mk.window_returns(pd.Series(nifty["close"].to_numpy(dtype=float)), WINDOW)
    print(f"\nCalibration — a {WINDOW}-bar window here is {WINDOW*15/60:.0f} hours of crude, "
          f"not 20 trading days:")
    print(f"  crude {WINDOW}-bar return: sd {ret.std():.2%}, "
          f"range {ret.min():+.1%} to {ret.max():+.1%}")
    print(f"  NIFTY {WINDOW}-day return: sd {nret.std():.2%}, "
          f"range {nret.min():+.1%} to {nret.max():+.1%}")
    base = mk.label_states(bars, WINDOW, 0.05, -0.05)
    d0 = mk.state_distribution(base)
    print(f"  with the default ±5% thresholds: "
          + ", ".join(f"{k} {v:.1%}" for k, v in d0.items()))
    if min(d0.values()) < 0.10:
        print(f"  -> a state is under 10% of windows, so the percentile fallback applies "
              f"(Entry #014 permits this exactly when a state is starved).")

    bull_thr, bear_thr = mk.percentile_thresholds(bars, WINDOW, PCT)
    print(f"  percentile calibration (top/bottom {PCT:.0%}): "
          f"BULL >= {bull_thr:+.2%}, BEAR <= {bear_thr:+.2%}")

    # ── FIX 2, instrument-agnostic (the NIFTY anchors do not apply here) ──
    checks = mk.assert_labels_verified_generic(bars, WINDOW, bull_thr, bear_thr)
    print("\nLabel verification (FIX 2, generic anchors):")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    states = mk.label_states(bars, WINDOW, bull_thr, bear_thr)
    print("\nState distribution: "
          + ", ".join(f"{k} {v:.1%}" for k, v in mk.state_distribution(states).items()))

    # ── FIX 1: both matrices ──
    rep = mk.matrix_report(states, WINDOW)
    print(f"\nOVERLAPPING matrix (legacy — consecutive windows share {WINDOW-1} of {WINDOW} "
          f"bars; the diagonal is an artifact, NOT statistically honest):")
    print(mk.format_matrix(rep["overlap"]["P"], rep["overlap"]["counts"]))
    print(f"  transitions: {rep['overlap']['n_transitions']:,}")

    print(f"\nSTRIDE-SAMPLED matrix (true — non-overlapping {WINDOW}-bar windows):")
    print(mk.format_matrix(rep["stride"]["P"], rep["stride"]["counts"]))
    n_str = rep["stride"]["n_transitions"]
    print(f"  transitions: {n_str}")
    if rep["stride"]["unreliable"]:
        print(f"  ⚠ cells with <{mk.MIN_CELL_OBS} obs: " + ", ".join(rep["stride"]["unreliable"]))

    print("\nStickiness (diagonal):")
    print(f"  overlap: { {k: round(v,3) for k,v in mk.stickiness(rep['overlap']['P']).items()} }")
    print(f"  stride : { {k: round(v,3) for k,v in mk.stickiness(rep['stride']['P']).items()} }")

    # ── the independence question, applied BEFORE quoting the sample as evidence ──
    sampled = states.iloc[::WINDOW]
    n_sess = sampled.index.normalize().nunique()
    per_sess = len(sampled) / max(n_sess, 1)
    print(f"\n⚠ INDEPENDENCE CHECK — the number that matters is not {n_str}.")
    print(f"  Those {n_str} stride transitions sit inside just {n_sess} trading sessions "
          f"({per_sess:.1f} windows per session).")
    print(f"  Windows within one session share the same news, the same intraday trend and the")
    print(f"  same participants — they are not independent draws. The honest independent unit")
    print(f"  is the SESSION, so effective n is about {n_sess}, and all {n_sess} sit inside a")
    print(f"  single {(bars.index[-1]-bars.index[0]).days}-day stretch of one contract — one")
    print(f"  market environment, not a cross-section of them. For scale: NIFTY gave 266 stride")
    print(f"  transitions over 21 YEARS and that still proved too thin to validate an effect.")

    # ── signal and forecasts, reported as description only ──
    P = rep["stride"]["P"]
    cur = int(states.iloc[-1])
    print(f"\nCurrent state: {mk.STATE_NAMES[cur]}   "
          f"signal P(bull)-P(bear) = {mk.signal(P, cur):+.3f} per {WINDOW}-bar step")
    print("Signal from each state (check both gate legs are reachable before ever "
          "proposing a two-sided rule):")
    for s in range(mk.N_STATES):
        print(f"  {mk.STATE_NAMES[s]:<9} {mk.signal(P, s):+.3f}")
    sig = [mk.signal(P, s) for s in range(mk.N_STATES)]
    print(f"  range {min(sig):+.3f} to {max(sig):+.3f} — "
          f"{'BOTH legs reachable at ±0.10' if min(sig) < -0.10 < 0.10 < max(sig) else 'a ±0.10 two-sided gate would NOT fire on both sides'}")

    stat = mk.stationary_distribution(P)
    print("\nForecast convergence (stride matrix, k steps of 20 bars):")
    for k in (1, 2, 3, 5, 10):
        f = mk.forecast(P, cur, k)
        print(f"  k={k:>2}: " + "  ".join(f"{mk.STATE_NAMES[i]} {f[i]:.1%}" for i in range(3)))
    print("  stationary: " + "  ".join(f"{mk.STATE_NAMES[i]} {stat[i]:.1%}" for i in range(3))
          + "   <- forecasts converge here and carry no signal")

    pd.DataFrame({
        "state": [mk.STATE_NAMES[s] for s in range(3)],
        "stickiness_stride": [P[s, s] for s in range(3)],
        "stickiness_overlap": [rep["overlap"]["P"][s, s] for s in range(3)],
        "signal": sig,
        "stationary": stat,
    }).to_csv(OUT / "crude_regime_summary.csv", index=False)
    print(f"\nSaved: {OUT / 'crude_regime_summary.csv'}")
    print("\nDESCRIPTIVE ONLY — no P&L simulated, no tradeable claim made. "
          "See vault note 34 §S15.")


if __name__ == "__main__":
    main()
