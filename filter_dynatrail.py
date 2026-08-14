# =============================================================================
# filter_dynatrail.py — Markov 2.0 FILTER mode: gate DynaTrail with the regime signal
#
# The user's strategy stays theirs — DynaTrail (Supertrend flip -> ATM weekly
# option, pinned production config ST10/2.5/atr25/psl40%/hold12) decides the
# trades; Markov 2.0 decides WHEN it is allowed to act:
#   CE entries only on days where signal > +THR
#   PE entries only on days where signal < -THR
#   no entries in chop (|signal| <= THR)
#
# The gating signal for day D is the walk-forward stride-matrix signal computed
# at the close of the last trading day BEFORE D (no lookahead), from 21 years
# of NIFTY daily. Both arms run the identical engine, window, config, costs
# (₹50/order + 1 pt/leg premium slippage, VIX-proxy IV where available) — the
# gate is the only difference.
# =============================================================================

import sys
from bisect import bisect_left
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, r"D:\MyPython\SachinJ_Algo\backtester")

import markov_regime as mk
from data_loader import load_nifty
from strategies.dynatrail_nifty_option import DynaTrailNiftyOption, print_result

DAILY_CSV = Path(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv")
VIX_CSV = Path(r"D:\MyPython\Download_1min_History\data\vix\INDIAVIX_daily.csv")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05
SIGNAL_THR = 0.10
MIN_TRAIN = 1250

# Identical to the DynaTrail walk-forward validation cost policy
PRODUCTION_CONFIG = {
    "st_period": 10, "st_mult": 2.5, "atr_min": 25,
    "premium_sl_pct": 0.40, "max_hold_bars": 12,
    "capital": 500_000, "cost_per_order": 50.0, "slippage_pts_per_leg": 1.0,
}

START, END = "2022-01-03", "2026-07-24"  # full NIFTY index 1-min coverage on disk


def load_daily_close() -> pd.Series:
    df = pd.read_csv(DAILY_CSV, parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    s = pd.Series(df["close"].to_numpy(dtype=float), index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def build_gate(signals: pd.Series, session_dates, thr: float) -> dict:
    """gate[date] = allowed sides for that day, from the last signal strictly
    before the date (previous trading day's close — no lookahead)."""
    sig = signals.dropna()
    sig_dates = [ts.date() for ts in sig.index]
    sig_vals = sig.to_numpy()
    gate = {}
    for d in session_dates:
        i = bisect_left(sig_dates, d) - 1  # latest signal date strictly < d
        if i < 0:
            gate[d] = set()  # no signal yet -> strategy may not act
            continue
        s = float(sig_vals[i])
        allowed = set()
        if s > thr:
            allowed.add("CE")
        if s < -thr:
            allowed.add("PE")
        gate[d] = allowed
    return gate


def main():
    # ── Markov walk-forward signal from 21y of daily (FIX 2 gate first) ──
    close_d = load_daily_close()
    mk.assert_labels_verified(close_d, WINDOW, BULL_THR, BEAR_THR)
    wf = mk.walk_forward(close_d, WINDOW, BULL_THR, BEAR_THR, matrix_mode="stride",
                         min_train_bars=MIN_TRAIN, signal_threshold=SIGNAL_THR,
                         apply_costs=False)
    signals = wf["signals"]
    print(f"Markov signal (stride, walk-forward): {signals.dropna().index[0].date()} "
          f"-> {signals.dropna().index[-1].date()}")

    # ── DynaTrail data: NIFTY index 1-min ──
    df = load_nifty("1min", start=START, end=END)
    print(f"NIFTY 1-min: {len(df):,} bars, {df.index[0]} -> {df.index[-1]}")

    vix = pd.read_csv(VIX_CSV)
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    vix_map = dict(zip(vix["date"], vix["close"]))
    print(f"VIX-proxy IV: {min(vix_map)} -> {max(vix_map)} "
          f"({len(vix_map)} days; earlier bars fall back to fixed iv=0.13 in BOTH arms)")

    session_dates = sorted(set(df.index.date))
    gate = build_gate(signals, session_dates, SIGNAL_THR)
    n_ce = sum(1 for a in gate.values() if "CE" in a)
    n_pe = sum(1 for a in gate.values() if "PE" in a)
    n_block = sum(1 for a in gate.values() if not a)
    print(f"Gate exposure over {len(gate)} sessions: CE allowed {n_ce} "
          f"({n_ce/len(gate):.0%}), PE allowed {n_pe} ({n_pe/len(gate):.0%}), "
          f"fully blocked {n_block} ({n_block/len(gate):.0%})")

    # ── run both arms: identical everything, gate is the only difference ──
    base_cfg = {**PRODUCTION_CONFIG, "vix_by_date": vix_map, "vix_scale": 1.0}
    results = {}
    for label, extra in (("UNGATED", {}), ("GATED", {"side_gate_by_date": gate})):
        res = DynaTrailNiftyOption(df, {**base_cfg, **extra}).run()
        results[label] = res
        print_result(res, f"DynaTrail {label} — pinned production config, {START}->{END}")

    # ── comparison table ──
    mu, mg = results["UNGATED"].summary(), results["GATED"].summary()
    print("\nFILTER-mode comparison (identical engine/config/costs; gate only):")
    print(f"{'':<18}{'UNGATED':>14}{'GATED':>14}")
    for k, fmt in (("num_trades", "{:d}"), ("win_rate", "{:.1%}"),
                   ("profit_factor", "{:.3f}"), ("net_pnl", "₹{:,.0f}"),
                   ("max_drawdown", "{:.1%}"), ("sharpe_ratio", "{:.2f}"),
                   ("long_trades", "{:d}"), ("short_trades", "{:d}")):
        print(f"{k:<18}{fmt.format(mu[k]):>14}{fmt.format(mg[k]):>14}")

    # ── equity curves ──
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, color in (("UNGATED", "#7f7f7f"), ("GATED", "#1f77b4")):
        eq = results[label].equity
        ax.plot(eq.index, eq, color=color, lw=1.4,
                label=f"DynaTrail {label}" + (" (Markov 2.0 filter)" if label == "GATED" else ""))
    ax.set_title(f"Markov 2.0 FILTER mode — DynaTrail on NIFTY, {START} -> {END}, "
                 "net of costs (₹50/order + 1pt/leg)")
    ax.set_ylabel("Equity (₹, capital 500,000)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / "markov2_filter_dynatrail.png"
    fig.savefig(png, dpi=130)
    print(f"\nEquity curves saved: {png}")

    pd.DataFrame({"UNGATED": mu, "GATED": mg}).to_csv(OUT / "filter_dynatrail_summary.csv")
    print(f"Summary saved: {OUT / 'filter_dynatrail_summary.csv'}")


if __name__ == "__main__":
    main()
