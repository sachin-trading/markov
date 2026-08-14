# =============================================================================
# cross_sectional_bear.py — Registry Entry #017
#
# Does the BEAR-window forward-return effect exist across Indian equities, or
# is it an artifact of a few NIFTY index episodes (2008, 2011, 2020)?
#
# Method, exactly as pre-registered:
#   - label each NIFTY50 constituent's daily history with the Entry #014 states
#   - sample on a NON-OVERLAPPING grid: every 20th trading day of a common NSE
#     calendar, so each sample date is an independent block and every stock is
#     observed on the same dates
#   - per-stock edge  = mean fwd-20d return | BEAR  -  that stock's own uncond. mean
#   - pooled edge     = same difference across all stocks and sample dates
#   - block bootstrap resamples SAMPLE DATES with replacement (never stocks),
#     so correlated names are never counted as independent evidence
#
# Pass criteria (fixed before running):
#   (1) >= 35 qualifying stocks with a positive BEAR edge
#   (2) pooled edge survives the date-block bootstrap at p < 0.05, one-sided
#   Failing either => the effect is an index artifact, FILTER work is CLOSED.
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

DATA_DIR = Path(r"D:\MyPython\Download_1min_History\data\nifty50")
NIFTY_CSV = Path(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

WINDOW, BULL_THR, BEAR_THR = 20, 0.05, -0.05
MIN_YEARS = 10
N_BOOT = 10000
RNG = np.random.default_rng(20260814)


def load_daily(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    s = pd.Series(df["close"].to_numpy(dtype=float), index=ts.dt.normalize()).sort_index()
    return s[~s.index.duplicated(keep="last")]


def main():
    # ── common calendar and non-overlapping sample dates (from NIFTY) ──
    nifty = load_daily(NIFTY_CSV)
    calendar = nifty.index
    sample_dates = calendar[::WINDOW]
    print(f"Common NSE calendar: {len(calendar)} trading days "
          f"{calendar[0].date()} -> {calendar[-1].date()}")
    print(f"Non-overlapping sample dates (every {WINDOW}th): {len(sample_dates)}")

    files = sorted(DATA_DIR.glob("*_daily.csv"))
    print(f"Stock files found: {len(files)}")

    rows = []            # long panel: one row per (stock, sample date)
    per_stock = []
    excluded = []
    for f in files:
        sym = f.name.replace("_daily.csv", "")
        close = load_daily(f)
        years = (close.index[-1] - close.index[0]).days / 365.25
        if years < MIN_YEARS:
            excluded.append((sym, round(years, 1)))
            continue

        states = mk.label_states(close, WINDOW, BULL_THR, BEAR_THR)
        fwd = (close.shift(-WINDOW) / close - 1.0)

        # observe only on the shared sample dates the stock actually traded
        idx = states.index.intersection(sample_dates).intersection(fwd.dropna().index)
        if len(idx) < 20:
            excluded.append((sym, round(years, 1)))
            continue

        st = states.loc[idx].to_numpy()
        fw = fwd.loc[idx].to_numpy(dtype=float)
        rows.append(pd.DataFrame({"date": idx, "symbol": sym, "state": st, "fwd": fw}))

        bear = fw[st == mk.STATE_BEAR]
        per_stock.append({
            "symbol": sym, "obs": len(fw), "bear_obs": int(len(bear)),
            "bear_mean": float(bear.mean()) if len(bear) else np.nan,
            "uncond_mean": float(fw.mean()),
            "edge": float(bear.mean() - fw.mean()) if len(bear) else np.nan,
            "years": round(years, 1),
        })

    panel = pd.concat(rows, ignore_index=True)
    ps = pd.DataFrame(per_stock)
    qualifying = ps.dropna(subset=["edge"])

    print(f"\nQualifying stocks: {len(qualifying)} "
          f"(excluded {len(excluded)} with <{MIN_YEARS}y or too few samples)")
    if excluded:
        print("  excluded: " + ", ".join(f"{s} ({y}y)" for s, y in excluded))
    print(f"Panel: {len(panel):,} independent (stock, date) observations across "
          f"{panel['date'].nunique()} sample dates")

    # ── condition 1: how many stocks show a positive edge ──
    n_pos = int((qualifying["edge"] > 0).sum())
    n_qual = len(qualifying)
    print(f"\nCondition 1 — stocks with a positive BEAR edge: {n_pos}/{n_qual} "
          f"(need >= 35)  -> {'PASS' if n_pos >= 35 else 'FAIL'}")

    print("\n  Strongest 5:")
    for _, r in qualifying.nlargest(5, "edge").iterrows():
        print(f"    {r['symbol']:<12} edge {r['edge']:+.2%}  "
              f"(bear {r['bear_mean']:+.2%} vs uncond {r['uncond_mean']:+.2%}, "
              f"{int(r['bear_obs'])} bear obs)")
    print("  Weakest 5:")
    for _, r in qualifying.nsmallest(5, "edge").iterrows():
        print(f"    {r['symbol']:<12} edge {r['edge']:+.2%}  "
              f"(bear {r['bear_mean']:+.2%} vs uncond {r['uncond_mean']:+.2%}, "
              f"{int(r['bear_obs'])} bear obs)")

    # ── condition 2: date-block bootstrap on the pooled edge ──
    bear_mask = panel["state"].to_numpy() == mk.STATE_BEAR
    fwd_all = panel["fwd"].to_numpy(dtype=float)
    pooled_edge = fwd_all[bear_mask].mean() - fwd_all.mean()
    print(f"\nPooled edge: BEAR {fwd_all[bear_mask].mean():+.2%} vs unconditional "
          f"{fwd_all.mean():+.2%}  ->  {pooled_edge:+.2%} "
          f"({bear_mask.sum():,} bear observations)")

    # group observations by sample date; resample WHOLE DATES with replacement
    date_codes, uniq_dates = pd.factorize(panel["date"], sort=True)
    by_date = [np.where(date_codes == k)[0] for k in range(len(uniq_dates))]
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(by_date), len(by_date))
        idx = np.concatenate([by_date[k] for k in pick])
        f_b, m_b = fwd_all[idx], bear_mask[idx]
        boot[b] = (f_b[m_b].mean() - f_b.mean()) if m_b.any() else np.nan
    boot = boot[~np.isnan(boot)]
    p = float((boot <= 0).mean())

    print(f"\nCondition 2 — date-block bootstrap ({len(boot):,} replicates, "
          f"resampling {len(by_date)} sample dates, never stocks):")
    print(f"  pooled edge {pooled_edge:+.2%}   "
          f"95% CI [{np.percentile(boot,2.5):+.2%}, {np.percentile(boot,97.5):+.2%}]")
    print(f"  one-sided p = {p:.4f}  -> {'PASS' if p < 0.05 else 'FAIL'} (need p < 0.05)")

    # ── verdict ──
    c1, c2 = n_pos >= 35, p < 0.05
    print("\n" + "=" * 62)
    if c1 and c2:
        print("VERDICT: PASS — the BEAR-window effect replicates across Indian")
        print("equities and survives correlation-aware resampling.")
    else:
        print("VERDICT: FAIL — pre-registered criteria not met "
              f"(condition 1 {'PASS' if c1 else 'FAIL'}, condition 2 {'PASS' if c2 else 'FAIL'}).")
        print("Per Entry #017 this closes the FILTER line of work: the effect the")
        print("Entry #015/#016 gates relied on is an index artifact, not a property")
        print("of Indian equities.")
    print("=" * 62)

    # ── chart ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    q = qualifying.sort_values("edge")
    colors = ["#d62728" if e < 0 else "#2ca02c" for e in q["edge"]]
    axes[0].barh(q["symbol"], q["edge"] * 100, color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title(f"Per-stock BEAR edge ({n_pos}/{n_qual} positive; need 35)")
    axes[0].set_xlabel("mean fwd-20d return in BEAR minus unconditional (%)")
    axes[0].tick_params(axis="y", labelsize=6)

    axes[1].hist(boot * 100, bins=60, color="#1f77b4", alpha=0.85)
    axes[1].axvline(0, color="black", lw=1.2, label="no effect")
    axes[1].axvline(pooled_edge * 100, color="#d62728", lw=1.6,
                    label=f"observed {pooled_edge:+.2%}")
    axes[1].set_title(f"Date-block bootstrap of pooled edge (p = {p:.4f})")
    axes[1].set_xlabel("pooled BEAR edge (%)")
    axes[1].legend()
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / "entry017_cross_sectional_bear.png"
    fig.savefig(png, dpi=130)
    print(f"\nChart saved: {png}")

    ps.to_csv(OUT / "entry017_per_stock.csv", index=False)
    pd.DataFrame({"metric": ["pooled_edge", "p_value", "n_positive", "n_qualifying",
                             "bear_obs", "total_obs", "sample_dates"],
                  "value": [pooled_edge, p, n_pos, n_qual, int(bear_mask.sum()),
                            len(panel), len(by_date)]}).to_csv(
        OUT / "entry017_summary.csv", index=False)
    print(f"Saved: {OUT / 'entry017_per_stock.csv'}, {OUT / 'entry017_summary.csv'}")


if __name__ == "__main__":
    main()
