# =============================================================================
# prepare_wti.py — build a long WTI daily series, and VERIFY it tracks MCX
#
# MCX CrudeOil is cash-settled against NYMEX WTI, so WTI is a legitimate proxy
# for the long history MCX itself cannot provide (expired MCX contracts need
# Upstox Plus; reachable MCX history is ~4 months). Source: EIA, which
# publishes the daily front-month WTI futures settlement (RCLC1) back to 1983.
#
# The proxy claim is CHECKED here rather than assumed, against the 78 sessions
# of real MCX AUG-26 data already on disk. Two differences are expected and
# quantified: (1) MCX quotes INR, WTI quotes USD, so MCX returns carry a USDINR
# term; (2) MCX trades 09:00-23:30 IST while NYMEX settles ~23:59 IST, so a
# given calendar date is not the same information set.
#
# Usage: python prepare_wti.py
# =============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

WTI_DIR = Path(r"D:\MyPython\Download_1min_History\data\wti")
CRUDE_DIR = Path(r"D:\MyPython\Download_1min_History\data\crudeoil")
OUT_CSV = WTI_DIR / "WTI_FRONT_daily.csv"


def load_eia_xls(path: Path) -> pd.Series:
    """EIA .xls: sheet 'Data 1', two header rows, then Date | value."""
    df = pd.read_excel(path, sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.Series(pd.to_numeric(df["price"], errors="coerce").to_numpy(),
                  index=df["date"]).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def main():
    fut = load_eia_xls(WTI_DIR / "RCLC1d.xls")
    spot = load_eia_xls(WTI_DIR / "RWTCd.xls")
    print(f"WTI front-month futures (RCLC1): {len(fut):,} days, "
          f"{fut.index[0].date()} -> {fut.index[-1].date()}")
    print(f"WTI Cushing spot        (RWTC) : {len(spot):,} days, "
          f"{spot.index[0].date()} -> {spot.index[-1].date()}")

    # ── EIA discontinued the futures series; spot is the only current one ──
    print(f"\nEIA stopped publishing the futures series (RCLC1) after "
          f"{fut.index[-1].date()}. The spot series is current, so SPOT is the base "
          f"series and futures is used only to validate that choice.")
    both = pd.DataFrame({"fut": fut, "spot": spot}).dropna()
    ok = both[(both > 0).all(axis=1)]
    r1 = ok.pct_change().dropna()
    r20 = (ok / ok.shift(20) - 1).dropna()
    print(f"  spot vs front-month futures over {len(ok):,} shared days "
          f"({ok.index[0].date()} -> {ok.index[-1].date()}):")
    print(f"    daily-return correlation   {r1['fut'].corr(r1['spot']):.4f}")
    print(f"    20-day-return correlation  {r20['fut'].corr(r20['spot']):.4f}")
    print(f"    mean absolute basis        ${(ok['fut'] - ok['spot']).abs().mean():.3f}")

    series = spot

    # ── negative prices break every return calculation ──
    neg = series[series <= 0]
    print(f"\nNon-positive prices in the base (spot) series: {len(neg)}")
    for d, v in neg.items():
        print(f"  {d.date()}: ${v:.2f}")
    if len(neg):
        print("  -> percentage returns are undefined across these dates; windows spanning")
        print("     them must be dropped rather than silently producing nonsense.")

    # ── verify the MCX sync claim on real overlapping data ──
    mcx_files = sorted(CRUDE_DIR.glob("CRUDEOIL_*_1min.csv"))
    print(f"\nVerifying the WTI<->MCX proxy claim on {len(mcx_files)} MCX contracts:")
    print(f"  {'contract':<10}{'sessions':>9}{'daily ret corr':>16}{'20d ret corr':>14}")
    for f in mcx_files:
        label = f.name.split("_")[1]
        m = pd.read_csv(f, parse_dates=["timestamp"])
        ts = pd.to_datetime(m["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        mc = pd.Series(m["close"].to_numpy(dtype=float), index=ts).sort_index()
        mday = mc.resample("1D").last().dropna()
        joined = pd.DataFrame({"mcx": mday, "wti": series}).dropna()
        if len(joined) < 30:
            print(f"  {label:<10}{len(joined):>9}   too few overlapping days")
            continue
        r1 = joined.pct_change().dropna()
        r20 = (joined / joined.shift(20) - 1).dropna()
        print(f"  {label:<10}{len(joined):>9}{r1['mcx'].corr(r1['wti']):>16.3f}"
              f"{(r20['mcx'].corr(r20['wti']) if len(r20) > 5 else float('nan')):>14.3f}")

    series.rename("close").to_frame().to_csv(OUT_CSV, index_label="date")
    print(f"\nSaved: {OUT_CSV}  ({len(series):,} rows, "
          f"{series.index[0].date()} -> {series.index[-1].date()})")


if __name__ == "__main__":
    main()
