# =============================================================================
# download_nifty_daily.py — Build a long NIFTY 50 daily series via Upstox v3
#
# Daily endpoint returns at most 1 year per call, so we loop year-by-year.
# Uses the long-lived ANALYTICS TOKEN only (read-only; never the Trading token).
# Token is imported at runtime from the existing downloader config — it is
# NEVER stored in this repo.
#
# Usage:  python download_nifty_daily.py [--from 2005-01-01] [--instrument "NSE_INDEX|Nifty 50"] [--out path.csv]
# =============================================================================

import argparse
import importlib.util
import sys
import time
import urllib.parse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DOWNLOADER_DIR = Path(r"D:\MyPython\Download_1min_History")
DEFAULT_OUT = DOWNLOADER_DIR / "data" / "nifty" / "NIFTY_daily.csv"

# Rate limits per the platform downloader config
REQUEST_DELAY, BURST_EVERY, BURST_PAUSE = 0.5, 40, 5.0
MAX_RETRIES, RETRY_WAIT = 3, 10

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]


def load_analytics_token() -> str:
    spec = importlib.util.spec_from_file_location("dl_config", DOWNLOADER_DIR / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ANALYTICS_TOKEN


def fetch_year(token: str, instrument_key: str, from_date: str, to_date: str) -> list:
    key = urllib.parse.quote(instrument_key, safe="")
    url = f"https://api.upstox.com/v3/historical-candle/{key}/days/1/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json().get("data", {}).get("candles", [])
            if r.status_code == 400:
                print(f"  400 for {from_date}->{to_date}: {r.text[:150]}")
                return []
            print(f"  HTTP {r.status_code}, retry {attempt}/{MAX_RETRIES}")
        except requests.RequestException as e:
            print(f"  network error: {e}, retry {attempt}/{MAX_RETRIES}")
        time.sleep(RETRY_WAIT)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2005-01-01")
    ap.add_argument("--instrument", default="NSE_INDEX|Nifty 50")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    token = load_analytics_token()
    start_year = int(args.start[:4])
    today = date.today()

    frames = []
    n_req = 0
    for year in range(start_year, today.year + 1):
        f = f"{year}-01-01" if year > start_year else args.start
        t = f"{year}-12-31" if year < today.year else today.isoformat()
        candles = fetch_year(token, args.instrument, f, t)
        n_req += 1
        print(f"  {f} -> {t}: {len(candles)} candles")
        if candles:
            frames.append(pd.DataFrame(candles, columns=COLUMNS))
        time.sleep(BURST_PAUSE if n_req % BURST_EVERY == 0 else REQUEST_DELAY)

    if not frames:
        print("No data received — nothing saved.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            .reset_index(drop=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = pd.read_csv(out, parse_dates=["timestamp"])
        df = (pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=["timestamp"]).sort_values("timestamp"))
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
