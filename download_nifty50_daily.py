# =============================================================================
# download_nifty50_daily.py — daily history for all 50 NIFTY50 constituents
#
# Needed for Registry Entry #017 (cross-sectional validation). The nifty50
# folder on disk holds 1-min data from 2022 only; this builds the long daily
# series each name needs to contribute independent 20-day windows.
#
# Loops the Upstox v3 daily endpoint year-by-year (1 year per call) with the
# ANALYTICS TOKEN, read at runtime from the local downloader config — never
# stored here. Stocks that had not listed yet simply return empty years.
#
# Usage: python download_nifty50_daily.py [--from 2005-01-01] [--resume]
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
OUT_DIR = DOWNLOADER_DIR / "data" / "nifty50"

REQUEST_DELAY, BURST_EVERY, BURST_PAUSE = 0.5, 40, 5.0
MAX_RETRIES, RETRY_WAIT = 3, 10
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]


def load_config():
    spec = importlib.util.spec_from_file_location("dl_config", DOWNLOADER_DIR / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ANALYTICS_TOKEN, mod.NIFTY50_STOCKS


def fetch_year(token, instrument_key, from_date, to_date):
    key = urllib.parse.quote(instrument_key, safe="")
    url = f"https://api.upstox.com/v3/historical-candle/{key}/days/1/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json().get("data", {}).get("candles", [])
            if r.status_code == 400:
                return []          # pre-listing / invalid range for this year
            print(f"    HTTP {r.status_code} attempt {attempt}/{MAX_RETRIES}", flush=True)
        except requests.RequestException as e:
            print(f"    network error {e} attempt {attempt}/{MAX_RETRIES}", flush=True)
        time.sleep(RETRY_WAIT)
    return None                     # all retries failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2005-01-01")
    ap.add_argument("--resume", action="store_true",
                    help="skip symbols whose _daily.csv already exists")
    args = ap.parse_args()

    token, stocks = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_year, today = int(args.start[:4]), date.today()

    n_req = 0
    summary = []
    for i, stock in enumerate(stocks, 1):
        name, key = stock["name"], stock["instrument_key"]
        out = OUT_DIR / f"{name}_daily.csv"
        if args.resume and out.exists():
            print(f"[{i}/{len(stocks)}] {name}: exists, skipped", flush=True)
            continue

        frames, failed = [], 0
        for year in range(start_year, today.year + 1):
            f = f"{year}-01-01" if year > start_year else args.start
            t = f"{year}-12-31" if year < today.year else today.isoformat()
            candles = fetch_year(token, key, f, t)
            n_req += 1
            if candles is None:
                failed += 1
            elif candles:
                frames.append(pd.DataFrame(candles, columns=COLUMNS))
            time.sleep(BURST_PAUSE if n_req % BURST_EVERY == 0 else REQUEST_DELAY)

        if not frames:
            print(f"[{i}/{len(stocks)}] {name}: NO DATA ({failed} failed calls)", flush=True)
            summary.append({"symbol": name, "rows": 0, "start": None, "end": None})
            continue

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df.to_csv(out, index=False)
        lo, hi = df["timestamp"].min().date(), df["timestamp"].max().date()
        print(f"[{i}/{len(stocks)}] {name}: {len(df)} rows {lo} -> {hi}"
              + (f"  ({failed} failed calls)" if failed else ""), flush=True)
        summary.append({"symbol": name, "rows": len(df), "start": str(lo), "end": str(hi)})

    pd.DataFrame(summary).to_csv(Path(__file__).parent / "output" / "nifty50_daily_manifest.csv",
                                 index=False)
    ok = sum(1 for s in summary if s["rows"] > 0)
    print(f"\nDONE: {ok}/{len(summary)} symbols with data, {n_req} API calls.", flush=True)


if __name__ == "__main__":
    main()
