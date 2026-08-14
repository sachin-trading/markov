# =============================================================================
# download_crude_1min.py — 1-min history for active MCX CrudeOil contracts
#
# MCX keys are NUMERIC (MCX_FO|560977); name-based keys throw UDAPI100011.
# Expired contracts need Upstox Plus (this token has isPlusPlan=false), so the
# reachable history is only what the currently-listed contracts carry: roughly
# 4 months of 1-min data per contract. The 1-min endpoint serves ~1 month per
# call, so each contract is paged month by month.
#
# Saves per contract — NO stitching. A rolled "continuous" series injects basis
# jumps at every roll that are indistinguishable from real returns, which would
# corrupt the 20-bar window returns this method is built on.
#
# Usage: python download_crude_1min.py [--months-back 6]
# =============================================================================

import argparse
import importlib.util
import json
import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

DOWNLOADER_DIR = Path(r"D:\MyPython\Download_1min_History")
MCX_JSON = Path(r"D:\MyPython\SachinJ_Algo\Upstox Data\MCX.json")
OUT_DIR = DOWNLOADER_DIR / "data" / "crudeoil"

REQUEST_DELAY, BURST_EVERY, BURST_PAUSE = 0.5, 40, 5.0
MAX_RETRIES, RETRY_WAIT = 3, 10
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]


def load_token() -> str:
    spec = importlib.util.spec_from_file_location("c", DOWNLOADER_DIR / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.ANALYTICS_TOKEN


def crude_contracts() -> list[dict]:
    """Active CRUDEOIL futures from MCX.json. expiry is a millisecond epoch;
    asset_symbol distinguishes CRUDEOIL (full) from CRUDEOILM (mini)."""
    data = json.loads(MCX_JSON.read_text(encoding="utf-8"))
    out = []
    for r in data:
        if r.get("instrument_type") == "FUT" and r.get("asset_symbol") == "CRUDEOIL":
            exp = datetime.fromtimestamp(r["expiry"] / 1000, tz=timezone.utc).date()
            out.append({"key": r["instrument_key"], "expiry": exp,
                        "label": exp.strftime("%b%y").upper()})
    return sorted(out, key=lambda x: x["expiry"])


def fetch_1min(token, key, f, t):
    k = urllib.parse.quote(key, safe="")
    url = f"https://api.upstox.com/v3/historical-candle/{k}/minutes/1/{t}/{f}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json().get("data", {}).get("candles", [])
            if r.status_code == 400:
                return []          # outside this contract's trading window
            print(f"    HTTP {r.status_code} attempt {attempt}/{MAX_RETRIES}", flush=True)
        except requests.RequestException as e:
            print(f"    network error {e} attempt {attempt}/{MAX_RETRIES}", flush=True)
        time.sleep(RETRY_WAIT)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months-back", type=int, default=6)
    args = ap.parse_args()

    token = load_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    start = today - relativedelta(months=args.months_back)

    n_req = 0
    for c in crude_contracts():
        if c["expiry"] < start:
            continue
        frames, cur = [], start
        while cur < today:
            nxt = min(cur + relativedelta(months=1), today)
            candles = fetch_1min(token, c["key"], cur.isoformat(), nxt.isoformat())
            n_req += 1
            if candles:
                frames.append(pd.DataFrame(candles, columns=COLUMNS))
            cur = nxt
            time.sleep(BURST_PAUSE if n_req % BURST_EVERY == 0 else REQUEST_DELAY)

        if not frames:
            print(f"{c['label']} ({c['key']}): no data", flush=True)
            continue

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        out = OUT_DIR / f"CRUDEOIL_{c['label']}_1min.csv"
        df.to_csv(out, index=False)
        ist = df["timestamp"].dt.tz_convert("Asia/Kolkata")
        print(f"{c['label']} ({c['key']}): {len(df):,} bars, "
              f"{ist.min().date()} -> {ist.max().date()}, "
              f"{ist.dt.normalize().nunique()} sessions -> {out.name}", flush=True)

    print(f"\nDone. {n_req} API calls.", flush=True)


if __name__ == "__main__":
    main()
