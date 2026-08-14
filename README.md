# Markov 2.0 — Hedge Fund Method (Upstox / Indian Markets Edition)

Regime detection for Indian markets: label history into BULL / BEAR / SIDEWAYS states,
build a transition matrix, trade (or gate a strategy with) the differential
**P(bull next) − P(bear next)**.

Corrected "2.0" version — three documented flaws of the original are fixed:

| Fix | Flaw | Correction |
|---|---|---|
| 1 | Overlapping 20-day windows share 19 days → fake diagonal persistence | Stride-sampled (non-overlapping) matrix is the honest one; both are always shown side by side, cells with <10 observations flagged |
| 2 | A display once shipped with bull/bear swapped | `assert_labels_verified()` re-derives every label and checks known anchors (COVID crash = BEAR, 2020-21 recovery = BULL, flattest stretch = SIDEWAYS) before anything is displayed |
| 3 | Ambiguity about how the signal is used | Explicit **FILTER** mode (signal gates an existing strategy) vs **STANDALONE** mode (trade the differential, whole lots, real Indian F&O costs) |

## Files

- `markov_regime.py` — core library: states, matrices, signal, forecasts, verification, walk-forward, Upstox F&O cost model
- `run_demo.py` — proof run on NIFTY 50 daily: FIX 2 gate → calibration → both matrices → forecasts/stationary convergence → walk-forward before-fix vs after-fix, net of costs; writes `output/markov2_nifty_walkforward.png` + summary CSV
- `download_nifty_daily.py` — builds a 10+ year NIFTY daily series year-by-year via the Upstox v3 daily endpoint (Analytics Token read at runtime from the local downloader config — **never stored in this repo**)

## Data & credentials

Reads local CSVs from `D:\MyPython\Download_1min_History\data` (schema:
`timestamp(+05:30 IST), open, high, low, close, volume, oi`). The Upstox Analytics
Token stays in the local `config.py` outside this repo. Nothing secret is committed.

## Honest reporting

Walk-forward only (the matrix never sees its test data), expanding window, matrix
rebuilt as it walks, Upstox flat-fee F&O costs + slippage applied.

> Backtests flatter. The fixed matrix shows uglier, truer numbers — those are the
> only ones worth trading.
