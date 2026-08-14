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
- `filter_dynatrail.py` — FILTER mode: gates DynaTrail's entries with the regime signal, gated vs ungated on identical engine/config/costs
- `filter_dynatrail_control.py` — the control that FILTER result demands: separates a real regime edge from a plain long-only bias
- `filter_significance.py` — subset check + permutation test, because 41 trades is a thin sample

## The drift trap (found 2026-08-14, read before using FILTER mode)

On NIFTY the raw signal is positive from **all three** states (BEAR +0.48, BULL +0.13,
SIDEWAYS +0.04) because the index drifts upward. A gate written as "short only when
signal < −threshold" can therefore never fire: the filter silently becomes long-only.
In the DynaTrail test it allowed CE on 18% of sessions, PE on **0%**, and blocked 82%
entirely — improving profit factor (2.00 vs 1.39) while giving up ₹143,619 of net P&L,
including 270 profitable PE trades.

The regime effect on the side it *did* permit is real (₹956/trade gated vs ₹82 ungated on
CE trades, permutation p = 0.029), but always run the same-side control before believing a
FILTER headline.

**The de-meaned fix did not rescue it** (`filter_dynatrail_demeaned.py`, vault note 40
Entry #016, run 2026-08-14). Comparing the signal against its expanding mean instead of zero
makes the short leg reachable in principle — 215 days cleared −0.10 across the full history —
but all of them predate October 2013, while the matrix was still immature. Once the matrix
converges, the three states sit at raw signals of +0.04 / +0.13 / +0.48 against a running mean
of ≈+0.10, so the deepest attainable de-meaned value is **−0.066**, mathematically above the
−0.10 threshold. Result: 18 trades, all CE again, INCONCLUSIVE against its pre-registered bar.

The real constraint is structural: **with three states the signal takes only three values on
any given day**, and on a drifting index its downside range is too compressed to cross a
symmetric threshold. Gating shorts needs a different construction — more states, a lower
short threshold, or a signal that isn't a difference of two probabilities — not another
adjustment to this one. Each is a new registry entry, not a tweak.

## Data & credentials

Reads local CSVs from `D:\MyPython\Download_1min_History\data` (schema:
`timestamp(+05:30 IST), open, high, low, close, volume, oi`). The Upstox Analytics
Token stays in the local `config.py` outside this repo. Nothing secret is committed.

## Honest reporting

Walk-forward only (the matrix never sees its test data), expanding window, matrix
rebuilt as it walks, Upstox flat-fee F&O costs + slippage applied.

> Backtests flatter. The fixed matrix shows uglier, truer numbers — those are the
> only ones worth trading.
