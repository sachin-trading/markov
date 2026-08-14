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
| **4** | **Stride sampling still has to pick a grid START bar, and the answer depends on it** | `phase_report()` computes the matrix for all `stride` phases and reports mean + range; any signal whose sign flips across phases is an artifact (`phase_robustness.py`) |

## FIX 4 — the one the original three missed

FIX 1 correctly bans overlapping windows. But a non-overlapping grid must begin somewhere,
and there are 20 equally valid starting bars. Checking all of them changes the conclusions:

| | Phase 0 (naive) | Mean across 20 phases | Range | Sign stable? |
|---|---|---|---|---|
| NIFTY BEAR-row signal | +0.484 | +0.216 | [+0.000, +0.484] | **no** |
| NIFTY P(BULL \| BEAR) | 54.8% | 36.0% | [20.0%, 54.8%] | — |
| CrudeOil BULL-row signal | −0.132 | +0.015 | [−0.132, +0.180] | **no** |
| CrudeOil BEAR-row signal | +0.200 | +0.034 | [−0.123, +0.200] | **no** |

The naive phase-0 value is the *maximum* of the twenty in both NIFTY cases, and the low end of
NIFTY's P(BULL|BEAR) range sits below the 22.9% unconditional base rate. **The earlier claim
that NIFTY's 20-day signal is mean-reverting described one lucky grid alignment, not the
market.** A related trap worth checking alongside it: consecutive stride windows share an
endpoint price, inducing negative autocorrelation ≈ −(bar sd / window sd)² — enough to
manufacture fake mean reversion on its own. Compare `stride = window` against
`stride = window + 1` to isolate it.

This is the fourth appearance of a single recurring error: overlapping windows (FIX 1), trades
clustered inside regime episodes, correlated stocks (Entry #017), and now grid phase. Every
time, a sampling choice manufactured the pattern.

## Files

- `markov_regime.py` — core library: states, matrices, signal, forecasts, verification, walk-forward, Upstox F&O cost model
- `run_demo.py` — proof run on NIFTY 50 daily: FIX 2 gate → calibration → both matrices → forecasts/stationary convergence → walk-forward before-fix vs after-fix, net of costs; writes `output/markov2_nifty_walkforward.png` + summary CSV
- `download_nifty_daily.py` — builds a 10+ year NIFTY daily series year-by-year via the Upstox v3 daily endpoint (Analytics Token read at runtime from the local downloader config — **never stored in this repo**)
- `filter_dynatrail.py` — FILTER mode: gates DynaTrail's entries with the regime signal, gated vs ungated on identical engine/config/costs
- `filter_dynatrail_control.py` — the control that FILTER result demands: separates a real regime edge from a plain long-only bias
- `filter_significance.py` — subset check + permutation test, because 41 trades is a thin sample
- `cross_sectional_bear.py` — Entry #017: the mechanism tested across 46 NIFTY50 stocks with a date-block bootstrap
- `download_nifty50_daily.py` — long daily history for all 50 constituents
- `phase_robustness.py` — FIX 4: the matrix across all sampling-grid phases, on both instruments
- `crude_regime_report.py` / `download_crude_1min.py` — MCX CrudeOil, descriptive only (see below)

## MCX CrudeOil

**The daily horizon is not reachable.** Expired MCX contracts need Upstox Plus (this token has
`isPlusPlan: false`), so only currently-listed contracts are available: the deepest starts
2026-02-20, and the union across all six is ~119 trading days — about **4 non-overlapping
20-day transitions** for a 9-cell matrix. `crude_regime_report.py` therefore uses a 20-**bar**
window on 15-min data (5 hours ≈ ⅓ of an MCX session), which answers a different question.

There, the default ±5% thresholds label 97% of windows SIDEWAYS — a 20-bar crude window has
sd 1.98% against 5.99% for a 20-day NIFTY window — so percentile calibration is mandatory.
The single-phase matrix showed a clean two-sided mean-reversion signal (BULL −0.132,
BEAR +0.200) and, unlike NIFTY, both gate legs were reachable. **It did not survive FIX 4:**
all three signals flip sign across phases and the phase means are ≈0. Independence is also
worse than it looks — 207 stride transitions inside just 75 sessions of one contract over a
single 113-day stretch. Use one contract, never a stitched continuous series: each roll injects
a basis jump indistinguishable from a real return.

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
symmetric threshold.

## Verdict: the FILTER line is closed (2026-08-14)

Rather than keep adjusting the gate, Entry #017 tested the mechanism it depended on —
"BEAR windows are followed by unusual strength" — across **46 NIFTY50 constituents**,
11,859 non-overlapping (stock, date) observations on 267 shared sample dates
(`cross_sectional_bear.py`). Both pre-registered conditions failed: 33 of 46 stocks
positive against a bar of 35, and a pooled edge of +0.55% at bootstrap **p = 0.173**.

**The methodological result outlasts the trading one.** The same data analysed three other
ways all claim a discovery:

| Resampling unit | p-value | Verdict |
|---|---|---|
| **Dates** (pre-registered, correct) | **0.1728** | **FAIL** |
| Stocks (explicitly forbidden) | 0.0090 | PASS |
| Individual observations (naive) | 0.0050 | PASS |
| Naive sign test, 33/46 positive | 0.00227 | PASS |

Indian equities crash together — 2008, 2011, 2020 — so treating 46 names as 46 independent
samples is fiction. Only resampling whole dates preserves that co-movement. Three of four
plausible analyses would have manufactured a false positive, and the ban on the stock-level
bootstrap is credible only because it was written down before the numbers were seen. This is
the same error as FIX 1 (overlapping windows faking persistence), wearing different clothes.

Markov 2.0 is retained as a **descriptive regime monitor** — current state, honest
stride-sampled stickiness, forecast decay to the stationary distribution. It should not gate
any strategy without a new registry entry and a fresh correlation-aware test.

## Data & credentials

Reads local CSVs from `D:\MyPython\Download_1min_History\data` (schema:
`timestamp(+05:30 IST), open, high, low, close, volume, oi`). The Upstox Analytics
Token stays in the local `config.py` outside this repo. Nothing secret is committed.

## Honest reporting

Walk-forward only (the matrix never sees its test data), expanding window, matrix
rebuilt as it walks, Upstox flat-fee F&O costs + slippage applied.

> Backtests flatter. The fixed matrix shows uglier, truer numbers — those are the
> only ones worth trading.
