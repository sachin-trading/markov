# =============================================================================
# markov_regime.py — Markov 2.0 Hedge Fund Method (corrected), India edition
#
# States -> transition matrix -> stickiness -> signal, with the three fixes:
#   FIX 1: stride-sampled (non-overlapping) transition matrix alongside the
#          legacy overlapping one — only the stride matrix is statistically honest
#   FIX 2: programmatic label verification against known NIFTY periods
#   FIX 3: explicit FILTER vs STANDALONE modes (see run_demo.py / SKILL.md)
# =============================================================================

import numpy as np
import pandas as pd

# State encoding — verified programmatically by verify_label_mapping(), never trust by eye
STATE_SIDEWAYS, STATE_BULL, STATE_BEAR = 0, 1, 2
STATE_NAMES = {STATE_SIDEWAYS: "SIDEWAYS", STATE_BULL: "BULL", STATE_BEAR: "BEAR"}
N_STATES = 3

MIN_CELL_OBS = 10  # cells resting on fewer transitions than this are flagged unreliable


# ── States ───────────────────────────────────────────────────────────────────

def window_returns(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing cumulative return over `window` bars: close[t]/close[t-window] - 1."""
    return (close / close.shift(window) - 1.0).dropna()


def label_states(close: pd.Series, window: int = 20,
                 bull_thr: float = 0.05, bear_thr: float = -0.05) -> pd.Series:
    """
    Label each bar by its trailing `window`-bar cumulative return.
    >= bull_thr -> BULL, <= bear_thr -> BEAR, else SIDEWAYS.
    Uses only data up to and including bar t — no lookahead.
    """
    ret = window_returns(close, window)
    states = pd.Series(STATE_SIDEWAYS, index=ret.index, dtype=int)
    states[ret >= bull_thr] = STATE_BULL
    states[ret <= bear_thr] = STATE_BEAR
    return states


def percentile_thresholds(close: pd.Series, window: int = 20,
                          pct: float = 0.25) -> tuple[float, float]:
    """Alternative calibration: top/bottom `pct` of window returns as thresholds."""
    ret = window_returns(close, window)
    return float(ret.quantile(1 - pct)), float(ret.quantile(pct))


def state_distribution(states: pd.Series) -> dict:
    """Fraction of bars in each state, by name."""
    frac = states.value_counts(normalize=True)
    return {STATE_NAMES[s]: float(frac.get(s, 0.0)) for s in range(N_STATES)}


# ── FIX 2: label verification ────────────────────────────────────────────────

def verify_label_mapping(close: pd.Series, window: int = 20,
                         bull_thr: float = 0.05, bear_thr: float = -0.05) -> list[dict]:
    """
    Programmatic self-check of the state labels. Returns a list of check dicts
    with keys: name, passed, detail. Callers must abort any display if a check fails.

    Checks:
      1. Definition consistency — every label re-derived from raw returns must match.
      2. COVID crash (Feb–Mar 2020) windows must contain BEAR; 2020-03-23 must be BEAR.
      3. 2020-21 recovery rally must contain BULL.
      4. The flattest window in the history must be SIDEWAYS.
    If history starts after Feb 2020, checks 2–3 substitute the June-2024
    election-result crash day and the subsequent rally.
    """
    states = label_states(close, window, bull_thr, bear_thr)
    ret = window_returns(close, window)
    checks = []

    # 1 — definition consistency (catches any bull/bear swap in the mapping)
    ok = (bool((states[ret >= bull_thr] == STATE_BULL).all())
          and bool((states[ret <= bear_thr] == STATE_BEAR).all())
          and bool((states[(ret > bear_thr) & (ret < bull_thr)] == STATE_SIDEWAYS).all()))
    checks.append({"name": "definition-consistency", "passed": ok,
                   "detail": "every label re-derived from raw window returns matches its state code"})

    have_2020 = states.index.min() <= pd.Timestamp("2020-02-01")

    if have_2020:
        # 2 — COVID crash must label BEAR
        crash = states.loc["2020-03-15":"2020-04-10"]
        bottom = states.loc["2020-03-20":"2020-03-25"]
        ok = (not crash.empty and (crash == STATE_BEAR).any()
              and not bottom.empty and (bottom == STATE_BEAR).all())
        detail = f"2020-03-23 window return {ret.loc['2020-03-20':'2020-03-25'].min():+.1%} -> BEAR" if not crash.empty else "no data"
        checks.append({"name": "covid-crash-is-BEAR", "passed": ok, "detail": detail})

        # 3 — recovery rally must label BULL
        rally = states.loc["2020-11-01":"2021-02-28"]
        ok = not rally.empty and (rally == STATE_BULL).any()
        checks.append({"name": "2020-21-recovery-is-BULL", "passed": ok,
                       "detail": f"{(rally == STATE_BULL).mean():.0%} of Nov-2020→Feb-2021 windows BULL" if not rally.empty else "no data"})
    else:
        # Fallback anchors: June-2024 election-result crash + subsequent rally
        crash_day = ret.loc["2024-06-04":"2024-06-05"]
        ok = not crash_day.empty and float(crash_day.iloc[0]) < 0 and \
             int(states.loc[crash_day.index[0]]) != STATE_BULL
        checks.append({"name": "election-2024-crash-not-BULL", "passed": ok,
                       "detail": f"2024-06-04 window return {float(crash_day.iloc[0]):+.1%}" if not crash_day.empty else "no data"})
        rally = states.loc["2024-06-20":"2024-09-30"]
        ok = not rally.empty and (rally == STATE_BULL).any()
        checks.append({"name": "post-election-rally-is-BULL", "passed": ok,
                       "detail": f"{(rally == STATE_BULL).mean():.0%} of Jun→Sep-2024 windows BULL" if not rally.empty else "no data"})

    # 4 — flattest stretch must be SIDEWAYS
    flattest = ret.abs().idxmin()
    ok = int(states.loc[flattest]) == STATE_SIDEWAYS
    checks.append({"name": "flattest-window-is-SIDEWAYS", "passed": ok,
                   "detail": f"{flattest.date()} window return {float(ret.loc[flattest]):+.2%}"})

    return checks


def assert_labels_verified(close: pd.Series, window: int = 20,
                           bull_thr: float = 0.05, bear_thr: float = -0.05) -> list[dict]:
    """Run verify_label_mapping and raise if any check fails (FIX 2 gate)."""
    checks = verify_label_mapping(close, window, bull_thr, bear_thr)
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise AssertionError("Label verification FAILED: "
                             + "; ".join(f"{c['name']} ({c['detail']})" for c in failed))
    return checks


# ── Transition matrices ──────────────────────────────────────────────────────

def transition_counts(states: pd.Series, stride: int = 1) -> np.ndarray:
    """
    3x3 transition count matrix from the label series sampled every `stride` bars.

    stride=1        -> legacy OVERLAPPING matrix: consecutive daily labels whose
                       windows share (window-1) bars — fakes persistence (FIX 1 flaw).
    stride=window   -> stride-sampled TRUE matrix: non-overlapping windows only.
    """
    sampled = states.iloc[::stride].to_numpy()
    counts = np.zeros((N_STATES, N_STATES), dtype=float)
    for a, b in zip(sampled[:-1], sampled[1:]):
        counts[a, b] += 1
    return counts


def to_probabilities(counts: np.ndarray) -> np.ndarray:
    """Row-normalize counts to probabilities; all-zero rows stay uniform."""
    P = np.full((N_STATES, N_STATES), 1.0 / N_STATES)
    rows = counts.sum(axis=1)
    for i in range(N_STATES):
        if rows[i] > 0:
            P[i] = counts[i] / rows[i]
    return P


def stickiness(P: np.ndarray) -> dict:
    return {STATE_NAMES[i]: float(P[i, i]) for i in range(N_STATES)}


def unreliable_cells(counts: np.ndarray, min_obs: int = MIN_CELL_OBS) -> list[str]:
    """Cells resting on fewer than `min_obs` transitions — treat with suspicion."""
    out = []
    for i in range(N_STATES):
        for j in range(N_STATES):
            if counts[i, j] < min_obs:
                out.append(f"{STATE_NAMES[i]}->{STATE_NAMES[j]} ({int(counts[i, j])} obs)")
    return out


# ── Signal & forecasts ───────────────────────────────────────────────────────

def signal(P: np.ndarray, current_state: int) -> float:
    """P(bull next) - P(bear next). Sign = direction, magnitude = conviction."""
    return float(P[current_state, STATE_BULL] - P[current_state, STATE_BEAR])


def forecast(P: np.ndarray, current_state: int, steps: int) -> np.ndarray:
    """State distribution `steps` transitions ahead via matrix power."""
    return np.linalg.matrix_power(P, steps)[current_state]


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Long-run distribution — where all forecasts converge; carries no signal."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def matrix_report(states: pd.Series, window: int) -> dict:
    """Both matrices side by side (FIX 1): legacy overlapping + stride-sampled true."""
    c_overlap = transition_counts(states, stride=1)
    c_stride = transition_counts(states, stride=window)
    return {
        "overlap": {"counts": c_overlap, "P": to_probabilities(c_overlap),
                    "n_transitions": int(c_overlap.sum()),
                    "unreliable": unreliable_cells(c_overlap)},
        "stride": {"counts": c_stride, "P": to_probabilities(c_stride),
                   "n_transitions": int(c_stride.sum()),
                   "unreliable": unreliable_cells(c_stride)},
    }


def format_matrix(P: np.ndarray, counts: np.ndarray | None = None) -> str:
    names = [STATE_NAMES[i] for i in range(N_STATES)]
    lines = ["            " + "".join(f"{n:>12}" for n in names)]
    for i in range(N_STATES):
        row = "".join(f"{P[i, j]:>11.1%} " for j in range(N_STATES))
        obs = f"   (row n={int(counts[i].sum())})" if counts is not None else ""
        lines.append(f"{names[i]:>10}  {row}{obs}")
    return "\n".join(lines)


# ── Indian F&O costs (per vault note 26, Upstox flat-fee model) ──────────────

def futures_leg_cost(price: float, lots: int, lot_size: int, side: str,
                     slippage_pts: float = 1.0) -> float:
    """
    Cost in ₹ of one NSE F&O futures order leg (buy or sell) at `price`.
    Brokerage ₹20/order, STT 0.01% sell-side, exchange 0.002%, SEBI ₹10/cr,
    18% GST on fees, stamp 0.002% buy-side, plus slippage in index points.
    """
    value = price * lots * lot_size
    brokerage = 20.0
    stt = value * 0.0001 if side == "sell" else 0.0
    exchange = value * 0.00002
    sebi = value * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = value * 0.00002 if side == "buy" else 0.0
    slippage = slippage_pts * lots * lot_size
    return brokerage + stt + exchange + sebi + gst + stamp + slippage


# ── Walk-forward backtest (STANDALONE mode on the differential) ──────────────

def walk_forward(close: pd.Series, window: int = 20,
                 bull_thr: float = 0.05, bear_thr: float = -0.05,
                 matrix_mode: str = "stride",
                 min_train_bars: int = 1250,
                 signal_threshold: float = 0.10,
                 lot_size: int = 65, lots: int = 1,
                 slippage_pts: float = 1.0,
                 apply_costs: bool = True) -> dict:
    """
    Expanding-window walk-forward: at each bar t the matrix is rebuilt from
    transitions available up to t only (never test on data the matrix has
    learned from), the signal decides the position held over t -> t+1.

    matrix_mode: "stride" (FIX 1, honest) or "overlap" (legacy, flawed) —
    both exposed so before-fix vs after-fix can be compared on identical rules.

    Position: +lots if signal > +signal_threshold, -lots if < -signal_threshold,
    else flat. P&L on NIFTY futures notional (lots × lot_size), costs per leg.
    """
    states = label_states(close, window, bull_thr, bear_thr)  # label(t) uses data ≤ t only
    close = close.loc[states.index[0]:]
    stride = window if matrix_mode == "stride" else 1

    n = len(states)
    idx = states.index
    pos = np.zeros(n, dtype=int)
    sig = np.full(n, np.nan)

    sampled_labels = states.iloc[::stride]           # fixed non-overlapping grid
    counts = np.zeros((N_STATES, N_STATES))
    consumed = 0  # transitions of the sampled grid already added to counts

    state_arr = states.to_numpy()
    samp_arr = sampled_labels.to_numpy()

    for t in range(n - 1):
        # add any newly-completed sampled transitions up to time t
        k_avail = int(np.searchsorted(sampled_labels.index.values, idx[t].to_datetime64(), side="right"))
        while consumed < k_avail - 1:
            counts[samp_arr[consumed], samp_arr[consumed + 1]] += 1
            consumed += 1
        if t < min_train_bars:
            continue
        P = to_probabilities(counts)
        s = signal(P, int(state_arr[t]))
        sig[t] = s
        pos[t + 1] = lots if s > signal_threshold else (-lots if s < -signal_threshold else 0)

    # P&L: position held over bar t (set from info at t-1) earns close[t-1]->close[t]
    px = close.to_numpy(dtype=float)
    pnl = np.zeros(n)
    costs = np.zeros(n)
    for t in range(1, n):
        pnl[t] = pos[t] * (px[t] - px[t - 1]) * lot_size
        d = pos[t] - pos[t - 1]
        if d != 0 and apply_costs:
            # one order leg per lot-change direction; a reversal is exit+entry = 2 legs
            legs = []
            if pos[t - 1] != 0 and np.sign(pos[t]) != np.sign(pos[t - 1]):
                legs.append((abs(pos[t - 1]), "sell" if pos[t - 1] > 0 else "buy"))
                if pos[t] != 0:
                    legs.append((abs(pos[t]), "buy" if pos[t] > 0 else "sell"))
            else:
                legs.append((abs(d), "buy" if d > 0 else "sell"))
            costs[t] = sum(futures_leg_cost(px[t], q, lot_size, side, slippage_pts)
                           for q, side in legs)
    net = pnl - costs

    capital = px[min_train_bars] * lot_size * lots  # fully-funded notional, no leverage
    equity = pd.Series(capital + np.cumsum(net), index=idx)
    eq_test = equity.iloc[min_train_bars:]

    # trade-level stats: a trade = maximal run of a constant non-zero position
    trades = []
    t0 = None
    for t in range(1, n):
        if pos[t] != 0 and (pos[t] != pos[t - 1]):
            if t0 is not None:
                trades.append(net[t0:t].sum())
            t0 = t
        elif pos[t] == 0 and t0 is not None:
            trades.append(net[t0:t].sum())
            t0 = None
    if t0 is not None:
        trades.append(net[t0:].sum())
    trades = np.array(trades) if trades else np.array([0.0])

    wins, losses = trades[trades > 0], trades[trades < 0]
    years = max((idx[-1] - idx[min_train_bars]).days / 365.25, 1e-9)
    dd = (eq_test / eq_test.cummax() - 1.0).min()

    bh_ret = px[-1] / px[min_train_bars] - 1.0
    bh_curve = pd.Series(px, index=idx).iloc[min_train_bars:]
    bh_dd = (bh_curve / bh_curve.cummax() - 1.0).min()

    return {
        "mode": matrix_mode, "signal_threshold": signal_threshold,
        "equity": eq_test, "positions": pd.Series(pos, index=idx),
        "signals": pd.Series(sig, index=idx),
        "n_trades": int(len(trades)),
        "win_rate": float((trades > 0).mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size else float("inf"),
        "total_return": float(eq_test.iloc[-1] / capital - 1.0),
        "cagr": float((eq_test.iloc[-1] / capital) ** (1 / years) - 1.0),
        "max_drawdown": float(dd),
        "total_costs": float(costs.sum()),
        "net_pnl": float(net[min_train_bars:].sum()),
        "buyhold_return": float(bh_ret), "buyhold_maxdd": float(bh_dd),
        "test_start": str(idx[min_train_bars].date()), "test_end": str(idx[-1].date()),
        "capital": float(capital),
    }
