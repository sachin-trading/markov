"""
Damage-when-wrong, CORRECTED.

Bugs in v1:
  1. Position direction was redrawn every day while costs were charged only
     every 20/60 days. That is not a holding period -- daily flip-flopping
     cancels moves a real held position compounds, so futures tail risk was
     UNDERSTATED. Fixed: direction is drawn once per holding period and held.
  2. The option row used invented parameters (delta 0.5, "60% of daily range").
     Fixed: bootstrap the 519 REAL DynaTrail trades with theta charged
     (mean -Rs 84, sd Rs 3,154, worst -Rs 6,085).

Still assumes ZERO EDGE everywhere -- the honest base case.
"""
import numpy as np
import pandas as pd

CAPITAL, N_PATHS, DAYS = 500_000, 20_000, 250
RNG = np.random.default_rng(20260814)

d = pd.read_csv(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv",
                parse_dates=["timestamp"])
px = pd.Series(d["close"].to_numpy(dtype=float),
               index=pd.to_datetime(d["timestamp"], utc=True)
               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)).sort_index()
rets = px.pct_change().dropna().to_numpy()

opt = pd.read_csv(r"D:\MyPython\SachinJ_Algo\markov\output\theta_comparison.csv")
opt_pnl = opt.loc[opt["theta"] == "on", "pnl"].to_numpy(dtype=float)
OPT_RATE = len(opt_pnl) / 1127.0          # trades per session, measured

SPOT, LOT = 23_800.0, 65
FUT_NOTIONAL, FUT_MARGIN = SPOT * LOT, 120_000

print(f"NIFTY: {len(rets):,} real daily returns (sd {rets.std():.2%}, worst {rets.min():.1%})")
print(f"Options: {len(opt_pnl)} real theta-charged trades "
      f"(mean {opt_pnl.mean():+,.0f}, worst {opt_pnl.min():,.0f}), "
      f"{OPT_RATE:.2f} trades/session\n")


def boot_returns(n_paths, n_days, block=20):
    out = np.empty((n_paths, n_days))
    nb = int(np.ceil(n_days / block))
    starts = RNG.integers(0, len(rets) - block, size=(n_paths, nb))
    for b in range(nb):
        seg = np.stack([rets[s:s + block] for s in starts[:, b]])
        lo, hi = b * block, min((b + 1) * block, n_days)
        out[:, lo:hi] = seg[:, :hi - lo]
    return out


def held_side(n_paths, n_days, hold):
    """Direction drawn ONCE per holding period, then held (the v1 bug)."""
    n_blocks = int(np.ceil(n_days / hold))
    s = RNG.choice([-1, 1], size=(n_paths, n_blocks))
    return np.repeat(s, hold, axis=1)[:, :n_days]


R = boot_returns(N_PATHS, DAYS)


def report(name, eq, ruined, note):
    final = eq[:, -1]
    peak = np.maximum.accumulate(eq, axis=1)
    dd = ((eq - peak) / np.maximum(peak, 1)).min(axis=1)
    p = final - CAPITAL
    return dict(name=name, median=np.median(p), p05=np.percentile(p, 5),
                maxdd=np.median(dd), p_half=(final < CAPITAL * .5).mean(),
                p_ruin=ruined.mean(), note=note)


out = []

# ── equity delivery, 60-day holds, unleveraged ──
for frac, tag in [(1.0, "1x"), (0.5, "0.5x")]:
    side = held_side(N_PATHS, DAYS, 60)
    eq = np.full((N_PATHS, DAYS + 1), float(CAPITAL))
    for t in range(DAYS):
        exp = eq[:, t] * frac
        c = exp * 0.00241 if t % 60 == 0 else 0.0
        eq[:, t + 1] = eq[:, t] + exp * side[:, t] * R[:, t] - c
        eq[:, t + 1] = np.maximum(eq[:, t + 1], 0.0)
    out.append(report(f"Equity delivery 60d ({tag})", eq, np.zeros(N_PATHS, bool),
                      "no margin call, no expiry"))

# ── NIFTY futures, direction HELD 20 days ──
for lots in (1, 3):
    side = held_side(N_PATHS, DAYS, 20)
    fu = np.full((N_PATHS, DAYS + 1), float(CAPITAL))
    ruin = np.zeros(N_PATHS, bool)
    for t in range(DAYS):
        pnl = side[:, t] * R[:, t] * FUT_NOTIONAL * lots
        c = 918.0 * lots if t % 20 == 0 else 0.0
        fu[:, t + 1] = np.where(~ruin, fu[:, t] + pnl - c, fu[:, t])
        ruin |= fu[:, t + 1] < FUT_MARGIN * lots
    out.append(report(f"NIFTY futures {lots} lot{'s' if lots > 1 else ''}, 20d held",
                      fu, ruin, f"{FUT_NOTIONAL * lots / CAPITAL:.1f}x leverage, margin call"))

# ── intraday options: bootstrap the real theta-charged trades ──
op = np.full((N_PATHS, DAYS + 1), float(CAPITAL))
ruin_o = np.zeros(N_PATHS, bool)
for t in range(DAYS):
    trades = RNG.random(N_PATHS) < OPT_RATE
    draw = RNG.choice(opt_pnl, size=N_PATHS)
    op[:, t + 1] = np.where(~ruin_o, op[:, t] + np.where(trades, draw, 0.0), op[:, t])
    ruin_o |= op[:, t + 1] < 15_000
out.append(report("Intraday options (real trades)", op, ruin_o,
                  "loss capped per trade, theta daily"))

print("=" * 106)
print(f"DAMAGE WITH ZERO EDGE — Rs {CAPITAL:,}, {DAYS} days, {N_PATHS:,} simulated years")
print("=" * 106)
print(f"{'mode':<32}{'median':>12}{'bad year (5%)':>15}{'med maxDD':>12}"
      f"{'P(lose>50%)':>13}{'P(wiped out)':>14}")
print("-" * 106)
for r in sorted(out, key=lambda x: -x["median"]):
    print(f"{r['name']:<32}{r['median']:>11,.0f}{r['p05']:>15,.0f}{r['maxdd']:>11.0%}"
          f"{r['p_half']:>13.0%}{r['p_ruin']:>14.0%}")

print("\nCaveat that makes futures WORSE than shown: SPAN margin rises when")
print("volatility rises, so real margin calls arrive earlier than this fixed-margin model.")
