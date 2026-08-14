"""
Option SELLING -- the one option strategy with a documented structural edge.

Your own note 34 identified why N1/N2 (long straddle/strangle) failed:
"implied volatility structurally overstates realized volatility on average
(the volatility risk premium) -- option buyers pay for that premium."

Selling puts you on the RIGHT side of that premium. So this is not the same
question as buying. The real question is whether the edge survives (a) costs
and (b) the tail risk you are being paid to underwrite.

Modelled on real NIFTY weekly moves, 2005-2026, including 2008 and Mar-2020.
Premium is set from realised vol scaled by an explicit VRP assumption --
stated, not hidden, and swept for sensitivity below.
"""
import numpy as np
import pandas as pd

CAPITAL, LOT, SPOT = 500_000, 65, 23_800.0
RNG = np.random.default_rng(20260814)

d = pd.read_csv(r"D:\MyPython\Download_1min_History\data\nifty\NIFTY_daily.csv",
                parse_dates=["timestamp"])
px = pd.Series(d["close"].to_numpy(dtype=float),
               index=pd.to_datetime(d["timestamp"], utc=True)
               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)).sort_index()

# real overlapping 5-day (weekly) returns, and the trailing vol at each point
wk = (px.shift(-5) / px - 1.0).dropna()
rv20 = px.pct_change().rolling(20).std().reindex(wk.index)
ok = rv20.notna()
wk, rv20 = wk[ok].to_numpy(), rv20[ok].to_numpy()
print(f"{len(wk):,} real NIFTY weekly moves. "
      f"sd {wk.std():.2%}, worst {wk.min():.1%}, best {wk.max():+.1%}\n")

# costs for a 2-leg short strangle, current Apr-2026 rates, per lot round trip
def strangle_cost(premium_pts):
    val = premium_pts * LOT
    stt = val * 0.0015              # 0.15% on sell premium
    exch = val * 0.0003503
    brok = 40.0                     # 2 legs x Rs 20
    gst = (brok + exch) * 0.18
    return (stt + exch + brok + gst) * 2   # open + close, both legs


def simulate(vrp, width_sd, margin_per_lot=150_000, n_paths=20_000, weeks=50):
    """Sell a `width_sd`-sigma strangle every week for a year."""
    eq = np.full((n_paths, weeks + 1), float(CAPITAL))
    ruin = np.zeros(n_paths, bool)
    wins = np.zeros((n_paths, weeks), bool)
    for w in range(weeks):
        i = RNG.integers(0, len(wk), n_paths)
        move, vol = wk[i], rv20[i]
        wvol = vol * np.sqrt(5)                       # weekly sigma
        k = width_sd * wvol                           # strike distance, fraction
        # premium collected: both legs, priced at IV = RV * vrp
        prem_frac = 0.40 * wvol * vrp * 2             # rough 2-leg OTM premium
        premium = prem_frac * SPOT
        # payout: intrinsic beyond the strikes
        breach = np.maximum(np.abs(move) - k, 0.0) * SPOT
        pnl = (premium - breach) * LOT - strangle_cost(premium)
        wins[:, w] = pnl > 0
        eq[:, w + 1] = np.where(~ruin, eq[:, w] + pnl, eq[:, w])
        ruin |= eq[:, w + 1] < margin_per_lot
    return eq, ruin, wins


print("=" * 104)
print("SHORT STRANGLE, 1 lot weekly — zero skill, structural VRP edge only")
print("=" * 104)
print(f"{'VRP (IV/RV)':<13}{'strikes':<10}{'win rate':>10}{'median':>12}"
      f"{'bad yr (5%)':>14}{'worst':>13}{'P(wiped out)':>14}")
print("-" * 104)

rows = []
for vrp in (1.30, 1.15, 1.00):
    for wsd, tag in ((1.5, "1.5 sd"), (1.0, "1.0 sd")):
        eq, ruin, wins = simulate(vrp, wsd)
        p = eq[:, -1] - CAPITAL
        rows.append((vrp, tag, wins.mean(), np.median(p), np.percentile(p, 5),
                     p.min(), ruin.mean()))
        print(f"{vrp:<13.2f}{tag:<10}{wins.mean():>9.0%}{np.median(p):>12,.0f}"
              f"{np.percentile(p, 5):>14,.0f}{p.min():>13,.0f}{ruin.mean():>14.0%}")

print("\n" + "=" * 104)
print("THE SKEW — why a high win rate is not safety (VRP 1.15, 1.5 sd strikes)")
print("=" * 104)
eq, ruin, wins = simulate(1.15, 1.5)
p = eq[:, -1] - CAPITAL
print(f"  win rate per week            {wins.mean():.0%}")
print(f"  median year                  {np.median(p):>+12,.0f}")
print(f"  mean year                    {p.mean():>+12,.0f}")
print(f"  25th pct                     {np.percentile(p, 25):>+12,.0f}")
print(f"   5th pct                     {np.percentile(p, 5):>+12,.0f}")
print(f"   1st pct                     {np.percentile(p, 1):>+12,.0f}")
print(f"  worst of 20,000 years        {p.min():>+12,.0f}")
print(f"  P(losing year)               {(p < 0).mean():.0%}")
print(f"  P(lose > half capital)       {(p < -CAPITAL/2).mean():.0%}")
print(f"  P(wiped out / margin call)   {ruin.mean():.0%}")
mean_win = p[p > 0].mean() if (p > 0).any() else 0
mean_loss = p[p < 0].mean() if (p < 0).any() else 0
print(f"\n  average winning year {mean_win:>+11,.0f} | average losing year {mean_loss:>+11,.0f}"
      f"  -> one bad year erases {abs(mean_loss / max(mean_win, 1)):.1f} good ones")
