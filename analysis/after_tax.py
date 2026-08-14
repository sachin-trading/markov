"""
After-tax comparison, FY 2026-27 rates.

The critical fact: F&O is NOT a capital asset (Sec 2(14)), so F&O profit is
never STCG or LTCG -- it is non-speculative business income (Sec 43(5)(d)) at
slab rates, REGARDLESS of holding period. Holding NIFTY futures 60 days has the
same tax treatment as holding it 60 minutes. There is no holding-period tax
cliff in F&O at all.

Equity is where holding period changes the rate -- and it moves the OPPOSITE
way to the user's assumption:
    intraday equity  -> speculative business income, slab (up to 31.2%)
    <= 12 months     -> STCG 20% + cess = 20.8%
    >  12 months     -> LTCG 12.5% + cess = 13.0%, first Rs 1.25L exempt
Holding LONGER lowers the equity tax rate; it never raises it.
"""

CESS = 1.04
SLAB_TOP = 0.30 * CESS      # 31.2%  F&O business income, top bracket
STCG = 0.20 * CESS          # 20.8%  equity <= 12 months
LTCG = 0.125 * CESS         # 13.0%  equity > 12 months (above Rs 1.25L)

COST_RT = {"futures": 0.00059, "equity": 0.00241}   # per round trip, % of position

GROSS = 0.15   # assumed gross annual return before costs & tax, % of position

SPECS = [
    # label,               instrument, round trips/yr, tax rate,  tax label
    ("Futures  5-day hold",  "futures", 50, SLAB_TOP, "slab 31.2%"),
    ("Futures  20-day hold", "futures", 12, SLAB_TOP, "slab 31.2%"),
    ("Futures  60-day hold", "futures",  4, SLAB_TOP, "slab 31.2%"),
    ("Equity   20-day hold", "equity",  12, STCG,     "STCG 20.8%"),
    ("Equity   60-day hold", "equity",   4, STCG,     "STCG 20.8%"),
    ("Equity   >12-month",   "equity",   1, LTCG,     "LTCG 13.0%"),
]

print("=" * 96)
print(f"AFTER-TAX RETURN, assuming {GROSS:.0%} gross annual return before costs (% of position value)")
print("=" * 96)
print(f"{'strategy':<24}{'trips/yr':>9}{'cost drag':>11}{'net pre-tax':>13}"
      f"{'tax treatment':>15}{'after tax':>11}{'kept':>8}")
print("-" * 96)

rows = []
for label, inst, trips, tax, taxlabel in SPECS:
    drag = trips * COST_RT[inst]
    pre = GROSS - drag
    post = pre * (1 - tax)
    kept = post / GROSS
    rows.append((label, post, kept))
    print(f"{label:<24}{trips:>9}{drag:>10.2%}{pre:>13.2%}{taxlabel:>15}{post:>11.2%}{kept:>8.0%}")

print("\n" + "=" * 96)
print("RANKING — after-tax return kept from the same gross edge")
print("=" * 96)
for label, post, kept in sorted(rows, key=lambda x: -x[1]):
    print(f"  {label:<24}{post:>8.2%}   {'#' * max(int(post * 400), 1)}")

print("\n" + "=" * 96)
print("WHERE THE MONEY GOES — same gross edge, three horizons")
print("=" * 96)
for label, inst, trips, tax, taxlabel in SPECS:
    drag = trips * COST_RT[inst]
    pre = GROSS - drag
    taxamt = pre * tax
    print(f"  {label:<24} costs {drag / GROSS:>5.0%} of gross | "
          f"tax {taxamt / GROSS:>5.0%} | you keep {(pre - taxamt) / GROSS:>5.0%}")

print("\n" + "=" * 96)
print("F&O-ONLY ADVANTAGES (do not apply to equity capital gains)")
print("=" * 96)
print("  - STT on F&O is a DEDUCTIBLE business expense; STT on equity CG is not")
print("  - brokerage, data feeds, internet, VPS/EC2, computer depreciation all deductible")
print("  - losses carry forward 8 years, set off against ANY business income")
print("  - equity INTRADAY losses are speculative: offset only speculative gains, 4-yr carry")
eff = SLAB_TOP * (1 - 0.15)
print(f"  -> after typical expense deduction, effective F&O rate lands nearer {eff:.1%} than {SLAB_TOP:.1%}")
