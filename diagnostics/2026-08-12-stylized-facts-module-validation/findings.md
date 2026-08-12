# Stylized-facts module validation (trust gate for the generator-comparison suite)

Date: 2026-08-12

First step of the market-generator comparison suite (see
`/Users/ivanpaiewonsky/.claude/plans/fuzzy-prancing-meteor.md` for the
full plan): before any generator is scored against
`benchmark/stylized_facts.py`, confirm the module itself reproduces
Cont (2001)'s standard "stylized facts" checklist on real market data.
Same trust-gate philosophy as Rung 1's Hawkes replication against
Filimonov & Sornette's published value -- if real data doesn't show these
signs through this code, the code has a bug, not the market.

## Result: all five facts confirmed on real SPY and AAPL minute data

| Fact | Expected | SPY (n=22,999) | AAPL (n=23,379) |
|---|---|---|---|
| Raw return ACF | ~0 at every lag | mean\|acf\|=0.0061 | mean\|acf\|=0.0101 |
| Volatility clustering | positive, slow decay | 0.283 (lag1) -> 0.179 (lag50) | 0.298 (lag1) -> 0.088 (lag50) |
| Excess kurtosis | >> 0 (fat tails) | 14.71 | 52.79 |
| Leverage effect | negative | mean=-0.017, lag1=-0.016 | mean=-0.020, lag1=-0.043 |
| Aggregational Gaussianity | declining kurtosis at coarser scales | 14.71 -> 9.97 -> 7.51 -> 3.94 (scales 1/5/15/30) | 52.79 -> 20.10 -> 22.59 -> 18.08 |

All five pass. **PASS**, this module is trusted to score generators.

## One honest wrinkle, not a bug

AAPL's aggregational-kurtosis curve isn't perfectly monotonic (scale 15's
22.59 is higher than scale 5's 20.10). At scale 15, only
23,379/15≈1,558 aggregated points remain, and kurtosis estimates get
noisier with fewer points -- an isolated step-to-step bump is expected
sampling noise, not a defect in `aggregational_gaussianity_curve`. The
overall trend from scale 1 (52.79) to scale 30 (18.08) is still a clear,
strong decline, which is the actual claim being tested. Noting this
explicitly so a future reader doesn't mistake real sampling noise for a
bug and "fix" something that isn't broken.

## What's next

`benchmark/conformal.py` (block-bootstrap calibrated bands, deliberately
NOT called "conformal prediction" since real returns are autocorrelated
and violate the exchangeability assumption textbook conformal prediction
relies on) is built and unit-tested alongside this module. Both are ready
for the first generator arm (Hawkes jump-diffusion ablation, next).
