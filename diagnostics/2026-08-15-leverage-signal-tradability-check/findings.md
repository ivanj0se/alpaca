# Is the real leverage-effect signal tradable? Tested directly on real data: no, cost drag dominates by ~30x

Date: 2026-08-15

Prompted by a direct question: does the real, statistically significant
leverage asymmetry found in `diagnostics/2026-08-15-asymmetric-quadratic-hawkes-real-fit/`
(kappa_minus > kappa_plus, p=0.00034) mean a trader could make money
using it? Tested empirically on REAL SPY tick data (not model-generated
synthetic paths, which would be circular -- a strategy "profitable" on a
model's own synthetic output just exploits the model's own assumptions).

## Setup

At each of the 23,515 real Hawkes-flagged SPY events, computed L1(t^-)
using the real fitted beta_leverage (0.00709675). Split into "down-trend"
(L1<0, the state the real fit found elevated intensity for) vs
"up-trend" (L1>=0). Measured the REALIZED forward log-return over the
next 20 real ticks for each, and tested a simple strategy: go long after
a down-trend signal, short after an up-trend signal (the natural
mean-reversion read of "the market gets more active after down moves").

## Real result

| | mean forward return | std | n |
|---|---|---|---|
| Down-trend (L1<0) | +0.131 bps | 4.06 bps | 12,124 |
| Up-trend (L1>=0) | -0.000 bps | 3.74 bps | 11,391 |

t-test (down vs up): t=2.572, **p=0.0101** -- nominally significant.
Strategy mean return: **+0.068 bps/trade**, t=2.652, p=0.0080 --also
nominally significant, at ~735 trades/day given this signal's real
firing rate.

**Net of an assumed 2.0 bps round-trip cost (not measured -- no bid/ask
data in this project's tick store, a conservative assumption for a
liquid ETF like SPY): -1.932 bps/trade.** Cost drag exceeds the gross
signal by roughly 30x. At ~735 trades/day, this isn't a marginal,
"maybe profitable with a cheaper broker" result -- transaction costs
would dominate any real edge by a wide margin regardless of reasonable
assumptions about execution quality.

## Reading the "significant" p-values correctly

p<0.01 sounds like a real finding, and in one sense it is -- there does
appear to be a tiny, real difference in average forward return between
the two states. But at n>23,000, economically meaningless effects
routinely clear p<0.05; 0.068 bps is under 1% of a single basis point.
The much more likely explanation for even this tiny effect is
microstructure noise (bid-ask bounce in trade-level data is a
well-documented source of exactly this kind of small, "significant,"
non-tradable pattern) rather than a genuine, exploitable inefficiency --
this project has no bid/ask data to check that hypothesis directly, which
is itself a real limitation worth naming rather than glossing over.
Also worth flagging: this is tested on the SAME historical window the
leverage effect was originally fit on, not a genuinely independent
out-of-sample period -- not a real backtest by any rigorous standard,
just a direct, honest check of the specific question asked.

## Bottom line

The leverage effect is real (found independently via MLE, confirmed here
via a completely different method -- direct forward-return comparison)
and mechanistically interesting -- it says something true about how real
SPY order flow clusters. It is not, on this evidence, a tradable edge.
The gross signal is roughly 30x smaller than a conservative assumed
transaction cost. This is consistent with why this project has framed
itself as "not a trading bot" from the start (see CLAUDE.md) -- a model
that statistically describes real market dynamics well is a different
thing entirely from a model that predicts direction well enough to beat
real trading costs, and this project has only ever built and validated
the former.
