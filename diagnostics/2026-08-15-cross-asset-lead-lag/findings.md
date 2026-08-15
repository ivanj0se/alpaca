# Cross-asset lead-lag: mostly noise, no coherent or tradable structure

Date: 2026-08-15

Fourth signal source tested in this exploration (after leverage-effect
timing, news tone, and order-flow imbalance -- see
`diagnostics/2026-08-15-leverage-signal-tradability-check/` and
`diagnostics/2026-08-15-alternative-signal-sources/`). Question: does
SPY's return predict an individual large-cap constituent's SUBSEQUENT
return (or vice versa) at some real lag -- real microstructure
literature documents broad/liquid instruments sometimes leading
individual names as new information gets priced into the most-traded
instrument first.

## Setup

Real minute bars (`data/`, ~90 days, 2026-05-13 to 2026-08-07 -- far
more real history than the ~3.3 days of tick data available for names
other than SPY, and already on the regular time grid a cross-correlation
analysis needs). SPY vs. four large-cap names (AAPL, MSFT, NVDA, GOOGL),
cross-correlation at lags 1-10 minutes in both directions
(~22,700-22,800 aligned real observations per pair).

## Real result: much weaker and less coherent than the other three signals

Contemporaneous (lag=0) correlations were large and unsurprising
(0.36-0.64 -- the already-established shared market factor). The
lead-lag correlations themselves were all tiny (|r| < 0.023 across all
4 pairs and all 10 lags) -- smaller in magnitude than even
order-flow imbalance's weak full-sample average (r=0.0137).

More importantly, the pattern across pairs was **inconsistent, not just
weak**:

| Pair | Strongest lag | Direction | corr | p |
|---|---|---|---|---|
| SPY vs AAPL | 3 min | SPY leads | -0.0215 | 0.0012 |
| SPY vs MSFT | 1 min | SPY leads | +0.0178 | 0.0073 |
| SPY vs NVDA | 1 min | tied (both directions ~equal) | +0.0189 / +0.0188 | 0.0044 / 0.0046 |
| SPY vs GOOGL | 6 min | peer leads | -0.0224 | 0.00076 |

A genuine information-diffusion lead-lag effect would show a consistent
direction (the same instrument leading across multiple names) and a
coherent decay pattern (strongest at short lags, fading at longer ones).
Neither shows up here: the leading direction flips between pairs (SPY
leads AAPL/MSFT, is tied with NVDA, is LED BY GOOGL), signs flip
unpredictably, and significant lags are scattered across the 1-10 minute
range with no decay shape.

**Multiple-comparisons check, done explicitly rather than glossed over:**
80 total tests (4 pairs x 10 lags x 2 directions) at alpha=0.05 would
produce ~4 false positives by pure chance alone. 17 cells actually
cleared p<0.05 -- more than chance alone predicts, so this likely isn't
*purely* noise, but the total lack of a coherent, interpretable pattern
means whatever real structure exists here (if any) can't be responsibly
described as "SPY leads by N minutes" or turned into a trading signal
with a defensible sign and horizon.

## Bottom line

The weakest and least interpretable of the four signals tried today.
Not clearly tradable -- there isn't even a clean, single directional
claim to test for tradability the way the other three allowed. Whatever
is producing the above-chance count of significant cells is more likely
ticker-idiosyncratic microstructure noise (e.g. bid-ask bounce
accumulated differently into each name's minute bars) than genuine
cross-asset information diffusion, at least at this bar granularity and
lag range.

## Consolidated picture across all four signals tried today

| Signal | Best gross effect | Net of assumed costs | Verdict |
|---|---|---|---|
| Leverage-effect timing | 0.068 bps/trade | -1.93 bps | real, ~30x short |
| News tone | ~0 (p=0.93) | -- | clean null |
| Order-flow imbalance | 0.11 bps/trade (top decile) | -1.89 bps | real, ~18x short, best of the four |
| Cross-asset lead-lag | <0.02 corr, inconsistent direction | -- | too weak/incoherent to even test |

Same conclusion as before, now with four independent confirmations
rather than one: real structure exists in parts of this data, but
consistently falls at least an order of magnitude short of what
realistic trading costs require, and the structure that IS found doesn't
compound across signal types into something bigger -- each one is its
own small, mostly-independent effect, not pieces of one larger hidden
signal.
