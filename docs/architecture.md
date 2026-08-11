# Architecture

## Why not just PCA

PCA assumes the informative structure in the data is linear. There's decent
evidence against that for markets: volatility clustering (GARCH-type
conditional heteroskedasticity) is itself a nonlinearity linear models can't
produce, and cross-asset correlations spike during crises -- meaning a linear
basis fit on calm data is least valid exactly during the events this project
cares about. Robust PCA (`M = L + S`, low-rank + sparse) is the linear-but-honest
first step: `L` is the common/systematic/"endogenous market-wide" component,
`S` is the per-ticker idiosyncratic residual. A small TCN-VAE is the nonlinear
upgrade path once the linear baseline is validated.

## Two tracks (reused pattern)

1. **Cross-sectional decomposition** (`rpca/`) -- Robust PCA across the
   basket at each time step separates common movement from idiosyncratic
   residual.
2. **Nonlinear reconstruction** (`models/`) -- TCN-VAE trained on windowed
   price/volume only, no news/fundamentals. Reconstruction residual is the
   anomaly score.

Both feed the same downstream question: is a given window's activity
"explained by the market's own recent history," or not?

## The trust gate: Hawkes branching ratio

Filimonov & Sornette (2012, 2015) quantified market self-excitation via a
Hawkes-process **branching ratio**: the fraction of price-changing events
that are self-triggered vs. exogenous, ~0.81 on E-mini S&P 500 tick data.
`events/hawkes.py` implements this from scratch (Ozaki O(N) recursive MLE,
exact cluster-representation simulator) and validates itself synthetically
before touching real data: simulate a known process, refit, confirm
recovery (tests/unit/test_hawkes.py).

**On real SPY minute-bar data, the branching ratio comes out near zero --
this is the correct, expected result, not a failure.** Minute bars are
close to a regular grid (Fano factor 0.82, *under*dispersed), the opposite
of the bursty overdispersion a self-exciting process produces; the
self-excitation Filimonov & Sornette measured operates at the tick/trade
timescale (milliseconds-seconds), which 1-minute binning destroys entirely.
Verified this isn't a fitter bug by injecting deliberate bursts into an
otherwise-regular synthetic sequence and confirming the fitter recovers a
nonzero alpha. Full writeup:
diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/findings.md.

The actual "does this reproduce ~0.81" check
(tests/replication/test_hawkes_branching_ratio_replication.py) runs against
real accumulated ticks from ingest/tick_recorder.py, and skips gracefully
until there's enough (>= 5,000 ticks over >= 5 days for SPY) -- it's a real
test that will start actually checking something once the recorder has run
long enough, not a placeholder.

## Attribution, not prediction

The model never sees news. `attribution/` correlates high-residual windows
against real GDELT news events *after* scoring, using a permutation/null-control
test (shuffle news timestamps, compare the real match rate against the
shuffled distribution) rather than presenting raw correlation numbers as
precise -- GDELT has no tickers, so ticker matching is heuristic. This
inversion (news as a post-hoc attribution variable, never a model input) is
the point: most existing work uses news sentiment as a prediction *input*,
which answers a different question than "how much of this is the market
talking to itself."

## Non-goals (for now)

No brokerage/order execution. No live *decisioning* pipeline -- only
`ingest/tick_recorder.py` runs continuously, and it only records (needed
because free historical tick-level data doesn't exist; the bar-threshold
proxy is used until enough real ticks accumulate).
