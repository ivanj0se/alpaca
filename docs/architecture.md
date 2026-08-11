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
`events/hawkes.py` replicates this on SPY before anything else in the ladder
is trusted -- if the pipeline can't reproduce a published number, nothing
built on top of it should be believed either.

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
