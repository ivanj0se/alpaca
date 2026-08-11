# Finding: GARCH and the factor model aren't on the same comparison axis

Date: 2026-08-11
Data: real AAPL + SPY minute bars (30 days), purged 6-fold CV, 60-min embargo.

## What happened

First walk-forward comparison of Rung 2a (GARCH) vs 2b (factor model) on
real AAPL data showed the factor model with a much better (lower) mean NLL:

| Rung | Mean NLL | Std NLL |
|---|---|---|
| factor_model | -5.75 | 0.17 |
| garch | -5.04 | 0.03 |

Naive reading: "the factor model beats GARCH." That's not the right
conclusion -- the two models are not answering the same question, so ranking
them against each other is comparing apples to oranges.

## Why

GARCH's `mean_nll` scores `Var(return_t | past returns)` -- forecasted
using only information strictly before `t`.

The factor model's `mean_nll` scores `Var(return_t - alpha - beta *
market_return_t)` -- and `market_return_t` is SPY's return at the *same*
timestamp `t` as the AAPL return being scored. The factor model has access
to contemporaneous market information GARCH doesn't get. Since AAPL and SPY
co-move (beta ~0.4 measured earlier), residualizing against a
contemporaneous factor mechanically shrinks the scored variance (Var of an
OLS residual is <= Var of the raw series whenever beta != 0) -- of course
its NLL looks better, independent of which model actually captures more
real structure.

This isn't a bug to fix; it's a difference in what the two models are for:

- **GARCH (temporal lane)**: forecasts one instrument's own future
  behavior from its own past. A legitimate walk-forward prediction.
- **Factor model (cross-sectional lane)**: explains one instrument's
  contemporaneous move using other instruments' contemporaneous moves.
  Explicitly NOT a forecast -- it's the same kind of contemporaneous,
  cross-sectional decomposition Rung 3 (Robust PCA) and the SSA project's
  population/cross-sectional track both do. Useful for the
  endogenous/exogenous question (how much of AAPL's move at time t is
  explained by the market at time t), useless for actually predicting
  AAPL's return ahead of time.

## Design change

The ladder is two parallel lanes, not one ranked list, matching the
two-track pattern already used in the SSA project (reconstruction track vs.
population/cross-sectional track) and already planned for Rung 3/4 in
docs/architecture.md:

- **Temporal lane**: Rung 0 (random walk) -> Rung 2a (GARCH) -> Rung 4
  (TCN-VAE reconstruction). Each must beat the previous *within this lane*.
- **Cross-sectional lane**: Rung 2b (factor model) -> Rung 3 (Robust PCA).
  Each must beat the previous *within this lane*.

`benchmark/ladder.py`'s `evaluate_rung`/`gate_check`/`save_ladder_report`
stay fully generic (they don't know or care which lane a rung belongs to);
what changes is which comparisons get treated as meaningful. GARCH vs.
factor-model NLL is reported side by side for visibility but never gated
against each other.
