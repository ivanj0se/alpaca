# Zero-intelligence agent-based baseline: the expected clean negative result

Date: 2026-08-12

**CORRECTION (2026-08-13):** the "0.000 overall" result below was
computed against calibration bands with a real length-mismatch bug -- see
`diagnostics/2026-08-13-conformal-band-length-mismatch/findings.md`.
Corrected: **overall_score=0.400** (raw_return_acf and leverage_curve
now pass at 1.00 for every generator tested, not specific to this one --
those two facts have low discriminating power at this sample length;
volatility_clustering_acf, excess_kurtosis, and aggregational_kurtosis
all still fail at 0.00, matching `gbm_null`/`hawkes_control` exactly).
The core argument below -- this arm is indistinguishable from the other
structure-free arms and clearly loses to the Hawkes treatment arm -- is
unaffected and still holds.

Rung G1 of the market-generator comparison suite
(`/Users/ivanpaiewonsky/.claude/plans/fuzzy-prancing-meteor.md`), built
last and deliberately kept the simplest of all four generator arms: a
discrete-time linear-impact rule in the Gode & Sunder (1993)
"zero-intelligence trader" spirit. `n_agents=200` independently emit a
buy(+1)/sell(-1)/no-order(0) signal each step (`order_prob=0.05`,
i.e. ~5% chance of a nonzero order per agent per step); log-return is a
linear function of the net order imbalance plus i.i.d. Gaussian noise.
No limit order book, no matching engine, no bid/ask spread, and --
deliberately, this is the point -- no herding or volatility-feedback
mechanism. Every step is i.i.d. by construction.

## Calibration

`impact_lambda` calibrated so the model's unconditional return std
matches real SPY data's, the same vol-matching spirit as
`baselines/random_walk.py::estimate_gbm_params` (there's no real
order-imbalance series to regress impact against directly). Target
variance split evenly between the imbalance-driven component and pure
noise -- an explicit, documented 50/50 choice, not claimed to be "the"
correct decomposition.

## Result: 0.000 overall, exactly as predicted

| Fact | Coverage | Mean distance |
|---|---|---|
| raw_return_acf | 0% | 0.01900 |
| volatility_clustering_acf | 0% | 0.20965 |
| excess_kurtosis | 0% | 14.66672 |
| leverage_curve | 0% | 0.02553 |
| aggregational_kurtosis | 0% | 8.99640 |
| **overall** | **0.000** | -- |

Matches `hawkes_control` (also 0.000) and `gbm_null` (also 0.000) almost
exactly across every fact -- three structurally different constructions
(pure Poisson-driven jump-diffusion, a plain vol-matched random walk, and
this discrete-time agent model) converge on the same near-total failure
to reproduce real markets' stylized facts, because all three share the
one property that actually matters here: no genuine temporal
dependence/self-excitation in the underlying process. `volatility_clustering_acf`'s
mean distance (0.2097) is, like the other two i.i.d.-by-construction
arms, almost exactly the real reference's own mean ACF value (0.2094) --
the synthetic series shows essentially zero clustering, confirmed
directly in unit tests
(`test_zero_intelligence.py::test_no_volatility_clustering_by_construction`).

## Why this is a useful result, not a wasted build

This is the cleanest possible confirmation that realistic market
statistics require genuine temporal structure, not just "a plausible
mechanistic story." A reader could reasonably wonder whether the Hawkes
treatment arm's success (68% volatility-clustering coverage, see
`diagnostics/2026-08-12-hawkes-jump-diffusion-generator/`) was really
about self-excitation specifically, or just about *any* multi-agent
mechanistic construction looking more realistic than a bare random walk.
This rules that out directly: a mechanistically-motivated, multi-agent,
calibrated-to-real-scale construction with *no* self-exciting/persistent
structure performs identically to the least-sophisticated arm in the
whole comparison (plain GBM). The active ingredient in the Hawkes arm's
result is demonstrably the self-excitation itself, not "having agents"
or "having a plausible economic story."

## Not pursued further

A richer agent-based model (order persistence, momentum/mean-reversion
sub-populations, a real limit order book) would very plausibly close some
of this gap -- explicitly out of scope for this pass per the build plan,
which scoped this arm as the simplest-possible baseline specifically to
get this clean contrast, not as an attempt at a competitive generator.
