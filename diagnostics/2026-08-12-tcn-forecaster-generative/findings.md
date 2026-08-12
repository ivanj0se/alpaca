# TCNForecaster autoregressive generator: a real logvar-clamp bug, and an honest negative result

Date: 2026-08-12

Rung G3 of the market-generator comparison suite
(`/Users/ivanpaiewonsky/.claude/plans/fuzzy-prancing-meteor.md`):
`models/tcn_forecaster.py` reuses `TCNEncoder` for genuinely
autoregressive one-step-ahead generation (see that module's docstring
for why `TCNVAE.decode()` can't be repurposed for this).

## Bug found and fixed: logvar clamp range blindly copied from a different model

Trained on real SPY data (`window_len=45`, `dilations=(1,2,4,8)`, 30
epochs, early-stopped): predicted std on a real, unmodified window was
**pinned at exactly 0.049787 across every single position** (min==max to
5 decimal places). That number is `exp(0.5 * -6)` -- exactly the lower
bound of the `[-6, 6]` logvar clamp I'd copied from `models/tcn_vae.py`
without checking whether it fit this model's actual data scale.

Real minute-bar SPY log_return has std~0.00035-0.00046 (log-variance
~-15.9); even the single largest real move in the whole dataset
corresponds to log-variance~-10.4. The model was correctly trying to
predict a tiny variance and getting silently floored at `exp(-6)`'s
implied std -- ~140x too large -- on every prediction, real or
generated. TCNVAE's `[-6,6]` range made sense in *that* model's
KL-divergence-numerical-stability context; nothing carried the
assumption over to a Gaussian-NLL forecaster on raw small-scale returns,
and nothing checked it. Fixed to `[-20, 0]`, comfortably covering the
real range with margin on both sides (std from ~4.5e-5 to 1.0). Added a
regression test (`test_logvar_clamp_covers_real_minute_bar_return_scale`)
and widened the clamp-bounds test itself, which had been asserting the
*wrong* (too narrow) range and would never have caught this on its own.

Effect of the fix, same real window: predicted std range corrected from
a pinned 0.049787 to 0.003938-0.009535 -- variable across positions now
(the model is actually conditioning on input again, not just returning a
clamped constant), and much closer to real scale, though not fully
calibrated (~9-21x too large, down from ~140x). Not chased to full
convergence this pass -- flagged honestly below, not as a resolved
scale-matching problem.

## Seed-sensitivity / exposure-bias check: passes cleanly

Free-running generation has no real anchor after the seed window (unlike
`TCNVAE`'s reconstruction, permanently bounded by real per-position
`h_seq`) -- errors can in principle compound step to step. Checked 5
seeds over a 390-step horizon (one trading-day equivalent): overall_std
ranged 0.0036-0.0040 across seeds, early-window vs. late-window std ratio
stayed within 0.89-1.12 for all 5 -- no drift, blowup, or collapse over
the horizon. This part of the design works as intended.

## Stylized-facts result: an honest 0% overall, informative by contrast

25 independent 1,950-step (5-trading-day-equivalent) realizations,
scored against real SPY stylized facts with the identical
block-bootstrap-calibrated methodology used for the Hawkes ablation
(`diagnostics/2026-08-12-hawkes-jump-diffusion-generator/`):

| Fact | Coverage | Mean distance | Threshold |
|---|---|---|---|
| raw_return_acf | 0% | 0.01958 | 0.00850 |
| volatility_clustering_acf | 0% | 0.20919 | 0.07254 |
| excess_kurtosis | 0% | 14.71274 | 4.95632 |
| leverage_curve | 0% | 0.02449 | 0.01141 |
| aggregational_kurtosis | 0% | 9.07310 | 3.26948 |
| **overall** | **0%** | -- | -- |

Two of these distances are worth reading precisely, not just as
failures: `excess_kurtosis`'s mean distance (14.713) is almost exactly
equal to the real reference's own excess kurtosis (14.71 -- see
`diagnostics/2026-08-12-stylized-facts-module-validation/`), meaning the
synthetic series' kurtosis is close to zero -- i.e. close to Gaussian,
no fat tails at all. `volatility_clustering_acf`'s mean distance (0.209)
is likewise almost exactly the reference's own mean ACF value (0.2094),
meaning the synthetic series shows essentially *no* volatility
clustering. Both are the expected consequence of the same root cause: at
each ancestral step the model samples independently from
N(mean, exp(logvar)), and this training run's predicted variance stays
close to constant across a generation window (consistent with the
horizon-stability check above finding no drift) rather than genuinely
conditioning on recent realized volatility the way real markets do --
so the marginal distribution of a long generated path ends up close to a
single fixed-variance Gaussian, which is definitionally flat-ACF and
thin-tailed.

## Comparative read (informative for the eventual full comparison report)

The Hawkes jump-diffusion generator's *control* arm (zero self-excitation)
also scored 0% overall -- so on this first pass, `TCNForecaster` performs
comparably to a generator with *no* self-exciting structure at all,
while the Hawkes *treatment* arm (real self-excitation) reached 68%
coverage on volatility clustering specifically. That's a genuinely useful
comparative data point for the final "compare them all" report: a simple
one-step Gaussian-NLL autoregressive model, trained this way, does not
on its own learn to reproduce temporal volatility clustering during
free-running generation, whereas explicitly modeling point-process
self-excitation does. Not a claim that autoregressive neural generation
can't work here -- a richer loss (e.g. conditioning explicitly on
realized volatility as a target, not just an input feature; a
heavier-tailed output distribution; more training epochs/data) would be
the natural next iteration, but out of scope for this first pass per the
build-order plan (get the harness validated against multiple real arms
first, refine individual generators after the comparison exists).

## Not fully resolved

The ~9-21x variance-scale mismatch after the clamp fix wasn't chased to
full convergence -- more epochs, a learning-rate sweep, or a different
patience setting might close more of that gap, but doing so wasn't
necessary to get a real, honestly-measured, apples-to-apples comparison
point for this generator against the others. Flagged as a legitimate
follow-up, not silently left unmentioned.
