# Tier 3: multi-kernel Hawkes beats the single-exponential baseline; Cox-Hawkes/RPCA helps but underperforms it

Date: 2026-08-14

Tier 3 of the private self-excitation research extension: does modeling
the real structure this project's own extensions found (multiple
self-excitation timescales, Tier 1; a real exogenous RPCA baseline, Tier
4) produce measurably MORE realistic synthetic SPY paths than the
already-scored single-exponential Hawkes treatment arm (overall_score
0.896 in the original market-generator comparison suite, see CLAUDE.md)?
Scored all four arms through the exact same shared harness
(`benchmark/generator_ladder.py::evaluate_generator`, the same
Cont-2001-style stylized-facts/block-bootstrap-calibrated-band machinery
that produced that original result), refit consistently on the same real
IEX tick source (`data/`) for a genuinely controlled comparison.

## Real result

| Arm | overall_score | raw_return_acf | vol_clustering_acf | excess_kurtosis | leverage_curve | agg_kurtosis |
|---|---|---|---|---|---|---|
| hawkes_multi_kernel | **0.904** | 0.88 | 0.84 | 1.00 | 0.84 | 0.96 |
| hawkes_treatment (baseline) | 0.864 | 0.72 | 0.96 | 1.00 | 0.64 | 1.00 |
| cox_hawkes_rpca | 0.520 | 0.68 | 0.12 | 0.64 | 0.60 | 0.56 |
| hawkes_control (no excitation) | 0.400 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |

**Multi-kernel Hawkes beats the single-exponential baseline** (0.904 vs
0.864) -- modeling real multiple self-excitation timescales (fitted
here: tau ~ [91s, 9.1s, 0.9s], IEX/`data/` tick data) produces a
measurably more realistic synthetic market than a single relaxation
timescale does, on the same real reference data and the same calibrated
bands. This is a genuinely new, positive finding -- Tier 1's original
question was about detecting a real timescale-structure difference
between venues; this shows that structure, once found, actually pays off
when used generatively, not just descriptively.

**Cox-Hawkes/RPCA clears the no-excitation control by a wide margin**
(0.520 vs 0.400) but underperforms both other self-exciting arms. It's
not a bad model -- it beats the naive control on every single fact -- it
just doesn't beat plain self-excitation at this sample size, and is
notably weak specifically on volatility clustering (0.12 coverage, its
worst fact by far).

## A real bug caught and fixed before trusting this result

The first run of this comparison gave a badly wrong Cox-Hawkes score
(0.336 -- WORSE than the no-excitation control) that looked like a
genuine, surprising negative finding. Didn't take it at face value --
checked per-path bar counts directly and found `cox_hawkes_rpca` paths
were 10,133 bars long against every other arm's 1,950. Root cause: two
compounding bugs in `generators/hawkes_extensions_generator.py`'s
`generate_cox_hawkes_paths` --

1. It computed `T_days` in CALENDAR seconds (86400/day) while every
   other arm in this same module and `generators/hawkes_jump_diffusion.py`
   uses SESSION-trading seconds (23400/day) for the identical parameter
   name -- a silent ~3.7x unit mismatch.
2. Its covariate-grid truncation rounded UP to "at least T_requested"
   (`searchsorted(...) + 1`) instead of clipping the final segment, so a
   single large gap in the real (irregularly-sampled) RPCA grid could
   push the actual simulated span well past the target.

This is the exact same bug CLASS already caught once in this project
(`diagnostics/2026-08-13-conformal-band-length-mismatch/`): scoring a
generator's paths against bands calibrated for a DIFFERENT path length
measures the length mismatch, not realism. Confirmed the fix directly --
`fit_real_cox_hawkes_params`'s real RPCA slice used for simulation is not
an unusual/outlier stretch of history (mean=-0.045, std=0.801 vs the
full grid's mean=0.084, std=1.012 -- checked this FIRST and ruled it out
before finding the actual bug), so the length mismatch really was the
whole story. Fixed by (a) using the module's own `SESSION_SECONDS_PER_DAY`
constant for consistency with every sibling function, and (b) clipping
the final included grid segment's duration so the simulated span always
lands exactly on the target, never over. Added a direct regression test
(`test_simulated_span_never_overshoots_the_requested_t_days`).

## Reading Cox-Hawkes's remaining gap (not fully resolved)

With the length bug fixed, Cox-Hawkes/RPCA genuinely underperforms
plain self-excitation, particularly on volatility clustering. One
plausible mechanism, not confirmed here: `generate_cox_hawkes_paths`
holds the real historical covariate trajectory FIXED and identical
across all 25 simulated realizations (a deliberate design choice --
`x(t)` is a real, externally-measured driver, not something to invent,
see the module's own docstring) -- only the point-process randomness
varies between realizations. The `overall_score` metric specifically
rewards an ENSEMBLE whose across-realization variability matches real
markets' own natural sampling variability (that's what the
block-bootstrap-calibrated band measures). A fixed covariate path could
structurally suppress exactly that kind of ensemble diversity in a way
neither the constant-mu treatment arm (no covariate to fix) nor the
multi-kernel arm (also constant-mu) is vulnerable to. Not tested directly
here -- would need an ensemble that resamples different real historical
covariate WINDOWS per realization (not just different random seeds) to
check.

## What this means for the tiers overall

Two extensions, two different real answers: multi-timescale
self-excitation (Tier 1) generatively outperforms the already-shipped
baseline; the RPCA exogenous baseline (Tier 4) has real detective value
(the 2026-08-13 diagnostics entry's gamma=-0.1396/loglik+95.62 result)
but so far weaker generative value, likely for a structural reason (fixed
real covariate path) rather than the underlying idea being wrong. Both
honest, useful results -- not every real extension needs to be a clean
win to be worth having built and measured.
