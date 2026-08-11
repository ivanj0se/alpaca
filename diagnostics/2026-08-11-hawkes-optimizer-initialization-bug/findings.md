# Bug: fixed beta0=1.0 made Hawkes fits unreliable on real (sparse) event data

Date: 2026-08-11
Found while: first real run of scripts/run_ladder.py on freshly backfilled
90-day SPY data.

## What happened

The Rung 1 bar-proxy fit reported `alpha=0.4999999309653881,
beta=1.0000000371241717` -- essentially identical to the optimizer's
default starting point (`alpha0=0.5, beta0=1.0`), with `converged=True`.
This is not a coincidence: the optimizer barely moved from its initial
guess and still reported success.

## Root cause

`fit_hawkes_exponential`'s default `beta0=1.0` implies a 1-second
self-excitation decay timescale. Real SPY bar-proxy events have a median
gap of 480 seconds. At beta0=1.0, `exp(-1.0 * 480) ~ 3.46e-209` --
numerically zero for essentially every event, which makes
`d(loglik)/d(alpha)` zero at the starting point too (the recursive sum
`R(i)` that alpha multiplies is itself ~0 everywhere). The optimizer had
no gradient signal to work with and stopped almost immediately, having
made no real progress toward an actual MLE optimum.

This also retroactively explains why two earlier runs on similar sparse
data gave *different*-looking results: the original 120-day fixture test
found alpha pinned at the lower bound (1e-10), while this fresh 90-day run
found alpha stuck near the initial guess (0.5) instead. Both are symptoms
of the same underlying problem -- a numerically flat likelihood surface
near a badly-scaled starting point -- producing an optimizer path that
isn't reliable or reproducible across different (but qualitatively
similar) datasets.

## Does this change the earlier underdispersion conclusion?

No -- it refines the evidence quality, not the conclusion.
diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/ established minute
bars are too regular for self-excitation via two *other*, independent
lines of evidence (Fano factor = 0.82, and a burst-injection sensitivity
check), neither of which depended on this specific optimizer path. This
bug means the direct "branching ratio ~ 0 on SPY" number from that
analysis wasn't fully trustworthy on its own terms, but the broader
conclusion it supported already had independent backing.

## Fix

`beta0` now defaults to `1 / median(inter-event gap)` when not explicitly
provided, rather than a fixed constant -- puts the optimizer where the
likelihood surface actually has curvature, regardless of the event
stream's natural timescale.

## Validation

1. Refit real SPY bar-proxy events with the fix: `alpha=1e-10` (pinned at
   the lower bound), `branching_ratio ~ 3.6e-13` -- essentially zero,
   converged.
2. Robustness check (the property that was actually broken): refit from 4
   different random `alpha0` starting points (0.1 to 2.0) -- all converge
   to `branching_ratio = 0.000000` exactly. Before the fix, different
   starting points could land in qualitatively different places (boundary
   vs. stuck-at-initial-guess) on similar sparse data.
3. Existing tests unaffected: the synthetic simulate-refit-recover
   self-test and the frozen-fixture replication test both still pass
   unchanged (`tests/unit/test_hawkes.py`,
   `tests/replication/test_hawkes_branching_ratio_replication.py`).
4. New regression tests added:
   `test_default_beta0_is_data_adaptive` (fixed beta0=1.0 would land
   suspiciously close to the exact starting point on sparse data) and
   `test_refit_agrees_across_different_alpha0_starting_points` (the core
   robustness property, directly encoding what this bug violated).

## Takeaway

MLE optimizers reporting `converged=True` is not sufficient evidence of a
real optimum -- worth checking gradient-scale sanity (does the starting
point put you somewhere the likelihood surface has curvature relative to
the actual data scale?) and multi-start agreement for any future estimator
added to the ladder, not just Hawkes.
