# Permutation-count stability check (n_permutations=2000)

Date: 2026-08-12

Same audit spirit as diagnostics/2026-08-12-tcn-vae-seed-sensitivity/: a
permutation test's p-value is itself a Monte Carlo estimate with its own
sampling noise, and `attribution/null_control.py`'s default
`n_permutations=2000` had never been checked for how much that noise
matters near a decision boundary.

## Method

Real AAPL GDELT news (42 events, from the project's cached data) with 3
synthetic anomaly timestamps (2 placed near real news events, 1 random --
chosen to land the p-value in a decision-relevant middle range rather
than a trivial 0 or 1). Ran `permutation_test` 30 times with different
seeds, same inputs, `n_permutations=2000` each time.

## Result

p-value: mean=0.0423, std=0.0039, range 0.0335-0.0480 across the 30 runs.

A real but modest spread (~0.015), and notably it straddles the common
alpha=0.05 threshold in principle -- with std=0.0039, a true p-value
sitting right at ~0.05 could plausibly land on either side of
significance in a single 2000-permutation run, purely from Monte Carlo
noise.

## Why this isn't an active problem

None of the actual production Rung 5 results are anywhere near this
boundary: after Sidak correction across the universe, the closest ticker
to significance was p_sidak~0.88-0.90 (see
diagnostics/2026-08-11-full-ladder-run/report.md) -- multiple orders of
magnitude away from where 2000-permutation noise could matter. Not fixing
this now (bumping n_permutations to 5000-10000 for a wider safety margin
would be cheap if a future ticker ever does land close to the 0.05
boundary, but doing it pre-emptively for a scenario that hasn't occurred
in any real run isn't worth the extra compute right now).
