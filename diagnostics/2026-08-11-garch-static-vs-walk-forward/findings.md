# Bug: static GARCH forecast lost to the flat null on real data

Date: 2026-08-11
Found while: first full run of scripts/run_ladder.py (AAPL, real 90-day data).

## What happened

The temporal lane report showed `Rung2a (GARCH) beats Rung0 (flat null):
False` for both AAPL and MSFT, with a sizeable gap (Rung0 mean NLL -5.61,
Rung2a -4.51 for AAPL -- lower is better). GARCH modeling time-varying
volatility losing to "just use one constant variance for everything" is
backwards from the standard expectation, and worth investigating rather
than reporting as-is.

## Root cause

`make_fold_scorer` used `forecast_variance_path`, which projects a single
h-step-ahead variance path *once* from the end of the training fold and
holds it fixed across the entire test fold (h = however many rows are in
that fold). Diagnosed directly on a real AAPL fold (train n=20,215, test
n=4,044): GARCH persistence was **1.00003** -- essentially a perfect unit
root. At that persistence, the forecast barely decays toward any long-run
level at all; it just stays anchored near wherever volatility happened to
be at the exact moment training ended. On this fold, that meant the static
forecast ranged from 1.22e-06 to 4.31e-05 (a 35x span) while the *actual*
realized test-fold variance was 7.48e-07 -- the static forecast was
wildly, persistently too high for most of a 4,000-minute-long fold,
because it never got to react to what was actually happening during that
period.

Near-unit-root persistence isn't a fluke of this one fold -- it's the norm
for real intraday minute-bar GARCH fits (~0.98-1.0, measured repeatedly
across this project; see the near-IGARCH note in
diagnostics/2026-08-11-tcn-vae-nll-scale-invariance/'s neighbor
investigations). So this wasn't a one-off bad fold; it would have
depressed GARCH's apparent performance across the whole ladder.

## Fix

`forecast_variance_walk_forward` replaces the static path: at each test
point, applies the GARCH(1,1) recursion fed by the *actual* realized
return one step back -- already-known information by that point, not a
lookahead violation, and exactly how GARCH is used in real deployment
(update the variance estimate after every new observation, without
refitting). Only the very first test-point forecast uses training-fold
information (the last fitted conditional variance and return); every
step after that uses the test fold's own realized values as they're
"revealed."

## Validation

Same real AAPL fold: walk-forward variance range 1.04e-07 to 1.88e-05,
much closer to the realized 7.48e-07. Walk-forward GARCH mean NLL -5.82
vs. flat-null -5.63 -- **GARCH now correctly beats the null**, matching
the standard expectation that modeling volatility clustering should add
value over assuming one constant variance throughout.

## Takeaway

A model that's correct in principle (GARCH, well-fit, converged) can
still lose an evaluation for a reason that has nothing to do with the
model itself -- how it's *scored* over a walk-forward window matters as
much as the fit. Worth checking for any future rung that projects a
forecast forward over a multi-step horizon rather than updating on
realized values as they arrive.
