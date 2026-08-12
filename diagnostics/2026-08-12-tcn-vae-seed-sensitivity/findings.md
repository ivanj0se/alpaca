# Auditing other single-run results after the Hawkes multistart mistake

Date: 2026-08-12

Prompted by diagnostics/2026-08-11-sip-consolidated-tape-check/'s
CORRECTION section (a narrow optimizer-restart grid silently converged to
a wrong branching ratio, looked robust because all 4 starting points
agreed, wasn't): audited other places in the ladder for the same
structural risk -- a result reported from a single run of something with
randomness or multiple optima, never checked for stability.

## Checked and cleared

- **Every production Hawkes fit already goes through
  `fit_hawkes_exponential_multistart`** (grepped for direct
  `fit_hawkes_exponential(` calls outside `events/hawkes.py` itself --
  none found). No gap here.
- **GARCH `ConvergenceWarning: ... code 8` (seen in earlier ladder run
  logs) is real but inert.** Reproduced it directly: SPY, PurgedKFold
  folds 0 and 4 (n~19,100), `convergence_flag=8`. But `SPY` never goes
  through `run_temporal_lane` in production (`scripts/run_ladder.py`'s
  `tickers` list is the 20-ticker universe only; SPY is used solely for
  Rung 1 Hawkes and as the cross-sectional factor, neither of which fits
  GARCH). `baselines/garch.py::make_fold_scorer` also already checks
  `convergence_flag == 0` and skips the fold (returns `None`) otherwise --
  a real double-safety even if SPY's usage ever changes. No fix needed.
- **Robust PCA is structurally lower-risk than Hawkes here**: Principal
  Component Pursuit (nuclear norm + L1) is a convex problem, so Inexact
  ALM either converges toward the true global optimum or fails to
  converge -- there's no "multiple local optima" failure mode to begin
  with, unlike a 3-parameter nonlinear MLE. `rpca/rolling_rpca.py::make_fold_scorer`
  already checks `result.converged` and skips non-converged folds. No
  action needed.

## Checked and confirmed robust: TCN-VAE / GARCH win margin across random seeds

`models/train.py::TrainConfig.seed` defaults to 0 and every ladder run to
date has used exactly that one seed -- genuinely the same shape of risk
as the Hawkes bug (a single run standing in for "the answer" with no
check on whether a different starting point/seed would disagree),
just in a different part of the pipeline (SGD training instead of MLE
optimization). Neural net training is well known to be seed-sensitive in
general, so this deserved a direct check rather than an assumption either
way.

Tested 3 tickers x 3 seeds (0, 1, 42) -- 0 is the production default;
picked AMZN specifically because its seed=0 margin (2.2630 NLL) was the
smallest of any ticker in the 20-ticker production run, making it the
case most likely to flip sign if seed variance were comparable in size to
the margin. Also included BAC (largest production margin, 3.1298) and
MSFT (a mid-range case, 2.7239) for context.

| Ticker | seed=0 margin | seed=1 margin | seed=42 margin | range |
|---|---|---|---|---|
| AMZN | 2.2630 | 2.3334 | 2.3516 | 0.089 |
| BAC | 3.1298 | 3.0130 | 3.0595 | 0.117 |
| MSFT | 2.7239 | 2.7349 | 2.8281 | 0.104 |

(margin = rung2a_NLL − rung4_NLL; positive means TCN-VAE beats GARCH.)

TCN-VAE beat GARCH in all 9 runs, no exceptions. Seed-to-seed variance in
the margin is small (~0.09-0.12 NLL) relative to the margin itself
(2.26-3.13) -- roughly 20-30x smaller than even the tightest case's
margin. The production seed=0 run wasn't a lucky outlier in either
direction: for AMZN and MSFT it was actually the *smallest* margin of the
three seeds tested (if anything slightly understating the win), and for
BAC the largest (slightly overstating it), which is itself reassuring --
no consistent bias in one direction across tickers.

## Conclusion

No bug found here, unlike the Hawkes case. The "TCN-VAE beats GARCH on
20/20 tickers" finding is robust to random seed on the 3 tickers tested
(including the one with the smallest margin, the case most likely to
flip). Not exhaustively verified across all 20 tickers x many seeds --
that would cost a full extra ladder run's worth of CPU time for a
question this check already answers with reasonable confidence (small,
consistent variance on the most-at-risk case). Not recommending
`TrainConfig.seed` be swept for every future production run; this
one-time audit is enough evidence the existing seed=0 numbers aren't
fragile.
