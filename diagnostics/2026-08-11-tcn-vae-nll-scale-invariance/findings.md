# Finding: NLL is not scale-invariant -- two separate bugs, not one

Date: 2026-08-11
Data: real AAPL minute bars (30 days), 30-min windows, stride 5, purged
4-fold CV, 30-min embargo.

## Summary

Getting Rung 4 (TCN-VAE) to a fair comparison against Rung 2a (GARCH) in
the temporal lane took three iterations, and the first two "fixes" were
each real but insufficient on their own.

## Attempt 1: naive pooled NLL across 3 raw-scale features

`make_fold_scorer` trained on [log_return, realized_vol, volume_zscore]
jointly and computed one pooled Gaussian NLL across all three. Result:
TCN-VAE mean NLL = **+0.88**, GARCH mean NLL = **-4.99** (lower is better)
-- TCN-VAE looked much worse.

Root cause: real feature scales are wildly different --
`log_return` std ~0.00095, `realized_vol` std ~0.00065, `volume_zscore`
std ~1.04 (already z-scored by construction). Pooling them into one
variance for both the training MSE loss and the NLL means volume_zscore
(≈1000x larger scale) dominates both -- the model effectively learns to
reconstruct volume_zscore at the expense of the tiny-scale return
features, and the reported variance/NLL reflects mostly that.

## Attempt 2: per-fold standardization, still pooled NLL

Fixed the scale-domination problem by standardizing each fold (fit on
train only, `standardize_windows`) before training. Result: TCN-VAE mean
NLL = **+1.31** -- *worse*, not better.

Root cause (different from attempt 1): NLL is not scale-invariant. A
Gaussian's log-density grows without bound as variance shrinks
(`-log N(0, var) ~ 0.5*log(var)` for small var), so a model scored on a
standardized (~unit-variance) target will *always* look far worse in raw
NLL terms than one scored on a naturally tiny-variance target like real
per-minute log returns (~1e-6 variance), regardless of which model is
actually better. Standardizing for training was still correct and
necessary (fixes attempt 1's problem), but doesn't fix comparability with
GARCH by itself.

## Fix: standardize for training, score only the un-standardized primary channel

`make_fold_scorer` now: (1) standardizes all channels for training, so the
loss isn't dominated by volume_zscore; (2) computes the final NLL using
only the log_return channel's residual, multiplied back by that channel's
train-fold std to restore raw scale. This puts TCN-VAE on the same target
(the instrument's own raw-scale log return) and the same units as GARCH,
while still letting TCN-VAE use realized_vol and volume_zscore as
additional *input* signal (still history-only, no lookahead -- a
legitimate advantage, not a leak).

Result: TCN-VAE mean NLL = **-5.50**, GARCH mean NLL = **-4.99** -- now on
comparable scale, and TCN-VAE wins, consistent with it having access to
richer own-instrument information (price *and* volume dynamics) than
GARCH's return-history-only view.

## Takeaway for future rungs

Any NLL-based comparison across the ladder needs both: (a) matching scale
for training (standardize/normalize per fold, fit on train only), and (b)
scoring the *same underlying raw-scale quantity* across rungs being
compared, not a standardized proxy for it. Both were necessary; neither
alone was sufficient. Worth checking this explicitly if a future rung's
comparison "looks wrong" in either direction before trusting the number.
