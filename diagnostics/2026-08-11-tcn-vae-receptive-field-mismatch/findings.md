# Finding: TCN-VAE lost to GARCH on all 20/20 tickers at the real window_len

Date: 2026-08-11
Data: first full-universe run of scripts/run_ladder.py (20 tickers, 90-day
real data, 15 epochs, window_len=90 per config/settings.yaml).

## What happened

Temporal lane report: GARCH beat the flat null on 20/20 tickers (the
walk-forward fix from diagnostics/2026-08-11-garch-static-vs-walk-forward/
holding up at full scale), but TCN-VAE beat GARCH on **0/20** tickers --
consistently, by a small but nonzero margin on every single ticker (e.g.
AAPL: GARCH -5.8532 vs TCN-VAE -5.6136). A 20/20 sweep in one direction is
too consistent to be noise; this is a real, systematic effect, not a fluke
on one or two names.

Note this differs from the earlier isolated AAPL validation
(diagnostics/2026-08-11-tcn-vae-nll-scale-invariance/), which showed
TCN-VAE beating GARCH -- that test used `window_len=30`, not this run's
`window_len=90` (config/settings.yaml's actual default). The two runs
aren't the same comparison.

## Likely root cause

`models/tcn_vae.py`'s `TCNEncoder` uses `dilations=(1, 2, 4)` with
`kernel_size=3`. Receptive field of a causal dilated-conv stack:
`1 + sum((kernel_size-1) * dilation)` = `1 + 2 + 4 + 8` = **15 timesteps**.
The configured `window_len` is **90** -- a 6x mismatch. Each position in
the encoder's output can only "see" ~15 timesteps of true local context;
`encode()` then mean-pools across all 90 positions into a single latent
vector, but that pooling averages together many representations that each
only capture local structure, not genuine window-length dependencies. The
model is structurally underpowered for the window size it's being asked
to reconstruct.

This is a plausible, well-founded explanation (receptive field arithmetic
is exact, not speculative), but is **not yet directly verified** by
re-running with a wider receptive field -- flagged as the first thing to
try, not confirmed as the fix.

## Not yet done (deliberately, given time of night -- next session)

- Re-run with dilations giving a receptive field closer to (or covering)
  window_len -- e.g. `(1, 2, 4, 8, 16)` on the encoder (receptive field
  1+2+4+8+16+32=63, still short of 90 but much closer) or add a block.
- Alternative: keep the small receptive field but reduce `window_len` to
  match it more closely, if 90 minutes of context isn't actually necessary
  for the reconstruction task.
- Either way, re-validate on AAPL alone first (cheap) before re-running
  the full 20-ticker sweep.

## Not a correctness bug

Both sanity checks that would catch a broken NLL pipeline passed cleanly:
GARCH beat the null 20/20, and RPCA beat the factor model. The scoring
methodology itself (validated extensively in
diagnostics/2026-08-11-tcn-vae-nll-scale-invariance/) is sound; this is a
model-capacity/architecture-tuning question, not a repeat of the earlier
scale-invariance bug.
