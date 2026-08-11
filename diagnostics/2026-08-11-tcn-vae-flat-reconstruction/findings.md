# Fix: TCN-VAE reconstructions were structurally flat (skip connection added)

Date: 2026-08-11
Follow-up to diagnostics/2026-08-11-tcn-vae-receptive-field-mismatch/,
which flagged (but hadn't verified) a receptive-field explanation for
TCN-VAE losing to GARCH on 0/20 tickers.

## The real mechanism (receptive field was a red herring)

Directly measured the model's actual reconstructions on real AAPL data
before assuming anything: for 5 real test windows, reconstructed
log_return had std ~0.01-0.015 across the 90-timestep window, vs. the
*actual* data's std ~0.5-0.72 in the same windows -- a 50-70x gap. The
model was outputting an almost flat line regardless of what was in the
window.

Root cause, once measured: `TCNVAE.decode()` took a single pooled latent
`z` (one vector per window, from mean-pooling the encoder's per-timestep
output), broadcast it *identically* to all 90 timesteps, then ran it
through a translation-equivariant causal-conv decoder. A translation-equivariant
decoder given a constant input can only produce an
approximately constant output (up to boundary effects near the start of
the causal padding). No amount of encoder receptive-field tuning fixes
this -- the bottleneck itself structurally discards per-timestep
information before decoding. This is the well-documented "VAE
reconstructions are blurry/oversmoothed" failure mode from the wider VAE
literature (pure-bottleneck VAEs discard local detail); it happens to
apply here in the temporal dimension rather than the spatial dimension of
an image VAE.

## Fix

Added a skip connection: `encode()` now returns the encoder's per-timestep
hidden states (`h_seq`, shape (B, T, hidden_dim)) alongside the usual
pooled (mu, logvar). `decode(z, h_seq)` concatenates the broadcast global
latent with `h_seq` at every position and fuses them through a linear
layer before the decoder. `h_seq` is genuinely per-position (encoder's own
causal-conv output, not yet pooled), so this isn't a trivial identity
shortcut -- the decoder gets local detail *plus* the global-latent
"regime" summary, not a copy of the raw input.

## Validation (real AAPL data)

- Reconstructed log_return std now matches actual std almost exactly
  (ratio 0.99-1.0 across 5 test windows, vs. ~0.02 before).
- Per-position correlation between reconstruction and actual: 0.998+
  (genuine tracking, not just matched aggregate variance).
- Anomaly sensitivity preserved: an injected spike (added to 10 test
  windows) still scores 22x higher reconstruction error than normal
  windows -- the fix does not make it "too easy" to reconstruct anything,
  which was the real risk worth checking (skip connections can sometimes
  let an autoencoder cheat past anomalies too).
- Temporal-lane comparison on AAPL: TCN-VAE mean NLL -8.81 vs. GARCH -5.63
  vs. Rung0 null -5.60. TCN-VAE now clearly wins, a dramatic change from
  0/20 tickers before the fix.

## Honest caveat found while verifying causality

Not a new issue, but not previously stated plainly: the skip path (`h_seq`) is genuinely causal
per-position (verified directly -- unaffected by changes to later
positions in the same window), but the *global* latent (`mu`/`z`) pools
over the *whole* window by design and is therefore not position-causal --
confirmed directly (changing only late-window values changes `mu`). This
was already true before the skip-connection fix (pooling has always
looked at the whole window); it's just now stated precisely rather than
assumed. Practical implication: TCN-VAE's window-level reconstruction can
use "future-within-the-window" information that GARCH's genuinely
one-step-causal forecast never gets access to, so the temporal-lane NLL
comparison between them is an approximation of "which model captures this
instrument's structure better," not a strictly like-for-like forecasting
comparison. Defensible given Rung 4's actual purpose (retrospective/batch
window-reconstruction anomaly detection, not real-time forecasting), but
worth stating rather than leaving implicit -- especially now that
TCN-VAE's NLL margin over GARCH is large enough that this caveat matters
for interpreting *how much* better, not just whether.

## Next step

Re-run the full 20-ticker ladder (scripts/run_ladder.py) to confirm this
holds across the whole universe, not just AAPL.
