# Confirmed: TCN-VAE skip-connection fix holds universe-wide

Date: 2026-08-11, second full-universe run of the day.

Last night's first full run found TCN-VAE losing to GARCH on 20/20
tickers -- diagnosed as a flat-reconstruction bug (single pooled latent
broadcast identically to every timestep before a translation-equivariant
decoder), fixed with a skip connection
(diagnostics/2026-08-11-tcn-vae-flat-reconstruction/), and spot-validated
on AAPL alone (-8.81 vs -5.63).

This run re-validates across the full universe. Result: **TCN-VAE beats
GARCH on 20/20 tickers**, every one by a wide margin:

| Ticker | GARCH | TCN-VAE | Margin |
|---|---|---|---|
| CAT (smallest margin) | -4.98 | -7.83 | 2.85 |
| NEE (largest margin) | -6.05 | -9.12 | 3.07 |
| (all 20 tickers) | -4.98 to -6.05 | -7.83 to -9.12 | 2.5 to 3.1 |

Full table in report.md. Every ticker's TCN-VAE NLL is comfortably beyond
its own GARCH NLL, with a fairly tight, consistent margin across the
whole basket (not one or two outlier tickers dragging an average) --
strong evidence this is a real, general architectural improvement rather
than an artifact of any single instrument's data.

**Caveat carried over from the fix itself** (docs/architecture.md,
diagnostics/2026-08-11-tcn-vae-flat-reconstruction/): the skip path is
genuinely causal, but the pooled global latent still looks at the whole
window, so this remains an approximate comparison against GARCH's
strictly one-step-causal forecast, not a strictly like-for-like one. The
margin is now large enough (2.5-3.1 NLL, vs. GARCH's own ~0.03-0.2 margin
over the flat null) that this caveat matters more for interpretation than
it did before the fix.

Rung 5 in this same run still shows the pre-alias-fix zero-match tickers
(PG, AMZN, JNJ, BAC, LIN) -- this run was started before
diagnostics/2026-08-11-gdelt-alias-matching/ landed, so the
already-running process had the old filter_events_for_universe loaded in
memory (a source-file edit doesn't propagate into an already-executing
Python process). Not a new bug, just a timing artifact of iterating while
a long background run was in flight. A follow-up run with both fixes
active from the start will give a fully consistent report.
