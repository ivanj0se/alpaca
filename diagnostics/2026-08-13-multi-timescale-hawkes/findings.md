# Multi-timescale Hawkes kernel: IEX and SIP self-excitation operate at genuinely different clock speeds

Date: 2026-08-13

Tier 1 of the private self-excitation research extension. Original
hypothesis: the already-found IEX-vs-SIP branching-ratio gap
(`diagnostics/2026-08-11-sip-consolidated-tape-check/`) is concentrated
at fast timescales, evidence of single-venue microstructure noise
(quote-stuffing, internal matching artifacts). **Not confirmed** -- the
real finding is different and, honestly, more interesting.

## First attempt: a false start worth documenting

Built `research/multi_kernel_hawkes.py` (sum-of-exponentials kernel,
fixed data-adaptive timescale grid, fitted amplitudes only -- extends
`events/hawkes.py`'s Ozaki recursion to K independent components without
modifying that trust-gated module). First real run: fit IEX and SIP on a
**shared** 3-timescale grid centered on SIP's median gap (tau = 5.55s,
0.175s, 0.0055s). Result looked backwards from the hypothesis -- IEX's
total branching ratio (0.7246) came out *lower* than SIP's (0.9486), with
IEX's medium component pinned exactly at its lower bound (alpha=1e-12)
and almost all its signal crammed into the single slowest grid point.

That's a red flag, not a finding -- a parameter pinned exactly at a bound
usually means the model doesn't have the right building blocks, not that
the signal is genuinely absent. Refit IEX on **its own** data-adaptive
grid (centered on IEX's own median gap, not SIP's) before trusting
anything: **total branching ratio = 0.9970**, matching the earlier
free-beta single-exponential IEX result (0.9969,
`diagnostics/2026-08-11-real-tick-hawkes-replication/`) almost exactly.
Also reran with a much wider 12-point alpha0 grid -- identical
log-likelihood to two decimals, so this isn't a local-optimum artifact
either. The shared-grid comparison wasn't wrong because of a bug; it was
wrong because it forced two feeds with genuinely different natural
timescales onto the same ruler.

## The real finding: IEX's median event gap is ~52x slower than SIP's

| Feed | Median inter-event gap | Dominant timescale | Branching ratio by component | Total |
|---|---|---|---|---|
| IEX | 9.06s | tau=286s (~4.8 min), 86% of total | [0.860, 0.130, 0.007] at tau=[286s, 9.06s, 0.29s] | **0.997** |
| SIP | 0.175s | spread across all 3, no single dominant scale | [0.467, 0.371, 0.111] at tau=[5.55s, 0.175s, 0.0055s] | **0.949** |

IEX's self-excitation is dominated by a single **slow** (~5-minute)
component; SIP's is spread fairly evenly across **three much faster**
components, reaching down to single-digit milliseconds -- a timescale
IEX's own grid never even probes, because IEX doesn't have enough event
density to populate it meaningfully.

## Plausible mechanism (not proven, but a reasonable read)

SIP sees the whole consolidated market; IEX sees roughly its own slice
of volume. Genuine cross-venue, HFT-speed reflexivity (fast arbitrage
and liquidity-taking reactions across many venues within
milliseconds-to-seconds) is only *visible* in consolidated data -- a
single venue's own tape can't show a reaction that primarily plays out
across venues it isn't part of. What a single venue *can* show is
something slower: a single large order or algorithm methodically
routing flow to that specific venue over several minutes (a metaorder-persistence
effect), which plausibly explains IEX's dominant ~5-minute timescale.
This is broadly consistent with Hardiman, Bercot & Bouchaud (2013)'s
own two-regime finding (a faster HFT-driven decay below ~10^3 seconds, a
slower metaorder-driven decay above it) -- we're seeing an analogous
fast/slow split, just realized as a difference *between venues* rather
than within one feed's own kernel.

## What this means for the original hypothesis

Wrong in its specific prediction (excess concentrated at *fast*
timescales) but right in spirit (the two feeds' branching-ratio
difference is a real, timescale-structured phenomenon, not a uniform
inflation) -- just structured the opposite way from what I guessed.
Worth being direct about: the first hypothesis didn't survive contact
with the data, and the honest thing was to say so and dig for what
actually happened rather than keep the original framing.

## Reassuring cross-validation, despite everything above

Two structurally different model specifications -- the original
free-beta single-exponential fit and this fixed-multi-kernel fit -- land
on nearly identical *total* branching ratios for each feed independently
(IEX: 0.9969 vs 0.9970; SIP: 0.94-0.96 range previously vs 0.9486 here).
The near-criticality finding itself is robust across two genuinely
different methodologies. What's new here is the *decomposition* --
methodology 1 couldn't see that this near-criticality is built from
completely different timescale structures depending on which venue you
look at; this one can.

## Not resolved

Whether the ~5-minute IEX timescale really reflects metaorder-splitting
(as opposed to some other single-venue-specific mechanism) isn't tested
directly here -- would need order-size/participant data this project
doesn't have. Flagged as a plausible read, not a proven mechanism.
