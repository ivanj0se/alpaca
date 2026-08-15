# News-conditioned Hawkes marks: SPY blocked by a GDELT coverage gap; AMZN gives a real (null) answer

Date: 2026-08-14

Tier 2 of the private self-excitation research extension. Question: do
real Hawkes-flagged SPY jumps that fall near a real GDELT news event have
a different magnitude distribution than self-triggered (no nearby news)
jumps -- does news conditioning actually change the mark distribution,
not just correlate with event timing?

## Setup

- Events: 59,916 real SPY tick anomaly events from `data_sip_diagnostic/`
  (`tick_events_from_recorder`, `sigma_threshold=2.0`); 36,900 fall within
  the real GDELT coverage window used here (2026-07-25 to 2026-08-11, 17
  days, all 1,633 cached 15-minute GKG files -- 1,589,058 parsed rows).
- News: `filter_events_for_universe` with a hand-built SPY entry
  (`aliases=["S&P 500", "SPDR S&P 500", "S&P500", "S&P 500 Index"]`) --
  SPY isn't part of `config/universe.yaml`'s ~20-ticker universe (it's
  the separate `benchmark_instrument` entry), so this alias list was
  constructed specifically for this analysis.

## Result: zero real matches

`filter_events_for_universe` returned **0** SPY-tagged news events across
the entire 17-day, 1.59M-row real GDELT corpus. `compare_magnitude_distributions`
correctly refused to run on an empty matched group
(`ValueError: too few events in one group to compare (matched=0,
unmatched=36900)`) rather than silently producing a meaningless result.

Followed up directly rather than accepting this as a bare negative:
searched a broad 76,727-row sample (every 20th cached file, spanning the
full 17 days) across all four candidate GKG text fields
(`Organizations`, `V2Organizations`, `Themes`, `AllNames`) for `"S&P
500"`, `"SPDR"`, `"S&P"`, and the URL-escaped `"S%26P"` -- **zero
matches for every term, in every field, across the whole sample.**

## Reading this: a GDELT coverage gap specific to SPY, not a finding about news conditioning

This project has already directly confirmed (see
`ingest/gdelt_client.py::filter_events_for_universe`'s own docstring and
`diagnostics/2026-08-11-gdelt-alias-matching/`) that GDELT's GKG entity
extraction reliably surfaces real COMPANY names in the `Organizations`
field at meaningful volume -- "Amazon" alone matched 483 times on real
cached data, "Procter" 53, "Linde" 71. The same pipeline, on the same
kind of data, found **nothing at all** for an index/ETF name across four
different fields and four different phrasings. The most likely
explanation: GDELT's NLP entity extraction is tuned to recognize named
organizations (companies, institutions) in running prose, and "S&P 500"
functions grammatically as a financial index reference, not an
organization -- it may simply fall outside what the underlying NER model
tags as an "Organization" or "Person" entity at all, regardless of
phrasing. This is a real, structural mismatch between GDELT's extraction
target and what SPY actually is, not a bug in this project's matching
code, and not evidence that news proximity has no effect on jump
magnitude.

## What this means for the tier

The core mechanism this tier set out to test -- does a real jump's
magnitude differ conditional on real external news proximity -- remains
untested for SPY specifically (a GDELT coverage gap, not a falsification),
but **is now tested for AMZN, and the real answer is a null result**: no
detectable magnitude difference, at real 2026-08-11..14 sample sizes.
`research/news_conditioned_marks.py` itself is built, tested (6/6 passing
unit tests against synthetic planted effects and a duplicate-timestamp
regression check), and mechanically sound throughout both runs.

## Follow-up: AMZN has real coverage, and the real answer is a null result

Reran against AMZN, which already has a confirmed high-coverage GDELT
alias (`"Amazon"`, 483 matches on real data per the paragraph above) and
real tick history in `data/` (always-on IEX recorder). AMZN's tick
history only spans 2026-08-11 to 2026-08-14 (~3.3 days, the recorder's
own startup date for this ticker) -- extended the cached GDELT window to
cover it first (`data/gdelt_cache` grew from 1,633 to 2,025 files,
2026-07-25 through 2026-08-15), then parsed only the 321 files actually
overlapping the tick span (not the full cache -- parsing all 2,025 files
in one process was silently killed twice, exit code 137/SIGKILL, almost
certainly memory pressure from holding ~1.6M+ wide-string GKG rows at
once; narrowing to the real overlap window avoided this and finished in
under 30s).

**Real result:** 302 AMZN Hawkes-flagged tick events (`sigma_threshold=2.0`)
within the 3.33-day GDELT coverage window; 417 real AMZN-tagged GDELT news
events; calibrated `match_window=5min`. Of the 302 events, 152 matched
(news-adjacent), 150 unmatched --

| | matched (n=152) | unmatched (n=150) |
|---|---|---|
| mean \|z-score\| | 3.447 | 4.548 |
| std | 3.333 | 8.877 |

mean_gap = -1.10 (matched events were, if anything, SMALLER on average,
opposite the a-priori direction). Mann-Whitney p=0.268; null-control
p=0.199 (observed gap is ~1.3 null-standard-deviations from zero,
null_std_gap=0.844) -- **not statistically significant by either test.**

**Honest reading:** over this real 3.3-day AMZN window, jump magnitude
does not measurably differ between news-adjacent and self-triggered
Hawkes-flagged events. This is a genuine null result on real data, not a
coverage artifact (GDELT matching worked fine here) and not a code bug
(`research/news_conditioned_marks.py` is the same module already
validated against synthetic planted effects). The unmatched group's much
larger std (8.88 vs 3.33) hints at a few large idiosyncratic jumps
unrelated to any tracked news event, but the effect isn't strong enough
to move the mean gap outside chance variation. Caveat: only ~3.3 days of
real coverage overlap and 302 total events is a fairly small, likely
underpowered sample -- this doesn't rule out a real effect that would
show up with more real history once the always-on recorder accumulates
more AMZN ticks; it just says none was detectable here.
