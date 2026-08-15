# News-conditioned Hawkes marks: inconclusive for SPY (a GDELT coverage gap, not a null result)

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

**The core mechanism this tier set out to test -- does a real jump's
magnitude differ conditional on real external news proximity -- remains
genuinely untested, not falsified.** `research/news_conditioned_marks.py`
itself is built, tested (5/5 passing unit tests against synthetic planted
effects), and mechanically sound; the blocker is entirely that SPY,
specifically, doesn't have usable GDELT coverage under an
Organizations/Themes/entity-based matching approach.

## Not resolved / concrete follow-up

The natural next step -- not done here -- is to rerun
`compare_magnitude_distributions` against one of the universe tickers
already confirmed to have real GDELT coverage (e.g. `AAPL` or the
"Amazon" alias already validated above) instead of SPY, using that
ticker's own Hawkes-flagged tick events. That would give the tier's
actual real result; this entry documents why SPY specifically couldn't
answer it and rules out the two cheap, likely explanations (wrong
alias, wrong field) before concluding a coverage gap rather than a code
bug.
