# Tick storage was silently discarding real trades on every merge write

Date: 2026-08-11

## What was wrong

`ingest/storage.py::_write_partitioned` de-duplicated on `subset="timestamp"`
alone whenever a write merged with an already-existing partition file (the
normal case for `ingest/tick_recorder.py`, which flushes many times
throughout a single trading day into the same date partition). That key is
correct for bars (one bar per ticker per minute, by construction) but
wrong for ticks: two genuinely different real trades can share a
timestamp down to the microsecond -- a single incoming order sweeping
several resting price levels produces multiple separate trade prints,
often in the same tick of the matching engine. `drop_duplicates(subset="timestamp")`
silently kept only one of each such pair.

Found this while backfilling real historical SPY tick data (see
diagnostics/2026-08-11-real-tick-hawkes-replication/) for Rung 1: freshly
backfilled days (first-ever write to a new partition, which happens to
skip the merge/dedup branch entirely) showed real distinct-price
collisions preserved; today's partition, which the live recorder had
already been incrementally writing to all day, showed *zero* such
collisions -- exactly the signature of them having already been quietly
collapsed away.

## Measured impact

Re-fetched 2026-08-11's SPY trades directly from Alpaca's historical
trades endpoint (the authoritative record) after market close and
compared against what the live recorder had stored:

- Stored by the live recorder (before fix): 12,180 ticks
- True count for the same day (historical API, post-fix write path): 16,841 ticks
- **4,661 real trades (27.7%) had been silently dropped**, every single
  day the recorder has run.

Across the 29-day historical backfill (all first-time writes, so
unaffected by this specific bug -- see diagnostics/2026-08-11-real-tick-hawkes-replication/
for how that data was obtained), 16,664 of 18,955 timestamp-collision
groups had genuinely different price/size, confirming this isn't rare:
roughly 4% of real ticks share a timestamp with at least one other real,
distinct trade.

Why this specifically matters for the project's own research question:
the dropped trades are disproportionately the clustered/bursty ones --
exactly the self-exciting activity a Hawkes fit is trying to
characterize. Silently thinning exactly the signal of interest, in a way
that would have been invisible without deliberately cross-checking
against an independent authoritative source.

## Fix

`_write_partitioned` now de-duplicates on full-row equality
(`combined.drop_duplicates()`, no `subset`) instead of `timestamp` alone,
applied consistently on both the merge path and the first-write path.
Still correctly collapses genuine duplicates (the same trade re-fetched
by an overlapping incremental window, or the same flush written twice) --
regression test `test_exact_duplicate_row_is_still_collapsed` -- while no
longer discarding distinct trades that happen to share a timestamp --
regression test `test_distinct_trades_sharing_a_timestamp_are_both_kept`.

Re-ran `ingest.historical_trades` for 2026-08-11 specifically after the
fix, which merged against the live recorder's already-written partition
and recovered the previously-dropped trades (12,180 -> 16,841 stored).
Going forward the live recorder no longer loses this data on its own
incremental flushes either, since it shares the same `write_ticks` code
path.

## Not fully attributed

The 4,661-trade gap is most likely dominated by this bug (the mechanism
and the "zero collisions in the one already-merged partition" signature
both point the same direction), but a small part of it could plausibly
reflect a genuine difference between the live IEX websocket feed and the
historical IEX REST endpoint (e.g. late-reported or corrected prints
reconciled after the fact into the historical record but never present
in the original live stream). Not chased further -- doesn't change the
fix, and the dominant mechanism is well-evidenced.
