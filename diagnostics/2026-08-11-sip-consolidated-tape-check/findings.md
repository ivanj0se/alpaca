# Checking the IEX-single-venue hypothesis against real consolidated-tape (SIP) data

Date: 2026-08-11

Follow-up to diagnostics/2026-08-11-real-tick-hawkes-replication/, which
left one hypothesis unverified for why the real-tick branching ratio
(0.9969, IEX feed) disagreed with Filimonov & Sornette's published 0.81
(E-mini S&P 500 consolidated futures tape): that single-venue IEX order
flow might be a mechanically different, more self-referential process
than genuine cross-market consolidated order flow. Set out to look for a
paid vendor to test this -- turned out not to be necessary.

## No paid vendor needed -- this account already has SIP access

Before shopping for Polygon/Databento/etc., checked whether Alpaca's own
API would even respond to a request for `DataFeed.SIP` (the real
consolidated tape) on the current account. It did, immediately, with no
separate subscription step: `client.get_stock_trades(..., feed=DataFeed.SIP)`
returned real data. Confirmed it's genuinely different (not silently
falling back to IEX): in the same 5-minute window, IEX returned 616
trades from one exchange code (`V`); SIP returned 18,234 trades (29.6x
denser) across 17 distinct exchange codes. SIP historical depth also
goes back at least a year, same as IEX.

Added a `feed` parameter to `ingest/historical_trades.py::backfill_trades`/
`update_incremental` (defaults to `DataFeed.IEX`, unchanged behavior for
existing callers) and a `--feed` CLI flag, so this is reusable rather than
a one-off script. Deliberately did **not** point a SIP pull at the
production `data/ticks` directory the live recorder writes to -- SIP and
IEX are structurally different samples of the same trades (SIP is a
strict superset), mixing them in the same partition would produce
misleading merged data. Backfilled 1 week (2026-08-04 to 2026-08-11) of
SIP data for SPY into a separate `data_sip_diagnostic/` directory: 3.27M
rows, 28MB on disk.

## Controlled comparison: same week, same event-definition methodology, only venue composition differs

Both datasets restricted to regular trading session hours (9:30-16:00 ET)
before comparing, since SIP includes real extended-hours prints (4.7% of
volume) that IEX barely has (0.08%) -- conflating "consolidated vs.
single-venue" with "extended-hours included vs. not" would have muddied
the result.

| Feed | n_ticks (regular session, same week) | n_events (sigma>=2.0) | branching_ratio |
|---|---|---|---|
| IEX (single venue) | 113,664 | 5,731 | **1.0006** |
| SIP (consolidated, 27.2x denser) | 3,092,223 | 20,168 | **0.9601** |

The IEX same-week result (1.0006) closely matches the full 29-day IEX
result from the earlier entry (0.9969) -- reassurance this particular
week isn't an outlier driving the comparison.

**The consolidated-tape result is lower, in the predicted direction, and
-- more strikingly -- far more stable across the event-selectivity
threshold than IEX ever was:**

| sigma>= | IEX branching_ratio (from the threshold-sweep entry) | SIP branching_ratio |
|---|---|---|
| 1.5 | (not tested at multistart) | 1.1394 |
| 2.0 | 0.9969 (robust) | 0.9601 |
| 2.5 | 0.9933, 0.9933, 1.0988, 0.0274 (**unstable across starts**) | 0.9500 |
| 3.0 | 1.2378, 0.7987, 0.9889, 0.9889 (**unstable across starts**) | 0.9411 |

At the exact thresholds where IEX's likelihood surface was so poorly
identified that different optimizer starting points landed on branching
ratios anywhere from 0.03 to 1.6+, SIP's 27x-denser event stream gives a
tight, self-consistent 0.94-0.96 -- which makes sense statistically (more
events, better-powered MLE) as well as suggesting a real difference in
the underlying process, not just a sample-size artifact.

## Honest reading

The hypothesis is **partially confirmed, not fully**. Moving to real
consolidated data measurably reduced the estimate (1.0006 -> 0.9601-0.94,
depending on threshold) and dramatically stabilized it, which is real,
positive evidence that single-venue IEX order flow was inflating the
apparent self-excitation. But even the most stable, best-identified
consolidated-tape estimate (~0.94-0.96) still sits right at or just above
the configured `plausible_band` upper edge (0.95), not comfortably inside
it, and well above the published 0.81. Remaining gap could be:

- **Genuine instrument/regime difference**: Filimonov & Sornette studied
  E-mini S&P 500 *futures* in 2011-2012; this is SPY *equity* in 2026 --
  different instrument (futures vs. ETF, no creation/redemption mechanics
  in a future), different market-structure era (pre- vs. post- several
  waves of HFT/regulatory change), different sample period. Not a
  controlled comparison in either of those dimensions, only in
  venue-consolidation.
- **Event-definition still imperfect**: this uses the same
  z-score-on-raw-trade-price-returns definition throughout: no attempt
  made to reconstruct the actual methodology F&S used (their paper likely
  defines events via mid-price changes from a full limit order book, not
  raw trade prints -- LOB data isn't available from Alpaca at any tier).
- Not testing further this session -- this has answered the specific
  question asked (does a paid/upgraded data source resolve the
  discrepancy) about as far as it can go without literally reproducing
  F&S's exact instrument, period, and event construction.

## Not decided here: should the live pipeline switch to SIP going forward?

This diagnostic pull was deliberately kept separate from the production
`data/ticks` store and the always-on `ingest/tick_recorder.py`. Switching
production to SIP would mean ~30x more data volume flowing through the
live recorder continuously (storage, and plausibly the same
connection/throughput considerations documented in
diagnostics/2026-08-11-tick-recorder-connection-limit-incident/, though
untested at this volume) -- a real operational change to an always-on
process, not something to flip silently. Left as an explicit open
decision rather than made unilaterally.
