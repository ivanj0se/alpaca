# Real-tick Hawkes replication: unblocked via historical backfill, result is a genuine open question

Date: 2026-08-11

## Historical trade-level data is available on the free tier -- no need to wait on the live recorder

The original plan assumed real tick-level data could only come from
`ingest/tick_recorder.py`'s live websocket feed, accumulating slowly in
real time (the Rung 1 real-tick gate -- `>=5000 ticks over >=5 days` --
was designed around that constraint). Checked directly against the real
API: `StockHistoricalDataClient.get_stock_trades` (alpaca-py) returns real
historical trade-level data on the same free IEX feed already used for
bars, not just aggregated bars. Verified depth of history with fixed
regular-trading-hours sample windows (an earlier same-clock-time test
across dates falsely suggested no history existed beyond 1 day -- that
was a sampling artifact: testing the same after-hours wall-clock time on
every past date is empty regardless of how far back real data goes, since
markets are closed then every day):

| Days back | Date sampled | Trades in a 1h window |
|---|---|---|
| 1-30 | various | 2,798-4,576 |
| 90-365 | various | 1,652-3,861 |

Real trade-level history goes back at least a year on the free tier. Added
`ingest/historical_trades.py` (mirrors `ingest/historical_bars.py`'s
chunked-backfill pattern, one day per request since trade volume is much
higher than bar volume) to pull it directly into the same
`data/ticks` partitioned store the live recorder writes to -- both paths
produce byte-identical schema (verified via
`test_matches_live_recorder_schema`), so `read_ticks()` and the Rung 1
gate see backfilled and live-recorded ticks identically.

Backfilled the most recent 30 calendar days (2026-07-13 to 2026-08-11) of
real SPY trades: 472,734 rows. Chose ~30 trading days deliberately, not
the full year available -- a Hawkes MLE fit assumes locally stationary
dynamics, and a full year would very plausibly straddle different
volatility/liquidity regimes and bias the estimate; a month is long
enough to clear the gate by a wide margin while staying closer to a
single regime. Also found and fixed a real data-loss bug in the storage
layer along the way -- see
diagnostics/2026-08-11-tick-dedup-key-bug/findings.md.

## The real-tick Hawkes fit: result

With the recovered/backfilled dataset (489,575 SPY ticks over 29.3 days):

- Event definition: `events/price_events.py::tick_events_from_recorder`,
  same z-score-threshold-on-log-returns definition as the bar proxy
  (sigma_threshold=2.0), now also session-boundary-aware (see
  diagnostics/2026-08-11-session-boundary-returns/).
- n_events = 23,470 (4.8% of ticks flagged -- almost exactly the ~4.55%
  a Gaussian two-tailed threshold at sigma=2 would predict, so the
  *marginal* rate isn't obviously miscalibrated).
- **branching_ratio = 0.9969** (alpha=0.005588, beta=0.005605, converged=True).
- Configured plausible_band is [0.5, 0.95] (Filimonov & Sornette 2012
  published 0.81 on E-mini S&P 500 futures tick data) -- **0.9969 is
  outside the band, on the high side**. This is the polar opposite
  failure mode from the bar-proxy result (branching_ratio~0.0000, too low
  -- see diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/).

## Investigated the obvious mechanical-artifact hypothesis -- it's not the (whole) explanation

Median inter-event gap is ~9 milliseconds, with 96% of consecutive event
gaps under 100ms. That's consistent with a single real economic event (one
large order sweeping several price levels) generating a cascade of
separate trade prints within milliseconds, each individually crossing the
z-score threshold and getting counted as its own "self-triggered" event --
a mechanical artifact of the event definition, not genuine market-wide
self-excitation.

Tested directly: collapsed any event within 100ms of the previously-kept
event into a single representative event (23,470 -> 3,643 events, an 84%
reduction) and re-fit. **branching_ratio = 0.9925 -- barely moved from
0.9969.** The near-critical self-excitation survives collapsing away the
obvious millisecond-scale burst artifact almost entirely intact. This
rules out "it's just order-sweep prints being double-counted" as the
(main) explanation, which is itself a useful negative result -- whatever
is driving this is a real property of the process at a coarser timescale
than individual sweep prints, not a trivial counting artifact.

## What this means (honest assessment, not resolved)

The real-tick trust gate does **not** currently pass, and I'm not
adjusting `plausible_band` to make it pass -- that would defeat the
purpose of a trust gate. Leaving this as an open, documented question
rather than a silent failure:

- Plausible real explanation: Alpaca's free tier is IEX-only, a single
  venue that historically carries on the order of 2-3% of consolidated
  SPY volume -- already flagged as a caveat in docs/architecture.md for
  volume/trade-count purposes, but this may also mean single-venue order
  flow is a mechanically different (more self-referential -- e.g. the
  same market maker's own quote updates, or smart-order-router behavior
  specific to that venue) process than the true cross-market/consolidated
  order flow Filimonov & Sornette's E-mini futures data represents. Not
  verified directly -- would need a paid consolidated-tape feed to check.
- Alternative explanation: the z-score-threshold event definition itself,
  while adequate for the coarse bar proxy, may not be the right
  translation of "event" at true tick-level granularity even after
  burst-collapsing at 100ms -- e.g. it may need a longer collapse window,
  a minimum price-change-in-ticks threshold instead of a return z-score,
  or an entirely different construction. Not chased further this session.
- Not a bug in the Hawkes fitter itself: `converged=True`, and
  alpha/beta both landed at small, well-separated-from-zero, sane values
  (not pinned at an optimizer bound the way the earlier
  diagnostics/2026-08-11-hawkes-optimizer-initialization-bug/ case was).

## Follow-up: tested the event-definition question directly, found and fixed a real optimizer robustness gap

Went back to check the two hypotheses left open above, both against real
data rather than speculation.

**Volatility-clustering confound -- tested and refuted.** The z-score
threshold uses one *global* std over the whole 29-day series as the
denominator. If local volatility clusters over time (well established --
this whole project's other rungs confirm it), a global-std threshold
flags disproportionately more events during high-vol stretches and fewer
during calm ones, which would look like temporal clustering (self-excitation)
to a Hawkes fit even with zero genuine causal triggering -- a "doubly
stochastic" process misread as self-exciting. Confirmed the mechanism is
present: local event-flagging rate (1000-tick windows) correlates 0.754
with local realized volatility. But testing it directly kills it as *the*
explanation: refitting on a local-std-normalized (500-tick rolling window)
version of the same threshold gives branching_ratio=0.9983 -- statistically
indistinguishable from the global-std result (0.9969), not the substantial
drop the confound hypothesis would predict.

**Literal price-change event definition (closer to Filimonov & Sornette's
actual construction -- every distinct price change is an event, no
z-score threshold at all) is unstable, not lower.** n=290,487 (59% of all
ticks), branching_ratio=1.5937 -- *above* 1, mathematically outside the
range a stationary Hawkes process can produce. Confirmed this isn't an
optimizer-bound artifact (bounds are pure positivity constraints, no
upper limit) -- it's a genuine finite-sample MLE result. At maximal event
density the exponential-kernel model is even less well-behaved, not more.

**A full threshold sweep (sigma 1.5 to 6.0) exposed a real gap in the
fitter, not just an interesting market fact.** Refitting the *same* event
set from different `alpha0` starting points should agree (this exact
property is what the data-adaptive-beta0 fix guaranteed for the sparse
bar-proxy case, see diagnostics/2026-08-11-hawkes-optimizer-initialization-bug/)
-- but at several of these tick-level thresholds it doesn't:

| sigma>= | n_events | branching_ratio across alpha0=[0.1, 0.5, 0.9, 2.0] |
|---|---|---|
| 2.0 | 23,470 | 0.9969, 0.9969, 0.9969, 0.9969 (robust) |
| 2.5 | 13,578 | 0.9933, 0.9933, 1.0988, 0.0274 (**not robust**) |
| 3.0 | 9,013 | 1.2378, 0.7987, 0.9889, 0.9889 (**not robust**) |
| 3.5 | 5,859 | 0.9820, 1.6114, 0.0180, 0.0461 (**not robust**) |
| 6.0 | 932 | 0.9128, 0.3929, 0.9128, 0.1433 (**not robust**) |

The "moderate, plausible-looking" values that showed up in a naive
single-start sweep (e.g. 0.80 at sigma=3.0, 0.39 at sigma=6.0) are
**not trustworthy** -- they're arbitrary local optima, not well-identified
estimates. This means my first-pass threshold sweep (in an earlier,
unlogged scratch run) was misleading on its face; only re-fitting from
multiple starts and taking the best log-likelihood revealed which numbers
were real.

**Fix:** added `events/hawkes.py::fit_hawkes_exponential_multistart`
(refits from a small alpha0 grid, keeps the highest-log-likelihood
converged result) and wired it into both the bar-proxy and real-tick
fits in `scripts/run_ladder.py::run_rung1_hawkes` -- the trust gate should
not be silently trusting a single arbitrary local optimum. 4 new tests in
`tests/unit/test_hawkes.py::TestFitHawkesExponentialMultistart`.

**Re-ran the production sigma>=2.0 fit through the hardened multistart
path: branching_ratio=0.9969, identical to the single-start result
(loglik=-88229.78, the best among the grid).** The original number was
already sitting at the true optimum for this configuration -- multistart
doesn't change the answer here, but now actively guards against the
instability documented above recurring silently in a future run with
different data or thresholds.

## Bottom line

This is real progress, not a dead end: Rung 1's real-tick check produced
its first-ever actual number today instead of being permanently gated on
data that would take weeks to accumulate live, and that number is now
more defensible than when it was first measured. The event-definition
follow-up did real work: it killed the volatility-clustering-confound
hypothesis outright, showed the "any price change" definition is
*worse* (unstable) rather than a fix, and -- most concretely -- found and
fixed a real robustness gap in the fitter itself (single-start MLE
silently landing on arbitrary local optima at several plausible
thresholds). None of that made the near-critical result go away; if
anything it survived every test aimed at explaining it away, which raises
confidence that 0.9969 is a real, well-identified property of this
specific event stream (global-std z>=2 on IEX trade prints), not an
artifact.

What's still open is *why* single-venue tick-level order flow looks this
different from Filimonov & Sornette's consolidated-tape futures result --
the IEX-only-venue hypothesis remains the leading, unverified explanation
(would need a paid consolidated feed to check directly). Worth a follow-up
pass on this question before treating either the bar-proxy near-zero or
this near-one result as "the" answer for SPY's true tick-level branching
ratio.
