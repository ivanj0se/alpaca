# Finding: minute-bar events cannot reproduce the tick-level Hawkes branching ratio

Date: 2026-08-11
Source data: 120 days of real SPY minute bars via Alpaca (2026-04-13 to
2026-08-07), `tests/replication/fixtures/spy_bars_120d.parquet` (32,792 rows).

## Summary

Fitting the exponential-kernel Hawkes model (events/hawkes.py) to
bar-threshold "events" derived from real SPY minute bars produces a
branching ratio of essentially zero (alpha pins at its lower optimization
bound), regardless of the event-definition threshold. This does **not**
mean the Hawkes fitter is broken -- it means 1-minute bar aggregation
structurally destroys the timescale at which real order-flow
self-excitation lives. This is a stronger and more precise version of the
risk already flagged in docs/architecture.md ("bar-threshold proxies likely
bias the branching ratio downward").

## Evidence

1. **Threshold sweep** (2-sigma down to 0.25-sigma, plus "any price change"
   at all, 97.6% of bars): branching ratio is ~0 at every density from 1,060
   events up to 32,016 events. Density alone doesn't fix it.
2. **Dispersion test**: counting "any price change" events in 10-minute
   windows gives a Fano factor (variance/mean of counts) of 0.82 -- *under*-dispersed
   relative to a Poisson process. Real self-excitation produces
   *over*-dispersion (Fano > 1, bursty clustering). Minute bars are close to
   a regular grid (a bar closes ~every 60s whenever the market is active),
   which is close to the opposite of what a self-exciting process looks
   like.
3. **Fitter sanity checks** (rule out an implementation bug):
   - The Ozaki O(N) recursive log-likelihood matches a naive O(N^2) double
     sum to float precision on synthetic data.
   - Simulating a known Hawkes process (mu=0.05, alpha=0.5, beta=1.0) and
     refitting recovers the branching ratio within ~1-3% (tests/unit/test_hawkes.py).
   - Injecting deliberate bursts on top of an otherwise-regular synthetic
     sequence (mimicking minute-bar regularity) makes the fitter correctly
     recover a nonzero alpha (0 -> 0.40) -- the fitter *can* detect
     self-excitation when it's actually present; minute-bar SPY data just
     doesn't have any at this timescale.

## Conclusion

The published Filimonov & Sornette (2012) branching ratio (~0.81) was
measured on E-mini S&P 500 **trade-level tick data**, where inter-event
gaps are milliseconds to seconds. SPY minute bars have ~60-second inter-event
spacing by construction -- three to five orders of magnitude coarser than
the timescale where order-flow self-excitation actually operates. No
amount of threshold tuning on minute bars fixes this; it requires real
tick-level event timestamps, which is exactly why ingest/tick_recorder.py
exists and was started immediately rather than deferred.

## Design change

The Rung 1 trust gate is split into two separate claims rather than one:

1. **Fitter correctness** (synthetic, no real data needed): the
   simulate-refit-recover and burst-injection checks in
   tests/unit/test_hawkes.py. This validates the math.
2. **Real-data behavior**: tests/replication/test_hawkes_branching_ratio_replication.py
   now asserts the *correct, expected* minute-bar result (branching ratio
   near zero, with the dispersion reasoning above as justification) rather
   than the tick-level published figure. A second test in the same file,
   gated on enough accumulated real tick data existing
   (ingest/tick_recorder.py output), performs the actual "does this land
   near 0.81" check against config/settings.yaml's `plausible_band` -- it
   skips gracefully until there's enough tick data, and will start running
   for real once the recorder has accumulated a few weeks of trades.
