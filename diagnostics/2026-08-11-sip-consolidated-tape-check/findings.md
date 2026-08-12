# Checking the IEX-single-venue hypothesis against real consolidated-tape (SIP) data

Date: 2026-08-11

Follow-up to diagnostics/2026-08-11-real-tick-hawkes-replication/, which
left one hypothesis unverified for why the real-tick branching ratio
(0.9969, IEX feed) disagreed with Filimonov & Sornette's published 0.81
(E-mini S&P 500 consolidated futures tape): that single-venue IEX order
flow might be a mechanically different, more self-referential process
than genuine cross-market consolidated order flow. Set out to look for a
paid vendor to test this -- turned out not to be necessary.

## CORRECTION (same day): the first version of this analysis reported a wrong number

The first pass through this investigation reported SIP branching_ratio=0.7910
at sigma>=2.5 -- close enough to the published 0.81 that it read as the
resolution to the open question. **It was wrong.** A wider optimizer
starting-point sweep found a different, genuinely better-fitting optimum
(higher log-likelihood) at the same configuration: branching_ratio=0.9520,
not 0.7910. The `fit_hawkes_exponential_multistart` hardening built
earlier the same day (diagnostics/2026-08-11-real-tick-hawkes-replication/)
used only 4 starting points (alpha0 in 0.1, 0.5, 0.9, 2.0); on this
specific dataset all 4 happened to converge to the same *worse* local
optimum, which passed every robustness check that narrow grid could
perform (agreement across all 4 starts) while still being wrong. Widening
to 9 points (0.02, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 7.0, 15.0) found the true
optimum; a one-time 55-point 2D sweep over both alpha0 and beta0 confirmed
0.9520 as the actual best (highest log-likelihood among everything tried).
A second number in the original write-up (IEX sigma>=3.0=0.8220) turned
out to have the identical problem and is also corrected below (->0.9947).
`events/hawkes.py::fit_hawkes_exponential_multistart`'s default grid is
now the wider 9-point version; both diagnostics entries below are updated
to the re-verified numbers. Everything past this notice is the corrected
version.

## No paid vendor needed -- this account already has SIP access

Before shopping for Polygon/Databento/etc., checked whether Alpaca's own
API would even respond to a request for `DataFeed.SIP` (the real
consolidated tape) on the current account. It did, immediately, with no
separate subscription step: `client.get_stock_trades(..., feed=DataFeed.SIP)`
returned real data. Confirmed it's genuinely different (not silently
falling back to IEX): in the same 5-minute window, IEX returned 616
trades from one exchange code (`V`); SIP returned 18,234 trades (29.6x
denser) across 17 distinct exchange codes. SIP historical depth also
goes back at least a year, same as IEX. Separately confirmed the account
does **not** have live SIP *streaming* access (`insufficient subscription`
on connecting `StockDataStream` with `feed=SIP`) -- historical REST and
live websocket access are gated independently on this account; only the
historical side works. This means the always-on `ingest/tick_recorder.py`
cannot be switched to SIP on the current plan, regardless of any decision
about whether it would be worth doing.

Added a `feed` parameter to `ingest/historical_trades.py::backfill_trades`/
`update_incremental` (defaults to `DataFeed.IEX`, unchanged behavior for
existing callers) and a `--feed` CLI flag, so this is reusable rather than
a one-off script. Deliberately did **not** point a SIP pull at the
production `data/ticks` directory the live recorder writes to -- SIP and
IEX are structurally different samples of the same trades (SIP is a
strict superset), mixing them in the same partition would produce
misleading merged data. Pulled into a separate `data_sip_diagnostic/`
directory: first 1 week (3.27M rows, 28MB), then extended to a full month
(2026-07-13 to 2026-08-11, matching the IEX comparison window exactly)
once the 1-week sample showed the comparison was worth doing properly:
14.0M rows written, 13.87M read back, 119MB on disk.

## The properly-verified comparison (full month, corrected fitter)

Both datasets restricted to regular trading session hours (9:30-16:00 ET)
before comparing, since SIP includes real extended-hours prints (4.7% of
volume) that IEX barely has (0.08%) -- conflating "consolidated vs.
single-venue" with "extended-hours included vs. not" would have muddied
the result. All six results below re-run through the corrected 9-point
multistart grid.

| sigma>= | IEX n_events | IEX branching_ratio | SIP n_events | SIP branching_ratio | SIP − IEX |
|---|---|---|---|---|---|
| 2.0 | 26,574 | 0.9989 | 107,628 | 0.9573 | −0.0416 |
| 2.5 | 16,668 | 0.9975 | 76,154 | 0.9520 | −0.0455 |
| 3.0 | 10,860 | 0.9947 | 54,326 | 0.9445 | −0.0502 |

(IEX density: 472,347 regular-session ticks over the month. SIP: 13,228,750
-- 28.0x denser.) Also re-verified the original headline result from the
earlier entry (full IEX recorder history, all hours, sigma>=2.0, n=23,470)
against the corrected wide grid: **unchanged at 0.9969**, loglik=-88229.78,
identical to the narrow-grid result -- that particular number holds up.

**This corrected picture is more coherent than the (wrong) first pass.**
Every one of the six configurations lands in the same tight 0.94-1.00
band -- no more "one threshold mysteriously drops near the published
value while everything around it stays near-critical," which in hindsight
should have been the tell that something was off before it was ever
written up. Instead there's a small, consistent, monotonically-widening
gap: SIP is 0.04-0.05 lower than IEX at every threshold tested, growing
slightly as the threshold gets stricter. That's a real, reproducible
signal in the predicted direction, not a resolution.

## Honest reading

The hypothesis is **confirmed as a real, small, directionally-correct
effect -- not as an explanation that closes the gap to 0.81.** Real
consolidated-tape data does show measurably less apparent self-excitation
than single-venue IEX data, consistently across three thresholds. But the
effect size (~0.04-0.05) is far smaller than the ~0.15-0.19 gap that would
be needed to reach the published value, let alone land comfortably inside
`plausible_band` ([0.5, 0.95] -- SIP's best result, 0.9445, is still just
outside it). Remaining gap most likely reflects:

- **Genuine instrument/regime difference**: Filimonov & Sornette studied
  E-mini S&P 500 *futures* in 2011-2012; this is SPY *equity* in 2026 --
  different instrument (futures vs. ETF, no creation/redemption mechanics
  in a future), different market-structure era, different sample period.
  Not a controlled comparison in either of those dimensions, only in
  venue-consolidation.
- **Event-definition still imperfect**: this uses the same
  z-score-on-raw-trade-price-returns definition throughout every test in
  this project so far -- no attempt made to reconstruct F&S's actual
  methodology (their paper likely defines events via mid-price changes
  from a full limit order book, not raw trade prints -- LOB data isn't
  available from Alpaca at any tier).
- Not testing further this session -- this has answered the specific
  question asked (does a paid/upgraded data source resolve the
  discrepancy) about as far as it can go without literally reproducing
  F&S's exact instrument, period, and event construction.

## Not decided here: should the live pipeline switch to SIP going forward?

Moot for now, and not just left open -- confirmed above that this account
doesn't have live SIP *streaming* access even though it has historical
SIP REST access, so `ingest/tick_recorder.py` cannot switch to SIP without
a plan upgrade. Whether that upgrade would be worth it is a separate,
still-open question: the corrected numbers show SIP genuinely helps
(lower, more stable estimates) but doesn't turn Rung 1 into a clean pass
by itself, so the case for paying for a plan change rests more on general
data quality than on "this one gate would pass." Not decided here.
