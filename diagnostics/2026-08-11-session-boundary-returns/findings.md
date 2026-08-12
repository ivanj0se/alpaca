# Overnight/weekend session-boundary returns treated as normal 1-minute returns

Date: 2026-08-11

## What was wrong

`log_returns()` (features/returns.py) computes `np.diff(log(close))` across
consecutive rows in whatever order the data arrives -- it has no concept
of a trading session. `features/bars.py::trading_session_index` already
exists and correctly builds a calendar-aware minute grid for the
cross-sectional alignment path, but neither of the two places that build a
*return series* for the temporal lane (`features/returns.py::build_feature_frame`,
used by Rung 0/2a/4) or the cross-sectional lane
(`scripts/run_ladder.py::log_returns_from_bars`, used by Rung 2b/3) went
through it. Both simply diffed sorted timestamps, so the first bar of
every new session got a "1-minute return" computed against the *previous
session's last close* -- silently folding an overnight, weekend, or
holiday move into the intraday return series as if it were an ordinary
minute-to-minute change.

## How big a deal, measured on real AAPL data

- 312 session-boundary points out of 24,264 rows (1.29% of rows).
- Boundary-point returns average 4.4x larger in magnitude than genuine
  intraday returns (mean |return| 0.00219 vs 0.00050); max boundary return
  3.37% vs 1.10% max intraday.
- Contamination spreads via the rolling `realized_vol` window (window=15):
  6.42% of rows had a vol estimate touched by a boundary return, and those
  rows' realized_vol averaged 3.04x higher than clean rows.
- This is the well-documented overnight-vs-intraday-return asymmetry in
  the market microstructure literature (e.g. Berkman et al. 2012) --
  not a data error, just two different statistical populations getting
  mixed into one series.

Why this matters beyond "one biased feature": a session-open jump getting
scored as a return outlier would inflate GARCH's fitted variance
persistence, distort the TCN-VAE's learned notion of "normal", and --most
concerning for the actual research question-- get flagged by Rung 4 as a
reconstruction-residual "anomaly" on every single trading day regardless
of whether anything newsworthy happened overnight, which would show up in
Rung 5 as spurious correlation with whatever news happened to be
published near that day's open. A confound specific to the thing this
project is trying to measure, not a generic modeling nuisance.

## Fix

`features/returns.py::session_boundary_mask(timestamps)`: True at
position i if `timestamps[i]` falls on a different NYSE trading-session
date (America/New_York calendar date) than `timestamps[i-1]`. Simple
date-changed heuristic rather than a full `pandas_market_calendars`
session lookup per bar -- correct for regular-session US equities (no
overnight sessions to confuse it) and cheap.

Wired into both places that build a return series:
- `features/returns.py::build_feature_frame` -- boundary returns are
  masked to NaN before `realized_vol` is computed, then dropped by the
  existing `dropna()` (same "fail loud, don't silently smooth" policy
  already used for missing bars).
- `scripts/run_ladder.py::log_returns_from_bars` -- same masking; also
  had to add an explicit `dropna` step to `run_cross_sectional_lane`
  before RPCA, since a NaN previously couldn't occur there and nothing
  handled it (RPCA's SVD can't take NaN input, and a boundary shared
  across the whole market -- e.g. Monday's open -- would otherwise NaN
  nearly every column on the same row).

7 new regression tests (`tests/unit/test_returns.py::TestSessionBoundaryMask`,
`TestBuildFeatureFrame::test_excludes_session_boundary_return`,
`tests/unit/test_run_ladder.py::TestLogReturnsFromBars::test_excludes_session_boundary_return`).
Full suite: 266 passed before this change, all still pass after (fix adds
tests, doesn't touch unrelated modules).

Post-fix, real AAPL data: feature-frame rows drop from 24,264 to 23,379
(885 rows / 3.6% lost, consistent with the ~6.4% contaminated-row estimate
above once vol_window overlap between adjacent boundaries is accounted
for); max |log_return| in the whole series drops from 3.37% to 2.20%, with
zero returns exceeding 5%.

## Measured effect on the full ladder (before -> after, mean across 20 tickers)

| Rung | Before (buggy) | After (fixed) | Change |
|---|---|---|---|
| Rung 0 (random-walk null) | -5.4483 | -5.7803 | -0.332 (6.1%) |
| Rung 2a (GARCH) | -5.6535 | -5.9348 | -0.281 (5.0%) |
| Rung 4 (TCN-VAE) | -8.5425 | -8.8998 | -0.357 (4.2%) |
| Rung 2b (factor model) | -5.4580 | -5.6080 | -0.150 (2.7%) |
| Rung 3 (RPCA) | -6.0271 | -6.2063 | -0.179 (3.0%) |

All NLLs improved (more negative) by a comparable few-percent margin
across every rung -- consistent with "removed a shared source of
variance-inflating outlier contamination" rather than favoring any one
model. Every rung's gate check (2a beats 0, 4 beats 2a, RPCA beats factor
model) still passes, 20/20 tickers, same as before the fix -- **the
ladder's relative conclusions are unchanged**, but the absolute NLL
numbers reported from here on are the trustworthy ones; anything computed
before this commit used the contaminated series.

Rung 1 (Hawkes bar-proxy) goes through `events/price_events.py`'s own
event proxy, not `build_feature_frame` -- had the identical bug (both
`bar_threshold_events` and `tick_events_from_recorder` diffed prices with
no session awareness), fixed the same way by masking boundary returns
before the z-score is computed. Effect on real SPY data: event count went
from 760 to 1178 (not down, despite removing ~90 boundary-crossing rows --
the boundary returns had also been inflating the z-score's std baseline,
suppressing sensitivity to genuine intraday moves; removing them tightens
the baseline enough that more real moves cross the sigma_threshold=2.0
bar than the ~90 removed boundary events cost). Branching ratio unchanged
at 0.0000 either way -- consistent with the existing
diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/ finding that
minute-bar events are structurally near-regular regardless of this
specific contamination.

Rung 5 (attribution): still no ticker survives Sidak correction (closest
is META at p_sidak=0.90, was AAPL at 0.97) -- expected, this fix reduces
noise in the *input* series, it doesn't add more news data or anomaly
samples, which remain the actual bottleneck (see
diagnostics/2026-08-11-match-window-calibration/findings.md).

## Not fixed here

`events/price_events.py` (Hawkes bar-proxy event extraction) wasn't
audited for the same issue -- flagged above, not chased further since it
doesn't currently affect the trust-gate conclusion.
