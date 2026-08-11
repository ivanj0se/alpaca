# Per-ticker match_window calibration

Date: 2026-08-11
Follow-up to diagnostics/2026-08-11-full-ladder-run/rung5_notes.md, which
flagged (without fixing) that a flat 30-min match_window saturates
high-coverage tickers (real and null match rates both ~100%, i.e. zero
statistical power to distinguish signal from noise) while under-using
sparse ones.

## Approach

`attribution/null_control.py::calibrate_match_window` uses the 1D Boolean
coverage-process formula: for `n` events scattered at rate
`lambda = n / observation_minutes`, the probability a random query point
falls within +/-W of at least one event is `1 - exp(-2*W*lambda)`. Solving
for W given a target probability (default 0.2) yields a per-ticker window
that targets the *same null match rate* for every ticker, regardless of
how newsy it is -- denser coverage gets a narrower window, sparser
coverage gets a wider one.

Validated the formula itself by Monte Carlo (simulate the process,
measure empirical coverage against the calibrated window): matched the
target to within ~1 percentage point across event counts 20-1000 and
target rates 0.1-0.3.

## Real calibrated windows (3-day observation period, target 0.2)

| Ticker | n_events | Calibrated window |
|---|---|---|
| WMT, GS, JPM, BA, AMZN, CVX, CAT, PFE (8 tickers) | 232-1179 | 5 min (floor) |
| GOOGL, MSFT, LIN, XOM, NEE, META, PG, NVDA, AAPL | 42-73 | 6.6-11.5 min |
| BAC, UNH | 18-21 | 23-27 min |
| JNJ | 6 | 1h20m |

## Honest limitation: the floor doesn't fully de-saturate the densest names

The 8 highest-coverage tickers all hit the 5-minute `min_window` floor --
their *unclamped* calibrated window would be well under a minute (matches
the earlier Monte Carlo finding that 500-1000 events over 3 days
calibrates to ~0.5-1.5 min). At the 5-minute floor, WMT's actual
theoretical null coverage is still `1 - exp(-2*5*(1179/4320)) ~= 0.94` --
still highly saturated, just less than the previous 30-min window's
near-100%.

This is an intentional trade-off, not an oversight: a match window
meaningfully narrower than the underlying data's own temporal resolution
(TCN-VAE windows have a 10-min stride; GDELT publishes in 15-min batches)
isn't physically meaningful even if the formula says it would de-saturate
the test. For the handful of extremely newsy tickers in this universe
(major banks, Walmart), some residual saturation is unavoidable without
either (a) accepting matches finer than the data supports, or (b) finding
a fundamentally different test design for high-frequency-coverage names.
Not solved here -- documented so it isn't mistaken for the calibration
having no effect.

## Wiring

`scripts/run_ladder.py::run_rung5_attribution` now calibrates a window per
ticker (falling back to the flat `config/settings.yaml` setting only when
a ticker has zero news events, where there's nothing to calibrate from
anyway) instead of using one fixed window for the whole universe. The
report table now includes each ticker's calibrated `match_window_minutes`
for transparency.

## Next: re-run the full ladder to measure the actual effect

This entry documents the mechanism and its real limitation; a follow-up
full-universe run will show whether null_mean values are now more
consistent across tickers (the actual goal) for the mid-tier and sparse
names, and by how much residual saturation remains for the top 8.
