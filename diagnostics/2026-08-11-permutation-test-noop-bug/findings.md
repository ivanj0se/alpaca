# Critical bug: the null-control permutation test was a silent no-op

Date: 2026-08-11

## Severity

This is the most consequential bug found anywhere in the project so far.
`attribution/null_control.py` is the actual statistical mechanism the
whole project's deliverable rests on -- "what fraction of anomalous
activity has a real external trigger, at a defensible confidence level."
A broken null control doesn't produce a wrong number; it produces a number
that *looks* statistically validated but isn't.

## What happened

First implementation of `shuffle_news_timestamps` permuted the existing
timestamp values among the news dataframe's own rows:

```python
shuffled[time_col] = rng.permutation(shuffled[time_col].to_numpy())
```

Ran a real end-to-end test against live GDELT data (AAPL/MSFT/NVDA/GOOGL,
6-hour window). Result: `null_std = 0.000`, `null_mean` exactly equal to
`observed_rate`, on every run.

## Root cause

`match_to_news` (and everything downstream of it) only depends on the
*set* of news timestamps present -- not which row/headline owns which
timestamp. Permuting timestamp values among a fixed set of rows produces
an identical *set* of timestamps every time. Every one of the "2000
random shuffles" was silently computing the exact same thing as the
unshuffled original. The permutation test was, structurally, incapable of
ever disagreeing with itself -- p-values would have been meaningless
degenerate constants (either exactly 0 or exactly 1 depending on
floating-point tie-breaking) for every ticker, every time, regardless of
whether the real correlation was genuine or coincidental.

This is exactly the failure mode good practice test-generation is
supposed to catch, and did: real-data validation surfaced `null_std=0.0`
as an obviously wrong number before this shipped as a passing test suite
with plausible-looking (but meaningless) synthetic data.

## Fix

`shuffle_news_timestamps` now draws `len(news_df)` fresh, independent
uniform-random timestamps over `[window_start, window_end]`, replacing the
real values entirely rather than permuting them:

```python
random_offsets_ns = rng.integers(0, span_ns, size=n)
shuffled[time_col] = window_start + pd.to_timedelta(random_offsets_ns, unit="ns")
```

`permutation_test` now takes (or derives from `news_df`'s own min/max) an
explicit window to draw the null from -- "how much correlation would
appear by chance if this many news events were scattered randomly across
the period they actually covered."

## Validation after the fix

Calibrated synthetic scenario (sparse news: 20 events over 10 days, 15-min
match tolerance -- deliberately sparse relative to the tolerance window,
unlike the real-data smoke test below, to get a discriminating result
rather than a saturated one):

| Scenario | observed_rate | null_mean | null_std | p-value |
|---|---|---|---|---|
| Anomalies genuinely near news | 1.000 | 0.046 | 0.053 | 0.0000 |
| Anomalies unrelated to news (random) | 0.000 | 0.035 | 0.047 | 1.0000 |

Correctly significant for genuine correlation, correctly not significant
for none. Re-ran the original real-GDELT smoke test too (6-hour window,
4-ticker universe, ~16 matched news events): now shows genuine null
variance (std ~0.11) instead of the degenerate 0.0, though that specific
scenario's news density (avg gap ~20 min) relative to the 30-min tolerance
window saturates the correlation rate for both real and null cases (both
land near 90%+) -- an expected small-sample/parameter-choice effect, not a
bug, and a reminder that `time_window` needs to be calibrated relative to
the real news density at full project scale (~20 tickers, weeks of data),
not left at an arbitrary default.

## Takeaway

Regression test added
(`test_null_control.py::TestShuffleNewsTimestamps::test_is_not_a_permutation_of_the_same_values`)
asserting the shuffled timestamp *set* differs from the original --
directly encodes the failure mode found here so it can't silently
reappear.
