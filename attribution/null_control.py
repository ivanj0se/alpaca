"""Permutation/null-control significance testing for the news-correlation
attribution layer. GDELT's Organizations-field matching is heuristic
(confirmed false positive during development, see attribution/correlate.py),
so a raw correlation rate is not trustworthy on its own -- this compares
the observed rate against the distribution of rates you'd see from
randomly shuffled news timestamps, which controls for "how much
correlation would appear by chance alone, given how many news events and
anomalies there are." Mirrors the SSA project's Sidak-corrected
significance-testing culture (its own C4/null-control gates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from attribution.correlate import correlation_rate, match_to_news


def shuffle_news_timestamps(
    news_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    seed: int | None = None,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Replaces the real news timestamps with len(news_df) independently
    drawn uniform-random timestamps over [window_start, window_end] --
    keeps the same *count* of news events (so the null accounts for "how
    much correlation appears by chance given this many news events in this
    period") while breaking any real temporal structure.

    Does NOT permute the existing timestamp *values* among rows -- that
    would be a no-op here. match_to_news only depends on the set of
    timestamps present, not which row/headline owns which time, so
    permuting a fixed multiset among itself produces an identical result
    every time. Confirmed the hard way on real GDELT data: the first
    version of this function did exactly that, and the resulting
    "permutation test" came back with null_std=0.0 and null_mean exactly
    equal to the observed rate on every run -- every single shuffle was
    silently identical to the original.
    """
    rng = np.random.default_rng(seed)
    n = len(news_df)
    span_ns = int((window_end - window_start).value)
    if span_ns <= 0:
        raise ValueError("window_end must be after window_start")
    random_offsets_ns = rng.integers(0, span_ns, size=n)
    shuffled = news_df.copy()
    shuffled[time_col] = window_start + pd.to_timedelta(random_offsets_ns, unit="ns")
    return shuffled


def calibrate_match_window(
    n_events: int,
    observation_minutes: float,
    target_null_rate: float = 0.2,
    min_window: pd.Timedelta = pd.Timedelta(minutes=5),
    max_window: pd.Timedelta = pd.Timedelta(hours=6),
) -> pd.Timedelta | None:
    """Per-ticker match_window, instead of one fixed value for every
    ticker. A fixed window is either too generous or too tight depending on
    how often a ticker appears in the news -- confirmed on real data
    (diagnostics/2026-08-11-full-ladder-run/rung5_notes.md): at a flat
    30-min window, high-coverage tickers (WMT, GS, JPM, BA, CVX) had both
    real and null match rates near 100% (saturated -- the test has no
    power to distinguish signal from noise, since almost any timestamp
    matches something), while low-coverage tickers got a much weaker test
    than they could have.

    Uses the 1D Boolean coverage-process formula: for `n_events` scattered
    uniformly at rate `lambda = n_events / observation_minutes` over a
    period, the probability a random query point falls within +/- W of at
    least one event is `1 - exp(-2*W*lambda)`. Solving for W given a
    target probability yields a window that targets the same *null* match
    rate for every ticker, regardless of how newsy it is. Verified by
    Monte Carlo (simulate the process, measure the empirical coverage
    rate against the calibrated window): matched the target rate to within
    ~1 percentage point across n_events from 20 to 1000 and target rates
    0.1-0.3.

    Returns None if there are no events to calibrate against (nothing
    sensible to compute). Clamped to [min_window, max_window]: very dense
    coverage would otherwise calibrate to a window narrower than the
    anomaly-detection stride itself (a few tens of seconds), and very
    sparse coverage would calibrate to implausibly wide windows spanning
    much of the observation period.
    """
    if n_events <= 0:
        return None
    if not 0 < target_null_rate < 1:
        raise ValueError("target_null_rate must be in (0, 1)")

    rate_per_minute = n_events / observation_minutes
    w_minutes = -math.log(1 - target_null_rate) / (2 * rate_per_minute)
    window = pd.Timedelta(minutes=w_minutes)
    return max(min_window, min(window, max_window))


@dataclass
class PermutationTestResult:
    observed_rate: float
    null_mean: float
    null_std: float
    null_distribution: np.ndarray
    p_value: float


def permutation_test(
    anomaly_times: pd.DatetimeIndex,
    news_df: pd.DataFrame,
    time_window: pd.Timedelta,
    n_permutations: int = 2000,
    seed: int | None = None,
    time_col: str = "timestamp",
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
) -> PermutationTestResult:
    """Observed correlation rate vs. the distribution of rates from
    `n_permutations` null draws (see shuffle_news_timestamps -- fresh
    uniform-random timestamps, not a permutation of the existing ones).
    p_value is the fraction of null draws whose rate is >= the observed
    rate (one-sided: "is the real correlation higher than chance").

    window_start/window_end default to the real news_df's own min/max
    timestamp -- "how much correlation would appear by chance if this many
    news events were scattered randomly across the period they actually
    covered." Pass explicit bounds if anomaly_times spans a different
    period than news_df.
    """
    if news_df.empty:
        observed = correlation_rate(match_to_news(anomaly_times, news_df, time_window, time_col))
        null_rates = np.zeros(n_permutations)
        return PermutationTestResult(observed, 0.0, 0.0, null_rates, p_value=1.0)

    window_start = window_start if window_start is not None else news_df[time_col].min()
    window_end = window_end if window_end is not None else news_df[time_col].max()

    observed = correlation_rate(match_to_news(anomaly_times, news_df, time_window, time_col))

    rng = np.random.default_rng(seed)
    null_rates = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = shuffle_news_timestamps(
            news_df, window_start, window_end, seed=int(rng.integers(0, 2**32 - 1)), time_col=time_col
        )
        null_rates[i] = correlation_rate(match_to_news(anomaly_times, shuffled, time_window, time_col))

    p_value = float(np.mean(null_rates >= observed))
    return PermutationTestResult(
        observed_rate=observed,
        null_mean=float(null_rates.mean()),
        null_std=float(null_rates.std()),
        null_distribution=null_rates,
        p_value=p_value,
    )


def sidak_correction(p_values: np.ndarray, n_tests: int | None = None) -> np.ndarray:
    """Sidak-corrected p-values for multiple comparisons (e.g. one
    permutation test per ticker in the universe): 1 - (1 - p)^n.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = n_tests if n_tests is not None else len(p_values)
    return 1 - (1 - p_values) ** n


def summarize_attribution(results: dict[str, PermutationTestResult], alpha: float = 0.05) -> pd.DataFrame:
    """Per-ticker summary: observed rate, null mean/std, raw p-value,
    Sidak-corrected p-value (correcting for running one permutation test
    per ticker in the universe), and significance at `alpha`. This table is
    the project's actual deliverable: what fraction of each instrument's
    anomalous activity has a statistically defensible external trigger.
    """
    tickers = list(results.keys())
    raw_p = np.array([results[t].p_value for t in tickers])
    corrected_p = sidak_correction(raw_p, n_tests=len(tickers))

    return pd.DataFrame(
        {
            "ticker": tickers,
            "observed_rate": [results[t].observed_rate for t in tickers],
            "null_mean": [results[t].null_mean for t in tickers],
            "null_std": [results[t].null_std for t in tickers],
            "p_value": raw_p,
            "p_value_sidak": corrected_p,
            "significant": corrected_p < alpha,
        }
    ).sort_values("p_value_sidak")
