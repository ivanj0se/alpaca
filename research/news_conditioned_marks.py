"""Tests whether real Hawkes-flagged events' jump MAGNITUDE differs
between news-adjacent and self-triggered events, using this project's
own already-validated news-attribution machinery
(attribution/correlate.py, attribution/null_control.py) -- not a
textbook two-sample test alone, but one calibrated against this
project's own specific matching mechanism via the same null-control
philosophy already used for Rung 5.

Standard marked Hawkes processes (Bacry & Muzy order-flow literature)
condition the mark distribution on the process's OWN internal state
(recent intensity, volume, prior mark). This conditions it on something
external and independently validated instead: real proximity to a GDELT
news event. Directly operationalizes this project's founding
endogenous/exogenous question as a testable GENERATIVE mechanism -- do
news-adjacent jumps actually look different, not just correlate with
news timing -- rather than only a post-hoc detection statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from attribution.correlate import match_to_news
from attribution.null_control import shuffle_news_timestamps


@dataclass
class MagnitudeComparison:
    n_matched: int
    n_unmatched: int
    matched_mean: float
    unmatched_mean: float
    matched_std: float
    unmatched_std: float
    mean_gap: float  # matched_mean - unmatched_mean
    mannwhitney_p: float
    null_control_p: float  # fraction of shuffled-news trials with a gap >= observed
    null_mean_gap: float
    null_std_gap: float


def label_events_by_news_proximity(
    event_times: pd.DatetimeIndex, news_df: pd.DataFrame, match_window: pd.Timedelta
) -> np.ndarray:
    """Boolean array, one per event, in the SAME order as `event_times`:
    True if a real GDELT news event fell within +/- match_window (reuses
    attribution/correlate.py::match_to_news exactly, the same mechanism
    Rung 5 uses for detection -- a like-for-like reuse, not a parallel
    reimplementation).

    match_to_news sorts its output by timestamp internally
    (`.sort_values("anomaly_time")`, needed for merge_asof) -- its
    existing callers evidently never depended on output order matching
    input order. This function does, since the result gets zipped
    positionally against a separate `magnitudes` array by the caller, so
    it explicitly restores the original event order via an inverted
    sort permutation rather than assuming match_to_news preserves it.
    """
    event_times = pd.DatetimeIndex(event_times)
    order = np.argsort(event_times.to_numpy(), kind="stable")
    matches = match_to_news(event_times[order], news_df, match_window)
    matched_sorted = matches["matched"].to_numpy()
    result = np.empty(len(event_times), dtype=bool)
    result[order] = matched_sorted
    return result


def compare_magnitude_distributions(
    magnitudes: np.ndarray,
    event_times: pd.DatetimeIndex,
    news_df: pd.DataFrame,
    match_window: pd.Timedelta,
    n_permutations: int = 2000,
    seed: int | None = None,
) -> MagnitudeComparison:
    """Compares |jump magnitude| between news-matched and unmatched real
    events, both with a standard nonparametric test (Mann-Whitney U,
    doesn't assume a normal magnitude distribution -- these are heavy-tailed)
    and a null-control significance check: is the observed mean-magnitude
    gap larger than what independently-shuffled news timestamps would
    produce by chance (same shuffle_news_timestamps mechanism already
    validated for Rung 5, see diagnostics/2026-08-11-permutation-test-noop-bug/
    -- fresh uniform-random timestamps, not a permutation of the existing
    set, which would be a no-op for anything depending only on the
    timestamp set).
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    matched_mask = label_events_by_news_proximity(event_times, news_df, match_window)
    matched = magnitudes[matched_mask]
    unmatched = magnitudes[~matched_mask]
    if len(matched) < 2 or len(unmatched) < 2:
        raise ValueError(f"too few events in one group to compare (matched={len(matched)}, unmatched={len(unmatched)})")

    observed_gap = float(matched.mean() - unmatched.mean())
    _, mw_p = mannwhitneyu(matched, unmatched, alternative="two-sided")

    window_start = news_df["timestamp"].min()
    window_end = news_df["timestamp"].max()
    rng = np.random.default_rng(seed)
    null_gaps = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_news = shuffle_news_timestamps(
            news_df, window_start, window_end, seed=int(rng.integers(0, 2**32 - 1))
        )
        shuffled_mask = label_events_by_news_proximity(event_times, shuffled_news, match_window)
        s_matched = magnitudes[shuffled_mask]
        s_unmatched = magnitudes[~shuffled_mask]
        null_gaps[i] = s_matched.mean() - s_unmatched.mean() if len(s_matched) >= 2 and len(s_unmatched) >= 2 else np.nan

    valid_null = null_gaps[~np.isnan(null_gaps)]
    null_control_p = float(np.mean(np.abs(valid_null) >= abs(observed_gap))) if len(valid_null) > 0 else float("nan")

    return MagnitudeComparison(
        n_matched=len(matched),
        n_unmatched=len(unmatched),
        matched_mean=float(matched.mean()),
        unmatched_mean=float(unmatched.mean()),
        matched_std=float(matched.std(ddof=1)),
        unmatched_std=float(unmatched.std(ddof=1)),
        mean_gap=observed_gap,
        mannwhitney_p=float(mw_p),
        null_control_p=null_control_p,
        null_mean_gap=float(np.nanmean(null_gaps)),
        null_std_gap=float(np.nanstd(null_gaps)),
    )
