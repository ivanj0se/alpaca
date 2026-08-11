"""Correlates anomaly/residual scores against real news events -- Rung 5,
and the actual point of the project: what fraction of "unexplained by the
market's own internal dynamics" activity has an identifiable external
trigger. A raw correlation rate from this module is never trustworthy on
its own -- GDELT's Organizations-field matching is heuristic (confirmed
false positive during development: "Group Alphabet Bar Grill" matched
"Alphabet"), so every use of correlation_rate should be compared against
attribution/null_control.py's permutation test, not read in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def high_residual_windows(score_series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Timestamps where an anomaly/residual score exceeds `threshold`."""
    return pd.DatetimeIndex(score_series[score_series > threshold].index)


def match_to_news(
    anomaly_times: pd.DatetimeIndex,
    news_df: pd.DataFrame,
    time_window: pd.Timedelta,
    news_time_col: str = "timestamp",
) -> pd.DataFrame:
    """For each anomaly timestamp, finds the nearest news event within
    +/- time_window (if any), via pandas' merge_asof (verified semantics:
    "nearest" direction with a tolerance correctly leaves unmatched
    anomalies as NaT/NaN rather than falling back to a far-away event, and
    one news event can legitimately match multiple nearby anomalies).
    Returns one row per anomaly: [anomaly_time, matched, gap_seconds,
    ...news columns for the match].
    """
    base_columns = ["anomaly_time", "matched", "gap_seconds"]
    if len(anomaly_times) == 0:
        return pd.DataFrame(columns=base_columns)

    # merge_asof requires *identical* datetime64 resolution on both join
    # keys, not just matching timezone -- anomaly_times and news_df
    # commonly come from different code paths (e.g. pandas-native
    # ns-resolution timestamps vs. GDELT's parsed us-resolution ones) and
    # silently differ, raising a MergeError. pd.to_datetime(..., utc=True)
    # alone does NOT fix this: it passes through the existing resolution
    # unchanged when the input is already a proper datetime64 dtype
    # (confirmed the hard way -- an earlier fix using only to_datetime
    # still failed on real GDELT data). Explicitly coerce both sides to a
    # single unit with as_unit().
    target_unit = "us"
    anomaly_df = pd.DataFrame(
        {"anomaly_time": pd.DatetimeIndex(anomaly_times).tz_convert("UTC").as_unit(target_unit)}
    ).sort_values("anomaly_time")

    if news_df.empty:
        anomaly_df["matched"] = False
        anomaly_df["gap_seconds"] = np.nan
        return anomaly_df.reset_index(drop=True)

    news_sorted = news_df.copy()
    news_sorted[news_time_col] = (
        pd.to_datetime(news_sorted[news_time_col], utc=True).dt.as_unit(target_unit)
    )
    news_sorted = news_sorted.sort_values(news_time_col).reset_index(drop=True)
    merged = pd.merge_asof(
        anomaly_df,
        news_sorted,
        left_on="anomaly_time",
        right_on=news_time_col,
        direction="nearest",
        tolerance=time_window,
    )
    merged["matched"] = merged[news_time_col].notna()
    merged["gap_seconds"] = (merged["anomaly_time"] - merged[news_time_col]).abs().dt.total_seconds()
    return merged.reset_index(drop=True)


def correlation_rate(matched_df: pd.DataFrame) -> float:
    """Fraction of anomaly windows that matched a news event within the
    time window. Not statistically meaningful on its own -- see
    attribution/null_control.py's permutation test for significance.
    """
    if matched_df.empty:
        return 0.0
    return float(matched_df["matched"].mean())
