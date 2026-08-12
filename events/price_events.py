"""Constructs point processes of "price events" for the Hawkes rung. Two
sources, explicitly labeled by provenance:

- `bar_threshold_events`: a coarse proxy from minute-bar returns (available
  immediately, from historical data).
- `tick_events_from_recorder`: real trade-level events (only available once
  ingest/tick_recorder.py has accumulated enough live data).

The bar proxy likely biases the branching ratio downward relative to true
tick-level order flow (see docs/architecture.md) -- this is exactly why the
two sources are kept distinguishable rather than silently merged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.returns import session_boundary_mask


def bar_threshold_events(bars_df: pd.DataFrame, sigma_threshold: float = 2.0) -> pd.DataFrame:
    """Per-ticker point process: a bar counts as an "event" if its log
    return's absolute z-score (relative to that ticker's own return
    distribution) exceeds `sigma_threshold`. Returns columns
    [timestamp, ticker, abs_zscore], sorted by timestamp, provenance="bar_proxy".

    Session-boundary returns (overnight/weekend/holiday -- see
    features.returns.session_boundary_mask) are excluded before the z-score
    is even computed: left in, every session open would register as an
    oversized return and get flagged as an "event" on a perfectly regular
    once-a-day cadence, which is the opposite of the self-exciting
    clustering this rung is trying to measure and would bias both the
    z-score baseline (std) and the branching ratio estimate (see
    diagnostics/2026-08-11-session-boundary-returns/findings.md).
    """
    if bars_df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "abs_zscore", "provenance"])

    frames = []
    for ticker, group in bars_df.sort_values("timestamp").groupby("ticker"):
        timestamps = pd.DatetimeIndex(group["timestamp"])
        close = group["close"].to_numpy()
        log_returns = np.diff(np.log(close))
        if len(log_returns) == 0:
            continue
        boundary = session_boundary_mask(timestamps).to_numpy()[1:]
        log_returns = log_returns[~boundary]
        event_times_all = timestamps.to_numpy()[1:][~boundary]
        if len(log_returns) == 0:
            continue
        std = log_returns.std(ddof=1)
        if std == 0:
            continue
        z = np.abs(log_returns) / std
        event_mask = z >= sigma_threshold
        if not event_mask.any():
            continue
        event_times = event_times_all[event_mask]
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": event_times,
                    "ticker": ticker,
                    "abs_zscore": z[event_mask],
                    "provenance": "bar_proxy",
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["timestamp", "ticker", "abs_zscore", "provenance"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def tick_events_from_recorder(ticks_df: pd.DataFrame, sigma_threshold: float = 2.0) -> pd.DataFrame:
    """Same construction as bar_threshold_events but on real trade-level
    prices -- every trade is a "tick", and events are trades whose price
    change from the previous trade (same ticker) has an unusually large
    z-scored magnitude. provenance="real_tick". Session-boundary price
    changes are excluded for the same reason as bar_threshold_events.
    """
    columns = ["timestamp", "ticker", "abs_zscore", "provenance"]
    if ticks_df.empty:
        return pd.DataFrame(columns=columns)

    frames = []
    for ticker, group in ticks_df.sort_values("timestamp").groupby("ticker"):
        timestamps = pd.DatetimeIndex(group["timestamp"])
        price = group["price"].to_numpy()
        if len(price) < 3:
            continue
        log_returns = np.diff(np.log(price))
        boundary = session_boundary_mask(timestamps).to_numpy()[1:]
        log_returns = log_returns[~boundary]
        event_times_all = timestamps.to_numpy()[1:][~boundary]
        if len(log_returns) == 0:
            continue
        std = log_returns.std(ddof=1)
        if std == 0:
            continue
        z = np.abs(log_returns) / std
        event_mask = z >= sigma_threshold
        if not event_mask.any():
            continue
        event_times = event_times_all[event_mask]
        frames.append(
            pd.DataFrame(
                {"timestamp": event_times, "ticker": ticker, "abs_zscore": z[event_mask], "provenance": "real_tick"}
            )
        )

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def merge_event_sources(*event_frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate event frames (e.g. bar-proxy and real-tick) while
    keeping the `provenance` column intact so downstream consumers (the
    Hawkes fitter, diagnostics) can distinguish or filter by source rather
    than silently blending different-quality event definitions.
    """
    columns = ["timestamp", "ticker", "abs_zscore", "provenance"]
    non_empty = [f for f in event_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def event_times_array(events_df: pd.DataFrame, ticker: str | None = None) -> np.ndarray:
    """Extract a sorted array of seconds-since-first-event, the format
    events/hawkes.py's fitter and simulator expect. Optionally filtered to
    one ticker (a Hawkes fit is per-instrument, not pooled across tickers).
    """
    df = events_df if ticker is None else events_df[events_df["ticker"] == ticker]
    if df.empty:
        return np.array([])
    times = pd.to_datetime(df["timestamp"]).sort_values()
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    return seconds
