"""Per-ticker feature engineering: log returns, realized volatility, and
volume z-score -- the three input channels for the Rung 4 TCN-VAE (temporal
lane, one instrument, no cross-sectional/news information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """log(close_t / close_{t-1}), one element shorter than the input."""
    values = np.log(close.to_numpy())
    diffs = np.diff(values)
    return pd.Series(diffs, index=close.index[1:])


def session_boundary_mask(timestamps: pd.DatetimeIndex, tz: str = "America/New_York") -> pd.Series:
    """True at position i if timestamps[i] falls on a different NYSE
    trading-session date (in exchange-local time) than timestamps[i-1] --
    i.e. this is the first bar of a new session (open, after an overnight
    gap, a weekend, or a holiday). First position is always False (no prior
    bar to compare against).

    Exists because log_returns()/np.diff() has no concept of session
    boundaries -- it happily computes a "1-minute return" between the last
    bar of one session and the first bar of the next, which is actually an
    overnight or weekend return masquerading as an intraday one. Confirmed
    on real AAPL data: these boundary points average 4.4x larger in
    magnitude than genuine intraday returns, and contaminate 6.4% of
    realized_vol rows (3x inflated) via the rolling window that includes
    them (see diagnostics/2026-08-11-session-boundary-returns/findings.md).
    Callers should exclude (not merely flag) returns at these positions --
    an overnight/weekend return is a different statistical animal
    (Berkman et al. 2012), not a noisy version of the same thing.
    """
    local_dates = pd.DatetimeIndex(timestamps).tz_convert(tz).normalize()
    boundary = local_dates.to_series().diff().fillna(pd.Timedelta(0)) != pd.Timedelta(0)
    return pd.Series(boundary.to_numpy(), index=timestamps)


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of returns -- a simple realized-vol proxy.
    The first `window - 1` values are NaN (insufficient history), left as
    NaN rather than filled, consistent with features/bars.py's no-silent-fill
    policy.
    """
    return returns.rolling(window).std(ddof=1)


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of volume: (v - rolling_mean) / rolling_std. A
    rolling std of exactly 0 (e.g. constant volume) produces NaN rather
    than a divide-by-zero inf/nan mismatch.
    """
    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std(ddof=1)
    z = (volume - rolling_mean) / rolling_std
    return z.replace([np.inf, -np.inf], np.nan)


def build_feature_frame(bars: pd.DataFrame, vol_window: int = 30, volume_window: int = 30) -> pd.DataFrame:
    """bars: DataFrame with 'timestamp', 'close', 'volume' columns for a
    single ticker (see ingest/storage.py's BAR_COLUMNS). Returns a frame of
    [log_return, realized_vol, volume_zscore], indexed by timestamp,
    dropping leading rows that don't yet have enough history for the
    rolling windows.

    Returns spanning a session boundary (overnight, weekend, holiday -- see
    session_boundary_mask) are excluded, not just flagged: they get NaN'd
    out here and removed by the final dropna(), the same "fail loud, don't
    silently smooth" policy features/bars.py already uses for missing bars.
    A boundary return is real (not a data error) but is a categorically
    different quantity than an intraday 1-minute return, so mixing it into
    log_return/realized_vol would corrupt both the GARCH/TCN-VAE input
    series and any downstream anomaly score -- a session-open jump would
    get flagged as an "anomaly" independent of any news that morning,
    confounding Rung 5's attribution test specifically.
    """
    bars = bars.sort_values("timestamp")
    close = bars.set_index("timestamp")["close"]
    r = log_returns(close)
    boundary = session_boundary_mask(close.index).loc[r.index]
    r = r.mask(boundary)
    vol = realized_vol(r, vol_window)
    volume_z = volume_zscore(bars.set_index("timestamp")["volume"].iloc[1:], volume_window)

    frame = pd.DataFrame({"log_return": r, "realized_vol": vol, "volume_zscore": volume_z})
    return frame.dropna()
