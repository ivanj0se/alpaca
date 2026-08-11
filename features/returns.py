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
    """
    bars = bars.sort_values("timestamp")
    r = log_returns(bars.set_index("timestamp")["close"])
    vol = realized_vol(r, vol_window)
    volume_z = volume_zscore(bars.set_index("timestamp")["volume"].iloc[1:], volume_window)

    frame = pd.DataFrame({"log_return": r, "realized_vol": vol, "volume_zscore": volume_z})
    return frame.dropna()
