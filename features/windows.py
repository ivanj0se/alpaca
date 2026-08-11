"""Windowing for the TCN-VAE (Rung 4). Splitting/purging logic lives in
benchmark/cv.py -- this module only builds the (N, T, C) window tensor and
a matching set of "as-of" end-times for benchmark/cv.make_t1, and never
reimplements purge/embargo logic itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    feature_frame: pd.DataFrame, window_len: int, stride: int = 1
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """feature_frame: index=timestamp, columns=features (e.g. from
    features/returns.py's build_feature_frame). Returns (windows, end_times):
    windows has shape (N, window_len, n_features); end_times[i] is the
    timestamp of window i's *last* row (its as-of time -- feed this to
    benchmark/cv.make_t1 to build a purge/embargo-aware CV split).
    """
    if window_len < 2:
        raise ValueError("window_len must be at least 2")
    values = feature_frame.to_numpy()
    n_rows = len(feature_frame)
    if window_len > n_rows:
        raise ValueError("window_len cannot exceed the number of rows in feature_frame")

    starts = list(range(0, n_rows - window_len + 1, stride))
    windows = np.stack([values[s : s + window_len] for s in starts])
    end_times = feature_frame.index[[s + window_len - 1 for s in starts]]
    return windows, pd.DatetimeIndex(end_times)
