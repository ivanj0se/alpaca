import numpy as np
import pandas as pd
import pytest

from features.windows import make_windows


def _feature_frame(n=20, n_features=3):
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min", tz="UTC")
    data = np.arange(n * n_features).reshape(n, n_features).astype(float)
    return pd.DataFrame(data, index=idx, columns=[f"f{i}" for i in range(n_features)])


class TestMakeWindows:
    def test_output_shape_step_one(self):
        frame = _feature_frame(n=20, n_features=3)
        windows, end_times = make_windows(frame, window_len=5, stride=1)
        assert windows.shape == (16, 5, 3)  # 20 - 5 + 1 = 16
        assert len(end_times) == 16

    def test_windows_contain_correct_values(self):
        frame = _feature_frame(n=10, n_features=2)
        windows, end_times = make_windows(frame, window_len=3, stride=1)
        assert np.array_equal(windows[0], frame.iloc[0:3].to_numpy())
        assert np.array_equal(windows[1], frame.iloc[1:4].to_numpy())
        assert end_times[0] == frame.index[2]
        assert end_times[1] == frame.index[3]

    def test_stride_greater_than_one(self):
        frame = _feature_frame(n=20, n_features=2)
        windows, end_times = make_windows(frame, window_len=5, stride=5)
        assert windows.shape[0] == 4  # starts at 0, 5, 10, 15
        assert end_times[1] == frame.index[9]

    def test_raises_when_window_exceeds_data(self):
        frame = _feature_frame(n=5)
        with pytest.raises(ValueError, match="cannot exceed"):
            make_windows(frame, window_len=10)

    def test_raises_when_window_too_small(self):
        frame = _feature_frame(n=5)
        with pytest.raises(ValueError, match="at least 2"):
            make_windows(frame, window_len=1)
