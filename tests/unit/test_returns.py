import numpy as np
import pandas as pd
import pytest

from features.returns import build_feature_frame, log_returns, realized_vol, volume_zscore


def _price_series(closes, start="2026-01-02 09:30", freq="1min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.Series(closes, index=idx)


class TestLogReturns:
    def test_computes_known_values(self):
        closes = _price_series([100.0, 110.0, 99.0])
        r = log_returns(closes)
        assert len(r) == 2
        assert r.iloc[0] == pytest.approx(np.log(1.10))
        assert r.iloc[1] == pytest.approx(np.log(99.0 / 110.0))

    def test_one_shorter_than_input(self):
        closes = _price_series([100.0] * 10)
        assert len(log_returns(closes)) == 9


class TestRealizedVol:
    def test_first_window_minus_one_values_are_nan(self):
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.001, 20))
        vol = realized_vol(returns, window=5)
        assert vol.iloc[:4].isna().all()
        assert vol.iloc[4:].notna().all()

    def test_constant_returns_give_zero_vol(self):
        returns = pd.Series([0.001] * 20)
        vol = realized_vol(returns, window=5)
        assert (vol.dropna() == 0).all()


class TestVolumeZscore:
    def test_recovers_known_zscore(self):
        # Constant mean/std window, check the formula directly on the last value.
        volume = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0])
        z = volume_zscore(volume, window=4)
        window_vals = volume.iloc[1:5]
        expected = (200.0 - window_vals.mean()) / window_vals.std(ddof=1)
        assert z.iloc[-1] == pytest.approx(expected)

    def test_zero_variance_window_is_nan_not_inf(self):
        volume = pd.Series([100.0] * 10)
        z = volume_zscore(volume, window=5)
        assert not np.isinf(z.dropna()).any()
        assert z.iloc[4:].isna().all()


class TestBuildFeatureFrame:
    def test_output_columns_and_no_nans(self):
        rng = np.random.default_rng(1)
        n = 100
        idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min", tz="UTC")
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        volumes = rng.integers(50, 500, n).astype(float)
        bars = pd.DataFrame({"timestamp": idx, "close": closes, "volume": volumes})

        frame = build_feature_frame(bars, vol_window=10, volume_window=10)
        assert list(frame.columns) == ["log_return", "realized_vol", "volume_zscore"]
        assert not frame.isna().any().any()
        assert len(frame) < n  # leading rows without enough rolling history are dropped

    def test_handles_unsorted_input(self):
        rng = np.random.default_rng(2)
        n = 50
        idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min", tz="UTC")
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        volumes = rng.integers(50, 500, n).astype(float)
        bars = pd.DataFrame({"timestamp": idx, "close": closes, "volume": volumes}).sample(
            frac=1.0, random_state=0
        )  # shuffled
        frame = build_feature_frame(bars, vol_window=10, volume_window=10)
        assert frame.index.is_monotonic_increasing
