import numpy as np
import pandas as pd
import pytest

from features.returns import build_feature_frame, log_returns, realized_vol, session_boundary_mask, volume_zscore


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


class TestSessionBoundaryMask:
    def test_first_position_is_false(self):
        idx = pd.date_range("2026-01-02 09:30", periods=3, freq="1min", tz="UTC")
        mask = session_boundary_mask(idx)
        assert mask.iloc[0] == False  # noqa: E712

    def test_no_boundary_within_same_session(self):
        idx = pd.date_range("2026-01-02 09:30", periods=5, freq="1min", tz="UTC")
        mask = session_boundary_mask(idx)
        assert not mask.any()

    def test_detects_overnight_boundary(self):
        # Same-week overnight gap (Tuesday close -> Wednesday open), not a weekend.
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-06 15:59", tz="America/New_York"),  # Tuesday close
                pd.Timestamp("2026-01-07 09:30", tz="America/New_York"),  # Wednesday open
            ]
        ).tz_convert("UTC")
        mask = session_boundary_mask(idx)
        assert mask.tolist() == [False, True]

    def test_detects_weekend_boundary(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02 15:59", tz="America/New_York"),  # Friday close
                pd.Timestamp("2026-01-05 09:30", tz="America/New_York"),  # Monday open
            ]
        ).tz_convert("UTC")
        mask = session_boundary_mask(idx)
        assert mask.iloc[1] == True  # noqa: E712


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

    def test_excludes_session_boundary_return(self):
        # Two trading days of 10 bars each with a large synthetic jump
        # planted exactly at the day boundary. If the boundary return
        # weren't excluded, it would dominate log_return (a 50% jump vs.
        # ~0.1%-scale intraday moves) and appear as an extreme outlier.
        day1 = pd.date_range("2026-01-02 09:30", periods=10, freq="1min", tz="America/New_York")
        day2 = pd.date_range("2026-01-05 09:30", periods=10, freq="1min", tz="America/New_York")
        idx = day1.append(day2).tz_convert("UTC")
        rng = np.random.default_rng(3)
        closes_day1 = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 10)))
        closes_day2 = closes_day1[-1] * 1.5 * np.exp(np.cumsum(rng.normal(0, 0.0005, 10)))  # planted jump
        closes = np.concatenate([closes_day1, closes_day2])
        volumes = rng.integers(50, 500, 20).astype(float)
        bars = pd.DataFrame({"timestamp": idx, "close": closes, "volume": volumes})

        frame = build_feature_frame(bars, vol_window=3, volume_window=3)
        assert day2[0].tz_convert("UTC") not in frame.index
        assert frame["log_return"].abs().max() < 0.01  # the ~40% jump never appears

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
