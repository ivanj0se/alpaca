import numpy as np
import pandas as pd
import pytest

from research.rough_volatility import block_realized_vol, block_realized_vol_by_session, estimate_roughness


def _minute_returns(n, seed=0, start="2026-01-02 09:30"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.Series(rng.normal(0, 0.001, n), index=idx)


class TestBlockRealizedVol:
    def test_produces_non_overlapping_blocks(self):
        r = _minute_returns(100)
        rv = block_realized_vol(r, block_minutes=10)
        # 100 minutes / 10-minute blocks -> 10 blocks
        assert len(rv) == 10

    def test_drops_short_trailing_partial_block(self):
        r = _minute_returns(92)  # 9 full 10-min blocks + a 2-min partial (clearly below the 5-min threshold)
        rv = block_realized_vol(r, block_minutes=10)
        assert len(rv) == 9

    def test_keeps_partial_block_at_exactly_half_threshold(self):
        r = _minute_returns(95)  # 9 full blocks + a 5-min partial, exactly at the >=50% keep threshold
        rv = block_realized_vol(r, block_minutes=10)
        assert len(rv) == 10

    def test_empty_for_too_short_input(self):
        r = _minute_returns(1)
        rv = block_realized_vol(r, block_minutes=10)
        assert len(rv) == 0

    def test_values_are_positive(self):
        r = _minute_returns(100)
        rv = block_realized_vol(r, block_minutes=10)
        assert (rv > 0).all()


class TestBlockRealizedVolBySession:
    def test_splits_at_session_boundaries(self):
        day1 = _minute_returns(100, seed=0, start="2026-01-02 09:30")
        day2 = _minute_returns(100, seed=1, start="2026-01-05 09:30")
        combined = pd.concat([day1, day2])
        sessions = block_realized_vol_by_session(combined, block_minutes=10)
        assert len(sessions) == 2  # two independent sessions, not one contaminated series

    def test_single_session_returns_one_series(self):
        r = _minute_returns(200)
        sessions = block_realized_vol_by_session(r, block_minutes=10)
        assert len(sessions) == 1


class TestEstimateRoughness:
    def test_iid_log_vol_gives_hurst_near_zero(self):
        # log(RV) i.i.d. across blocks (no cumulative structure) -> the
        # structure function E[|diff|] doesn't depend on lag at all for
        # lag>=1 (consecutive values are independent regardless of lag),
        # so the log-log slope (Hurst estimate) should be ~0 -- an
        # analytically clean "maximally rough" reference case.
        rng = np.random.default_rng(0)
        n_sessions = 30
        session_rvs = [
            pd.Series(np.exp(rng.normal(0, 1, 60)), index=pd.RangeIndex(60)) for _ in range(n_sessions)
        ]
        result = estimate_roughness(session_rvs, lags=(1, 2, 3, 5, 8, 12))
        assert abs(result.hurst) < 0.15

    def test_random_walk_log_vol_gives_hurst_near_half(self):
        # log(RV) as a cumulative random walk -> Var(diff) scales linearly
        # with lag, so E[|diff|] ~ lag^0.5 exactly for Gaussian increments
        # -- the analytically known H=0.5 reference case (standard
        # Brownian motion, not rough).
        rng = np.random.default_rng(1)
        n_sessions = 30
        session_rvs = []
        for _ in range(n_sessions):
            steps = rng.normal(0, 1, 80)
            log_rv = np.cumsum(steps)
            session_rvs.append(pd.Series(np.exp(log_rv), index=pd.RangeIndex(80)))
        result = estimate_roughness(session_rvs, lags=(1, 2, 3, 5, 8, 12, 16))
        assert 0.35 < result.hurst < 0.65

    def test_raises_on_empty_sessions(self):
        with pytest.raises(ValueError, match="no sessions"):
            estimate_roughness([])

    def test_r_squared_is_high_for_clean_synthetic_data(self):
        # The two analytically-known cases above should also fit very
        # cleanly (high R^2) -- if either came out with a reasonable
        # Hurst estimate but a poor fit, that would itself be suspicious.
        rng = np.random.default_rng(2)
        session_rvs = [
            pd.Series(np.exp(np.cumsum(rng.normal(0, 1, 80))), index=pd.RangeIndex(80)) for _ in range(30)
        ]
        result = estimate_roughness(session_rvs, lags=(1, 2, 3, 5, 8, 12, 16))
        assert result.r_squared > 0.9
