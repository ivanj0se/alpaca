import numpy as np
import pandas as pd
import pytest

from baselines.random_walk import estimate_gbm_params, simulate_gbm, simulate_matched_to


class TestEstimateGbmParams:
    def test_recovers_known_constant_drift(self):
        # Deterministic log returns of exactly 0.001 -> mu=0.001, sigma=0.
        prices = pd.Series(100.0 * np.exp(np.cumsum([0.001] * 50)))
        params = estimate_gbm_params(prices)
        assert params.mu == pytest.approx(0.001, abs=1e-9)
        assert params.sigma == pytest.approx(0.0, abs=1e-9)

    def test_raises_on_too_short_series(self):
        with pytest.raises(ValueError):
            estimate_gbm_params(pd.Series([100.0, 100.1]))


class TestSimulateGbm:
    def test_output_shape(self):
        from baselines.random_walk import GBMParams

        paths = simulate_gbm(GBMParams(mu=0.0, sigma=0.01), n_steps=20, n_sims=5, seed=0)
        assert paths.shape == (5, 21)

    def test_first_column_is_s0(self):
        from baselines.random_walk import GBMParams

        paths = simulate_gbm(GBMParams(mu=0.0, sigma=0.01), n_steps=10, n_sims=3, s0=50.0, seed=0)
        assert np.allclose(paths[:, 0], 50.0)

    def test_reproducible_with_seed(self):
        from baselines.random_walk import GBMParams

        params = GBMParams(mu=0.0001, sigma=0.01)
        p1 = simulate_gbm(params, n_steps=50, n_sims=4, seed=42)
        p2 = simulate_gbm(params, n_steps=50, n_sims=4, seed=42)
        assert np.array_equal(p1, p2)

    def test_recovers_matched_moments_over_many_sims(self):
        from baselines.random_walk import GBMParams

        params = GBMParams(mu=0.0005, sigma=0.02)
        paths = simulate_gbm(params, n_steps=200, n_sims=20000, seed=0)
        log_returns = np.diff(np.log(paths), axis=1).flatten()
        assert np.mean(log_returns) == pytest.approx(params.mu, abs=2e-4)
        assert np.std(log_returns) == pytest.approx(params.sigma, rel=0.05)


class TestSimulateMatchedTo:
    def test_matches_length_and_start_price(self):
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.full(30, 0.0002))))
        paths = simulate_matched_to(prices, n_sims=2, seed=0)
        assert paths.shape == (2, len(prices))
        assert np.allclose(paths[:, 0], prices.iloc[0])
