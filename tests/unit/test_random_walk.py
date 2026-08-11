import numpy as np
import pandas as pd
import pytest

from baselines.random_walk import estimate_gbm_params, make_fold_scorer, simulate_gbm, simulate_matched_to
from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung


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


class TestMakeFoldScorer:
    def _returns(self, n=500, seed=0):
        idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min", tz="UTC")
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0, 0.001, n), index=idx)

    def test_returns_none_for_too_short_training_fold(self):
        returns = self._returns(n=50)
        scorer = make_fold_scorer(returns)
        assert scorer(np.arange(0, 5), np.arange(5, 10)) is None

    def test_returns_finite_score_on_a_reasonable_fold(self):
        returns = self._returns(n=300)
        scorer = make_fold_scorer(returns)
        n = len(returns)
        score = scorer(np.arange(0, n - 50), np.arange(n - 50, n))
        assert score is not None
        assert np.isfinite(score)

    def test_returns_none_for_zero_variance_training_fold(self):
        idx = pd.date_range("2026-01-02 09:30", periods=50, freq="1min", tz="UTC")
        constant_returns = pd.Series([0.0] * 50, index=idx)
        scorer = make_fold_scorer(constant_returns)
        assert scorer(np.arange(0, 30), np.arange(30, 50)) is None

    def test_matches_manual_nll_computation(self):
        returns = self._returns(n=200)
        scorer = make_fold_scorer(returns)
        train_idx, test_idx = np.arange(0, 150), np.arange(150, 200)
        score = scorer(train_idx, test_idx)

        variance = float(np.var(returns.iloc[train_idx].to_numpy(), ddof=1))
        test_vals = returns.iloc[test_idx].to_numpy()
        expected = float(np.mean(0.5 * (np.log(2 * np.pi * variance) + test_vals**2 / variance)))
        assert score == pytest.approx(expected)

    def test_integrates_with_evaluate_rung(self):
        returns = self._returns(n=600)
        t1 = make_t1(returns.index, pd.Timedelta(minutes=1))
        result = evaluate_rung("random_walk", make_fold_scorer(returns), t1, n_splits=5, embargo_td=pd.Timedelta(minutes=10))
        assert result.n_folds > 0
        assert np.isfinite(result.mean_nll)
