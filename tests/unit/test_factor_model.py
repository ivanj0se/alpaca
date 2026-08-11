import numpy as np
import pandas as pd
import pytest

from baselines.factor_model import (
    factor_model_anomaly_score,
    fit_linear_factor_model,
    fit_universe_factor_models,
    make_fold_scorer,
)
from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung


def _series(values, start="2026-01-02 09:30", freq="1min"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


class TestFitLinearFactorModel:
    def test_recovers_known_alpha_and_beta_exactly_noiseless(self):
        rng = np.random.default_rng(0)
        market = _series(rng.normal(0, 0.001, 200))
        true_alpha, true_beta = 0.0002, 1.3
        y = true_alpha + true_beta * market.to_numpy()
        returns = pd.Series(y, index=market.index)

        fit = fit_linear_factor_model(returns, market, ticker="TEST")
        assert fit.alpha == pytest.approx(true_alpha, abs=1e-10)
        assert fit.beta == pytest.approx(true_beta, abs=1e-8)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(fit.residuals.to_numpy(), 0.0, atol=1e-8)

    def test_recovers_known_beta_with_noise_within_tolerance(self):
        rng = np.random.default_rng(1)
        market = _series(rng.normal(0, 0.001, 5000))
        true_alpha, true_beta = 0.0, 0.8
        noise = rng.normal(0, 0.0005, 5000)
        y = true_alpha + true_beta * market.to_numpy() + noise
        returns = pd.Series(y, index=market.index)

        fit = fit_linear_factor_model(returns, market)
        assert fit.beta == pytest.approx(true_beta, rel=0.1)
        assert 0 <= fit.r_squared <= 1

    def test_raises_with_fewer_than_three_observations(self):
        idx = pd.date_range("2026-01-02", periods=2, freq="1min", tz="UTC")
        with pytest.raises(ValueError, match="at least 3"):
            fit_linear_factor_model(pd.Series([0.001, 0.002], index=idx), pd.Series([0.001, 0.002], index=idx))

    def test_raises_when_market_returns_have_zero_variance(self):
        n = 10
        idx = pd.date_range("2026-01-02", periods=n, freq="1min", tz="UTC")
        market = pd.Series([0.0] * n, index=idx)
        returns = pd.Series(np.linspace(0.001, 0.002, n), index=idx)
        with pytest.raises(ValueError, match="zero variance"):
            fit_linear_factor_model(returns, market)

    def test_drops_unaligned_timestamps_rather_than_erroring(self):
        idx1 = pd.date_range("2026-01-02 09:30", periods=10, freq="1min", tz="UTC")
        idx2 = pd.date_range("2026-01-02 09:32", periods=10, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.001, 10), index=idx1)
        market = pd.Series(rng.normal(0, 0.001, 10), index=idx2)
        fit = fit_linear_factor_model(returns, market)
        assert len(fit.residuals) == 8  # 8 overlapping timestamps


class TestFactorModelAnomalyScore:
    def test_nonnegative(self):
        rng = np.random.default_rng(2)
        market = _series(rng.normal(0, 0.001, 500))
        returns = pd.Series(0.5 * market.to_numpy() + rng.normal(0, 0.0003, 500), index=market.index)
        fit = fit_linear_factor_model(returns, market)
        score = factor_model_anomaly_score(fit)
        assert (score >= 0).all()

    def test_zero_residual_std_does_not_crash(self):
        # Perfect linear fit -> zero residuals -> zero std, must not divide by zero.
        market = _series([0.001, 0.002, 0.003, 0.004])
        returns = pd.Series(2 * market.to_numpy(), index=market.index)
        fit = fit_linear_factor_model(returns, market)
        score = factor_model_anomaly_score(fit)
        assert (score == 0).all()

    def test_outlier_gets_higher_score(self):
        rng = np.random.default_rng(3)
        market = _series(rng.normal(0, 0.001, 300))
        y = 0.5 * market.to_numpy() + rng.normal(0, 0.0002, 300)
        y[-1] += 0.01  # inject an obvious idiosyncratic outlier
        returns = pd.Series(y, index=market.index)
        fit = fit_linear_factor_model(returns, market)
        score = factor_model_anomaly_score(fit)
        assert score.iloc[-1] > score.iloc[:-1].median() * 3


class TestFitUniverseFactorModels:
    def test_fits_one_model_per_ticker(self):
        rng = np.random.default_rng(4)
        market = _series(rng.normal(0, 0.001, 300))
        returns_by_ticker = {
            "AAPL": pd.Series(0.5 * market.to_numpy() + rng.normal(0, 0.0003, 300), index=market.index),
            "MSFT": pd.Series(0.8 * market.to_numpy() + rng.normal(0, 0.0003, 300), index=market.index),
        }
        fits = fit_universe_factor_models(returns_by_ticker, market)
        assert set(fits.keys()) == {"AAPL", "MSFT"}
        assert fits["AAPL"].ticker == "AAPL"
        assert fits["MSFT"].ticker == "MSFT"


class TestMakeFoldScorer:
    def _aligned_pair(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        market = _series(rng.normal(0, 0.001, n))
        returns = pd.Series(0.5 * market.to_numpy() + rng.normal(0, 0.0003, n), index=market.index)
        return returns, market

    def test_raises_on_mismatched_index(self):
        returns, market = self._aligned_pair()
        shifted_market = market.copy()
        shifted_market.index = shifted_market.index + pd.Timedelta(minutes=1)
        with pytest.raises(ValueError, match="same index"):
            make_fold_scorer(returns, shifted_market)

    def test_returns_finite_score_on_a_reasonable_fold(self):
        returns, market = self._aligned_pair()
        scorer = make_fold_scorer(returns, market)
        n = len(returns)
        score = scorer(np.arange(0, n - 200), np.arange(n - 200, n))
        assert score is not None
        assert np.isfinite(score)

    def test_returns_none_for_too_short_training_fold(self):
        returns, market = self._aligned_pair()
        scorer = make_fold_scorer(returns, market)
        assert scorer(np.arange(0, 2), np.arange(2, 5)) is None

    def test_returns_none_when_residual_variance_is_zero(self):
        # Perfect noiseless linear relationship -> zero residual variance
        # in the training fold -> must not divide by zero.
        market = _series(np.linspace(0.0001, 0.002, 100))
        returns = pd.Series(2.0 * market.to_numpy(), index=market.index)
        scorer = make_fold_scorer(returns, market)
        assert scorer(np.arange(0, 80), np.arange(80, 100)) is None

    def test_integrates_with_evaluate_rung(self):
        returns, market = self._aligned_pair(n=3000)
        t1 = make_t1(returns.index, pd.Timedelta(minutes=1))
        result = evaluate_rung(
            "factor_model", make_fold_scorer(returns, market), t1, n_splits=5, embargo_td=pd.Timedelta(minutes=10)
        )
        assert result.n_folds > 0
        assert np.isfinite(result.mean_nll)
