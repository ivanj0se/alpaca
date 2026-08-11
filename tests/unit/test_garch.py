import types

import numpy as np
import pandas as pd
import pytest

from baselines.garch import (
    fit_garch,
    forecast_one_step_variance,
    garch_anomaly_score,
    make_fold_scorer,
    out_of_sample_neg_log_likelihood,
)
from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung


def _simulate_garch11(omega, alpha, beta, n, seed=0):
    """Simulate a GARCH(1,1) process with known parameters -- the reference
    generator for this module's self-test, same discipline as
    events/hawkes.py's simulate-refit-recover check.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    r = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * rng.normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.normal()
    return r


def _returns_series(values, freq="1min"):
    idx = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


class TestFitGarch:
    def test_raises_on_too_few_observations(self):
        with pytest.raises(ValueError):
            fit_garch(_returns_series(np.random.default_rng(0).normal(0, 0.001, 10)))

    def test_adaptive_scale_targets_reasonable_range_regardless_of_data_magnitude(self):
        rng = np.random.default_rng(0)
        tiny = _returns_series(rng.normal(0, 0.0005, 500))  # intraday-minute-like
        big = _returns_series(rng.normal(0, 0.02, 500))  # daily-like
        fit_tiny = fit_garch(tiny)
        fit_big = fit_garch(big)
        # Both fits should use a scale that brings the rescaled series' std
        # near the target (10), regardless of the raw data's magnitude.
        assert fit_tiny.scale * tiny.std() == pytest.approx(10.0, rel=0.05)
        assert fit_big.scale * big.std() == pytest.approx(10.0, rel=0.05)


class TestSimulateRefitRecover:
    TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 1e-6, 0.1, 0.85
    N = 10000

    def test_recovers_alpha_beta_within_tolerance(self):
        r = _simulate_garch11(self.TRUE_OMEGA, self.TRUE_ALPHA, self.TRUE_BETA, self.N, seed=42)
        fit = fit_garch(_returns_series(r))
        alpha = fit.model_result.params["alpha[1]"]
        beta = fit.model_result.params["beta[1]"]
        assert alpha == pytest.approx(self.TRUE_ALPHA, abs=0.04)
        assert beta == pytest.approx(self.TRUE_BETA, rel=0.15)
        persistence = alpha + beta
        assert persistence == pytest.approx(self.TRUE_ALPHA + self.TRUE_BETA, rel=0.10)


class TestGarchAnomalyScore:
    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        r = _simulate_garch11(1e-6, 0.1, 0.85, 2000, seed=1)
        fit = fit_garch(_returns_series(r))
        score = garch_anomaly_score(fit)
        assert (score >= 0).all()

    def test_large_return_gets_higher_score_than_typical_return(self):
        r = _simulate_garch11(1e-6, 0.1, 0.85, 2000, seed=1)
        r = r.copy()
        r[-1] = r.std() * 15  # inject an obvious outlier at the end
        fit = fit_garch(_returns_series(r))
        score = garch_anomaly_score(fit)
        assert score.iloc[-1] > score.iloc[:-1].median() * 3


class TestForecastAndNll:
    def test_forecast_variance_positive(self):
        r = _simulate_garch11(1e-6, 0.1, 0.85, 1000, seed=2)
        fit = fit_garch(_returns_series(r))
        assert forecast_one_step_variance(fit) > 0

    def test_nll_matches_manual_gaussian_formula(self):
        # Pure math check, independent of any fitted model.
        variance = 4.0e-6
        next_return = 0.001
        expected = 0.5 * (np.log(2 * np.pi * variance) + next_return**2 / variance)

        fake_forecast_result = types.SimpleNamespace(variance=pd.DataFrame([[variance]]))
        fake_model_result = types.SimpleNamespace(forecast=lambda horizon, reindex: fake_forecast_result)
        fake_fit = types.SimpleNamespace(scale=1.0, model_result=fake_model_result)

        actual = out_of_sample_neg_log_likelihood(fake_fit, next_return)
        assert actual == pytest.approx(expected)


class TestMakeFoldScorer:
    def _returns(self, n=3000, seed=0):
        r = _simulate_garch11(1e-6, 0.1, 0.85, n, seed=seed)
        return _returns_series(r)

    def test_returns_finite_score_on_a_reasonable_fold(self):
        returns = self._returns()
        scorer = make_fold_scorer(returns)
        n = len(returns)
        train_idx = np.arange(0, n - 200)
        test_idx = np.arange(n - 200, n)
        score = scorer(train_idx, test_idx)
        assert score is not None
        assert np.isfinite(score)

    def test_returns_none_for_too_short_training_fold(self):
        returns = self._returns()
        scorer = make_fold_scorer(returns)
        score = scorer(np.arange(0, 5), np.arange(5, 10))
        assert score is None

    def test_integrates_with_evaluate_rung(self):
        returns = self._returns(n=4000)
        t1 = make_t1(returns.index, pd.Timedelta(minutes=1))
        result = evaluate_rung("garch", make_fold_scorer(returns), t1, n_splits=5, embargo_td=pd.Timedelta(minutes=10))
        assert result.n_folds > 0
        assert np.isfinite(result.mean_nll)
