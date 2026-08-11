import numpy as np
import pandas as pd
import pytest

from baselines.random_walk import GBMParams, simulate_gbm
from events.hawkes import (
    HawkesFitResult,
    _neg_log_likelihood,
    _recursive_sum,
    branching_ratio,
    fit_hawkes_exponential,
    simulate_hawkes,
)
from events.price_events import bar_threshold_events, event_times_array


class TestRecursiveSum:
    def test_matches_naive_double_sum(self):
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 100, 30))
        beta = 0.8
        naive = np.array([sum(np.exp(-beta * (t[i] - t[j])) for j in range(i)) for i in range(len(t))])
        fast = _recursive_sum(t, beta)
        assert np.allclose(naive, fast, atol=1e-10)

    def test_first_element_is_zero(self):
        t = np.array([0.0, 1.0, 2.0])
        assert _recursive_sum(t, 1.0)[0] == 0.0


class TestNegLogLikelihood:
    def test_infinite_for_nonpositive_params(self):
        t = np.array([0.0, 1.0, 2.0])
        assert _neg_log_likelihood(np.array([-1.0, 0.5, 1.0]), t, 3.0) == np.inf
        assert _neg_log_likelihood(np.array([0.5, -1.0, 1.0]), t, 3.0) == np.inf
        assert _neg_log_likelihood(np.array([0.5, 0.5, 0.0]), t, 3.0) == np.inf

    def test_finite_for_valid_params(self):
        t = np.array([0.0, 1.0, 2.0, 3.5])
        val = _neg_log_likelihood(np.array([0.5, 0.3, 1.0]), t, 4.0)
        assert np.isfinite(val)


class TestBranchingRatio:
    def test_computes_alpha_over_beta(self):
        fit = HawkesFitResult(mu=0.1, alpha=0.4, beta=0.8, loglik=-10.0, converged=True, n_events=50)
        assert branching_ratio(fit) == pytest.approx(0.5)


class TestFitHawkesExponential:
    def test_raises_with_fewer_than_two_events(self):
        with pytest.raises(ValueError):
            fit_hawkes_exponential(np.array([1.0]))

    def test_converges_on_simple_synthetic_data(self):
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=3000, seed=1)
        fit = fit_hawkes_exponential(events)
        assert fit.converged
        assert fit.n_events == len(events)


class TestSimulateHawkes:
    def test_rejects_supercritical_alpha(self):
        with pytest.raises(ValueError):
            simulate_hawkes(mu=0.1, alpha=1.0, beta=1.0, T=100)
        with pytest.raises(ValueError):
            simulate_hawkes(mu=0.1, alpha=1.5, beta=1.0, T=100)

    def test_events_sorted_and_within_window(self):
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=1000, seed=0)
        assert np.all(np.diff(events) >= 0)
        assert np.all(events >= 0) and np.all(events < 1000)

    def test_reproducible_with_seed(self):
        e1 = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=1000, seed=7)
        e2 = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=1000, seed=7)
        assert np.array_equal(e1, e2)

    def test_higher_branching_ratio_produces_more_events_on_average(self):
        low = [len(simulate_hawkes(0.05, 0.1, 1.0, T=2000, seed=s)) for s in range(10)]
        high = [len(simulate_hawkes(0.05, 0.8, 1.0, T=2000, seed=s)) for s in range(10)]
        assert np.mean(high) > np.mean(low)


class TestSimulateRefitRecover:
    """The pipeline's own internal correctness check: simulate a Hawkes
    process with known parameters, refit, and confirm recovery -- this
    validates the MLE fitter itself before it's trusted on real SPY data
    in tests/replication/.
    """

    TRUE_MU, TRUE_ALPHA, TRUE_BETA = 0.05, 0.5, 1.0
    T = 5000  # calibrated empirically: ~500 events, tight branching-ratio recovery

    def test_recovers_branching_ratio_within_10_percent(self):
        events = simulate_hawkes(self.TRUE_MU, self.TRUE_ALPHA, self.TRUE_BETA, T=self.T, seed=42)
        fit = fit_hawkes_exponential(events, T=self.T)
        true_ratio = self.TRUE_ALPHA / self.TRUE_BETA
        assert branching_ratio(fit) == pytest.approx(true_ratio, rel=0.10)

    def test_recovers_individual_params_within_25_percent(self):
        events = simulate_hawkes(self.TRUE_MU, self.TRUE_ALPHA, self.TRUE_BETA, T=self.T, seed=42)
        fit = fit_hawkes_exponential(events, T=self.T)
        assert fit.mu == pytest.approx(self.TRUE_MU, rel=0.25)
        assert fit.alpha == pytest.approx(self.TRUE_ALPHA, rel=0.25)
        assert fit.beta == pytest.approx(self.TRUE_BETA, rel=0.25)


class TestNullRandomWalkSpecificity:
    """Complementary specificity check to the SPY replication test: a GBM
    path has i.i.d. increments, so a price-event point process built from
    it should show ~no self-excitation. If this pipeline reported a high
    branching ratio on pure noise, that would mean it hallucinates
    endogeneity -- the SPY replication alone (sensitivity) wouldn't catch
    that failure mode.
    """

    def test_gbm_derived_events_yield_near_zero_branching_ratio(self):
        params = GBMParams(mu=0.0, sigma=0.001)
        paths = simulate_gbm(params, n_steps=20000, n_sims=1, s0=100.0, seed=0)
        idx = pd.date_range("2026-01-02 09:30", periods=paths.shape[1], freq="1min", tz="UTC")
        bars_df = pd.DataFrame(
            {
                "timestamp": idx,
                "ticker": "NULL",
                "open": paths[0],
                "high": paths[0],
                "low": paths[0],
                "close": paths[0],
                "volume": 100,
            }
        )
        events_df = bar_threshold_events(bars_df, sigma_threshold=2.0)
        event_times = event_times_array(events_df, ticker="NULL")
        assert len(event_times) > 20, "need enough synthetic events for a meaningful fit"

        fit = fit_hawkes_exponential(event_times)
        assert branching_ratio(fit) < 0.15, (
            "GBM increments are i.i.d. -- a well-behaved pipeline should not "
            "detect substantial self-excitation in pure random-walk noise"
        )
