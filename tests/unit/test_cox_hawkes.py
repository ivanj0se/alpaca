import numpy as np
import pytest

from events.hawkes import fit_hawkes_exponential, simulate_hawkes
from research.cox_hawkes import CoxHawkesFitResult, fit_cox_hawkes, simulate_cox_hawkes


class TestFitCoxHawkes:
    def test_raises_with_fewer_than_two_events(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_cox_hawkes(
                np.array([1.0]), np.array([0.0]), beta=1.0, grid_covariate=np.array([0.0]), grid_durations=np.array([1.0])
            )

    def test_raises_on_mismatched_covariate_length(self):
        with pytest.raises(ValueError, match="one value per event"):
            fit_cox_hawkes(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0]),  # wrong length
                beta=1.0,
                grid_covariate=np.array([0.0]),
                grid_durations=np.array([1.0]),
            )

    def test_raises_on_nonpositive_beta(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            fit_cox_hawkes(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 0.0]),
                beta=0.0,
                grid_covariate=np.array([0.0]),
                grid_durations=np.array([1.0]),
            )

    def test_raises_on_grid_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            fit_cox_hawkes(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 0.0, 0.0]),
                beta=1.0,
                grid_covariate=np.array([0.0, 0.0]),
                grid_durations=np.array([1.0]),
            )

    def test_zero_covariate_reduces_to_standard_hawkes_fit(self):
        # With event_covariate and grid_covariate all zero, mu(t) =
        # mu0*exp(gamma*0) = mu0 for ANY gamma -- the model degenerates
        # to a standard constant-baseline Hawkes process regardless of
        # gamma. Fix beta at the standard fitter's OWN converged value
        # (beta0 is only a starting point there, not fixed -- it moved to
        # 1.128, not the nominal 1.0) for a true apples-to-apples
        # comparison; mu0/alpha should then closely match.
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=5000, seed=1)
        standard_fit = fit_hawkes_exponential(events, beta0=1.0)

        n = len(events)
        grid_n = 500
        grid_durations = np.full(grid_n, 5000.0 / grid_n)
        result = fit_cox_hawkes(
            events,
            event_covariate=np.zeros(n),
            beta=standard_fit.beta,
            grid_covariate=np.zeros(grid_n),
            grid_durations=grid_durations,
            T=5000.0,
        )
        assert isinstance(result, CoxHawkesFitResult)
        assert result.mu0 == pytest.approx(standard_fit.mu, rel=0.05)
        assert result.alpha == pytest.approx(standard_fit.alpha, rel=0.05)

    def test_detects_correct_sign_of_covariate_effect(self):
        # Construct two segments: a LOW-covariate, LOW-event-density
        # segment and a HIGH-covariate, HIGH-event-density segment
        # (concatenated). A real positive relationship between the
        # covariate and event rate should recover gamma > 0.
        rng = np.random.default_rng(2)
        low_events = np.sort(rng.uniform(0, 2000, 40))  # sparse
        high_events = np.sort(rng.uniform(2000, 4000, 400))  # dense
        event_times = np.concatenate([low_events, high_events])
        event_covariate = np.concatenate([np.zeros(40), np.ones(400)])

        grid_covariate = np.concatenate([np.zeros(200), np.ones(200)])
        grid_durations = np.full(400, 4000.0 / 400)

        result = fit_cox_hawkes(
            event_times,
            event_covariate,
            beta=1.0,
            grid_covariate=grid_covariate,
            grid_durations=grid_durations,
            T=4000.0,
        )
        assert result.gamma > 0

    def test_converged_flag_true_on_well_behaved_data(self):
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=3000, seed=3)
        n = len(events)
        result = fit_cox_hawkes(
            events,
            event_covariate=np.zeros(n),
            beta=1.0,
            grid_covariate=np.zeros(300),
            grid_durations=np.full(300, 10.0),
            T=3000.0,
        )
        assert result.converged


class TestSimulateCoxHawkes:
    def test_rejects_supercritical_alpha(self):
        with pytest.raises(ValueError, match="branching ratio"):
            simulate_cox_hawkes(
                mu0=0.1, gamma=0.0, alpha=1.0, beta=1.0, grid_covariate=np.zeros(10), grid_durations=np.full(10, 10.0)
            )

    def test_raises_on_grid_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            simulate_cox_hawkes(
                mu0=0.1, gamma=0.0, alpha=0.4, beta=1.0, grid_covariate=np.zeros(10), grid_durations=np.full(5, 10.0)
            )

    def test_events_sorted_and_within_window(self):
        events = simulate_cox_hawkes(
            mu0=0.05, gamma=0.5, alpha=0.4, beta=1.0,
            grid_covariate=np.zeros(100), grid_durations=np.full(100, 10.0), seed=0,
        )
        T = 1000.0
        assert np.all(np.diff(events) >= 0)
        assert np.all(events >= 0) and np.all(events < T)

    def test_reproducible_with_seed(self):
        kwargs = dict(
            mu0=0.05, gamma=0.5, alpha=0.4, beta=1.0,
            grid_covariate=np.zeros(100), grid_durations=np.full(100, 10.0), seed=7,
        )
        e1 = simulate_cox_hawkes(**kwargs)
        e2 = simulate_cox_hawkes(**kwargs)
        assert np.array_equal(e1, e2)

    def test_zero_gamma_matches_constant_mu_simulator_in_distribution(self):
        # gamma=0 collapses mu(t) = mu0*exp(0*x) = mu0 for any covariate --
        # should be statistically indistinguishable from the plain
        # constant-mu simulator at the same mu0/alpha/beta, the same
        # cross-implementation consistency check already used elsewhere
        # in this project (e.g. multi-kernel K=1 vs single-kernel).
        grid_covariate = np.concatenate([np.full(50, -3.0), np.full(50, 3.0)])  # varies, but gamma=0 nullifies it
        grid_durations = np.full(100, 20.0)  # T = 2000
        cox_counts = [
            len(simulate_cox_hawkes(0.05, 0.0, 0.4, 1.0, grid_covariate, grid_durations, seed=s)) for s in range(20)
        ]
        plain_counts = [len(simulate_hawkes(0.05, 0.4, 1.0, T=2000, seed=s)) for s in range(20)]
        assert np.mean(cox_counts) == pytest.approx(np.mean(plain_counts), rel=0.15)

    def test_positive_gamma_produces_more_events_when_covariate_is_higher(self):
        low_covariate = np.full(100, -1.0)
        high_covariate = np.full(100, 1.0)
        grid_durations = np.full(100, 10.0)
        low = [
            len(simulate_cox_hawkes(0.05, 1.0, 0.2, 1.0, low_covariate, grid_durations, seed=s)) for s in range(15)
        ]
        high = [
            len(simulate_cox_hawkes(0.05, 1.0, 0.2, 1.0, high_covariate, grid_durations, seed=s)) for s in range(15)
        ]
        assert np.mean(high) > np.mean(low)


class TestCoxHawkesSimulateRefitRecover:
    """Same trust-gate pattern as tests/unit/test_hawkes.py's
    TestSimulateRefitRecover, generalized to the covariate-driven
    baseline: simulate with known (mu0, gamma, alpha), refit at the SAME
    fixed beta, confirm recovery -- validates simulate_cox_hawkes and
    fit_cox_hawkes are mutually consistent before either is trusted on
    real data.
    """

    TRUE_MU0, TRUE_GAMMA, TRUE_ALPHA, TRUE_BETA = 0.05, 0.8, 0.4, 1.0

    def test_recovers_gamma_sign_and_rough_magnitude(self):
        rng = np.random.default_rng(1)
        grid_covariate = rng.normal(0, 1, 400)
        grid_durations = np.full(400, 20.0)  # T = 8000
        events = simulate_cox_hawkes(
            self.TRUE_MU0, self.TRUE_GAMMA, self.TRUE_ALPHA, self.TRUE_BETA,
            grid_covariate, grid_durations, seed=42,
        )
        grid_starts = np.concatenate([[0.0], np.cumsum(grid_durations)[:-1]])
        event_covariate = grid_covariate[np.searchsorted(grid_starts, events, side="right") - 1]
        fit = fit_cox_hawkes(
            events, event_covariate, beta=self.TRUE_BETA,
            grid_covariate=grid_covariate, grid_durations=grid_durations, T=float(grid_durations.sum()),
        )
        assert fit.gamma > 0  # correct sign recovered
        assert fit.gamma == pytest.approx(self.TRUE_GAMMA, rel=0.5)  # loose -- one realization, not a full study
