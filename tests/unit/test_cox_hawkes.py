import numpy as np
import pytest

from events.hawkes import fit_hawkes_exponential, simulate_hawkes
from research.cox_hawkes import CoxHawkesFitResult, fit_cox_hawkes


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
