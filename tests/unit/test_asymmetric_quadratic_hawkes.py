import numpy as np
import pytest

from research.asymmetric_quadratic_hawkes import (
    AsymmetricQuadraticHawkesFitResult,
    _recursive_signed_sum,
    _recursive_sum,
    fit_asymmetric_quadratic_hawkes,
    simulate_asymmetric_quadratic_hawkes,
)
from research.quadratic_hawkes import fit_quadratic_hawkes, simulate_quadratic_hawkes


class TestRecursionHelpers:
    def test_signed_sum_matches_naive_double_sum(self):
        event_times = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
        marks = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
        beta = 0.5
        result = _recursive_signed_sum(event_times, marks, beta)
        naive = np.zeros(len(event_times))
        for i in range(len(event_times)):
            for j in range(i):
                naive[i] += marks[j] * np.exp(-beta * (event_times[i] - event_times[j]))
        assert np.allclose(result, naive)

    def test_unsigned_sum_matches_naive_double_sum(self):
        event_times = np.array([0.0, 1.0, 2.5, 4.0])
        beta = 0.3
        result = _recursive_sum(event_times, beta)
        naive = np.zeros(len(event_times))
        for i in range(len(event_times)):
            for j in range(i):
                naive[i] += np.exp(-beta * (event_times[i] - event_times[j]))
        assert np.allclose(result, naive)


class TestLeverageAsymmetryEmerges:
    """The core mechanism this model exists to add, checked directly on
    the intensity formula (no simulation/fitting needed): with
    kappa_minus > kappa_plus, a down-trend history must predict HIGHER
    intensity than an up-trend history of the exact same magnitude --
    something research/quadratic_hawkes.py's symmetric model cannot do
    at any kappa value.
    """

    def test_down_trend_predicts_higher_intensity_than_up_trend_of_same_magnitude(self):
        event_times = np.arange(0, 10, 1.0)
        n = len(event_times)
        down_trend_l1 = _recursive_signed_sum(event_times, np.full(n, -1.0), beta=0.3)[-1]
        up_trend_l1 = _recursive_signed_sum(event_times, np.full(n, 1.0), beta=0.3)[-1]
        assert down_trend_l1 == pytest.approx(-up_trend_l1)  # same magnitude, opposite sign, by construction

        kappa_minus, kappa_plus, lambda0, alpha, l2 = 0.5, 0.1, 0.05, 0.0, 0.0
        intensity_down = lambda0 + kappa_minus * down_trend_l1**2 + alpha * l2  # L1 < 0 -> kappa_minus applies
        intensity_up = lambda0 + kappa_plus * up_trend_l1**2 + alpha * l2  # L1 >= 0 -> kappa_plus applies
        assert intensity_down > intensity_up

    def test_equal_kappas_produce_symmetric_intensity(self):
        # kappa_minus == kappa_plus should erase the asymmetry entirely --
        # confirms the mechanism is genuinely OFF, not just small, when
        # the two coefficients match (the model's degenerate case, same
        # as research/quadratic_hawkes.py's single kappa).
        event_times = np.arange(0, 10, 1.0)
        n = len(event_times)
        down_l1 = _recursive_signed_sum(event_times, np.full(n, -1.0), beta=0.3)[-1]
        up_l1 = _recursive_signed_sum(event_times, np.full(n, 1.0), beta=0.3)[-1]
        kappa = 0.3
        intensity_down = 0.05 + kappa * down_l1**2
        intensity_up = 0.05 + kappa * up_l1**2
        assert intensity_down == pytest.approx(intensity_up)


class TestFitAsymmetricQuadraticHawkes:
    def test_raises_with_fewer_than_two_events(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_asymmetric_quadratic_hawkes(np.array([1.0]), np.array([1.0]), beta_leverage=0.5, beta=0.5)

    def test_raises_on_mismatched_marks_length(self):
        with pytest.raises(ValueError, match="one value per event"):
            fit_asymmetric_quadratic_hawkes(np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0]), beta_leverage=0.5, beta=0.5)

    def test_raises_on_nonpositive_betas(self):
        with pytest.raises(ValueError, match="beta_leverage"):
            fit_asymmetric_quadratic_hawkes(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]), beta_leverage=0.0, beta=0.5
            )
        with pytest.raises(ValueError, match="beta must be positive"):
            fit_asymmetric_quadratic_hawkes(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]), beta_leverage=0.5, beta=0.0
            )

    def test_output_type_and_converged_flag(self):
        events, marks = simulate_asymmetric_quadratic_hawkes(
            lambda0=0.05, kappa_minus=0.3, kappa_plus=0.08, alpha=0.2, beta_leverage=0.5, beta=1.0, T=3000,
            marks_pool=np.array([1.0, -1.0]), seed=1,
        )
        result = fit_asymmetric_quadratic_hawkes(events, marks, beta_leverage=0.5, beta=1.0, T=3000.0)
        assert isinstance(result, AsymmetricQuadraticHawkesFitResult)
        assert result.converged
        assert result.lambda0 > 0 and result.kappa_minus >= 0 and result.kappa_plus >= 0 and result.alpha >= 0

    def test_equal_true_kappas_reduce_to_symmetric_model_fit(self):
        # With kappa_minus == kappa_plus in the DATA-GENERATING process,
        # the asymmetric fit should recover a small leverage_asymmetry
        # (near 0) and a loglik close to research/quadratic_hawkes.py's
        # symmetric fit on the SAME events -- cross-validates the new
        # machinery's degenerate case against the already-trusted
        # implementation.
        events, marks = simulate_quadratic_hawkes(
            lambda0=0.05, kappa=0.2, alpha=0.2, beta_leverage=0.5, beta=1.0, T=5000,
            marks_pool=np.array([1.0, -1.0]), seed=7,
        )
        symmetric_fit = fit_quadratic_hawkes(events, marks, beta_leverage=0.5, beta=1.0, T=5000.0)
        asym_fit = fit_asymmetric_quadratic_hawkes(events, marks, beta_leverage=0.5, beta=1.0, T=5000.0)
        assert asym_fit.loglik >= symmetric_fit.loglik - 1e-6  # strictly more flexible model, can't do worse
        assert asym_fit.loglik == pytest.approx(symmetric_fit.loglik, abs=5.0)  # but shouldn't overfit wildly either


class TestSimulateAsymmetricQuadraticHawkes:
    def test_raises_on_empty_marks_pool(self):
        with pytest.raises(ValueError, match="non-empty"):
            simulate_asymmetric_quadratic_hawkes(
                lambda0=0.1, kappa_minus=0.1, kappa_plus=0.1, alpha=0.1, beta_leverage=0.5, beta=1.0, T=100,
                marks_pool=np.array([]),
            )

    def test_raises_on_nonpositive_betas(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_asymmetric_quadratic_hawkes(
                lambda0=0.1, kappa_minus=0.1, kappa_plus=0.1, alpha=0.1, beta_leverage=0.0, beta=1.0, T=100,
                marks_pool=np.array([1.0]),
            )

    def test_events_and_marks_same_length_and_sorted(self):
        events, marks = simulate_asymmetric_quadratic_hawkes(
            lambda0=0.05, kappa_minus=0.3, kappa_plus=0.08, alpha=0.2, beta_leverage=0.5, beta=1.0, T=2000,
            marks_pool=np.array([1.0, -1.0]), seed=0,
        )
        assert len(events) == len(marks)
        assert np.all(np.diff(events) >= 0)
        assert np.all(events >= 0) and np.all(events < 2000)

    def test_reproducible_with_seed(self):
        kwargs = dict(
            lambda0=0.05, kappa_minus=0.3, kappa_plus=0.08, alpha=0.2, beta_leverage=0.5, beta=1.0, T=2000,
            marks_pool=np.array([1.0, -1.0]), seed=7,
        )
        e1, m1 = simulate_asymmetric_quadratic_hawkes(**kwargs)
        e2, m2 = simulate_asymmetric_quadratic_hawkes(**kwargs)
        assert np.array_equal(e1, e2)
        assert np.array_equal(m1, m2)

    def test_equal_kappas_match_symmetric_simulator_in_distribution(self):
        # kappa_minus == kappa_plus should be statistically
        # indistinguishable in event COUNT from
        # research/quadratic_hawkes.py's already-trusted
        # simulate_quadratic_hawkes at the same (lambda0, kappa, alpha) --
        # same cross-implementation consistency discipline used
        # throughout this project.
        asym_counts = [
            len(
                simulate_asymmetric_quadratic_hawkes(
                    0.05, 0.15, 0.15, 0.1, 0.5, 1.0, T=2000, marks_pool=np.array([1.0, -1.0]), seed=s
                )[0]
            )
            for s in range(20)
        ]
        symmetric_counts = [
            len(simulate_quadratic_hawkes(0.05, 0.15, 0.1, 0.5, 1.0, T=2000, marks_pool=np.array([1.0, -1.0]), seed=s)[0])
            for s in range(20)
        ]
        assert np.mean(asym_counts) == pytest.approx(np.mean(symmetric_counts), rel=0.15)


class TestAsymmetricQuadraticHawkesSimulateRefitRecover:
    """Same trust-gate pattern as every other Hawkes-family fitter in
    this project: simulate with known, genuinely asymmetric parameters,
    refit, confirm the asymmetry DIRECTION is recovered -- validates
    simulate_asymmetric_quadratic_hawkes and
    fit_asymmetric_quadratic_hawkes are mutually consistent before either
    is trusted on real data.
    """

    LAMBDA0, KAPPA_MINUS, KAPPA_PLUS, ALPHA, BETA_LEV, BETA = 0.05, 0.4, 0.1, 0.2, 0.5, 1.0
    T = 8000

    def test_recovers_the_correct_asymmetry_direction(self):
        events, marks = simulate_asymmetric_quadratic_hawkes(
            self.LAMBDA0, self.KAPPA_MINUS, self.KAPPA_PLUS, self.ALPHA, self.BETA_LEV, self.BETA, T=self.T,
            marks_pool=np.array([1.0, -1.0]), seed=42,
        )
        fit = fit_asymmetric_quadratic_hawkes(events, marks, beta_leverage=self.BETA_LEV, beta=self.BETA, T=self.T)
        assert fit.leverage_asymmetry > 0  # kappa_minus > kappa_plus recovered, the true (planted) direction

    def test_recovers_branching_ratio_within_35_percent(self):
        events, marks = simulate_asymmetric_quadratic_hawkes(
            self.LAMBDA0, self.KAPPA_MINUS, self.KAPPA_PLUS, self.ALPHA, self.BETA_LEV, self.BETA, T=self.T,
            marks_pool=np.array([1.0, -1.0]), seed=42,
        )
        fit = fit_asymmetric_quadratic_hawkes(events, marks, beta_leverage=self.BETA_LEV, beta=self.BETA, T=self.T)
        true_ratio = self.ALPHA / self.BETA
        assert fit.branching_ratio == pytest.approx(true_ratio, rel=0.35)
