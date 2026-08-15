import numpy as np
import pytest

from events.hawkes import fit_hawkes_exponential_multistart, simulate_hawkes
from research.quadratic_hawkes import (
    QuadraticHawkesFitResult,
    _recursive_signed_sum,
    _recursive_sum,
    fit_quadratic_hawkes,
    simulate_quadratic_hawkes,
)


class TestRecursiveSignedSum:
    def test_matches_naive_double_sum(self):
        event_times = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
        marks = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
        beta = 0.5
        result = _recursive_signed_sum(event_times, marks, beta)
        naive = np.zeros(len(event_times))
        for i in range(len(event_times)):
            for j in range(i):
                naive[i] += marks[j] * np.exp(-beta * (event_times[i] - event_times[j]))
        assert np.allclose(result, naive)

    def test_first_element_is_zero(self):
        result = _recursive_signed_sum(np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]), 0.5)
        assert result[0] == 0.0

    def test_unit_marks_reduce_to_unsigned_recursion(self):
        event_times = np.array([0.0, 1.0, 2.5, 4.0])
        marks = np.ones(len(event_times))
        signed = _recursive_signed_sum(event_times, marks, 0.3)
        unsigned = _recursive_sum(event_times, 0.3)
        assert np.allclose(signed, unsigned)


class TestLeverageEffectEmerges:
    """The core mechanism this model exists to add: a trending (same-sign)
    history should produce a LARGER |L1| (and thus more excess intensity
    once squared) than a choppy (alternating-sign) history of the exact
    same magnitudes -- something no model built earlier in this project
    (events/hawkes.py, multi_kernel_hawkes.py, cox_hawkes.py) can tell
    apart, since none of them look at sign at all. Deterministic, no
    simulation/fitting needed -- directly confirms the mechanism itself.
    """

    def test_trending_marks_produce_larger_magnitude_than_alternating(self):
        event_times = np.arange(0, 10, 1.0)
        n = len(event_times)
        trending = np.full(n, -1.0)  # a run of down-moves
        alternating = np.array([(-1.0) ** i for i in range(n)])  # chop

        beta = 0.3
        l1_trending = _recursive_signed_sum(event_times, trending, beta)
        l1_alternating = _recursive_signed_sum(event_times, alternating, beta)

        assert abs(l1_trending[-1]) > abs(l1_alternating[-1]) * 3

    def test_alternating_marks_stay_near_zero(self):
        event_times = np.arange(0, 20, 1.0)
        n = len(event_times)
        alternating = np.array([(-1.0) ** i for i in range(n)])
        l1 = _recursive_signed_sum(event_times, alternating, beta=0.3)
        assert abs(l1[-1]) < 1.0  # bounded, doesn't grow with history length


class TestFitQuadraticHawkes:
    def test_raises_with_fewer_than_two_events(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_quadratic_hawkes(np.array([1.0]), np.array([1.0]), beta_leverage=0.5, beta=0.5)

    def test_raises_on_mismatched_marks_length(self):
        with pytest.raises(ValueError, match="one value per event"):
            fit_quadratic_hawkes(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0]), beta_leverage=0.5, beta=0.5
            )

    def test_raises_on_nonpositive_beta_leverage(self):
        with pytest.raises(ValueError, match="beta_leverage"):
            fit_quadratic_hawkes(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]), beta_leverage=0.0, beta=0.5
            )

    def test_raises_on_nonpositive_beta(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            fit_quadratic_hawkes(
                np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]), beta_leverage=0.5, beta=0.0
            )

    def test_output_type_and_converged_flag(self):
        events, marks = simulate_quadratic_hawkes(
            lambda0=0.05, kappa=0.1, alpha=0.3, beta_leverage=0.5, beta=1.0, T=3000,
            marks_pool=np.array([1.0, -1.0]), seed=1,
        )
        result = fit_quadratic_hawkes(events, marks, beta_leverage=0.5, beta=1.0, T=3000.0)
        assert isinstance(result, QuadraticHawkesFitResult)
        assert result.converged
        assert result.lambda0 > 0 and result.kappa >= 0 and result.alpha >= 0

    def test_kappa_zero_start_still_lets_kappa_grow_when_needed(self):
        # kappa0_grid includes 0.0 as a starting point -- confirm the
        # optimizer isn't stuck there when the data genuinely has a
        # leverage effect (simulated with a real, substantial kappa).
        # kappa=0.4 kept well below stability_heuristic's ~1.0 danger
        # zone (here: 0.4/(2*0.5) + 0.2/1.0 = 0.6) -- an earlier version
        # of this test used kappa=1.5 (heuristic=1.7) and hung for
        # minutes before being killed; see stability_heuristic's
        # docstring.
        events, marks = simulate_quadratic_hawkes(
            lambda0=0.05, kappa=0.4, alpha=0.2, beta_leverage=0.5, beta=1.0, T=5000,
            marks_pool=np.array([1.0, -1.0]), seed=2,
        )
        result = fit_quadratic_hawkes(events, marks, beta_leverage=0.5, beta=1.0, T=5000.0)
        assert result.kappa > 0.15  # recovered a real, substantial leverage coefficient

    def test_kappa_zero_reduces_to_standard_hawkes_fit(self):
        # With kappa fixed near 0 in the data-generating process (no real
        # leverage effect), the L2/alpha/beta half of this model should
        # closely match a standard single-exponential Hawkes fit on the
        # SAME events -- validates the quadratic machinery's linear half
        # against the already-trusted events/hawkes.py implementation.
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=5000, seed=3)
        marks = np.ones(len(events))  # constant marks -> L1 nonzero but IDENTICAL every trial, not genuinely signed
        standard_fit = fit_hawkes_exponential_multistart(events)
        quad_fit = fit_quadratic_hawkes(events, marks, beta_leverage=1.0, beta=standard_fit.beta, T=5000.0)
        assert quad_fit.branching_ratio == pytest.approx(standard_fit.alpha / standard_fit.beta, rel=0.15)


class TestSimulateQuadraticHawkes:
    def test_raises_on_empty_marks_pool(self):
        with pytest.raises(ValueError, match="non-empty"):
            simulate_quadratic_hawkes(
                lambda0=0.1, kappa=0.1, alpha=0.1, beta_leverage=0.5, beta=1.0, T=100, marks_pool=np.array([])
            )

    def test_raises_on_nonpositive_betas(self):
        with pytest.raises(ValueError, match="positive"):
            simulate_quadratic_hawkes(
                lambda0=0.1, kappa=0.1, alpha=0.1, beta_leverage=0.0, beta=1.0, T=100, marks_pool=np.array([1.0])
            )

    def test_events_and_marks_same_length_and_sorted(self):
        events, marks = simulate_quadratic_hawkes(
            lambda0=0.05, kappa=0.1, alpha=0.3, beta_leverage=0.5, beta=1.0, T=2000,
            marks_pool=np.array([1.0, -1.0]), seed=0,
        )
        assert len(events) == len(marks)
        assert np.all(np.diff(events) >= 0)
        assert np.all(events >= 0) and np.all(events < 2000)

    def test_reproducible_with_seed(self):
        kwargs = dict(
            lambda0=0.05, kappa=0.1, alpha=0.3, beta_leverage=0.5, beta=1.0, T=2000,
            marks_pool=np.array([1.0, -1.0]), seed=7,
        )
        e1, m1 = simulate_quadratic_hawkes(**kwargs)
        e2, m2 = simulate_quadratic_hawkes(**kwargs)
        assert np.array_equal(e1, e2)
        assert np.array_equal(m1, m2)

    def test_marks_drawn_only_from_the_pool(self):
        # Non-zero-mean marks (E[mark]=-0.5 here) make stability_heuristic
        # unreliable (see its own docstring) -- kappa=0.05 (heuristic
        # "0.425") actually exploded in practice; empirically confirmed
        # kappa=0.01 finishes in well under a second before trusting it
        # here.
        pool = np.array([2.0, -3.0])
        _, marks = simulate_quadratic_hawkes(
            lambda0=0.1, kappa=0.01, alpha=0.1, beta_leverage=0.5, beta=1.0, T=3000, marks_pool=pool, seed=5
        )
        assert len(marks) > 0
        assert set(np.unique(marks)).issubset(set(pool))

    def test_kappa_zero_matches_standard_simulator_in_distribution(self):
        # kappa=0 nullifies the quadratic term regardless of marks --
        # should be statistically indistinguishable in event COUNT from
        # events/hawkes.py's already-trusted simulate_hawkes at the same
        # (lambda0, alpha, beta), the same K=1/gamma=0-style consistency
        # check already used for the other extensions in this project.
        quad_counts = [
            len(
                simulate_quadratic_hawkes(
                    0.05, 0.0, 0.4, 1.0, 1.0, T=2000, marks_pool=np.array([1.0, -1.0]), seed=s
                )[0]
            )
            for s in range(20)
        ]
        plain_counts = [len(simulate_hawkes(0.05, 0.4, 1.0, T=2000, seed=s)) for s in range(20)]
        assert np.mean(quad_counts) == pytest.approx(np.mean(plain_counts), rel=0.15)

    def test_higher_kappa_produces_more_events_on_average_for_trending_marks(self):
        # A pool with only SAME-SIGN marks (a permanent "trend") should
        # make higher kappa matter -- more events on average than kappa=0,
        # since L1 never cancels back toward zero. All-same-sign marks are
        # a worst case for stability: L1 becomes numerically IDENTICAL to
        # L2 (not just correlated with it), so stability_heuristic badly
        # underestimates real risk here (its own docstring flags this
        # exact failure mode) -- kappa=0.3 (heuristic "0.4") actually
        # exploded in practice; kappa=0.03 empirically confirmed safe
        # (n_events~107 over T=2000, comfortably under any cap).
        pool = np.array([1.0])  # always +1
        low = [
            len(simulate_quadratic_hawkes(0.05, 0.0, 0.1, 0.5, 1.0, T=2000, marks_pool=pool, seed=s)[0])
            for s in range(15)
        ]
        high = [
            len(simulate_quadratic_hawkes(0.05, 0.03, 0.1, 0.5, 1.0, T=2000, marks_pool=pool, seed=s)[0])
            for s in range(15)
        ]
        assert np.mean(high) > np.mean(low)


class TestQuadraticHawkesSimulateRefitRecover:
    """Same trust-gate pattern as tests/unit/test_hawkes.py's
    TestSimulateRefitRecover: simulate with known parameters, refit,
    confirm recovery -- validates simulate_quadratic_hawkes and
    fit_quadratic_hawkes are mutually consistent before either is trusted
    on real data.
    """

    # kappa=0.3 kept well under stability_heuristic's danger zone
    # (0.3/1.0 + 0.3 = 0.6, comfortable margin below 1).
    LAMBDA0, KAPPA, ALPHA, BETA_LEV, BETA = 0.05, 0.3, 0.3, 0.5, 1.0
    T = 8000  # wider than a single-kernel fit -- two feedback terms need more events to jointly identify

    def test_recovers_branching_ratio_within_30_percent(self):
        events, marks = simulate_quadratic_hawkes(
            self.LAMBDA0, self.KAPPA, self.ALPHA, self.BETA_LEV, self.BETA, T=self.T,
            marks_pool=np.array([1.0, -1.0]), seed=42,
        )
        fit = fit_quadratic_hawkes(events, marks, beta_leverage=self.BETA_LEV, beta=self.BETA, T=self.T)
        true_ratio = self.ALPHA / self.BETA
        assert fit.branching_ratio == pytest.approx(true_ratio, rel=0.30)

    def test_recovers_a_nonzero_kappa(self):
        events, marks = simulate_quadratic_hawkes(
            self.LAMBDA0, self.KAPPA, self.ALPHA, self.BETA_LEV, self.BETA, T=self.T,
            marks_pool=np.array([1.0, -1.0]), seed=42,
        )
        fit = fit_quadratic_hawkes(events, marks, beta_leverage=self.BETA_LEV, beta=self.BETA, T=self.T)
        assert fit.kappa > 0.1  # recovered real, substantial leverage signal, not collapsed to ~0
