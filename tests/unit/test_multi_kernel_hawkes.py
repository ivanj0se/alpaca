import numpy as np
import pytest

from events.hawkes import branching_ratio, fit_hawkes_exponential_multistart, simulate_hawkes
from research.multi_kernel_hawkes import (
    MultiKernelFitResult,
    _recursive_sums,
    default_timescale_grid,
    fit_multi_kernel_hawkes,
    simulate_multi_kernel_hawkes,
)


class TestDefaultTimescaleGrid:
    def test_centered_on_inverse_median_gap(self):
        # regular 10-second gaps -> median_gap=10 -> center beta = 0.1
        event_times = np.arange(0, 1000, 10.0)
        grid = default_timescale_grid(event_times, n_components=3, span_decades=1.0)
        assert grid[1] == pytest.approx(0.1, rel=1e-6)  # middle component = center

    def test_geometrically_spaced(self):
        event_times = np.arange(0, 1000, 10.0)
        grid = default_timescale_grid(event_times, n_components=3, span_decades=1.0)
        assert grid[0] == pytest.approx(grid[1] / 10, rel=1e-6)
        assert grid[2] == pytest.approx(grid[1] * 10, rel=1e-6)

    def test_raises_on_degenerate_gap(self):
        with pytest.raises(ValueError, match="positive"):
            default_timescale_grid(np.array([1.0, 1.0, 1.0]))


class TestRecursiveSums:
    def test_matches_single_component_recursion_for_k_equals_one(self):
        from events.hawkes import _recursive_sum

        event_times = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
        beta = 0.5
        single = _recursive_sum(event_times, beta)
        multi = _recursive_sums(event_times, np.array([beta]))
        assert np.allclose(single, multi[0])

    def test_shape_is_k_by_n(self):
        event_times = np.array([0.0, 1.0, 2.0, 3.0])
        betas = np.array([0.1, 0.5, 2.0])
        r = _recursive_sums(event_times, betas)
        assert r.shape == (3, 4)


class TestFitMultiKernelHawkes:
    def test_raises_with_fewer_than_two_events(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_multi_kernel_hawkes(np.array([1.0]), betas=np.array([0.5]))

    def test_raises_for_nonpositive_betas(self):
        with pytest.raises(ValueError, match="positive"):
            fit_multi_kernel_hawkes(np.array([0.0, 1.0, 2.0]), betas=np.array([0.5, -1.0]))

    def test_output_shapes(self):
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=3000, seed=1)
        result = fit_multi_kernel_hawkes(events, betas=np.array([0.3, 1.0, 3.0]))
        assert isinstance(result, MultiKernelFitResult)
        assert result.alphas.shape == (3,)
        assert result.betas.shape == (3,)
        assert len(result.branching_ratio_per_component) == 3

    def test_k_equals_one_recovers_single_kernel_result(self):
        # Consistency check: a single-component multi-kernel fit at the
        # SAME beta as the standard single-exponential fitter should
        # recover essentially the same branching ratio on the same real
        # (simulated) data -- validates the multi-kernel machinery
        # against the already-trusted events/hawkes.py implementation.
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=5000, seed=2)
        single_fit = fit_hawkes_exponential_multistart(events)
        multi_fit = fit_multi_kernel_hawkes(events, betas=np.array([single_fit.beta]))
        assert multi_fit.branching_ratio_total == pytest.approx(branching_ratio(single_fit), rel=0.05)

    def test_correctly_attributes_signal_to_the_matching_timescale(self):
        # Data simulated from a single true timescale (beta=1.0) fit with
        # a K=2 grid where one beta matches and one is very different --
        # the matching component should carry almost all the BRANCHING
        # RATIO contribution (alpha_k/beta_k), not necessarily the larger
        # raw alpha: a fast-decaying (high-beta) component needs a large
        # alpha just to survive its own rapid decay, but that alpha still
        # translates to a tiny alpha/beta ratio -- confirmed directly
        # (alphas ~[0.39, 0.46], but branching ratio contributions
        # ~[0.39, 0.009]). Comparing raw alphas was the wrong check.
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=5000, seed=3)
        result = fit_multi_kernel_hawkes(events, betas=np.array([1.0, 50.0]))
        contributions = result.branching_ratio_per_component
        assert contributions[0] > contributions[1] * 5

    def test_converged_flag_true_on_well_behaved_data(self):
        events = simulate_hawkes(mu=0.05, alpha=0.4, beta=1.0, T=3000, seed=4)
        result = fit_multi_kernel_hawkes(events, betas=np.array([0.3, 1.0, 3.0]))
        assert result.converged


class TestSimulateMultiKernelHawkes:
    def test_rejects_supercritical_total_branching_ratio(self):
        with pytest.raises(ValueError, match=">= 1"):
            simulate_multi_kernel_hawkes(mu=0.1, alphas=np.array([0.6, 0.6]), betas=np.array([1.0, 1.0]), T=100)

    def test_events_sorted_and_within_window(self):
        events = simulate_multi_kernel_hawkes(
            mu=0.05, alphas=np.array([0.2, 0.1]), betas=np.array([1.0, 10.0]), T=1000, seed=0
        )
        assert np.all(np.diff(events) >= 0)
        assert np.all(events >= 0) and np.all(events < 1000)

    def test_reproducible_with_seed(self):
        kwargs = dict(mu=0.05, alphas=np.array([0.2, 0.1]), betas=np.array([1.0, 10.0]), T=1000, seed=7)
        e1 = simulate_multi_kernel_hawkes(**kwargs)
        e2 = simulate_multi_kernel_hawkes(**kwargs)
        assert np.array_equal(e1, e2)

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            simulate_multi_kernel_hawkes(mu=0.05, alphas=np.array([0.2, 0.1]), betas=np.array([1.0]), T=100)

    def test_k_equals_one_matches_single_kernel_simulator_in_distribution(self):
        # Not sample-equal (different internal RNG call sequence: the
        # single-kernel simulator draws one Poisson/Exponential pair per
        # parent, the multi-kernel one loops per-component even at K=1) --
        # but should produce statistically indistinguishable event counts
        # at the same seed *distribution* (many seeds), the same
        # consistency-with-the-trusted-implementation check style already
        # used for fit_multi_kernel_hawkes's K=1 case.
        single_counts = [len(simulate_hawkes(0.05, 0.4, 1.0, T=2000, seed=s)) for s in range(20)]
        multi_counts = [
            len(simulate_multi_kernel_hawkes(0.05, np.array([0.4]), np.array([1.0]), T=2000, seed=s))
            for s in range(20)
        ]
        assert np.mean(multi_counts) == pytest.approx(np.mean(single_counts), rel=0.15)

    def test_more_components_produce_more_events_on_average(self):
        low = [
            len(simulate_multi_kernel_hawkes(0.05, np.array([0.4]), np.array([1.0]), T=2000, seed=s))
            for s in range(15)
        ]
        high = [
            len(simulate_multi_kernel_hawkes(0.05, np.array([0.4, 0.3]), np.array([1.0, 5.0]), T=2000, seed=s))
            for s in range(15)
        ]
        assert np.mean(high) > np.mean(low)


class TestMultiKernelSimulateRefitRecover:
    """Same trust-gate pattern as tests/unit/test_hawkes.py's
    TestSimulateRefitRecover: simulate with known per-component
    parameters, refit on the SAME betas, confirm recovery -- validates
    that the branching-representation generalization to K components
    (simulate_multi_kernel_hawkes) and the K-component MLE fitter
    (fit_multi_kernel_hawkes) are mutually consistent before either is
    trusted on real data.
    """

    TRUE_MU = 0.05
    TRUE_ALPHAS = np.array([0.3, 0.15])
    TRUE_BETAS = np.array([1.0, 10.0])
    T = 8000  # wider than the single-kernel case -- two components split the signal, needs more events to identify both

    def test_recovers_total_branching_ratio_within_25_percent(self):
        events = simulate_multi_kernel_hawkes(self.TRUE_MU, self.TRUE_ALPHAS, self.TRUE_BETAS, T=self.T, seed=42)
        fit = fit_multi_kernel_hawkes(events, betas=self.TRUE_BETAS, T=self.T)
        true_total = float(np.sum(self.TRUE_ALPHAS / self.TRUE_BETAS))
        assert fit.branching_ratio_total == pytest.approx(true_total, rel=0.25)
