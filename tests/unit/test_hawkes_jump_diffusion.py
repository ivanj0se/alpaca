import numpy as np
import pandas as pd
import pytest

from events.hawkes import HawkesFitResult
from generators.hawkes_jump_diffusion import (
    HawkesAblationArm,
    build_ablation_arms,
    calibrate_jump_and_background_std,
    fit_real_hawkes_params,
    generate_ablation_paths,
    hawkes_events_to_returns,
    session_elapsed_seconds,
    simulate_hawkes_bounded,
)
from ingest.storage import write_bars, write_ticks


class TestSessionElapsedSeconds:
    def test_no_boundary_gives_raw_elapsed_seconds(self):
        idx = pd.date_range("2026-01-02 09:30", periods=5, freq="1min", tz="UTC")
        result = session_elapsed_seconds(idx)
        assert np.allclose(result, [0, 60, 120, 180, 240])

    def test_boundary_gap_is_replaced_not_kept_raw(self):
        day1 = pd.date_range("2026-01-02 09:30", periods=3, freq="1min", tz="America/New_York")
        day2 = pd.date_range("2026-01-05 09:30", periods=3, freq="1min", tz="America/New_York")
        idx = day1.append(day2).tz_convert("UTC")
        result = session_elapsed_seconds(idx)
        # the boundary gap (position 3) should be replaced by the typical
        # 60s intra-session gap, not the real multi-day gap.
        assert result[3] - result[2] == pytest.approx(60.0)

    def test_single_timestamp_returns_zero(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-01-02 09:30", tz="UTC")])
        result = session_elapsed_seconds(idx)
        assert list(result) == [0.0]


def _synthetic_ticks(ticker, start, n, base_price=100.0, burst_at=None):
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    rng = np.random.default_rng(0)
    prices = base_price + np.cumsum(rng.normal(0, 0.001, n))
    if burst_at is not None:
        # plant a cluster of large moves so sigma>=2 events actually fire
        for i in burst_at:
            prices[i:] += rng.choice([-1, 1]) * 0.5
    return pd.DataFrame({"timestamp": idx, "ticker": ticker, "price": prices, "size": 10})


class TestFitRealHawkesParams:
    def test_raises_when_no_data(self, tmp_path):
        with pytest.raises(ValueError, match="no real tick data"):
            fit_real_hawkes_params(tmp_path, ticker="SPY")

    def test_raises_when_too_few_events(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 50), tmp_path)
        with pytest.raises(ValueError, match="not enough to fit"):
            fit_real_hawkes_params(tmp_path, ticker="SPY", sigma_threshold=2.0)

    def test_fits_successfully_on_synthetic_tick_data(self, tmp_path):
        bursts = list(range(100, 3000, 100))  # frequent bursts -> plenty of sigma>=2 events
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=bursts), tmp_path)
        fit = fit_real_hawkes_params(tmp_path, ticker="SPY", sigma_threshold=2.0)
        assert isinstance(fit, HawkesFitResult)
        assert fit.mu > 0 and fit.alpha >= 0 and fit.beta > 0


class TestBuildAblationArms:
    def test_control_has_zero_alpha(self):
        fit = HawkesFitResult(mu=0.001, alpha=0.005, beta=0.01, loglik=-1.0, converged=True, n_events=100)
        control, treatment = build_ablation_arms(fit)
        assert control.alpha == 0.0
        assert treatment.alpha == fit.alpha

    def test_expected_total_events_matches_between_arms(self):
        # The core correctness property: mu/(1-branching_ratio) must be
        # equal for both arms, or the ablation confounds self-excitation
        # with event density.
        fit = HawkesFitResult(mu=0.001, alpha=0.005, beta=0.01, loglik=-1.0, converged=True, n_events=100)
        control, treatment = build_ablation_arms(fit)
        control_total = control.mu / (1.0 - control.alpha / control.beta)
        treatment_total = treatment.mu / (1.0 - treatment.alpha / treatment.beta)
        assert control_total == pytest.approx(treatment_total, rel=1e-9)

    def test_raises_for_supercritical_branching_ratio(self):
        fit = HawkesFitResult(mu=0.001, alpha=0.02, beta=0.01, loglik=-1.0, converged=True, n_events=100)
        with pytest.raises(ValueError):
            build_ablation_arms(fit)


class TestSimulateHawkesBounded:
    def test_raises_when_exceeding_max_events(self):
        # branching_ratio close to 1 with a long T -> plenty of events;
        # set max_events tiny to force the cap to trigger deterministically.
        with pytest.raises(RuntimeError, match="max_events"):
            simulate_hawkes_bounded(mu=0.05, alpha=0.09, beta=0.1, T=10000, max_events=2, seed=0)

    def test_returns_events_when_under_cap(self):
        events = simulate_hawkes_bounded(mu=0.01, alpha=0.02, beta=0.1, T=100, max_events=100_000, seed=0)
        assert isinstance(events, np.ndarray)
        assert (events < 100).all()


class TestCalibrateJumpAndBackgroundStd:
    def test_splits_correctly(self):
        rng = np.random.default_rng(0)
        small = rng.normal(0, 0.001, 950)
        large = rng.normal(0, 0.02, 50)
        returns = np.concatenate([small, large])
        background_std, jump_std = calibrate_jump_and_background_std(returns, sigma_threshold=2.0)
        assert jump_std > background_std

    def test_raises_on_zero_variance(self):
        with pytest.raises(ValueError, match="zero variance"):
            calibrate_jump_and_background_std(np.zeros(100), sigma_threshold=2.0)

    def test_raises_when_one_side_too_sparse(self):
        # An enormous threshold puts everything below it, leaving nothing above.
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="too few observations"):
            calibrate_jump_and_background_std(rng.normal(0, 1, 100), sigma_threshold=100.0)


class TestHawkesEventsToReturns:
    def test_output_length_matches_n_bars(self):
        result = hawkes_events_to_returns(
            np.array([10.0, 50.0]), T=300, bar_seconds=60, background_std=0.001, jump_std=0.01, seed=0
        )
        assert len(result) == 5

    def test_events_produce_larger_moves_than_background_only(self):
        no_events = hawkes_events_to_returns(
            np.array([]), T=6000, bar_seconds=60, background_std=0.001, jump_std=0.05, seed=0
        )
        with_events = hawkes_events_to_returns(
            np.arange(0, 6000, 60), T=6000, bar_seconds=60, background_std=0.001, jump_std=0.05, seed=0
        )
        assert np.abs(with_events).mean() > np.abs(no_events).mean() * 5

    def test_reproducible_with_seed(self):
        a = hawkes_events_to_returns(np.array([10.0]), T=120, bar_seconds=60, background_std=0.001, jump_std=0.01, seed=5)
        b = hawkes_events_to_returns(np.array([10.0]), T=120, bar_seconds=60, background_std=0.001, jump_std=0.01, seed=5)
        assert np.array_equal(a, b)


class TestGenerateAblationPaths:
    def test_end_to_end_on_synthetic_data(self, tmp_path):
        bursts = list(range(100, 3000, 100))
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=bursts), tmp_path)
        idx = pd.date_range("2026-01-02 09:30", periods=4000, freq="1min", tz="UTC")
        rng = np.random.default_rng(1)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 4000)))
        volumes = rng.integers(50, 500, 4000).astype(float)  # non-constant, or volume_zscore is NaN everywhere
        bars = pd.DataFrame(
            {"timestamp": idx, "ticker": "SPY", "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
        )
        write_bars(bars, tmp_path)

        results = generate_ablation_paths(tmp_path, ticker="SPY", T_days=0.01, n_sims=2, seed=0)
        assert set(results.keys()) == {"control", "treatment"}
        assert len(results["control"]) == 2
        assert len(results["treatment"]) == 2
        for path in results["control"] + results["treatment"]:
            assert len(path.log_returns) > 0
