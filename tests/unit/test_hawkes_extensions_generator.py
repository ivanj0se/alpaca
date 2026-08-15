import numpy as np
import pandas as pd
import pytest

from generators.hawkes_extensions_generator import (
    fit_real_cox_hawkes_params,
    fit_real_multi_kernel_params,
    generate_cox_hawkes_paths,
    generate_multi_kernel_paths,
    simulate_multi_kernel_hawkes_bounded,
)
from ingest.storage import write_bars, write_ticks
from research.cox_hawkes import CoxHawkesFitResult
from research.multi_kernel_hawkes import MultiKernelFitResult


def _synthetic_ticks(ticker, start, n, base_price=100.0, burst_at=None):
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    rng = np.random.default_rng(0)
    prices = base_price + np.cumsum(rng.normal(0, 0.001, n))
    if burst_at is not None:
        for i in burst_at:
            prices[i:] += rng.choice([-1, 1]) * 0.5
    return pd.DataFrame({"timestamp": idx, "ticker": ticker, "price": prices, "size": 10})


def _synthetic_bars(start, n):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    volumes = rng.integers(50, 500, n).astype(float)
    return pd.DataFrame(
        {"timestamp": idx, "ticker": "SPY", "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
    )


BURSTS = list(range(100, 3000, 100))


class TestSimulateMultiKernelHawkesBounded:
    def test_raises_when_exceeding_max_events(self):
        with pytest.raises(RuntimeError, match="max_events"):
            simulate_multi_kernel_hawkes_bounded(
                mu=0.05, alphas=np.array([0.09]), betas=np.array([0.1]), T=10000, max_events=2, seed=0
            )

    def test_returns_events_when_under_cap(self):
        events = simulate_multi_kernel_hawkes_bounded(
            mu=0.01, alphas=np.array([0.02]), betas=np.array([0.1]), T=100, max_events=100_000, seed=0
        )
        assert isinstance(events, np.ndarray)
        assert (events < 100).all()


class TestFitRealMultiKernelParams:
    def test_raises_when_no_data(self, tmp_path):
        with pytest.raises(ValueError, match="no real tick data"):
            fit_real_multi_kernel_params(tmp_path, ticker="SPY")

    def test_fits_successfully_on_synthetic_tick_data(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        fit = fit_real_multi_kernel_params(tmp_path, ticker="SPY", sigma_threshold=2.0)
        assert isinstance(fit, MultiKernelFitResult)
        assert fit.mu > 0
        assert len(fit.alphas) == 3
        assert len(fit.betas) == 3


class TestGenerateMultiKernelPaths:
    def test_end_to_end_on_synthetic_data(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        write_bars(_synthetic_bars("2026-01-02 09:30", 4000), tmp_path)

        paths = generate_multi_kernel_paths(tmp_path, ticker="SPY", T_days=0.01, n_sims=2, seed=0)
        assert len(paths) == 2
        for p in paths:
            assert p.generator_id == "hawkes_multi_kernel"
            assert len(p.log_returns) > 0
            assert len(p.params["alphas"]) == 3


def _synthetic_common_factor(start, n_points, freq="1D"):
    idx = pd.date_range(start, periods=n_points, freq=freq, tz="UTC")
    rng = np.random.default_rng(2)
    return pd.Series(rng.normal(0, 1, n_points), index=idx)


class TestFitRealCoxHawkesParams:
    def test_raises_when_no_data(self, tmp_path):
        cf = _synthetic_common_factor("2026-01-02", 30)
        with pytest.raises(ValueError, match="no real tick data"):
            fit_real_cox_hawkes_params(tmp_path, cf, ticker="SPY")

    def test_fits_successfully_on_synthetic_data_within_coverage(self, tmp_path):
        # Ticks span 2026-01-02 to roughly 2026-01-03 (4000s); common
        # factor must cover that span with fine enough resolution to
        # leave a real grid, so use hourly points over a few days.
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        cf = _synthetic_common_factor("2026-01-01", 96, freq="1h")  # 4 days hourly, covers the tick span
        fit, grid_vals, grid_durations = fit_real_cox_hawkes_params(tmp_path, cf, ticker="SPY")
        assert isinstance(fit, CoxHawkesFitResult)
        assert len(grid_vals) == len(grid_durations)
        assert len(grid_vals) > 0


class TestGenerateCoxHawkesPaths:
    def test_end_to_end_on_synthetic_data(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        write_bars(_synthetic_bars("2026-01-02 09:30", 4000), tmp_path)
        cf = _synthetic_common_factor("2026-01-01", 96, freq="1h")

        paths = generate_cox_hawkes_paths(tmp_path, cf, ticker="SPY", T_days=0.01, n_sims=2, seed=0)
        assert len(paths) == 2
        for p in paths:
            assert p.generator_id == "cox_hawkes_rpca"
            assert len(p.log_returns) > 0
            assert "gamma" in p.params

    def test_raises_when_covariate_window_too_short(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        write_bars(_synthetic_bars("2026-01-02 09:30", 4000), tmp_path)
        cf = _synthetic_common_factor("2026-01-01", 96, freq="1h")  # ~4 days real coverage

        with pytest.raises(ValueError, match="less than half"):
            generate_cox_hawkes_paths(tmp_path, cf, ticker="SPY", T_days=100.0, n_sims=1, seed=0)

    def test_simulated_span_never_overshoots_the_requested_t_days(self, tmp_path):
        # Regression test for a real bug: an earlier version rounded UP to
        # "at least T_requested" instead of clipping the final grid
        # segment, AND used calendar days (86400s) instead of the same
        # session-day convention every other arm in this module uses --
        # together those produced paths ~5x longer than
        # generate_multi_kernel_paths/generate_ablation_paths at the same
        # T_days, which silently invalidated a real comparison against
        # bands calibrated at the shorter length (see
        # diagnostics/2026-08-14-tier3-hawkes-extensions-ablation/). Every
        # path's implied duration (n_bars*bar_seconds) must land within
        # one bar of T_days*SESSION_SECONDS_PER_DAY, never over.
        from generators.hawkes_extensions_generator import SESSION_SECONDS_PER_DAY

        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        write_bars(_synthetic_bars("2026-01-02 09:30", 4000), tmp_path)
        cf = _synthetic_common_factor("2026-01-01", 96, freq="1h")

        T_days = 0.05
        bar_seconds = 60.0
        paths = generate_cox_hawkes_paths(tmp_path, cf, ticker="SPY", T_days=T_days, bar_seconds=bar_seconds, n_sims=1, seed=0)
        expected_bars = T_days * SESSION_SECONDS_PER_DAY / bar_seconds
        assert len(paths[0].log_returns) <= expected_bars + 1
