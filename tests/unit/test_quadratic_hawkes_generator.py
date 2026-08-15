import numpy as np
import pandas as pd
import pytest

from generators.quadratic_hawkes_generator import (
    fit_real_quadratic_hawkes_params,
    generate_quadratic_hawkes_paths,
    signed_tick_events,
)
from ingest.storage import write_bars, write_ticks
from research.quadratic_hawkes import QuadraticHawkesFitResult


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


class TestSignedTickEvents:
    def test_raises_when_no_data_for_ticker(self):
        ticks = _synthetic_ticks("SPY", "2026-01-02 09:30", 100)
        with pytest.raises(ValueError, match="no real tick data"):
            signed_tick_events(ticks, "AAPL")

    def test_returns_signed_marks_matching_return_direction(self):
        ticks = _synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS)
        event_times, marks = signed_tick_events(ticks, "SPY", sigma_threshold=2.0)
        assert len(event_times) == len(marks)
        assert len(event_times) > 0
        # signed marks should include both signs on real bidirectional bursts
        assert (marks > 0).any() and (marks < 0).any()

    def test_marks_match_abs_zscore_from_the_production_event_definition(self):
        # Cross-check against the already-trust-gated event definition in
        # events/price_events.py::tick_events_from_recorder -- same event
        # SET and same magnitudes, this function only adds sign back in.
        from events.price_events import tick_events_from_recorder

        ticks = _synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS)
        event_times, marks = signed_tick_events(ticks, "SPY", sigma_threshold=2.0)
        production_events = tick_events_from_recorder(ticks, sigma_threshold=2.0)
        production_events = production_events[production_events["ticker"] == "SPY"]

        assert len(event_times) == len(production_events)
        assert np.allclose(np.sort(np.abs(marks)), np.sort(production_events["abs_zscore"].to_numpy()))


class TestFitRealQuadraticHawkesParams:
    def test_raises_when_no_data(self, tmp_path):
        with pytest.raises(ValueError, match="no real tick data"):
            fit_real_quadratic_hawkes_params(tmp_path, ticker="SPY")

    def test_raises_when_too_few_events(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 50), tmp_path)
        with pytest.raises(ValueError):
            fit_real_quadratic_hawkes_params(tmp_path, ticker="SPY", sigma_threshold=2.0)

    def test_fits_successfully_on_synthetic_tick_data(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        fit, marks = fit_real_quadratic_hawkes_params(tmp_path, ticker="SPY", sigma_threshold=2.0)
        assert isinstance(fit, QuadraticHawkesFitResult)
        assert fit.lambda0 > 0
        assert fit.kappa >= 0 and fit.alpha >= 0
        assert len(marks) > 0

    def test_default_betas_are_equal_and_data_adaptive(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        fit, _ = fit_real_quadratic_hawkes_params(tmp_path, ticker="SPY", sigma_threshold=2.0)
        assert fit.beta_leverage == fit.beta
        assert fit.beta > 0


class TestGenerateQuadraticHawkesPaths:
    def test_end_to_end_on_synthetic_data(self, tmp_path):
        write_ticks(_synthetic_ticks("SPY", "2026-01-02 09:30", 4000, burst_at=BURSTS), tmp_path)
        write_bars(_synthetic_bars("2026-01-02 09:30", 4000), tmp_path)

        paths = generate_quadratic_hawkes_paths(tmp_path, ticker="SPY", T_days=0.01, n_sims=2, seed=0)
        assert len(paths) == 2
        for p in paths:
            assert p.generator_id == "quadratic_hawkes"
            assert len(p.log_returns) > 0
            assert "kappa" in p.params
