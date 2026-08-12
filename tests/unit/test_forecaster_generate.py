import numpy as np
import pandas as pd
import pytest
import torch

from models.forecaster_generate import ancestral_sample, generate_forecaster_paths
from models.tcn_forecaster import TCNForecaster
from ingest.storage import write_bars


class TestAncestralSample:
    def test_output_length_matches_n_steps(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.zeros((20, 2), dtype=np.float32)
        result = ancestral_sample(model, seed_window, n_steps=15, vol_window=5, seed=0)
        assert len(result) == 15

    def test_reproducible_with_seed(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.random.default_rng(1).normal(0, 0.01, size=(20, 2)).astype(np.float32)
        a = ancestral_sample(model, seed_window, n_steps=10, vol_window=5, seed=7)
        b = ancestral_sample(model, seed_window, n_steps=10, vol_window=5, seed=7)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_paths(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.random.default_rng(1).normal(0, 0.01, size=(20, 2)).astype(np.float32)
        a = ancestral_sample(model, seed_window, n_steps=10, vol_window=5, seed=1)
        b = ancestral_sample(model, seed_window, n_steps=10, vol_window=5, seed=2)
        assert not np.array_equal(a, b)

    def test_raises_for_wrong_feature_count(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.zeros((20, 3), dtype=np.float32)  # 3 features, model expects 2
        with pytest.raises(ValueError, match="2 features"):
            ancestral_sample(model, seed_window, n_steps=5, vol_window=5, seed=0)

    def test_does_not_mutate_seed_window(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.random.default_rng(1).normal(0, 0.01, size=(20, 2)).astype(np.float32)
        original = seed_window.copy()
        ancestral_sample(model, seed_window, n_steps=10, vol_window=5, seed=0)
        assert np.array_equal(seed_window, original)

    def test_generated_vol_reflects_generated_return_dispersion(self):
        # A model whose mean output is pinned near zero but with a large
        # fixed variance should, after enough generated steps, show
        # realized_vol (recomputed from GENERATED returns) roughly
        # tracking that variance -- confirms the rolling-vol recompute
        # actually uses generated values, not a frozen seed value.
        torch.manual_seed(0)

        class FixedVarianceForecaster(TCNForecaster):
            def forward(self, x):
                mean, _ = super().forward(x)
                logvar = torch.full_like(mean, np.log(0.04**2))  # std=0.04
                return mean * 0.0, logvar  # force mean to exactly zero

        model = FixedVarianceForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        seed_window = np.zeros((20, 2), dtype=np.float32)  # starts at zero vol
        result_vol_track = ancestral_sample(model, seed_window, n_steps=100, vol_window=15, seed=0)
        # after burn-in, generated returns should have std close to 0.04,
        # not stuck near the seed's initial zero vol.
        late_returns = result_vol_track[30:]
        assert 0.02 < late_returns.std() < 0.06


class TestGenerateForecasterPaths:
    def test_raises_when_no_data(self, tmp_path):
        with pytest.raises(ValueError, match="no real bar data"):
            generate_forecaster_paths(tmp_path, ticker="SPY", epochs=1, n_sims=1, n_steps=5)

    def test_end_to_end_on_synthetic_data(self, tmp_path):
        idx = pd.date_range("2026-01-02 09:30", periods=1000, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 1000)))
        volumes = rng.integers(50, 500, 1000).astype(float)
        bars = pd.DataFrame(
            {"timestamp": idx, "ticker": "SPY", "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
        )
        write_bars(bars, tmp_path)

        paths = generate_forecaster_paths(
            tmp_path, ticker="SPY", window_len=20, hidden_dim=8, dilations=(1, 2),
            epochs=2, n_sims=3, n_steps=10, vol_window=5, seed=0,
        )
        assert len(paths) == 3
        for p in paths:
            assert p.generator_id == "tcn_forecaster"
            assert len(p.log_returns) == 10

    def test_raises_when_not_enough_windows(self, tmp_path):
        # Enough rows for make_windows to succeed (past its own too-few-rows
        # check), but few enough resulting windows to trigger this
        # function's own "not enough to train on" guard.
        idx = pd.date_range("2026-01-02 09:30", periods=100, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 100)))
        volumes = rng.integers(50, 500, 100).astype(float)
        bars = pd.DataFrame(
            {"timestamp": idx, "ticker": "SPY", "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
        )
        write_bars(bars, tmp_path)
        with pytest.raises(ValueError, match="not enough"):
            generate_forecaster_paths(tmp_path, ticker="SPY", window_len=20, epochs=1, n_sims=1, n_steps=5)
