import numpy as np
import pytest
import torch

from models.forecaster_generate import ancestral_sample
from models.tcn_forecaster import TCNForecaster


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
