"""Most of scripts/run_generator_comparison.py is orchestration (reads
real data, trains real models, runs real simulations) -- an integration
target validated by actually running it (see diagnostics/2026-08-12-*/),
the same way scripts/run_ladder.py's main() isn't unit tested but its
constituent pure helpers are. This covers the pure, non-orchestration
helpers.
"""

import pandas as pd
import yaml

from scripts.run_generator_comparison import build_gbm_paths, load_settings


class TestLoadSettings:
    def test_loads_yaml_file(self, tmp_path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(yaml.dump({"generators": {"hawkes_jump_diffusion": {"ticker": "SPY"}}}))
        settings = load_settings(settings_path)
        assert settings["generators"]["hawkes_jump_diffusion"]["ticker"] == "SPY"


class TestBuildGbmPaths:
    def test_returns_expected_number_of_paths(self):
        prices = pd.Series([100.0 + i * 0.01 for i in range(200)])
        paths = build_gbm_paths(prices, n_steps=50, n_sims=5, seed=0)
        assert len(paths) == 5

    def test_paths_have_expected_generator_id_and_length(self):
        prices = pd.Series([100.0 + i * 0.01 for i in range(200)])
        paths = build_gbm_paths(prices, n_steps=50, n_sims=3, seed=0)
        for p in paths:
            assert p.generator_id == "gbm_null"
            assert len(p.log_returns) == 50  # diff of n_steps+1 price points

    def test_reproducible_with_seed(self):
        prices = pd.Series([100.0 + i * 0.01 for i in range(200)])
        a = build_gbm_paths(prices, n_steps=30, n_sims=2, seed=7)
        b = build_gbm_paths(prices, n_steps=30, n_sims=2, seed=7)
        for pa, pb in zip(a, b):
            assert (pa.log_returns == pb.log_returns).all()
