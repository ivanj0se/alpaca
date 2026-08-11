"""Most of scripts/run_ladder.py is orchestration (reads real config/data,
trains real models, downloads real news) -- inherently an integration
target validated by actually running it (see diagnostics/2026-08-11-hawkes-optimizer-initialization-bug/
and the full-ladder-run reports for real-data validation history), the
same way ingest/historical_bars.py's main() isn't unit tested but its
constituent functions are. This covers the pure, non-orchestration helpers.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_ladder import load_config, log_returns_from_bars


class TestLoadConfig:
    def test_loads_both_yaml_files(self, tmp_path):
        universe_path = tmp_path / "universe.yaml"
        settings_path = tmp_path / "settings.yaml"
        universe_path.write_text(yaml.dump({"universe": [{"ticker": "AAPL", "name": "Apple"}]}))
        settings_path.write_text(yaml.dump({"cv": {"n_splits": 6}}))

        universe_cfg, settings = load_config(universe_path, settings_path)
        assert universe_cfg["universe"][0]["ticker"] == "AAPL"
        assert settings["cv"]["n_splits"] == 6


class TestLogReturnsFromBars:
    def test_computes_log_returns_sorted_by_timestamp(self):
        idx = pd.date_range("2026-01-02 09:30", periods=4, freq="1min", tz="UTC")
        bars = pd.DataFrame({"timestamp": idx, "close": [100.0, 101.0, 99.0, 100.0]}).sample(
            frac=1.0, random_state=0
        )  # shuffled input
        returns = log_returns_from_bars(bars)
        assert len(returns) == 3
        assert returns.index.is_monotonic_increasing
        assert returns.iloc[0] == pytest.approx(np.log(101.0 / 100.0))
