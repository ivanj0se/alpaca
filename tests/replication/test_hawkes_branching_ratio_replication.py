"""Rung 1 trust gate, real-data half. See
tests/unit/test_hawkes.py::TestSimulateRefitRecover and
::TestNullRandomWalkSpecificity for the synthetic fitter-correctness half
(does the MLE math work at all, and does it avoid hallucinating
self-excitation in pure noise).

This file is about what real data actually shows. It does NOT expect
minute-bar SPY data to reproduce Filimonov & Sornette's published ~0.81
branching ratio -- diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/findings.md
documents why that's structurally impossible (minute bars are
near-regularly spaced, Fano factor 0.82 on real data -- underdispersed, the
opposite of the bursty self-excitation a Hawkes process detects). Instead:

1. `test_minute_bar_proxy_shows_near_zero_self_excitation` asserts the
   *correct, expected* minute-bar result on a frozen real SPY fixture.
2. `test_tick_level_replication_against_published_branching_ratio` is the
   actual "does it land near 0.81" check, gated on enough real tick data
   having accumulated via ingest/tick_recorder.py -- it skips gracefully
   until there's enough, then runs for real.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from events.hawkes import fit_hawkes_exponential, branching_ratio
from events.price_events import bar_threshold_events, event_times_array, tick_events_from_recorder
from ingest.storage import read_ticks

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "spy_bars_120d.parquet"
SETTINGS_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

MIN_TICKS_FOR_REPLICATION = 5000
MIN_SPAN_DAYS_FOR_REPLICATION = 5


def _settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


class TestMinuteBarProxy:
    def test_minute_bar_proxy_shows_near_zero_self_excitation(self):
        settings = _settings()
        df = pd.read_parquet(FIXTURE_PATH)
        events = bar_threshold_events(df, sigma_threshold=settings["hawkes"]["bar_event_sigma_threshold"])
        times = event_times_array(events, ticker="SPY")
        assert len(times) > 100, "fixture should produce a substantial number of threshold events"

        fit = fit_hawkes_exponential(times)
        assert fit.converged
        ratio = branching_ratio(fit)
        max_expected = settings["hawkes"]["bar_proxy_max_branching_ratio"]
        assert ratio < max_expected, (
            f"branching ratio {ratio:.4f} exceeds the expected near-zero bar-proxy "
            f"result ({max_expected}) -- if this starts failing, something about the "
            f"proxy construction changed; see diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/"
        )


class TestTickLevelReplication:
    def test_tick_level_replication_against_published_branching_ratio(self):
        settings = _settings()
        ticks = read_ticks(DATA_DIR, tickers=["SPY"])

        if ticks.empty:
            pytest.skip("no real SPY tick data recorded yet -- run ingest/tick_recorder.py for a while first")

        span_days = (ticks["timestamp"].max() - ticks["timestamp"].min()).total_seconds() / 86400
        if len(ticks) < MIN_TICKS_FOR_REPLICATION or span_days < MIN_SPAN_DAYS_FOR_REPLICATION:
            pytest.skip(
                f"only {len(ticks)} SPY ticks over {span_days:.1f} days recorded so far "
                f"(need >= {MIN_TICKS_FOR_REPLICATION} ticks over >= {MIN_SPAN_DAYS_FOR_REPLICATION} days) "
                "-- this is the real trust-gate check, it will start running once enough live data has accumulated"
            )

        events = tick_events_from_recorder(ticks, sigma_threshold=settings["hawkes"]["bar_event_sigma_threshold"])
        times = event_times_array(events, ticker="SPY")
        fit = fit_hawkes_exponential(times)
        assert fit.converged

        ratio = branching_ratio(fit)
        low, high = settings["hawkes"]["plausible_band"]
        assert low <= ratio <= high, (
            f"tick-level branching ratio {ratio:.4f} falls outside the documented plausible "
            f"band [{low}, {high}] around the published Filimonov & Sornette (2012) figure "
            f"({settings['hawkes']['published_branching_ratio']}) -- this is the actual trust "
            f"gate; downstream rungs should not be trusted until this passes"
        )
