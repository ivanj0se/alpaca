import numpy as np
import pandas as pd

from events.price_events import (
    bar_threshold_events,
    event_times_array,
    merge_event_sources,
    tick_events_from_recorder,
)


def _bars(ticker, closes, start="2026-01-02 09:30", freq_min=1):
    idx = pd.date_range(start, periods=len(closes), freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "ticker": ticker, "open": closes, "high": closes, "low": closes, "close": closes, "volume": 100}
    )


def _ticks(ticker, prices, start="2026-01-02 09:30:00"):
    idx = pd.date_range(start, periods=len(prices), freq="1s", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "ticker": ticker, "price": prices, "size": 10})


class TestBarThresholdEvents:
    def test_flags_large_moves_only(self):
        # Mostly tiny noise, one clear outlier jump.
        rng = np.random.default_rng(0)
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 40)))
        base[20] *= 1.05  # a large, obvious jump relative to the noise scale
        df = _bars("AAPL", base)
        events = bar_threshold_events(df, sigma_threshold=3.0)
        assert len(events) >= 1
        assert set(events["ticker"]) == {"AAPL"}
        assert (events["abs_zscore"] >= 3.0).all()
        assert set(events["provenance"]) == {"bar_proxy"}

    def test_empty_input_returns_empty_with_columns(self):
        out = bar_threshold_events(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == ["timestamp", "ticker", "abs_zscore", "provenance"]

    def test_constant_prices_produce_no_events_no_crash(self):
        df = _bars("AAPL", [100.0] * 20)
        out = bar_threshold_events(df, sigma_threshold=2.0)
        assert out.empty

    def test_multiple_tickers_scored_independently(self):
        rng = np.random.default_rng(1)
        aapl = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 30)))
        aapl[15] *= 1.05
        msft = 200 * np.exp(np.cumsum(rng.normal(0, 0.0005, 30)))
        msft[10] *= 1.05
        df = pd.concat([_bars("AAPL", aapl), _bars("MSFT", msft)])
        events = bar_threshold_events(df, sigma_threshold=3.0)
        assert set(events["ticker"]) >= {"AAPL", "MSFT"}


class TestTickEventsFromRecorder:
    def test_flags_large_price_jumps(self):
        rng = np.random.default_rng(2)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, 50)))
        prices[25] *= 1.03
        df = _ticks("AAPL", prices)
        events = tick_events_from_recorder(df, sigma_threshold=3.0)
        assert len(events) >= 1
        assert set(events["provenance"]) == {"real_tick"}

    def test_too_few_ticks_no_crash(self):
        df = _ticks("AAPL", [100.0, 100.1])
        out = tick_events_from_recorder(df, sigma_threshold=2.0)
        assert out.empty

    def test_empty_input(self):
        out = tick_events_from_recorder(pd.DataFrame())
        assert out.empty


class TestMergeEventSources:
    def test_concatenates_and_preserves_provenance(self):
        bar_ev = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-02 09:31", tz="UTC")],
                "ticker": ["AAPL"],
                "abs_zscore": [3.5],
                "provenance": ["bar_proxy"],
            }
        )
        tick_ev = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-02 09:30:05", tz="UTC")],
                "ticker": ["AAPL"],
                "abs_zscore": [4.0],
                "provenance": ["real_tick"],
            }
        )
        merged = merge_event_sources(bar_ev, tick_ev)
        assert len(merged) == 2
        assert set(merged["provenance"]) == {"bar_proxy", "real_tick"}
        assert merged["timestamp"].is_monotonic_increasing

    def test_all_empty_returns_empty(self):
        out = merge_event_sources(pd.DataFrame(), pd.DataFrame())
        assert out.empty


class TestEventTimesArray:
    def test_converts_to_seconds_since_first_event(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-02 09:30:00", "2026-01-02 09:30:10", "2026-01-02 09:31:00"], utc=True
                ),
                "ticker": ["AAPL"] * 3,
            }
        )
        arr = event_times_array(df)
        assert np.allclose(arr, [0.0, 10.0, 60.0])

    def test_filters_by_ticker(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-02 09:30:00", "2026-01-02 09:30:10"], utc=True),
                "ticker": ["AAPL", "MSFT"],
            }
        )
        arr = event_times_array(df, ticker="AAPL")
        assert np.allclose(arr, [0.0])

    def test_empty_returns_empty_array(self):
        arr = event_times_array(pd.DataFrame(columns=["timestamp", "ticker"]))
        assert arr.size == 0
