import asyncio
from dataclasses import dataclass, field

import pandas as pd

from ingest.storage import read_ticks
from ingest.tick_recorder import TickRecorder


@dataclass
class _FakeTrade:
    symbol: str
    timestamp: pd.Timestamp
    price: float
    size: int
    exchange: str = "V"
    conditions: list[str] = field(default_factory=lambda: ["@"])


def _run(coro):
    return asyncio.run(coro)


class TestOnTrade:
    def test_buffers_without_flushing_below_thresholds(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=10, flush_interval=pd.Timedelta(seconds=60))
        trade = _FakeTrade("AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10)
        _run(rec.on_trade(trade))
        assert len(rec._buffer) == 1
        assert read_ticks(tmp_path).empty

    def test_flushes_when_buffer_size_reached(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=3, flush_interval=pd.Timedelta(seconds=3600))
        base = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        for i in range(3):
            _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=i), 100.0, 10)))
        assert rec._buffer == []
        out = read_ticks(tmp_path)
        assert len(out) == 3

    def test_flushes_when_interval_exceeded(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1000, flush_interval=pd.Timedelta(seconds=5))
        base = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        _run(rec.on_trade(_FakeTrade("AAPL", base, 100.0, 10)))
        assert len(rec._buffer) == 1  # not yet stale
        _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=6), 100.0, 10)))
        assert rec._buffer == []  # second trade's timestamp made the buffer stale

    def test_naive_timestamp_localized_to_utc(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        naive_ts = pd.Timestamp("2026-01-02 09:30:00")
        _run(rec.on_trade(_FakeTrade("AAPL", naive_ts, 100.0, 10)))
        out = read_ticks(tmp_path)
        assert out["timestamp"].dt.tz is not None

    def test_conditions_joined_to_string(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        trade = _FakeTrade(
            "AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10, conditions=["@", "T"]
        )
        _run(rec.on_trade(trade))
        out = read_ticks(tmp_path)
        assert out.iloc[0]["conditions"] == "@,T"

    def test_empty_conditions_becomes_empty_string(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        trade = _FakeTrade("AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10, conditions=[])
        _run(rec.on_trade(trade))
        out = read_ticks(tmp_path)
        assert out.iloc[0]["conditions"] == ""


class TestFlushBuffer:
    def test_empty_buffer_is_noop(self, tmp_path):
        rec = TickRecorder(tmp_path)
        assert rec.flush_buffer() == 0

    def test_flush_resets_buffer_and_tracks_total(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1000, flush_interval=pd.Timedelta(seconds=3600))
        base = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        for i in range(5):
            _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=i), 100.0, 10)))
        n = rec.flush_buffer()
        assert n == 5
        assert rec.total_flushed == 5
        assert rec._buffer == []
        assert rec._buffer_opened_at is None

    def test_second_flush_merges_without_duplicates(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        base = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        _run(rec.on_trade(_FakeTrade("AAPL", base, 100.0, 10)))
        _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=1), 100.0, 10)))
        out = read_ticks(tmp_path)
        assert len(out) == 2
