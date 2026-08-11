import asyncio
import os
from dataclasses import dataclass, field

import pandas as pd
import pytest

from ingest.storage import read_ticks
from ingest.tick_recorder import (
    AlreadyRunningError,
    TickRecorder,
    acquire_singleton_lock,
    release_singleton_lock,
)


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
        rec.wait_for_pending_flushes()
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
        rec.wait_for_pending_flushes()
        out = read_ticks(tmp_path)
        assert out["timestamp"].dt.tz is not None

    def test_conditions_joined_to_string(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        trade = _FakeTrade(
            "AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10, conditions=["@", "T"]
        )
        _run(rec.on_trade(trade))
        rec.wait_for_pending_flushes()
        out = read_ticks(tmp_path)
        assert out.iloc[0]["conditions"] == "@,T"

    def test_empty_conditions_becomes_empty_string(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        trade = _FakeTrade("AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10, conditions=[])
        _run(rec.on_trade(trade))
        rec.wait_for_pending_flushes()
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
        rec.wait_for_pending_flushes()  # ensure the first background write lands before the second starts
        _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=1), 100.0, 10)))
        rec.wait_for_pending_flushes()
        out = read_ticks(tmp_path)
        assert len(out) == 2


class TestAsyncFlushBehavior:
    """Regression coverage for the actual bug found on real data: on_trade
    calling flush_buffer() synchronously blocked the websocket's event
    loop during real trade bursts, plausibly causing the frequent
    live "no close frame received or sent" disconnects observed against
    the real Alpaca feed (see diagnostics/2026-08-11-tick-recorder-blocking-flush/).
    Automatic flushes now run on a background thread instead.
    """

    def test_on_trade_clears_buffer_immediately_without_waiting_for_disk_write(self, tmp_path, monkeypatch):
        # Prove the buffer swap (the part that must stay fast, in-loop) is
        # decoupled from the disk write (the part now pushed to a
        # background thread) -- block the background writer with an event
        # so we can observe buffer state mid-flight.
        import threading

        writer_may_proceed = threading.Event()
        real_write = TickRecorder._write_to_disk

        def blocking_write(self, records):
            writer_may_proceed.wait(timeout=2)
            return real_write(self, records)

        monkeypatch.setattr(TickRecorder, "_write_to_disk", blocking_write)

        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        _run(rec.on_trade(_FakeTrade("AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10)))

        # on_trade has returned, but the background write is still
        # deliberately blocked -- buffer must already be clear (swapped
        # out synchronously) even though nothing has hit disk yet.
        assert rec._buffer == []
        assert read_ticks(tmp_path).empty

        writer_may_proceed.set()
        rec.wait_for_pending_flushes()
        assert len(read_ticks(tmp_path)) == 1

    def test_wait_for_pending_flushes_clears_completed_futures(self, tmp_path):
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        _run(rec.on_trade(_FakeTrade("AAPL", pd.Timestamp("2026-01-02 09:30:00", tz="UTC"), 100.0, 10)))
        assert len(rec._pending_flushes) == 1
        rec.wait_for_pending_flushes()
        assert rec._pending_flushes == []

    def test_rapid_back_to_back_flushes_do_not_corrupt_or_lose_data(self, tmp_path):
        # max_workers=1 on the executor should serialize writes to the
        # same partition file rather than racing -- fire many
        # threshold-triggering trades back to back with no manual waiting
        # in between, then confirm every single one made it to disk.
        rec = TickRecorder(tmp_path, flush_size=1, flush_interval=pd.Timedelta(seconds=3600))
        base = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        for i in range(30):
            _run(rec.on_trade(_FakeTrade("AAPL", base + pd.Timedelta(seconds=i), 100.0, 10)))
        rec.wait_for_pending_flushes()
        out = read_ticks(tmp_path)
        assert len(out) == 30
        assert out["timestamp"].is_unique


class TestSingletonLock:
    """Regression coverage for the real operational incident this
    prevents: an old instance not fully stopped plus a manual restart both
    holding live connections at once exhausted Alpaca's per-account
    connection limit, and the SDK's own reconnect loop has no backoff on
    that specific error path, hammering the API until manually killed. See
    diagnostics/2026-08-11-tick-recorder-connection-limit-incident/.
    """

    def test_acquires_lock_when_none_exists(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        acquire_singleton_lock(lock_path)
        assert lock_path.exists()
        assert int(lock_path.read_text()) == os.getpid()

    def test_raises_when_another_live_process_holds_the_lock(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        # Simulate another process's lock via a PID that's genuinely alive
        # right now (this test process itself).
        lock_path.write_text(str(os.getpid()))
        with pytest.raises(AlreadyRunningError):
            acquire_singleton_lock(lock_path)

    def test_recovers_from_stale_lock_left_by_a_crashed_process(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        # A PID essentially guaranteed not to be a live process.
        lock_path.write_text("999999")
        acquire_singleton_lock(lock_path)  # must not raise
        assert int(lock_path.read_text()) == os.getpid()

    def test_ignores_corrupt_lock_file_contents(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        lock_path.write_text("not-a-pid")
        acquire_singleton_lock(lock_path)  # must not raise
        assert int(lock_path.read_text()) == os.getpid()

    def test_creates_parent_directory_if_missing(self, tmp_path):
        lock_path = tmp_path / "nested" / "dir" / "recorder.lock"
        acquire_singleton_lock(lock_path)
        assert lock_path.exists()

    def test_release_removes_the_lock_file(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        acquire_singleton_lock(lock_path)
        release_singleton_lock(lock_path)
        assert not lock_path.exists()

    def test_release_is_a_noop_when_no_lock_exists(self, tmp_path):
        lock_path = tmp_path / "recorder.lock"
        release_singleton_lock(lock_path)  # must not raise
