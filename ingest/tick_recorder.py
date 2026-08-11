"""Continuous live trade recorder. Records only -- makes no decisions, and
runs independently of everything else in the pipeline. This exists because
free historical tick-level data doesn't exist any other way: the Hawkes
branching-ratio rung (events/hawkes.py) needs true event timestamps, not
bars, so this should start running under a supervisor (launchd/tmux/systemd)
as soon as Alpaca credentials are configured -- well before the rest of the
pipeline is built.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import signal
import threading
from pathlib import Path

import pandas as pd

from ingest.alpaca_client import get_stream_client
from ingest.historical_bars import load_universe
from ingest.storage import write_ticks

DEFAULT_LOCK_PATH = Path("logs/tick_recorder.lock")


class AlreadyRunningError(RuntimeError):
    pass


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_singleton_lock(lock_path: Path) -> None:
    """Refuses to start if another tick_recorder process is already
    running. Prevents a real failure mode found operationally: an old
    instance not yet fully stopped plus a manual restart both holding live
    websocket connections at once exhausts Alpaca's per-account connection
    limit -- and once that happens, alpaca-py's own reconnect loop hits a
    ValueError code path that (unlike its WebSocketException handling)
    isn't followed by any backoff, so it retries essentially instantly,
    hammering the API in a tight loop until manually killed.
    """
    lock_path = Path(lock_path)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text().strip())
        except ValueError:
            existing_pid = None
        if existing_pid is not None and _pid_is_alive(existing_pid):
            raise AlreadyRunningError(
                f"another tick_recorder is already running (pid {existing_pid}, lock file {lock_path}) "
                "-- stop it first (`launchctl bootout ...`), don't start a second one. If you're certain "
                "it's a stale lock from a crash, delete the lock file and retry."
            )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))


def release_singleton_lock(lock_path: Path) -> None:
    Path(lock_path).unlink(missing_ok=True)


class TickRecorder:
    """Buffers incoming trades in memory and flushes to partitioned parquet
    (ingest/storage.py) when the buffer grows past `flush_size` or the
    oldest buffered trade is older than `flush_interval`, whichever comes
    first. `run_forever` also flushes on shutdown so no buffered trades are
    lost on a clean stop.

    Automatic flushes triggered from `on_trade` run on a background thread,
    not inline on the event loop. `on_trade` runs directly on the
    websocket connection's asyncio event loop; the original implementation
    called flush_buffer() (synchronous parquet read-merge-write, one
    round-trip per distinct ticker in the buffer) straight from there,
    which blocks that event loop -- including the low-level keepalive
    frames the websocket connection needs serviced -- for however long the
    write takes. Confirmed as the likely cause of frequent live
    "no close frame received or sent" disconnects on real data: burst of
    trades -> buffer hits flush_size mid-burst -> blocking write stalls
    the loop -> connection drops -> reconnect -> repeat, with real
    unrecoverable trade data lost in each gap. A dedicated single-worker
    executor serializes writes (avoiding concurrent-write races on the
    same partition file) off the event loop instead.
    """

    def __init__(
        self,
        data_dir: Path,
        flush_size: int = 500,
        flush_interval: pd.Timedelta = pd.Timedelta(seconds=30),
    ):
        self.data_dir = Path(data_dir)
        self.flush_size = flush_size
        self.flush_interval = flush_interval
        self._buffer: list[dict] = []
        self._buffer_opened_at: pd.Timestamp | None = None
        self.total_flushed = 0
        self._buffer_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending_flushes: list[concurrent.futures.Future] = []

    async def on_trade(self, trade) -> None:
        ts = pd.Timestamp(trade.timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        conditions = getattr(trade, "conditions", None)

        record = {
            "timestamp": ts,
            "ticker": trade.symbol,
            "price": float(trade.price),
            "size": int(trade.size),
            "exchange": getattr(trade, "exchange", None),
            "conditions": ",".join(conditions) if conditions else "",
        }

        to_flush = None
        with self._buffer_lock:
            self._buffer.append(record)
            if self._buffer_opened_at is None:
                self._buffer_opened_at = ts

            buffer_full = len(self._buffer) >= self.flush_size
            buffer_stale = ts - self._buffer_opened_at >= self.flush_interval
            if buffer_full or buffer_stale:
                to_flush, self._buffer = self._buffer, []
                self._buffer_opened_at = None

        if to_flush:
            self._pending_flushes.append(self._executor.submit(self._write_to_disk, to_flush))

    def _write_to_disk(self, records: list[dict]) -> int:
        df = pd.DataFrame(records)
        n = write_ticks(df, self.data_dir)
        self.total_flushed += n
        return n

    def wait_for_pending_flushes(self, timeout: float | None = None) -> None:
        """Blocks until every background flush submitted so far has
        completed. The live recorder never needs this itself (shutdown
        uses executor.shutdown(wait=True)) -- it exists so tests/tools can
        deterministically wait for a background write instead of sleeping.
        """
        concurrent.futures.wait(self._pending_flushes, timeout=timeout)
        self._pending_flushes = [f for f in self._pending_flushes if not f.done()]

    def flush_buffer(self) -> int:
        """Synchronous flush -- writes inline and blocks until done. Used
        at shutdown (where blocking briefly is correct: we want the
        process to stay alive until the write finishes) and for direct/manual
        calls; the automatic hot-path flush from on_trade never calls this,
        it uses the background executor instead.
        """
        with self._buffer_lock:
            to_flush, self._buffer = self._buffer, []
            self._buffer_opened_at = None
        if not to_flush:
            return 0
        return self._write_to_disk(to_flush)

    def run_forever(self, universe_path: Path, feed=None, lock_path: Path = DEFAULT_LOCK_PATH) -> None:
        """Subscribe to trades for the universe and block until stopped.

        SIGTERM is translated into SIGINT (self-signaled) rather than
        handled directly, so shutdown flows through alpaca-py's own tested
        KeyboardInterrupt-catching path inside `StockDataStream.run()`
        instead of racing its internal asyncio event loop from a separate
        signal-handler code path.

        Acquires a singleton lock first (see acquire_singleton_lock) --
        raises AlreadyRunningError immediately, before ever touching the
        network, if another instance is already live.
        """
        from alpaca.data.enums import DataFeed

        acquire_singleton_lock(lock_path)
        try:
            stream = get_stream_client(feed=feed or DataFeed.IEX)
            tickers = load_universe(universe_path)
            stream.subscribe_trades(self.on_trade, *tickers)

            def _sigterm_to_sigint(signum, frame):
                os.kill(os.getpid(), signal.SIGINT)

            signal.signal(signal.SIGTERM, _sigterm_to_sigint)

            try:
                stream.run()
            finally:
                self._executor.shutdown(wait=True)  # let any in-flight background flush finish first
                self.flush_buffer()
        finally:
            release_singleton_lock(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record live trades to parquet. Records only, no decisions.")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--flush-size", type=int, default=500)
    parser.add_argument("--flush-interval-seconds", type=int, default=30)
    args = parser.parse_args()

    recorder = TickRecorder(
        args.data_dir,
        flush_size=args.flush_size,
        flush_interval=pd.Timedelta(seconds=args.flush_interval_seconds),
    )
    recorder.run_forever(args.universe)


if __name__ == "__main__":
    main()
