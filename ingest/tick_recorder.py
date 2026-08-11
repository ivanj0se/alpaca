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
import os
import signal
from pathlib import Path

import pandas as pd

from ingest.alpaca_client import get_stream_client
from ingest.historical_bars import load_universe
from ingest.storage import write_ticks


class TickRecorder:
    """Buffers incoming trades in memory and flushes to partitioned parquet
    (ingest/storage.py) when the buffer grows past `flush_size` or the
    oldest buffered trade is older than `flush_interval`, whichever comes
    first. `run_forever` also flushes on shutdown so no buffered trades are
    lost on a clean stop.
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

    async def on_trade(self, trade) -> None:
        ts = pd.Timestamp(trade.timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        conditions = getattr(trade, "conditions", None)

        self._buffer.append(
            {
                "timestamp": ts,
                "ticker": trade.symbol,
                "price": float(trade.price),
                "size": int(trade.size),
                "exchange": getattr(trade, "exchange", None),
                "conditions": ",".join(conditions) if conditions else "",
            }
        )
        if self._buffer_opened_at is None:
            self._buffer_opened_at = ts

        buffer_full = len(self._buffer) >= self.flush_size
        buffer_stale = ts - self._buffer_opened_at >= self.flush_interval
        if buffer_full or buffer_stale:
            self.flush_buffer()

    def flush_buffer(self) -> int:
        if not self._buffer:
            return 0
        df = pd.DataFrame(self._buffer)
        n = write_ticks(df, self.data_dir)
        self.total_flushed += n
        self._buffer = []
        self._buffer_opened_at = None
        return n

    def run_forever(self, universe_path: Path, feed=None) -> None:
        """Subscribe to trades for the universe and block until stopped.

        SIGTERM is translated into SIGINT (self-signaled) rather than
        handled directly, so shutdown flows through alpaca-py's own tested
        KeyboardInterrupt-catching path inside `StockDataStream.run()`
        instead of racing its internal asyncio event loop from a separate
        signal-handler code path.
        """
        from alpaca.data.enums import DataFeed

        stream = get_stream_client(feed=feed or DataFeed.IEX)
        tickers = load_universe(universe_path)
        stream.subscribe_trades(self.on_trade, *tickers)

        def _sigterm_to_sigint(signum, frame):
            os.kill(os.getpid(), signal.SIGINT)

        signal.signal(signal.SIGTERM, _sigterm_to_sigint)

        try:
            stream.run()
        finally:
            self.flush_buffer()


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
