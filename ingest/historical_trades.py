"""Historical trade-level (tick) backfill via Alpaca's historical trades
endpoint (`StockHistoricalDataClient.get_stock_trades`) -- confirmed
available on the free IEX feed with at least a year of real depth (see
diagnostics/2026-08-11-historical-tick-backfill/findings.md), not just the
live websocket. This unblocks the Rung 1 real-tick Hawkes replication test
immediately instead of waiting days/weeks for ingest/tick_recorder.py to
accumulate enough live data -- both write into the same
`data/ticks` partitioned store (ingest/storage.py) via the same schema, so
`read_ticks()` and the Rung 1 gate in scripts/run_ladder.py see backfilled
and live-recorded ticks identically.

Chunked by day (unlike historical_bars.py's 30-day chunks) because trade
volume is orders of magnitude higher than bar volume -- keeps individual
requests and partition writes a manageable size and makes a partial
failure cheap to resume (storage.write_ticks de-duplicates on timestamp).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockTradesRequest

from ingest.alpaca_client import get_client
from ingest.storage import latest_timestamp, write_ticks


def trades_to_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Converts alpaca-py's trades .df (multi-indexed by symbol, timestamp)
    into the storage schema (ingest/storage.py's TICK_COLUMNS +
    TICK_OPTIONAL_COLUMNS). `conditions` is joined into a comma-separated
    string (not left as a list) to match ingest/tick_recorder.py's on_trade
    exactly -- a live-recorded and a backfilled tick for the same trade
    must produce byte-identical rows, or de-duplication and any downstream
    schema assumption could silently break.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "price", "size", "exchange", "conditions"])

    df = trades_df.reset_index().rename(columns={"symbol": "ticker"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(int)
    df["conditions"] = df["conditions"].apply(lambda c: ",".join(c) if isinstance(c, list) and c else "")
    return df[["timestamp", "ticker", "price", "size", "exchange", "conditions"]]


def fetch_historical_trades(
    client,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    feed: DataFeed = DataFeed.IEX,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch real historical trades for `symbols` in [start, end];
    alpaca-py paginates internally (confirmed: a full trading day, ~17.7k
    SPY trades, returns in under a second with no explicit pagination
    loop needed here). Retries with exponential backoff on API errors,
    matching ingest/alpaca_client.py::fetch_minute_bars.
    """
    request = StockTradesRequest(
        symbol_or_symbols=symbols,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=feed,
    )
    last_error: APIError | None = None
    for attempt in range(max_retries):
        try:
            return trades_to_frame(client.get_stock_trades(request).df)
        except APIError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise last_error


def backfill_trades(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: Path,
    chunk_days: int = 1,
    client=None,
) -> dict[str, int]:
    """Full historical tick backfill, ticker by ticker, one day per request
    by default. Returns {ticker: rows_written}.
    """
    client = client or get_client()
    rows_written: dict[str, int] = {}

    for ticker in symbols:
        total = 0
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
            df = fetch_historical_trades(client, [ticker], chunk_start, chunk_end)
            if not df.empty:
                total += write_ticks(df, data_dir)
            chunk_start = chunk_end
        rows_written[ticker] = total

    return rows_written


def update_incremental(
    symbols: list[str],
    data_dir: Path,
    end: pd.Timestamp | None = None,
    client=None,
) -> dict[str, int]:
    """Fetch only trades newer than the last stored tick per ticker -- safe
    to run repeatedly. A ticker with no stored ticks yet is a no-op (run
    backfill_trades first).
    """
    client = client or get_client()
    end = end or pd.Timestamp.now(tz="UTC")
    rows_written: dict[str, int] = {}

    for ticker in symbols:
        last = latest_timestamp(data_dir, "ticks", ticker)
        if last is None:
            rows_written[ticker] = 0
            continue
        start = last + pd.Timedelta(microseconds=1)
        if start >= end:
            rows_written[ticker] = 0
            continue
        df = fetch_historical_trades(client, [ticker], start, end)
        rows_written[ticker] = write_ticks(df, data_dir) if not df.empty else 0

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill real historical trade-level (tick) data.")
    parser.add_argument("--tickers", nargs="+", default=["SPY"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=str, help="e.g. 2026-06-01 (required unless --incremental)")
    parser.add_argument("--end", type=str, help="e.g. 2026-08-01 (defaults to now if --incremental)")
    parser.add_argument("--incremental", action="store_true", help="Fetch only new ticks since last stored timestamp")
    args = parser.parse_args()

    if args.incremental:
        end = pd.Timestamp(args.end, tz="UTC") if args.end else None
        result = update_incremental(args.tickers, args.data_dir, end=end)
    else:
        if not args.start or not args.end:
            parser.error("--start and --end are required unless --incremental")
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC")
        result = backfill_trades(args.tickers, start, end, args.data_dir)

    for ticker, n in result.items():
        print(f"{ticker}: {n} rows written")


if __name__ == "__main__":
    main()
