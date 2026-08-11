"""Historical minute-bar backfill for the universe defined in
config/universe.yaml. Resumable/idempotent -- safe to re-run after a
rate-limit failure (storage.write_bars de-duplicates) or to pick up where a
previous run left off (update_incremental).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from alpaca.data.enums import Adjustment

from ingest.alpaca_client import fetch_minute_bars, get_client
from ingest.storage import latest_timestamp, write_bars


def load_universe(universe_path: Path) -> list[str]:
    with open(universe_path) as f:
        cfg = yaml.safe_load(f)
    tickers = [row["ticker"] for row in cfg["universe"]]
    tickers.append(cfg["benchmark_instrument"]["ticker"])
    return tickers


def backfill_universe(
    universe_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: Path,
    chunk_days: int = 30,
    client=None,
) -> dict[str, int]:
    """Full historical pull, ticker by ticker, chunked to keep individual
    requests small -- friendlier to the free-tier rate limit and cheaper to
    resume after a partial failure than one giant multi-year request.
    Returns {ticker: rows_written}.
    """
    client = client or get_client()
    tickers = load_universe(universe_path)
    rows_written: dict[str, int] = {}

    for ticker in tickers:
        total = 0
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days), end)
            df = fetch_minute_bars(client, [ticker], chunk_start, chunk_end, adjustment=Adjustment.ALL)
            if not df.empty:
                total += write_bars(df, data_dir)
            chunk_start = chunk_end
        rows_written[ticker] = total

    return rows_written


def update_incremental(
    universe_path: Path,
    data_dir: Path,
    end: pd.Timestamp | None = None,
    client=None,
) -> dict[str, int]:
    """Fetch only bars newer than the last stored timestamp per ticker. Safe
    to run repeatedly (e.g. daily). A ticker with no stored data yet is a
    no-op here -- run backfill_universe first.
    """
    client = client or get_client()
    tickers = load_universe(universe_path)
    end = end or pd.Timestamp.now(tz="UTC")
    rows_written: dict[str, int] = {}

    for ticker in tickers:
        last = latest_timestamp(data_dir, "bars", ticker)
        if last is None:
            rows_written[ticker] = 0
            continue
        start = last + pd.Timedelta(minutes=1)
        if start >= end:
            rows_written[ticker] = 0
            continue
        df = fetch_minute_bars(client, [ticker], start, end, adjustment=Adjustment.ALL)
        rows_written[ticker] = write_bars(df, data_dir) if not df.empty else 0

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical minute bars for the universe.")
    parser.add_argument("--universe", type=Path, default=Path("config/universe.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=str, help="e.g. 2023-01-01 (required unless --incremental)")
    parser.add_argument("--end", type=str, help="e.g. 2026-08-01 (defaults to now if --incremental)")
    parser.add_argument(
        "--incremental", action="store_true", help="Fetch only new bars since last stored timestamp"
    )
    args = parser.parse_args()

    if args.incremental:
        end = pd.Timestamp(args.end, tz="UTC") if args.end else None
        result = update_incremental(args.universe, args.data_dir, end=end)
    else:
        if not args.start or not args.end:
            parser.error("--start and --end are required unless --incremental")
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC")
        result = backfill_universe(args.universe, start, end, args.data_dir)

    for ticker, n in result.items():
        print(f"{ticker}: {n} rows written")


if __name__ == "__main__":
    main()
