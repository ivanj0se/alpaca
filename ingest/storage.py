"""Partitioned parquet storage for bars and ticks.

Partition layout: ``{data_dir}/{kind}/ticker={TICKER}/date={YYYY-MM-DD}/data.parquet``.
One file per ticker per trading day keeps writes small and idempotent (the
tick recorder flushes continuously; historical backfills run once) and reads
cheap to filter by ticker/date range without loading everything into memory.

This is the schema/partitioning contract every other module relies on --
`features/`, `events/`, `rpca/`, and `models/` all read through
`read_bars`/`read_ticks` rather than touching parquet files directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BAR_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
BAR_OPTIONAL_COLUMNS = ["trade_count", "vwap"]

TICK_COLUMNS = ["timestamp", "ticker", "price", "size"]
TICK_OPTIONAL_COLUMNS = ["exchange", "conditions"]


def _validate_schema(df: pd.DataFrame, required: list[str], kind: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{kind} dataframe missing required columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError(f"{kind} dataframe 'timestamp' column must be datetime64")
    if df["timestamp"].dt.tz is None:
        raise ValueError(f"{kind} dataframe 'timestamp' column must be tz-aware (UTC)")


def _partition_path(data_dir: Path, kind: str, ticker: str, date: pd.Timestamp) -> Path:
    return data_dir / kind / f"ticker={ticker}" / f"date={date.date().isoformat()}" / "data.parquet"


def _write_partitioned(df: pd.DataFrame, data_dir: Path, kind: str, required: list[str]) -> int:
    """Write df to per-ticker/per-day partitions. Merges with any existing
    partition file and de-duplicates on full-row equality, so this is safe
    to call repeatedly (backfill re-runs, live recorder flush cycles)
    without creating duplicate rows. Returns the number of rows written.

    De-duplicating on `timestamp` alone (an earlier version of this
    function) is correct for bars (one bar per ticker per minute, by
    construction) but wrong for ticks: real trades legitimately share a
    timestamp down to the microsecond during a burst (e.g. one incoming
    order sweeping several resting orders produces multiple separate trade
    prints, often in the same tick of the matching engine) -- confirmed on
    real SPY data, ~4% of stored ticks were involved in a timestamp
    collision with a *different* price/size, i.e. a real distinct trade,
    not a duplicate write. `drop_duplicates(subset="timestamp")` was
    silently discarding one side of every such pair whenever a write
    merged with an existing partition file (the common case for the live
    recorder's repeated same-day flushes) -- exactly the kind of clustered
    activity a Hawkes fit is trying to characterize, so this was quietly
    suppressing the project's own signal of interest. See
    diagnostics/2026-08-11-tick-dedup-key-bug/findings.md. Full-row
    equality still correctly collapses genuine duplicates (the same trade
    re-fetched by an overlapping incremental window, or the same flush
    written twice).
    """
    _validate_schema(df, required, kind)
    data_dir = Path(data_dir)
    rows_written = 0

    df = df.copy()
    df["_date"] = df["timestamp"].dt.tz_convert("UTC").dt.date

    for (ticker, date), group in df.groupby(["ticker", "_date"]):
        group = group.drop(columns="_date").sort_values("timestamp")
        path = _partition_path(data_dir, kind, ticker, pd.Timestamp(date))
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, group], ignore_index=True)
        else:
            combined = group
        combined = combined.drop_duplicates().sort_values("timestamp")

        combined.to_parquet(path, index=False)
        rows_written += len(group)

    return rows_written


def write_bars(df: pd.DataFrame, data_dir: Path) -> int:
    return _write_partitioned(df, Path(data_dir), "bars", BAR_COLUMNS)


def write_ticks(df: pd.DataFrame, data_dir: Path) -> int:
    return _write_partitioned(df, Path(data_dir), "ticks", TICK_COLUMNS)


def _read_partitioned(
    data_dir: Path,
    kind: str,
    tickers: list[str] | None,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    data_dir = Path(data_dir) / kind
    if not data_dir.exists():
        return pd.DataFrame()

    ticker_dirs = (
        [data_dir / f"ticker={t}" for t in tickers]
        if tickers is not None
        else sorted(data_dir.glob("ticker=*"))
    )

    frames = []
    for ticker_dir in ticker_dirs:
        if not ticker_dir.exists():
            continue
        for date_dir in sorted(ticker_dir.glob("date=*")):
            date_str = date_dir.name.removeprefix("date=")
            date = pd.Timestamp(date_str, tz="UTC")
            if start is not None and date < start.normalize():
                continue
            if end is not None and date > end.normalize():
                continue
            path = date_dir / "data.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if start is not None:
        result = result[result["timestamp"] >= start]
    if end is not None:
        result = result[result["timestamp"] <= end]
    return result.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def read_bars(
    data_dir: Path,
    tickers: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return _read_partitioned(Path(data_dir), "bars", tickers, start, end)


def read_ticks(
    data_dir: Path,
    tickers: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return _read_partitioned(Path(data_dir), "ticks", tickers, start, end)


def latest_timestamp(data_dir: Path, kind: str, ticker: str) -> pd.Timestamp | None:
    """Last stored timestamp for a ticker, for incremental/resumable backfills."""
    df = _read_partitioned(Path(data_dir), kind, [ticker], None, None)
    if df.empty:
        return None
    return df["timestamp"].max()
