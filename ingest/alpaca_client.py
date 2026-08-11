"""Thin wrappers around alpaca-py's REST and websocket clients. Credentials
are read from environment variables via python-dotenv -- get free keys at
https://alpaca.markets/ (a data/paper account; this project never places an
order, so no funding is needed).

Verified against alpaca-py 0.43.5's actual API surface (field names,
signatures) rather than assumed from memory.
"""

from __future__ import annotations

import os
import time

import pandas as pd
from dotenv import load_dotenv

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()


class MissingCredentialsError(RuntimeError):
    pass


def _get_keys() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise MissingCredentialsError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Copy .env.example to "
            ".env and fill in free keys from https://alpaca.markets/ (data/paper account)."
        )
    return api_key, secret_key


def get_client() -> StockHistoricalDataClient:
    api_key, secret_key = _get_keys()
    return StockHistoricalDataClient(api_key, secret_key)


def get_stream_client(feed: DataFeed = DataFeed.IEX) -> StockDataStream:
    api_key, secret_key = _get_keys()
    return StockDataStream(api_key, secret_key, feed=feed)


def bars_to_frame(bar_set: BarSet) -> pd.DataFrame:
    """Convert alpaca-py's BarSet into the storage schema
    (timestamp, ticker, open, high, low, close, volume, trade_count, vwap),
    matching ingest/storage.py's BAR_COLUMNS.
    """
    df = bar_set.df
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"])
    df = df.reset_index().rename(columns={"symbol": "ticker"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_minute_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjustment: Adjustment = Adjustment.ALL,
    feed: DataFeed = DataFeed.IEX,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch minute bars for `symbols` in [start, end]; alpaca-py paginates
    internally. Retries with exponential backoff on API errors -- the free
    tier rate-limits aggressively during a large historical backfill, and
    this is the one place in the pipeline where that's expected, not
    exceptional.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment=adjustment,
        feed=feed,
    )
    last_error: APIError | None = None
    for attempt in range(max_retries):
        try:
            return bars_to_frame(client.get_stock_bars(request))
        except APIError as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise last_error
