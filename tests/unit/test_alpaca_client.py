"""Unit tests using mocked/fake objects -- no real network calls or
credentials. See tests/integration for mocked-HTTP-layer tests and the
`live` marker for tests that hit the real Alpaca API.
"""

import os

import pandas as pd
import pytest

from alpaca.common.exceptions import APIError
from ingest import alpaca_client


class _FakeBarSet:
    """Stands in for alpaca-py's BarSet -- bars_to_frame only touches `.df`,
    so a fake object with the same multiindex shape is sufficient and avoids
    depending on alpaca-py's internal constructors.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df


def _fake_bars_multiindex(symbols=("AAPL",), n=3):
    frames = []
    for sym in symbols:
        idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "trade_count": 10,
                "vwap": 100.2,
            },
            index=pd.MultiIndex.from_product([[sym], idx], names=["symbol", "timestamp"]),
        )
        frames.append(df)
    return pd.concat(frames)


class TestGetKeys:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        with pytest.raises(alpaca_client.MissingCredentialsError):
            alpaca_client.get_client()

    def test_present_credentials_construct_client(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
        client = alpaca_client.get_client()
        assert client is not None


class TestBarsToFrame:
    def test_converts_multiindex_to_flat_schema(self):
        fake = _FakeBarSet(_fake_bars_multiindex(symbols=("AAPL", "MSFT"), n=3))
        out = alpaca_client.bars_to_frame(fake)
        assert set(out.columns) >= {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
        assert set(out["ticker"]) == {"AAPL", "MSFT"}
        assert len(out) == 6
        assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
        assert out["timestamp"].dt.tz is not None

    def test_empty_barset_returns_empty_frame_with_expected_columns(self):
        fake = _FakeBarSet(pd.DataFrame())
        out = alpaca_client.bars_to_frame(fake)
        assert out.empty
        assert list(out.columns) == ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


class TestFetchMinuteBarsRetry:
    def test_retries_on_api_error_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        class _FlakyClient:
            def get_stock_bars(self, request):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise APIError("rate limited")
                return _FakeBarSet(_fake_bars_multiindex(n=2))

        monkeypatch.setattr(alpaca_client.time, "sleep", lambda _: None)
        out = alpaca_client.fetch_minute_bars(
            _FlakyClient(),
            ["AAPL"],
            pd.Timestamp("2026-01-02", tz="UTC"),
            pd.Timestamp("2026-01-03", tz="UTC"),
        )
        assert calls["n"] == 3
        assert len(out) == 2

    def test_raises_after_exhausting_retries(self, monkeypatch):
        class _AlwaysFailsClient:
            def get_stock_bars(self, request):
                raise APIError("persistent failure")

        monkeypatch.setattr(alpaca_client.time, "sleep", lambda _: None)
        with pytest.raises(APIError):
            alpaca_client.fetch_minute_bars(
                _AlwaysFailsClient(),
                ["AAPL"],
                pd.Timestamp("2026-01-02", tz="UTC"),
                pd.Timestamp("2026-01-03", tz="UTC"),
                max_retries=2,
            )
