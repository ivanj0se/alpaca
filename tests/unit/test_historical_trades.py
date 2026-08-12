import pandas as pd

from ingest.historical_trades import backfill_trades, fetch_historical_trades, trades_to_frame, update_incremental
from ingest.storage import read_ticks, write_ticks


def _trades_df(ticker, start, n, conditions=None):
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ticker,
            "timestamp": idx,
            "exchange": "V",
            "price": [100.0 + i for i in range(n)],
            "size": [10] * n,
            "id": list(range(n)),
            "conditions": [conditions or [] for _ in range(n)],
            "tape": "B",
        }
    ).set_index(["symbol", "timestamp"])


class _FakeClient:
    def __init__(self, trades_by_symbol=None):
        self.calls = []
        self.trades_by_symbol = trades_by_symbol or {}

    def get_stock_trades(self, request):
        class _FakeTradeSet:
            def __init__(self, df):
                self.df = df

        self.calls.append(
            {
                "symbols": request.symbol_or_symbols,
                "start": pd.Timestamp(request.start, tz="UTC")
                if pd.Timestamp(request.start).tz is None
                else pd.Timestamp(request.start),
                "end": pd.Timestamp(request.end),
            }
        )
        symbol = request.symbol_or_symbols[0]
        df = self.trades_by_symbol.get(symbol, _trades_df(symbol, "2026-01-02 09:30", 3))
        return _FakeTradeSet(df)


class TestTradesToFrame:
    def test_converts_expected_columns(self):
        df = _trades_df("SPY", "2026-01-02 09:30", 2, conditions=["@", "F"])
        out = trades_to_frame(df)
        assert list(out.columns) == ["timestamp", "ticker", "price", "size", "exchange", "conditions"]
        assert out["ticker"].iloc[0] == "SPY"
        assert out["conditions"].iloc[0] == "@,F"

    def test_empty_conditions_becomes_empty_string(self):
        df = _trades_df("SPY", "2026-01-02 09:30", 1, conditions=[])
        out = trades_to_frame(df)
        assert out["conditions"].iloc[0] == ""

    def test_empty_input_returns_empty_with_columns(self):
        out = trades_to_frame(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == ["timestamp", "ticker", "price", "size", "exchange", "conditions"]

    def test_matches_live_recorder_schema(self):
        # Regression: a backfilled and a live-recorded row for the same
        # trade must be schema-identical (dtypes + conditions format), or
        # storage.write_ticks's de-dup/concat could silently misbehave.
        # ingest/tick_recorder.py's on_trade produces this exact shape.
        df = _trades_df("SPY", "2026-01-02 09:30", 1, conditions=["F"])
        out = trades_to_frame(df)
        live_record = pd.DataFrame(
            [{"timestamp": out["timestamp"].iloc[0], "ticker": "SPY", "price": 100.0, "size": 10, "exchange": "V", "conditions": "F"}]
        )
        assert out["conditions"].dtype == live_record["conditions"].dtype
        assert isinstance(out["size"].iloc[0].item(), int)


class TestFetchHistoricalTrades:
    def test_fetches_and_converts(self):
        client = _FakeClient()
        df = fetch_historical_trades(
            client, ["SPY"], pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC")
        )
        assert len(df) == 3
        assert list(df["ticker"].unique()) == ["SPY"]


class TestBackfillTrades:
    def test_writes_ticks_for_every_ticker(self, tmp_path):
        data_dir = tmp_path / "data"
        client = _FakeClient()
        result = backfill_trades(
            ["SPY"], pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC"), data_dir, client=client
        )
        assert result["SPY"] == 3
        out = read_ticks(data_dir, tickers=["SPY"])
        assert len(out) == 3

    def test_chunks_by_day_by_default(self, tmp_path):
        data_dir = tmp_path / "data"
        client = _FakeClient()
        backfill_trades(
            ["SPY"], pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-04", tz="UTC"), data_dir, client=client
        )
        spy_calls = [c for c in client.calls if c["symbols"] == ["SPY"]]
        assert len(spy_calls) == 3  # 3 calendar days, chunk_days=1


class TestUpdateIncremental:
    def test_noop_when_no_existing_data(self, tmp_path):
        data_dir = tmp_path / "data"
        client = _FakeClient()
        result = update_incremental(["SPY"], data_dir, client=client)
        assert result == {"SPY": 0}
        assert client.calls == []

    def test_fetches_only_ticks_after_last_stored_timestamp(self, tmp_path):
        data_dir = tmp_path / "data"
        write_ticks(trades_to_frame(_trades_df("SPY", "2026-01-02 09:30", 3)), data_dir)

        new_trades = {"SPY": _trades_df("SPY", "2026-01-02 09:33", 2)}
        client = _FakeClient(trades_by_symbol=new_trades)

        result = update_incremental(
            ["SPY"], data_dir, end=pd.Timestamp("2026-01-02 10:00", tz="UTC"), client=client
        )
        assert result["SPY"] == 2
        out = read_ticks(data_dir, tickers=["SPY"])
        assert len(out) == 5  # 3 original + 2 new
