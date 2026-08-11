import pandas as pd
import yaml

from ingest.historical_bars import backfill_universe, load_universe, update_incremental
from ingest.storage import read_bars, write_bars


def _universe_yaml(tmp_path, tickers=("AAPL", "MSFT")):
    cfg = {
        "benchmark_instrument": {"ticker": "SPY", "name": "SPDR S&P 500"},
        "universe": [{"ticker": t, "name": t, "sector": "test"} for t in tickers],
    }
    path = tmp_path / "universe.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def _bars_df(ticker, start, n, freq_min=1):
    idx = pd.date_range(start, periods=n, freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": idx, "ticker": ticker, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
    )


class _FakeClient:
    """Records every call and returns a deterministic bars frame per call,
    so tests can assert both the data written and the chunking behavior.
    """

    def __init__(self, bars_by_symbol=None):
        self.calls = []
        self.bars_by_symbol = bars_by_symbol or {}

    def get_stock_bars(self, request):
        from ingest.alpaca_client import bars_to_frame

        class _FakeBarSet:
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
        df = self.bars_by_symbol.get(symbol, _bars_df(symbol, "2026-01-02 09:30", 3))
        multi = df.set_index(["ticker", "timestamp"])
        multi.index.set_names(["symbol", "timestamp"], inplace=True)
        return _FakeBarSet(multi)


class TestLoadUniverse:
    def test_includes_benchmark_and_universe_tickers(self, tmp_path):
        path = _universe_yaml(tmp_path, tickers=("AAPL", "MSFT"))
        tickers = load_universe(path)
        assert tickers == ["AAPL", "MSFT", "SPY"]


class TestBackfillUniverse:
    def test_writes_bars_for_every_ticker(self, tmp_path):
        universe_path = _universe_yaml(tmp_path, tickers=("AAPL",))
        data_dir = tmp_path / "data"
        client = _FakeClient()

        result = backfill_universe(
            universe_path,
            pd.Timestamp("2026-01-02", tz="UTC"),
            pd.Timestamp("2026-01-03", tz="UTC"),
            data_dir,
            client=client,
        )
        assert result["AAPL"] == 3
        assert result["SPY"] == 3
        out = read_bars(data_dir, tickers=["AAPL"])
        assert len(out) == 3

    def test_chunks_long_ranges(self, tmp_path):
        universe_path = _universe_yaml(tmp_path, tickers=("AAPL",))
        data_dir = tmp_path / "data"
        client = _FakeClient()

        backfill_universe(
            universe_path,
            pd.Timestamp("2026-01-01", tz="UTC"),
            pd.Timestamp("2026-03-01", tz="UTC"),  # ~60 days, chunk_days=30 -> 2 chunks
            data_dir,
            chunk_days=30,
            client=client,
        )
        aapl_calls = [c for c in client.calls if c["symbols"] == ["AAPL"]]
        assert len(aapl_calls) == 2


class TestUpdateIncremental:
    def test_noop_when_no_existing_data(self, tmp_path):
        universe_path = _universe_yaml(tmp_path, tickers=("AAPL",))
        data_dir = tmp_path / "data"
        client = _FakeClient()
        result = update_incremental(universe_path, data_dir, client=client)
        assert result == {"AAPL": 0, "SPY": 0}
        assert client.calls == []

    def test_fetches_only_bars_after_last_stored_timestamp(self, tmp_path):
        universe_path = _universe_yaml(tmp_path, tickers=("AAPL",))
        data_dir = tmp_path / "data"
        write_bars(_bars_df("AAPL", "2026-01-02 09:30", 3), data_dir)
        write_bars(_bars_df("SPY", "2026-01-02 09:30", 3), data_dir)

        new_bars = {
            "AAPL": _bars_df("AAPL", "2026-01-02 09:33", 2),
            "SPY": _bars_df("SPY", "2026-01-02 09:33", 2),
        }
        client = _FakeClient(bars_by_symbol=new_bars)

        result = update_incremental(
            universe_path, data_dir, end=pd.Timestamp("2026-01-02 10:00", tz="UTC"), client=client
        )
        assert result["AAPL"] == 2
        call = next(c for c in client.calls if c["symbols"] == ["AAPL"])
        assert call["start"] == pd.Timestamp("2026-01-02 09:33", tz="UTC")

        out = read_bars(data_dir, tickers=["AAPL"])
        assert len(out) == 5  # 3 original + 2 new
