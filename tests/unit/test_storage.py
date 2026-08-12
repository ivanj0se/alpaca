import pandas as pd
import pytest

from ingest.storage import (
    latest_timestamp,
    read_bars,
    read_ticks,
    write_bars,
    write_ticks,
)


def _bars_df(ticker="AAPL", n=5, start="2026-01-02 09:30", freq_min=1):
    idx = pd.date_range(start, periods=n, freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "ticker": ticker,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )


def _ticks_df(ticker="AAPL", n=5, start="2026-01-02 09:30:00"):
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "ticker": ticker, "price": 100.0, "size": 100})


class TestBarsRoundTrip:
    def test_write_then_read_returns_same_rows(self, tmp_path):
        df = _bars_df()
        write_bars(df, tmp_path)
        out = read_bars(tmp_path)
        assert len(out) == len(df)
        assert set(out["ticker"]) == {"AAPL"}

    def test_write_is_idempotent_no_duplicates(self, tmp_path):
        df = _bars_df()
        write_bars(df, tmp_path)
        write_bars(df, tmp_path)  # re-run, e.g. backfill retry
        out = read_bars(tmp_path)
        assert len(out) == len(df)

    def test_write_merges_new_rows_into_existing_partition(self, tmp_path):
        df1 = _bars_df(n=3, start="2026-01-02 09:30")
        df2 = _bars_df(n=3, start="2026-01-02 09:33")  # overlapping day, new minutes
        write_bars(df1, tmp_path)
        write_bars(df2, tmp_path)
        out = read_bars(tmp_path)
        assert len(out) == 6
        assert out["timestamp"].is_monotonic_increasing

    def test_partitions_across_multiple_days(self, tmp_path):
        day1 = _bars_df(n=2, start="2026-01-02 09:30")
        day2 = _bars_df(n=2, start="2026-01-05 09:30")
        write_bars(pd.concat([day1, day2]), tmp_path)
        out = read_bars(tmp_path)
        assert len(out) == 4
        assert (tmp_path / "bars" / "ticker=AAPL" / "date=2026-01-02").exists()
        assert (tmp_path / "bars" / "ticker=AAPL" / "date=2026-01-05").exists()

    def test_filter_by_ticker(self, tmp_path):
        write_bars(_bars_df(ticker="AAPL"), tmp_path)
        write_bars(_bars_df(ticker="MSFT"), tmp_path)
        out = read_bars(tmp_path, tickers=["AAPL"])
        assert set(out["ticker"]) == {"AAPL"}

    def test_filter_by_date_range(self, tmp_path):
        day1 = _bars_df(n=2, start="2026-01-02 09:30")
        day2 = _bars_df(n=2, start="2026-01-10 09:30")
        write_bars(pd.concat([day1, day2]), tmp_path)
        out = read_bars(tmp_path, start=pd.Timestamp("2026-01-09", tz="UTC"))
        assert (out["timestamp"] >= pd.Timestamp("2026-01-09", tz="UTC")).all()
        assert len(out) == 2

    def test_read_empty_dir_returns_empty_frame(self, tmp_path):
        out = read_bars(tmp_path)
        assert out.empty

    def test_missing_required_column_raises(self, tmp_path):
        df = _bars_df().drop(columns="volume")
        with pytest.raises(ValueError, match="missing required columns"):
            write_bars(df, tmp_path)

    def test_tz_naive_timestamp_rejected(self, tmp_path):
        df = _bars_df()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            write_bars(df, tmp_path)


class TestTicksRoundTrip:
    def test_write_then_read(self, tmp_path):
        df = _ticks_df()
        write_ticks(df, tmp_path)
        out = read_ticks(tmp_path)
        assert len(out) == len(df)

    def test_write_is_idempotent_no_duplicates(self, tmp_path):
        df = _ticks_df()
        write_ticks(df, tmp_path)
        write_ticks(df, tmp_path)  # re-run, e.g. an overlapping incremental fetch
        out = read_ticks(tmp_path)
        assert len(out) == len(df)

    def test_distinct_trades_sharing_a_timestamp_are_both_kept(self, tmp_path):
        # Regression for a real bug: two DIFFERENT real trades can share a
        # timestamp down to the microsecond (a burst -- one order sweeping
        # several resting orders). De-duplicating on timestamp alone
        # silently discarded one of them whenever a write merged with an
        # existing partition file. See
        # diagnostics/2026-08-11-tick-dedup-key-bug/findings.md.
        ts = pd.Timestamp("2026-01-02 09:30:05.123456", tz="UTC")
        first = pd.DataFrame({"timestamp": [ts], "ticker": ["AAPL"], "price": [100.0], "size": [10]})
        second = pd.DataFrame({"timestamp": [ts], "ticker": ["AAPL"], "price": [100.05], "size": [25]})
        write_ticks(first, tmp_path)
        write_ticks(second, tmp_path)  # merges with the existing partition file
        out = read_ticks(tmp_path)
        assert len(out) == 2
        assert set(out["price"]) == {100.0, 100.05}

    def test_exact_duplicate_row_is_still_collapsed(self, tmp_path):
        # The legitimate case full-row dedup must still catch: the same
        # trade re-fetched (e.g. an overlapping incremental window).
        ts = pd.Timestamp("2026-01-02 09:30:05.123456", tz="UTC")
        row = pd.DataFrame({"timestamp": [ts], "ticker": ["AAPL"], "price": [100.0], "size": [10]})
        write_ticks(row, tmp_path)
        write_ticks(row.copy(), tmp_path)
        out = read_ticks(tmp_path)
        assert len(out) == 1


class TestLatestTimestamp:
    def test_none_when_no_data(self, tmp_path):
        assert latest_timestamp(tmp_path, "bars", "AAPL") is None

    def test_returns_max_timestamp(self, tmp_path):
        df = _bars_df(n=5, start="2026-01-02 09:30")
        write_bars(df, tmp_path)
        assert latest_timestamp(tmp_path, "bars", "AAPL") == df["timestamp"].max()
