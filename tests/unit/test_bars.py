import pandas as pd

from features.bars import align_universe_bars, trading_session_index


class TestTradingSessionIndex:
    def test_single_trading_day_has_390_minutes(self):
        # 2026-01-02 is a Friday, a normal NYSE trading day.
        idx = trading_session_index(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02"))
        assert len(idx) == 390

    def test_excludes_weekend(self):
        # 2026-01-02 (Fri) through 2026-01-04 (Sun): only Friday trades.
        idx = trading_session_index(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-04"))
        assert len(idx) == 390
        assert all(ts.date() == pd.Timestamp("2026-01-02").date() for ts in idx)

    def test_is_tz_aware_utc(self):
        idx = trading_session_index(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02"))
        assert idx.tz is not None


class TestAlignUniverseBars:
    # 2026-01-02 (a Friday, normal NYSE session) opens 14:30 UTC / 9:30 ET
    # (EST, UTC-5, in January); first minute bar is timestamped 14:31 UTC.
    # Naive "09:30" strings would localize straight to 09:30 UTC = 4:30 AM
    # ET, outside real trading hours -- confirmed the hard way via a
    # KeyError when the synthetic bars fell entirely outside
    # trading_session_index's grid.
    SESSION_START = "2026-01-02 14:31"

    def _bars(self, ticker, closes, start=SESSION_START):
        idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
        return pd.DataFrame({"timestamp": idx, "close": closes})

    def test_empty_input_returns_empty_frame(self):
        assert align_universe_bars({}).empty

    def test_produces_wide_frame_with_one_column_per_ticker(self):
        bars_by_ticker = {
            "AAPL": self._bars("AAPL", [100.0, 101.0, 102.0]),
            "MSFT": self._bars("MSFT", [200.0, 201.0, 202.0]),
        }
        wide = align_universe_bars(bars_by_ticker)
        assert set(wide.columns) == {"AAPL", "MSFT"}

    def test_missing_minutes_are_nan_not_forward_filled(self):
        # MSFT is missing a minute that AAPL has data for.
        aapl = self._bars("AAPL", [100.0, 101.0, 102.0])
        msft = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-02 14:31", "2026-01-02 14:33"], utc=True
                ),  # gap at 14:32
                "close": [200.0, 202.0],
            }
        )
        wide = align_universe_bars({"AAPL": aapl, "MSFT": msft})
        missing_ts = pd.Timestamp("2026-01-02 14:32", tz="UTC")
        assert pd.isna(wide.loc[missing_ts, "MSFT"])
        assert wide.loc[missing_ts, "AAPL"] == 101.0
