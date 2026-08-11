"""Trading-session-aware minute grid and cross-ticker alignment. Verified
against pandas_market_calendars 5.4.0's actual API (schedule() + date_range())
rather than assumed -- naive fixed-frequency resampling would silently
include non-trading minutes (a classic corruptor of cross-sectional
matrices, see docs/architecture.md's risk list).
"""

from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal


def trading_session_index(start: pd.Timestamp, end: pd.Timestamp, calendar: str = "XNYS") -> pd.DatetimeIndex:
    """Canonical UTC minute-level index covering only real trading minutes
    (holidays, half-days, and non-trading hours excluded) between start and
    end, inclusive.
    """
    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start.date(), end_date=end.date())
    if schedule.empty:
        return pd.DatetimeIndex([], tz="UTC")
    return mcal.date_range(schedule, frequency="1min")


def align_universe_bars(bars_by_ticker: dict[str, pd.DataFrame], calendar: str = "XNYS") -> pd.DataFrame:
    """bars_by_ticker: {ticker: DataFrame with a 'timestamp' and 'close'
    column}. Reindexes every ticker onto the shared canonical trading-minute
    grid (spanning the union of all tickers' data) and returns a wide
    DataFrame (index=timestamp, columns=ticker) of close prices.

    Explicit, not silent: minutes with no trade for a ticker are left as
    NaN rather than forward-filled -- forward-filling would dampen real
    gaps/halts and defeat the point of reconstruction-residual anomaly
    detection (a risk flagged in docs/architecture.md). Callers decide how
    to handle NaNs (drop, or an explicit fill policy) for their specific use.
    """
    if not bars_by_ticker:
        return pd.DataFrame()

    all_starts = [df["timestamp"].min() for df in bars_by_ticker.values() if not df.empty]
    all_ends = [df["timestamp"].max() for df in bars_by_ticker.values() if not df.empty]
    if not all_starts:
        return pd.DataFrame()

    grid = trading_session_index(min(all_starts), max(all_ends), calendar=calendar)

    columns = {}
    for ticker, df in bars_by_ticker.items():
        series = df.set_index("timestamp")["close"]
        columns[ticker] = series.reindex(grid)

    return pd.DataFrame(columns, index=grid)
