"""One-shot check that ALPACA_API_KEY / ALPACA_SECRET_KEY in .env are valid
and the account can actually pull data. Requires credentials, so it's
excluded from the default run (see the `live` marker in pyproject.toml) --
run explicitly with `pytest -m live tests/integration/test_alpaca_connectivity.py -s`.
"""

import pandas as pd
import pytest

from ingest import alpaca_client


@pytest.mark.live
def test_can_fetch_recent_minute_bars():
    client = alpaca_client.get_client()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=5)

    df = alpaca_client.fetch_minute_bars(client, ["AAPL"], start, end)

    assert not df.empty, "query succeeded but returned no bars -- check the symbol/date range"
    assert set(df["ticker"]) == {"AAPL"}
    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(df.columns)

    print(
        f"\nOK: fetched {len(df)} AAPL minute bars, "
        f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    )
