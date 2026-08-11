"""One-off connectivity check: confirms ALPACA_API_KEY / ALPACA_SECRET_KEY in
.env are valid by fetching a handful of real historical bars. Prints only
row counts and sample data, never the credentials themselves.
"""

from __future__ import annotations

import sys

import pandas as pd
from alpaca.common.exceptions import APIError

from ingest.alpaca_client import MissingCredentialsError, fetch_minute_bars, get_client


def main() -> None:
    try:
        client = get_client()
    except MissingCredentialsError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=5)

    try:
        df = fetch_minute_bars(client, ["AAPL"], start, end, max_retries=2)
    except APIError as exc:
        print(f"FAILED: Alpaca API rejected the request -- {exc}")
        sys.exit(1)

    if df.empty:
        print("FAILED: request succeeded but returned zero rows (unexpected for AAPL over 5 days)")
        sys.exit(1)

    print("SUCCESS: credentials are valid and historical data access works.")
    print(f"Fetched {len(df)} AAPL minute bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
