# Data sources and caveats

## Prices (Alpaca, free tier)

- Historical + live minute bars and trade-level ticks via `alpaca-py`.
- **IEX-sourced, not the full consolidated SIP tape.** Undercounts total
  volume/trade count vs. the real tape. This directly affects the fidelity
  of the bar-threshold Hawkes proxy -- treat as a documented approximation,
  not ground truth.
- Adjustment convention is fixed to `all` (split+dividend adjusted) in
  `config/settings.yaml` and must stay consistent across every fetch, or
  unadjusted corporate actions will masquerade as anomalies.

## Ticks (Alpaca websocket, live only)

- `ingest/tick_recorder.py` records continuously starting immediately --
  there is no free historical tick-level source, so this is the only way to
  eventually get real event timestamps for the Hawkes rung.
- Liquid large-caps trade thousands of times/day, so a few weeks of
  recording is enough to redo the Hawkes fit on real ticks and compare
  against the bar-threshold proxy.

## News (GDELT, free)

- Uses the **GKG (Global Knowledge Graph)** feed, not the bare Event table:
  GDELT's Event table codes actor names via CAMEO (a political-science
  taxonomy -- countries, generic roles like "BUSINESS"/"POLICE" -- not
  literal company names), while GKG's `Organizations` field is NLP-extracted
  plain-text org names per article, which is what "was there news about
  company X around time T" actually needs. Downloaded directly
  (`data.gdeltproject.org/gdeltv2/*.gkg.csv.zip`, files every 15 min), not
  via the unmaintained `gdeltPyR` package. Schema verified against a live
  file during development, not assumed from memory.
- **No tickers in GDELT.** Ticker/company matching (`gdelt_client.py`,
  `filter_events_for_universe`) is name/org-string heuristic matching --
  expect false positives and false negatives. Confirmed during development:
  a live query for "Alphabet" matched "Group Alphabet Bar Grill," an
  unrelated restaurant. This is exactly why the attribution layer (Rung 5)
  uses a permutation/null-control significance test rather than reporting
  raw correlation rates as precise.

## Upgrade paths (not needed to start)

- Polygon.io (rebranded "Massive," Oct 2025): flat-rate, deep historical
  coverage, full consolidated tape.
- Databento: tick + L2 order book, metered pricing -- the realistic option
  if the project ever needs true limit-order-book depth rather than trades.
