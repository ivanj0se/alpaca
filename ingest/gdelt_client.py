"""GDELT ingest -- free, no auth required.

Uses the GKG (Global Knowledge Graph) feed rather than the bare Event
table: GDELT's Event table codes actor names via CAMEO (a political-science
actor taxonomy -- countries, generic roles like "BUSINESS" or "POLICE", not
literal company names), while GKG's `Organizations` field is NLP-extracted
plain-text organization names per article -- what "was there news about
company X around time T" actually needs.

Schema below verified against a live GDELT export
(2026-08-11 01:45 UTC GKG file) rather than assumed from memory: 27
tab-separated columns, no header row.

No tickers anywhere in GDELT -- ticker/company matching against
`Organizations` is name-substring heuristic matching, not authoritative.
See docs/data_sources.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"

GKG_COLUMNS = [
    "GKGRecordID",
    "Date",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]


def gkg_file_url(timestamp: pd.Timestamp) -> str:
    """GDELT publishes a GKG file every 15 minutes at deterministic,
    UTC-aligned filenames -- no need to scrape the multi-hundred-MB
    masterfilelist to discover what's available.
    """
    slot = timestamp.floor("15min")
    return f"{GDELT_BASE_URL}/{slot.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"


def _slots(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("15min"), end.ceil("15min"), freq="15min", tz="UTC")


def download_gdelt_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_dir: Path,
    session=None,
) -> list[Path]:
    """Download every 15-minute GKG file in [start, end] into out_dir. Skips
    files already on disk (idempotent, safe to re-run/resume) and tolerates
    individual missing slots (GDELT occasionally has gaps) without aborting
    the whole range.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests

    paths = []
    for slot in _slots(start, end):
        url = gkg_file_url(slot)
        dest = out_dir / Path(url).name
        if dest.exists():
            paths.append(dest)
            continue
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            continue  # missing slot -- not fatal, GDELT has occasional gaps
        dest.write_bytes(resp.content)
        paths.append(dest)
    return paths


def parse_gkg_file(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=GKG_COLUMNS,
                dtype=str,
                na_filter=False,
                on_bad_lines="skip",
                encoding="utf-8",
                encoding_errors="replace",
            )
    if df.empty:
        df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        return df
    df["timestamp"] = pd.to_datetime(df["Date"], format="%Y%m%d%H%M%S", utc=True)
    return df


def parse_gdelt_range(paths: list[Path]) -> pd.DataFrame:
    frames = [parse_gkg_file(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=[*GKG_COLUMNS, "timestamp"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def filter_events_for_universe(gkg_df: pd.DataFrame, universe: list[dict]) -> pd.DataFrame:
    """Heuristic name-substring match of each universe entry's `name`
    against the GKG `Organizations` field. Approximate by construction:
    false positives (unrelated org with a similar name) and false negatives
    (company mentioned under a name variant GDELT didn't extract) are both
    expected. This is exactly why the attribution layer (Rung 5) uses a
    permutation/null-control significance test rather than presenting raw
    match counts as precise. `universe` entries need `ticker` and `name`
    keys (see config/universe.yaml).
    """
    columns = ["timestamp", "ticker", "matched_org", "SourceCommonName", "DocumentIdentifier"]
    if gkg_df.empty:
        return pd.DataFrame(columns=columns)

    orgs_lower = gkg_df["Organizations"].str.lower()
    matches = []
    for entry in universe:
        name_lower = entry["name"].lower()
        hit_mask = orgs_lower.str.contains(name_lower, regex=False, na=False)
        if not hit_mask.any():
            continue
        hits = gkg_df.loc[hit_mask, ["timestamp", "Organizations", "SourceCommonName", "DocumentIdentifier"]].copy()
        hits["ticker"] = entry["ticker"]
        hits = hits.rename(columns={"Organizations": "matched_org"})
        matches.append(hits[columns])

    if not matches:
        return pd.DataFrame(columns=columns)
    return pd.concat(matches, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
