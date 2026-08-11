"""Offline tests -- no real network calls. See the live smoke test notes in
docs/data_sources.md; this module was manually verified against real GDELT
data during development (real Organizations-field false positive observed:
"Alphabet Bar Grill" matching "Alphabet" -- confirms the permutation-test
design in attribution/ is load-bearing, not decorative).
"""

import zipfile

import pandas as pd
import pytest

from ingest.gdelt_client import (
    GKG_COLUMNS,
    download_gdelt_range,
    filter_events_for_universe,
    gkg_file_url,
    parse_gdelt_range,
    parse_gkg_file,
)


def _gkg_row(date="20260811014500", organizations="", source="example.com"):
    row = {col: "" for col in GKG_COLUMNS}
    row["GKGRecordID"] = f"{date}-0"
    row["Date"] = date
    row["SourceCommonName"] = source
    row["DocumentIdentifier"] = f"https://{source}/article"
    row["Organizations"] = organizations
    return [row[c] for c in GKG_COLUMNS]


def _write_gkg_zip(tmp_path, filename_stem, rows):
    csv_path = tmp_path / f"{filename_stem}.gkg.csv"
    csv_path.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    zip_path = tmp_path / f"{filename_stem}.gkg.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(csv_path, arcname=csv_path.name)
    csv_path.unlink()
    return zip_path


class TestGkgFileUrl:
    def test_floors_to_15_minute_boundary(self):
        ts = pd.Timestamp("2026-08-11 01:52:37", tz="UTC")
        url = gkg_file_url(ts)
        assert url.endswith("20260811014500.gkg.csv.zip")

    def test_exact_boundary_unchanged(self):
        ts = pd.Timestamp("2026-08-11 01:45:00", tz="UTC")
        assert gkg_file_url(ts).endswith("20260811014500.gkg.csv.zip")


class _FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class _FakeSession:
    def __init__(self, status_by_suffix=None, default_status=200):
        self.status_by_suffix = status_by_suffix or {}
        self.default_status = default_status
        self.calls = []

    def get(self, url, timeout=30):
        self.calls.append(url)
        for suffix, status in self.status_by_suffix.items():
            if url.endswith(suffix):
                return _FakeResponse(status, b"fake-zip-bytes")
        return _FakeResponse(self.default_status, b"fake-zip-bytes")


class TestDownloadGdeltRange:
    def test_downloads_expected_number_of_15min_slots(self, tmp_path):
        session = _FakeSession()
        start = pd.Timestamp("2026-08-11 01:00", tz="UTC")
        end = pd.Timestamp("2026-08-11 01:30", tz="UTC")
        paths = download_gdelt_range(start, end, tmp_path, session=session)
        assert len(paths) == 3  # 01:00, 01:15, 01:30
        assert all(p.exists() for p in paths)

    def test_skips_files_already_on_disk(self, tmp_path):
        session = _FakeSession()
        start = end = pd.Timestamp("2026-08-11 01:00", tz="UTC")
        download_gdelt_range(start, end, tmp_path, session=session)
        assert len(session.calls) == 1
        download_gdelt_range(start, end, tmp_path, session=session)
        assert len(session.calls) == 1  # second call served from disk, no new request

    def test_tolerates_missing_slot_without_raising(self, tmp_path):
        session = _FakeSession(status_by_suffix={"20260811011500.gkg.csv.zip": 404})
        start = pd.Timestamp("2026-08-11 01:00", tz="UTC")
        end = pd.Timestamp("2026-08-11 01:30", tz="UTC")
        paths = download_gdelt_range(start, end, tmp_path, session=session)
        assert len(paths) == 2  # the 404'd slot is skipped, not fatal


class TestParseGkgFile:
    def test_parses_columns_and_timestamp(self, tmp_path):
        rows = [_gkg_row(date="20260811014500", organizations="microsoft corporation")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        assert len(df) == 1
        assert list(df.columns[: len(GKG_COLUMNS)]) == GKG_COLUMNS
        assert df.iloc[0]["timestamp"] == pd.Timestamp("2026-08-11 01:45:00", tz="UTC")
        assert df.iloc[0]["Organizations"] == "microsoft corporation"

    def test_empty_file_returns_empty_frame_with_timestamp_column(self, tmp_path):
        path = _write_gkg_zip(tmp_path, "20260811014500", [])
        df = parse_gkg_file(path)
        assert df.empty
        assert "timestamp" in df.columns


class TestParseGdeltRange:
    def test_concatenates_and_sorts_multiple_files(self, tmp_path):
        p1 = _write_gkg_zip(tmp_path, "20260811020000", [_gkg_row(date="20260811020000")])
        p2 = _write_gkg_zip(tmp_path, "20260811014500", [_gkg_row(date="20260811014500")])
        df = parse_gdelt_range([p1, p2])
        assert len(df) == 2
        assert df["timestamp"].is_monotonic_increasing

    def test_empty_path_list_returns_empty_frame(self):
        df = parse_gdelt_range([])
        assert df.empty
        assert "timestamp" in df.columns


class TestFilterEventsForUniverse:
    UNIVERSE = [{"ticker": "MSFT", "name": "Microsoft Corp"}, {"ticker": "AAPL", "name": "Apple Inc"}]

    def test_matches_case_insensitive_substring(self, tmp_path):
        rows = [_gkg_row(organizations="MICROSOFT CORPORATION;LEVI KORSINSKY", source="pr-inside.com")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, self.UNIVERSE)
        assert len(matches) == 1
        assert matches.iloc[0]["ticker"] == "MSFT"

    def test_false_positive_on_substring_match_is_expected_behavior(self, tmp_path):
        # Documents the known limitation: "Alphabet Bar Grill" contains
        # "alphabet" as a substring of a company name variant. This is
        # exactly why Rung 5 uses a permutation test rather than raw counts.
        universe = [{"ticker": "GOOGL", "name": "Alphabet"}]
        rows = [_gkg_row(organizations="GROUP ALPHABET BAR GRILL")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, universe)
        assert len(matches) == 1  # matched, even though it's not really about the company

    def test_no_matches_returns_empty_with_expected_columns(self, tmp_path):
        rows = [_gkg_row(organizations="national weather service")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, self.UNIVERSE)
        assert matches.empty
        assert list(matches.columns) == ["timestamp", "ticker", "matched_org", "SourceCommonName", "DocumentIdentifier"]

    def test_empty_gkg_df_returns_empty(self):
        matches = filter_events_for_universe(pd.DataFrame(), self.UNIVERSE)
        assert matches.empty


class TestFilterEventsForUniverseAliases:
    """Regression coverage for a real, confirmed data-quality gap: matching
    on a single formal/legal name misses real coverage entirely. Confirmed
    on real cached GDELT data -- "Procter & Gamble" (0 matches) vs.
    "Procter" (53), "Amazon.com" (0) vs. "Amazon" (483), "Linde plc" (0)
    vs. "Linde" (71). See diagnostics/2026-08-11-gdelt-alias-matching/.
    """

    def test_matches_on_any_alias(self, tmp_path):
        universe = [{"ticker": "PG", "name": "Procter & Gamble", "aliases": ["Procter & Gamble", "Procter"]}]
        rows = [_gkg_row(organizations="procter reports quarterly earnings")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, universe)
        assert len(matches) == 1
        assert matches.iloc[0]["ticker"] == "PG"

    def test_falls_back_to_name_when_no_aliases_given(self, tmp_path):
        # Backward compatible: an entry without "aliases" still matches on
        # "name" alone, same as before this feature existed.
        universe = [{"ticker": "MSFT", "name": "Microsoft Corp"}]
        rows = [_gkg_row(organizations="microsoft corporation")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, universe)
        assert len(matches) == 1

    def test_does_not_double_count_a_single_article_matching_multiple_aliases(self, tmp_path):
        universe = [{"ticker": "PG", "name": "Procter & Gamble", "aliases": ["Procter & Gamble", "Procter"]}]
        rows = [_gkg_row(organizations="procter & gamble announces procter brand refresh")]
        path = _write_gkg_zip(tmp_path, "20260811014500", rows)
        df = parse_gkg_file(path)
        matches = filter_events_for_universe(df, universe)
        assert len(matches) == 1  # one article, one row -- not one row per matching alias
