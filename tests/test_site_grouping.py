"""Site grouping preserves records while limiting search results by physical site."""

from copy import deepcopy
from datetime import date
from unittest.mock import patch

import pytest

from aquascope.archive.catalog import group_station_sites, search_stations
from aquascope.mcp_server import find_stations
from aquascope.schemas.station import Station


@pytest.mark.parametrize("fields, expected", [({}, "station-1"), ({"site_id": ""}, "station-1"),
                                             ({"site_id": "site-1"}, "site-1")])
def test_station_site_default_and_explicit_value(fields, expected):
    station = Station(source="test", station_id="station-1", latitude=0, longitude=0, **fields)
    assert station.site_id == expected
    assert station.model_dump()["site_id"] == expected


def _record(station_id, start=None, end=None, **extra):
    return {
        "source": "hubeau_hydrometrie", "station_id": station_id, "site_id": "A8910301",
        "name": "Example gauge", "latitude": 48.0, "longitude": 7.0,
        "variables": ["discharge"], "period_start": start, "period_end": end, **extra,
    }


def test_longest_span_represents_site_without_losing_or_mutating_records():
    rows = [_record("A891030101", "1980-01-01", "1985-01-01"),
            _record("A891030102", "2000-01-01", "2025-01-01")]
    original = deepcopy(rows)
    hits = group_station_sites(rows)
    assert len(hits) == 1
    assert hits[0]["station_id"] == "A891030102"
    assert hits[0]["record_count"] == 2
    assert {r["station_id"] for r in hits[0]["records"]} == {"A891030101", "A891030102"}
    assert rows == original


def test_open_ended_and_unknown_spans_and_ties():
    rows = [_record("unknown"), _record("invalid", "bad-date"),
            _record("reversed", "2020-01-01", "1900-01-01"),
            _record("closed", "1900-01-01", "1901-01-01"),
            _record("ongoing-b", date(1900, 1, 1)), _record("ongoing-a", "1900-01-01")]
    assert group_station_sites(rows)[0]["station_id"] == "ongoing-a"


def test_source_scoping_and_legacy_rows_remain_separate():
    rows = [_record("a"), _record("b", source="other"),
            _record("c", site_id=None), _record("d", site_id="")]
    del rows[-1]["site_id"]
    hits = group_station_sites(rows)
    assert len(hits) == 4
    assert [r["site_id"] for r in hits] == ["A8910301", "A8910301", "c", "d"]


def test_search_groups_before_limit_and_preserves_filters_and_raw_mode():
    rows = [_record("A891030101"), _record("A891030102"),
            _record("elsewhere", site_id="elsewhere", latitude=49.0)]
    hits = search_stations(rows, near=(48, 7), limit=2, group_sites=True)
    assert [r["site_id"] for r in hits] == ["A8910301", "elsewhere"]
    assert len(search_stations(rows)) == 3
    exact = search_stations(rows, query="A891030102", group_sites=True)
    assert exact[0]["station_id"] == "A891030102"
    assert exact[0]["record_count"] == 2
    assert {r["station_id"] for r in exact[0]["records"]} == {"A891030101", "A891030102"}
    assert search_stations(rows, variable="water_level", group_sites=True) == []
    assert search_stations(rows, limit=0, group_sites=True) == []


def test_find_stations_exposes_members_count_and_note():
    rows = [_record("A891030101", "1980-01-01", "1985-01-01"),
            _record("A891030102", "2000-01-01", "2025-01-01")]
    with patch("aquascope.archive.catalog.load_stations", return_value=rows):
        result = find_stations(limit=1)
    assert result["n_catalog"] == 2
    assert result["n_returned"] == 1
    site = result["stations"][0]
    assert site["station_id"] == "A891030102"
    assert site["site_note"] == "2 records at this site"
    assert site["record_count"] == 2
    assert {r["station_id"] for r in site["records"]} == {"A891030101", "A891030102"}


def test_representative_matches_variable_filter_but_all_members_remain_reachable():
    rows = [_record("level", "1900-01-01", variables=["water_level"]),
            _record("flow", "2000-01-01", variables=["discharge"])]
    hits = search_stations(rows, variable="discharge", group_sites=True)
    assert hits[0]["station_id"] == "flow"
    assert hits[0]["record_count"] == 2
    assert {r["station_id"] for r in hits[0]["records"]} == {"level", "flow"}
