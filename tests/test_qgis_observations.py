"""Tests for AquaScope QGIS daily observation reader."""

import gzip
from datetime import date
from urllib.error import HTTPError

import pytest

from integrations.qgis.aquascope_gauges.observations import (
    ObservationError,
    ObservationNotFoundError,
    available_station_variables,
    available_variables,
    fetch_observations,
    load_observation_manifest,
    observation_url,
    parse_observations,
)


def test_available_variables_for_usgs():
    assert available_variables("usgs") == ("discharge", "water_level")


def test_available_variables_for_unknown_source():
    assert available_variables("unknown") == ()


def test_observation_url_builds_expected_path():
    url = observation_url("usgs", "12345", "discharge")
    assert url.endswith("obs/discharge/usgs/12345.csv.gz")


def test_observation_url_encodes_station_id():
    url = observation_url("usgs", "station 123", "discharge")
    assert "station%20123.csv.gz" in url


def test_observation_url_rejects_unknown_source():
    with pytest.raises(ValueError):
        observation_url("unknown", "12345", "discharge")


def test_observation_url_rejects_unsupported_variable():
    with pytest.raises(ValueError):
        observation_url("usgs", "12345", "precipitation")


def test_observation_url_sanitizes_forward_slash():
    url = observation_url("usgs", "station/123", "discharge")
    assert url.endswith("obs/discharge/usgs/station_123.csv.gz")


def test_observation_url_rejects_backslash():
    with pytest.raises(ValueError):
        observation_url("usgs", "station\\\\123", "discharge")


def test_observation_url_rejects_traversal_id():
    with pytest.raises(ValueError):
        observation_url("usgs", "..", "discharge")


def test_observation_url_rejects_http_base_url():
    with pytest.raises(ValueError):
        observation_url(
            "usgs", "12345", "discharge", base_url="http://example.com/"
        )


def test_parse_observations_reads_valid_csv_gzip():
    csv_data = b"date,value\n2026-01-01,12.5\n2026-01-02,14.2\n"
    records = parse_observations(gzip.compress(csv_data))

    assert records == [
        {"date": date(2026, 1, 1), "value": 12.5},
        {"date": date(2026, 1, 2), "value": 14.2},
    ]


def test_parse_observations_skips_invalid_rows():
    csv_data = (
        b"date,value\n"
        b"2026-01-01,12.5\n"
        b"invalid-date,10\n"
        b"2026-01-03,not-a-number\n"
        b"2026-01-04,nan\n"
        b"2026-01-05,inf\n"
    )
    records = parse_observations(gzip.compress(csv_data))

    assert records == [
        {"date": date(2026, 1, 1), "value": 12.5},
    ]


def test_parse_observations_rejects_invalid_gzip():
    with pytest.raises(ObservationError):
        parse_observations(b"not a gzip archive")


def test_parse_observations_rejects_missing_columns():
    compressed = gzip.compress(b"timestamp,measurement\n2026-01-01,10\n")
    with pytest.raises(ObservationError):
        parse_observations(compressed)


def test_parse_observations_handles_empty_csv():
    compressed = gzip.compress(b"")
    with pytest.raises(ObservationError):
        parse_observations(compressed)


def test_parse_observations_returns_empty_list_for_header_only():
    compressed = gzip.compress(b"date,value\n")
    assert parse_observations(compressed) == []


def test_fetch_observations_uses_parser(monkeypatch):
    csv_data = b"date,value\n2026-01-01,20.5\n"
    compressed = gzip.compress(csv_data)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return compressed

    def fake_urlopen(url, timeout):
        assert url.endswith("obs/discharge/usgs/12345.csv.gz")
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(
        "integrations.qgis.aquascope_gauges.observations.urlopen",
        fake_urlopen,
    )

    records = fetch_observations("usgs", "12345", "discharge")

    assert records == [
        {"date": date(2026, 1, 1), "value": 20.5},
    ]


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (404, ObservationNotFoundError),
        (503, ObservationError),
    ],
)
def test_fetch_observations_handles_http_errors(
    monkeypatch, status_code, expected_error
):
    def fake_urlopen(url, timeout):
        raise HTTPError(
            url,
            status_code,
            "Archive response",
            {},
            None,
        )

    monkeypatch.setattr(
        "integrations.qgis.aquascope_gauges.observations.urlopen",
        fake_urlopen,
    )

    with pytest.raises(expected_error):
        fetch_observations("usgs", "12345", "discharge")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://",
        "https:///missing-host",
        "https://user:password@example.com/",
        "https://example.com/?token=secret",
        "https://example.com/#fragment",
        "https://example.com:invalid/",
    ],
)
def test_observation_url_rejects_invalid_base_urls(base_url):
    with pytest.raises(ValueError):
        observation_url("usgs", "12345", "discharge", base_url=base_url)


def test_observation_url_accepts_https_base_url_with_path():
    url = observation_url(
        "usgs",
        "12345",
        "discharge",
        base_url="https://example.com/archive/",
    )
    assert url == (
        "https://example.com/archive/obs/discharge/usgs/12345.csv.gz"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("usgs", ("discharge", "water_level")),
        (
            "uk_ea",
            ("discharge", "water_level", "precipitation", "groundwater_level"),
        ),
        ("hubeau_hydrometrie", ("discharge",)),
        ("taiwan_cwa", ("precipitation",)),
        ("greece_openhi", ("discharge", "water_level", "precipitation")),
        ("poland_imgw", ("discharge", "water_level")),
    ],
)
def test_available_variables_for_supported_sources(source, expected):
    assert available_variables(source) == expected


@pytest.mark.parametrize("source", ["pegelonline", "ireland_opw", "unknown", ""])
def test_available_variables_for_unsupported_sources(source):
    assert available_variables(source) == ()


def test_available_station_variables_returns_station_metadata():
    manifest = {
        "sources": {
            "usgs/discharge": {
                "attribution": "USGS",
                "license": "US-PD",
                "stations": {
                    "12345": {
                        "file": "obs/discharge/usgs/12345.csv.gz",
                        "first": "2020-01-01",
                        "last": "2026-01-01",
                        "n": 100,
                        "unit": "m3/s",
                    }
                },
            },
            "usgs/water_level": {
                "attribution": "USGS",
                "license": "US-PD",
                "stations": {
                    "12345": {
                        "empty": True,
                        "n": 0,
                    }
                },
            },
        }
    }

    result = available_station_variables("usgs", "12345", manifest)

    assert result == {
        "discharge": {
            "first": "2020-01-01",
            "last": "2026-01-01",
            "count": 100,
            "unit": "m3/s",
            "attribution": "USGS",
            "license": "US-PD",
        }
    }


def test_available_station_variables_excludes_empty_station_records():
    manifest = {
        "sources": {
            "taiwan_cwa/precipitation": {
                "stations": {
                    "12J990": {
                        "empty": True,
                        "n": 0,
                    }
                }
            }
        }
    }

    assert available_station_variables(
        "taiwan_cwa", "12J990", manifest
    ) == {}


def test_available_station_variables_handles_missing_station():
    manifest = {
        "sources": {
            "usgs/discharge": {
                "stations": {
                    "12345": {"file": "archive.csv.gz", "n": 10}
                }
            }
        }
    }

    assert available_station_variables("usgs", "99999", manifest) == {}
    assert available_station_variables("", "12345", manifest) == {}


def test_load_observation_manifest_downloads_once(monkeypatch):
    import json

    from integrations.qgis.aquascope_gauges import observations

    manifest = {"sources": {"usgs/discharge": {"stations": {}}}}
    payload = json.dumps(manifest).encode("utf-8")
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return payload

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeResponse()

    monkeypatch.setattr(observations, "urlopen", fake_urlopen)
    load_observation_manifest.cache_clear()

    try:
        first = load_observation_manifest()
        second = load_observation_manifest()

        assert first == manifest
        assert second is first
        assert len(calls) == 1
        assert calls[0][0].endswith("/obs/manifest.json")
    finally:
        load_observation_manifest.cache_clear()
