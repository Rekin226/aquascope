"""Tests for the German PEGELONLINE collector."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from aquascope.collectors.pegelonline import PegelonlineCollector
from aquascope.schemas.water_data import (
    DataSource,
    StreamflowReading,
    WaterLevelReading,
)

STATION_UUID = "593647aa-9fea-43ec-a7d6-6476a76ae868"

STATION = {
    "uuid": STATION_UUID,
    "number": "2710080",
    "shortname": "BONN",
    "longname": "BONN",
    "longitude": 7.108045,
    "latitude": 50.736398,
    "water": {"shortname": "RHEIN", "longname": "RHEIN"},
    "timeseries": [
        {"shortname": "W", "longname": "WASSERSTAND ROHDATEN", "unit": "cm"},
        {"shortname": "Q", "longname": "ABFLUSS", "unit": "m³/s"},
    ],
}


def test_fetches_station_metadata_and_both_timeseries():
    collector = PegelonlineCollector()
    get_json = Mock(
        side_effect=[
            STATION,
            [{"timestamp": "2026-07-21T08:30:00+02:00", "value": 213.0}],
            [{"timestamp": "2026-07-21T08:30:00+02:00", "value": 892.0}],
        ]
    )
    collector.client.get_json = get_json

    raw = collector.fetch_raw(station_id=STATION_UUID, days=15)

    assert [item["shortname"] for item in raw["series"]] == ["W", "Q"]
    assert get_json.call_args_list[0].args[0] == f"/stations/{STATION_UUID}.json"
    assert get_json.call_args_list[0].kwargs["params"] == {"includeTimeseries": "true"}
    assert get_json.call_args_list[1].args[0].endswith("/W/measurements.json")
    assert get_json.call_args_list[1].kwargs["params"] == {"start": "P15D"}
    assert get_json.call_args_list[2].args[0].endswith("/Q/measurements.json")


def test_fetches_only_requested_timeseries():
    collector = PegelonlineCollector()
    get_json = Mock(
        side_effect=[
            STATION,
            [{"timestamp": "2026-07-21T08:30:00+02:00", "value": 892.0}],
        ]
    )
    collector.client.get_json = get_json

    raw = collector.fetch_raw(station_id=STATION_UUID, timeseries="q")

    assert [item["shortname"] for item in raw["series"]] == ["Q"]
    assert get_json.call_args_list[1].args[0].endswith("/Q/measurements.json")


@pytest.mark.parametrize("days", [0, 32])
def test_rejects_history_outside_api_window(days):
    collector = PegelonlineCollector()
    with pytest.raises(ValueError, match="between 1 and 31"):
        collector.fetch_raw(station_id=STATION_UUID, days=days)


def test_normalises_level_and_discharge_records():
    collector = PegelonlineCollector()
    raw = {
        "station": STATION,
        "series": [
            {
                "shortname": "W",
                "unit": "cm",
                "measurements": [{"timestamp": "2026-07-21T08:30:00+02:00", "value": 213.0}],
            },
            {
                "shortname": "Q",
                "unit": "m³/s",
                "measurements": [{"timestamp": "2026-07-21T08:30:00+02:00", "value": 892.0}],
            },
        ],
    }

    readings = collector.normalise(raw)

    assert len(readings) == 2
    level, discharge = readings
    assert isinstance(level, WaterLevelReading)
    assert level.source == DataSource.PEGELONLINE
    assert level.station_id == STATION_UUID
    assert level.station_name == "BONN"
    assert level.location is not None
    assert level.location.latitude == 50.736398
    assert level.water_level == 213.0
    assert level.unit == "cm"
    assert level.reading_datetime.utcoffset().total_seconds() == 7200

    assert isinstance(discharge, StreamflowReading)
    assert discharge.source == DataSource.PEGELONLINE
    assert discharge.discharge_cms == 892.0
    assert discharge.source_type == "in_situ"
    assert discharge.unit == "m3/s"


def test_skips_invalid_measurements_and_warns(caplog):
    collector = PegelonlineCollector()
    raw = {
        "station": STATION,
        "series": [
            {
                "shortname": "W",
                "unit": "cm",
                "measurements": [
                    {"timestamp": "2026-07-21T08:30:00+02:00", "value": 213.0},
                    {"timestamp": "2026-07-21T08:45:00+02:00", "value": None},
                ],
            }
        ],
    }

    with caplog.at_level("WARNING"):
        readings = collector.normalise(raw)

    assert len(readings) == 1
    assert any("skipped 1/2" in record.message for record in caplog.records)


def test_skips_timeseries_not_published_by_station():
    collector = PegelonlineCollector()
    station = {**STATION, "timeseries": [STATION["timeseries"][0]]}
    get_json = Mock(side_effect=[station, []])
    collector.client.get_json = get_json

    raw = collector.fetch_raw(station_id=STATION_UUID)

    assert [item["shortname"] for item in raw["series"]] == ["W"]
    assert get_json.call_count == 2
