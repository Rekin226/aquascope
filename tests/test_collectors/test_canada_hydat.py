"""Offline tests for the Canada HYDAT collector."""

from unittest.mock import Mock

import pytest

from aquascope.collectors.base import CollectorError
from aquascope.collectors.canada_hydat import CanadaHydatCollector
from aquascope.schemas.water_data import Quality, StreamflowReading

_DEFAULT_GEOMETRY = object()


def feature(*, discharge=4.98, symbol=None, geometry=_DEFAULT_GEOMETRY, date="2000-01-01"):
    if geometry is _DEFAULT_GEOMETRY:
        geometry = {"type": "Point", "coordinates": [-123.2, 50.8]}
    return {
        "geometry": geometry,
        "properties": {
            "STATION_NUMBER": "05BB001",
            "STATION_NAME": "Example River",
            "DATE": date,
            "DISCHARGE": discharge,
            "DISCHARGE_SYMBOL_EN": symbol,
        },
    }


def test_normalise_maps_discharge_and_coordinates():
    [reading] = CanadaHydatCollector(client=Mock()).normalise([feature()])

    assert isinstance(reading, StreamflowReading)
    assert reading.station_id == "05BB001"
    assert reading.station_name == "Example River"
    assert reading.reading_datetime.isoformat().startswith("2000-01-01")
    assert reading.discharge_cms == 4.98
    assert (reading.location.longitude, reading.location.latitude) == (-123.2, 50.8)
    assert reading.quality is Quality.APPROVED


def test_normalise_skips_missing_discharge_but_keeps_zero():
    rows = [feature(discharge=None), feature(discharge=0, date="2000-01-02")]

    [reading] = CanadaHydatCollector(client=Mock()).normalise(rows)

    assert reading.discharge_cms == 0


def test_normalise_maps_known_and_unknown_symbols():
    [ice, other] = CanadaHydatCollector(client=Mock()).normalise(
        [feature(symbol="Ice Conditions"), feature(symbol="Unmapped flag", date="2000-01-02")]
    )

    assert ice.quality is Quality.ESTIMATED
    assert ice.quality_raw == "Ice Conditions"
    assert other.quality is Quality.UNKNOWN
    assert other.quality_raw == "Unmapped flag"


def test_normalise_skips_malformed_rows_and_missing_geometry():
    rows = [feature(geometry=None), {"properties": {"STATION_NUMBER": "05BB001", "DISCHARGE": "bad"}}]

    [reading] = CanadaHydatCollector(client=Mock()).normalise(rows)

    assert reading.location is None


def test_fetch_raw_applies_filters_and_pagination():
    client = Mock()
    full_page = [feature() for _ in range(1000)]
    client.get_json.side_effect = [
        {"features": full_page},
        {"features": [feature(date="2000-01-02")]},
    ]
    collector = CanadaHydatCollector(client=client)

    rows = collector.fetch_raw(
        station_id="05BB001",
        datetime_start="2000-01-01",
        datetime_end="2000-01-02",
        bbox=(-124, 49, -122, 51),
        max_items=1001,
    )

    assert len(rows) == 1001
    first_params = client.get_json.call_args_list[0].kwargs["params"]
    second_params = client.get_json.call_args_list[1].kwargs["params"]
    assert first_params["STATION_NUMBER"] == "05BB001"
    assert first_params["datetime"] == "2000-01-01/2000-01-02"
    assert first_params["bbox"] == "-124,49,-122,51"
    assert second_params["offset"] == 1000


def test_fetch_raw_caps_results():
    client = Mock()
    client.get_json.return_value = {"features": [feature() for _ in range(1000)]}

    rows = CanadaHydatCollector(client=client).fetch_raw(max_items=12)

    assert len(rows) == 12
    client.get_json.assert_called_once()


def test_fetch_raw_wraps_request_errors():
    client = Mock()
    client.get_json.side_effect = RuntimeError("upstream unavailable")

    with pytest.raises(CollectorError, match="Canada HYDAT fetch failed"):
        CanadaHydatCollector(client=client).fetch_raw(station_id="05BB001")
