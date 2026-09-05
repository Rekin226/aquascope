"""Offline tests for the Taiwan data.gov.tw collector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aquascope.collectors.taiwan_datagov import (
    DATASET_GROUNDWATER,
    DATASET_WATER_LEVEL,
    TaiwanDataGovCollector,
)
from aquascope.schemas.water_data import DataSource


def test_constructor_keeps_tls_verification_enabled():
    with patch("aquascope.collectors.taiwan_datagov.CachedHTTPClient") as client_cls:
        TaiwanDataGovCollector()

    kwargs = client_cls.call_args.kwargs
    assert kwargs["relax_strict_tls"] is True
    assert "verify" not in kwargs


def test_legacy_dataset_ids_are_mapped():
    client = MagicMock()

    water = TaiwanDataGovCollector(dataset_id="25768", client=client)
    groundwater = TaiwanDataGovCollector(dataset_id="161082", client=client)

    assert water.dataset_id == DATASET_WATER_LEVEL
    assert groundwater.dataset_id == DATASET_GROUNDWATER


def test_fetch_raw_returns_all_records_by_default_and_supports_limit():
    client = MagicMock()
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    client.get_json.return_value = records
    collector = TaiwanDataGovCollector(client=client)

    assert collector.fetch_raw() == records
    assert collector.fetch_raw(limit=1, offset=1) == [{"id": 2}]
    assert client.get_json.call_args_list[0].args == (DATASET_WATER_LEVEL,)
    assert client.get_json.call_args_list[0].kwargs == {"params": {"format": "json"}}


def test_normalise_lowercase_water_level_and_groundwater_records():
    collector = TaiwanDataGovCollector(client=MagicMock())
    raw = [
        {
            "stationid": "river-1",
            "waterlevel": "2.75",
            "datetime": "2026-08-31T12:00:00",
        },
        {
            "wellidentifier": "well-1",
            "waterlevel": "-1.25",
            "recordtime": "2026-08-31T13:00:00",
        },
        {
            "stationid": "empty",
            "waterlevel": "",
            "datetime": "2026-08-31T14:00:00",
        },
    ]

    readings = list(collector.normalise(raw))

    assert len(readings) == 2
    assert [reading.station_id for reading in readings] == ["river-1", "well-1"]
    assert [reading.water_level for reading in readings] == [2.75, -1.25]
    assert all(reading.source == DataSource.TAIWAN_DATAGOV for reading in readings)
