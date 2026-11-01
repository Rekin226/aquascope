"""Unit tests for the Copernicus EWDS (GloFAS) collector.

``cdsapi.Client`` and the downloaded-file parser are mocked: these tests never
contact the EWDS and need no API key.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from aquascope.collectors.copernicus import CopernicusCollector

EWDS_URL = "https://ewds.climate.copernicus.eu/api"


@pytest.fixture
def fake_cdsapi(monkeypatch):
    module = types.ModuleType("cdsapi")
    module.Client = MagicMock(name="cdsapi.Client")
    monkeypatch.setitem(sys.modules, "cdsapi", module)
    return module


def _fetch(collector: CopernicusCollector, **kwargs):
    with patch.object(CopernicusCollector, "_parse_download", return_value=[{"value": 1.0}]) as parse:
        rows = collector.fetch_raw(latitude=48.85, longitude=2.35, **kwargs)
    return rows, parse


class TestCopernicusCollectorInit:
    def test_default_dataset_is_historical(self):
        assert CopernicusCollector().dataset == CopernicusCollector.GLOFAS_HISTORICAL

    def test_collector_name(self):
        assert CopernicusCollector().name == "copernicus"


class TestCopernicusCollectorFetchRaw:
    def test_client_is_pinned_to_ewds(self, fake_cdsapi):
        rows, _ = _fetch(CopernicusCollector())

        assert rows == [{"value": 1.0}]
        assert fake_cdsapi.Client.call_args.kwargs["url"] == EWDS_URL
        assert fake_cdsapi.Client.return_value.retrieve.called

    def test_default_request_uses_ewds_keys(self, fake_cdsapi):
        _, parse = _fetch(CopernicusCollector())

        dataset, request, target = fake_cdsapi.Client.return_value.retrieve.call_args.args
        assert dataset == "cems-glofas-historical"
        assert request == {
            "variable": "average_river_discharge_in_the_last_24_hours",
            "product_type": "consolidated",
            "system_version": "version_5_0",
            "timespan": "time_mean",
            "hydrological_model": "lisflood",
            "year": "2023",
            "month": "01",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "data_format": "grib2",
            "download_format": "unarchived",
            "area": [49.35, 1.85, 48.35, 2.85],
        }
        assert "format" not in request
        assert target.endswith("download.grib2")
        parse.assert_called_once_with(target, 48.85, 2.35, "grib2")

    def test_default_day_covers_the_whole_month(self, fake_cdsapi):
        _fetch(CopernicusCollector())

        _, request, _ = fake_cdsapi.Client.return_value.retrieve.call_args.args
        assert request["day"] == [f"{d:02d}" for d in range(1, 32)]
        assert isinstance(request["day"], list)

    def test_a_single_day_string_is_sent_as_a_list(self, fake_cdsapi):
        _fetch(CopernicusCollector(), day="05")

        _, request, _ = fake_cdsapi.Client.return_value.retrieve.call_args.args
        assert request["day"] == ["05"]

    def test_instantaneous_timespan_is_selectable(self, fake_cdsapi):
        _fetch(
            CopernicusCollector(),
            variable="snow_depth_water_equivalent",
            timespan="instantaneous",
        )

        _, request, _ = fake_cdsapi.Client.return_value.retrieve.call_args.args
        assert request["timespan"] == "instantaneous"

    def test_older_system_and_netcdf_are_still_configurable(self, fake_cdsapi):
        _, parse = _fetch(
            CopernicusCollector(),
            year=["2024"],
            month=["03"],
            day=["01"],
            variable="runoff_water_equivalent",
            system_version="version_4_0",
            data_format="netcdf",
        )

        _, request, target = fake_cdsapi.Client.return_value.retrieve.call_args.args
        assert request["variable"] == "runoff_water_equivalent"
        assert request["system_version"] == "version_4_0"
        assert request["year"] == ["2024"]
        assert request["day"] == ["01"]
        assert request["data_format"] == "netcdf"
        assert target.endswith("download.nc")
        parse.assert_called_once_with(target, 48.85, 2.35, "netcdf")

    def test_key_from_environment(self, fake_cdsapi, monkeypatch):
        monkeypatch.setenv("CDSAPI_KEY", "12345:secret")

        _fetch(CopernicusCollector())

        assert fake_cdsapi.Client.call_args.kwargs["key"] == "12345:secret"

    def test_explicit_key_wins_over_environment(self, fake_cdsapi, monkeypatch):
        monkeypatch.setenv("CDSAPI_KEY", "12345:environment")

        _fetch(CopernicusCollector(), api_key="67890:explicit")

        assert fake_cdsapi.Client.call_args.kwargs["key"] == "67890:explicit"

    def test_missing_key_is_left_to_cdsapi(self, fake_cdsapi, monkeypatch):
        monkeypatch.delenv("CDSAPI_KEY", raising=False)

        _fetch(CopernicusCollector())

        assert fake_cdsapi.Client.call_args.kwargs["key"] is None

    def test_rejects_unknown_data_format(self):
        with pytest.raises(ValueError, match="data_format"):
            CopernicusCollector().fetch_raw(latitude=0.0, longitude=0.0, data_format="csv")

    def test_rejects_unknown_timespan(self):
        with pytest.raises(ValueError, match="timespan"):
            CopernicusCollector().fetch_raw(latitude=0.0, longitude=0.0, timespan="daily")

    def test_rejects_zip_download(self):
        with pytest.raises(ValueError, match="unarchived"):
            CopernicusCollector().fetch_raw(latitude=0.0, longitude=0.0, download_format="zip")


class TestCopernicusCollectorParse:
    def test_netcdf_uses_xarray_default_engine(self, monkeypatch):
        xarray = MagicMock(name="xarray")
        xarray.open_dataset.return_value.data_vars = {}
        monkeypatch.setitem(sys.modules, "xarray", xarray)

        records = CopernicusCollector()._parse_download("download.nc", 48.85, 2.35, "netcdf")

        assert records == []
        xarray.open_dataset.assert_called_once_with("download.nc")

    def test_grib2_uses_cfgrib(self, monkeypatch):
        xarray = MagicMock(name="xarray")
        xarray.open_dataset.return_value.data_vars = {}
        monkeypatch.setitem(sys.modules, "xarray", xarray)

        records = CopernicusCollector()._parse_download("download.grib2", 48.85, 2.35, "grib2")

        assert records == []
        xarray.open_dataset.assert_called_once_with("download.grib2", engine="cfgrib")
