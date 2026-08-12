"""Config-level tests for the USGS API-key resolution and the Taiwan WRA
water-level location extraction added in 0.6.0. These do not hit the network."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from aquascope.collectors.taiwan_wra import (
    TaiwanWRAWaterLevelCollector,
    _extract_location,
)
from aquascope.collectors.usgs import USGSCollector
from aquascope.schemas.water_data import GeoLocation


class TestUSGSKeyResolution:
    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("USGS_API_KEY", "from-env")
        assert USGSCollector(api_key="explicit").api_key == "explicit"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("USGS_API_KEY", "from-env")
        assert USGSCollector().api_key == "from-env"

    def test_demo_key_fallback_warns(self, monkeypatch, caplog):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        with caplog.at_level("WARNING"):
            collector = USGSCollector()
        assert collector.api_key == "DEMO_KEY"
        assert any("DEMO_KEY" in r.message for r in caplog.records)


class TestExtractLocation:
    def test_returns_geolocation_when_coords_present(self):
        loc = _extract_location({"Latitude": "24.15", "Longitude": "120.68"})
        assert isinstance(loc, GeoLocation)
        assert loc.latitude == 24.15
        assert loc.longitude == 120.68

    def test_alternate_key_names(self):
        loc = _extract_location({"TWD97Lat": 23.5, "TWD97Lon": 121.0})
        assert isinstance(loc, GeoLocation)

    def test_none_when_coords_absent(self):
        assert _extract_location({"StationName": "X"}) is None

    def test_none_when_coords_unparseable(self):
        assert _extract_location({"lat": "n/a", "lon": "n/a"}) is None


class TestWRANormaliseUsesLocation:
    def test_normalise_populates_location(self):
        collector = TaiwanWRAWaterLevelCollector()
        raw = [
            {
                "StationIdentifier": "1140H013",
                "StationName": "Test",
                "WaterLevel": "12.3",
                "RecordTime": "2026-06-01T08:00:00",
                "Latitude": "24.15",
                "Longitude": "120.68",
            }
        ]
        readings = collector.normalise(raw)
        assert len(readings) == 1
        assert readings[0].location is not None
        assert readings[0].location.latitude == 24.15

    def test_normalise_without_coords_is_none(self):
        collector = TaiwanWRAWaterLevelCollector()
        raw = [
            {
                "StationIdentifier": "1140H013",
                "WaterLevel": "12.3",
                "RecordTime": "2026-06-01T08:00:00",
            }
        ]
        readings = collector.normalise(raw)
        assert len(readings) == 1
        assert readings[0].location is None


class TestUSGSCollectorKeyless:
    def test_keyless_raises_error_for_unsupported_collection(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        with pytest.raises(ValueError, match="Collection 'discrete' is not supported"):
            collector.fetch_raw(collection="discrete", station_id="01646500")

    def test_keyless_raises_error_without_filter(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        with pytest.raises(ValueError, match="USGS keyless path requires a filter parameter"):
            collector.fetch_raw(collection="daily")

    def test_keyless_daily_fetch_and_normalise_water_quality_sample(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()

        mock_response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteName": "Test Site",
                            "siteCode": [{"value": "01646500"}],
                            "geoLocation": {
                                "geogLocation": {
                                    "latitude": 38.9,
                                    "longitude": -77.1
                                }
                            }
                        },
                        "variable": {
                            "variableCode": [{"value": "00095"}],
                            "unit": {"unitCode": "uS/cm"},
                            "noDataValue": -999999.0
                        },
                        "values": [
                            {
                                "value": [
                                    {"value": "2960", "dateTime": "2026-07-20T00:00:00.000"},
                                    {"value": "-999999", "dateTime": "2026-07-21T00:00:00.000"},
                                    {"value": "2430", "dateTime": "2026-07-22T00:00:00.000"}
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        mock_get_json = Mock(return_value=mock_response)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(collection="daily", station_id="01646500", days=5)

        assert len(raw) == 2
        assert raw[0]["properties"]["parameter_code"] == "00095"
        assert raw[0]["properties"]["value"] == "2960"
        assert raw[0]["properties"]["monitoring_location_id"] == "01646500"
        assert raw[1]["properties"]["value"] == "2430"

        samples = collector.normalise(raw)
        assert len(samples) == 2
        assert samples[0].parameter == "Conductivity"
        assert samples[0].value == 2960.0
        assert samples[0].unit == "uS/cm"
        assert samples[0].location.latitude == 38.9
        assert samples[0].location.longitude == -77.1
        assert samples[1].value == 2430.0

    def test_keyless_daily_fetch_and_normalise_streamflow_reading(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()

        mock_response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteName": "Test Site",
                            "siteCode": [{"value": "01646500"}],
                            "geoLocation": {
                                "geogLocation": {
                                    "latitude": 38.9,
                                    "longitude": -77.1
                                }
                            }
                        },
                        "variable": {
                            "variableCode": [{"value": "00060"}],
                            "unit": {"unitCode": "ft3/s"},
                            "noDataValue": -999999.0
                        },
                        "values": [
                            {
                                "value": [
                                    {"value": "2960", "dateTime": "2026-07-20T00:00:00.000"},
                                    {"value": "-999999", "dateTime": "2026-07-21T00:00:00.000"},
                                    {"value": "2430", "dateTime": "2026-07-22T00:00:00.000"}
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        mock_get_json = Mock(return_value=mock_response)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(collection="daily", station_id="01646500", days=5)

        assert len(raw) == 2
        assert raw[0]["properties"]["value"] == "2960"
        assert raw[0]["properties"]["monitoring_location_id"] == "01646500"
        assert raw[1]["properties"]["value"] == "2430"

        mock_get_json.assert_called_once()
        args, kwargs = mock_get_json.call_args
        assert args[0].startswith("https://waterservices.usgs.gov/nwis/dv/")
        assert kwargs["params"]["sites"] == "01646500"

        samples = collector.normalise(raw)
        assert len(samples) == 2
        assert samples[0].discharge_cms == 83.8
        assert samples[0].location.latitude == 38.9
        assert samples[0].location.longitude == -77.1
        assert samples[1].discharge_cms == 68.8

    def test_keyless_sta_fetch_and_normalise(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()

        mock_response = {
            "value": {
                "timeSeries": [
                    {
                        "sourceInfo": {
                            "siteName": "Test Site",
                            "siteCode": [{"value": "01646500"}],
                            "geoLocation": {
                                "geogLocation": {
                                    "latitude": 38.9,
                                    "longitude": -77.1
                                }
                            }
                        },
                        "variable": {
                            "variableCode": [{"value": "00065"}],
                            "unit": {"unitCode": "ft"},
                            "noDataValue": -999999.0
                        },
                        "values": [
                            {
                                "value": [
                                    {"value": "12.3", "dateTime": "2026-07-24T20:10:00.000-04:00"}
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        mock_get_json = Mock(return_value=mock_response)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(collection="sta", bbox="-77.2,38.8,-77.0,39.0", days=1)
        assert len(raw) == 1
        assert raw[0]["properties"]["value"] == "12.3"

        mock_get_json.assert_called_once()
        args, kwargs = mock_get_json.call_args
        assert args[0].startswith("https://waterservices.usgs.gov/nwis/iv/")
        assert kwargs["params"]["bBox"] == "-77.2,38.8,-77.0,39.0"


class TestUSGSMonitoringLocationCatchmentArea:
    def test_returns_none_for_empty_location_id(self):
        collector = USGSCollector(api_key="valid-key")

        assert collector._get_monitoring_location_catchment_area("") is None

    def test_converts_and_rounds_drainage_area(self):
        collector = USGSCollector(api_key="valid-key")
        collector.client.get_json = Mock(
            return_value={"properties": {"drainage_area": "12.5"}}
        )

        area = collector._get_monitoring_location_catchment_area("01646500")

        assert area == pytest.approx(32.4)
        collector.client.get_json.assert_called_once_with(
            "collections/monitoring-locations/items/USGS-01646500",
            params={"f": "json"},
        )

    def test_returns_none_when_feature_has_no_drainage_area(self):
        collector = USGSCollector(api_key="valid-key")
        collector.client.get_json = Mock(return_value={"properties": {}})

        assert collector._get_monitoring_location_catchment_area("01646500") is None


class TestUSGSSignificantFiguresHelpers:
    def test_count_sig_figs_handles_common_formats(self):
        collector = USGSCollector(api_key="valid-key")

        assert collector._count_sig_figs("12.50") == 4
        assert collector._count_sig_figs("-12.50") == 4
        assert collector._count_sig_figs("+12.50") == 4
        assert collector._count_sig_figs("0.001230") == 4
        assert collector._count_sig_figs("1000") == 1
        assert collector._count_sig_figs("1000.0") == 5
        assert collector._count_sig_figs("0") == 1
        assert collector._count_sig_figs("100.") == 3

    def test_round_to_sig_figs_rounds_correctly(self):
        assert USGSCollector._round_to_sig_figs(1234.567, 4) == pytest.approx(1235.0)
        assert USGSCollector._round_to_sig_figs(0.0012345, 3) == pytest.approx(0.00123)
        assert USGSCollector._round_to_sig_figs(0.0012365, 3) == pytest.approx(0.00124)
        assert USGSCollector._round_to_sig_figs(0, 5) == 0
        assert USGSCollector._round_to_sig_figs(5, 0) == 5


class TestUSGSCollectorKeyed:
    def test_keyed_uses_ogc_api(self, monkeypatch):
        collector = USGSCollector(api_key="valid-key")

        mock_response = {
            "features": [
                {
                    "geometry": {"coordinates": [-77.1, 38.9]},
                    "properties": {
                        "monitoring_location_id": "01646500",
                        "parameter_code": "00060",
                        "value": 2960.0,
                        "time": "2026-07-20T00:00:00Z",
                        "unit_of_measure": "ft3/s"
                    }
                }
            ]
        }

        mock_get_json = Mock(return_value=mock_response)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(collection="daily", station_id="01646500")
        assert len(raw) == 1
        assert raw[0]["properties"]["value"] == 2960.0

        mock_get_json.assert_called_once()
        args, kwargs = mock_get_json.call_args
        assert args[0] == "collections/daily/items"
