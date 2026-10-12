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
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    Quality,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)


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


def _ogc_feature(code, value, time, *, unit="ft^3/s", approval="Approved", qualifier=None, site="USGS-01646500"):
    """One feature as the USGS Water Data OGC API (v1) returns it from ``collections/daily/items``."""
    return {
        "type": "Feature",
        "id": f"{site}-{code}-{time}",
        "geometry": {"type": "Point", "coordinates": [-77.1, 38.9]},
        "properties": {
            "monitoring_location_id": site, "parameter_code": code, "statistic_id": "00003", "time": time,
            "value": value, "unit_of_measure": unit, "approval_status": approval, "qualifier": qualifier,
        },
    }


class TestUSGSCollectorKeyless:
    """Keyless calls go to the same OGC API as keyed ones, just without an ``api_key`` (#515)."""

    def test_keyless_raises_error_without_filter(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        with pytest.raises(ValueError, match="USGS keyless path requires a filter parameter"):
            collector.fetch_raw(collection="daily")

    def test_keyless_field_measurements_are_allowed(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        collector.client.get_json = Mock(return_value={"features": [], "links": []})
        assert collector.fetch_raw(collection="discrete", station_id="01646500") == []
        args, kwargs = collector.client.get_json.call_args
        assert args[0] == "collections/field-measurements/items"
        assert "api_key" not in kwargs["params"]

    def test_keyless_daily_fetch_and_normalise_water_quality_sample(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        page = {"features": [_ogc_feature("00095", "2960", "2026-07-20", unit="uS/cm"),
                             _ogc_feature("00095", "2430", "2026-07-22", unit="uS/cm")], "links": []}
        collector.client.get_json = Mock(return_value=page)

        raw = collector.fetch_raw(collection="daily", station_id="01646500", days=5)

        assert [f["properties"]["value"] for f in raw] == ["2960", "2430"]
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
        collector = USGSCollector(lookup_catchment_area=False)
        page = {"features": [_ogc_feature("00060", "2960", "2026-07-20"),
                             _ogc_feature("00060", "2430", "2026-07-22")], "links": []}
        mock_get_json = Mock(return_value=page)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(collection="daily", station_id="01646500", days=5, statCd="00003")

        mock_get_json.assert_called_once()
        args, kwargs = mock_get_json.call_args
        assert args[0] == "collections/daily/items"
        params = kwargs["params"]
        assert params["monitoring_location_id"] == "USGS-01646500"
        assert params["statistic_id"] == "00003"
        assert "api_key" not in params  # keyless means no key at all, not the shared DEMO_KEY

        samples = collector.normalise(raw)
        assert len(samples) == 2
        assert samples[0].station_id == "USGS-01646500"
        assert samples[0].discharge_cms == 83.8
        assert samples[0].location.latitude == 38.9
        assert samples[0].location.longitude == -77.1
        assert samples[1].discharge_cms == 68.8

    def test_keyless_daily_quality_flags_are_preserved(self, monkeypatch):
        """The #481 mapping, on the fields the v1 API returns (approval_status plus a qualifier list)."""
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector(lookup_catchment_area=False)
        page = {"features": [
            _ogc_feature("00060", "10.0", "2026-07-20", approval="Approved"),
            _ogc_feature("00060", "20.0", "2026-07-21", approval="Provisional"),
            _ogc_feature("00060", "30.0", "2026-07-22", approval="Approved", qualifier=["ESTIMATED"]),
            _ogc_feature("00060", "40.0", "2026-07-23", approval="Approved", qualifier=["ICE"]),
        ], "links": []}
        collector.client.get_json = Mock(return_value=page)

        samples = collector.normalise(collector.fetch_raw(collection="daily", station_id="01646500", days=5))

        assert [s.quality for s in samples] == [Quality.APPROVED, Quality.PROVISIONAL, Quality.ESTIMATED,
                                                Quality.SUSPECT]
        assert samples[2].quality_raw == "Approved | ESTIMATED"

    def test_keyless_continuous_fetch_and_normalise(self, monkeypatch):
        monkeypatch.delenv("USGS_API_KEY", raising=False)
        collector = USGSCollector()
        feature = _ogc_feature("00065", "12.3", "2026-07-24T20:10:00+00:00", unit="ft", approval="Provisional")
        mock_get_json = Mock(return_value={"features": [feature], "links": []})
        collector.client.get_json = mock_get_json

        # "sta" is the v0 name of the continuous collection; it still works.
        raw = collector.fetch_raw(collection="sta", bbox="-77.2,38.8,-77.0,39.0", days=1)
        assert len(raw) == 1
        args, kwargs = mock_get_json.call_args
        assert args[0] == "collections/continuous/items"
        assert kwargs["params"]["bbox"] == "-77.2,38.8,-77.0,39.0"

        samples = collector.normalise(raw)
        assert len(samples) == 1
        reading = samples[0]
        assert isinstance(reading, WaterLevelReading)
        assert reading.source == DataSource.USGS
        assert reading.station_id == "USGS-01646500"
        assert reading.unit == "m"
        # 12.3 has 3 sig figs: 12.3 * 0.3048 = 3.74904 -> 3.75
        assert reading.water_level == pytest.approx(3.75)
        assert reading.quality == Quality.PROVISIONAL
        assert reading.location is not None
        assert reading.location.latitude == 38.9
        assert reading.location.longitude == -77.1


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
            params={"f": "json", "api_key": "valid-key"},
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

    def test_keyed_normalise_gage_height_to_water_level_reading(self):
        collector = USGSCollector(api_key="valid-key")
        raw = [
            {
                "geometry": {"coordinates": [-77.1, 38.9]},
                "properties": {
                    "monitoring_location_id": "01646500",
                    "station_name": "Potomac River",
                    "parameter_code": "00065",
                    "value": "5.40",
                    "time": "2026-07-20T12:00:00Z",
                    "unit_of_measure": "ft",
                },
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        reading = samples[0]
        assert isinstance(reading, WaterLevelReading)
        assert reading.source == DataSource.USGS
        assert reading.station_id == "01646500"
        assert reading.station_name == "Potomac River"
        assert reading.unit == "m"
        # 5.40 has 3 sig figs: 5.40 * 0.3048 = 1.64592 -> 1.65
        assert reading.water_level == pytest.approx(1.65)
        assert reading.location is not None
        assert reading.location.latitude == 38.9
        assert reading.location.longitude == -77.1


class TestUSGSGageHeightNormalisation:
    """#240: USGS gage height (00065) normalises to WaterLevelReading in metres."""

    @pytest.mark.parametrize(
        ("val_str", "expected_water_level"),
        [
            ("10.0", 3.05),    # 3 sig figs: 10.0 * 0.3048 = 3.048 -> 3.05
            ("12.3", 3.75),    # 3 sig figs: 12.3 * 0.3048 = 3.74904 -> 3.75
            ("5.400", 1.646),  # 4 sig figs: 5.400 * 0.3048 = 1.64592 -> 1.646
            ("0.5", 0.2),      # 1 sig fig:  0.5 * 0.3048 = 0.1524 -> 0.2
            ("100", 30.0),     # 1 sig fig:  100 * 0.3048 = 30.48 -> 30.0
        ],
    )
    def test_gage_height_sig_figs_and_metre_conversion(self, val_str, expected_water_level):
        collector = USGSCollector(api_key="valid-key")
        raw = [
            {
                "geometry": {"coordinates": [-77.1, 38.9]},
                "properties": {
                    "monitoring_location_id": "01646500",
                    "parameter_code": "00065",
                    "value": val_str,
                    "time": "2026-07-20T12:00:00Z",
                },
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        reading = samples[0]
        assert isinstance(reading, WaterLevelReading)
        assert reading.unit == "m"
        assert reading.water_level == pytest.approx(expected_water_level)

    def test_discharge_and_water_quality_schemas_preserved(self):
        collector = USGSCollector(api_key="valid-key")
        raw = [
            {
                "geometry": {"coordinates": [-77.1, 38.9]},
                "properties": {
                    "monitoring_location_id": "01646500",
                    "parameter_code": "00060",
                    "value": "100.0",
                    "time": "2026-07-20T12:00:00Z",
                },
            },
            {
                "geometry": {"coordinates": [-77.1, 38.9]},
                "properties": {
                    "monitoring_location_id": "01646500",
                    "parameter_code": "00065",
                    "value": "10.0",
                    "time": "2026-07-20T12:00:00Z",
                },
            },
            {
                "geometry": {"coordinates": [-77.1, 38.9]},
                "properties": {
                    "monitoring_location_id": "01646500",
                    "parameter_code": "00400",
                    "value": "7.5",
                    "time": "2026-07-20T12:00:00Z",
                    "unit_of_measure": "pH units",
                },
            },
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 3
        assert isinstance(samples[0], StreamflowReading)
        assert isinstance(samples[1], WaterLevelReading)
        assert isinstance(samples[2], WaterQualitySample)
        assert samples[0].unit == "m3/s"
        assert samples[1].unit == "m"
        assert samples[2].parameter == "pH"


class TestUSGSKeyedFiltersReachTheOGCPath:
    """#160: with an API key the OGC path must filter, not crawl the nation."""

    def test_station_parameter_and_area_filters_are_mapped(self):
        from unittest.mock import MagicMock

        from aquascope.collectors.usgs import USGSCollector

        c = USGSCollector(api_key="a-real-key")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="01646500", days=30, collection="daily", parameter="00060", max_items=None)
        params = c.client.get_json.call_args.kwargs["params"]
        assert params["monitoring_location_id"] == "USGS-01646500"
        assert params["parameter_code"] == "00060" and params["api_key"] == "a-real-key"
        assert c.client.get_json.call_args.args[0] == "collections/daily/items"

        c.fetch_raw(collection="daily", days=1, stateCd="24", huc="02070008", countyCd="031",
                    station_id=["01646500", "USGS-01646000"])
        params = c.client.get_json.call_args.kwargs["params"]
        assert params["monitoring_location_id"] == "USGS-01646500,USGS-01646000"
        assert params["state_code"] == "24" and params["county_code"] == "031"
        assert params["hydrologic_unit_code"] == "02070008"


class TestUSGSOGCPagination:
    def test_next_link_is_fetched_as_is_and_loops_are_guarded(self):
        from unittest.mock import MagicMock

        from aquascope.collectors.usgs import USGSCollector

        c = USGSCollector(api_key="a-real-key")
        c.client = MagicMock()
        page1 = {"features": [{"a": 1}], "links": [{"rel": "next", "href": "https://x/items?cursor=abc&f=json"}]}
        page2 = {"features": [{"a": 2}], "links": [{"rel": "next", "href": "https://x/items?cursor=abc&f=json"}]}
        c.client.get_json.side_effect = [page1, page2, page2, page2]
        out = c.fetch_raw(station_id="01646500", days=3, collection="daily", parameter="00060", max_items=None)
        assert len(out) == 2  # page1 + page2; the repeated cursor stops the loop
        second_call = c.client.get_json.call_args_list[1]
        # USGS builds next links without the key, so a keyed walk puts it back on (#515).
        assert second_call.args[0] == "https://x/items?cursor=abc&f=json&api_key=a-real-key"
        assert second_call.kwargs["params"] is None

    def test_keyless_walk_follows_every_page_until_the_last(self):
        from unittest.mock import MagicMock

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        base = "https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items"
        pages = [
            {"features": [{"n": 1}, {"n": 2}], "links": [{"rel": "next", "href": f"{base}?cursor=p2&f=json"}]},
            {"features": [{"n": 3}, {"n": 4}], "links": [{"rel": "self", "href": "x"},
                                                        {"rel": "next", "href": f"{base}?cursor=p3&f=json"}]},
            {"features": [{"n": 5}], "links": [{"rel": "self", "href": "x"}]},
        ]
        c.client.get_json.side_effect = pages
        out = c.fetch_raw(station_id="01646500", days=3, collection="daily", max_items=None, limit=2)
        assert [f["n"] for f in out] == [1, 2, 3, 4, 5]
        urls = [call.args[0] for call in c.client.get_json.call_args_list]
        assert urls == ["collections/daily/items", f"{base}?cursor=p2&f=json", f"{base}?cursor=p3&f=json"]
        assert all("api_key" not in u for u in urls)  # keyless pages stay keyless

    def test_max_items_stops_the_walk_mid_page(self):
        from unittest.mock import MagicMock

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        c.client.get_json.side_effect = [
            {"features": [{"n": i} for i in range(3)], "links": [{"rel": "next", "href": "https://x/p2"}]},
            {"features": [{"n": i} for i in range(3, 6)], "links": [{"rel": "next", "href": "https://x/p3"}]},
        ]
        out = c.fetch_raw(station_id="01646500", days=3, collection="daily", max_items=4)
        assert [f["n"] for f in out] == [0, 1, 2, 3]
        assert c.client.get_json.call_count == 2

    def test_slim_series_options_reach_the_query(self):
        from unittest.mock import MagicMock

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="01646500", days=3, collection="daily", parameter="00060", statCd="00003",
                    limit=50_000, skip_geometry=True, properties=("time", "value", "approval_status"))
        params = c.client.get_json.call_args.kwargs["params"]
        assert params["limit"] == 50_000 and params["statistic_id"] == "00003"
        assert params["skipGeometry"] == "true" and params["properties"] == "time,value,approval_status"

    def test_statistic_filter_is_not_sent_to_field_measurements(self):
        from unittest.mock import MagicMock

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="01646500", days=3, collection="field-measurements", statCd="00003")
        assert "statistic_id" not in c.client.get_json.call_args.kwargs["params"]  # the API 400s on it


class TestUSGSv1Endpoints:
    def test_base_url_is_ogc_v1(self):
        from aquascope.collectors.usgs import USGS_BASE

        assert USGS_BASE == "https://api.waterdata.usgs.gov/ogcapi/v1"
        assert USGSCollector().client.base_url == USGS_BASE

    @pytest.mark.parametrize("old,new", [("sta", "continuous"), ("iv", "continuous"), ("dv", "daily"),
                                         ("discrete", "field-measurements"), ("daily", "daily")])
    def test_old_collection_names_map_to_v1(self, old, new):
        from unittest.mock import MagicMock

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="01646500", days=3, collection=old)
        assert c.client.get_json.call_args.args[0] == f"collections/{new}/items"

    def test_keyed_area_lookup_carries_the_key(self):
        c = USGSCollector(api_key="a-real-key")
        USGSCollector._shared_area_cache.pop("USGS-09999999", None)
        c.client.get_json = Mock(return_value={"properties": {"drainage_area": "10"}})
        c._get_monitoring_location_catchment_area("09999999")
        c.client.get_json.assert_called_once_with(
            "collections/monitoring-locations/items/USGS-09999999", params={"f": "json", "api_key": "a-real-key"})
        USGSCollector._shared_area_cache.pop("USGS-09999999", None)


class TestUSGSAgencyPrefixedIds:
    def test_keyed_path_keeps_other_agency_prefixes(self):
        from unittest.mock import MagicMock

        from aquascope.collectors.usgs import USGSCollector

        c = USGSCollector(api_key="a-real-key")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="CA574-09527500", days=3, collection="daily", parameter="00060", max_items=None)
        assert c.client.get_json.call_args.kwargs["params"]["monitoring_location_id"] == "CA574-09527500"
        c.fetch_raw(station_id="01646500", days=3, collection="daily", max_items=None)
        assert c.client.get_json.call_args.kwargs["params"]["monitoring_location_id"] == "USGS-01646500"

    def test_keyless_path_keeps_the_agency_prefix(self):
        from unittest.mock import MagicMock

        from aquascope.collectors.usgs import USGSCollector

        c = USGSCollector(api_key="DEMO_KEY")
        c.client = MagicMock()
        c.client.get_json.return_value = {"features": [], "links": []}
        c.fetch_raw(station_id="CA574-09527500", days=3, collection="daily", parameter="00060", max_items=None)
        params = c.client.get_json.call_args.kwargs["params"]
        assert params["monitoring_location_id"] == "CA574-09527500" and "api_key" not in params
        c.fetch_raw(station_id="USGS-01646500", days=3, collection="daily", max_items=None)
        assert c.client.get_json.call_args.kwargs["params"]["monitoring_location_id"] == "USGS-01646500"


class TestUSGSSharedPacingAndAreaLookup:
    """The harvest builds one collector per station; pacing and the area cache must span them (#harvest 429s)."""

    def test_collectors_share_one_rate_limiter_under_the_keyed_quota(self):
        a, b = USGSCollector(api_key="k"), USGSCollector(api_key="k")
        assert a.client.rate_limiter is b.client.rate_limiter
        lim = a.client.rate_limiter
        assert lim.max_calls * 3600 / lim.period < 1000  # a keyed USGS quota is 1,000 requests an hour

    def test_the_area_lookup_is_asked_once_across_collectors(self):
        first, second = USGSCollector(api_key="k"), USGSCollector(api_key="k")
        first.client.get_json = Mock(return_value={"properties": {"drainage_area": "12.5"}})
        second.client.get_json = Mock()
        assert first._get_monitoring_location_catchment_area("01646500") == pytest.approx(32.4)
        assert second._get_monitoring_location_catchment_area("01646500") == pytest.approx(32.4)
        second.client.get_json.assert_not_called()

    def test_a_throttled_area_lookup_is_not_remembered_as_missing(self):
        from aquascope.utils.http_client import RateLimitedError

        c = USGSCollector(api_key="k")
        c.client.get_json = Mock(side_effect=RateLimitedError("https://example.test"))
        assert c._get_monitoring_location_catchment_area("01646500") is None
        c.client.get_json = Mock(return_value={"properties": {"drainage_area": "12.5"}})
        assert c._get_monitoring_location_catchment_area("01646500") == pytest.approx(32.4)

    def test_the_series_path_skips_the_area_lookup(self):
        c = USGSCollector(api_key="k", lookup_catchment_area=False)
        c._get_monitoring_location_catchment_area = Mock(return_value=99.0)
        raw = {"features": [{"geometry": {"coordinates": [-77.1, 38.9]}, "properties": {
            "monitoring_location_id": "USGS-01646500", "parameter_code": "00060", "value": "2960",
            "time": "2026-07-20", "unit_of_measure": "ft^3/s"}}]}
        recs = c.normalise(raw["features"])
        assert recs and recs[0].catchment_area_km2 is None
        c._get_monitoring_location_catchment_area.assert_not_called()
