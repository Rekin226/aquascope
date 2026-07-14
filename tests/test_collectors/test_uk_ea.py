"""Config-level tests for the USGS API-key resolution and the Taiwan WRA
water-level location extraction added in 0.6.0. These do not hit the network."""

from __future__ import annotations

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
