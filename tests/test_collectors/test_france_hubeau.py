"""Tests for the Hub'Eau (France) hydrometrie collector's normalise().

These test normalise() directly with hand-built fixture rows shaped like
Hub'Eau's real observations_tr response - no network calls."""

from __future__ import annotations

from aquascope.collectors.france_hubeau import HubeauHydrometrieCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample


class TestFranceHubeauNormaliseGrandeur:
    def test_water_level_row(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "H",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 1250.0,
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert isinstance(samples[0], WaterQualitySample)
        assert samples[0].source == DataSource.HUBEAU
        assert samples[0].station_id == "K002000101"
        assert samples[0].parameter == "Water level"
        assert samples[0].value == 1250.0
        assert samples[0].unit == "mm"

    def test_discharge_row(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "Q",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 84.3,
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].parameter == "Discharge"
        assert samples[0].value == 84.3
        assert samples[0].unit == "L/s"

    def test_unknown_grandeur_falls_back_to_raw_code(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "X",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 1.0,
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].parameter == "X"
        assert samples[0].unit == ""


class TestFranceHubeauNormaliseLocation:
    def test_populates_location_when_coords_present(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "A402061001",
                "grandeur_hydro": "H",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": -212.0,
                "latitude": 47.866921289,
                "longitude": 6.796285291,
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].location is not None
        assert samples[0].location.latitude == 47.866921289
        assert samples[0].location.longitude == 6.796285291

    def test_location_none_when_coords_absent(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "Q",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 84.3,
            }
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].location is None


class TestFranceHubeauNormaliseEdgeCases:
    def test_skips_row_with_null_value(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "Q",
                "date_obs": "2026-07-08T10:05:00Z",
                "resultat_obs": None,
            }
        ]
        assert collector.normalise(raw) == []

    def test_skips_row_missing_station_id(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "grandeur_hydro": "Q",
                "date_obs": "2026-07-08T10:05:00Z",
                "resultat_obs": 10.0,
            }
        ]
        assert collector.normalise(raw) == []

    def test_skips_row_with_unparseable_datetime(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "Q",
                "date_obs": "not-a-date",
                "resultat_obs": 10.0,
            }
        ]
        assert collector.normalise(raw) == []

    def test_mixed_batch_skips_only_invalid_rows(self):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "H",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 1250.0,
            },
            {
                "code_station": "K002000101",
                "grandeur_hydro": "Q",
                "date_obs": "2026-07-08T10:05:00Z",
                "resultat_obs": None,
            },
        ]
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].parameter == "Water level"
