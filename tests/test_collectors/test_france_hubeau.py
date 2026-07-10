"""Tests for the Hub'Eau (France) hydrometrie collector's normalise().

These test normalise() directly with hand-built fixture rows shaped like
Hub'Eau's real observations_tr response - no network calls."""

from __future__ import annotations

from unittest.mock import Mock

from aquascope.collectors.france_hubeau import HubeauHydrometrieCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample


class TestFranceHubeauFetchRawPagination:
    def test_follows_next_link_across_pages(self):
        collector = HubeauHydrometrieCollector()

        page1 = {
            "data": [
                {
                    "code_station": "A1",
                    "grandeur_hydro": "Q",
                    "date_obs": "2026-07-08T10:00:00Z",
                    "resultat_obs": 1.0,
                }
            ],
            "next": "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr?cursor=abc&size=1",
        }
        page2 = {
            "data": [
                {
                    "code_station": "A1",
                    "grandeur_hydro": "Q",
                    "date_obs": "2026-07-08T10:05:00Z",
                    "resultat_obs": 2.0,
                }
            ],
            # no "next" key - this is the last page
        }

        mock_get_json = Mock(side_effect=[page1, page2])
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(code_station="A1", size=1, max_items=None)

        assert len(raw) == 2
        assert raw[0]["resultat_obs"] == 1.0
        assert raw[1]["resultat_obs"] == 2.0
        assert mock_get_json.call_count == 2

        first_args, first_kwargs = mock_get_json.call_args_list[0]
        assert first_args[0] == "/observations_tr"
        assert first_kwargs["params"]["code_station"] == "A1"

        second_args, second_kwargs = mock_get_json.call_args_list[1]
        assert second_args[0] == page1["next"]
        assert second_kwargs["params"] == {}

    def test_stops_when_max_items_reached_mid_page(self):
        collector = HubeauHydrometrieCollector()
        page1 = {
            "data": [
                {
                    "code_station": "A1",
                    "grandeur_hydro": "Q",
                    "date_obs": "2026-07-08T10:00:00Z",
                    "resultat_obs": 1.0,
                },
                {
                    "code_station": "A1",
                    "grandeur_hydro": "Q",
                    "date_obs": "2026-07-08T10:01:00Z",
                    "resultat_obs": 2.0,
                },
            ],
            "next": "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr?cursor=abc",
        }
        mock_get_json = Mock(return_value=page1)
        collector.client.get_json = mock_get_json

        raw = collector.fetch_raw(code_station="A1", size=2, max_items=1)

        assert len(raw) == 1  # truncated to max_items even mid-page
        assert mock_get_json.call_count == 1  # never followed next_link


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


class TestFranceHubeauNormaliseDatetime:
    def test_z_suffix_parses_to_tz_naive(self):
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
        dt = samples[0].sample_datetime
        assert dt.tzinfo is None
        assert dt.isoformat() == "2026-07-08T10:00:00"


class TestFranceHubeauNormaliseLogging:
    def test_warns_with_skip_count_when_rows_skipped(self, caplog):
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
        with caplog.at_level("WARNING"):
            samples = collector.normalise(raw)
        assert len(samples) == 1
        assert any("skipped 1/2" in r.message for r in caplog.records)

    def test_no_warning_when_nothing_skipped(self, caplog):
        collector = HubeauHydrometrieCollector()
        raw = [
            {
                "code_station": "K002000101",
                "grandeur_hydro": "H",
                "date_obs": "2026-07-08T10:00:00Z",
                "resultat_obs": 1250.0,
            }
        ]
        with caplog.at_level("WARNING"):
            collector.normalise(raw)
        assert not any("skipped" in r.message for r in caplog.records)


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
