"""Tests for the Japan MLIT water data collector."""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.japan_mlit import (
    JAPAN_RIVER_SYSTEMS,
    PARAMETER_MAP_JA,
    PREFECTURE_CODES,
    QUALITY_STANDARDS_JAPAN,
    JapanMLITCollector,
)
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "station_code": "305041281005030",
        "station_name": "利根川 栗橋",
        "parameter": "水位",
        "value": 3.45,
        "unit": "m",
        "datetime": "2023-07-15T10:00:00",
        "latitude": 36.133,
        "longitude": 139.717,
        "river_system": "利根川",
    },
    {
        "station_code": "305041281005031",
        "station_name": "石狩川 旭橋",
        "parameter": "流量",
        "value": 125.8,
        "unit": "m3/s",
        "datetime": "2023-07-15T12:00:00",
        "latitude": 43.770,
        "longitude": 142.370,
        "river_system": "石狩川",
    },
    {
        "station_code": "WQ001",
        "station_name": "淀川 枚方",
        "parameter": "pH",
        "value": 7.2,
        "unit": "",
        "datetime": "2023-08-01T09:00:00",
        "latitude": 34.816,
        "longitude": 135.649,
        "river_system": "淀川",
    },
    {
        "station_code": "WQ002",
        "station_name": "信濃川 大手大橋",
        "parameter": "DO",
        "value": 8.5,
        "unit": "mg/L",
        "datetime": "2023-08-01T10:00:00",
        "latitude": 37.900,
        "longitude": 139.020,
        "river_system": "信濃川",
    },
]

SAMPLE_RAW_SKIP = [
    {
        "station_code": "SKIP01",
        "station_name": "test",
        "parameter": "水位",
        "value": "ND",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
    {
        "station_code": "SKIP02",
        "station_name": "test",
        "parameter": "水位",
        "value": "",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
]


class TestJapanMLITInit:
    def setup_method(self):
        self.collector = JapanMLITCollector()

    def test_collector_name(self):
        assert self.collector.name == "japan_mlit"

    def test_base_url(self):
        assert self.collector.BASE_URL == "http://www1.river.go.jp/cgi-bin/"


class TestJapanMLITNormalise:
    def setup_method(self):
        self.collector = JapanMLITCollector()

    def test_normalise_produces_correct_count(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 4

    def test_normalise_maps_japanese_to_english(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].parameter == "water_level"
        assert records[1].parameter == "discharge"
        assert records[0].source.value == "japan_mlit"

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.JAPAN_MLIT

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - 36.133) < 0.01
        assert abs(rec.location.longitude - 139.717) < 0.01

    def test_normalise_maps_river_system(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].basin == "Tone"
        assert records[1].basin == "Ishikari"

    def test_normalise_preserves_station_code(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_id == "305041281005030"

    def test_normalise_skips_nd_and_empty(self):
        records = self.collector.normalise(SAMPLE_RAW_SKIP)
        assert len(records) == 0

    def test_normalise_empty_input(self):
        records = self.collector.normalise([])
        assert records == []

    def test_normalise_keeps_passthrough_params(self):
        records = self.collector.normalise(SAMPLE_RAW)
        params = {r.parameter for r in records}
        assert "pH" in params
        assert "DO" in params


class TestJapanMLITFetchRaw:
    def setup_method(self):
        self.collector = JapanMLITCollector()

    def test_fetch_raw_invalid_parameter(self):
        try:
            self.collector.fetch_raw(parameter="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "Unsupported parameter" in str(exc)

    def test_fetch_raw_handles_connection_error(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("test error")
        collector = JapanMLITCollector(client=mock_client)
        result = collector.fetch_raw(station_id="12345")
        assert result == []

    def test_fetch_raw_builds_station_param(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response
        collector = JapanMLITCollector(client=mock_client)
        collector.fetch_raw(station_id="305041281005030")
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["StationID"] == "305041281005030"

    def test_fetch_raw_builds_date_params(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response
        collector = JapanMLITCollector(client=mock_client)
        collector.fetch_raw(start_date="2023-01-01", end_date="2023-12-31")
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["StartDate"] == "20230101"
        assert call_args[1]["params"]["EndDate"] == "20231231"


class TestPrefectureCodes:
    def test_tokyo_code(self):
        assert PREFECTURE_CODES["Tokyo"] == "13"

    def test_osaka_code(self):
        assert PREFECTURE_CODES["Osaka"] == "27"

    def test_hokkaido_code(self):
        assert PREFECTURE_CODES["Hokkaido"] == "01"

    def test_aichi_code(self):
        assert PREFECTURE_CODES["Aichi"] == "23"

    def test_fukuoka_code(self):
        assert PREFECTURE_CODES["Fukuoka"] == "40"


class TestParameterMapJA:
    def test_all_core_mappings_present(self):
        assert PARAMETER_MAP_JA["水位"] == "water_level"
        assert PARAMETER_MAP_JA["流量"] == "discharge"
        assert PARAMETER_MAP_JA["pH"] == "pH"
        assert PARAMETER_MAP_JA["DO"] == "DO"
        assert PARAMETER_MAP_JA["BOD"] == "BOD"
        assert PARAMETER_MAP_JA["COD"] == "COD"
        assert PARAMETER_MAP_JA["SS"] == "SS"


class TestRiverSystems:
    def test_at_least_ten_systems(self):
        assert len(JAPAN_RIVER_SYSTEMS) >= 10

    def test_tone_river(self):
        assert JAPAN_RIVER_SYSTEMS["利根川"] == "Tone"

    def test_shinano_river(self):
        assert JAPAN_RIVER_SYSTEMS["信濃川"] == "Shinano"


class TestQualityStandards:
    def test_grade_aa_thresholds(self):
        aa = QUALITY_STANDARDS_JAPAN["AA"]
        assert aa["BOD"] == 1.0
        assert aa["DO"] == 7.5

    def test_all_grades_present(self):
        expected = {"AA", "A", "B", "C", "D", "E"}
        assert expected == set(QUALITY_STANDARDS_JAPAN.keys())

    def test_bod_increases_with_worse_grade(self):
        grades = ["AA", "A", "B", "C", "D", "E"]
        bod_values = [QUALITY_STANDARDS_JAPAN[g]["BOD"] for g in grades]
        assert bod_values == sorted(bod_values)
