"""Tests for the Korea WAMIS water data collector."""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.korea_wamis import (
    KOREA_DAM_LIST,
    KOREA_MAJOR_BASINS,
    KOREA_QUALITY_GRADES,
    PARAMETER_MAP_KO,
    KoreaWAMISCollector,
)
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "station_code": "1018680",
        "station_name": "한강 팔당",
        "parameter": "수위",
        "value": 25.34,
        "unit": "m",
        "datetime": "2023-06-20T08:00:00",
        "latitude": 37.520,
        "longitude": 127.300,
        "basin": "한강",
    },
    {
        "station_code": "2018100",
        "station_name": "낙동강 물금",
        "parameter": "유량",
        "value": 85.2,
        "unit": "m3/s",
        "datetime": "2023-06-20T09:00:00",
        "latitude": 35.320,
        "longitude": 128.990,
        "basin": "낙동강",
    },
    {
        "obscd": "WQ3001",
        "obsnm": "금강 공주",
        "parameter": "pH",
        "value": 7.8,
        "unit": "",
        "datetime": "2023-07-01T10:00:00",
        "lat": 36.465,
        "lon": 127.125,
        "basin": "금강",
    },
    {
        "station_code": "WQ4001",
        "station_name": "영산강 승촌보",
        "parameter": "T-N",
        "value": 2.15,
        "unit": "mg/L",
        "datetime": "2023-07-01T11:00:00",
        "latitude": 35.030,
        "longitude": 126.840,
        "basin": "영산강",
    },
]

SAMPLE_RAW_DAM = [
    {
        "station_code": "DAM001",
        "station_name": "충주댐",
        "parameter": "저수량",
        "value": 1850.5,
        "unit": "MCM",
        "datetime": "2023-08-15T00:00:00",
        "latitude": 36.970,
        "longitude": 128.000,
        "basin": "한강",
    },
]

SAMPLE_RAW_SKIP = [
    {
        "station_code": "SKIP01",
        "station_name": "test",
        "parameter": "수위",
        "value": "ND",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
    {
        "station_code": "SKIP02",
        "station_name": "test",
        "parameter": "수위",
        "value": "--",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
]


class TestKoreaWAMISInit:
    def setup_method(self):
        self.collector = KoreaWAMISCollector()

    def test_collector_name(self):
        assert self.collector.name == "korea_wamis"

    def test_base_url(self):
        assert self.collector.BASE_URL == "http://www.wamis.go.kr/openapi/"


class TestKoreaWAMISNormalise:
    def setup_method(self):
        self.collector = KoreaWAMISCollector()

    def test_normalise_produces_correct_count(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 4

    def test_normalise_maps_korean_to_english(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].parameter == "water_level"
        assert records[1].parameter == "discharge"

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.KOREA_WAMIS

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - 37.520) < 0.01
        assert abs(rec.location.longitude - 127.300) < 0.01

    def test_normalise_maps_basin_korean_to_english(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].basin == "Han"
        assert records[1].basin == "Nakdong"
        assert records[2].basin == "Geum"

    def test_normalise_maps_total_nitrogen(self):
        records = self.collector.normalise(SAMPLE_RAW)
        tn_records = [r for r in records if r.parameter == "Total Nitrogen"]
        assert len(tn_records) == 1
        assert tn_records[0].value == 2.15

    def test_normalise_dam_storage(self):
        records = self.collector.normalise(SAMPLE_RAW_DAM)
        assert len(records) == 1
        assert records[0].parameter == "dam_storage"
        assert records[0].value == 1850.5

    def test_normalise_skips_nd_and_dashes(self):
        records = self.collector.normalise(SAMPLE_RAW_SKIP)
        assert len(records) == 0

    def test_normalise_empty_input(self):
        records = self.collector.normalise([])
        assert records == []

    def test_normalise_uses_obscd_fallback(self):
        records = self.collector.normalise(SAMPLE_RAW)
        # Third record uses obscd instead of station_code
        assert records[2].station_id == "WQ3001"

    def test_normalise_uses_obsnm_fallback(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[2].station_name == "금강 공주"

    def test_normalise_location_lat_lon_fallback(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[2]
        assert rec.location is not None
        assert abs(rec.location.latitude - 36.465) < 0.01


class TestKoreaWAMISFetchRaw:
    def setup_method(self):
        self.collector = KoreaWAMISCollector()

    def test_fetch_raw_invalid_parameter(self):
        try:
            self.collector.fetch_raw(parameter="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "Unsupported parameter" in str(exc)

    def test_fetch_raw_handles_connection_error(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("test error")
        collector = KoreaWAMISCollector(client=mock_client)
        result = collector.fetch_raw(station_id="12345")
        assert result == []

    def test_fetch_raw_builds_station_param(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response
        collector = KoreaWAMISCollector(client=mock_client)
        collector.fetch_raw(station_id="1018680")
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["obscd"] == "1018680"

    def test_fetch_raw_builds_date_params(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response
        collector = KoreaWAMISCollector(client=mock_client)
        collector.fetch_raw(start_date="2023-01-01", end_date="2023-12-31")
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["startdt"] == "20230101"
        assert call_args[1]["params"]["enddt"] == "20231231"

    def test_fetch_raw_basin_filter(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response
        collector = KoreaWAMISCollector(client=mock_client)
        collector.fetch_raw(basin="Han")
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["basin"] == "한강"


class TestKoreaBasins:
    def test_at_least_five_basins(self):
        assert len(KOREA_MAJOR_BASINS) >= 5

    def test_han_basin(self):
        assert KOREA_MAJOR_BASINS["Han"] == "한강"

    def test_nakdong_basin(self):
        assert KOREA_MAJOR_BASINS["Nakdong"] == "낙동강"


class TestKoreaQualityGrades:
    def test_all_grades_present(self):
        expected = {"Ia", "Ib", "II", "III", "IV", "V", "VI"}
        assert expected == set(KOREA_QUALITY_GRADES.keys())

    def test_grade_ia_strictest(self):
        ia = KOREA_QUALITY_GRADES["Ia"]
        assert ia["BOD"] == 1.0
        assert ia["DO"] == 7.5

    def test_bod_increases_with_worse_grade(self):
        grades = ["Ia", "Ib", "II", "III", "IV", "V"]
        bod_values = [KOREA_QUALITY_GRADES[g]["BOD"] for g in grades]
        assert bod_values == sorted(bod_values)


class TestParameterMapKO:
    def test_core_mappings(self):
        assert PARAMETER_MAP_KO["수위"] == "water_level"
        assert PARAMETER_MAP_KO["유량"] == "discharge"
        assert PARAMETER_MAP_KO["pH"] == "pH"
        assert PARAMETER_MAP_KO["DO"] == "DO"
        assert PARAMETER_MAP_KO["BOD"] == "BOD"
        assert PARAMETER_MAP_KO["COD"] == "COD"
        assert PARAMETER_MAP_KO["SS"] == "SS"
        assert PARAMETER_MAP_KO["T-N"] == "Total Nitrogen"
        assert PARAMETER_MAP_KO["T-P"] == "Total Phosphorus"


class TestKoreaDamList:
    def test_at_least_five_dams(self):
        assert len(KOREA_DAM_LIST) >= 5

    def test_chungju_metadata(self):
        dam = KOREA_DAM_LIST["Chungju"]
        assert dam["korean_name"] == "충주댐"
        assert dam["basin"] == "Han"
        assert dam["capacity_mcm"] > 0

    def test_soyang_metadata(self):
        dam = KOREA_DAM_LIST["Soyang"]
        assert dam["korean_name"] == "소양강댐"
        assert dam["completion_year"] == 1973
