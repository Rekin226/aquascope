"""Tests for the Japan MLIT water data collector."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock

import pytest

from aquascope.collectors.base import CollectorError
from aquascope.collectors.japan_mlit import (
    JAPAN_RIVER_SYSTEMS,
    PARAMETER_MAP_JA,
    PREFECTURE_CODES,
    QUALITY_STANDARDS_JAPAN,
    JapanMLITCollector,
    month_range,
    parse_datetime,
    parse_export,
)
from aquascope.schemas.water_data import DataSource

SAMPLE_CSV = (
    "日時,データ,属性,暫定フラグ\r\n"
    "2019/08/01 01:00,0.06,0,\r\n"
    "2019/08/01 02:00,-9999,1,\r\n"
    "2019/08/01 03:00,0.07,0,*\r\n"
    "2019/08/01 04:00,0.09,3,\r\n"
    "2019/08/01 24:00,0.10,0,\r\n"
)

SAMPLE_STATION = {
    "id": "303031283301030",
    "name_data": "高砂",
    "suikei_name_data": "利根川",
    "wlatitude": 36.133,
    "wlongitude": 139.717,
}

SAMPLE_RAW = [
    {
        "station_id": "305041281005030",
        "station_name": "利根川 栗橋",
        "parameter": "water_level",
        "value": 3.45,
        "unit": "m",
        "datetime": "2023-07-15T10:00:00",
        "latitude": 36.133,
        "longitude": 139.717,
        "river": "利根川",
    },
    {
        "station_id": "305041281005031",
        "station_name": "石狩川 旭橋",
        "parameter": "discharge",
        "value": 125.8,
        "unit": "m3/s",
        "datetime": "2023-07-15T12:00:00",
        "latitude": 43.770,
        "longitude": 142.370,
        "river": "石狩川",
    },
    {
        "station_id": "WQ001",
        "station_name": "淀川 枚方",
        "parameter": "pH",
        "value": 7.2,
        "unit": "",
        "datetime": "2023-08-01T09:00:00",
        "latitude": 34.816,
        "longitude": 135.649,
        "river": "淀川",
    },
    {
        "station_id": "WQ002",
        "station_name": "信濃川 大手大橋",
        "parameter": "DO",
        "value": 8.5,
        "unit": "mg/L",
        "datetime": "2023-08-01T10:00:00",
        "latitude": 37.900,
        "longitude": 139.020,
        "river": "信濃川",
    },
]

SAMPLE_RAW_SKIP = [
    {
        "station_id": "SKIP01",
        "station_name": "test",
        "parameter": "water_level",
        "value": "ND",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
    {
        "station_id": "SKIP02",
        "station_name": "test",
        "parameter": "water_level",
        "value": "",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
    {
        "station_id": "SKIP03",
        "station_name": "test",
        "parameter": "water_level",
        "value": "-9999",
        "unit": "m",
        "datetime": "2023-01-01T00:00:00",
    },
]


def make_zip(csv_text: str, name: str = "303031283301030_hv.csv") -> bytes:
    """Build a Shift_JIS export the way the site serves one."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv_text.encode("cp932"))
    return buffer.getvalue()


class TestJapanMLITInit:
    def setup_method(self):
        self.collector = JapanMLITCollector()

    def test_collector_name(self):
        assert self.collector.name == "japan_mlit"

    def test_base_url_is_the_spa_api(self):
        assert self.collector.BASE_URL == "https://www1.river.go.jp/water-inf-app/api/"


class TestParseDatetime:
    def test_plain_timestamp(self):
        assert parse_datetime("2019/08/01 01:00").isoformat() == "2019-08-01T01:00:00"

    def test_24_00_rolls_over_to_the_next_day(self):
        assert parse_datetime("2019/08/31 24:00").isoformat() == "2019-09-01T00:00:00"

    def test_empty_is_none(self):
        assert parse_datetime("") is None
        assert parse_datetime("   ") is None

    def test_garbage_is_none(self):
        assert parse_datetime("not a date") is None


class TestMonthRange:
    def test_single_month(self):
        assert month_range("2019-08-01", None) == ["2019-08"]

    def test_range_spans_months(self):
        assert month_range("2019-01-01", "2019-03-31") == ["2019-01", "2019-02", "2019-03"]

    def test_range_crosses_year(self):
        assert month_range("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]

    def test_reversed_range_raises(self):
        with pytest.raises(ValueError, match="before"):
            month_range("2020-01-01", "2019-01-01")

    def test_no_dates_defaults_to_this_month(self):
        months = month_range(None, None)
        assert len(months) == 1
        assert len(months[0]) == 7


class TestParseExport:
    def test_reads_rows_and_skips_missing_values(self):
        rows = parse_export(make_zip(SAMPLE_CSV), station=SAMPLE_STATION, parameter="water_level", unit="m")
        values = [row["value"] for row in rows]
        assert values == [0.06, 0.07, 0.10]

    def test_skips_invalid_attribute_codes(self):
        rows = parse_export(make_zip(SAMPLE_CSV), station=SAMPLE_STATION, parameter="water_level", unit="m")
        assert all(row["attribute"] == "0" for row in rows)

    def test_marks_provisional_rows(self):
        rows = parse_export(make_zip(SAMPLE_CSV), station=SAMPLE_STATION, parameter="water_level", unit="m")
        provisional = [row["value"] for row in rows if row["provisional"]]
        assert provisional == [0.07]

    def test_carries_station_metadata(self):
        rows = parse_export(make_zip(SAMPLE_CSV), station=SAMPLE_STATION, parameter="water_level", unit="m")
        assert rows[0]["station_id"] == "303031283301030"
        assert rows[0]["station_name"] == "高砂"
        assert rows[0]["river"] == "利根川"
        assert rows[0]["latitude"] == 36.133

    def test_empty_csv_yields_no_rows(self):
        body = make_zip("日時,データ,属性,暫定フラグ\r\n")
        rows = parse_export(body, station=SAMPLE_STATION, parameter="water_level", unit="m")
        assert rows == []

    def test_non_zip_body_yields_no_rows(self):
        assert parse_export(b"<html>403</html>", station=SAMPLE_STATION, parameter="water_level", unit="m") == []

    def test_zip_without_csv_yields_no_rows(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("凡例.txt", "00:00は前日の24:00とする".encode("cp932"))
        rows = parse_export(buffer.getvalue(), station=SAMPLE_STATION, parameter="water_level", unit="m")
        assert rows == []


class TestJapanMLITNormalise:
    def setup_method(self):
        self.collector = JapanMLITCollector()

    def test_normalise_produces_correct_count(self):
        assert len(self.collector.normalise(SAMPLE_RAW)) == 4

    def test_normalise_keeps_parameter_names(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].parameter == "water_level"
        assert records[1].parameter == "discharge"

    def test_normalise_sets_correct_source(self):
        for record in self.collector.normalise(SAMPLE_RAW):
            assert record.source == DataSource.JAPAN_MLIT

    def test_normalise_parses_location(self):
        record = self.collector.normalise(SAMPLE_RAW)[0]
        assert record.location is not None
        assert abs(record.location.latitude - 36.133) < 0.01
        assert abs(record.location.longitude - 139.717) < 0.01

    def test_normalise_maps_river_system(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].basin == "Tone"
        assert records[1].basin == "Ishikari"

    def test_normalise_preserves_station_id(self):
        assert self.collector.normalise(SAMPLE_RAW)[0].station_id == "305041281005030"

    def test_normalise_skips_nd_empty_and_sentinel(self):
        assert self.collector.normalise(SAMPLE_RAW_SKIP) == []

    def test_normalise_empty_input(self):
        assert self.collector.normalise([]) == []

    def test_normalise_keeps_passthrough_params(self):
        params = {record.parameter for record in self.collector.normalise(SAMPLE_RAW)}
        assert "pH" in params
        assert "DO" in params


class TestJapanMLITFetchRaw:
    def test_refuses_and_names_the_robots_disallow(self):
        client = MagicMock()
        with pytest.raises(CollectorError) as exc_info:
            JapanMLITCollector(client=client).fetch_raw(prefecture="Tokyo")

        message = str(exc_info.value)
        assert "robots.txt disallows /water-inf-app/" in message
        assert "水防災オープンデータ" in message
        assert exc_info.value.source == "japan_mlit"
        assert "water-inf-app" in exc_info.value.url
        # The site is never called, not even the station search.
        client.get_json.assert_not_called()
        client.get_bytes.assert_not_called()

    def test_refuses_a_station_id_too(self):
        client = MagicMock()
        with pytest.raises(CollectorError):
            JapanMLITCollector(client=client).fetch_raw(station_id="303031283301030")
        client.get_bytes.assert_not_called()


class TestPrefectureCodes:
    def test_tokyo_code(self):
        assert PREFECTURE_CODES["Tokyo"] == "1301"

    def test_osaka_code(self):
        assert PREFECTURE_CODES["Osaka"] == "2701"

    def test_hokkaido_code(self):
        assert PREFECTURE_CODES["Hokkaido"] == "0101"

    def test_aichi_code(self):
        assert PREFECTURE_CODES["Aichi"] == "2301"

    def test_fukuoka_code(self):
        assert PREFECTURE_CODES["Fukuoka"] == "4001"

    def test_all_codes_are_four_digits(self):
        assert all(len(code) == 4 and code.isdigit() for code in PREFECTURE_CODES.values())

    def test_covers_all_forty_seven_prefectures(self):
        assert len(PREFECTURE_CODES) == 47


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
        bod_values = [QUALITY_STANDARDS_JAPAN[grade]["BOD"] for grade in grades]
        assert bod_values == sorted(bod_values)
