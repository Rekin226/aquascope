"""Tests for Taiwan MOENV collector normalisation."""

from datetime import datetime
from unittest.mock import Mock

from aquascope.collectors.taiwan_moenv import TaiwanMOENVCollector
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "sitename": "新店溪-秀朗橋",
        "siteengname": "Xindian River - Xiulang Bridge",
        "county": "新北市",
        "township": "中和區",
        "basin": "淡水河",
        "river": "新店溪",
        "twd97lon": "121.511",
        "twd97lat": "25.001",
        "sampledate": "2025-11-15",
        "sampletime": "09:30",
        "itemname": "溶氧量",
        "itemengabbreviation": "DO",
        "itemvalue": "6.8",
        "itemunit": "mg/L",
    },
    {
        "sitename": "新店溪-秀朗橋",
        "siteengname": "Xindian River - Xiulang Bridge",
        "county": "新北市",
        "township": "中和區",
        "basin": "淡水河",
        "river": "新店溪",
        "twd97lon": "121.511",
        "twd97lat": "25.001",
        "sampledate": "2025-11-15",
        "sampletime": "09:30",
        "itemname": "生化需氧量",
        "itemengabbreviation": "BOD5",
        "itemvalue": "2.3",
        "itemunit": "mg/L",
    },
    # This record should be skipped (ND value)
    {
        "sitename": "test",
        "sampledate": "2025-11-15",
        "sampletime": "10:00",
        "itemname": "大腸桿菌群",
        "itemvalue": "ND",
        "itemunit": "CFU/100mL",
    },
]


def test_moenv_default_fetches_and_parses_river_water_quality():
    # Field names and sample values from the official WQX_P_01 catalog preview.
    # https://data.moenv.gov.tw/dataset/detail/WQX_P_01
    record = {
        "siteid": "1685",
        "sitename": "永安橋",
        "sampledate": "2026-09-21 12:30:59",
        "itemname": "氣溫",
        "itemengabbreviation": "TMP",
        "itemvalue": "33.2",
        "itemunit": "℃",
    }
    client = Mock(get_json=Mock(return_value={"records": [record]}))
    collector = TaiwanMOENVCollector(client=client)

    samples = collector.collect(limit=2)

    client.get_json.assert_called_once_with("WQX_P_01", params={"limit": 2, "offset": 0, "format": "json"})
    [sample] = samples
    assert sample.station_id == "1685"
    assert sample.station_name == "永安橋"
    assert sample.sample_datetime == datetime(2026, 9, 21, 12, 30, 59)
    assert (sample.parameter, sample.value, sample.unit) == ("TMP", 33.2, "℃")


class TestTaiwanMOENVCollector:
    def setup_method(self):
        self.collector = TaiwanMOENVCollector(api_key="test")

    def test_normalise_produces_correct_records(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 2
        assert records[0].sample_datetime == datetime(2025, 11, 15, 9, 30)
        assert records[0].station_id == "新店溪-秀朗橋"

    def test_normalise_maps_parameter_names(self):
        records = self.collector.normalise(SAMPLE_RAW)
        params = {r.parameter for r in records}
        assert "DO" in params
        assert "BOD5" in params

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.TAIWAN_MOENV

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - 25.001) < 0.01
        assert abs(rec.location.longitude - 121.511) < 0.01

    def test_normalise_skips_nd_values(self):
        records = self.collector.normalise(SAMPLE_RAW)
        stations = {r.station_id for r in records}
        assert "test" not in stations
