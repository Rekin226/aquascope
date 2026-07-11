"""Tests for the GRDC (Global Runoff Data Centre) water discharge collector."""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.grdc import GRDCCollector
from aquascope.schemas.water_data import DataSource

SAMPLE_RAW = [
    {
        "station_id": "6435060",
        "station_name": "TEST STATION ON RIVER",
        "latitude": 51.234,
        "longitude": 7.891,
        "date": "2020-01-15",
        "discharge": 45.2,
        "source_type": "in_situ",
    },
    {
        "station_id": "RSEG_00123",
        "station_name": None,
        "latitude": -3.5,
        "longitude": 29.1,
        "date": "2021-06-01",
        "discharge": 812.7,
        "source_type": "satellite",
        "uncertainty": 55.3,
    },
]

SAMPLE_RAW_MALFORMED = [
    # Missing required "discharge" key
    {
        "station_id": "BAD01",
        "date": "2020-01-01",
        "source_type": "in_situ",
    },
    # Non-numeric discharge value
    {
        "station_id": "BAD02",
        "date": "2020-01-01",
        "discharge": "not_a_number",
        "source_type": "in_situ",
    },
]

STATION_FILE_TEXT = """# GRDC STATION DATA FILE
# Station: TEST STATION ON RIVER
# GRDC-No.: 6435060
# Latitude: 51.234
# Longitude: 7.891
#************************************************************
YYYY-MM-DD;HH:MM;Value
2020-01-15;00:00;   45.200
2020-01-16;00:00;   47.800
2020-01-17;00:00;  -999.000
2020-01-18;00:00; -9998.000
"""


class TestGRDCInit:
    def setup_method(self):
        self.collector = GRDCCollector()

    def test_collector_name(self):
        assert self.collector.name == "grdc"


class TestGRDCNormalise:
    def setup_method(self):
        self.collector = GRDCCollector()

    def test_normalise_produces_correct_count(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert len(records) == 2

    def test_normalise_sets_correct_source(self):
        records = self.collector.normalise(SAMPLE_RAW)
        for r in records:
            assert r.source == DataSource.GRDC

    def test_normalise_tags_in_situ(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].source_type == "in_situ"

    def test_normalise_tags_satellite(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[1].source_type == "satellite"

    def test_normalise_carries_uncertainty_for_satellite_only(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].uncertainty_cms is None
        assert records[1].uncertainty_cms == 55.3

    def test_normalise_parses_location(self):
        records = self.collector.normalise(SAMPLE_RAW)
        rec = records[0]
        assert rec.location is not None
        assert abs(rec.location.latitude - 51.234) < 0.001
        assert abs(rec.location.longitude - 7.891) < 0.001

    def test_normalise_discharge_value(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].discharge_cms == 45.2
        assert records[0].unit == "m3/s"

    def test_normalise_preserves_station_id(self):
        records = self.collector.normalise(SAMPLE_RAW)
        assert records[0].station_id == "6435060"

    def test_normalise_skips_malformed_rows(self):
        records = self.collector.normalise(SAMPLE_RAW_MALFORMED)
        assert records == []

    def test_normalise_empty_input(self):
        records = self.collector.normalise([])
        assert records == []


class TestGRDCFetchRaw:
    def setup_method(self):
        self.collector = GRDCCollector()

    def test_fetch_raw_invalid_source_type(self):
        try:
            self.collector.fetch_raw(source_type="bogus")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "source_type" in str(exc)

    def test_fetch_raw_no_files_in_zenodo_record(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"files": []}
        collector = GRDCCollector(client=mock_client)
        result = collector.fetch_raw(source_type="in_situ")
        assert result == []

    def test_fetch_raw_no_csv_in_darus_dataset(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"data": {"latestVersion": {"files": []}}}
        collector = GRDCCollector(client=mock_client)
        result = collector.fetch_raw(source_type="satellite")
        assert result == []


class TestGRDCParseStationFile:
    def test_parses_metadata_and_rows(self):
        rows = GRDCCollector._parse_grdc_station_file(STATION_FILE_TEXT, "6435060_Q_Day.Cmd.txt")
        # The -999 sentinel row should be dropped
        assert len(rows) == 2
        assert rows[0]["station_id"] == "6435060"
        assert rows[0]["station_name"] == "TEST STATION ON RIVER"
        assert rows[0]["latitude"] == 51.234
        assert rows[0]["discharge"] == 45.2
        assert rows[0]["source_type"] == "in_situ"

    def test_drops_negative_sentinel_values(self):
        rows = GRDCCollector._parse_grdc_station_file(STATION_FILE_TEXT, "6435060_Q_Day.Cmd.txt")
        discharges = [r["discharge"] for r in rows]
        assert all(d >= 0 for d in discharges)

    def test_empty_text_returns_no_rows(self):
        rows = GRDCCollector._parse_grdc_station_file("", "empty.txt")
        assert rows == []
