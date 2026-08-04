"""Tests for the Ireland OPW (waterlevel.ie) hydrometric collector.

fetch_raw/fetch_stations are tested with mocked HTTP; normalise() and the
CSV/datetime parsing helpers are tested directly with hand-built fixture
data shaped like waterlevel.ie's real GeoJSON/CSV responses - no network
calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.ireland_opw import IrelandOPWCollector
from aquascope.schemas.water_data import DataSource

FAKE_STATIONS_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-6.267, 53.35]},
            "properties": {
                "ref": "0000016010",
                "name": "Anner",
                "csv_file": "/data/month/16010_0001.csv",
            },
        },
        {
            # Above the valid republication range (1-41000) - must be skipped
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-8.5, 51.9]},
            "properties": {
                "ref": "0000099999",
                "name": "Out Of Range Station",
                "csv_file": "/data/month/99999_0001.csv",
            },
        },
        {
            # No ref at all - must be skipped without raising
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-7.0, 52.0]},
            "properties": {"name": "No Ref Station"},
        },
    ],
}

CSV_RAW_STYLE = (
    "Datetime,Value\n"
    "2026-01-15 00:00:00,0.452\n"
    "2026-01-15 00:15:00,0.458\n"
    "2026-01-15 00:30:00,\n"  # blank value - must be skipped
)

CSV_SUMMARY_STYLE = "Datetime,Value,Min,Mean,Max\n2025-01-16,,0.171,0.189,0.208\n2025-01-17,0.150,0.139,0.155,0.171\n"


class TestIrelandOPWFetchStations:
    def test_fetch_stations_returns_features(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_GEOJSON
        collector = IrelandOPWCollector(client=mock_client)

        stations = collector.fetch_stations()
        assert len(stations) == 3
        mock_client.get_json.assert_called_once()


class TestIrelandOPWFetchRaw:
    def test_fetch_raw_uses_csv_file_property(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_GEOJSON
        mock_client.get_text.return_value = CSV_RAW_STYLE
        collector = IrelandOPWCollector(client=mock_client)

        rows = collector.fetch_raw()

        # Only the one valid station (ref 16010) should be fetched -
        # the out-of-range and ref-less stations are skipped.
        assert mock_client.get_text.call_count == 1
        called_url = mock_client.get_text.call_args[0][0]
        assert called_url.endswith("/data/month/16010_0001.csv")

        # The blank-value row is dropped by _parse_csv, leaving 2 rows.
        assert len(rows) == 2
        assert rows[0]["station_ref"] == "0000016010"
        assert rows[0]["station_name"] == "Anner"
        assert rows[0]["latitude"] == 53.35
        assert rows[0]["longitude"] == -6.267

    def test_fetch_raw_respects_max_stations(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_GEOJSON
        mock_client.get_text.return_value = CSV_RAW_STYLE
        collector = IrelandOPWCollector(client=mock_client)

        collector.fetch_raw(max_stations=0)
        assert mock_client.get_text.call_count == 0

    def test_fetch_raw_skips_station_on_http_failure(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_GEOJSON
        mock_client.get_text.side_effect = ConnectionError("boom")
        collector = IrelandOPWCollector(client=mock_client)

        rows = collector.fetch_raw()  # must not raise
        assert rows == []

    def test_fetch_raw_accepts_pre_fetched_stations(self):
        mock_client = MagicMock()
        mock_client.get_text.return_value = CSV_RAW_STYLE
        collector = IrelandOPWCollector(client=mock_client)

        collector.fetch_raw(stations=FAKE_STATIONS_GEOJSON["features"])
        mock_client.get_json.assert_not_called()


class TestIrelandOPWParseCSV:
    def test_parses_raw_style_csv(self):
        rows = IrelandOPWCollector._parse_csv(CSV_RAW_STYLE)
        assert len(rows) == 2
        assert rows[0] == {"datetime": "2026-01-15 00:00:00", "value": "0.452"}

    def test_parses_summary_style_csv_skipping_blank_values(self):
        rows = IrelandOPWCollector._parse_csv(CSV_SUMMARY_STYLE)
        assert len(rows) == 1
        assert rows[0]["datetime"] == "2025-01-17"

    def test_unresolvable_header_returns_empty_not_raise(self):
        rows = IrelandOPWCollector._parse_csv("Foo,Bar\n1,2\n")
        assert rows == []

    def test_empty_csv_returns_empty(self):
        rows = IrelandOPWCollector._parse_csv("")
        assert rows == []


class TestIrelandOPWParseDatetime:
    def test_parses_bare_date(self):
        dt = IrelandOPWCollector._parse_datetime("2025-01-16")
        assert dt.year == 2025 and dt.month == 1 and dt.day == 16
        assert dt.tzinfo is None

    def test_parses_full_datetime(self):
        dt = IrelandOPWCollector._parse_datetime("2026-01-15 00:15:00")
        assert dt.hour == 0 and dt.minute == 15

    def test_parses_z_suffix_iso_datetime(self):
        # Regression test: datetime.fromisoformat() only accepts a bare
        # "Z" on Python 3.11+; CI runs 3.10, so this must be handled
        # explicitly rather than passed straight through.
        dt = IrelandOPWCollector._parse_datetime("2026-01-15T00:15:00Z")
        assert dt.hour == 0 and dt.minute == 15
        assert dt.tzinfo is None  # stored tz-naive per codebase convention


class TestIrelandOPWNormalise:
    def setup_method(self):
        self.collector = IrelandOPWCollector()

    def test_normalise_produces_correct_records(self):
        raw = [
            {
                "datetime": "2026-01-15 00:00:00",
                "value": "0.452",
                "station_ref": "0000016010",
                "station_name": "Anner",
                "latitude": 53.35,
                "longitude": -6.267,
            }
        ]
        records = self.collector.normalise(raw)
        assert len(records) == 1
        rec = records[0]
        assert rec.source == DataSource.IRELAND_OPW
        assert rec.station_id == "0000016010"
        assert rec.station_name == "Anner"
        assert rec.water_level == 0.452
        assert rec.unit == "m"
        assert rec.location.latitude == 53.35
        assert rec.location.longitude == -6.267

    def test_normalise_skips_missing_value(self):
        raw = [{"datetime": "2026-01-15 00:00:00", "station_ref": "X"}]
        records = self.collector.normalise(raw)
        assert records == []

    def test_normalise_skips_invalid_value(self):
        raw = [{"datetime": "2026-01-15 00:00:00", "value": "not_a_number", "station_ref": "X"}]
        records = self.collector.normalise(raw)
        assert records == []

    def test_normalise_handles_missing_location(self):
        raw = [{"datetime": "2026-01-15 00:00:00", "value": "0.5", "station_ref": "X"}]
        records = self.collector.normalise(raw)
        assert len(records) == 1
        assert records[0].location is None

    def test_normalise_empty_input(self):
        assert self.collector.normalise([]) == []


class TestIrelandOPWCollectRoundTrip:
    def test_collect_end_to_end(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = FAKE_STATIONS_GEOJSON
        mock_client.get_text.return_value = CSV_RAW_STYLE
        collector = IrelandOPWCollector(client=mock_client)

        records = collector.collect()
        assert len(records) == 2
        assert all(r.source == DataSource.IRELAND_OPW for r in records)
