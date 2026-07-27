"""Tests for the NOAA NWPS collector."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aquascope.collectors import NOAA_NWPSCollector
from aquascope.schemas.water_data import DataSource, StreamflowReading

SAMPLE_OBSERVED_RAW = [
    {
        "validTime": "2026-06-28T14:45:00Z",
        "generatedTime": "2026-06-28T15:40:05Z",
        "primary": 5.03,
        "secondary": 20.6,
        "secondaryUnit": "kcfs",
    }
]

SAMPLE_DISCOVERY_RAW = {
    "gauges": [
        {
            "lid": "AGSI4",
            "name": "Skunk River at Augusta",
            "rfc": {
                "abbreviation": "NCRFC",
                "name": "North Central River Forecast Center",
            },
        }
    ]
}

class TestNOAANWPSCollectorInit:
    def test_initialization(self):
        collector = NOAA_NWPSCollector()
        assert collector is not None

    def test_collector_name(self):
        collector = NOAA_NWPSCollector()
        assert collector.name == "noaa_nwps"

class TestNOAANWPSCollectorURLBuilder:
    # Gauge URLs
    def test_expected_gauge_url(self):
        collector = NOAA_NWPSCollector()
        url = collector._build_gauge_url("AGSI4")
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges/AGSI4"

    def test_unexpected_gauge_url(self):
        collector = NOAA_NWPSCollector()
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("A")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("AGSI45")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("")

    # Observation URLs
    def test_expected_observation_url(self):
        collector = NOAA_NWPSCollector()
        url = collector._build_observed_url("AGSI4")
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges/AGSI4/stageflow/observed"

    def test_unexpected_observation_urls(self):
        collector = NOAA_NWPSCollector()
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("A")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("AGSI45")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("")

    # Discovery URLs
    def test_expected_discovery_url(self):
        collector = NOAA_NWPSCollector()
        bbox = (-95, 38, -89, 41)
        url = collector._build_discovery_url(bbox)
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges?bbox.xmin=-95&bbox.ymin=38&bbox.xmax=-89&bbox.ymax=41&srid=EPSG_4326"

    def test_discovery_url_param_order_is_stable(self):
        collector = NOAA_NWPSCollector()
        url = collector._build_discovery_url((-95, 38, -89, 41))
        assert url.index("bbox.xmin=") < url.index("bbox.ymin=")
        assert url.index("bbox.ymin=") < url.index("bbox.xmax=")
        assert url.index("bbox.xmax=") < url.index("bbox.ymax=")

    def test_unexpected_discovery_urls(self):
        collector = NOAA_NWPSCollector()
        with pytest.raises(ValueError, match="bbox search parameters require 4 values"):
            collector._build_discovery_url((-95, 38, -89))
        with pytest.raises(ValueError, match="bbox search parameters require 4 values"):
            collector._build_discovery_url(())
        with pytest.raises(ValueError, match="bbox search parameters require 4 values"):
            collector._build_discovery_url(None)
        with pytest.raises(ValueError, match="NOAA-NWPS: bbox.xmin cannot be greater than bbox.xmax."):
            collector._build_discovery_url((-89.0, 38.0, -95.0, 41.0))
        with pytest.raises(ValueError, match="NOAA-NWPS: bbox.ymin cannot be greater than bbox.ymax."):
            collector._build_discovery_url((-95.0, 41.0, -89.0, 38.0))
        with pytest.raises(ValueError, match="bbox x values"):
            collector._build_discovery_url((1.0, 38.0, 2.0, 41.0))
        with pytest.raises(ValueError, match="bbox y values"):
            collector._build_discovery_url((-95.0, -41.0, -89.0, -38.0))


class TestNOAANWPSCollectorDiscovery:
    def test_extract_name_lid_valid_payload(self):
        collector = NOAA_NWPSCollector()
        pairs = collector._extract_name_lid(SAMPLE_DISCOVERY_RAW)
        assert pairs == [("AGSI4", "Skunk River at Augusta")]

    def test_extract_name_lid_non_dict_returns_empty(self):
        collector = NOAA_NWPSCollector()
        assert collector._extract_name_lid([{"gauges": []}]) == []

    def test_extract_name_lid_missing_or_non_list_gauges_returns_empty(self):
        collector = NOAA_NWPSCollector()
        assert collector._extract_name_lid({}) == []
        assert collector._extract_name_lid({"gauges": "invalid"}) == []

    def test_extract_name_lid_skips_malformed_entries(self):
        collector = NOAA_NWPSCollector()
        payload = {
            "gauges": [
                {"lid": "GOOD1", "name": "Good"},
                {"lid": "", "name": "Empty lid"},
                {"lid": "LID02"},
                {"name": "Missing lid"},
                "not-a-dict",
            ]
        }
        assert collector._extract_name_lid(payload) == [("GOOD1", "Good")]

class TestNOAANWPSCollectorFetchRaw:
    def test_fetch_raw_with_lid_and_bbox_raises(self):
        collector = NOAA_NWPSCollector()
        with pytest.raises(ValueError, match="Provide only one of 'lid' or 'bbox'."):
            collector.fetch_raw(lid="AGSI4", bbox=(-95, 38, -89, 41))

    def test_fetch_raw_with_neither_raises(self):
        collector = NOAA_NWPSCollector()
        with pytest.raises(ValueError, match="One of 'lid' or 'bbox' is required"):
            collector.fetch_raw()

    def test_fetch_raw_lid_returns_combined_dictionary(self):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            {
                "lid": "AGSI4",
                "name": "Skunk River at Augusta",
                "latitude": 40.0,
                "longitude": -91.0,
            },
            {"data": SAMPLE_OBSERVED_RAW},
        ]
        collector = NOAA_NWPSCollector(client=mock_client)

        result = collector.fetch_raw(lid="AGSI4")

        assert result["source"] == DataSource.NOAA_NWPS
        assert result["station_id"] == "AGSI4"
        assert result["station_name"] == "Skunk River at Augusta"
        assert result["source_type"] == "in_situ"
        assert result["unit"] == "m3/s"
        assert isinstance(result["data_container"], list)

    def test_fetch_raw_bbox_prints_and_returns_empty(self, capsys):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {
            "gauges": [
                {"lid": "AGSI4", "name": "Skunk River at Augusta"},
                {"lid": "ANAW1", "name": "Snake River near Anatone"},
            ]
        }
        collector = NOAA_NWPSCollector(client=mock_client)

        result = collector.fetch_raw(bbox=(-95, 38, -89, 41))

        out = capsys.readouterr().out
        assert "AGSI4\tSkunk River at Augusta" in out
        assert "ANAW1\tSnake River near Anatone" in out
        assert result == []

    def test_fetch_raw_bbox_no_stations_returns_empty(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"gauges": []}
        collector = NOAA_NWPSCollector(client=mock_client)

        result = collector.fetch_raw(bbox=(-95, 38, -89, 41))
        assert result == []

class TestNOAANWPSCollectorHelpers:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (1, "kcfs", 28.316846592),
            (1, "cfs", 0.028316846592),
            (1, "m3/s", 1.0),
            (1, "cms", 1.0),
            (1, "ft", 0.3048),
        ],
    )
    def test_to_discharge_cms_known_units(self, value, unit, expected):
        collector = NOAA_NWPSCollector()
        assert collector._to_discharge_cms(value, unit) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "unit"),
        [
            (None, "kcfs"),
            ("not-a-number", "kcfs"),
            (1, None),
            (1, "unknown"),
        ],
    )
    def test_to_discharge_cms_invalid_returns_none(self, value, unit):
        collector = NOAA_NWPSCollector()
        assert collector._to_discharge_cms(value, unit) is None

    def test_parse_nwps_datetime_handles_z(self):
        parsed = NOAA_NWPSCollector._parse_nwps_datetime("2026-06-28T15:40:05Z")
        assert parsed == datetime(2026, 6, 28, 15, 40, 5, tzinfo=timezone.utc)

    def test_parse_nwps_datetime_handles_naive_iso(self):
        parsed = NOAA_NWPSCollector._parse_nwps_datetime("2026-06-28T15:40:05")
        assert parsed == datetime(2026, 6, 28, 15, 40, 5)

    @pytest.mark.parametrize("value", ["not-a-date", 123, None, ""])
    def test_parse_nwps_datetime_invalid_returns_none(self, value):
        assert NOAA_NWPSCollector._parse_nwps_datetime(value) is None

class TestNOAANWPSCollectorNormalise:
    def test_normalise_produces_streamflow_records(self):
        collector = NOAA_NWPSCollector()
        combined = {
            "source": DataSource.NOAA_NWPS,
            "station_id": "ANAW1",
            "station_name": "Snake River (WA) near Anatone",
            "location": None,
            "data_container": SAMPLE_OBSERVED_RAW,
            "source_type": "in_situ",
            "unit": "m3/s",
        }

        records = collector.normalise(combined)

        assert isinstance(records, list)
        assert len(records) == 1
        assert isinstance(records[0], StreamflowReading)
        first = records[0]
        assert first.source == DataSource.NOAA_NWPS
        assert first.station_id == "ANAW1"
        assert first.station_name == "Snake River (WA) near Anatone"
        assert first.reading_datetime is not None
        assert first.discharge_cms == pytest.approx(20.6 * 28.316846592)
        assert first.unit == "m3/s"

    def test_normalise_skips_invalid_datetime(self):
        collector = NOAA_NWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {
                    "generatedTime": "invalid",
                    "secondary": 20.6,
                    "secondaryUnit": "kcfs",
                }
            ],
        }

        assert collector.normalise(combined) == []

    def test_normalise_skips_missing_or_sentinel_secondary(self):
        collector = NOAA_NWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": -999, "secondaryUnit": "kcfs"},
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": None, "secondaryUnit": "kcfs"},
            ],
        }

        assert collector.normalise(combined) == []

    def test_normalise_missing_or_non_list_data_container(self):
        collector = NOAA_NWPSCollector()
        assert collector.normalise({"station_id": "ANAW1"}) == []
        assert collector.normalise({"station_id": "ANAW1", "data_container": "bad"}) == []

    def test_normalise_skips_non_dict_entries(self):
        collector = NOAA_NWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                "bad-entry",
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": 20.6, "secondaryUnit": "kcfs"},
            ],
        }

        records = collector.normalise(combined)
        assert len(records) == 1

    def test_normalise_list_input_flattens_station_records(self):
        collector = NOAA_NWPSCollector()
        stations = [
            {
                "station_id": "A1",
                "data_container": [
                    {"generatedTime": "2026-06-28T15:40:05Z", "secondary": 1, "secondaryUnit": "kcfs"}
                ],
            },
            {
                "station_id": "A2",
                "data_container": [
                    {"generatedTime": "2026-06-28T16:40:05Z", "secondary": 2, "secondaryUnit": "kcfs"}
                ],
            },
        ]

        records = collector.normalise(stations)
        assert len(records) == 2


class TestNOAANWPSCollectorEdgeCases:
    def test_build_combined_dictionary_missing_coordinates_falls_back(self, caplog):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            {"lid": "AGSI4", "name": "Skunk River at Augusta", "latitude": None, "longitude": None},
            {"data": []},
        ]
        collector = NOAA_NWPSCollector(client=mock_client)

        with caplog.at_level("WARNING"):
            result = collector._build_combined_dictionary("AGSI4")

        assert result["location"].latitude == 0.0
        assert result["location"].longitude == 0.0
        assert "Missing latitude or longitude" in caplog.text


class TestNOAANWPSCollectorLogging:
    def test_extract_name_lid_logs_for_malformed_payload(self, caplog):
        collector = NOAA_NWPSCollector()
        with caplog.at_level("WARNING"):
            collector._extract_name_lid("not-a-dict")
        assert "Discovery raw data is not a dictionary" in caplog.text

    def test_extract_name_lid_logs_for_non_list_gauges(self, caplog):
        collector = NOAA_NWPSCollector()
        with caplog.at_level("WARNING"):
            collector._extract_name_lid({"gauges": "bad"})
        assert "'gauges' key is not a list" in caplog.text

    def test_to_discharge_logs_for_unknown_unit(self, caplog):
        collector = NOAA_NWPSCollector()
        with caplog.at_level("WARNING"):
            value = collector._to_discharge_cms(1, "weird")
        assert value is None
        assert "Unknown NWPS unit" in caplog.text

    def test_normalise_logs_for_sentinel_secondary(self, caplog):
        collector = NOAA_NWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": -999, "secondaryUnit": "kcfs"}
            ],
        }

        with caplog.at_level("WARNING"):
            records = collector.normalise(combined)

        assert records == []
        assert "Skipping" in caplog.text
