"""Tests for the NOAA NWPS collector."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from aquascope.collectors import NOAANWPSCollector
from aquascope.schemas.water_data import DataSource, StreamflowReading

SAMPLE_OBSERVED_RAW = [
    {
        "pedts": "HGIRG",
        "issuedTime": "2026-07-28T13:15:00Z",
        "wfo": "PDT",
        "timeZone": "PST8PDT",
        "primaryName": "Stage",
        "primaryUnits": "ft",
        "secondaryName": "Flow",
        "secondaryUnits": "kcfs",
        "data": [
            {
                "validTime": "2026-06-28T14:45:00Z",
                "generatedTime": "2026-06-28T15:40:05Z",
                "primary": 5.03,
                "secondary": 20.6
            }
        ]
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
            "latitude": 46.0971,
            "longitude": -116.9776,
        }
    ]
}

class TestNOAANWPSCollectorInit:
    def test_initialization(self):
        collector = NOAANWPSCollector()
        assert collector is not None

    def test_collector_name(self):
        collector = NOAANWPSCollector()
        assert collector.name == "noaa_nwps"

class TestNOAANWPSCollectorURLBuilder:
    # Gauge URLs
    def test_expected_gauge_url(self):
        collector = NOAANWPSCollector()
        url = collector._build_gauge_url("AGSI4")
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges/AGSI4"

    def test_unexpected_gauge_url(self):
        collector = NOAANWPSCollector()
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("A")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("AGSI45")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_gauge_url("")

    # Observation URLs
    def test_expected_observation_url(self):
        collector = NOAANWPSCollector()
        url = collector._build_observed_url("AGSI4")
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges/AGSI4/stageflow/observed"

    def test_unexpected_observation_urls(self):
        collector = NOAANWPSCollector()
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("A")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("AGSI45")
        with pytest.raises(ValueError, match="NOAA-NWPS: LID must be exactly 5 characters long."):
            collector._build_observed_url("")

    # Discovery URLs
    def test_expected_discovery_url(self):
        collector = NOAANWPSCollector()
        bbox = (-95, 38, -89, 41)
        url = collector._build_discovery_url(bbox)
        assert url == "https://api.water.noaa.gov/nwps/v1/gauges?bbox.xmin=-95&bbox.ymin=38&bbox.xmax=-89&bbox.ymax=41&srid=EPSG_4326"

    def test_discovery_url_param_order_is_stable(self):
        collector = NOAANWPSCollector()
        url = collector._build_discovery_url((-95, 38, -89, 41))
        assert url.index("bbox.xmin=") < url.index("bbox.ymin=")
        assert url.index("bbox.ymin=") < url.index("bbox.xmax=")
        assert url.index("bbox.xmax=") < url.index("bbox.ymax=")

    def test_unexpected_discovery_urls(self):
        collector = NOAANWPSCollector()
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
    def test_extract_lid_valid_payload(self):
        collector = NOAANWPSCollector()
        lids = collector._extract_lid(SAMPLE_DISCOVERY_RAW)
        assert lids == [("AGSI4")]

    def test_extract_lid_non_dict_returns_empty(self):
        collector = NOAANWPSCollector()
        assert collector._extract_lid([{"gauges": []}]) == []

    def test_extract_lid_missing_or_non_list_gauges_returns_empty(self):
        collector = NOAANWPSCollector()
        assert collector._extract_lid({}) == []
        assert collector._extract_lid({"gauges": "invalid"}) == []

    def test_extract_lid_skips_malformed_entries(self):
        collector = NOAANWPSCollector()
        payload = {
            "gauges": [
                {"lid": "GOOD1", "name": "Good"},
                {"lid": "", "name": "Empty lid"},
                {"lid": "LID02"},
                {"name": "Missing lid"},
                "not-a-dict",
            ]
        }
        assert collector._extract_lid(payload) == ['GOOD1', 'LID02']

class TestNOAANWPSCollectorFetchRaw:
    def test_fetch_raw_with_lid_and_bbox_raises(self):
        collector = NOAANWPSCollector()
        with pytest.raises(ValueError, match="Provide only one of 'lid' or 'bbox'."):
            collector.fetch_raw(lid="AGSI4", bbox=(-95, 38, -89, 41))

    def test_fetch_raw_with_neither_raises(self):
        collector = NOAANWPSCollector()
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

        collector = NOAANWPSCollector(client=mock_client)
        result = collector.fetch_raw(lid="AGSI4")
        assert isinstance(result["data_container"], list)

    def test_fetch_raw_bbox_no_stations_returns_empty(self):
        mock_client = MagicMock()
        mock_client.get_json.return_value = {"gauges": []}
        collector = NOAANWPSCollector(client=mock_client)

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
            (1, "ft", None),
        ],
    )
    def test_to_discharge_cms_known_units(self, value, unit, expected):
        collector = NOAANWPSCollector()
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
        collector = NOAANWPSCollector()
        assert collector._to_discharge_cms(value, unit) is None

    def test_parse_nwps_datetime_handles_z(self):
        parsed = NOAANWPSCollector._parse_nwps_datetime("2026-06-28T15:40:05Z")
        assert parsed == datetime(2026, 6, 28, 15, 40, 5, tzinfo=None)

    def test_parse_nwps_datetime_handles_naive_iso(self):
        parsed = NOAANWPSCollector._parse_nwps_datetime("2026-06-28T15:40:05")
        assert parsed == datetime(2026, 6, 28, 15, 40, 5)

    @pytest.mark.parametrize("value", ["not-a-date", 123, None, ""])
    def test_parse_nwps_datetime_invalid_returns_none(self, value):
        assert NOAANWPSCollector._parse_nwps_datetime(value) is None

class TestNOAANWPSCollectorNormalise:
    def test_normalise_produces_streamflow_records(self):
        collector = NOAANWPSCollector()
        combined = {
            "source": DataSource.NOAA_NWPS,
            "station_id": "ANAW1",
            "station_name": "Snake River (WA) near Anatone",
            "latitude": None,
            "longitude": None,
            "data_container": [{"validTime": "2026-06-28T14:45:00Z", "generatedTime": "2026-06-28T15:40:05Z",
                                "primary": 5.03, "secondary": 20.6}],
            "source_type": "in_situ",
            "unit": "kcfs",
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
        collector = NOAANWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {
                    "validTimeKey": "invalid",
                    "secondary": 20.6,
                    "secondaryUnits": "kcfs",
                }
            ],
        }

        assert collector.normalise(combined) == []

    def test_normalise_skips_missing_or_sentinel_secondary(self):
        collector = NOAANWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": -999, "secondaryUnits": "kcfs"},
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": None, "secondaryUnits": "kcfs"},
            ],
        }

        assert collector.normalise(combined) == []

    def test_normalise_missing_or_non_list_data_container(self):
        collector = NOAANWPSCollector()
        assert collector.normalise({"station_id": "ANAW1"}) == []
        assert collector.normalise({"station_id": "ANAW1", "data_container": "bad"}) == []

    def test_normalise_skips_non_dict_entries(self):
        collector = NOAANWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "unit": "kcfs",
            "data_container": [
                "bad-entry",
                {"validTime": "2026-06-28T15:40:05Z", "secondary": 20.6},
            ],
        }

        records = collector.normalise(combined)
        assert len(records) == 1


class TestNOAANWPSCollectorEdgeCases:
    def test_build_combined_dictionary_missing_coordinates_falls_back(self, caplog):
        mock_client = MagicMock()
        mock_client.get_json.side_effect = [
            {"lid": "AGSI4", "name": "Skunk River at Augusta", "latitude": None, "longitude": None},
            {"data": []},
        ]
        collector = NOAANWPSCollector(client=mock_client)

        with caplog.at_level("WARNING"):
            result = collector._build_combined_dictionary("AGSI4")

        assert result["location"].latitude == 0.0
        assert result["location"].longitude == 0.0
        assert "Missing latitude or longitude" in caplog.text


class TestNOAANWPSCollectorLogging:
    def test_extract_name_lid_logs_for_malformed_payload(self, caplog):
        collector = NOAANWPSCollector()
        with caplog.at_level("WARNING"):
            collector._extract_lid("not-a-dict")
        assert "Discovery raw data is not a dictionary" in caplog.text

    def test_extract_name_lid_logs_for_non_list_gauges(self, caplog):
        collector = NOAANWPSCollector()
        with caplog.at_level("WARNING"):
            collector._extract_lid({"gauges": "bad"})
        assert "'gauges' key is not a list" in caplog.text

    def test_to_discharge_logs_for_unknown_unit(self, caplog):
        collector = NOAANWPSCollector()
        with caplog.at_level("WARNING"):
            value = collector._to_discharge_cms(1, "weird")
        assert value is None
        assert "Unknown NWPS unit" in caplog.text

    def test_normalise_logs_for_sentinel_secondary(self, caplog):
        collector = NOAANWPSCollector()
        combined = {
            "station_id": "ANAW1",
            "data_container": [
                {"generatedTime": "2026-06-28T15:40:05Z", "secondary": -999, "secondaryUnits": "kcfs"}
            ],
        }

        with caplog.at_level("WARNING"):
            records = collector.normalise(combined)

        assert records == []
        assert "skipping" in caplog.text
