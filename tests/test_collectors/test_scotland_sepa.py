"""Tests for the Scotland SEPA hydrometric time-series collector.

Station reference data:

- 133171 "Newton Stewart" -- station number, coordinates, flow series,
  and river-level series confirmed against the live SEPA KiWIS API.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aquascope.collectors.scotland_sepa import (
    FLOW_PARAMETER,
    LEVEL_PARAMETER,
    PARAMETER_UNITS,
    ScotlandSepaCollector,
)
from aquascope.schemas.water_data import (
    DataSource,
    StreamflowReading,
    WaterLevelReading,
)

TS_LIST_FLOW_RESPONSE = [
    [
        "station_no",
        "station_name",
        "station_latitude",
        "station_longitude",
        "stationparameter_name",
        "ts_name",
        "ts_id",
        "ts_path",
    ],
    [
        "133171",
        "Newton Stewart",
        "54.95726338",
        "-4.480525846",
        "Flow",
        "15minute",
        "67947010",
        "1/133171/Q/15m.Cmd",
    ],
]

TS_LIST_LEVEL_RESPONSE = [
    [
        "station_no",
        "station_name",
        "station_latitude",
        "station_longitude",
        "stationparameter_name",
        "ts_name",
        "ts_id",
        "ts_path",
    ],
    [
        "133171",
        "Newton Stewart",
        "54.95726338",
        "-4.480525846",
        "Level",
        "15minute",
        "67948010",
        "1/133171/SG/15m.Cmd",
    ],
]

TS_LIST_NO_MATCH = ["No matches."]

# getTimeseriesValues with metadata=true returns station coordinates and
# the API's unit alongside the observation data.
TS_VALUES_FLOW_RESPONSE = [
    {
        "ts_id": "67947010",
        "station_latitude": "54.95726338",
        "station_longitude": "-4.480525846",
        "ts_unitsymbol": "m³/s",
        "columns": "Timestamp,Value,Quality Code",
        "data": [
            ["2026-09-10T00:00:00.000Z", "8.315", "254"],
            ["2026-09-10T00:15:00.000Z", "8.262", "254"],
            ["2026-09-10T00:30:00.000Z", "", "255"],
        ],
    }
]

TS_VALUES_LEVEL_RESPONSE = [
    {
        "ts_id": "67948010",
        "station_latitude": "54.95726338",
        "station_longitude": "-4.480525846",
        "ts_unitsymbol": "m",
        "columns": "Timestamp,Value,Quality Code",
        "data": [
            ["2026-09-10T00:00:00.000Z", "0.708", "254"],
        ],
    }
]

SAMPLE_FLOW_RAW = {
    "station_no": "133171",
    "station_name": "Newton Stewart",
    "parameter": "Q",
    "unit": "m³/s",
    "latitude": 54.95726338,
    "longitude": -4.480525846,
    "timestamp": "2026-09-10T00:00:00.000Z",
    "value": "8.315",
    "quality_code": "254",
}

SAMPLE_LEVEL_RAW = {
    "station_no": "133171",
    "station_name": "Newton Stewart",
    "parameter": "SG",
    "unit": "m",
    "latitude": 54.95726338,
    "longitude": -4.480525846,
    "timestamp": "2026-09-10T00:00:00.000Z",
    "value": "0.708",
    "quality_code": "254",
}


class TestScotlandSepaInit:
    def test_collector_name(self):
        assert ScotlandSepaCollector().name == "scotland_sepa"


class TestScotlandSepaFetchRaw:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.collector = ScotlandSepaCollector(client=self.mock_client)

    def test_fetch_raw_requires_station_id(self):
        try:
            self.collector.fetch_raw(station_id="")
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "station_id" in str(exc)

    def test_fetch_raw_rejects_unknown_parameter(self):
        try:
            self.collector.fetch_raw(
                station_id="133171",
                parameter="UNKNOWN",
            )
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "'Q'" in str(exc)
            assert "'SG'" in str(exc)

    def test_fetch_raw_returns_merged_flow_rows(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        rows = self.collector.fetch_raw(
            station_id="133171",
            parameter=FLOW_PARAMETER,
            days=1,
        )

        # Missing values are deliberately retained here because normalise()
        # owns filtering and validation of individual observations.
        assert len(rows) == 3
        assert rows[0]["station_no"] == "133171"
        assert rows[0]["station_name"] == "Newton Stewart"
        assert rows[0]["parameter"] == FLOW_PARAMETER
        assert rows[0]["unit"] == "m³/s"
        assert rows[0]["latitude"] == 54.95726338
        assert rows[0]["longitude"] == -4.480525846
        assert rows[0]["value"] == "8.315"

    def test_fetch_raw_calls_timeseries_list_then_values(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        self.collector.fetch_raw(
            station_id="133171",
            parameter=FLOW_PARAMETER,
        )

        calls = self.mock_client.get_json.call_args_list
        assert calls[0][1]["params"]["request"] == "getTimeseriesList"
        assert calls[1][1]["params"]["request"] == "getTimeseriesValues"

    def test_fetch_raw_requests_15_minute_series(self):
        # A station can expose annual maxima, daily means, gaugings, and other
        # products for the same parameter. Explicitly requesting "15minute"
        # prevents AquaScope from accidentally selecting an aggregate series.
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        self.collector.fetch_raw(
            station_id="133171",
            parameter=FLOW_PARAMETER,
        )

        params = self.mock_client.get_json.call_args_list[0][1]["params"]

        assert params["station_no"] == "133171"
        assert params["stationparameter_no"] == FLOW_PARAMETER
        assert params["ts_name"] == "15minute"

    def test_fetch_raw_requests_timeseries_list_fields(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        self.collector.fetch_raw(station_id="133171")

        params = self.mock_client.get_json.call_args_list[0][1]["params"]

        # Unlike BOM's KiWIS deployment, SEPA supports returnfields on
        # getTimeseriesList, so request only the metadata AquaScope needs.
        assert "returnfields" in params
        assert "ts_id" in params["returnfields"]
        assert "station_name" in params["returnfields"]

    def test_fetch_raw_requests_metadata_on_values_call(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        self.collector.fetch_raw(station_id="133171")

        params = self.mock_client.get_json.call_args_list[1][1]["params"]
        assert params["metadata"] == "true"
        assert params["returnfields"] == "Timestamp,Value,Quality Code"

    def test_fetch_raw_no_matches_returns_empty(self):
        self.mock_client.get_json.side_effect = [TS_LIST_NO_MATCH]

        rows = self.collector.fetch_raw(
            station_id="999999",
            parameter=FLOW_PARAMETER,
        )

        assert rows == []

    def test_fetch_raw_handles_request_failure(self):
        self.mock_client.get_json.side_effect = ConnectionError("boom")

        rows = self.collector.fetch_raw(
            station_id="133171",
            parameter=FLOW_PARAMETER,
        )

        assert rows == []

    def test_fetch_raw_passes_explicit_dates(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_FLOW_RESPONSE,
            TS_VALUES_FLOW_RESPONSE,
        ]

        self.collector.fetch_raw(
            station_id="133171",
            parameter=FLOW_PARAMETER,
            start_date="2026-09-01",
            end_date="2026-09-10",
        )

        params = self.mock_client.get_json.call_args_list[1][1]["params"]
        assert params["from"] == "2026-09-01"
        assert params["to"] == "2026-09-10"

    def test_fetch_raw_level_resolves_correct_timeseries(self):
        self.mock_client.get_json.side_effect = [
            TS_LIST_LEVEL_RESPONSE,
            TS_VALUES_LEVEL_RESPONSE,
        ]

        rows = self.collector.fetch_raw(
            station_id="133171",
            parameter=LEVEL_PARAMETER,
        )

        assert len(rows) == 1
        assert rows[0]["parameter"] == LEVEL_PARAMETER
        assert rows[0]["unit"] == "m"
        assert rows[0]["value"] == "0.708"


class TestScotlandSepaNormalise:
    def setup_method(self):
        self.collector = ScotlandSepaCollector()

    def test_normalise_produces_streamflow_reading_for_flow(self):
        records = self.collector.normalise([SAMPLE_FLOW_RAW])

        assert len(records) == 1
        assert isinstance(records[0], StreamflowReading)
        assert records[0].source == DataSource.SCOTLAND_SEPA
        assert records[0].station_id == "133171"
        assert records[0].station_name == "Newton Stewart"
        assert records[0].discharge_cms == 8.315
        assert records[0].source_type == "in_situ"
        assert records[0].unit == "m³/s"

    def test_normalise_produces_water_level_reading_for_level(self):
        records = self.collector.normalise([SAMPLE_LEVEL_RAW])

        assert len(records) == 1
        assert isinstance(records[0], WaterLevelReading)
        assert records[0].source == DataSource.SCOTLAND_SEPA
        assert records[0].water_level == 0.708
        assert records[0].unit == "m"

    def test_normalise_parses_location(self):
        records = self.collector.normalise([SAMPLE_FLOW_RAW])

        assert records[0].location is not None
        assert abs(records[0].location.latitude - 54.95726338) < 0.001
        assert abs(records[0].location.longitude - (-4.480525846)) < 0.001

    def test_normalise_preserves_quality_code(self):
        records = self.collector.normalise([SAMPLE_FLOW_RAW])

        assert records[0].remark == "Quality Code: 254"

    def test_normalise_skips_missing_values(self):
        raw = [
            {
                "station_no": "133171",
                "parameter": FLOW_PARAMETER,
                "unit": "m³/s",
                "timestamp": "2026-09-10T00:30:00.000Z",
                "value": "",
                "quality_code": "255",
            }
        ]

        assert self.collector.normalise(raw) == []

    def test_normalise_empty_input(self):
        assert self.collector.normalise([]) == []

    def test_normalise_falls_back_to_default_flow_unit(self):
        raw = [
            {
                "station_no": "133171",
                "parameter": FLOW_PARAMETER,
                "unit": "",
                "timestamp": "2026-09-10T00:00:00.000Z",
                "value": "8.315",
            }
        ]

        records = self.collector.normalise(raw)

        assert records[0].unit == PARAMETER_UNITS[FLOW_PARAMETER]

    def test_normalise_falls_back_to_default_level_unit(self):
        raw = [
            {
                "station_no": "133171",
                "parameter": LEVEL_PARAMETER,
                "unit": "",
                "timestamp": "2026-09-10T00:00:00.000Z",
                "value": "0.708",
            }
        ]

        records = self.collector.normalise(raw)

        assert records[0].unit == PARAMETER_UNITS[LEVEL_PARAMETER]


class TestScotlandSepaConstants:
    def test_parameter_codes(self):
        assert FLOW_PARAMETER == "Q"
        assert LEVEL_PARAMETER == "SG"

    def test_default_units(self):
        assert PARAMETER_UNITS[FLOW_PARAMETER] == "m3/s"
        assert PARAMETER_UNITS[LEVEL_PARAMETER] == "m"
