"""Tests for the new v0.2.0 collectors: GEMStat, WQP, Taiwan Civil IoT."""

import logging
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from aquascope.collectors.gemstat import GEMStatCollector
from aquascope.collectors.taiwan_civil_iot import TaiwanCivilIoTCollector
from aquascope.collectors.wqp import WQPCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample


# Streaming fakes, used in many WQP tests
class _StreamResp:
    """Minimal fake for an httpx streaming response context manager."""

    def __init__(self, lines, raise_on_status=None):
        self._lines = lines
        self._raise = raise_on_status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def iter_lines(self):
        yield from self._lines


class _FakeStreamClient:
    """Minimal fake for an httpx.Client that supports `.stream()`."""

    def __init__(self, lines, raise_on_status=None):
        self._lines = lines
        self._raise = raise_on_status
        self.stream_calls = []  # records (method, url, params) for assertions

    def stream(self, method, url, params=None, **kwargs):
        self.stream_calls.append((method, url, params))
        return _StreamResp(self._lines, raise_on_status=self._raise)


def _make_mock_client(lines, raise_on_status=None):
    """Return a (MagicMock CachedHTTPClient, _FakeStreamClient transport) pair."""
    fake_transport = _FakeStreamClient(lines, raise_on_status=raise_on_status)
    mc = mock.MagicMock()
    mc.base_url = "https://www.waterqualitydata.us/wqx3"
    mc._client = fake_transport
    return mc, fake_transport



class TestGEMStatCollector:
    def test_init(self):
        collector = GEMStatCollector()
        assert collector.name == "gemstat"

    def test_parse_csv_valid(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit,Latitude,Longitude,Country Code\n"
            "GEM001,2023-01-15,pH,7.2,pH units,47.3,8.5,CH\n"
            "GEM002,2023-02-20,DO,8.1,mg/L,48.1,11.6,DE\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 2
        assert all(isinstance(s, WaterQualitySample) for s in samples)
        assert samples[0].source == DataSource.GEMSTAT
        assert samples[0].parameter == "pH"
        assert samples[0].value == 7.2

    def test_parse_csv_skips_nd_values(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
            "GEM001,2023-01-15,pH,ND,pH units\n"
            "GEM002,2023-02-20,DO,8.1,mg/L\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 1

    def test_parse_csv_empty(self):
        csv_content = "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert len(samples) == 0

    def test_parse_csv_max_records(self):
        header = "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit\n"
        rows = "".join(f"GEM{i:03d},2023-01-{(i % 28) + 1:02d},pH,{7.0 + i * 0.01},pH units\n" for i in range(100))
        samples = GEMStatCollector.parse_gemstat_csv(header + rows, max_records=10)
        assert len(samples) == 10

    def test_parse_csv_with_location(self):
        csv_content = (
            "GEMS Station Number,Sample Date,Parameter,Analysis Result,Unit,Latitude,Longitude\n"
            "GEM001,2023-06-01,DO,7.5,mg/L,25.033,121.565\n"
        )
        samples = GEMStatCollector.parse_gemstat_csv(csv_content)
        assert samples[0].location is not None
        assert abs(samples[0].location.latitude - 25.033) < 0.001


class TestWQPCollector:
    def test_init(self):
        collector = WQPCollector()
        assert collector.name == "wqp"

    def test_default_client_timeout_is_payload_appropriate(self):
        collector = WQPCollector()
        assert collector.client.timeout == 600.0

    def test_normalise_valid(self):
        raw = [
            {
                "Location_Identifier": "USGS-01010000",
                "Location_Name": "Test Station",
                "Location_Latitude": "44.5",
                "Location_Longitude": "-67.5",
                "Activity_StartDate": "2023-05-15",
                "Activity_StartTime": "10:30:00",
                "Result_Characteristic": "Dissolved oxygen (DO)",
                "Result_Measure": "8.5",
                "Result_MeasureUnit": "mg/l",
            },
        ]
        collector = WQPCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].source == DataSource.WQP
        assert samples[0].value == 8.5
        assert samples[0].parameter == "Dissolved oxygen (DO)"


    def test_normalise_skips_empty_values(self):
        raw = [
            {
                "Location_Identifier": "USGS-01010000",
                "Activity_StartDate": "2023-05-15",
                "Result_Characteristic": "pH",
                "Result_Measure": "",
            },
            {
                "Location_Identifier": "USGS-01010000",
                "Activity_StartDate": "2023-05-15",
                "Result_Characteristic": "pH",
                "Result_Measure": "-",
            },
        ]
        collector = WQPCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0

    def test_fetch_raw_bounded_by_max_results(self):
        # 5 data rows offered, but we only asked for 2 — stream must stop early.
        csv_lines = [
            "Location_Identifier,Result_Characteristic,Result_Measure",
            "A,pH,7.0",
            "B,pH,7.1",
            "C,pH,7.2",
            "D,pH,7.3",
            "E,pH,7.4",
        ]
        mock_client, _ = _make_mock_client(csv_lines)
        collector = WQPCollector(client=mock_client)

        rows = collector.fetch_raw(state_code="US:11", max_results=2)

        assert len(rows) == 2
        assert rows[0]["Location_Identifier"] == "A"
        assert rows[1]["Location_Identifier"] == "B"
        # Test that rows C, D, E were never requested.
        returned_ids = {r["Location_Identifier"] for r in rows}
        assert "C" not in returned_ids
        assert "D" not in returned_ids
        assert "E" not in returned_ids

    def test_fetch_raw_streams_from_underlying_transport(self):
        csv_lines = [
            "Location_Identifier,Result_Measure",
            "USGS-01010000,8.5",
        ]
        mock_client, transport = _make_mock_client(csv_lines)
        collector = WQPCollector(client=mock_client)

        rows = collector.fetch_raw(state_code="US:11")

        assert rows[0]["Location_Identifier"] == "USGS-01010000"
        assert rows[0]["Result_Measure"] == "8.5"
        assert len(transport.stream_calls) == 1
        _, called_url, _ = transport.stream_calls[0]
        assert called_url.endswith("/Result/search")
        mock_client.get_text.assert_not_called()

    def test_fetch_raw_falls_back_to_buffered_get_text(self, caplog):
        # When streaming is unavailable (e.g. browser/WASM) the collector warns
        # and falls back to the shared client's buffered read, without caching.
        csv_body = (
            "Location_Identifier,Result_Measure\n"
            "USGS-01010000,8.5\n"
        )
        mock_client = mock.MagicMock()
        # Test with a plain object with no streaming support.
        mock_client._client = SimpleNamespace()
        mock_client.get_text.return_value = csv_body
        collector = WQPCollector(client=mock_client)

        with caplog.at_level(logging.WARNING, logger="aquascope.collectors.wqp"):
            rows = collector.fetch_raw(state_code="US:11")

        assert rows[0]["Location_Identifier"] == "USGS-01010000"
        mock_client.get_text.assert_called_once()
        assert mock_client.get_text.call_args.kwargs.get("use_cache") is False
        assert any("streaming is not available" in r.message for r in caplog.records)

    def test_fetch_raw_returns_empty_on_genuine_no_data(self):
        csv_lines = ["Location_Identifier,Result_Measure"]
        mock_client, _ = _make_mock_client(csv_lines)
        collector = WQPCollector(client=mock_client)

        assert collector.fetch_raw(state_code="US:11") == []

    def test_fetch_raw_raises_on_failure(self):
        mock_client, _ = _make_mock_client(
            lines=[],
            raise_on_status=httpx.ConnectError("dead endpoint"),
        )
        collector = WQPCollector(client=mock_client)

        with pytest.raises(httpx.ConnectError):
            collector.fetch_raw(state_code="US:11")

    def test_fetch_raw_calls_rate_limiter_before_streaming(self):
        csv_lines = ["Location_Identifier,Result_Measure"]
        mock_client, _ = _make_mock_client(csv_lines)
        mock_client.rate_limiter = mock.MagicMock()
        collector = WQPCollector(client=mock_client)

        collector.fetch_raw(state_code="US:11")

        mock_client.rate_limiter.wait_if_needed.assert_called_once()


class TestTaiwanCivilIoTCollector:
    def test_init(self):
        collector = TaiwanCivilIoTCollector()
        assert collector.name == "taiwan_civil_iot"

    def test_normalise_valid_datastream(self):
        raw = [
            {
                "name": "Water Level",
                "unitOfMeasurement": {"symbol": "m"},
                "Thing": {
                    "@iot.id": 1,
                    "name": "Station Alpha",
                    "Locations": [
                        {"location": {"coordinates": [121.5, 25.0]}}
                    ],
                },
                "Observations": [
                    {
                        "result": 3.45,
                        "phenomenonTime": "2024-03-15T10:00:00Z",
                    }
                ],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 1
        assert samples[0].source == DataSource.TAIWAN_CIVIL_IOT
        assert samples[0].value == 3.45
        assert samples[0].location is not None
        assert abs(samples[0].location.latitude - 25.0) < 0.01

    def test_normalise_skips_no_observations(self):
        raw = [
            {
                "name": "Flow Rate",
                "Thing": {"@iot.id": 2, "name": "Empty Station"},
                "Observations": [],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0

    def test_normalise_skips_null_result(self):
        raw = [
            {
                "name": "pH",
                "Thing": {"@iot.id": 3, "name": "Null Station"},
                "Observations": [{"result": None, "phenomenonTime": "2024-01-01T00:00:00Z"}],
            }
        ]
        collector = TaiwanCivilIoTCollector()
        samples = collector.normalise(raw)
        assert len(samples) == 0
