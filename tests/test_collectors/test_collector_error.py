"""Tests for CollectorError exception and error distinguishing across collectors."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aquascope.cli import cmd_collect
from aquascope.collectors import (
    CollectorError,
    EUWFDCollector,
    JapanMLITCollector,
    KoreaWAMISCollector,
    SDG6Collector,
    WQPCollector,
)
from aquascope.utils.http_client import CachedHTTPClient


class TestCollectorErrorClass:
    def test_init_and_attributes(self):
        cause = ValueError("root cause")
        err = CollectorError(
            "Service unavailable",
            source="japan_mlit",
            url="http://example.com/api",
            status_code=503,
            cause=cause,
        )
        assert str(err) == "Service unavailable"
        assert err.source == "japan_mlit"
        assert err.url == "http://example.com/api"
        assert err.status_code == 503
        assert err.cause is cause
        assert isinstance(err, RuntimeError)

    def test_defaults(self):
        err = CollectorError("Failed", source="test")
        assert err.url == ""
        assert err.status_code is None
        assert err.cause is None

    def test_extract_status_code_from_cause_chain(self):
        request = httpx.Request("GET", "http://example.com/api")
        response = httpx.Response(404, request=request)
        http_err = httpx.HTTPStatusError("Not Found", request=request, response=response)
        runtime_err = RuntimeError("All 3 attempts failed")
        runtime_err.__cause__ = http_err

        err = CollectorError("Collector failed", source="test", cause=runtime_err)
        assert err.status_code == 404


def _make_http_status_runtime_error(url: str, status_code: int) -> RuntimeError:
    request = httpx.Request("GET", url)
    response = httpx.Response(status_code, request=request)
    status_err = httpx.HTTPStatusError(f"Status {status_code}", request=request, response=response)
    err = RuntimeError(f"All 3 attempts failed for {url} (status {status_code})")
    err.__cause__ = status_err
    return err


class TestCachedHTTPClientFormatting:
    def setup_method(self):
        self.client = CachedHTTPClient(retries=3)

    def test_format_retry_error_http_status(self):
        request = httpx.Request("GET", "http://example.com/api")
        response = httpx.Response(404, request=request)
        status_err = httpx.HTTPStatusError("Not Found", request=request, response=response)

        msg = self.client._format_retry_error("http://example.com/api", status_err)
        assert "All 3 attempts failed for http://example.com/api (status 404)" in msg

    def test_format_retry_error_connect_error(self):
        conn_err = httpx.ConnectError("Connection refused")
        msg = self.client._format_retry_error("http://example.com/api", conn_err)
        assert "All 3 attempts failed for http://example.com/api (ConnectError: Connection refused)" in msg

    def test_format_retry_error_post_method(self):
        conn_err = httpx.ConnectError("timeout")
        msg = self.client._format_retry_error("http://example.com/api", conn_err, method="POST")
        assert "POST http://example.com/api" in msg

    def test_format_retry_error_unknown(self):
        msg = self.client._format_retry_error("http://example.com/api", None)
        assert "unknown error" in msg


class TestJapanMLITCollectorError:
    def test_fetch_raw_raises_collector_error_on_404(self):
        client = MagicMock()
        client.get_json.side_effect = _make_http_status_runtime_error("http://www1.river.go.jp/cgi-bin/", 404)
        collector = JapanMLITCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(station_id="12345")

        assert exc_info.value.source == "japan_mlit"
        assert exc_info.value.status_code == 404
        assert "404" in str(exc_info.value)

    def test_fetch_raw_raises_collector_error_on_network_error(self):
        client = MagicMock()
        client.get_json.side_effect = httpx.ConnectError("host unreachable")
        collector = JapanMLITCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(station_id="12345")

        assert exc_info.value.source == "japan_mlit"
        assert isinstance(exc_info.value.cause, httpx.ConnectError)

    def test_fetch_raw_returns_empty_list_on_empty_data(self):
        client = MagicMock()
        client.get_json.return_value = []
        collector = JapanMLITCollector(client=client)
        assert collector.fetch_raw(station_id="12345") == []


class TestKoreaWAMISCollectorError:
    def test_fetch_raw_raises_collector_error_on_500(self):
        client = MagicMock()
        client.get_json.side_effect = _make_http_status_runtime_error("http://www.wamis.go.kr/api", 500)
        collector = KoreaWAMISCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(station_id="12345")

        assert exc_info.value.source == "korea_wamis"
        assert exc_info.value.status_code == 500

    def test_fetch_raw_raises_collector_error_on_network_error(self):
        client = MagicMock()
        client.get_json.side_effect = ConnectionResetError("connection reset")
        collector = KoreaWAMISCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(station_id="12345")

        assert exc_info.value.source == "korea_wamis"
        assert isinstance(exc_info.value.cause, ConnectionResetError)

    def test_fetch_raw_returns_empty_list_on_empty_data(self):
        client = MagicMock()
        client.get_json.return_value = []
        collector = KoreaWAMISCollector(client=client)
        assert collector.fetch_raw(station_id="12345") == []


class TestEUWFDCollectorError:
    def test_fetch_raw_raises_collector_error_on_404(self):
        client = MagicMock()
        client.get_json.side_effect = _make_http_status_runtime_error("https://discomap.eea.europa.eu", 404)
        collector = EUWFDCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(country="DE")

        assert exc_info.value.source == "eu_wfd"
        assert exc_info.value.status_code == 404

    def test_fetch_raw_raises_collector_error_on_network_error(self):
        client = MagicMock()
        client.get_json.side_effect = httpx.ReadTimeout("timed out")
        collector = EUWFDCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(country="DE")

        assert exc_info.value.source == "eu_wfd"
        assert isinstance(exc_info.value.cause, httpx.ReadTimeout)

    def test_fetch_raw_returns_empty_list_on_empty_data(self):
        client = MagicMock()
        client.get_json.return_value = []
        collector = EUWFDCollector(client=client)
        assert collector.fetch_raw(country="DE") == []

        client.get_json.return_value = {"results": []}
        assert collector.fetch_raw(country="DE") == []


class TestSDG6CollectorError:
    def test_fetch_raw_raises_collector_error_on_failure(self):
        client = MagicMock()
        client.get_json.side_effect = _make_http_status_runtime_error("https://sdg6data.org/api", 503)
        collector = SDG6Collector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(indicator="6.3.2")

        assert exc_info.value.source == "sdg6"
        assert exc_info.value.status_code == 503

    def test_fetch_raw_returns_empty_list_on_empty_data(self):
        client = MagicMock()
        client.get_json.return_value = {"data": []}
        collector = SDG6Collector(client=client)
        assert collector.fetch_raw(indicator="6.3.2") == []


class TestWQPCollectorError:
    def test_fetch_raw_raises_collector_error_on_failure(self):
        client = MagicMock()
        client.rate_limiter = MagicMock()
        client._client = MagicMock()
        client._client.stream.side_effect = httpx.ConnectError("connection refused")
        collector = WQPCollector(client=client)

        with pytest.raises(CollectorError) as exc_info:
            collector.fetch_raw(state_code="US:11")

        assert exc_info.value.source == "wqp"
        assert isinstance(exc_info.value.cause, httpx.ConnectError)


class TestCLICollectorError:
    def test_cmd_collect_exits_on_collector_error(self):
        args = argparse.Namespace(
            source="japan_mlit",
            station="12345",
            days=None,
            start_date=None,
            end_date=None,
            format="csv",
            parameter=None,
            country=None,
            year=None,
            water_body_type=None,
            bbox=None,
            max_results=None,
            max_stations=None,
            state_code=None,
            site_id=None,
            characteristic_name=None,
            indicator=None,
            area=None,
            parameter_type=None,
            api_key=None,
        )

        with patch("aquascope.registry.build_collector") as mock_build, pytest.raises(SystemExit) as exc_info:
            mock_collector = MagicMock()
            mock_collector.collect.side_effect = CollectorError(
                "Japan MLIT endpoint failed: 404",
                source="japan_mlit",
                url="http://www1.river.go.jp/cgi-bin/",
                status_code=404,
            )
            mock_build.return_value = mock_collector

            cmd_collect(args)

        assert exc_info.value.code == 1
