"""Tests for the shared CachedHTTPClient, focusing on the get_text path
that the WQP collector relies on (retries, rate-limit, disk caching)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from aquascope.utils.http_client import CachedHTTPClient


def _fake_response(text: str, content_type: str = "text/csv") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.raise_for_status.return_value = None
    return resp


class TestGetText:
    def test_returns_body_text(self, tmp_path):
        client = CachedHTTPClient(cache_dir=tmp_path)
        client._client = MagicMock()
        client._client.get.return_value = _fake_response("a,b\n1,2\n")

        out = client.get_text("https://example.test/data")

        assert out == "a,b\n1,2\n"
        assert client._client.get.call_count == 1

    def test_second_call_hits_cache(self, tmp_path):
        client = CachedHTTPClient(cache_dir=tmp_path)
        client._client = MagicMock()
        client._client.get.return_value = _fake_response("col\nval\n")

        first = client.get_text("https://example.test/data", params={"q": "1"})
        second = client.get_text("https://example.test/data", params={"q": "1"})

        assert first == second == "col\nval\n"
        # Network hit only once; the second read came from disk cache.
        assert client._client.get.call_count == 1

    def test_text_cache_does_not_collide_with_json_cache(self, tmp_path):
        # A get_text and a get_json for the same URL must not clobber each other.
        client = CachedHTTPClient(cache_dir=tmp_path)
        client._client = MagicMock()
        client._client.get.return_value = _fake_response("raw text body")

        text = client.get_text("https://example.test/thing")

        assert text == "raw text body"
        # The text payload lives under a *-text.txt key, not the JSON .json key.
        assert list(tmp_path.glob("*-text.txt"))
        assert not list(tmp_path.glob("*[!t][!e][!x][!t].json"))

    def test_retries_then_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("aquascope.utils.http_client.time.sleep", lambda *_: None)
        client = CachedHTTPClient(cache_dir=tmp_path, retries=3)
        client._client = MagicMock()
        client._client.get.side_effect = httpx.ConnectError("boom")

        with pytest.raises(RuntimeError, match="All 3 attempts failed"):
            client.get_text("https://example.test/data", use_cache=False)

        assert client._client.get.call_count == 3


class TestWQPRoutesThroughClient:
    def test_fetch_raw_uses_get_text(self):
        from aquascope.collectors.wqp import WQPCollector

        csv_body = (
            "MonitoringLocationIdentifier,ResultMeasureValue\n"
            "USGS-01010000,8.5\n"
        )
        mock_client = MagicMock()
        mock_client.get_text.return_value = csv_body
        collector = WQPCollector(client=mock_client)

        rows = collector.fetch_raw(state_code="US:06")

        mock_client.get_text.assert_called_once()
        assert rows[0]["MonitoringLocationIdentifier"] == "USGS-01010000"

    def test_fetch_raw_returns_empty_on_error(self):
        from aquascope.collectors.wqp import WQPCollector

        mock_client = MagicMock()
        mock_client.get_text.side_effect = RuntimeError("all attempts failed")
        collector = WQPCollector(client=mock_client)

        assert collector.fetch_raw(state_code="US:06") == []
