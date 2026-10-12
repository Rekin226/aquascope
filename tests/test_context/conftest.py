"""A fake web for the place-context tests: every URL a layer reads is served from memory, never the network."""

from __future__ import annotations

import csv
import gzip
import io
import json
from typing import Any

import pytest

from aquascope.context import _common, gauges, rasters
from aquascope.utils import cog


class _Resp:
    def __init__(self, status: int, content: bytes):
        self.status_code = status
        self.content = content


class FakeWeb:
    """Files by URL, with HTTP range support, and the CachedHTTPClient surface the layers use."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []

    # httpx.Client surface (range_get and get_bytes)
    def get(self, url: str, headers: dict | None = None, timeout: float | None = None) -> _Resp:
        self.calls.append(url)
        data = self.files.get(url)
        if data is None:
            return _Resp(404, b"")
        rng = (headers or {}).get("Range")
        if rng:
            start, end = (int(x) for x in rng.split("=", 1)[1].split("-"))
            return _Resp(206, data[start:end + 1])
        return _Resp(200, data)

    # CachedHTTPClient surface
    def get_text(self, url: str, params: dict | None = None) -> str:
        self.calls.append(url)
        if url not in self.files:
            raise RuntimeError(f"404 {url}")
        return self.files[url].decode("utf-8")

    def get_json(self, url: str, params: dict | None = None) -> Any:
        return json.loads(self.get_text(url, params))

    def add_json(self, url: str, value: Any) -> None:
        self.files[url] = json.dumps(value).encode("utf-8")

    def add_csv_gz(self, url: str, header: list[str], rows: list[list[Any]]) -> None:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        self.files[url] = gzip.compress(buf.getvalue().encode("utf-8"))

    def requested(self, fragment: str) -> int:
        return sum(1 for c in self.calls if fragment in c)


@pytest.fixture()
def web(monkeypatch) -> FakeWeb:
    fake = FakeWeb()
    monkeypatch.setattr(cog, "_httpx_client", lambda: fake)
    monkeypatch.setattr(rasters, "http_client", lambda: fake)
    monkeypatch.setattr(gauges, "http_client", lambda: fake)
    cog._OPEN.clear()
    _common.clear_memo()
    yield fake
    cog._OPEN.clear()
    _common.clear_memo()
