"""aquascope-mcp (#113): tools are plain functions over the registry, the catalog and aquascope.explore.

The catalog download and the collectors are replaced by fakes; one test also
registers the tools on a real MCP server object and calls them through it,
so the schema generation and result wrapping of the installed SDK are covered.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mcp")

from aquascope import mcp_server as m  # noqa: E402
from aquascope.archive import catalog  # noqa: E402

CATALOG = [
    {"source": "usgs", "station_id": "USGS-1", "name": "Potomac River at Little Falls", "latitude": 38.95,
     "longitude": -77.13, "variables": ["discharge"], "period_start": "1930-01-01", "period_end": None,
     "url": "https://waterdata.usgs.gov/monitoring-location/1/", "river": None, "country": "USA",
     "agency": "USGS", "license": "US-PD", "redistributable": True, "extra": {}},
    {"source": "taiwan_cwa", "station_id": "466920", "name": "臺北", "latitude": 25.04, "longitude": 121.51,
     "variables": ["climate", "precipitation"], "period_start": "1896-01-01", "period_end": None, "url": None,
     "river": None, "country": "TWN", "agency": "CWA", "license": "OGDL-Taiwan-1.0", "redistributable": True,
     "extra": {}},
    {"source": "uk_ea", "station_id": "abc", "name": "Kingston", "latitude": 51.41, "longitude": -0.31,
     "variables": ["discharge", "water_level"], "period_start": "1883-10-01", "period_end": None, "url": None,
     "river": "River Thames", "country": "GBR", "agency": "EA", "license": "OGL-UK-3.0", "redistributable": True,
     "extra": {}},
]


def _flow(years=15):
    idx = pd.date_range("2000-01-01", periods=int(365.25 * years), freq="D")
    rng = np.random.default_rng(3)
    return pd.Series(np.exp(rng.normal(0, 0.5, len(idx))) * 40, index=idx)


def test_list_sources_shape():
    out = m.list_sources()
    assert out["n_sources"] >= 30
    assert "usgs" in out["with_station_catalog"] and "grdc" not in out["redistributable"]
    usgs = next(s for s in out["sources"] if s["key"] == "usgs")
    assert usgs["variables"] == ["discharge", "water_level", "water_quality"] and usgs["license"] == "US-PD"


def test_find_stations_filters_and_caps():
    with patch.object(catalog, "load_stations", return_value=CATALOG), \
         patch("aquascope.archive.catalog.load_stations", return_value=CATALOG):
        out = m.find_stations(query="kingston")
        assert out["n_returned"] == 1 and out["stations"][0]["source"] == "uk_ea"
        near = m.find_stations(near=[25.0, 121.5], limit=2)
        assert [s["station_id"] for s in near["stations"]][0] == "466920"
        bbox = m.find_stations(bbox=[-78, 38, -76, 40], variable="discharge")
        assert [s["station_id"] for s in bbox["stations"]] == ["USGS-1"]
        capped = m.find_stations(limit=10_000)
        assert capped["limit"] == m.MAX_STATIONS
        assert "error" in m.find_stations(variable="lava")
        assert "license" in bbox["stations"][0] and "extra" not in bbox["stations"][0]


def test_get_timeseries_is_bounded():
    s = _flow(10)
    fetched = {"series": s, "variable": "discharge", "unit": "m3/s", "note": "fake"}
    with patch("aquascope.explore.fetch_series", return_value=fetched):
        out = m.get_timeseries("usgs", "USGS-1", years=10, resample="M", max_points=50)
    assert out["variable"] == "discharge" and out["n_observations"] == len(s)
    assert out["resample"] == "MS" and out["n_points"] <= 50
    assert out["points"][0][0] == "2000-01-01" and isinstance(out["points"][0][1], float)
    assert out["license"] == "US-PD" and "attribution" in out
    with patch("aquascope.explore.fetch_series", return_value=fetched):
        big = m.get_timeseries("usgs", "USGS-1", years=10, resample="D", max_points=999_999)
    assert big["n_points"] <= m.MAX_POINTS


def test_get_timeseries_unknown_source_and_empty():
    assert "error" in m.get_timeseries("nope", "1")
    empty = {"series": None, "variable": "", "unit": "", "note": "n"}
    with patch("aquascope.explore.fetch_series", return_value=empty):
        out = m.get_timeseries("usgs", "USGS-0")
    assert out["n"] == 0 and "error" in out


def test_analyze_and_flood_frequency_drop_bulk_and_can_bootstrap():
    s = _flow(15)
    fetched = {"series": s, "variable": "discharge", "unit": "m3/s", "note": "fake"}
    with patch("aquascope.explore.fetch_series", return_value=fetched):
        out = m.analyze_station("usgs", "USGS-1", years=15)
        assert "series" not in out and set(out["fdc"]) == {"q95", "q50", "q10"}
        assert out["ffa"]["n_years"] >= 13 and "gev_bootstrap" not in out["ffa"]["fits"]
        ff = m.flood_frequency("usgs", "USGS-1", years=15, bootstrap_ci=True)
    assert set(ff) >= {"ffa", "methods", "license"} and "series" not in ff
    assert "gev_bootstrap" in ff["ffa"]["fits"] and len(ff["ffa"]["fits"]["gev_bootstrap"]["ci"]) == 6
    assert any(mm["name"].startswith("GEV (MLE") for mm in ff["methods"])


def test_describe_methods_lists_citations():
    out = m.describe_methods()
    assert out["return_periods"] == [2, 5, 10, 25, 50, 100]
    assert "lp3" in out["methods"] and "Bulletin 17C" in out["methods"]["lp3"]["citation"]


def test_server_registers_tools_and_calls_through_sdk():
    server = m.build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names >= {"list_sources", "find_stations", "get_timeseries", "analyze_station", "flood_frequency",
                     "describe_methods", "describe_catchment", "similar_basins", "regionalize_signatures",
                     "archive_health"}
    with patch("aquascope.archive.catalog.load_stations", return_value=CATALOG):
        res = asyncio.run(server.call_tool("find_stations", {"query": "potomac"}))
    payload = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if payload is None and isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        payload = res[1]  # mcp 1.10+: (unstructured content, structured content)
    if payload is None:  # older SDKs return only text content
        import json

        blocks = res[0] if isinstance(res, tuple) else res
        payload = json.loads(blocks[0].text if isinstance(blocks, list) else blocks.content[0].text)
    assert payload["n_returned"] == 1 and payload["stations"][0]["station_id"] == "USGS-1"


def test_describe_catchment_tool_wraps_basins(monkeypatch):
    calls = {}

    def fake(lat, lon, upstream=True):
        calls["args"] = (lat, lon, upstream)
        return {"sub_basin": {"hybas_id": 1}, "attributes": {}, "license": "CC-BY-4.0"}

    monkeypatch.setattr("aquascope.archive.basins.describe_catchment", fake)
    out = m.describe_catchment(48.85, 2.35, upstream=False)
    assert out["sub_basin"]["hybas_id"] == 1 and calls["args"] == (48.85, 2.35, False)

    def boom(lat, lon, upstream=True):
        raise RuntimeError("no basins yet")

    monkeypatch.setattr("aquascope.archive.basins.describe_catchment", boom)
    assert "no basins yet" in m.describe_catchment(0, 0)["error"]


def test_similar_basins_tool_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr("aquascope.archive.similar.similar_for_point",
                        lambda lat, lon, **kw: calls.append(("point", lat, lon, kw)) or {"stations": [], "k": 0})
    monkeypatch.setattr("aquascope.archive.similar.similar_for_station",
                        lambda s, i, **kw: calls.append(("station", s, i, kw)) or {"stations": [], "k": 0})
    assert m.similar_basins(lat=1.0, lon=2.0, k=99)["k"] == 0 and calls[-1][0] == "point" and calls[-1][3]["k"] == 50
    assert m.similar_basins(source="usgs", station_id="USGS-1", method="similarity")["k"] == 0
    assert calls[-1][:3] == ("station", "usgs", "USGS-1")
    assert "give lat and lon" in m.similar_basins()["error"]


def test_regionalize_signatures_tool_dispatches(monkeypatch):
    calls = []
    monkeypatch.setattr("aquascope.archive.regionalize.regionalize_point",
                        lambda lat, lon, **kw: calls.append((lat, lon, kw)) or {"estimates": {},
                                                                                "method": kw["method"]})
    assert m.regionalize_signatures(1.0, 2.0, k=99, method="both")["method"] == "both" and calls[-1][2]["k"] == 50

    def boom(*a, **k):
        raise RuntimeError("no signatures yet")

    monkeypatch.setattr("aquascope.archive.regionalize.regionalize_point", boom)
    assert "no signatures yet" in m.regionalize_signatures(0, 0)["error"]


# ── inline views for clients that render them (MCP Apps, #236) ──────────────


def test_station_view_carries_a_renderable_html_view() -> None:
    from unittest.mock import patch

    from aquascope.mcp_server import station_view

    fake = {
        "name": "Fish River", "source": "usgs", "station_id": "USGS-1", "start": "1986-01-01",
        "end": "2026-01-01", "years": 40, "unit": "m3/s", "stats": {"mean": 43.4, "max": 507.0},
        "series": {"v": [1, 5, 3, 9, 2, 7, 4, 8, 3, 6] * 20},
        "ffa": {"return_periods": [2, 100], "fits": {"gev_lmoments": {"q": [228.6, 583.2]}}},
        "attribution": "U.S. Geological Survey", "license": "public domain",
    }
    with patch("aquascope.mcp_server.analyze_station", return_value=fake):
        result = station_view("usgs", "USGS-1")

    # The ordinary result is untouched, so a client without the extension loses nothing.
    assert result["stats"]["mean"] == 43.4
    view = result["_meta"]["mcp/view"]
    assert view["mimeType"] == "text/html"
    assert "<svg" in view["html"], "the hydrograph is drawn inline, with no library"
    assert "583.2" in view["html"], "the 100-year flood is shown"
    assert "U.S. Geological Survey" in view["html"], "attribution travels with the view"


def test_station_view_passes_an_error_straight_through() -> None:
    from unittest.mock import patch

    from aquascope.mcp_server import station_view

    with patch("aquascope.mcp_server.analyze_station", return_value={"error": "no such station"}):
        result = station_view("usgs", "nope")
    assert result["error"] == "no such station"
    assert "_meta" not in result
