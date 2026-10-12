"""Study this area: the CLI verb and the MCP tool are thin faces over aquascope.area_study.study_area."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from aquascope import cli

CANNED = {
    "headline": "2 of 3 gauges studied (2 from the Archive, 0 live).",
    "sites": [
        {"source": "usgs", "station_id": "A", "status": "studied", "record_years": 40.0, "q100": 812.5,
         "unit": "m3/s", "trend": "up", "trend_p": 0.012, "note": ""},
        {"source": "usgs", "station_id": "B", "status": "studied", "record_years": 6.0, "q100": None,
         "unit": "m3/s", "trend": "untested", "trend_p": None, "note": ""},
        {"source": "usgs", "station_id": "C", "status": "skipped", "note": "live-fetch cap"},
    ],
    "notes": ["1 gauge(s) were skipped."],
    "table": {"columns": ["source", "station_id"], "rows": [["usgs", "A"], ["usgs", "B"], ["usgs", "C"]]},
    "geojson": {"type": "FeatureCollection", "features": []},
}


def test_cli_area_study_passes_the_box_and_prints_the_headline(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake(stations, **kw):
        seen.update(stations=stations, **kw)
        return CANNED

    monkeypatch.setattr("aquascope.area_study.study_area", fake)
    out_csv = tmp_path / "area.csv"
    monkeypatch.setattr(sys, "argv", ["aquascope", "area-study", "--bbox=-77,38,-76,39", "--max-live", "5",
                                      "-o", str(out_csv)])
    cli.main()
    out = capsys.readouterr().out
    assert seen["stations"] is None and seen["bbox"] == (-77.0, 38.0, -76.0, 39.0) and seen["max_live"] == 5
    assert out.startswith("2 of 3 gauges studied")
    assert "usgs/A: 40.0 yr, Q100 812.5 m3/s, trend up, p = 0.012" in out
    assert "usgs/C: skipped (live-fetch cap)" in out
    assert out_csv.read_text().startswith("source,station_id")


def test_cli_area_study_with_stations_and_json(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr("aquascope.area_study.study_area", lambda st, **kw: seen.update(st=st, **kw) or CANNED)
    monkeypatch.setattr(sys, "argv", ["aquascope", "area-study", "--station", "usgs/A", "--station", "uk_ea/x",
                                      "--json"])
    cli.main()
    assert seen["st"] == ["usgs/A", "uk_ea/x"] and seen["bbox"] is None
    assert json.loads(capsys.readouterr().out)["headline"] == CANNED["headline"]


def test_cli_area_study_needs_an_area(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aquascope", "area-study"])
    with pytest.raises(SystemExit):
        cli.main()


def test_mcp_study_area_is_registered_and_clamps_the_caps(monkeypatch):
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    seen = {}
    monkeypatch.setattr("aquascope.area_study.study_area", lambda st, **kw: seen.update(st=st, **kw) or CANNED)
    assert m.study_area(bbox=[-77, 38, -76, 39], max_live=500)["headline"] == CANNED["headline"]
    assert seen["bbox"] == (-77, 38, -76, 39) and seen["max_live"] == 25 and seen["st"] is None
    assert "error" in m.study_area()
    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "study_area" in names
