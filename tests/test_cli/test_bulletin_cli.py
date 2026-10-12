"""`aquascope bulletin` and the MCP tool over aquascope.bulletin (#523): thin faces over one engine."""

from __future__ import annotations

import json
import sys

import pytest

from aquascope import bulletin, cli, mcp_server

B = {
    "month": "2026-09", "label": "September 2026", "title": "State of the rivers", "origin": "published",
    "summary": "In September 2026, 3 gauges in 2 countries had at least 25 days of flow.",
    "headline": "September 2026: 33 % of 3 gauges below normal, 33 % normal, 33 % above.",
    "counts": {"much_below": 1, "below": 0, "normal": 1, "above": 0, "much_above": 1},
    "coverage": {"considered": 5, "classed": 3, "excluded": {"no_days": 2, "few_days": 0, "short_record": 0},
                 "excluded_text": dict(bulletin.REASONS), "by_source": {"usgs": {"considered": 5, "classed": 3}}},
    "countries": [{"country": "USA", "name": "United States", "n": 2, "median_percentile": 50.0,
                   "median_label": "normal",
                   "counts": {"much_below": 0, "below": 0, "normal": 1, "above": 0, "much_above": 1}},
                  {"country": "GBR", "name": "United Kingdom", "n": 1, "median_percentile": 3.0,
                   "median_label": "much below normal",
                   "counts": {"much_below": 1, "below": 0, "normal": 0, "above": 0, "much_above": 0}}],
    "basins": [{"basin_id": i} for i in range(20)],
    "gauges": [{"source": "usgs", "station_id": "A", "country": "USA"}, {"source": "usgs", "station_id": "B",
                                                                         "country": "USA"},
               {"source": "uk_ea", "station_id": "C", "country": "GBR"}],
}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "bulletin", *argv])
    cli.main()


def test_bulletin_prints_the_summary_and_the_country_table(monkeypatch, capsys):
    seen = {}

    def fake(month, sources, **kw):
        seen.update(month=month, sources=sources, **kw)
        return json.loads(json.dumps(B))

    monkeypatch.setattr(bulletin, "status_bulletin", fake)
    _run(monkeypatch, "2026-09", "--sources", "usgs", "--top-up", "10")
    out = capsys.readouterr().out
    assert seen["month"] == "2026-09" and seen["sources"] == ["usgs"] and seen["top_up"] == 10
    assert "State of the rivers, September 2026 (published)" in out
    assert "In September 2026, 3 gauges" in out
    assert "United States" in out and "50 (normal)" in out
    assert "Left out: 2 with no day of data in the month." in out
    assert "--out DIR" in out


def test_bulletin_json_leaves_the_gauges_out_unless_asked(monkeypatch, capsys):
    monkeypatch.setattr(bulletin, "status_bulletin", lambda *a, **k: json.loads(json.dumps(B)))
    _run(monkeypatch, "--json")
    assert "gauges" not in json.loads(capsys.readouterr().out)
    _run(monkeypatch, "--json", "--gauges")
    assert len(json.loads(capsys.readouterr().out)["gauges"]) == 3


def test_bulletin_out_writes_the_files(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(bulletin, "status_bulletin", lambda *a, **k: json.loads(json.dumps(B)))
    got = {}

    def fake_write(b, out, figure=True):
        got.update(out=out, figure=figure)
        return {"html": f"{out}/bulletins/2026-09/bulletin.html"}

    monkeypatch.setattr(bulletin, "write_bulletin", fake_write)
    _run(monkeypatch, "2026-09", "--out", str(tmp_path), "--no-map")
    out = capsys.readouterr().out
    assert got == {"out": str(tmp_path), "figure": False}
    assert "bulletin.html" in out


def test_a_bad_month_exits_with_the_reason(monkeypatch, capsys):
    def bad(*a, **k):
        raise ValueError("not a month: 'Sept' (use YYYY-MM)")

    monkeypatch.setattr(bulletin, "status_bulletin", bad)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "Sept")
    assert exc.value.code == 2
    assert "use YYYY-MM" in capsys.readouterr().out


def test_the_mcp_tool_drops_the_gauge_list_and_filters_by_country(monkeypatch):
    monkeypatch.setattr(bulletin, "status_bulletin", lambda *a, **k: json.loads(json.dumps(B)))
    res = mcp_server.status_bulletin("2026-09")
    assert "gauges" not in res and len(res["basins"]) == 15
    uk = mcp_server.status_bulletin("2026-09", country="gbr")
    assert [g["station_id"] for g in uk["gauges"]] == ["C"]

    def bad(*a, **k):
        raise ValueError("not a month")

    monkeypatch.setattr(bulletin, "status_bulletin", bad)
    assert mcp_server.status_bulletin("x") == {"error": "not a month"}


def test_the_mcp_server_registers_the_tool():
    pytest.importorskip("mcp")
    import asyncio

    tools = asyncio.run(mcp_server.build_server().list_tools())
    assert "status_bulletin" in {t.name for t in tools}
