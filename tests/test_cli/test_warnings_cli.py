"""`aquascope warnings` and the MCP tool `flood_warnings` over aquascope.archive.warnings (#546): thin faces."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from aquascope import cli, mcp_server
from aquascope.archive import warnings as fw

R = {
    "available": True, "issue_date": "2026-10-09", "n": 1,
    "counts": {"2": 0, "5": 0, "10": 1, "25": 0, "50": 0, "100": 0},
    "sentence": "The GEOGLOWS forecast from 2026-10-09 expects 1 river reach to reach the 2-year flow within 15 days.",
    "reaches": [{"river_id": 123, "lat": 10.5, "lon": 100.25, "rp": 10, "peak_cms": 2500.0, "q2": 900.0,
                 "peak_date": "2026-10-12", "share": 0.71}],
    "truncated": False, "not": "Model output, not an official warning.",
}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "warnings", *argv])
    cli.main()


def test_warnings_prints_the_sentence_and_the_reaches(monkeypatch, capsys):
    seen = {}

    def fake(bbox, **kw):
        seen.update(bbox=bbox, **kw)
        return R

    monkeypatch.setattr(fw, "flood_warnings", fake)
    _run(monkeypatch, "--bbox", "99", "9", "101", "11", "--min-rp", "5")
    out = capsys.readouterr().out
    assert seen["bbox"] == [99.0, 9.0, 101.0, 11.0] and seen["min_rp"] == 5 and seen["limit"] == 20
    assert "expects 1 river reach" in out and "1 at 10-year" in out
    assert "123" in out and "10-yr" in out and "2,500" in out and "71%" in out
    assert "not an official warning" in out


def test_warnings_json_and_nothing_published(monkeypatch, capsys):
    monkeypatch.setattr(fw, "flood_warnings", lambda bbox, **kw: R)
    _run(monkeypatch, "--json")
    assert json.loads(capsys.readouterr().out)["reaches"][0]["river_id"] == 123
    monkeypatch.setattr(fw, "flood_warnings", lambda bbox, **kw: {"available": False, "sentence": "No issue yet."})
    _run(monkeypatch)
    assert capsys.readouterr().out.strip() == "No issue yet."


def test_the_mcp_tool_wraps_the_engine(monkeypatch):
    pytest.importorskip("mcp")
    assert "bbox" in mcp_server.flood_warnings([1, 2, 3])["error"]     # refused before any network read
    seen = {}

    def fake(bbox, **kw):
        seen.update(bbox=bbox, **kw)
        return R

    monkeypatch.setattr(fw, "flood_warnings", fake)
    assert mcp_server.flood_warnings([1, 2, 3, 4], min_rp=25, limit=5) == R
    assert seen == {"bbox": [1, 2, 3, 4], "min_rp": 25, "limit": 5}
    names = {t.name for t in asyncio.run(mcp_server.build_server().list_tools())}
    assert "flood_warnings" in names
