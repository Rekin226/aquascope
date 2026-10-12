"""`aquascope watch` and the MCP tool over aquascope.watch (#521): thin faces over one engine."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from aquascope import cli, mcp_server, watch

DIGEST = {
    "today": "2026-10-08", "since": "2026-10-01", "n_items": 2, "n_changed": 1, "n_alerts": 1,
    "summary": "Since 1 October: 1 of 2 watched places changed, 1 with something to look at (Potomac: now above "
               "normal).",
    "items": [{"id": "usgs/USGS-01646500", "name": "Potomac", "line": "6 new days of data since 1 October.",
               "alerts": ["now above normal"], "changed": True, "notes": ["Added 2 recent days from the agency."]},
              {"id": "river:7", "name": "River reach 7", "line": "Nothing new since 1 October.", "alerts": [],
               "changed": False}],
    "notes": ["The Groundsource mirror is not published yet."],
}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "watch", *argv])
    cli.main()


def test_watch_prints_the_summary_and_a_line_per_item(monkeypatch, capsys):
    seen = {}

    def fake(items, since, **kw):
        seen.update(items=items, since=since, **kw)
        return DIGEST

    monkeypatch.setattr(watch, "watch_digest", fake)
    _run(monkeypatch, "usgs/USGS-01646500", "river:7", "--since", "2026-10-01",
         "--threshold", "usgs/USGS-01646500=10y", "--no-floods")
    out = capsys.readouterr().out
    assert seen["since"] == "2026-10-01" and seen["floods"] is False and seen["forecast"] == "auto"
    assert seen["items"][0]["threshold"] == "10y" and seen["items"][1]["kind"] == "reach"
    assert "Since 1 October: 1 of 2" in out and "* Potomac (usgs/USGS-01646500)" in out
    assert "  River reach 7 (river:7)" in out and "(Added 2 recent days from the agency.)" in out
    assert "Groundsource mirror" in out and "model output" in out


def test_watch_json_and_bad_input(monkeypatch, capsys):
    monkeypatch.setattr(watch, "watch_digest", lambda items, since, **kw: DIGEST)
    _run(monkeypatch, "usgs/1", "--json", "--no-forecast")
    assert json.loads(capsys.readouterr().out)["n_alerts"] == 1
    with pytest.raises(SystemExit):
        _run(monkeypatch, "nonsense")
    with pytest.raises(SystemExit):
        _run(monkeypatch, "usgs/1", "--threshold", "no-equals-sign")


def test_the_mcp_tool_is_registered_and_passes_thresholds(monkeypatch):
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "watch_digest" in names
    seen = {}

    def fake(items, since, **kw):
        seen.update(items=items, since=since, **kw)
        return DIGEST

    monkeypatch.setattr(watch, "watch_digest", fake)
    res = mcp_server.watch_digest(["usgs/1", {"kind": "reach", "river_id": 7}, "junk"], since="2026-10-01",
                                  thresholds={"usgs/1": 300, "river:7": "5y"})
    assert res is DIGEST and seen["since"] == "2026-10-01"
    assert seen["items"][0] == {"id": "usgs/1", "threshold": 300}
    assert seen["items"][1]["threshold"] == "5y" and seen["items"][2] == "junk"
