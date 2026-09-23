"""The studio_* MCP tools: stateless over the workspace dict, registered on the server."""

from __future__ import annotations

import asyncio

from aquascope import mcp_server as m
from tests.test_studio.conftest import PROBLEM, patched


def test_studio_tools_round_trip(tmp_path, no_deliverables):
    with patched():
        start = m.studio_start(PROBLEM, 51.415, -0.308, intake={"return_period": 50})
        assert start["reply"]["kind"] == "plan" and start["status"] == "review"
        assert start["summary"]["steps"] == 4 and start["workspace"]["brief"]["intake"]["return_period"] == 50
        done = m.studio_approve(start["workspace"], edits={"s1": None})
        assert done["reply"]["kind"] == "report" and done["status"] == "done"
        assert done["workspace"]["study"]["plan"]["edited"] and len(done["workspace"]["run"]["results"]) == 3
        assert "50-year" in str(done["reply"]["payload"]["report"]["key_numbers"])
        # the study on the map comes back with every reply: the site, then the cells the anywhere step sampled
        roles = [f["properties"]["role"] for f in done["map"]["features"]]
        assert done["map"]["type"] == "FeatureCollection" and roles[0] == "site" and "grid_cell" in roles
        q = m.studio_follow_up(done["workspace"], "what is the 50-year flow?")
        assert q["reply"]["kind"] == "answer" and "m3/s" in q["reply"]["text"]
        ch = m.studio_say(q["workspace"], "redo it with a 20-year return period")
        assert ch["reply"]["kind"] == "report" and ch["workspace"]["brief"]["intake"]["return_period"] == 20
        exported = m.studio_export(ch["workspace"], str(tmp_path / "mcp"))
        assert set(exported["paths"]) >= {"report.md", "study.yaml", "workspace.json"}
    assert "error" in m.studio_say({}, "x") and "error" in m.studio_export({"status": "done"}, str(tmp_path))


def test_questions_over_mcp(no_deliverables):
    with patched():
        start = m.studio_start("Can the river supply the town reliably?", 51.415, -0.308)
        assert start["reply"]["kind"] == "questions" and start["reply"]["payload"]["questions"][0]["id"] == "demand_m3s"
        nxt = m.studio_say(start["workspace"], "2 m3/s, a licence")
        assert nxt["reply"]["kind"] == "plan" and nxt["workspace"]["brief"]["intake"]["demand_m3s"] == 2.0
        assert nxt["workspace"]["brief"]["decision"] == "an abstraction licence"


def test_the_server_registers_the_studio_tools():
    server = m.build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"studio_start", "studio_say", "studio_approve", "studio_follow_up", "studio_export"} <= names
