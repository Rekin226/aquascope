"""Ask for data (#419): a decline about data the user could bring becomes a request, the study waits, and
a table or "continue without" moves it on."""

from __future__ import annotations

import io

import pandas as pd

from aquascope.studio import requests
from tests.test_studio.conftest import RECON, fake_tools, patched

GW_BRIEF = ("The water table under our farm near Tetbury seems to be falling. Is it the new abstraction licence "
            "upstream that is pulling it down? Tell me why it is falling.")


def _gw_recon() -> dict:
    import json

    recon = json.loads(json.dumps(RECON))
    ctx = recon.setdefault("context", {})
    ctx.setdefault("years_by_variable", {})["groundwater_level"] = 48.0
    recon.setdefault("stations", []).append({"source": "uk_ea", "station_id": "GW1", "name": "Brokenborough",
                                             "variables": ["groundwater_level"], "years": 48.0,
                                             "distance_km": 3.9, "lat": 51.6, "lon": -2.2})
    return recon


def test_a_cause_question_without_pumping_data_asks_instead_of_declining(studio_factory):
    s, calls = studio_factory(recon_value=_gw_recon())
    r = s.say(GW_BRIEF)
    while r.kind == "questions":
        r = s.say("just go")
    assert r.kind == "data_request", (r.kind, r.text)
    ws = s.workspace
    assert ws.status == "waiting" and ws.pending_request["what"].startswith("abstraction (pumping) records")
    req = ws.pending_request
    assert req["can_continue"] and req["continue_without"] == {"attribute_cause": False}
    assert "continue without" in r.text and "What it changes" in r.text
    assert ws.messages[-1].kind == "data_request" and calls == [], "nothing runs while the crew waits"
    again = s.say("what do you mean?")
    assert again.kind == "data_request", "anything but a table or 'continue' repeats the request"
    r2 = s.say("continue without")
    assert r2.kind == "plan" and ws.status == "review" and ws.brief.intake["attribute_cause"] is False
    assert any("was not provided" in a for a in ws.brief.assumptions) and ws.pending_request is None
    assert ws.study["plan"]["playbook"] == "groundwater_decline"


def test_a_table_dropped_while_waiting_is_inventoried_and_planned_on(studio_factory):
    s, _ = studio_factory(recon_value=_gw_recon())
    r = s.say(GW_BRIEF)
    while r.kind == "questions":
        r = s.say("just go")
    assert r.kind == "data_request"
    csv = "date,well,volume_m3\n" + "\n".join(f"20{y:02d}-01-01,W1,{1000 + y}" for y in range(10, 26))
    r2 = s.add_table("abstraction.csv", pd.read_csv(io.StringIO(csv)))
    ws = s.workspace
    assert ws.status in ("review", "declined", "waiting"), ws.status
    assert "upload:abstraction.csv" in ws.tables
    assert any(d.id == "upload:abstraction.csv" for d in ws.inventory.datasets), "the Scout saw the table"
    assert any(e["event"] == "table" and "waiting" in e["detail"] for e in ws.events)
    assert r2.kind in ("plan", "data_request", "declined")


def test_a_request_that_allows_no_continuation_declines_on_continue(studio_factory):
    s, _ = studio_factory()
    ws = s.workspace
    ws.brief.problem, ws.brief.playbook, ws.brief.kind = "Is the water fit to drink?", "water_quality", "water_quality"
    request = requests.data_request_for(ws, "No water-quality samples within reach of the site.")
    assert request and request["what"].startswith("a table of your own water-quality samples")
    assert not request["can_continue"]
    assert requests.continue_without(ws, request) is False
    assert requests.data_request_for(ws, "Map the inundation extent") is None, "not a data problem"
    assert requests.is_continue("Continue without it") and requests.is_continue("just go")
    assert not requests.is_continue("here is the file")


def test_a_table_after_the_report_is_a_follow_up_change(studio_factory):
    s, calls = studio_factory()
    s.say("Design flow for a road crossing, 100-year return period")
    r = s.approve()
    assert r.kind == "report"
    n = len(calls)
    r2 = s.add_table("flows.csv", "datetime,value\n2000-01-01,1\n2000-01-02,2\n")
    ws = s.workspace
    assert ws.status == "done" and "upload:flows.csv" in ws.tables
    assert ws.follow_ups[-1]["text"].startswith("use the new table upload:flows.csv")
    assert r2.kind in ("report", "answer")
    assert len(calls) >= n


def test_the_cli_continues_without_and_the_mcp_tools_take_tables(monkeypatch, capsys):
    import sys

    from aquascope import cli, mcp_server

    with patched(_gw_recon(), tools=fake_tools([])):
        monkeypatch.setattr(sys, "argv", ["aquascope", "studio", GW_BRIEF, "--lat", "51.6", "--lon", "-2.2", "--yes",
                                          "-q", "--continue-without", "--out", "/tmp/aquascope-studio-test-cw"])
        cli.main()
        out = capsys.readouterr()
        assert "Before this can be planned, the crew needs abstraction" in out.out or "abstraction" in out.err
        start = mcp_server.studio_start(GW_BRIEF, 51.6, -2.2)
        while start["reply"]["kind"] == "questions":
            start = mcp_server.studio_say(start["workspace"], "just go")
        assert start["reply"]["kind"] == "data_request" and start["status"] == "waiting"
        csv = "date,well,volume_m3\n2015-01-01,W1,1000\n2016-01-01,W1,1100\n"
        nxt = mcp_server.studio_say(start["workspace"], "", tables={"abstraction.csv": csv})
        assert nxt["status"] in ("review", "declined", "waiting")
        assert "upload:abstraction.csv" in nxt["workspace"]["tables"]
        cont = mcp_server.studio_say(start["workspace"], "continue without")
        assert cont["reply"]["kind"] == "plan" and cont["status"] == "review"
