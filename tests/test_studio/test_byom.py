"""Bring your own model: a brief, a plan and prose a caller's own model wrote go through the crew's own
coercion, validator and checks; the contexts the roles would send are exported; the MCP tools mirror it;
the prompts ship as JSON."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from aquascope import mcp_server as m
from aquascope.studio import prompts
from tests.test_studio.conftest import PROBLEM, RECON, RICH, FakeModel, patched
from tests.test_studio.test_methodologist import VALID_PLAN

ROOT = Path(__file__).resolve().parents[2]


# ── say(proposed=...) ──


def test_a_proposed_brief_is_merged_and_coerced_then_the_gaps_are_asked(studio_factory):
    s, _ = studio_factory(recon_value=RICH)
    r = s.say("How dry is it here?", proposed={"brief": {
        "decision": "hosepipe restrictions", "quantities": ["SPI-3 and SPI-12 now"], "kind": "drought",
        "intake": {"timescales": "3, 12", "drought_concern": "water supply", "bogus": 1, "flash_drought": "no"},
        "assumptions": ["the ERA5 cell stands for the town"]}, "source": "device"})
    b = s.workspace.brief
    assert b.source == "device" and b.playbook == "drought_status" and b.kind == "drought"
    assert b.decision == "hosepipe restrictions" and b.quantities == ["SPI-3 and SPI-12 now"]
    assert b.intake == {"timescales": [3, 12], "drought_concern": "water supply", "flash_drought": False}, \
        "coerced to the playbook's fields, the unknown one dropped"
    assert "the ERA5 cell stands for the town" in b.assumptions
    assert r.kind == "questions" and [q["id"] for q in r.questions] == ["period"], "the gap the device left"
    r2 = s.say("the last 12 months")
    assert r2.kind == "plan" and b.period == "the last 12 months" and b.ready
    assert s.workspace.study["steps"][0]["arguments"]["timescales"] == [3, 12]


def test_a_proposed_brief_that_covers_everything_goes_straight_to_the_plan(studio_factory):
    s, calls = studio_factory()
    r = s.say("A culvert on the Thames", proposed={"brief": {"decision": "design flow", "playbook": "flood_risk",
                                                            "intake": {"return_period": "200"}}})
    assert r.kind == "plan" and s.workspace.brief.source == "device"
    assert s.workspace.brief.intake["return_period"] == 200 and s.workspace.brief.decision == "design flow"
    assert "T = 200" in r.text and calls == []


def test_a_proposed_brief_answers_open_questions(studio_factory):
    s, _ = studio_factory()
    r = s.say("Can the river supply the town reliably?")
    assert [q["id"] for q in r.questions] == ["demand_m3s", "decision"]
    r2 = s.say("", proposed={"brief": {"intake": {"demand_m3s": 2}, "decision": "an abstraction licence"}})
    assert r2.kind == "plan" and s.workspace.brief.intake["demand_m3s"] == 2.0
    assert s.workspace.brief.decision == "an abstraction licence" and s.workspace.brief.source == "device"


# ── approve(plan=...) ──


def test_a_proposed_plan_is_adopted_and_says_who_wrote_it(studio_factory):
    s, calls = studio_factory()
    assert s.say(PROBLEM).kind == "plan"
    r = s.approve(plan={**VALID_PLAN, "source": "device"})
    ws = s.workspace
    assert r.kind == "report" and r.payload["plan_used"] == "proposed" and r.payload["plan_errors"] == []
    assert ws.study["author"] == "device" and ws.study["plan"]["author"] == "device"
    assert ws.study["plan"]["proposal"] == {"source": "device", "errors": [], "used": "proposed"}
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    assert ws.report["footer"]["plan_author"] == "device"
    assert any(m.kind == "plan" and "device" in m.text for m in ws.messages), "the adopted plan is announced"
    assert any("Wasko" in c for c in ws.study["plan"]["caveats"]), "the playbook's caveats still apply"


def test_a_proposed_plan_is_repaired_and_pruned(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    plan = dict(VALID_PLAN, steps=[VALID_PLAN["steps"][0], dict(VALID_PLAN["steps"][1], method="fao56_et0"),
                                   VALID_PLAN["steps"][2],
                                   {"id": "s4", "tool": "frobnicate", "arguments": {}, "rationale": "x"},
                                   dict(VALID_PLAN["steps"][3], id="s5", depends_on=["s4"])])
    r = s.approve(plan=plan)
    ws = s.workspace
    assert r.kind == "report" and r.payload["plan_used"] == "proposed"
    assert any("frobnicate" in e for e in r.payload["plan_errors"])
    assert [st["id"] for st in ws.study["steps"]] == ["s1", "s2", "s3"], "the invalid step and its dependant went"
    assert ws.study["steps"][1]["method"] == "at_site_flood_frequency", "a wrong method is replaced, not fatal"
    notes = ws.study["plan"]["notes"]
    assert any("fao56_et0" in n for n in notes) and any("s4 removed" in n for n in notes)
    assert ws.study["plan"]["author"] == "device" and [c[0] for c in calls][:3] == ["describe_catchment",
                                                                                     "analyze_station",
                                                                                     "flood_frequency"]


def test_a_singular_return_period_argument_is_repaired_onto_the_list(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    plan = {"objective": "the design flow", "source": "device", "steps": [
        {"id": "s1", "tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": "3400TH"}},
        {"id": "s2", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "3400TH",
                                                              "return_period": 200, "return_periods": [2, 100]}}]}
    r = s.approve(plan=plan)
    ws = s.workspace
    assert r.kind == "report" and r.payload["plan_used"] == "proposed" and r.payload["plan_errors"] == []
    assert [st["tool"] for st in ws.study["steps"]] == ["analyze_station", "flood_frequency"]
    assert ws.study["steps"][1]["arguments"] == {"source": "uk_ea", "station_id": "3400TH",
                                                 "return_periods": [2, 100, 200]}
    assert any("return_period 200 became return_periods" in n for n in ws.study["plan"]["notes"])
    assert calls[1][1]["return_periods"] == [2, 100, 200]


def test_a_proposed_plan_with_nothing_valid_falls_back_to_the_tree(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    r = s.approve(plan={"steps": [{"id": "s1", "tool": "frobnicate", "arguments": {}}], "source": "gemma"})
    ws = s.workspace
    assert r.kind == "report" and r.payload["plan_used"] == "tree" and r.payload["plan_errors"]
    assert ws.study["plan"]["author"] == "playbook" and ws.study["plan"]["branch"] == "at_site"
    assert ws.study["plan"]["proposal"]["used"] == "tree" and ws.study["plan"]["proposal"]["source"] == "gemma"
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    assert any(e["event"] == "fallback" and "gemma" in e["detail"] for e in ws.events)
    s2, _ = studio_factory()
    s2.say(PROBLEM)
    r2 = s2.approve(plan={"steps": []})
    assert r2.kind == "report" and r2.payload["plan_used"] == "tree" and "no steps" in r2.payload["plan_errors"][0]


def test_a_proposed_plan_with_edits_and_before_review(studio_factory):
    s, calls = studio_factory()
    assert s.approve(plan=VALID_PLAN).kind == "answer", "no plan to approve before the review"
    s.say(PROBLEM)
    r = s.approve(edits={"s1": None}, plan=VALID_PLAN)
    assert r.kind == "report" and r.payload["plan_used"] == "proposed"
    assert [c[0] for c in calls] == ["analyze_station", "flood_frequency", "anywhere"], "the edit applies after"


# ── narrate(...) ──


def test_narrate_replaces_sections_after_the_number_check(studio_factory):
    from aquascope.studio.roles.author import to_markdown

    s, _ = studio_factory()
    s.say(PROBLEM)
    s.approve()
    ws = s.workspace
    before = {x["id"]: x["text"] for x in ws.report["sections"]}
    events = len([e for e in ws.events if e["event"] == "deliverables_unavailable"])
    r = s.narrate({
        "summary": "The 100-year flow at Kingston (uk_ea 3400TH) is 520 m3/s by GEV, 548 m3/s by LP3. "
                   "The town has 77777 residents. The record runs 39.9 years.",
        "results-s3": "- GEV gives 520 m3/s (90 % band 420 to 650 m3/s).\n- LP3 gives 548 m3/s.\n"
                      "- A 1968 flood reached 900 m3/s.",
        "answer": "About 520 m3/s at Kingston (uk_ea 3400TH), 90 % band 420 to 650 m3/s.",
        "references": "not allowed", "nope": "unknown section", "limitations": "   ",
    }, source="device")
    assert r.kind == "report" and r.payload["dropped"] == 4
    assert r.payload["ignored"] == ["answer", "references", "nope"]
    after = {x["id"]: x["text"] for x in ws.report["sections"]}
    assert after["summary"] == ("The 100-year flow at Kingston (uk_ea 3400TH) is 520 m3/s by GEV, 548 m3/s by LP3. "
                                "The record runs 39.9 years.")
    assert after["results-s3"] == "- LP3 gives 548 m3/s."
    assert after["problem"] == before["problem"] and after["limitations"] == before["limitations"]
    assert "About 520" not in ws.report["answer"] and "(established)" in r.text
    written_by = ws.report["written_by"]
    assert written_by["summary"] == "device" and written_by["results-s3"] == "device"
    assert written_by["answer"] == "template"
    assert written_by["problem"] == "template" and ws.report["footer"]["written_by"] == written_by
    assert r.payload["written_by"] == written_by
    md = to_markdown(ws)
    assert "device wrote summary, results-s3" in md
    assert ws.report["critique"]["checks"] and ws.report["not_established"] == []
    assert any(e["event"] == "dropped" for e in ws.events if e["role"] == "critic")
    assert len([e for e in ws.events if e["event"] == "deliverables_unavailable"]) == events + 1, "rebuilt"
    assert ws.messages[-1].kind == "report" and ws.messages[-1].payload["written_by"] == written_by


def test_narrate_rebuilds_the_deliverables_and_keeps_recommendations_as_a_list(studio_factory, monkeypatch):
    s, _ = studio_factory()
    s.say(PROBLEM)
    s.approve()
    built: list = []
    pkg = types.ModuleType("aquascope.studio.deliverables")
    pkg.build = lambda ws: built.append(ws.id) or []
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables", pkg)
    r = s.narrate([{"id": "recommendations", "text": "- Adopt 520 m3/s.\n- Quote the 420 to 650 m3/s band."}])
    assert r.kind == "report" and r.payload["dropped"] == 1 and built == [s.workspace.id]
    assert s.workspace.report["recommendations"] == ["Adopt 520 m3/s."]
    assert any(e["event"] == "deliverables" for e in s.workspace.events)


def test_narrate_needs_a_report(studio_factory):
    s, _ = studio_factory()
    s.say(PROBLEM)
    assert s.narrate({"summary": "x"}).kind == "answer"
    s.approve()
    r = s.narrate({"summary": "The town has 77777 residents."})
    assert r.payload["dropped"] == 1 and r.payload["ignored"] == ["summary"], "nothing survived: the template stands"
    assert s.workspace.report["written_by"]["summary"] == "template"


# ── the contexts ──


def test_the_contexts_are_what_the_roles_send_a_model(studio_factory):
    client = FakeModel({
        "consultant": [{"decision": "size the crossing", "kind": "flood_risk", "playbook": "flood_risk",
                        "intake": {"return_period": 100}, "questions": [], "ready": True}],
        "methodologist": [VALID_PLAN],
        "author": [{"answer": "About 520 m3/s at uk_ea 3400TH (band 420 to 650).", "sections": {}}],
        "critic": [{"issues": []}],
    })
    s, _ = studio_factory(client=client)
    s.say("A culvert on the Thames at Kingston, 100-year")
    s.approve()
    k, _ = studio_factory()
    ctx = k.consultant_context("A culvert on the Thames at Kingston, 100-year")
    assert ctx["system"] == prompts.CONSULTANT
    assert json.loads(json.dumps({x: y for x, y in ctx.items() if x != "system"})) == \
        client.calls("consultant")[0]["context"]
    k.say("A culvert on the Thames at Kingston, 100-year", proposed={"brief": {"intake": {"return_period": 100},
                                                                             "decision": "size the crossing"}})
    mctx = k.methodologist_context()
    assert mctx["system"] == prompts.METHODOLOGIST and mctx["exemplar"]["branch"] == "at_site"
    assert json.loads(json.dumps({x: y for x, y in mctx.items() if x != "system"})) == \
        client.calls("methodologist")[0]["context"]
    k.approve(plan=VALID_PLAN)
    actx = k.author_context()
    assert actx["system"] == prompts.AUTHOR and actx["section_ids"][:3] == ["summary", "decision", "findings"]
    sent = client.calls("author")[0]["context"]
    assert json.loads(json.dumps({x: y for x, y in actx.items() if x != "system"})) == sent
    fix = k.author_context(issues=[{"section": "summary", "severity": "fix", "text": "t", "fix": "f"}])
    assert fix["system"] == prompts.AUTHOR_FIX and fix["draft"]["sections"]["summary"] and fix["issues"]
    chg = k.methodologist_context("add the donors")
    assert chg["system"] == prompts.METHODOLOGIST_CHANGE and chg["request"] == "add the donors"
    assert chg["steps"][0]["outcome"]["ok"] is True


# ── MCP ──


def test_the_mcp_tools_mirror_the_entry_points(no_deliverables):
    with patched():
        start = m.studio_start("Can the river supply the town reliably?", 51.415, -0.308)
        assert start["reply"]["kind"] == "questions"
        ctx = m.studio_context(start["workspace"], "consultant", text="2 m3/s for a licence")
        assert ctx["role"] == "consultant" and ctx["context"]["system"] == prompts.CONSULTANT_ANSWERS
        assert ctx["context"]["questions"][0]["id"] == "demand_m3s" and ctx["status"] == "intake"
        said = m.studio_say(start["workspace"], "", proposed={"brief": {"intake": {"demand_m3s": 2},
                                                                        "decision": "an abstraction licence"}})
        assert said["reply"]["kind"] == "plan" and said["workspace"]["brief"]["source"] == "device"
        mctx = m.studio_context(said["workspace"], "methodologist")
        assert mctx["context"]["system"] == prompts.METHODOLOGIST and mctx["context"]["brief"]["playbook"]
        plan = {"steps": [{"id": "s1", "tool": "supply_reliability",
                           "arguments": {"lat": 51.415, "lon": -0.308, "demand_m3s": 2}, "rationale": "screen",
                           "expects": [{"check": "not_empty", "path": "reliability"}]}], "objective": "screen",
                "methodology": ["Screen the demand against the FDC."], "source": "device"}
        done = m.studio_approve(said["workspace"], plan=plan)
        assert done["reply"]["kind"] == "report" and done["reply"]["payload"]["plan_used"] == "proposed"
        assert done["workspace"]["study"]["plan"]["author"] == "device"
        actx = m.studio_context(done["workspace"], "author")
        assert actx["context"]["system"] == prompts.AUTHOR
        told = m.studio_narrate(done["workspace"], {"summary": "The demand of 2 m3/s is met on 61 % of days. "
                                                              "Paris has 2000000 people."}, source="device")
        assert told["reply"]["kind"] == "report" and told["reply"]["payload"]["dropped"] == 1
        assert told["workspace"]["report"]["written_by"]["summary"] == "device"
        assert "error" in m.studio_context(done["workspace"], "scout")
        assert "error" in m.studio_narrate({}, {"summary": "x"})
    import asyncio

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert {"studio_narrate", "studio_context"} <= names


# ── the prompts ship as JSON ──


def test_explorer_prompts_json_matches_the_module():
    shipped = (ROOT / "explorer" / "prompts.json").read_text(encoding="utf-8")
    assert shipped == prompts.as_json(), "explorer/prompts.json is stale: run `python -m aquascope.studio.prompts`"
    data = json.loads(shipped)
    assert set(data) >= {"consultant", "consultant_answers", "consultant_follow_up", "methodologist",
                         "methodologist_repair", "methodologist_change", "author", "author_fix", "critic",
                         "specialist", "schemas", "version"}
    assert data["consultant"] == prompts.CONSULTANT and data["version"] == prompts.VERSION
    assert set(data["schemas"]) == {"brief", "plan", "sections", "findings"}
    assert data["schemas"]["plan"]["properties"]["steps"]["items"]["required"] == ["id", "tool", "arguments",
                                                                                     "rationale"]
    assert "\u2014" not in shipped and "\u2013" not in shipped, "no dashes in what the page reads"


def test_the_prompts_command_writes_a_file(tmp_path):
    out = tmp_path / "p.json"
    prompts.main([str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["schemas"]["brief"]["required"]


def test_prompts_stay_tight():
    for key, value in prompts.as_dict().items():
        if isinstance(value, str) and key not in ("generated_by",):
            assert len(value) < 2500, key


def _two_gauges() -> dict:
    """RECON with a second discharge gauge, for the follow-ups that add one."""
    second = dict(RECON["stations"][0], station_id="3401TH", name="Teddington", distance_km=3.0, years=30.0)
    return {**RECON, "stations": [RECON["stations"][0], second]}


def test_follow_ups_add_steps_without_a_model(studio_factory):
    s, calls = studio_factory(recon_value=_two_gauges())
    s.say(PROBLEM)
    s.approve()
    ws = s.workspace
    n = len(calls)
    r = s.follow_up("add the flow duration curve and a trend test")
    assert r.kind == "report" and [st["id"] + ":" + st["tool"] for st in ws.study["steps"]] == [
        "s1:describe_catchment", "s2:analyze_station", "s3:flood_frequency", "s4:anywhere", "s5:low_flow_context",
        "s6:analyze_station"]
    assert ws.study["steps"][5]["method"] == "trend_mann_kendall" and ws.study["plan"]["added"] == ["s5", "s6"]
    assert [c[0] for c in calls[n:]] == ["low_flow_context", "analyze_station"], "the earlier steps are reused"
    assert ws.follow_ups[-1]["kind"] == "change" and ws.follow_ups[-1]["steps"][-1] == "s6"
    assert len(ws.study["plan"]["methodology"]) == 6
    assert any(k["label"] == "Baseflow index" for k in ws.report["key_numbers"])
    r = s.follow_up("compare with the donors")
    assert [st["tool"] for st in ws.study["steps"]][-2:] == ["similar_basins", "regionalize_signatures"]
    assert ws.study["steps"][-1]["arguments"] == {"lat": 51.415, "lon": -0.308, "k": 10}
    r = s.follow_up("also include the upstream gauge")
    assert r.kind == "report" and ws.study["steps"][-1]["arguments"] == {"source": "uk_ea", "station_id": "3401TH"}
    assert "Teddington" in ws.study["steps"][-1]["rationale"]
    r = s.follow_up("is this area in drought? add the SPI")
    assert ws.study["steps"][-1]["tool"] == "drought_indices" and ws.study["steps"][-1]["method"] == "spei_reanalysis"
    r = s.follow_up("and the baseflow, plus the GloFAS cross-check")
    assert [st["tool"] for st in ws.study["steps"]][-2:] == ["low_flow_context", "anywhere"]
    assert ws.study["steps"][-2]["method"] == "baseflow_separation"
    assert [st["id"] for st in ws.study["steps"]] == [f"s{i}" for i in range(1, 13)]
    assert ws.status == "done" and ws.run["ok"]


def test_a_return_period_change_carries_the_added_steps_and_unknown_requests_are_honest(studio_factory):
    s, calls = studio_factory(recon_value=_two_gauges())
    s.say(PROBLEM)
    s.approve()
    ws = s.workspace
    s.follow_up("add the flow duration curve")
    n = len(calls)
    r = s.follow_up("redo it with a 50-year return period")
    assert r.kind == "report" and ws.brief.intake["return_period"] == 50
    assert [st["tool"] for st in ws.study["steps"]] == ["describe_catchment", "analyze_station", "flood_frequency",
                                                        "anywhere", "low_flow_context"]
    assert ws.study["plan"]["added"] == ["s5"] and [c[0] for c in calls[n:]] == ["flood_frequency", "anywhere"]
    r = s.follow_up("add a rainfall-runoff model instead")
    assert r.kind == "answer" and "cannot add that without a model" in r.text and "another gauge" in r.text
    assert ws.follow_ups[-1]["kind"] == "change" and ws.status == "done"
    assert len(ws.study["steps"]) == 5, "the plan is untouched"
    s2, _ = studio_factory()
    s2.say(PROBLEM)
    s2.approve()
    s2.follow_up("also include the upstream gauge")
    r = s2.follow_up("also include the upstream gauge")
    assert r.kind == "answer" and "no other discharge gauge within reach" in r.text
