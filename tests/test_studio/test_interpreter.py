"""The Interpreter (#417) and the grades (#418): findings with basis paths, the decision block, the rules
keyless, the model held to the results, and the report built from them."""

from __future__ import annotations

import json

from aquascope.studio.model import Model
from aquascope.studio.roles import author, critic, interpreter
from aquascope.studio.workspace import Workspace
from tests.test_studio.conftest import FLOW, RECON, FakeModel, fake_tools
from tests.test_studio.test_critic_author import _ran
from tests.test_studio.test_methodologist import VALID_PLAN


def test_the_rules_write_findings_that_point_at_the_results_and_a_decision_with_its_band():
    ws = _ran()
    out = interpreter.interpret(ws, None)
    assert out["written_by"] == "rules" and out["findings"] and ws.findings is out
    for f in out["findings"]:
        assert interpreter.resolve_basis(ws, f["basis"][0]) is not None, f
        assert f["grade"] in interpreter.GRADES
    head = next(f for f in out["findings"] if f["claim"].startswith("100-year return level, GEV"))
    assert head["basis"] == ["s3.ffa.fits.gev_lmoments.q.5"] and head["grade"] == "established"
    d = out["decision"]
    assert d["value"] == 520 and d["unit"] == "m3/s" and d["band"] is None and d["grade"] == "established"
    assert d["answer"].startswith("Design flow for a road crossing") and "(established)" in d["answer"]
    assert d["basis"] == ["s3.ffa.fits.gev_lmoments.q.5"]
    kinds = {c["a"].split(": ")[1] for c in out["consistency"]}
    assert {"spread_within", "fit_envelopes_max", "cross_check_ratio", "trend_on_series"} <= kinds
    assert all(c["agree"] for c in out["consistency"])
    assert any(e["role"] == "interpreter" and e["event"] == "findings" for e in ws.events)


def test_grades_follow_the_run_a_fallback_is_indicative_a_failed_answer_step_is_not_established():
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900        # the fits disagree: s3 fails its gate, the donors run
    ws = _ran(tools=fake_tools([], flood_frequency=wide, analyze_station=wide,
                               similar_basins={"k": 5, "stations": [{"source": "usgs", "station_id": "1"}] * 5}))
    out = interpreter.interpret(ws, None)
    assert interpreter.grade_for_step(ws, "s3") == "indicative", "the fallback carried the step"
    assert out["decision"]["grade"] == "indicative"
    assert any("spread_within" in c for c in out["decision"]["limitations"])
    ws2 = _ran(tools=fake_tools([], flood_frequency=wide, analyze_station=wide,
                                similar_basins={"k": 1, "stations": []}))
    out2 = interpreter.interpret(ws2, None)
    assert interpreter.grade_for_step(ws2, "s3") == "not_established"
    assert out2["decision"]["grade"] == "not_established" and "No number" in out2["decision"]["answer"] or \
        out2["decision"]["grade"] == "not_established"
    assert any("s3" in w for w in out2["decision"]["what_would_change_it"])


def test_a_regional_plan_is_screening_grade():
    from tests.test_studio.conftest import UNGAUGED

    ws = _ran(playbook="ungauged_flow", intake={}, recon_value=UNGAUGED)
    out = interpreter.interpret(ws, None)
    assert out["decision"]["grade"] == "screening", (ws.study["plan"].get("branch"), out["decision"])
    assert all(f["grade"] in ("screening", "not_established") for f in out["findings"])


def test_the_model_is_held_to_the_results_and_may_only_lower_a_grade():
    ws = _ran()
    reply = {
        "findings": [
            {"id": "a", "claim": "The 100-year flow by GEV is 520 m3/s at Kingston.",
             "basis": ["s3.ffa.fits.gev_lmoments.q.5"], "grade": "established"},
            {"id": "b", "claim": "The 100-year flow is 777 m3/s.", "basis": ["s3.ffa.fits.gev_lmoments.q.5"],
             "grade": "established"},
            {"id": "c", "claim": "Something at nowhere.", "basis": ["s9.nothing.here"], "grade": "established"},
            {"id": "d", "claim": "The LP3 fit gives 548 m3/s, read with caution.",
             "basis": ["s3.ffa.fits.lp3.q.5"], "grade": "indicative"},
            {"id": "e", "claim": "Upstream area 9948 km2.", "basis": ["s1.attributes.upstream_area_km2"],
             "grade": "established"},
        ],
        "decision": {"answer": "Adopt 520 m3/s (band 420 to 650 m3/s) as the design flow (established).",
                     "value": 520, "unit": "m3/s", "band": [420, 650], "grade": "established",
                     "conditions": ["the record stays stationary"], "what_would_change_it": ["a longer record"]},
        "data_requests": [{"what": "the agency's peak series", "why": "to firm the fit", "effect_on_grade": "none"}],
        "assumptions": ["the gauge is representative"],
    }
    client = FakeModel({"interpreter": [reply]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    out = interpreter.interpret(ws, model)
    assert out["written_by"] == "model" and out["dropped"] == 2, "777 is at no basis; s9 does not exist"
    claims = [f["claim"] for f in out["findings"]]
    assert claims[0].startswith("The 100-year flow by GEV") and "777" not in " ".join(claims)
    lp3 = next(f for f in out["findings"] if "LP3" in f["claim"])
    assert lp3["grade"] == "indicative", "a model may lower a grade"
    area = next(f for f in out["findings"] if "Upstream" in f["claim"])
    assert area["grade"] == "screening", "a model may not raise the rule's grade (BasinATLAS is screening)"
    d = out["decision"]
    assert d["value"] == 520 and d["band"] is None and d["grade"] == "established"
    assert "band 420 to 650" not in d["answer"] and d["conditions"] == ["the record stays stationary"]
    assert out["data_requests"][0]["what"] == "the agency's peak series"
    assert ws.ledger["interpreter"]["calls"] == 1


def test_a_model_decision_value_at_no_basis_is_replaced_by_the_rules_and_a_grade_raised_is_lowered():
    ws = _ran()
    gev = {"claim": "GEV 520 m3/s.", "basis": ["s3.ffa.fits.gev_lmoments.q.5"], "grade": "established"}
    reply = {"findings": [gev],
             "decision": {"answer": "Adopt 999 m3/s (established).", "value": 999, "unit": "m3/s",
                          "grade": "established"}}
    model = Model.resolve(ws, client=FakeModel({"interpreter": [reply]}), model="fake", provider="custom")
    out = interpreter.interpret(ws, model)
    assert out["decision"]["value"] == 520 and "999" not in out["decision"]["answer"]
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    ws2 = _ran(tools=fake_tools([], flood_frequency=wide, analyze_station=wide,
                                similar_basins={"k": 5, "stations": [{"source": "usgs", "station_id": "1"}] * 5}))
    reply2 = {"findings": [dict(gev)],
              "decision": {"answer": "Adopt 520 m3/s (established).", "value": 520, "unit": "m3/s",
                           "grade": "established"}}
    model2 = Model.resolve(ws2, client=FakeModel({"interpreter": [reply2]}), model="fake", provider="custom")
    out2 = interpreter.interpret(ws2, model2)
    assert out2["findings"][0]["grade"] == "indicative" and out2["decision"]["grade"] == "indicative"
    assert out2["decision"]["answer"].endswith("(indicative).")


def test_the_report_opens_with_the_decision_and_carries_the_findings_sections_and_the_bundle_files():
    ws = _ran()
    interpreter.interpret(ws, None)
    report = author.author_report(ws, None)
    assert report["answer"].startswith("Design flow for a road crossing") and "(established)" in report["answer"]
    assert report["grade"] == "established" and report["decision"]["value"] == 520 and report["findings"]
    ids = [s["id"] for s in report["sections"]]
    assert ids[:3] == ["summary", "decision", "findings"]
    decision = next(s["text"] for s in report["sections"] if s["id"] == "decision")
    assert "(established)" in decision and "band 420 to 650" not in decision
    findings = next(s["text"] for s in report["sections"] if s["id"] == "findings")
    assert "[established] 100-year return level, GEV" in findings and "s3.ffa.fits.gev_lmoments.q.5" in findings
    recs = next(s["text"] for s in report["sections"] if s["id"] == "recommendations")
    assert recs.startswith("- Adopt this as the answer to the decision")
    out = critic.critique(ws, None)
    names = {c["name"]: c["passed"] for c in out["checks"]}
    assert names["findings_resolve"] and names["decision_in_answer"]
    from aquascope.studio.deliverables import bundle, workbook

    made = bundle.build(ws, formats=["findings", "xlsx"])
    assert any(a.name == "findings.json" for a in made)
    data = json.loads(next(a for a in made if a.name == "findings.json").data)
    assert data["decision"]["value"] == 520 and data["findings"][0]["basis"]
    import io

    import openpyxl

    book = openpyxl.load_workbook(io.BytesIO(workbook.workbook_bytes(ws)), read_only=True)
    assert "Findings" in book.sheetnames


def test_a_keyed_author_answer_without_the_grade_gets_the_decision_in_front(studio_factory):
    client = FakeModel({
        "consultant": [{"decision": "size the crossing", "quantities": ["the 100-year flow with a band"],
                        "kind": "flood_risk", "playbook": "flood_risk", "intake": {"return_period": 100},
                        "assumptions": [], "questions": [], "ready": True}],
        "methodologist": [VALID_PLAN],
        "author": [{"title": "t", "answer": "About 520 m3/s at uk_ea 3400TH by GEV L-moments.",
                    "sections": {"summary": "The flow is 520 m3/s at uk_ea 3400TH."}}],
        "critic": [{"issues": []}],
    })
    s, _ = studio_factory(client=client)
    s.say("A culvert on the Thames at Kingston, 100-year")
    r = s.approve()
    assert r.kind == "report" and r.text.startswith("size the crossing:") and "(established)" in r.text
    assert "About 520" in r.text and r.payload["grade"] == "established"
    assert set(s.workspace.ledger) >= {"interpreter", "author", "critic"}
    ctx = client.calls("interpreter")[0]["context"]
    assert ctx["rule_grade"] == "established" and ctx["draft"]["decision"]["value"] == 520


def test_the_workspace_round_trips_the_findings():
    ws = _ran()
    interpreter.interpret(ws, None)
    back = Workspace.from_dict(json.loads(ws.to_json()))
    assert back.findings["decision"]["value"] == 520 and len(back.findings["findings"]) == len(ws.findings["findings"])
    assert RECON is not None


def test_a_wrong_basis_path_is_re_anchored_when_the_number_is_in_that_steps_result():
    ws = _ran()
    gev = {"claim": "The GEV 100-year flow is 520 m3/s.", "basis": ["s3.ffa.gev.q100"], "grade": "established"}
    none = {"claim": "Nothing at 999 anywhere.", "basis": ["s3.ffa.gev.q999"], "grade": "established"}
    reply = {"findings": [gev, none],
             "decision": {"answer": "Adopt 520 m3/s (established).", "value": 520, "unit": "m3/s",
                          "grade": "established"}}
    model = Model.resolve(ws, client=FakeModel({"interpreter": [reply]}), model="fake", provider="custom")
    out = interpreter.interpret(ws, model)
    assert out["written_by"] == "model" and out["reanchored"] == 1 and out["dropped"] == 1
    assert out["findings"][0]["basis"] == ["s3.ffa.fits.gev_lmoments.q.5"]


def test_the_headline_follows_the_briefs_first_quantity_and_says_when_nothing_answers():
    ws = _ran()
    ws.brief.quantities = ["long-term trend in the level (m/year)", "the SGI for the last ten years"]
    ws.brief.kind = "groundwater_decline"
    out = interpreter.rules_findings(ws)
    head = "Design flow for a road crossing, 100-year return period: Sen's slope"
    assert out["decision"]["answer"].startswith(head), out["decision"]["answer"]
    ws2 = _ran()
    ws2.brief.quantities = ["the seasonal crop water requirement"]
    ws2.brief.kind = "irrigation_feasibility"
    out2 = interpreter.rules_findings(ws2)
    answer2 = out2["decision"]["answer"]
    assert out2["decision"]["value"] is None and "No number in the results answers the decision" in answer2
    assert "the study established" in answer2


def test_a_percentage_decision_is_found_in_the_answer():
    ws = _ran()
    interpreter.interpret(ws, None)
    ws.findings["decision"].update({"value": 82.53, "unit": "%",
                                    "answer": "Days the demand is met 82.53 % (established)."})
    author.author_report(ws, None)
    out = critic.critique(ws, None)
    assert {c["name"]: c["passed"] for c in out["checks"]}["decision_in_answer"]
