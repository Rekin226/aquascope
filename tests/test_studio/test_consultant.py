"""The Consultant: the brief from the rules, its questions, the answers, "just go", and the model path."""

from __future__ import annotations

from aquascope.studio.model import Model
from aquascope.studio.roles import consultant
from aquascope.studio.workspace import Workspace
from tests.test_studio.conftest import PROBLEM, FakeModel


def _ws() -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    return ws


def test_rules_write_the_brief_and_ask_nothing_when_every_field_has_a_default():
    ws = _ws()
    msg = consultant.consult(ws, None, PROBLEM)
    b = ws.brief
    assert msg.kind == "brief" and b.ready and b.source == "rules"
    assert b.playbook == "flood_risk" and b.kind == "flood_risk" and b.intake["return_period"] == 100
    assert b.quantities and "100-year" in b.quantities[0]
    assert b.decision == "design flow" and b.intake["decision"] == "design flow", "the decision is read off the text"
    assert not any("design flow" in a for a in b.assumptions), "a decision the text names is not an assumption"
    assert [m.role for m in ws.messages] == ["user", "consultant"]


def test_a_supply_problem_asks_for_the_demand_and_the_answer_is_coerced():
    ws = _ws()
    msg = consultant.consult(ws, None, "Can the river supply the town reliably?")
    assert msg.kind == "questions" and not ws.brief.ready
    ids = [q.id for q in ws.brief.open_questions]
    assert ids == ["demand_m3s", "decision"], "one of the demand pair, then the decision the text does not name"
    assert "1." in msg.text and "just go" in msg.text
    assert ws.brief.open_questions[1].options == ["an abstraction licence", "the size of the scheme",
                                                  "a screening of the source"]
    msg2 = consultant.consult(ws, None, "2.5 cubic metres per second, for a licence")
    assert ws.brief.ready and msg2.kind == "brief"
    assert ws.brief.intake["demand_m3s"] == 2.5 and ws.brief.intake["use"] == "municipal"
    assert ws.brief.questions[0].answer == 2.5 and ws.brief.decision == "an abstraction licence"
    assert "decision" not in ws.brief.intake, "a decision that is no playbook field stays on the brief"


def test_just_go_proceeds_on_the_defaults():
    ws = _ws()
    consultant.consult(ws, None, "Can the river supply the town reliably?")
    consultant.consult(ws, None, "just go")
    assert ws.brief.ready and all(q.answer is not None for q in ws.brief.questions)
    assert any("proceed on the defaults" in a for a in ws.brief.assumptions)


def test_no_playbook_asks_which_and_the_answer_picks_one():
    ws = _ws()
    msg = consultant.consult(ws, None, "Tell me about the water here")
    assert msg.kind == "questions" and ws.brief.open_questions[0].id == "playbook"
    assert "flood_risk" in ws.brief.open_questions[0].options
    msg2 = consultant.consult(ws, None, "drought_status please")
    assert ws.brief.playbook == "drought_status" and ws.brief.kind == "drought"
    assert msg2.kind == "questions" and [q.id for q in ws.brief.open_questions] == ["decision", "period"], \
        "the chosen playbook's gaps are asked once"
    consultant.consult(ws, None, "restrictions, the last 12 months")
    assert ws.brief.ready and ws.brief.decision == "drought restrictions" and ws.brief.period == "the last 12 months"


def test_answers_fill_open_questions_in_order_and_options_match_loosely():
    ws = _ws()
    ws.brief.problem = "x"
    ws.brief.playbook, ws.brief.kind = "flood_risk", "flood_risk"
    from aquascope.studio.workspace import Question

    ws.brief.questions = [Question(id="decision", text="What is decided?",
                                   options=["design flow", "risk screening", "insurance", "inundation extent"]),
                          Question(id="return_period", text="T?")]
    consultant.consult(ws, None, "screening, 200 years")
    assert ws.brief.intake["decision"] == "risk screening" and ws.brief.intake["return_period"] == 200
    assert ws.brief.ready


def test_the_model_writes_the_brief_within_the_playbooks(monkeypatch):
    import aquascope.explore

    monkeypatch.setattr(aquascope.explore, "assess_site", lambda *a, **k: {"stations": [], "context": {}},
                        raising=False)
    ws = _ws()
    ws.add_table("upload:flows.csv", "date,flow\n2020-01-01,1\n")
    client = FakeModel({"consultant": [
        {"decision": "size a culvert", "quantities": ["the 200-year flow with a band"], "kind": "flood_risk",
         "playbook": "flood_risk", "intake": {"return_period": "200", "decision": "design flow", "bogus": 1},
         "assumptions": ["the gauge is representative"],
         "questions": [{"id": "horizon", "text": "Design life in years?", "default": 50},
                       {"id": "q2", "text": "a"}, {"id": "q3", "text": "b"}, {"id": "q4", "text": "too many"}],
         "ready": False},
        {"answers": {"horizon": 100, "q2": "yes"}, "ready": True},
    ]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    msg = consultant.consult(ws, model, "A culvert on the Hogsmill, 200-year", tables={"upload:flows.csv": None})
    b = ws.brief
    assert msg.kind == "brief" and b.source == "model" and b.ready
    assert b.decision == "size a culvert"
    assert b.intake == {"return_period": 200, "decision": "design flow"}
    assert b.questions == [], "a checklist playbook asks only its checklist: the model's own questions are dropped"
    ctx = client.requests[0]["context"]
    assert ctx["uploads"] == {"upload:flows.csv": ["date", "flow"]} and ctx["playbooks"][0]["intake"]
    flood = next(p for p in ctx["playbooks"] if p["id"] == "flood_risk")
    assert [i["field"] for i in flood["checklist"]] == ["decision", "return_period", "years"]
    assert ws.ledger["consultant"]["calls"] == 1


def test_follow_ups_are_classified_keyless():
    ws = _ws()
    ws.brief.problem, ws.brief.playbook = PROBLEM, "flood_risk"
    ws.brief.intake = {"return_period": 100}
    ws.report = {"answer": "The 100-year flow is 520 m3/s.", "key_numbers": [{"label": "Q100", "value": 520,
                                                                             "unit": "m3/s", "step": "s3"}]}
    q = consultant.classify_follow_up(ws, None, "how sure can we be about that?")
    assert q["kind"] == "question" and "520" in q["answer"] and "Q100: 520 m3/s" in q["answer"]
    c = consultant.classify_follow_up(ws, None, "redo it with a 500-year return period")
    assert c == {"kind": "change", "intake": {"return_period": 500}, "request": "redo it with a 500-year return period"}
    c2 = consultant.classify_follow_up(ws, None, "add the GloFAS cross-check instead")
    assert c2["kind"] == "change" and c2["intake"] == {}


def test_follow_ups_are_classified_by_the_model():
    ws = _ws()
    ws.brief.problem = PROBLEM
    ws.report = {"answer": "520 m3/s"}
    ws.run = {"results": [{"id": "s1", "tool": "flood_frequency", "ok": True, "result": {"unit": "m3/s"}}]}
    client = FakeModel({"consultant": [{"kind": "question", "answer": "About 520 m3/s, uk_ea 3400TH."},
                                       {"kind": "change", "intake": {"return_period": 200}, "request": "T = 200"}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    assert consultant.classify_follow_up(ws, model, "what was it?")["answer"].startswith("About 520")
    assert consultant.classify_follow_up(ws, model, "200 years")["intake"] == {"return_period": 200}
    assert client.requests[0]["context"]["results"][0]["tool"] == "flood_frequency"
