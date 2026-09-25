"""The flood checklist: what the study must know, asked one question at a time, only when the words leave it
open, and each answer changes the plan."""

from __future__ import annotations

from aquascope import playbooks as pbk
from aquascope.studio.roles import consultant, interpreter
from aquascope.studio.workspace import Workspace


def _ws() -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    return ws


def test_the_flood_checklist_is_valid_and_ordered():
    pb = pbk.load("flood_risk")
    assert pbk.validate(pb) == []
    assert [i.field for i in pb.checklist] == ["decision", "return_period", "years"]
    assert [i.field for i in pbk.checklist_open(pb, {}, [])] == ["decision"], "later items wait for the goal"
    assert [i.field for i in pbk.checklist_open(pb, {"decision": "flood trend"}, ["decision"])] == ["years"]
    assert [i.field for i in pbk.checklist_open(pb, {"decision": "insurance"}, ["decision"])] == ["return_period"]
    assert pbk.checklist_open(pb, {"return_period": 100}, [])[0].field == "decision", "a default is not an answer"


def test_a_bad_checklist_is_refused():
    broken = {"id": "b", "title": "B", "problem": "flood_risk",
              "intake": [{"name": "rp", "type": "int", "min": 2}],
              "checklist": [{"field": "nope", "ask": "?"},
                            {"field": "rp", "ask": "?", "options": [{"value": 1}, {"value": 5, "match": "("}]}],
              "branches": [{"id": "x", "steps": [{"id": "s1", "tool": "anywhere"}]}]}
    errors = " ".join(pbk.validate(broken))
    assert "checklist nope: not an intake field" in errors
    assert "option 1 is not a value" in errors and "is not a pattern" in errors


def test_getting_worse_is_a_trend_and_only_the_window_is_asked():
    ws = _ws()
    msg = consultant.consult(ws, None, "Is flooding at this river getting worse?")
    b = ws.brief
    assert b.decision == "flood trend" and b.stated == ["decision"]
    assert msg.kind == "questions" and [q.id for q in b.open_questions] == ["years"]
    q = b.open_questions[0]
    assert q.why and q.options == ["The whole record", "The last 50 years", "The last 30 years"]
    assert q.default == "The whole record" and "1. The whole record" in msg.text
    assert "trend" in b.quantities[0], "the goal sets what the answer is"
    consultant.consult(ws, None, "2")
    assert b.open_questions and b.open_questions[0].id == "years", "a reply that names no option is asked again"
    consultant.consult(ws, None, "The last 30 years")
    assert b.ready and b.intake["years"] == 30 and "trend in annual flood peaks" in b.quantities[0]


def test_one_question_at_a_time_and_the_goal_opens_the_next():
    ws = _ws()
    consultant.consult(ws, None, "Tell me about floods here")
    assert [q.id for q in ws.brief.open_questions] == ["decision"]
    consultant.consult(ws, None, "The flow to design for (bridge, culvert, levee)")
    assert ws.brief.decision == "design flow" and [q.id for q in ws.brief.open_questions] == ["return_period"]
    consultant.consult(ws, None, "200-year")
    assert ws.brief.ready and ws.brief.intake["return_period"] == 200
    assert [q.id for q in ws.brief.questions] == ["decision", "return_period"]


def test_a_question_that_says_everything_is_not_asked_anything():
    ws = _ws()
    msg = consultant.consult(ws, None, "Design flow for a road crossing, 100-year return period")
    assert msg.kind == "brief" and ws.brief.ready and ws.brief.questions == []


def test_just_go_takes_the_defaults_and_says_so():
    ws = _ws()
    consultant.consult(ws, None, "flood question")
    consultant.consult(ws, None, "just go")
    b = ws.brief
    assert b.ready and b.intake["decision"] == "design flow" and b.intake["return_period"] == 100
    assert any("100-year" in a for a in b.assumptions)


def test_the_trend_goal_plans_the_trend_branch_with_the_window():
    from tests.test_playbooks import LONG

    study = pbk.plan("flood_risk", LONG, {"decision": "flood trend", "years": 30})
    assert study.plan["branch"] == "trend"
    assert [s.tool for s in study.steps] == ["describe_catchment", "analyze_station", "flood_frequency"]
    assert study.step_by_id("s2").arguments["years"] == 30
    assert "trend_on_series" not in {g["check"] for g in study.step_by_id("s2").expects}, (
        "a significant trend is the answer, not a failed gate")


def test_a_trend_goal_is_answered_by_the_trend():
    ws = _ws()
    ws.brief.intake = {"decision": "flood trend"}
    key = [{"label": "Mann-Kendall p-value (annual maxima)", "value": 0.32},
           {"label": "Sen's slope", "value": 0.24, "unit": "m3/s per year"}]
    assert interpreter._asks_trend(ws)
    assert interpreter._trend_verdict(key) == ", Mann-Kendall p = 0.32: no significant trend at 5 %"
    assert interpreter._trend_verdict([{"label": "Mann-Kendall p-value", "value": 0.01}]).endswith(
        "a significant trend at 5 %")


# ── a goal the gauge cannot answer, and the replies that cannot be used ──

def _trend_ws(here_years: float, here_end: str = "2026-09-01") -> Workspace:
    from aquascope.studio.workspace import Dataset, Inventory

    ws = _ws()
    ws.brief.playbook, ws.brief.intake = "flood_risk", {"decision": "flood trend", "years": 30}
    ws.inventory = Inventory(datasets=[
        Dataset(id="a", kind="station", variable="discharge", source="usgs", station_id="SHORT", name="Short",
                distance_km=0.0, start="1906-10-01", end=here_end, years=here_years),
        Dataset(id="b", kind="station", variable="discharge", source="usgs", station_id="LONG", name="Long",
                lat=47.2, lon=-68.5, distance_km=2.1, start="1903-07-29", end="2026-09-22", years=123.2),
        Dataset(id="c", kind="station", variable="water_level", source="usgs", station_id="STAGE", name="Stage",
                lat=47.3, lon=-68.6, distance_km=1.0, start="2026-01-01", end="2026-09-22", years=0.2)])
    return ws


def test_a_trend_at_a_short_or_stopped_gauge_offers_the_long_records_nearby():
    from aquascope.studio.coordinator import gauge_offer

    offer = gauge_offer(_trend_ws(9.0, "1915-09-30"))
    assert offer["kind"] == "gauge" and [g["station_id"] for g in offer["gauges"]] == ["LONG"]
    assert "9 years of discharge (1906 to 1915)" in offer["why"] and "at least 30 years" in offer["why"]
    assert offer["keep"].startswith("Keep this gauge")
    assert gauge_offer(_trend_ws(40.0, "1960-01-01")) is not None, "long but stopped: no recent years"
    assert gauge_offer(_trend_ws(80.0)) is None, "a gauge that can answer is not questioned"
    design = _trend_ws(9.0)
    design.brief.intake = {"decision": "design flow"}
    assert gauge_offer(design) is None, "only a goal whose branch needs a record asks"


def test_the_gauge_choice_moves_the_study_or_keeps_it():
    from aquascope.studio.coordinator import Studio

    s = Studio(47.0, -68.0)
    s.ws.pending_request = {"kind": "gauge", "ask": "Use a gauge?", "why": "short", "keep": "Keep this gauge (x)",
                            "gauges": [{"source": "usgs", "station_id": "LONG", "name": "Long", "lat": 47.2,
                                        "lon": -68.5, "label": "Long (123 years, 2.1 km away)"}]}
    s.ws.set_status("waiting")
    reply = s._request_reply()
    q = reply.payload["questions"][0]
    assert q["options"] == ["Long (123 years, 2.1 km away)", "Keep this gauge (x)"] and q["why"] == "short"
    moved = []
    s._scout_and_plan = lambda: moved.append(dict(s.ws.site)) or "planned"
    assert s._answer_request("Long (123 years, 2.1 km away)") == "planned" and moved == [{"lat": 47.2, "lon": -68.5}]
    assert s.ws.pending_request is None
    s.ws.pending_request = {"kind": "gauge", "gauges": [], "keep": "Keep this gauge (x)", "ask": "?", "why": "."}
    s._plan = lambda: "kept"
    assert s._answer_request("Keep this gauge (x)") == "kept"


def test_a_reply_that_cannot_be_used_is_asked_again_with_the_reason():
    ws = _ws()
    consultant.consult(ws, None, "Is flooding at this river getting worse?")
    msg = consultant.consult(ws, None, "the last 10 years")
    q = ws.brief.open_questions[0]
    assert q.id == "years" and q.retry == "10 is too few: a trend needs at least 20."
    assert msg.text.startswith("10 is too few")
    consultant.consult(ws, None, "something else entirely")
    assert "could not match" in ws.brief.open_questions[0].retry
    consultant.consult(ws, None, "The last 30 years")
    assert ws.brief.ready and ws.brief.questions[-1].retry is None
