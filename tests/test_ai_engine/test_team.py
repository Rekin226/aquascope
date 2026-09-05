"""The team: keyless end to end, a mocked model that replans after a failed gate, and the review callback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import aquascope.explore
from aquascope.ai_engine import team
from aquascope.study import loads


class FakeChat:
    """Scripted OpenAI-shaped client: one text reply per call, with token usage."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.turns.pop(0)
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


STATION = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "distance_km": 0.4,
           "variables": ["discharge"], "years": 39.5}
RECON = {"point": {"lat": 51.415, "lon": -0.308}, "stations": [STATION],
         "catchment": {"upstream_area_km2": 9948, "dams": 1},
         "context": {"years_by_variable": {"discharge": 39.5}, "resolution_by_variable": {"discharge": "daily"},
                     "area_km2": 9948, "donors": 8, "available": ["glofas"], "ungauged": False},
         "sufficiency": [], "notes": ["served record starts 1986"]}
FLOW = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "license": "OGL-UK-3.0",
        "attribution": "Environment Agency", "unit": "m3/s", "variable": "discharge", "start": "1986-08-17",
        "end": "2026-08-15", "years": 39.9, "stats": {"mean": 65.2, "min": 3.1, "max": 520.0},
        "trend": {"on": "annual mean", "p_value": 0.41, "sens_slope_per_year": 0.12, "n_years": 39},
        "ffa": {"n_years": 39, "return_periods": [2, 5, 10, 25, 50, 100],
                "fits": {"gev_lmoments": {"q": [250, 330, 380, 440, 480, 520]},
                         "lp3": {"q": [252, 335, 388, 452, 500, 548], "ci": [[200, 300]] * 5 + [[410, 690]]},
                         "gev_bootstrap": {"q": [250, 330, 380, 440, 480, 520],
                                           "ci": [[210, 290]] * 5 + [[420, 650]]}}},
        "methods": [{"name": "GEV fitted by L-moments", "text": "t", "citation": "Hosking 1990"}]}
CATCHMENT = {"latitude": 51.415, "longitude": -0.308, "sub_basin": {"hybas_id": 1},
             "attributes": {"upstream_area_km2": 9948.0}, "license": "CC BY 4.0", "attribution": "BasinATLAS"}


def _tools(calls, *, flow=FLOW, donors_k=5):
    def rec(name):
        def f(**kw):
            calls.append((name, kw))
            return {
                "describe_catchment": CATCHMENT, "analyze_station": flow, "flood_frequency": flow,
                "similar_basins": {"k": donors_k, "method": "combined",
                                   "stations": [{"source": "usgs", "station_id": str(i), "name": f"D{i}"}
                                                for i in range(donors_k)]},
                "anywhere": {"latitude": 51.415, "longitude": -0.308, "start": "2006-01-01", "end": "2026-01-01",
                             "climate": {"precipitation_mm_per_year": 700.0, "et0_mm_per_year": 600.0,
                                         "aridity_index": 1.17, "aridity_class": "humid"},
                             "glofas": {"stats": {"mean": 60.0}}, "attribution": "Open-Meteo"},
            }[name]
        return f
    names = ("describe_catchment", "analyze_station", "flood_frequency", "similar_basins", "anywhere")
    return {n: rec(n) for n in names}


def _run(*args, **kwargs):
    calls = kwargs.pop("calls", [])
    tools = kwargs.pop("tools", None) or _tools(calls)
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=kwargs.pop("recon_value", RECON)), \
         patch("aquascope.study._tools", return_value=tools):
        return team.solve(*args, lat=51.415, lon=-0.308, **kwargs)


def test_keyless_solve_runs_end_to_end_with_zero_model_calls():
    calls: list = []
    events: list = []
    res = _run("Design flow for a road crossing, 100-year return period", calls=calls, on_event=events.append)
    assert not res.declined and res.ok and res.cost == {} and res.model is None
    assert not any(e["event"] == "model_call" for e in res.timeline)
    roles = [e["role"] for e in res.timeline]
    assert roles[:3] == ["coordinator", "scout", "coordinator"] and roles[-1] == "reviewer"
    assert "narrator" in roles and events == res.timeline
    assert res.study.plan["playbook"] == "flood_risk" and res.study.plan["branch"] == "at_site"
    assert res.problem["params"]["return_period"] == 100, "the return period was read off the text"
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency"]
    assert all(g["passed"] for g in res.gates) and len(res.gates) == 7
    assert "520" in res.answer and "m3/s" in res.answer and "Kingston" in res.answer
    assert all(c["passed"] for c in res.checks), res.checks
    md = res.to_markdown()
    for section in ("## Plan", "## Steps and gates", "gate spread_within: passed", "## Caveats", "Wasko et al. 2024",
                    "## Data", "uk_ea / 3400TH", "## Methods and citations", "Hosking 1990", "Model calls: 0"):
        assert section in md, section
    back = loads(res.study_yaml)
    assert [s.id for s in back.steps] == ["s1", "s2", "s3"] and back.results["s3"]["ok"]
    assert res.to_dict()["study"]["version"] == 2


def test_a_model_run_plans_replans_after_a_failed_gate_and_narrates():
    calls: list = []
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900          # the fits disagree at T = 100: spread_within fails
    tools = _tools(calls, flow=wide, donors_k=1)      # the playbook's own fallback (donors) fails its gate too
    proposal = {"tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": 20},
                "rationale": "GloFAS as an independent cross-check",
                "expects": [{"check": "not_empty", "path": "glofas"}]}
    client = FakeChat([
        "The plan rests on 39.5 years at Kingston; the gates check the record and the spread.",
        json.dumps(proposal),
        "The 100-year flow at Kingston (uk_ea 3400TH, 1986-08-17 to 2026-08-15, 39.9 years) is about 520 m3/s "
        "by GEV (90 % CI 420 to 650 m3/s); the LP3 fit gives 900 m3/s, so the fits disagree and GloFAS was "
        "used as a cross-check.",
    ])
    res = _run("Design flow for a road crossing, 100-year return period", client=client, model="fake",
               provider="custom", tools=tools)
    assert not res.declined and res.model == "fake"
    assert set(res.cost) == {"coordinator", "specialist", "narrator"}
    assert all(v == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20} for v in res.cost.values())
    assert res.study.plan["rationale"].startswith("The plan rests") and res.study.plan["tree_rationale"]
    # each role call was stateless: two messages, no growing transcript
    assert all(len(r["messages"]) == 2 and r["messages"][0]["role"] == "system" for r in client.requests)
    assert "tools" not in client.requests[0]
    step = res.study.step_by_id("s3")
    assert step.fallback["step"]["tool"] == "anywhere" and res.study.plan["replans"][0]["step"] == "s3"
    r3 = [r for r in res.run.results if r["id"] == "s3"][0]
    assert r3["fallback_used"] and r3["fallback"]["tool"] == "anywhere" and r3["fallback"]["ok"]
    assert res.run.stop_reason is None and res.ok
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "similar_basins",
                                     "flood_frequency", "anywhere"], "the passed steps were reused, not fetched again"
    kinds = [(e["role"], e["event"]) for e in res.timeline]
    assert ("specialist", "replan") in kinds and ("runner", "reused") in kinds and ("narrator", "template") not in kinds
    assert res.answer.startswith("The 100-year flow")
    assert any("spread_within" in n for n in res.not_established)
    md = res.to_markdown()
    assert "fallback `anywhere(" in md and "model fake via custom" in md and "Model calls: 3" in md


def test_without_a_model_a_failed_gate_and_failed_fallback_stop_and_are_reported():
    calls: list = []
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    res = _run("Design flow, 100-year return period", tools=_tools(calls, flow=wide, donors_k=1))
    assert not res.declined and not res.ok and res.run.stopped_at == "s3"
    assert any("stopped at s3" in n for n in res.not_established)
    assert "The study stopped at step s3" in res.answer and "**Stopped at s3:**" in res.to_markdown()
    assert res.cost == {}


def test_the_review_callback_can_edit_or_decline_the_plan():
    calls: list = []
    seen = {}

    def drop_first(study):
        seen["steps"] = [s.id for s in study.steps]
        study.steps = [s for s in study.steps if s.id != "s1"]
        return study

    res = _run("Design flow, 100-year return period", calls=calls, review=drop_first)
    assert seen["steps"] == ["s1", "s2", "s3"] and [s.id for s in res.study.steps] == ["s2", "s3"]
    assert [c[0] for c in calls] == ["analyze_station", "flood_frequency"]
    assert ("coordinator", "review") in [(e["role"], e["event"]) for e in res.timeline]

    res = _run("Design flow, 100-year return period", calls=[], review=lambda s: None)
    assert res.declined and "declined at review" in res.declined_reason and res.run is None
    assert res.study.steps, "the declined plan travels with the result"
    assert "**Declined.**" in res.to_markdown()


def test_a_playbook_decline_is_the_playbooks_own_sentence():
    res = _run("Why is the water table in my well falling? Is it the pumping?", calls=[],
               recon_value={**RECON, "stations": [dict(STATION, variables=["groundwater_level"], years=15)],
                            "context": {"years_by_variable": {"groundwater_level": 15},
                                        "resolution_by_variable": {"groundwater_level": "monthly"}}})
    assert res.declined and res.problem["kind"] == "groundwater_decline"
    assert "pumping" in res.declined_reason and res.problem["params"]["attribute_cause"] is True
    assert res.run is None and res.not_established == [res.declined_reason]


def test_no_playbook_means_a_decline_not_a_guess():
    res = _run("What is the meaning of water?", calls=[])
    assert res.declined and "No playbook covers" in res.declined_reason and "flood_risk" in res.declined_reason


def test_an_explicit_playbook_and_intake_win_over_the_text():
    calls: list = []
    res = _run("Some vague words about a river", playbook="flood_risk", intake={"return_period": "50"}, calls=calls)
    assert not res.declined and res.problem["params"]["return_period"] == 50
    gate = [g for g in res.gates if g["check"] == "max_return_period_factor"][0]
    assert "T = 50" in gate["detail"]


def test_execute_false_returns_the_plan_without_running_it():
    calls: list = []
    res = _run("Design flow, 100-year return period", calls=calls, execute=False)
    assert res.run is None and res.answer == "" and calls == [] and len(res.study.steps) == 3
    assert res.timeline[-1]["event"] == "plan_ready"


def test_when_the_scout_fails_the_plan_says_so_and_goes_regional():
    with patch.object(aquascope.explore, "assess_site", create=True, side_effect=RuntimeError("offline")), \
         patch("aquascope.study._tools", return_value=_tools([])):
        res = team.solve("Design flow, 100-year return period", lat=1.0, lon=2.0)
    assert ("scout", "error") in [(e["role"], e["event"]) for e in res.timeline]
    assert res.study.plan["branch"] == "regional" and "offline" in res.study.plan["recon_notes"][0]


@pytest.mark.parametrize("text, expected", [
    ("Design flow for a culvert, 100-year return period", ("flood_risk", False)),
    ("How much water can this ungauged stream give an irrigation scheme?", ("ungauged_flow", False)),
    ("Is the aquifer under the farm declining?", ("groundwater_decline", False)),
    ("Tell me about the weather", (None, True)),
])
def test_the_keyword_rules(text, expected):
    assert team.choose_playbook(text) == expected


def test_intake_hints_read_the_text():
    assert team.intake_hints("a 1-in-200 flood for an insurance quote") == {"return_period": 200,
                                                                            "decision": "insurance"}
    assert team.intake_hints("which streets flood, how deep, 50-year")["decision"] == "inundation extent"
    assert team.intake_hints("Q95 for an environmental flow at an ungauged creek", "ungauged_flow") == {
        "statistic": "Q95", "purpose": "environmental flow"}


# ── run_reviewed: the page's half of the loop ────────────────────────────────
# The Explorer plans with execute=False, shows the checklist, lets the reader
# edit the arguments, and hands the study back. The run must be the same code
# path as solve(): gates, replan, the Reviewer's list, the Narrator.


def _run_reviewed(study, **kwargs):
    calls = kwargs.pop("calls", [])
    tools = kwargs.pop("registry", None) or _tools(calls)     # `tools=` goes through to run_reviewed
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=RECON), \
         patch("aquascope.study._tools", return_value=tools):
        return team.run_reviewed(study, **kwargs)


def test_a_planned_study_runs_through_run_reviewed_like_solve_would():
    planned = _run("Design flow for a road crossing, 100-year return period", execute=False)
    assert planned.run is None and not planned.declined
    calls: list = []
    events: list = []
    res = _run_reviewed(planned.study.to_dict(), recon=RECON, calls=calls, on_event=events.append)
    assert res.ok and not res.declined and res.cost == {} and res.model is None
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency"]
    assert len(res.gates) == 7 and all(g["passed"] for g in res.gates)
    assert "520" in res.answer and "Kingston" in res.answer
    assert res.not_established == [] and all(c["passed"] for c in res.checks)
    assert events == res.timeline
    roles = [e["role"] for e in res.timeline]
    assert roles[0] == "scout" and "narrator" in roles and roles[-1] == "reviewer"
    assert res.timeline[0]["detail"] == "reconnaissance supplied by the caller"
    # the problem, site and intake travel inside the study; nothing is asked twice
    assert res.problem["kind"] == "flood_risk" and res.problem["params"]["return_period"] == 100
    assert res.problem["site"] == {"lat": 51.415, "lon": -0.308}
    assert "## What this answer does not establish" not in res.to_markdown()
    assert loads(res.study_yaml).results["s3"]["ok"]


def test_an_edit_made_at_review_is_what_runs():
    planned = _run("Design flow for a road crossing, 100-year return period", execute=False)
    study = planned.study.to_dict()
    study["steps"][2]["arguments"]["bootstrap_ci"] = False      # the reader turned the slow band off
    calls: list = []
    res = _run_reviewed(study, recon=RECON, calls=calls)
    assert res.ok
    assert calls[2] == ("flood_frequency", {"source": "uk_ea", "station_id": "3400TH", "bootstrap_ci": False})


def test_run_reviewed_takes_yaml_and_scouts_when_no_reconnaissance_is_given():
    planned = _run("Design flow for a road crossing, 100-year return period", execute=False)
    res = _run_reviewed(planned.study_yaml)
    assert res.ok
    assert res.timeline[0]["role"] == "scout" and "stations within reach" in res.timeline[0]["detail"]


def test_run_reviewed_replans_on_a_branch_fallback_and_reports_a_failed_gate():
    planned = _run("Design flow for a road crossing, 100-year return period", execute=False)
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900          # spread_within fails at T = 100
    calls: list = []
    res = _run_reviewed(planned.study.to_dict(), recon=RECON, calls=calls,
                        registry=_tools(calls, flow=wide, donors_k=1))
    assert not res.ok
    assert any("spread_within" in n for n in res.not_established)
    assert any(e["event"] == "fallback" for e in res.timeline), "the playbook's own fallback ran"
    assert "does not establish" in res.to_markdown()


def test_run_reviewed_refuses_an_empty_study():
    with pytest.raises(ValueError):
        team.run_reviewed({"question": "nothing", "version": 2, "steps": []})


def test_a_caller_may_serve_a_tool_itself():
    """The browser worker cannot read BasinATLAS; the page hands describe_catchment's answer in as a tool."""
    planned = _run("Design flow for a road crossing, 100-year return period", execute=False)
    served: list = []

    def page_catchment(lat=None, lon=None, **_kw):
        served.append((lat, lon))
        return {"latitude": lat, "longitude": lon, "sub_basin": {"hybas_id": 1}, "license": "CC BY 4.0",
                "attributes": {"upstream_area_km2": 9948.0}, "attribution": "BasinATLAS"}

    calls: list = []
    res = _run_reviewed(planned.study.to_dict(), recon=RECON, calls=calls, tools={"describe_catchment": page_catchment})
    assert res.ok and served == [(51.415, -0.308)]
    assert [c[0] for c in calls] == ["analyze_station", "flood_frequency"], "the registry's tool was not called"
    assert "9,948" in res.answer
