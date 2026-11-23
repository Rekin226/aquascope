"""A routing Study plans the wave specified by the client on the selected reach."""

from __future__ import annotations

import pytest

from aquascope.studio import Studio, catalogue
from aquascope.studio.roles import consultant, methodologist
from aquascope.studio.workspace import Workspace

SCENARIO = {"river_id": 230301988, "peak_cms": 500, "duration_h": 12}
PROBLEM = "Route a flood wave downstream."


@pytest.mark.parametrize("magnitude", [{"peak_cms": 500}, {"return_period": 20}])
def test_route_plan_preserves_the_clients_wave_and_names_its_outputs(monkeypatch, magnitude):
    def unexpected(*args, **kwargs):
        pytest.fail("Planning a wave on a selected reach does not need a station inventory")

    monkeypatch.setattr("aquascope.ai_engine.team._scout", unexpected)
    scenario = {"river_id": 230301988, "duration_h": 12, "shape": "block", "hours": 48,
                "max_reaches": 4, **magnitude}
    studio = Studio(50, 8, intake={"route_flood": scenario})
    assert studio.say(PROBLEM).kind == "plan"
    ws = studio.workspace
    assert ws.inventory.sufficiency[0]["status"] == "defensible"
    step, = ws.study["steps"]
    assert step["tool"] == step["method"] == ws.brief.kind == "route_flood"
    assert step["arguments"] == scenario and ws.brief.stated == ["route_flood"]
    assert {(o["kind"], o["id"]) for o in step["outputs"]} == {
        ("figure", "s1_route_hydrograph"), ("table", "s1_route_hydrograph"),
    }
    assert "added discharge" in ws.study["plan"]["caveats"][0]


@pytest.mark.parametrize(("scenario", "missing"), [
    (None, "river_id"), ({"river_id": 230301988, "duration_h": 4}, "peak_cms"),
    ({"river_id": 230301988, "peak_cms": 500}, "duration_h"),
    ({**SCENARIO, "return_period": 20}, "exactly one"), ("default", "object"),
])
def test_route_declines_when_only_the_model_has_a_complete_scenario(scenario, missing):
    studio = Studio(50, 8, intake={"route_flood": scenario})
    reply = studio.say("Route it.", proposed={"brief": {"intake": {"route_flood": SCENARIO}}})
    assert reply.kind == "declined" and missing in reply.text
    assert studio.workspace.inventory.sufficiency[0]["status"] == "not_defensible"


def test_review_keeps_the_declared_reach_even_if_a_model_switches_the_playbook():
    studio = Studio(50, 8, intake={"route_flood": SCENARIO})
    assert studio.say(PROBLEM).kind == "plan"
    proposed = {"brief": {"playbook": "flood_risk", "intake": {"route_flood": {**SCENARIO, "peak_cms": 700}}}}
    assert studio.say("Keep my reach and peak.", proposed=proposed).kind == "plan"
    ws = studio.workspace
    assert ws.brief.kind == "route_flood" and ws.brief.intake["route_flood"] == SCENARIO
    assert ws.study["steps"][0]["arguments"] == SCENARIO


@pytest.mark.parametrize("alteration", ["invented reach", "missing check", "accept failed check"])
def test_a_proposed_plan_cannot_change_the_users_reach_or_skip_the_wave_check(alteration):
    studio = Studio(50, 8, intake={"route_flood": SCENARIO})
    assert studio.say(PROBLEM).kind == "plan"
    ws = studio.workspace
    original, = ws.study["steps"]
    step = {**original, "expects": [dict(g) for g in original["expects"]]}
    if alteration == "invented reach":
        step["arguments"] = {**SCENARIO, "river_id": 230301989}
    else:
        gate = next(g for g in step["expects"] if g["path"] == "hydrograph_check.ok")
        if alteration == "missing check":
            step["expects"].remove(gate)
        else:
            gate["value"] = False
    study, errors, used = methodologist.adopt(ws, {"steps": [step]})
    assert errors and used == "tree"
    assert study.steps[0].arguments == SCENARIO
    assert study.steps[0].expects == catalogue.get("route_flood").gates


@pytest.mark.parametrize("brief", ["What is the 100-year flood risk here?",
                                    "Map the inundation extent of a 100-year flood.",
                                    "Which streets flood if we route a flood wave downstream?"])
def test_flood_risk_and_inundation_keep_the_flood_playbook(brief):
    ws = Workspace()
    consultant.consult(ws, None, brief)
    assert ws.brief.playbook == "flood_risk"
