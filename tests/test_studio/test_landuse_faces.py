"""The three Studio faces use the drawn or supplied TR-55 scenario, keyless."""

from __future__ import annotations

import json
import sys

import pytest

from aquascope import cli, mcp_server
from aquascope.studio import Studio
from aquascope.studio.roles.consultant import consult
from aquascope.studio.workspace import Workspace
from tests.test_explorer.test_studio_worker import _face
from tests.test_study_landuse import SCENARIO

PROBLEM = "Compare runoff before and after this land-use change."


def test_landuse_plan_only_uses_the_supplied_scenario(no_deliverables):
    studio = Studio(45.70, 4.955)
    guessed = {"brief": {"intake": {"landuse_change": SCENARIO}}}
    decline = studio.say(PROBLEM, proposed=guessed)
    assert decline.kind == "declined"
    assert "polygon" in decline.text and "rain_mm" in decline.text

    studio = Studio(45.70, 4.955, intake={"landuse_change": {**SCENARIO, "rain_mm": 0}})
    decline = studio.say(PROBLEM)
    assert decline.kind == "declined" and "rain_mm" in decline.text

    studio = Studio(45.70, 4.955, intake={"landuse_change": SCENARIO})
    assert studio.say(PROBLEM).kind == "plan"
    studio.say("Keep my storm.", proposed={"brief": {"intake": {"landuse_change": {**SCENARIO, "rain_mm": 200}}}})
    assert studio.workspace.brief.intake["landuse_change"] == SCENARIO


def test_groundwater_attribution_keeps_its_own_playbook():
    ws = Workspace()
    consult(ws, None, "Could a groundwater decline be due to drought or land-use change?")
    assert ws.brief.playbook == "groundwater_decline"


@pytest.mark.parametrize("alteration", ["invented_storm", "missing_checks"])
def test_mcp_rejects_model_changes_to_the_scenario_and_its_checks(no_deliverables, alteration):
    opened = mcp_server.studio_start(PROBLEM, 45.70, 4.955, intake={"landuse_change": SCENARIO})
    assert opened["status"] == "review"
    step, = opened["workspace"]["study"]["steps"]
    assert step["arguments"] == SCENARIO
    proposed = {"id": "s1", "tool": "landuse_change", "arguments": SCENARIO}
    if alteration == "invented_storm":
        proposed = {**step, "arguments": {**SCENARIO, "rain_mm": 200}}
    done = mcp_server.studio_approve(opened["workspace"], plan={"steps": [proposed]})
    assert done["status"] == "done"
    assert done["reply"]["payload"]["plan_used"] == "tree"
    assert done["workspace"]["run"]["ok"]
    assert done["workspace"]["study"]["steps"][0]["arguments"] == SCENARIO


def test_cli_accepts_the_scenario_as_structured_intake(monkeypatch, tmp_path, no_deliverables):
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", PROBLEM, "--lat", "45.70", "--lon", "4.955",
                                      "--intake", "landuse_change=" + json.dumps(SCENARIO), "--yes", "--quiet",
                                      "--out", str(tmp_path)])
    cli.main()
    ws = json.loads((tmp_path / "workspace.json").read_text(encoding="utf-8"))
    assert ws["run"]["ok"] and ws["study"]["steps"][0]["arguments"] == SCENARIO


def test_map_worker_runs_the_same_landuse_study(no_deliverables):
    call = _face()["studio_call"]
    out = call({"op": "start", "lat": 45.70, "lon": 4.955, "text": PROBLEM, "intake": {"landuse_change": SCENARIO}})
    assert out["workspace"]["study"]["steps"][0]["arguments"] == SCENARIO
    completed = call({"op": "approve", "workspace": out["workspace"]})
    assert completed["status"] == "done" and completed["workspace"]["run"]["ok"]
