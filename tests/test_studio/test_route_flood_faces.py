"""Route the same specified wave through each keyless Study entry point."""

import json
import sys

import pytest

from aquascope import cli, mcp_server, routing
from tests.test_explorer.test_studio_worker import _face
from tests.test_routing import TOY_IDS, toy  # noqa: F401 - shared offline GEOGLOWS network fixture


@pytest.mark.parametrize("face", ["cli", "mcp", "explorer"])
@pytest.mark.usefixtures("toy", "no_deliverables")
def test_study_faces_keep_the_routed_flows(face, tmp_path, monkeypatch):
    problem = "Route this flood wave downstream."
    scenario = {"river_id": TOY_IDS["A"], "peak_cms": 600, "duration_h": 12, "hours": 24}
    intake = {"route_flood": scenario}
    if face == "cli":
        monkeypatch.setattr(sys, "argv", ["aquascope", "studio", problem, "--lat", "47", "--lon", "8.2",
                                          "--intake", "route_flood=" + json.dumps(scenario), "--yes", "--quiet",
                                          "--out", str(tmp_path)])
        cli.main()
        workspace = json.loads((tmp_path / "workspace.json").read_text(encoding="utf-8"))
    elif face == "mcp":
        opened = mcp_server.studio_start(problem, 47, 8.2, intake=intake)
        workspace = mcp_server.studio_approve(opened["workspace"])["workspace"]
    else:
        call = _face()["studio_call"]
        opened = call({"op": "start", "lat": 47, "lon": 8.2, "text": problem, "intake": intake})
        workspace = call({"op": "approve", "workspace": opened["workspace"]})["workspace"]

    assert workspace["run"]["ok"]
    step, = workspace["run"]["results"]
    wave = step["result"]["hydrograph"]
    assert wave == routing.study_route_flood(**scenario)["hydrograph"]
    assert wave["end_river_id"] == TOY_IDS["C"]
    assert wave["end_cms"] != wave["start_cms"]
