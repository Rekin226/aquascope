"""Run the real reservoir routing through all three Study clients with offline terrain and reaches."""

import base64
import csv
import io
import json
import sys
import zipfile

import pytest

from aquascope import cli, floodsim, mcp_server, playbooks
from aquascope.sketch import reservoir, reservoir_study
from tests.test_explorer.test_studio_worker import _face
from tests.test_sketch_reservoir import IDS, _flat_lake, toy  # noqa: F401 - offline routing network


@pytest.mark.usefixtures("toy")
@pytest.mark.parametrize(("face", "hours"), [("cli", 200), ("mcp", 200), ("explorer", 200), ("cli", 8)])
def test_reservoir_study_exports_inflow_and_outflow(face, hours, monkeypatch, tmp_path):
    win, terrain, polygon = _flat_lake(monkeypatch)
    polygon = [[float(lon), float(lat)] for lon, lat in polygon]
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    monkeypatch.setattr(reservoir, "plan", lambda ring: {"ok": True, "window": win, "dam": dam})
    monkeypatch.setattr(floodsim, "terrain_window", lambda *args: {"z": terrain})
    scenario = {"polygon": polygon, "crest_m": 5, "top_m": 12, "width_m": 10,
                "peak_cms": 600, "from_river_id": IDS["A"], "duration_h": 12, "hours": hours}
    question = "How does this reservoir change the flood?"
    intake = {"reservoir_routing": scenario}
    table = "tables/s1_reservoir_hydrograph.csv"
    figure = "figures/s1_reservoir_hydrograph.png"
    if face == "explorer":
        call = _face()["studio_call"]
        started = call({"op": "start", "lat": 47, "lon": 8.2, "text": "Flooding at this site"})
        picker = started["workspace"]["brief"]["questions"][-1]
        label = "How a drawn reservoir reduces a flood peak"
        assert picker["id"] == "decision" and label in picker["options"]
        selected = call({"op": "say", "text": label, "workspace": started["workspace"],
                         "reservoir_routing": scenario})
        workspace = call({"op": "approve", "workspace": selected["workspace"]})["workspace"]
        result = call({"op": "export", "workspace": workspace})
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["data"]))) as bundle:
            rows = bundle.read(table).decode("utf-8-sig")
            png = bundle.read(figure)
            report = bundle.read("report.md").decode()
    else:
        if face == "mcp":
            started = mcp_server.studio_start(question, 47, 8.2, intake=intake)
            workspace = mcp_server.studio_approve(started["workspace"])["workspace"]
            mcp_server.studio_export(workspace, str(tmp_path))
        else:
            monkeypatch.setattr(sys, "argv", ["aquascope", "studio", question, "--lat", "47", "--lon", "8.2",
                                              "--intake", "reservoir_routing=" + json.dumps(scenario),
                                              "--yes", "--quiet", "--out", str(tmp_path)])
            cli.main()
            workspace = json.loads((tmp_path / "workspace.json").read_text(encoding="utf-8"))
        rows = (tmp_path / table).read_text(encoding="utf-8-sig")
        png = (tmp_path / figure).read_bytes()
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert workspace["study"]["plan"]["playbook"] == "flood_risk"
    assert workspace["study"]["plan"]["branch"] == "reservoir_attenuation"
    assert workspace["run"]["ok"]
    step, = workspace["run"]["results"]
    payload = step["result"]
    assert payload["grade"] == "screening"
    assert payload["check"]["ok"] and payload["check"]["max_abs_diff_cfs"] == 0.51
    if hours == 8:
        assert payload["summary"]["reduction_pct"] is None
        assert payload["summary"]["delay_h"] is None
        assert "did not capture both peaks" in payload["message"]
    else:
        assert payload["summary"]["reduction_pct"] == 95.1
        assert payload["summary"]["delay_h"] == 9
    data = list(csv.DictReader(io.StringIO(rows)))
    assert len(data) == len(payload["hydrograph"]["hour"])
    assert [float(row["inflow_m3_s"]) for row in data] == payload["hydrograph"]["inflow_cms"]
    assert [float(row["outflow_m3_s"]) for row in data] == payload["hydrograph"]["outflow_cms"]
    assert png.startswith(b"\x89PNG")
    assert figure in report and "Example 17-2" in report
    assert "Flood routing through a drawn reservoir" in report


@pytest.mark.parametrize("change", [{"width_m": 0}, {"return_period": 50}, {"duration_h": 1},
                                    {"polygon": [[1, 1], [2, 2], [3, 3]]}, {"polygon": [None]}])
def test_invalid_reservoir_scenario_declines_before_routing(change):
    scenario = {"polygon": [[8, 47], [8.01, 47], [8.01, 47.01]], "crest_m": 5, "top_m": 12,
                "width_m": 10, "duration_h": 12, "peak_cms": 600}
    with pytest.raises(ValueError):
        reservoir_study.validate_scenario({**scenario, **change})


def test_flood_risk_decision_routes_only_with_a_supplied_reservoir():
    pb = playbooks.load("flood_risk")
    picker = next(item for item in pb.checklist if item.field == "decision")
    decision = next(opt.value for opt in picker.options if opt.value == "flood attenuation")
    recon = {"point": {"lat": 47, "lon": 8}, "context": {"years_by_variable": {"discharge": 5}, "donors": 0}}
    intake = {"decision": decision}
    with pytest.raises(playbooks.Declined, match="reservoir polygon"):
        playbooks.plan(pb, recon, intake)
    scenario = {"polygon": [[8, 47], [8.01, 47], [8.01, 47.01]], "crest_m": 5, "top_m": 12,
                "width_m": 10, "duration_h": 12, "peak_cms": 600}
    study = playbooks.plan(pb, recon, {**intake, "reservoir_routing": scenario})
    step, = study.steps
    assert step.method == step.tool == "reservoir_routing" and step.arguments == scenario
    assert {"check": "status_is", "path": "check.ok", "value": True} in step.expects
