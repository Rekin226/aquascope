"""Offline flood depth products retain values, orientation and comparison provenance."""

import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from aquascope import floodsim
from aquascope.studio.deliverables import build, figures, tables
from aquascope.studio.roles import analysts, author, interpreter
from aquascope.studio.workspace import Brief, Workspace
from aquascope.study import Step, Study


@pytest.mark.parametrize("kind", ["flood_risk", "flood_inundation"])
@pytest.mark.parametrize("csi", [0.333, None])
@pytest.mark.parametrize("custom", [False, True])
def test_inundation_products(kind, csi, custom):
    window = {"x0": 100, "y0": 200, "nx": 2, "ny": 2, "zoom": 10}
    depth = np.array([[1.25, 0.0], [2.5, 3.75]])
    payload = {"depth": floodsim.depth_payload({"window": window}, depth),
               "summary": {"river_id": 123, "return_period": 77 if custom else 100,
                           "flow": "12 m³/s" if custom else "the 100-year flow", "flow_cms": 12.5,
                           "hours_run": 3, "steady_at_h": None, "wet_km2": 0.2, "max_depth_m": 3.75},
               "check": {"ok": True, "status": "compared" if csi is not None else "unavailable",
                         "csi": csi, "summary": "Flood depth computed"},
               "grade": "screening", "message": "Screening flood depth."}
    study = Study("Flood depth", steps=[Step("flood_inundation", id="s1", method="flood_inundation")])
    ws = Workspace(brief=Brief(problem=study.question, kind=kind,
                               playbook="flood_risk" if kind == "flood_risk" else ""))
    ws.set_study(study)
    ws.run = {"ok": True, "results": [{"id": "s1", "tool": "flood_inundation", "ok": True, "result": payload}]}
    fig, _ = figures.make("inundation_map", payload)
    try:
        assert fig.axes[0].images[0].origin == "upper"
        np.testing.assert_allclose(fig.axes[0].images[0].get_array().filled(0), depth)
    finally:
        figures.close(fig)
    raster = analysts._depth_raster(ws, "s1", "flood_inundation", payload)
    assert raster is not None and "depth" in payload
    image = Image.open(io.BytesIO(raster.data))
    assert image.mode == "F"
    np.testing.assert_allclose(np.asarray(image), [[1.25, -9999], [2.5, 3.75]])
    artifacts = [raster, *figures.figures_for("s1", "flood_inundation", payload),
                 *tables.tables_for("s1", "flood_inundation", payload)]
    expected = "12.5 m3/s (specified flow)" if custom else "the 100-year flow"
    comparison = "CSI of 0.333" if csi is not None else "no CSI is reported"
    for artifact in artifacts:
        assert expected in artifact.caption and "grade: screening" in artifact.caption
        assert comparison in artifact.caption and "Simulated 3 hours" in artifact.caption
        assert "steady state not reached" in artifact.caption
        assert ("status: compared" if csi is not None else "status: unavailable") in artifact.caption
        ws.add_artifact(artifact)
    interpreter.interpret(ws, None)
    author.author_report(ws, None)
    ws = Workspace.from_json(ws.to_json())
    assert ws.run["results"][0]["result"]["depth"] == payload["depth"]
    assert ws.artifact(raster.id).data == raster.data
    archive_data = build(ws, formats=["md", "zip"])[-1].data
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        assert archive.read(raster.name) == raster.data
        assert archive.read("figures/s1_inundation_map.png").startswith(b"\x89PNG")
        rows = list(csv.DictReader(io.StringIO(archive.read("tables/s1_inundation_summary.csv").decode())))
        assert rows[0]["river_id"] == "123" and float(rows[0]["max_depth_m"]) == 3.75
        assert rows[0]["grade"] == "screening" and rows[0]["check_status"] == ("compared" if csi else "unavailable")
        assert rows[0]["csi"] == ("0.333" if csi is not None else "")
        assert rows[0]["hours_run"] == "3" and rows[0]["steady_at_h"] == ""
        assert rows[0]["steady_status"] == "not_reached"
        report = archive.read("report.md").decode()
        assert comparison in report and expected.replace("m3/s", "m³/s") in report
        assert "steady state not reached" in report
        assert "grade: screening" in report and "T = None" not in report
        assert "No design flood" not in report if kind == "flood_inundation" else True
