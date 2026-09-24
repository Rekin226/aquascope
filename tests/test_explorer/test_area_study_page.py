"""Study this area in the Explorer: the page is wired to the engine, and its pure helpers behave.

The page half is a thin face: area-study.js posts the selected rows to the
worker, which calls aquascope.area_study. These checks stand in for a browser:
the hooks exist at both ends of the message, and the colour, legend and sort
helpers (area-study-view.js, no DOM) do what the panel relies on.
"""

from __future__ import annotations

import inspect
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from aquascope import area_study

EXPLORER = Path(__file__).resolve().parents[2] / "explorer"
VIEW = EXPLORER / "src" / "area-study-view.js"


def test_the_area_select_offers_the_study_and_the_worker_answers_it():
    layer_ui = (EXPLORER / "src" / "layer-ui.js").read_text(encoding="utf-8")
    assert 'import { openAreaStudy } from "./area-study.js?v=__BUILD__";' in layer_ui
    assert "Study this area" in layer_ui and "openAreaStudy(rows, bbox)" in layer_ui

    worker = (EXPLORER / "worker.js").read_text(encoding="utf-8")
    assert 'if (m.type === "area_study") return await areaStudy(m);' in worker
    # every op the page sends is one the worker's Python handles, through package functions only
    page = (EXPLORER / "src" / "area-study.js").read_text(encoding="utf-8")
    ops = set(re.findall(r'callCancelable\("area_study", \{ op: "(\w+)"', page))
    assert ops == {"run", "areas"} and 'download("csv")' in page and 'download("xlsx"' in page
    for op in ("run", "areas", "csv", "xlsx"):
        assert f'_a["op"] == "{op}"' in worker
    for fn in ("study_area", "apply_areas", "to_csv", "to_xlsx"):
        assert f"_area_mod.{fn}(" in worker and callable(getattr(area_study, fn))
    assert "on_progress" in inspect.signature(area_study.study_area).parameters

    client = (EXPLORER / "src" / "worker-client.js").read_text(encoding="utf-8")
    assert 'm.type === "area_progress"' in client and "export function onAreaProgress" in client


def test_the_panel_lives_in_the_page():
    html = (EXPLORER / "index.html").read_text(encoding="utf-8")
    assert 'id="area-study"' in html


def _node(script: str):
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_view_helpers_colour_legend_and_sort():
    script = f"""
    const m = await import({json.dumps(VIEW.as_uri())});
    const feats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      .map((v) => ({{ properties: {{ q100: v, trend: v > 8 ? "up" : "none" }} }}));
    feats.push({{ properties: {{ q100: null, trend: "untested" }} }});
    console.log(JSON.stringify({{
      breaks: m.classBreaks(feats.map((f) => f.properties.q100)),
      trendExpr: m.pinColor("trend", feats)[0],
      rampExpr: m.pinColor("q100", feats),
      trendLegend: m.legend("trend", feats).map((r) => r.label),
      rampLegend: m.legend("q100", feats).map((r) => r.label),
      sorted: m.sortRows([{{ q100: 3 }}, {{ q100: null }}, {{ q100: 1 }}], "q100", 1).map((r) => r.q100),
      sortedDown: m.sortRows([{{ q100: 3 }}, {{ q100: null }}, {{ q100: 1 }}], "q100", -1).map((r) => r.q100),
      progress: [m.progressText({{ phase: "fetch", done: 2, total: 10, site: "usgs/1" }}),
                 m.progressShare({{ phase: "fetch", done: 10, total: 10 }})],
    }}));
    """
    got = _node(script)
    assert got["breaks"] == [3, 5, 7, 9]
    assert got["trendExpr"] == "match"
    assert got["rampExpr"][0] == "case" and got["rampExpr"][2][0] == "step"
    assert got["trendLegend"] == ["Up (2)", "Down (0)", "No trend (8)", "Not tested (1)"]
    assert got["rampLegend"][0] == "below 3" and got["rampLegend"][-1] == "no value (1)"
    assert got["sorted"] == [1, 3, None] and got["sortedDown"] == [3, 1, None]
    assert got["progress"][0] == "Reading records 3 of 10 (usgs/1)"
    assert got["progress"][1] == pytest.approx(0.8)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_colour_mode_is_a_field_the_engine_returns():
    got = _node(f"const m = await import({json.dumps(VIEW.as_uri())}); "
                "console.log(JSON.stringify({colors: m.COLOR_BY.map((c) => c.id), table: m.TABLE.map((c) => c[0])}));")
    fields = set(area_study.TABLE_COLUMNS) | {"key", "trend_tau", "discordant"}
    assert set(got["colors"]) <= fields
    assert set(got["table"]) <= fields
