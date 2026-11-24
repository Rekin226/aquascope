"""A short routing window in the study findings, plot, workbook and exported hourly CSV."""

import csv
import io
import zipfile

import openpyxl
import pytest

from aquascope import routing
from aquascope.studio.deliverables import build, figures, tables
from aquascope.studio.document.compose import terminal_summary
from aquascope.studio.roles import author, interpreter
from aquascope.studio.workspace import Brief, Workspace
from aquascope.study import Step, Study


@pytest.fixture
def payload():
    return {
        "grade": "screening", "hydrograph_check": {"ok": True},
        "hydrograph": {"hours": [0, 1, 2, 3], "start_river_id": 11, "end_river_id": 22,
                       "start_cms": [0, 12.5, 4, 0], "end_cms": [0, 0, 0.125, 0.375]},
        "reaches": [{"river_id": 11, "peak_h": 1}, {"river_id": 22, "peak_h": None}],
        "check": {k: routing.CHECK[k] for k in ("summary", "tolerance")},
        "notes": [routing.NOTE_MODEL, routing.NOTE_LINEAR], "licence": routing.LICENCE,
        "attribution": routing.ATTRIBUTION,
        "message": "The screening hydrograph gives hourly added flow from reach 11 to reach 22 over 3 hours.",
    }


def test_screening_report_and_export_preserve_the_hourly_added_flow(payload):
    study = Study("Route this flood wave", steps=[Step("route_flood", id="s1", method="route_flood")])
    ws = Workspace(brief=Brief(problem=study.question, kind="route_flood"))
    ws.set_study(study)
    ws.run = {"ok": True, "results": [{"id": "s1", "tool": "route_flood", "ok": True, "result": payload}]}
    decision = interpreter.interpret(ws, None)["decision"]
    assert decision["grade"] == "screening"
    assert decision["basis"] == ["s1.hydrograph.end_cms.3"]
    assert interpreter.resolve_basis(ws, decision["basis"][0]) == decision["value"] == 0.375
    assert payload["check"]["summary"] in decision["evidence"]["method_check"]
    assert payload["check"]["tolerance"] in decision["evidence"]["method_check"]
    assert decision["evidence"]["licence"] == payload["licence"]
    assert set(payload["notes"]) <= set(decision["limitations"])
    author.author_report(ws, None)
    summary = "\n".join(terminal_summary(ws))
    assert "No design flood" not in summary and "during the simulated window" in summary
    for a in [*tables.tables_for("s1", "route_flood", payload), *figures.figures_for("s1", "route_flood", payload)]:
        ws.add_artifact(a)
    bundle = build(ws, formats=["md", "xlsx", "zip"])[-1]
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("tables/s1_route_hydrograph.csv").decode())))
        assert [(int(r["hour"]), float(r["start_added_flow_m3_s"]), float(r["end_added_flow_m3_s"]))
                for r in rows] == [(0, 0, 0), (1, 12.5, 0), (2, 4, 0.125), (3, 0, 0.375)]
        assert all(r["start_river_id"] == "11" and r["end_river_id"] == "22" for r in rows)
        book = openpyxl.load_workbook(io.BytesIO(archive.read("workbook.xlsx")), read_only=True)
        sheet = list(book["s1_route_hydrograph"].values)
        assert [r[:3] for r in sheet[1:]] == [(0, 0, 0), (1, 12.5, 0), (2, 4, 0.125), (3, 0, 0.375)]
        assert archive.read("figures/s1_route_hydrograph.png").startswith(b"\x89PNG")
        report = archive.read("report.md").decode()
        assert "Flood-wave routing" in report and "during the simulated window" in report
        assert payload["check"]["summary"] in report and "CC BY-SA 4.0" in report
        assert "figures/s1_route_hydrograph.png" in report


def test_plot_keeps_raw_flows_and_omits_unchecked_or_misaligned_series(payload):
    fig, caption = figures.make("route_hydrograph", payload)
    try:
        ax = fig.axes[0]
        assert "m³/s" in ax.get_ylabel()
        assert all(list(line.get_xdata()) == [0, 1, 2, 3] for line in ax.lines)
        assert [list(line.get_ydata()) for line in ax.lines] == [[0, 12.5, 4, 0], [0, 0, 0.125, 0.375]]
        assert "reach 11" in caption and "reach 22" in caption
    finally:
        figures.close(fig)
    payload["hydrograph_check"]["ok"] = False
    assert tables.make("route_hydrograph", payload) is None and figures.make("route_hydrograph", payload) is None
    payload["hydrograph_check"]["ok"] = True
    payload["hydrograph"]["end_cms"].pop()
    assert tables.make("route_hydrograph", payload) is None and figures.make("route_hydrograph", payload) is None
