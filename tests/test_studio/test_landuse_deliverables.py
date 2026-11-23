"""The TR-55 study's screening evidence and comparison artifacts, from offline scenarios."""

import csv
import io

import pytest

from aquascope.sketch.landuse import study_landuse_change
from aquascope.studio.deliverables import figures, tables
from aquascope.studio.deliverables.workbook import workbook_bytes
from aquascope.studio.document.compose import build_report
from aquascope.studio.document.facts import facts_of
from aquascope.studio.roles import author, interpreter
from aquascope.studio.workspace import Brief, Workspace
from aquascope.study import Step, Study


def scenario(rain_mm, cover, use):
    return study_landuse_change(polygon=[[4.95, 45.70], [4.96, 45.70], [4.96, 45.71], [4.95, 45.71]],
                                rain_mm=rain_mm, cover=cover, use=use, soil_group="A", tc_hours=1)


@pytest.mark.parametrize("rain_mm, cover, use, direction", [(1, {10: 1}, "woods_good", 0),
                                                          (100, {50: 1}, "woods_good", -1)])
def test_screening_claims_and_report_keep_the_model_check(rain_mm, cover, use, direction):
    payload = scenario(rain_mm, cover, use)
    study = Study("How does runoff change?", steps=[Step("landuse_change", id="s1")])
    ws = Workspace(brief=Brief(problem=study.question, kind="landuse_change"))
    ws.set_study(study)
    ws.run = {"ok": True, "results": [{"id": "s1", "tool": "landuse_change", "ok": True, "result": payload}]}

    findings = interpreter.interpret(ws, None)
    decision = findings["decision"]
    assert decision["grade"] == "screening" and decision["basis"] == ["s1.change.q_mm"]
    assert decision["evidence"]["method_check"] == payload["check"]
    assert payload["assumptions"] in decision["conditions"]
    assert payload["notes"] and set(payload["notes"]) <= set(decision["limitations"])
    for finding, key in zip(findings["findings"], ("q_mm", "peak_cms"), strict=True):
        assert finding["grade"] == "screening" and finding["basis"] == [f"s1.change.{key}"]
        assert interpreter.resolve_basis(ws, finding["basis"][0]) == payload["change"][key]
        assert finding["evidence"]["method_check"] == payload["check"]
    if direction == 0:
        assert decision["value"] == 0 and payload["change"]["peak_pct"] is None
    else:
        assert decision["value"] < 0

    report = author.author_report(ws, None)
    result = next(s["text"] for s in report["sections"] if s["id"] == "results-s1")
    assert all(s in result for s in (payload["summary"], payload["assumptions"], payload["check"]))
    assert payload["check"] in facts_of(ws).grade_reason
    assert build_report(ws).title == "Land-use change and storm runoff"


def test_before_and_after_in_csv_workbook_and_figure():
    import openpyxl

    payload = scenario(100, {10: 1}, "paved")
    table, = tables.tables_for("s1", "landuse_change", payload)
    rows = list(csv.DictReader(io.StringIO(table.data.decode())))
    assert [r["scenario"] for r in rows] == ["before", "after"]
    for row in rows:
        run = payload[row["scenario"]]
        assert float(row["runoff_mm"]) == run["q_mm"]
        assert float(row["peak_m3_s"]) == run["peak_cms"]
        assert float(row["curve_number"]) == run["cn"]
        assert row["grade"] == "screening" and row["method_check"] == payload["check"]
    assert rows[1]["land_use"] == payload["use_label"]

    ws = Workspace()
    ws.add_artifact(table)
    book = openpyxl.load_workbook(io.BytesIO(workbook_bytes(ws)), read_only=True)
    sheet = list(book["s1_landuse_runoff"].values)
    assert [row[sheet[0].index("runoff_mm")] for row in sheet[1:]] == [
        payload["before"]["q_mm"], payload["after"]["q_mm"]]
    assert sheet[2][sheet[0].index("method_check")] == payload["check"]

    figure, caption = figures.make("landuse_runoff", payload)
    try:
        axis = figure.axes[0]
        assert [bar.get_width() for bar in axis.patches] == [payload["before"]["q_mm"], payload["after"]["q_mm"]]
        assert axis.get_xlabel() == "Runoff depth (mm)"
        assert "Screening" in axis.get_title() and "TR-55" in axis.get_title()
        assert payload["use_label"] in caption and payload["check"] in caption
    finally:
        figures.close(figure)


def test_fallback_keeps_its_screening_grade():
    fallback = {"tool": "landuse_change", "ok": True, "gates_passed": True, "result": scenario(100, {10: 1}, "paved")}
    ws = Workspace(run={"results": [{"id": "s1", "tool": "catchment_model", "ok": True, "gates_passed": False,
                                     "fallback_used": True, "fallback": fallback}]})
    assert interpreter.grade_for_step(ws, "s1.fallback") == "screening"
    assert interpreter.grade_for_step(ws, "s1") == "screening"
