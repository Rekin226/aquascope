"""The fixes from the live runs: wrong methods survive and invalid steps are pruned (A), the figures per step
and the series kept for them (E), an attached table planned on keyless (F), a report the checker does not
flag for its own counts (G), one key-numbers table and deduplicated references (C), recommendations that are
not the answer (H), records named (D)."""

from __future__ import annotations

import json
import sys
import types

from aquascope.studio.model import Model
from aquascope.studio.roles import analysts, author, consultant, critic, methodologist, scout
from aquascope.studio.workspace import Artifact, Workspace
from tests.test_studio.conftest import FLOW, PROBLEM, RECON, SERIES_CSV, UNGAUGED, FakeModel, fake_tools, patched
from tests.test_studio.test_methodologist import VALID_PLAN, _ws

# ── A ──


def test_a_wrong_method_on_a_valid_step_is_replaced_not_fatal():
    ws = _ws()
    plan = dict(VALID_PLAN, steps=[dict(VALID_PLAN["steps"][0]),
                                   dict(VALID_PLAN["steps"][3], method="fao56_et0"),
                                   dict(VALID_PLAN["steps"][1], method="not_a_method")])
    client = FakeModel({"methodologist": [plan]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist" and ws.ledger["methodologist"]["calls"] == 1
    assert study.step_by_id("s4").method == "glofas_cross_check", "the entry's first method serving flood_risk"
    assert study.step_by_id("s2").method == "at_site_flood_frequency"
    notes = study.plan["notes"]
    assert any("fao56_et0" in n and "glofas_cross_check" in n for n in notes)
    assert any("not_a_method" in n for n in notes)
    assert not any(e["event"] == "invalid" for e in ws.events)


def test_an_empty_repair_keeps_the_valid_remainder():
    ws = _ws()
    broken = dict(VALID_PLAN, steps=[*VALID_PLAN["steps"][:3],
                                     {"id": "s4", "tool": "frobnicate", "arguments": {}, "rationale": "x"},
                                     {"id": "s5", "tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308},
                                      "depends_on": ["s4"], "rationale": "needs s4"}])
    client = FakeModel({"methodologist": [broken, {"steps": []}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist"
    assert [s.id for s in study.steps] == ["s1", "s2", "s3"], "the invalid step and its dependant went"
    assert any(e["event"] == "pruned" for e in ws.events if e["role"] == "methodologist")
    assert any("s4 removed" in n for n in study.plan["notes"]) and any("s5 removed" in n for n in study.plan["notes"])


def test_a_repair_that_is_still_wrong_prunes_and_nothing_valid_falls_back_to_the_tree():
    ws = _ws()
    broken = dict(VALID_PLAN, steps=[*VALID_PLAN["steps"][:2], {"id": "s3", "tool": "frobnicate", "arguments": {}}])
    client = FakeModel({"methodologist": [broken, broken]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study.author == "methodologist" and [s.id for s in study.steps] == ["s1", "s2"]
    ws2 = _ws()
    only_bad = dict(VALID_PLAN, steps=[VALID_PLAN["steps"][0], {"id": "s2", "tool": "frobnicate", "arguments": {}}])
    client2 = FakeModel({"methodologist": [only_bad, only_bad]})
    model2 = Model.resolve(ws2, client=client2, model="fake", provider="custom")
    study2 = methodologist.plan(ws2, model2)
    assert study2.author == "playbook", "describe_catchment alone is not an analysis"
    assert any(e["event"] == "fallback" for e in ws2.events)


def test_the_methodologist_context_lists_methods_per_tool_and_uploads_first():
    ws = _ws()
    ws.add_table("upload:flows.csv", SERIES_CSV)
    with patched(RECON):
        scout.scout(ws)
    client = FakeModel({"methodologist": [VALID_PLAN]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    methodologist.plan(ws, model)
    ctx = client.requests[0]["context"]
    assert list(ctx)[0] == "uploads" and ctx["uploads"][0]["id"] == "upload:flows.csv"
    assert ctx["uploads"][0]["variable"] == "discharge" and ctx["uploads"][0]["columns"] == ["date", "flow_m3s"]
    entry = next(e for e in ctx["catalogue"] if e["tool"] == "analyze_station")
    assert "trend_mann_kendall" in entry["methods"]
    assert "one of the \"methods\"" in client.requests[0]["system"] or "methods" in client.requests[0]["system"]


# ── E ──

FULL_FLOW = {**FLOW, "fdc": {"exceedance": [1, 50, 99], "q": [400, 40, 5], "q95": 7.5, "q50": 40.0, "q10": 162.0},
             "series": {"t": ["2000-01-01", "2000-01-02"], "v": [10.0, 12.0]},
             "annual_max": {"year": [2000, 2001], "v": [300.0, 350.0]}}
FULL_FLOW.pop("name")


def _fake_makers(monkeypatch, made: list, *, with_kinds: bool):
    def figures_for(step_id, tool, payload, *, unit=None, site=None, **kw):
        if with_kinds and "kinds" not in kw:
            kinds = None
        elif not with_kinds and kw:
            raise TypeError("unexpected keyword argument 'kinds'")
        else:
            kinds = kw.get("kinds")
        made.append((step_id, tool, kinds, "series" in payload))
        wanted = kinds if kinds is not None else ["series", "annual_maxima", "frequency_curve", "fdc", "trend"]
        return [Artifact(id=f"fig-{step_id}-{k}", kind="figure", name=f"figures/{step_id}_{k}.png", data=b"png",
                         media_type="image/png", caption=k, step=step_id, meta={"kind": k, "tool": tool})
                for k in wanted]

    pkg = types.ModuleType("aquascope.studio.deliverables")
    figs = types.ModuleType("aquascope.studio.deliverables.figures")
    tabs = types.ModuleType("aquascope.studio.deliverables.tables")
    figs.figures_for, tabs.tables_for = figures_for, (lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables", pkg)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.figures", figs)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.tables", tabs)


def _flood_ws() -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem, ws.brief.playbook, ws.brief.kind = PROBLEM, "flood_risk", "flood_risk"
    ws.brief.intake = {"return_period": 100}
    with patched(RECON):
        scout.scout(ws)
        methodologist.plan(ws, None)
    return ws


def test_figures_follow_the_steps_method_and_the_series_is_stripped_after(monkeypatch):
    for with_kinds in (True, False):
        made: list = []
        _fake_makers(monkeypatch, made, with_kinds=with_kinds)
        ws = _flood_ws()
        with patched(RECON, tools=fake_tools([], analyze_station=FULL_FLOW, flood_frequency=FULL_FLOW)):
            analysts.run(ws, None)
        ids = sorted(a.id for a in ws.artifacts if not a.id.startswith("fig-s4-") and a.id != "study-map")
        assert ids == ["fig-s1-annual_maxima", "fig-s1-fdc", "fig-s1-frequency_curve", "fig-s1-series",
                       "fig-s1-trend", "fig-s2-series", "fig-s2-trend", "fig-s3-annual_maxima",
                       "fig-s3-frequency_curve"], with_kinds
        assert any(a.id.startswith("fig-s4-") for a in ws.artifacts), "the cross-check step draws too"
        assert ids.count("fig-s2-frequency_curve") == 0 and ids.count("fig-s3-frequency_curve") == 1
        s2 = next(m for m in made if m[0] == "s2")
        assert s2[3] is True, "the makers saw the series"
        for r in ws.run["results"][1:3]:
            assert "series" not in r["result"] and r["result"]["fdc"]["exceedance"] == [1, 50, 99]
            assert r["result"]["station_name"] == "Kingston" and r["result"]["name"] == "Kingston"
        assert '"t": [' not in json.dumps(ws.run) and '"t": [' not in json.dumps(ws.study)


def test_analyze_station_full_keeps_the_series_and_adds_the_band(monkeypatch):
    import aquascope.explore

    def fake(source, station_id, *, years=None, store=None, variable=None, period_start=None):
        store["series"] = "S"
        return {"source": source, "station_id": station_id, "series": {"t": ["2000-01-01"], "v": [1.0]},
                "fdc": {"exceedance": [1], "q": [1], "q95": 1, "q50": 1, "q10": 1},
                "ffa": {"fits": {"gev_lmoments": {"q": [1]}}}}

    monkeypatch.setattr(aquascope.explore, "analyze_station", fake)
    monkeypatch.setattr(aquascope.explore, "flood_ci", lambda s: {"q": [2], "ci": [[1, 3]], "method": {"name": "b"}})
    out = analysts.analyze_station_full("uk_ea", "3400TH", bootstrap_ci=True)
    assert out["series"]["v"] == [1.0] and out["fdc"]["exceedance"] == [1]
    assert out["ffa"]["fits"]["gev_bootstrap"] == {"q": [2], "ci": [[1, 3]]} and out["methods"] == [{"name": "b"}]
    assert "error" in analysts.analyze_station_full("nope", "1")
    import aquascope.studio.catalogue as cat

    with patched(RECON):
        callables = cat.callables({})
    assert callables["analyze_station"] is not analysts.analyze_station_full, "the swap happens inside run()"


# ── F ──


def _upload_ws(problem: str, recon_value: dict = RECON) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.add_table("upload:myflows.csv", SERIES_CSV)
    with patched(recon_value):
        consultant.consult(ws, None, problem)
        scout.scout(ws)
    return ws


def test_my_own_record_plans_on_the_upload_keyless():
    ws = _upload_ws("Study my own record: flood frequency for a 50-year design and the flow duration curve")
    assert "use the attached table upload:myflows.csv" in ws.brief.constraints
    assert not any("did not say" in a for a in ws.brief.assumptions)
    assert methodologist.wants_upload(ws)
    with patched(RECON):
        study = methodologist.plan(ws, None)
    assert [s.tool for s in study.steps] == ["load_table", "return_periods", "return_periods", "flow_duration"]
    assert study.plan["branch"] == "upload" and study.plan["upload"] == "upload:myflows.csv"
    s2 = study.step_by_id("s2")
    assert s2.arguments["from_step"] == "s1" and 50 in s2.arguments["periods"]
    assert s2.method == "at_site_flood_frequency"
    assert study.step_by_id("s3").arguments["distribution"] == "lp3"
    assert study.step_by_id("s4").expects == [{"check": "not_empty", "path": "percentiles"}]
    assert any("marginal" in n for n in study.plan["notes"]), "12 years of maxima is marginal for the fit"
    with patched(RECON):
        run = analysts.run(ws, None)
    assert run.ok, ws.run["stop_reason"]
    assert ws.run["results"][1]["result"]["return_levels"] and ws.run["results"][3]["result"]["percentiles"]
    report = author.author_report(ws, None)
    labels = [k["label"] for k in report["key_numbers"]]
    assert any(lab.startswith("50-year return level, GEV") for lab in labels) and "Q95" in labels
    assert critic.critique(ws, None)["checks"]


def test_without_the_words_the_gauge_wins_unless_it_is_missing():
    ws = _upload_ws(PROBLEM)
    assert any("did not say" in a for a in ws.brief.assumptions)
    assert not methodologist.wants_upload(ws), "a discharge gauge is within reach and the text says nothing"
    ws2 = _upload_ws(PROBLEM, UNGAUGED)
    assert methodologist.wants_upload(ws2), "no discharge within reach: the table is the record"
    with patched(UNGAUGED):
        study = methodologist.plan(ws2, None)
    assert study.steps[0].tool == "load_table"


def test_the_validator_judges_a_method_on_the_table_when_the_site_lacks_the_variable():
    ws = _upload_ws(PROBLEM, UNGAUGED)
    rows = {r["method"]: r for r in methodologist.sufficiency_for_validation(ws)}
    assert rows["at_site_flood_frequency"]["status"] == "marginal" and rows["flow_duration"]["status"] == "defensible"
    assert rows["at_site_flood_frequency"]["station"] == {"source": "upload", "station_id": "upload:myflows.csv"}


# ── C, D, G, H ──


def test_the_report_names_the_record_lists_long_rows_and_is_not_flagged_for_its_counts():
    ws = _flood_ws()
    short = dict(RECON["stations"][0], station_id="GPRSB5A", name="VILLIERS ROAD DS", years=0.1,
                 variables=["water_level"])
    ws.inventory.datasets.insert(1, scout._station_datasets({"stations": [short], "context": {}})[0])
    ws.inventory.datasets[0].start = "1883-10-01"
    unnamed = {**FLOW}
    unnamed.pop("name")
    with patched(RECON, tools=fake_tools([], analyze_station=unnamed, flood_frequency=unnamed)):
        analysts.run(ws, None)
    report = author.author_report(ws, None)
    assert "The record at Kingston (uk_ea 3400TH) runs" in report["answer"]
    by_id = {s["id"]: s["text"] for s in report["sections"]}
    assert "GPRSB5A" not in by_id["site_data"] and "1 more short record" in by_id["site_data"]
    assert "1883-10-01 to present" in by_id["site_data"]
    assert "| Quantity |" not in by_id["summary"] and by_id["summary"] != report["answer"]
    assert "4 step(s) ran" in by_id["summary"] and "12 of 12 gates passed" in by_id["summary"]
    md = author.to_markdown(ws)
    assert md.count("| Quantity | Value | Unit | Step |") == 1
    recs = report["recommendations"]
    assert 2 <= len(recs) <= 4 and not any("The record at" in r for r in recs)
    assert any("spread" in r for r in recs) and any("caveat" in r for r in recs)
    refs = report["references"]
    assert sum(1 for r in refs if "Hosking" in r) == 1 and sum(1 for r in refs if "Wasko" in r) == 1
    assert refs[-1].startswith("Rekin226")
    out = critic.critique(ws, None)
    assert all(c["passed"] for c in out["checks"]), out["checks"]
    assert out["not_established"] == []


def test_name_records_names_once_then_short():
    text = "The record at station uk_ea 3400TH runs. Low flows at gauge uk_ea 3400TH. Then uk_ea 3400TH again."
    assert author.name_records(text, [("uk_ea", "3400TH", "Kingston")]) == (
        "The record at Kingston (uk_ea 3400TH) runs. Low flows at Kingston. Then Kingston again.")
    already = "At Kingston (uk_ea 3400TH) and later uk_ea 3400TH."
    assert author.name_records(already, [("uk_ea", "3400TH", "Kingston")]) == \
        "At Kingston (uk_ea 3400TH) and later Kingston."
