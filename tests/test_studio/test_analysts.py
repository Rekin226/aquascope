"""The Analysts: the table loader, the run with gates, figures when the makers exist, the bounded replan."""

from __future__ import annotations

import json
import sys
import types

from aquascope.studio.model import Model
from aquascope.studio.roles import analysts, methodologist, scout
from aquascope.studio.workspace import Artifact, Workspace
from tests.test_studio.conftest import (
    FLOW,
    PROBLEM,
    RECON,
    SAMPLES_CSV,
    SERIES_CSV,
    FakeModel,
    fake_tools,
    patched,
)


def _ws(**tables) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem, ws.brief.playbook, ws.brief.kind = PROBLEM, "flood_risk", "flood_risk"
    ws.brief.intake = {"return_period": 100}
    for k, v in tables.items():
        ws.add_table(k, v)
    with patched(RECON):
        scout.scout(ws)
        methodologist.plan(ws, None)
    return ws


def test_load_table_serves_a_series_or_sample_rows():
    ws = Workspace()
    ws.add_table("upload:flows.csv", SERIES_CSV)
    ws.add_table("upload:samples.csv", SAMPLES_CSV)
    out = analysts.load_table(ws, "upload:flows.csv")
    assert out["n"] > 1000 and out["years"] > 10 and out["unit"] == "m3/s" and out["variable"] == "discharge"
    assert out["columns"] == ["date", "flow_m3s"] and len(out["series"]["t"]) == out["n"]
    assert out["series"]["t"][0].startswith("2010-01-01") and isinstance(out["series"]["v"][0], float)
    assert out["stats"]["mean"] > 0 and out["qa"]["coverage_pct"] > 0
    assert analysts.load_table(ws, "flows.csv")["n"] == out["n"], "a bare file name finds the upload"
    samples = analysts.load_table(ws, "upload:samples.csv")
    assert samples["n"] == 3 and samples["samples"][0] == {"station": "A", "parameter": "nitrate", "value": "3.1",
                                                          "unit": "mg/L"}
    assert "error" in analysts.load_table(ws, "upload:nope.csv")
    assert "error" in analysts.load_table(ws, "upload:flows.csv", value_column="nope")
    assert analysts.load_table(ws, "upload:flows.csv", value_column="flow_m3s", datetime_column="date")["n"] == out["n"]


def test_the_run_writes_results_gates_and_the_study_and_skips_figures_without_the_makers(no_deliverables):
    ws = _ws()
    calls: list = []
    with patched(RECON, tools=fake_tools(calls)):
        run = analysts.run(ws, None)
    assert run.ok and ws.run["ok"] and ws.run["stopped_at"] is None and ws.run["replans"] == 0
    assert [r["id"] for r in ws.run["results"]] == ["s1", "s2", "s3", "s4"] and len(ws.run["gates"]) == 12
    assert ws.run["failed_gates"] == [] and ws.study["results"]["s3"]["ok"]
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    kinds = [(e["role"], e["event"]) for e in ws.events]
    assert ("analyst", "figures_skipped") in kinds and ("runner", "done") in kinds and ("reviewer", "gate") in kinds
    # no figures or tables without the makers; the study map is pure Python and is always made
    assert ("analyst", "gates") in kinds and [a.id for a in ws.artifacts] == ["study-map"]


def test_figures_and_tables_are_made_per_step_when_the_makers_exist(monkeypatch):
    made: list = []

    def figures_for(step_id, tool, payload, *, unit=None, site=None):
        made.append(("fig", step_id, tool, unit, site))
        if tool == "flood_frequency":
            raise RuntimeError("no matplotlib here")
        return [Artifact(id=f"{step_id}_series", kind="figure", name=f"figures/{step_id}_series.png", data=b"png",
                         media_type="image/png", caption="the series", meta={"kind": "series"})]

    def tables_for(step_id, tool, payload):
        made.append(("tab", step_id, tool))
        return [{"id": f"{step_id}_table", "kind": "table", "name": f"tables/{step_id}.csv", "data": "YWJj",
                 "media_type": "text/csv"}]

    pkg = types.ModuleType("aquascope.studio.deliverables")
    figs = types.ModuleType("aquascope.studio.deliverables.figures")
    tabs = types.ModuleType("aquascope.studio.deliverables.tables")
    figs.figures_for, tabs.tables_for = figures_for, tables_for
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables", pkg)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.figures", figs)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.tables", tabs)
    ws = _ws()
    streamed: list = []
    with patched(RECON):
        analysts.run(ws, None, on_artifact=streamed.append)
    assert [m[1] for m in made if m[0] == "fig"] == ["s1", "s2", "s3", "s4"] and made[2][3] == "m3/s"
    assert made[2][4] == {"lat": 51.415, "lon": -0.308}
    ids = sorted(a.id for a in ws.artifacts if a.id != "study-map")
    assert ids == ["s1_series", "s1_table", "s2_series", "s2_table", "s3_table", "s4_series", "s4_table"], \
        "a maker's error skips one figure"
    assert [a.id for a in streamed] == [a.id for a in ws.artifacts]
    assert ws.artifact("s1_table").data == b"abc" and ws.artifact("s2_series").step == "s2"
    assert any(e["event"] == "figures_skipped" and e["step"] == "s3" and "matplotlib" in e["detail"]
               for e in ws.events)
    assert ws.figures("s1")[0].kind == "figure"


def test_a_failed_gate_runs_the_playbooks_fallback_then_the_specialists_proposal(no_deliverables):
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    calls: list = []
    tools = fake_tools(calls, flood_frequency=wide, analyze_station=wide,
                       similar_basins={"k": 1, "method": "combined",
                                       "stations": [{"source": "usgs", "station_id": "1"}]})
    ws = _ws()
    with patched(RECON, tools=tools):
        run = analysts.run(ws, None)
    assert not run.ok and run.stop_reason is None and ws.run["stopped_at"] is None
    failed_steps = ws.run["failed_steps"]
    assert [f["id"] for f in failed_steps if not f["skipped"]] == ["s3"]
    assert "spread_within" in failed_steps[0]["reason"] and failed_steps[1]["id"] == "s4", "s4 waits on s3"
    assert ws.run["summary"]["failed"] == 1 and ws.run["summary"]["planned"] == len(ws.run["results"])
    assert ws.run["results"][2]["fallback_used"] and ws.run["results"][2]["fallback"]["tool"] == "similar_basins"
    assert ws.run["replans"] == 0, "keyless: no specialist"

    proposal = {"tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": 20},
                "rationale": "GloFAS as an independent cross-check",
                "expects": [{"check": "not_empty", "path": "glofas"}]}
    ws2 = _ws()
    client = FakeModel({"analyst": [proposal]})
    model = Model.resolve(ws2, client=client, model="fake", provider="custom")
    calls2: list = []
    with patched(RECON, tools=fake_tools(calls2, flood_frequency=wide, analyze_station=wide,
                                         similar_basins={"k": 1, "stations": []})):
        run2 = analysts.run(ws2, model)
    assert not run2.ok and run2.stop_reason is None and ws2.run["replans"] == 1
    assert [f["id"] for f in run2.failed_steps] == ["s4"], "the cross-check cannot compare with a replaced fit"
    assert ws2.study["plan"]["replans"][0]["fallback"]["tool"] == "anywhere"
    assert ws2.study["steps"][2]["fallback"]["step"]["tool"] == "anywhere"
    r3 = ws2.run["results"][2]
    assert r3["fallback"]["tool"] == "anywhere" and r3["fallback"]["ok"] and r3["fallback"]["gates_passed"]
    # s4 is not called at all: it snaps its GloFAS cell to s3's mean flow, which the replaced fit does not carry
    assert [c[0] for c in calls2] == ["describe_catchment", "analyze_station", "flood_frequency", "similar_basins",
                                      "flood_frequency", "anywhere"], "passed steps are reused"
    assert ws2.ledger["analyst"]["calls"] == 2, "one proposal for s3, one (empty) for the cross-check s4"
    assert client.requests[0]["context"]["failed_step"]["id"] == "s3"
    assert any(e["event"] == "replan" and e["role"] == "analyst" for e in ws2.events)


def test_a_proposal_that_fails_the_validator_is_refused(no_deliverables):
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    ws = _ws()
    client = FakeModel({"analyst": [{"tool": "anywhere", "arguments": {"lat": 1, "lon": 2, "bogus": 3},
                                     "rationale": "x"}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    with patched(RECON, tools=fake_tools([], flood_frequency=wide, analyze_station=wide,
                                         similar_basins={"k": 1, "stations": []})):
        run = analysts.run(ws, model)
    assert not run.ok and run.stop_reason is None and ws.run["replans"] == 0, "a refused proposal is not a replan"
    assert [f["id"] for f in run.failed_steps if not f["skipped"]] == ["s3"], "the step stays not established"
    assert any(e["event"] == "no_fallback" and "bogus" in e["detail"] for e in ws.events)


def test_a_branch_fallback_replans_through_the_playbook(no_deliverables):
    wide = json.loads(json.dumps(FLOW))
    wide["ffa"]["fits"]["lp3"]["q"][5] = 900
    ws = _ws()
    study = ws.study
    study["steps"][2]["fallback"] = {"branch": "regional"}
    ws.study = study
    calls: list = []
    with patched(RECON, tools=fake_tools(calls, flood_frequency=wide, analyze_station=wide)):
        run = analysts.run(ws, None)
    assert ws.study["plan"]["branch"] == "regional" and ws.study["plan"]["replanned_from"]["step"] == "s3"
    assert ws.study["version"] == 3 and ws.study["plan"]["objective"], "the plan block carries over"
    assert [s["tool"] for s in ws.study["steps"]] == ["describe_catchment", "similar_basins",
                                                      "regionalize_signatures", "anywhere"]
    assert run.ok and ws.run["replans"] == 1
    assert calls[0][0] == "describe_catchment" and "describe_catchment" not in [c[0] for c in calls[1:]], "reused"


def test_prior_results_are_reused_unless_the_gates_changed(no_deliverables):
    ws = _ws()
    calls: list = []
    with patched(RECON, tools=fake_tools(calls)):
        analysts.run(ws, None)
        prior = analysts.prior_run(ws)
        assert prior is not None and len(prior.results) == 4
        methodologist.change(ws, None, "T = 50", intake={"return_period": 50})
        analysts.run(ws, None, prior=prior)
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere",
                                     "flood_frequency", "anywhere"]
    gate = next(g for g in ws.run["gates"] if g["check"] == "max_return_period_factor")
    assert "T = 50 years" in gate["detail"]


def test_the_requested_return_period_reaches_the_flood_steps() -> None:
    from aquascope.studio.roles.analysts import _ask_for_the_return_period
    from aquascope.studio.workspace import Workspace
    from aquascope.study import Step, Study

    ws = Workspace(site={"lat": 51.4, "lon": -0.3})
    ws.brief.intake["return_period"] = 200
    study = Study(question="q", version=3, steps=[
        Step(tool="describe_catchment", id="s1", arguments={"lat": 51.4, "lon": -0.3}),
        Step(tool="flood_frequency", id="s2", arguments={"source": "uk_ea", "station_id": "x", "bootstrap_ci": True}),
        Step(tool="analyze_station", id="s3", arguments={"source": "uk_ea", "station_id": "x",
                                                         "return_periods": [10, 200]}),
    ])
    _ask_for_the_return_period(ws, study)
    assert "return_periods" not in study.steps[0].arguments
    assert study.steps[1].arguments["return_periods"] == [2, 5, 10, 25, 50, 100, 200]
    assert study.steps[2].arguments["return_periods"] == [10, 200]
    ws.brief.intake["return_period"] = 100
    _ask_for_the_return_period(ws, study)
    assert study.steps[1].arguments["return_periods"] == [2, 5, 10, 25, 50, 100, 200]


def test_the_station_wrapper_and_the_gate_take_the_requested_return_period(monkeypatch) -> None:
    from aquascope import explore, gates
    from aquascope.studio.roles import analysts
    from aquascope.studio.workspace import Workspace
    from aquascope.study import Step, Study

    seen: dict = {}

    def fake_analyze(source, station_id, *, years=None, store=None, variable=None, period_start=None,
                     return_periods=None):
        seen["return_periods"] = return_periods
        return {"years": 40.0, "unit": "m3/s", "ffa": {"return_periods": return_periods or [2, 100]}}

    monkeypatch.setattr(explore, "analyze_station", fake_analyze)
    out = analysts.analyze_station_full("uk_ea", "x", return_periods=[2, 200])
    assert seen["return_periods"] == [2, 200] and out["years"] == 40.0
    ws = Workspace(site={"lat": 51.4, "lon": -0.3})
    ws.brief.intake["return_period"] = 200
    study = Study(question="q", version=3, steps=[
        Step(tool="flood_frequency", id="s1", arguments={"source": "uk_ea", "station_id": "x"},
             expects=[{"check": "max_return_period_factor", "value": 3, "path": "years"}]),
    ])
    analysts._ask_for_the_return_period(ws, study)
    assert study.steps[0].expects[0]["return_period"] == 200
    ok = gates.evaluate([{"check": "max_return_period_factor", "value": 3, "path": "years"}], {"years": 40.0})[0]
    assert ok["passed"], "a gate with no return period named is not applicable, not failed"


def test_a_method_argument_the_tool_rejects_is_rejected_with_the_choices_named() -> None:
    from aquascope.studio import catalogue

    steps = [{"id": "s1", "tool": "similar_basins", "arguments": {"lat": 51.4, "lon": -0.3, "k": 5,
                                                                    "method": "physio_climatic"}},
             {"id": "s2", "tool": "regionalize_signatures", "arguments": {"lat": 51.4, "lon": -0.3, "k": 5,
                                                                           "method": "regionalization"}}]
    errors = catalogue.validate_plan(steps)
    assert len(errors) == 2 and "'similarity', 'proximity', 'combined'" in errors[0] and "'both'" in errors[1]
    assert steps[0]["arguments"]["method"] == "physio_climatic", "never substituted (#413)"
    assert "notes" not in steps[0]
    bad = [{"id": "s1", "tool": "similar_basins", "arguments": {"lat": 1, "lon": 2, "method": "nope"}}]
    assert catalogue.validate_plan(bad, repair=False)
