"""The Methodologist: the tree keyless, a model's plan validated, repaired, refused or replaced by the tree,
the user's edits revalidated, a change after the report."""

from __future__ import annotations

import pytest

from aquascope.studio.model import Model
from aquascope.studio.roles import methodologist, scout
from aquascope.studio.workspace import Workspace
from aquascope.study import loads
from tests.test_studio.conftest import PROBLEM, RECON, FakeModel, patched, recon


def _ws(problem: str = PROBLEM, playbook: str | None = "flood_risk", intake: dict | None = None,
        recon_value: dict = RECON) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem = problem
    ws.brief.playbook = playbook
    ws.brief.kind = {"flood_risk": "flood_risk", "drought_status": "drought", None: None}.get(playbook, playbook)
    ws.brief.intake = dict(intake or {"return_period": 100})
    ws.brief.assumptions = ["the gauge is representative"]
    with patched(recon_value):
        scout.scout(ws)
    return ws


VALID_PLAN = {
    "objective": "The 100-year design flow at Kingston with its band",
    "decision": "size the crossing",
    "methodology": ["Frame the catchment.", "Test the record for a trend.", "Fit two distributions."],
    "steps": [
        {"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 51.415, "lon": -0.308},
         "rationale": "Catchment size and regulation.", "expects": [{"check": "not_empty", "path": "sub_basin"}],
         "outputs": [{"kind": "table", "id": "s1_catchment", "caption": "Catchment attributes"}]},
        {"id": "s2", "tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": "3400TH"},
         "method": "trend_mann_kendall", "rationale": "Trend pre-test.",
         "expects": [{"check": "min_years", "value": 20, "path": "years"}, {"check": "unit_present"}]},
        {"id": "s3", "tool": "flood_frequency", "depends_on": ["s2"],
         "arguments": {"source": "uk_ea", "station_id": "3400TH", "bootstrap_ci": True},
         "method": "at_site_flood_frequency", "rationale": "Two fits with a band.",
         "expects": [{"check": "max_return_period_factor", "value": 3, "path": "years", "return_period": 100},
                     {"check": "spread_within", "value": 0.25, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
                      "return_period": 100}],
         "fallback": {"step": {"tool": "similar_basins", "arguments": {"lat": 51.415, "lon": -0.308, "k": 5},
                               "rationale": "Donors as a cross-check."}}},
        {"id": "s4", "tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": 20},
         "method": "glofas_cross_check", "rationale": "GloFAS as an independent cross-check.",
         "expects": [{"check": "not_empty", "path": "glofas"}]},
    ],
    "assumptions": ["the record is stationary enough"],
    "alternatives": [{"method": "nonstationary fit", "why_not": "guidance is immature"}],
    "limitations_expected": ["GloFAS is a grid cell, not the gauge"],
    "citations": ["Hosking (1990) L-moments"],
}


def test_the_tree_plans_keyless_as_version_3():
    ws = _ws()
    study = methodologist.plan(ws, None)
    assert study is not None and study.version == 3 and study.author == "playbook"
    plan = study.plan
    assert plan["author"] == "playbook" and plan["playbook"] == "flood_risk" and plan["branch"] == "at_site"
    assert plan["objective"] == PROBLEM and len(plan["methodology"]) == 4
    assert plan["assumptions"] == ["the gauge is representative"] and plan["caveats"] and plan["citations"]
    assert all(s.outputs for s in study.steps) and study.steps[2].outputs[0] == {
        "kind": "figure", "id": "s3_frequency_curve", "caption": "frequency curve from flood_frequency"}
    assert ws.study["version"] == 3 and ws.messages[-1].kind == "plan"
    assert "[s3] flood_frequency" in ws.messages[-1].text
    back = loads(study.to_yaml())
    assert back.version == 3 and back.plan["methodology"] == plan["methodology"] and back.steps[2].outputs
    assert ws.brief.intake == {"return_period": 100, "decision": "design flow"}, "the filled intake is written back"


def test_keyless_declines_are_the_playbooks_words_or_the_missing_playbook():
    ws = _ws(playbook=None, intake={})
    assert methodologist.plan(ws, None) is None
    assert ws.status == "declined" and "flood_risk" in ws.declined_reason and "add a model" in ws.declined_reason
    short = recon({"discharge": 12}, [dict(RECON["stations"][0], years=12)], donors=1)
    ws2 = _ws(intake={"return_period": 100}, recon_value=short)
    assert methodologist.plan(ws2, None) is None
    assert ws2.status == "declined" and "36 years" in ws2.declined_reason and ws2.messages[-1].kind == "declined"


def test_a_valid_model_plan_is_accepted_with_the_playbooks_caveats_attached():
    ws = _ws()
    client = FakeModel({"methodologist": [VALID_PLAN]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist" and study.version == 3 and study.model == "fake"
    assert [s.tool for s in study.steps] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    plan = study.plan
    assert plan["author"] == "methodologist" and plan["objective"].startswith("The 100-year")
    assert plan["alternatives"][0]["why_not"] == "guidance is immature" and plan["limitations_expected"]
    assert any("Wasko" in c for c in plan["caveats"]) and "Hosking (1990) L-moments" in plan["citations"]
    assert any("Bulletin 17C" in c for c in plan["citations"]), "the playbook's citations come along"
    assert plan["assumptions"] == ["the gauge is representative", "the record is stationary enough"]
    assert study.steps[0].outputs == [{"kind": "table", "id": "s1_catchment", "caption": "Catchment attributes"}]
    assert study.steps[3].outputs and study.steps[3].outputs[0]["id"] == "s4_monthly_climate"
    assert ws.ledger["methodologist"]["calls"] == 1
    ctx = client.requests[0]["context"]
    assert ctx["exemplar"]["branch"] == "at_site" and ctx["gates"]["min_years"] and ctx["brief"]["playbook"]
    assert {e["tool"] for e in ctx["catalogue"]} >= {"analyze_station", "flood_frequency", "similar_basins",
                                                      "describe_catchment"}
    listed = {e["tool"] for e in ctx["catalogue"]}
    assert "eda" not in listed, "a table's descriptive tools are listed only with an upload"
    assert {"wqi", "who_screen", "return_periods"} <= listed, "the analytic table tools are listed: a step feeds them"
    assert len(__import__("json").dumps(ctx)) < 20_000


def test_an_invalid_plan_gets_one_repair_call():
    ws = _ws()
    broken = dict(VALID_PLAN, steps=[dict(VALID_PLAN["steps"][0], tool="frobnicate"),
                                     dict(VALID_PLAN["steps"][1], expects=[{"check": "min_yearz"}])])
    client = FakeModel({"methodologist": [broken, VALID_PLAN]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist" and len(study.steps) == 4
    kinds = [e["event"] for e in ws.events if e["role"] == "methodologist" and e["event"] != "model_call"]
    assert kinds[:2] == ["invalid", "repaired"] and ws.ledger["methodologist"]["calls"] == 2
    repair = client.requests[1]["context"]
    assert any("frobnicate" in e for e in repair["errors"]) and any("min_yearz" in e for e in repair["errors"])


def test_a_repair_reply_wrapped_as_plan_is_unwrapped():
    ws = _ws()
    broken = dict(VALID_PLAN, steps=[dict(VALID_PLAN["steps"][0], tool="frobnicate")])
    client = FakeModel({"methodologist": [broken, {"plan": VALID_PLAN}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist" and len(study.steps) == 4


def test_an_unrepairable_plan_falls_back_to_the_tree_or_declines():
    ws = _ws()
    broken = dict(VALID_PLAN, steps=[dict(VALID_PLAN["steps"][0], tool="frobnicate")])
    client = FakeModel({"methodologist": [broken, broken]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "playbook" and study.plan["branch"] == "at_site"
    assert any("frobnicate" in e for e in study.plan["model_plan_rejected"])
    assert any(e["event"] == "fallback" for e in ws.events if e["role"] == "methodologist")
    ws2 = _ws(playbook=None, intake={})
    client2 = FakeModel({"methodologist": [broken, broken]})
    model2 = Model.resolve(ws2, client=client2, model="fake", provider="custom")
    assert methodologist.plan(ws2, model2) is None
    assert ws2.status == "declined" and "frobnicate" in ws2.declined_reason


def test_a_method_the_registry_calls_not_defensible_is_refused():
    site = recon({"discharge": 7}, [dict(RECON["stations"][0], years=7)], donors=5, area=300)
    site["sufficiency"] = [{"method": "at_site_flood_frequency", "status": "not_defensible",
                            "reason": "7 years of discharge, below the 10-year floor"}]
    ws = _ws(recon_value=site)
    client = FakeModel({"methodologist": [VALID_PLAN, VALID_PLAN]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is not None and study.author == "methodologist", "the rest of the plan stands"
    assert study.step_by_id("s3") is None and [s.id for s in study.steps] == ["s1", "s2", "s4"]
    assert any("not defensible" in n and "at_site_flood_frequency" in n for n in study.plan["notes"])
    assert any("not defensible" in e for e in client.requests[1]["context"]["errors"])


def test_placeholders_other_than_results_are_rejected():
    ws = _ws()
    plan = dict(VALID_PLAN, steps=[
        dict(VALID_PLAN["steps"][1], id="s1", arguments={"source": "{{ station.source }}", "station_id": "3400TH"}),
        {"id": "s2", "tool": "anywhere", "arguments": {"lat": 51.415, "lon": -0.308, "years": "{{ result.s1.years }}"},
         "depends_on": ["s1"], "rationale": "ok"}])
    steps, errors, notes = methodologist._check(plan, ws)
    assert len(errors) == 1 and "station" in errors[0] and "concrete value" in errors[0] and notes == []


def test_revise_applies_overrides_moves_gate_keys_and_refuses_bad_edits():
    ws = _ws()
    methodologist.plan(ws, None)
    study = methodologist.revise(ws, None, {"s3": {"arguments": {"bootstrap_ci": False, "return_period": 200}},
                                            "s1": None})
    assert [s.id for s in study.steps] == ["s2", "s3", "s4"] and study.plan["edited"] is True
    s3 = study.step_by_id("s3")
    assert s3.arguments["bootstrap_ci"] is False and "return_period" not in s3.arguments
    assert {g["return_period"] for g in s3.expects if "return_period" in g} == {200}
    assert ws.messages[-1].kind == "plan" and ws.study["steps"][0]["id"] == "s2"
    with pytest.raises(ValueError, match="takes no argument"):
        methodologist.revise(ws, None, {"s2": {"arguments": {"nope": 1}}})
    assert ws.study["steps"][0]["id"] == "s2", "a refused edit leaves the plan as it was"
    with pytest.raises(ValueError, match="no steps"):
        methodologist.revise(ws, None, [])
    replaced = methodologist.revise(ws, None, [{"id": "a", "tool": "anywhere", "arguments": {"lat": 1, "lon": 2}}])
    assert [s.tool for s in replaced.steps] == ["anywhere"] and replaced.steps[0].outputs


def test_change_replans_the_tree_with_a_new_intake_or_takes_the_models_steps():
    ws = _ws()
    methodologist.plan(ws, None)
    ws.run = {"results": [{"id": "s3", "ok": True, "gates": [{"check": "min_years", "passed": True}]}]}
    study = methodologist.change(ws, None, "500 years", intake={"return_period": 500})
    assert study is not None and ws.brief.intake["return_period"] == 500 and study.plan["changed_for"] == "500 years"
    assert any(g.get("return_period") == 500 for g in study.step_by_id("s3").expects)
    client = FakeModel({"methodologist": [{"steps": VALID_PLAN["steps"][:2] + [
        {"id": "s9", "tool": "similar_basins", "arguments": {"lat": 51.415, "lon": -0.308, "k": 8},
         "rationale": "More donors.", "expects": [{"check": "min_donors", "value": 3, "path": "k"}]}],
        "methodology": ["a", "b", "c"]}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    changed = methodologist.change(ws, model, "add eight donors")
    assert [s.id for s in changed.steps] == ["s1", "s2", "s9"] and changed.plan["methodology"] == ["a", "b", "c"]
    assert changed.plan["objective"], "the plan block carries over"
    assert client.requests[0]["context"]["steps"][2]["outcome"]["gates"] == ["min_years: ok"]


def test_plan_text_is_a_numbered_checklist():
    ws = _ws()
    methodologist.plan(ws, None)
    text = methodologist.plan_text(ws.study)
    assert text.startswith("Plan (playbook, playbook flood_risk, branch at_site, 4 step(s))")
    assert "2. [s2] analyze_station(source='uk_ea', station_id='3400TH')  method trend_mann_kendall" in text
    assert "gate max_return_period_factor 3 on ffa.n_years" in text and "fallback: similar_basins" in text
    assert "caveat(s) will be printed verbatim" in text


def test_a_guessed_gate_path_is_repaired_to_the_one_the_tool_has() -> None:
    from aquascope.studio import catalogue

    steps = [
        {"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 51.4, "lon": -0.3},
         "expects": [{"check": "not_empty", "path": "catchment"},
                     {"check": "max_area_km2", "value": 20000, "path": "catchment.upstream_area_km2"}]},
        {"id": "s2", "tool": "similar_basins", "arguments": {"lat": 51.4, "lon": -0.3, "k": 10},
         "expects": [{"check": "min_donors", "value": 5, "path": "donors"}]},
        {"id": "s3", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "x"},
         "expects": [{"check": "spread_within", "value": 0.25, "path": "ffa.fits.gev_lmoments.q, ffa.fits.lp3.q"},
                     {"check": "ci_finite", "path": "ffa.fits.gev_bootstrap.ci"}]},
    ]
    assert catalogue.validate_plan(steps) == []
    assert steps[0]["expects"][0]["path"] == "sub_basin" and steps[0]["expects"][0]["repaired_from"] == "catchment"
    assert steps[0]["expects"][1]["path"] == "sub_basin.up_area"
    assert steps[1]["expects"][0]["path"] == "k"
    assert "repaired_from" not in steps[2]["expects"][0] and "repaired_from" not in steps[2]["expects"][1]
    # the model reads the paths it should copy
    entry = next(e for e in catalogue.compact("flood_risk") if e["tool"] == "similar_basins")
    assert {"check": "min_donors", "path": "k"} in entry["gates"]


def test_a_model_that_declines_is_final_and_the_tree_does_not_run(monkeypatch) -> None:
    from aquascope.studio.model import Model
    from aquascope.studio.roles import methodologist, scout
    from aquascope.studio.workspace import Workspace
    from tests.test_studio.conftest import RECON, FakeModel

    ws = Workspace(site={"lat": 51.415, "lon": -0.308})
    ws.brief.problem = "Map the inundation extent for the 100-year flood"
    ws.brief.kind = ws.brief.playbook = "flood_risk"
    ws.brief.intake = {"return_period": 100}

    monkeypatch.setattr("aquascope.explore.assess_site", lambda *a, **k: RECON, raising=False)
    scout.scout(ws)
    reason = "No tool in the catalogue maps an inundation extent; the flood tools give a flow, not a depth."
    client = FakeModel({"methodologist": [{"decline": True, "reason": reason}]})
    model = Model.resolve(ws, client=client, model="fake", provider="custom")
    study = methodologist.plan(ws, model)
    assert study is None and ws.status == "declined" and "inundation" in (ws.declined_reason or "")
    assert not any(e["event"] == "fallback" for e in ws.events), "a model's decline is final; the tree does not run"


def test_a_station_the_inventory_does_not_know_is_an_error() -> None:
    from aquascope.studio import catalogue

    steps = [{"id": "s1", "tool": "analyze_station", "arguments": {"source": "usgs", "station_id": "01646500"}}]
    assert catalogue.validate_plan(steps, stations={("uk_ea", "3400TH")}) == [
        "step s1: station usgs 01646500 is not in the inventory of this site"]
    assert catalogue.validate_plan(steps, stations={("usgs", "01646500")}) == []
    assert catalogue.validate_plan(steps, stations=None) == [], "no inventory, no check"
    entry = catalogue.get("supply_reliability")
    assert "regionalize_signatures" in entry.methods, "the ungauged mode of the supply screening is reachable"
