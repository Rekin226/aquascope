"""The advanced playbooks plan offline, the keyword rules route to them, the registry demotes on what a method
assumes (#376), and the runner runs a skipped step's fallback."""

from __future__ import annotations

import pytest

from aquascope import playbooks as pbk
from aquascope.ai_engine import team
from aquascope.methods import MARGINAL, METHODS, SiteContext, assess_method
from aquascope.study import Step, Study, run_study

GAUGE = {"source": "usgs", "station_id": "USGS-01013500", "name": "Fish River near Fort Kent", "distance_km": 0.1,
         "latitude": 47.2375, "longitude": -68.5828, "variables": ["discharge"], "years": 100.0}


def recon(years: float | None, *, stations=None) -> dict:
    by = {"discharge": years} if years else {}
    return {"point": {"lat": 47.2375, "lon": -68.5828}, "stations": list(stations if stations is not None else
                                                                         ([dict(GAUGE, years=years)] if years else [])),
            "catchment": {"area_km2": 2320, "upstream_area_km2": 2320, "dams": 0},
            "context": {"years_by_variable": by, "resolution_by_variable": {k: "daily" for k in by},
                        "area_km2": 2320, "donors": 8, "ungauged": not by,
                        "available": ["glofas", "temperature", "forcing", "gcms>=3"]},
            "sufficiency": [], "notes": []}


# ── routing ──


@pytest.mark.parametrize(("text", "playbook"), [
    ("What is the 100-year flood for a culvert here?", "flood_risk"),
    ("Is the 100-year flood getting worse?", "flood_change"),
    ("Have floods increased since 1970?", "flood_change"),
    ("How will climate change affect floods here?", "climate_change"),
    ("What will the river look like in 2050 under climate change?", "climate_change"),
    ("What if rainfall drops 10% and it gets 2 degrees warmer?", "catchment_response"),
    ("Calibrate a rainfall-runoff model for this catchment", "catchment_response"),
    ("Can this river supply 2 m3/s to a town?", "supply_reliability"),
])
def test_the_keyword_rules_route_the_advanced_questions(text, playbook):
    assert team.choose_playbook(text)[0] == playbook


def test_the_intake_hints_read_the_what_if_numbers():
    assert team.intake_hints("What if rainfall drops 10% and it gets 2 degrees warmer?") == \
        {"dp_pct": -10.0, "dt_c": 2.0}
    assert team.intake_hints("with 15% less rain and 1.5 °C warmer")["dp_pct"] == -15.0
    assert team.intake_hints("Is the 100-year flood getting worse by 2060?", "flood_change")["horizon_year"] == 2060


# ── the trees ──


def test_flood_change_plans_the_change_tests_with_the_stationary_reference():
    study = pbk.plan("flood_change", recon(100.0), {"return_period": 100})
    assert study.plan["branch"] == "long_record"
    assert [s.tool for s in study.steps] == ["change_points", "flood_frequency", "nonstationary_flood", "pot_flood",
                                             "regional_flood"]
    ns = study.steps[2]
    assert ns.arguments["horizon_year"] == 2050 and ns.arguments["return_periods"] == [10, 100]


def test_flood_change_without_a_long_record_goes_regional():
    study = pbk.plan("flood_change", recon(None), {})
    assert study.plan["branch"] == "regional_only" and [s.tool for s in study.steps] == ["regional_flood"]


def test_flood_change_declines_a_far_horizon():
    with pytest.raises(pbk.Declined, match="not a forecast"):
        pbk.plan("flood_change", recon(100.0), {"horizon_year": 2090})


def test_climate_change_chains_the_model_into_the_projection_with_a_climate_only_fallback():
    study = pbk.plan("climate_change", recon(100.0), {})
    assert study.plan["branch"] == "gauged"
    s1, s2, s3 = study.steps
    assert (s1.tool, s2.tool, s3.tool) == ("describe_catchment", "catchment_model", "climate_projection")
    assert s2.arguments["lat"] == 47.2375 and s2.arguments["area_km2"] == "{{ result.s1.sub_basin.up_area }}"
    assert s3.arguments["params"] == "{{ result.s2.params }}" and s3.depends_on == ["s1", "s2"]
    assert s3.fallback["step"]["tool"] == "climate_projection" and "params" not in s3.fallback["step"]["arguments"]


def test_climate_change_ungauged_is_climate_only_and_declines_2100():
    assert [s.tool for s in pbk.plan("climate_change", recon(None), {}).steps] == ["climate_projection"]
    with pytest.raises(pbk.Declined, match="end in 2050"):
        pbk.plan("climate_change", recon(100.0), {"horizon_year": 2100})


def test_catchment_response_carries_the_asked_scenarios():
    study = pbk.plan("catchment_response", recon(30.0), {"dp_pct": -15, "dt_c": 1.5})
    scen = study.steps[1].arguments["scenarios"]
    assert [s["label"] for s in scen] == ["rainfall -15%", "1.5 degrees warmer",
                                          "rainfall -15% and 1.5 degrees warmer"]
    assert scen[2]["dp_pct"] == -15 and scen[2]["dt_c"] == 1.5


def test_catchment_response_declines_without_a_gauge():
    with pytest.raises(pbk.Declined, match="no gauge with five years"):
        pbk.plan("catchment_response", recon(None), {})


# ── #376: what a method assumes ──


def _ctx(**kw) -> SiteContext:
    return SiteContext(years_by_variable={"discharge": 60}, resolution_by_variable={"discharge": "daily"},
                       available=set(kw.pop("available", ())), **kw)


def test_a_clean_long_record_stays_defensible():
    assert assess_method("at_site_flood_frequency", _ctx())["status"] == "defensible"


def test_a_change_point_inside_the_record_demotes_stationary_methods_naming_the_year():
    row = assess_method("at_site_flood_frequency", _ctx(change_points=[1991]))
    assert row["status"] == MARGINAL and "1991" in row["reason"] and "two regimes" in row["reason"]
    # a method that does not assume stationarity is not touched
    assert assess_method("nonstationary_gev", _ctx(change_points=[1991]))["status"] == "defensible"


def test_regulation_and_snow_are_caveats_not_blocks():
    reg = assess_method("at_site_flood_frequency", _ctx(available={"regulation"}))
    assert reg["status"] == MARGINAL and "operated" in reg["reason"]
    snow = assess_method("gr4j_calibration", _ctx(available={"snow", "forcing"}))
    assert snow["status"] == MARGINAL and "snow" in snow["reason"]


def test_every_method_declares_what_it_assumes():
    known = {"stationarity", "independence", "homogeneity"}
    for m in METHODS.values():
        assert set(m.assumes) <= known and set(m.sensitive_to) <= {"regulation", "snow"}, m.id
    assert "stationarity" in METHODS["at_site_flood_frequency"].assumes


# ── the runner: a step skipped for its input still runs its fallback ──


def test_a_skipped_step_runs_its_fallback_when_the_fallback_needs_nothing_that_failed():
    calls: list[str] = []
    tools = {
        "model": lambda **kw: calls.append("model") or {"kge": 0.2},
        "project": lambda **kw: calls.append("project:" + ("params" if "params" in kw else "climate"))
        or {"n_models": 7},
    }
    study = Study(question="q", version=3, steps=[
        Step(tool="model", id="s1", expects=[{"check": "kge_min", "value": 0.5, "path": "kge"}]),
        Step(tool="project", id="s2", depends_on=["s1"], arguments={"params": "{{ result.s1.kge }}"},
             expects=[{"check": "min_models", "value": 3}],
             fallback={"step": {"tool": "project", "arguments": {},
                                "expects": [{"check": "min_models", "value": 3}]}}),
    ])
    run = run_study(study, tools=tools)
    assert calls == ["model", "project:climate"]
    s2 = run.results[1]
    assert s2["fallback_used"] and not s2.get("skipped") and "fallback project ran instead" in s2["failed_reason"]
    assert s2["fallback"]["gates_passed"]


def test_a_skipped_step_with_no_fallback_stays_skipped():
    tools = {"a": lambda **kw: {"kge": 0.1}, "b": lambda **kw: {"x": 1}}
    study = Study(question="q", version=3, steps=[
        Step(tool="a", id="s1", expects=[{"check": "kge_min", "value": 0.5, "path": "kge"}]),
        Step(tool="b", id="s2", depends_on=["s1"], arguments={"v": "{{ result.s1.kge }}"}),
    ])
    run = run_study(study, tools=tools)
    assert run.results[1]["skipped"] and not run.results[1]["fallback_used"]


# ── #376 in the core flood fit: Pettitt on the annual maxima the stationary fit uses ──


def _flows(years: int, shift_at: int | None, seed: int) -> object:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-01", periods=int(years * 365.25), freq="D")
    q = 20 * rng.lognormal(0, 0.3, len(idx))
    for y in sorted(set(idx.year)):
        k = idx.get_loc(pd.Timestamp(f"{y}-04-20"))
        bump = 2.0 if shift_at is not None and y >= 1960 + shift_at else 1.0
        q[k] += bump * (80 + 20 * rng.gumbel())
    return pd.Series(q, index=idx)


def test_the_flood_fit_reports_a_step_change_in_its_maxima():
    from aquascope.explore import analyze_series

    shifted = analyze_series(_flows(50, 25, 3), "discharge", "m3/s")["ffa"]["amax_change"]
    assert shifted["significant"] and abs(shifted["change_year"] - 1985) <= 3 and shifted["mean_after"] > \
        shifted["mean_before"]
    clean = analyze_series(_flows(50, None, 4), "discharge", "m3/s")["ffa"]["amax_change"]
    assert clean["significant"] is False and clean["test"] == "Pettitt"


def test_a_regime_shift_behind_the_headline_makes_the_answer_indicative():
    from aquascope.studio.roles import interpreter
    from aquascope.studio.workspace import Workspace

    ws = Workspace()
    ws.run = {"results": [{"id": "s3", "tool": "flood_frequency", "ok": True, "gates_passed": True, "gates": [],
                           "result": {"ffa": {"amax_change": {"significant": True, "change_year": 1991,
                                                              "p_value": 0.01}}}}]}
    assert interpreter.regime_shift(ws, "s3")["change_year"] == 1991
    ws.run["results"][0]["result"]["ffa"]["amax_change"]["significant"] = False
    assert interpreter.regime_shift(ws, "s3") is None
    ws.run["results"][0]["result"] = {"ffa": {}}
    assert interpreter.regime_shift(ws, "s3") is None, "a payload without the test changes nothing"
