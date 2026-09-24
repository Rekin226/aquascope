"""Steerable steps: the declarations, the validation of a change, the rerun of a step and its dependants only,
the change recorded in study.yaml, the plan in plain words, and the Studio's steer after the report."""

from __future__ import annotations

import json

import pytest

from aquascope.gates import CHECKS, plain
from aquascope.studio import catalogue
from aquascope.studio.steering import (
    apply_change,
    controls_for,
    declared,
    dependants,
    plain_plan,
    rerun_step,
    steerable,
    validate_change,
)
from aquascope.study import Step, Study, loads
from tests.test_studio.conftest import PROBLEM, fake_tools

# ── a four-step study over fake tools ──────────────────────────────────────


def _flood(**kw):
    """flood_frequency whose payload reports the periods it was asked for, as the real tool does."""
    periods = [float(p) for p in (kw.get("return_periods") or [2, 5, 10, 25, 50, 100])]
    q = [100.0 + 2.0 * p for p in periods]
    return {"source": kw.get("source"), "station_id": kw.get("station_id"), "unit": "m3/s",
            "years": 80.0 if not kw.get("years") else float(kw["years"]),
            "ffa": {"return_periods": periods,
                    "fits": {"gev_lmoments": {"q": q, "q_by_T": {f"{p:g}": v for p, v in zip(periods, q)}},
                             "lp3": {"q": [v * 1.05 for v in q]}}}}


def _study() -> Study:
    return Study.from_dict({
        "version": 3, "question": "Design flood", "problem": {"kind": "flood_risk", "params": {"return_period": 100}},
        "plan": {"playbook": "flood_risk"},
        "steps": [
            {"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 51.4, "lon": -0.3}},
            {"id": "s2", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "X"},
             "method": "at_site_flood_frequency",
             "expects": [{"check": "min_years", "value": 20, "path": "years"},
                         {"check": "max_return_period_factor", "value": 3, "path": "years", "return_period": 100}]},
            {"id": "s3", "tool": "anywhere", "arguments": {"lat": 51.4, "lon": -0.3}, "depends_on": ["s2"],
             "expects": [{"check": "cross_check_ratio", "value": 0.5, "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                          "reference": "{{ result.s2.ffa.fits.gev_lmoments.q_by_T }}", "return_period": 100}]},
            {"id": "s4", "tool": "low_flow_context", "arguments": {"source": "uk_ea", "station_id": "X"}},
        ],
    })


def _tools(calls):
    return fake_tools(calls, flood_frequency=_flood)


def _first_run(calls):
    from aquascope.study import run_study

    return run_study(_study(), tools=_tools(calls))


# ── declarations ────────────────────────────────────────────────────────────


def test_every_declared_control_names_a_real_argument_of_its_tool():
    for tool, controls in steerable().items():
        entry = catalogue.get(tool)
        assert entry is not None, f"{tool} is not in the catalogue"
        assert entry.steer == declared(tool) and entry.to_dict()["steer"] == entry.steer
        for c in controls:
            assert c.type in ("choice", "number", "integer", "boolean")
            if c.argument:
                assert c.argument in entry.arguments, f"{tool}.{c.argument} is not an argument of the tool"
            else:
                assert c.param == "return_period" and "return_periods" in entry.arguments
            if c.type == "choice":
                assert c.choices
            enum = (entry.arguments.get(c.argument or "") or {}).get("enum")
            if enum:
                assert set(c.choices) <= set(enum), f"{tool}.{c.param} offers a value the tool refuses"


def test_the_examples_in_the_brief_are_declared():
    params = {t: {c.param for c in cs} for t, cs in steerable().items()}
    assert {"distribution"} <= params["return_periods"]
    assert {"return_period", "years"} <= params["flood_frequency"]
    assert {"threshold"} <= params["drought_indices"]
    assert "gumbel" in next(c for c in steerable()["return_periods"] if c.param == "distribution").choices


def test_a_method_limits_a_control_to_the_steps_that_apply_it():
    trend = Step(tool="analyze_station", id="s2", method="trend_mann_kendall")
    flood = Step(tool="analyze_station", id="s2", method="at_site_flood_frequency")
    assert "return_period" not in {c.param for c in controls_for(trend)}
    assert "return_period" in {c.param for c in controls_for(flood)}
    assert controls_for(Step(tool="describe_catchment")) == ()


# ── validation ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("changes, reason", [
    ({"distribution": "lp3"}, "cannot be adjusted on step s2"),
    ({"return_period": 7}, "must be one of 10, 20, 25, 50, 100, 200, 500, 1000"),
    ({"years": 2}, "must be at least 5"),
    ({"years": 30.5}, "whole number"),
    ({"years": "thirty"}, "is a number"),
    ({"return_period": 100}, "already 100"),
    ({"bootstrap_ci": "maybe"}, "yes or no"),
    ({}, "no change was given"),
])
def test_a_change_is_refused_with_the_reason(changes, reason):
    step = _study().step_by_id("s2")
    clean, errors = validate_change(step, changes, _study())
    assert errors and reason in "; ".join(errors) and not clean


def test_a_change_is_coerced_to_its_type():
    step = _study().step_by_id("s2")
    clean, errors = validate_change(step, {"return_period": "200", "years": "30", "bootstrap_ci": "true"}, _study())
    assert errors == [] and clean == {"return_period": 200, "years": 30, "bootstrap_ci": True}
    wb = Step(tool="return_periods", id="s5", arguments={"from_step": "s1", "distribution": "gev"})
    assert validate_change(wb, {"distribution": "LP3"})[0] == {"distribution": "lp3"}
    assert "must be one of gev, lp3, gumbel" in validate_change(wb, {"distribution": "weibull"})[1][0]
    drought = Step(tool="drought_indices", id="s6", arguments={"lat": 1, "lon": 2})
    assert validate_change(drought, {"threshold": -1.5})[0] == {"threshold": -1.5}
    assert "at most 0" in validate_change(drought, {"threshold": 0.5})[1][0]


def test_an_argument_taken_from_another_step_cannot_be_steered():
    step = Step(tool="similar_basins", id="s5", arguments={"k": "{{ result.s1.k }}"})
    _, errors = validate_change(step, {"k": 12})
    assert "taken from another step's result" in errors[0]


def test_an_optional_control_resets_to_the_default():
    study = _study()
    study.step_by_id("s2").arguments["years"] = 30
    out = apply_change(study, "s2", {"years": None})
    assert "years" not in out["study"].step_by_id("s2").arguments
    assert out["change"]["changes"] == {"years": {"from": 30, "to": None}}
    assert "to the default" in out["text"]


# ── dependants and the change ───────────────────────────────────────────────


def test_dependants_follow_depends_on_references_and_from_step():
    study = _study()
    assert dependants(study, "s2") == ["s2", "s3"]
    assert dependants(study, "s4") == ["s4"]
    study.steps.append(Step(tool="return_periods", id="s5", arguments={"from_step": "s3"}))
    study.steps.append(Step(tool="eda", id="s6", arguments={"from_step": "s5"}))
    assert dependants(study, "s2") == ["s2", "s3", "s5", "s6"], "transitive, in plan order"


def test_apply_change_copies_the_study_and_records_the_change():
    study = _study()
    out = apply_change(study, "s2", {"return_period": 200}, at="2026-09-23T00:00:00+00:00")
    new = out["study"]
    assert study.step_by_id("s2").arguments == {"source": "uk_ea", "station_id": "X"}, "the input is untouched"
    s2 = new.step_by_id("s2")
    assert s2.arguments["return_periods"] == [2, 5, 10, 25, 50, 100, 200]
    assert all(g["return_period"] == 200 for s in (s2, new.step_by_id("s3")) for g in s.expects
               if "return_period" in g), "the design T reaches the dependants' gates"
    assert new.problem["params"]["return_period"] == 200, "the report's headline quotes the new T"
    assert out["dirty"] == ["s2", "s3"]
    assert new.plan["steering"] == [{"step": "s2", "tool": "flood_frequency",
                                     "changes": {"return_period": {"from": 100, "to": 200}},
                                     "reran": ["s2", "s3"], "at": "2026-09-23T00:00:00+00:00"}]
    assert out["text"] == "Step s2: design return period (years) 100 to 200"
    with pytest.raises(ValueError, match="there is no step 's9'"):
        apply_change(study, "s9", {"years": 30})


def test_the_change_survives_study_yaml_and_the_notebook_run():
    new = apply_change(_study(), "s2", {"return_period": 200, "years": 40})["study"]
    back = loads(new.to_yaml())
    assert back.step_by_id("s2").arguments["return_periods"][-1] == 200
    assert back.step_by_id("s2").arguments["years"] == 40
    assert back.plan["steering"][0]["changes"]["years"] == {"from": None, "to": 40}
    # a second change appends
    again = apply_change(back, "s4", {"years": 20})["study"]
    assert [c["step"] for c in again.plan["steering"]] == ["s2", "s4"]


def test_a_version_1_study_is_promoted_so_the_record_is_written():
    v1 = Study(question="q", steps=[Step(tool="low_flow_context", arguments={"source": "a", "station_id": "b"})])
    out = apply_change(v1, "s1", {"years": 15})
    text = out["study"].to_yaml()
    assert "version: 2" in text and "years: 15" in text
    assert loads(text).plan["steering"][0]["changes"] == {"years": {"from": None, "to": 15}}


# ── the rerun ───────────────────────────────────────────────────────────────


def test_rerun_step_runs_the_step_and_its_dependants_only():
    calls: list = []
    prior = _first_run(calls)
    assert [c[0] for c in calls] == ["describe_catchment", "flood_frequency", "anywhere", "low_flow_context"]
    calls.clear()
    out = rerun_step(prior, "s2", {"return_period": 200}, tools=_tools(calls))
    assert out["ok"] and out["errors"] == []
    assert [c[0] for c in calls] == ["flood_frequency", "anywhere"], "s1 and s4 are reused, not fetched again"
    assert calls[0][1]["return_periods"] == [2, 5, 10, 25, 50, 100, 200]
    assert out["rerun"] == ["s2", "s3"] and out["reused"] == ["s1", "s4"]
    assert "T = 200" in next(g["detail"] for g in out["gates"] if g["check"] == "max_return_period_factor")
    assert "steering:" in out["study_yaml"] and json.dumps(out["study"])


def test_rerun_step_keeps_a_failed_independent_step_as_it_was():
    calls: list = []
    tools = _tools(calls)
    tools["low_flow_context"] = lambda **kw: calls.append(("low_flow_context", kw)) or {"error": "agency down"}
    from aquascope.study import run_study

    prior = run_study(_study(), tools=tools)
    assert not prior.ok
    calls.clear()
    out = rerun_step({"study": prior.study.to_dict(), "results": prior.results}, "s2", {"years": 40}, tools=tools)
    assert [c[0] for c in calls] == ["flood_frequency", "anywhere"], "the failed s4 is not retried"
    assert out["run_ok"] is False, "a failed result kept by the rerun still fails the run"
    assert next(r for r in out["results"] if r["id"] == "s4")["error"] == "agency down"


def test_a_refused_change_runs_nothing():
    calls: list = []
    prior = _first_run(calls)
    calls.clear()
    out = rerun_step(prior, "s2", {"distribution": "lp3"}, tools=_tools(calls))
    assert out["ok"] is False and "cannot be adjusted" in out["errors"][0] and calls == []


def test_rerun_step_without_a_prior_run_runs_everything():
    calls: list = []
    out = rerun_step(_study().to_yaml(), "s4", {"years": 25}, tools=_tools(calls))
    assert out["ok"] and len(calls) == 4 and out["reused"] == []


# ── the plan in plain words ─────────────────────────────────────────────────


def test_every_check_has_a_plain_sentence():
    for check in CHECKS:
        text = plain({"check": check, "value": 3, "path": "x", "return_period": 100})
        assert text and not text.startswith(check), f"{check} has no sentence"
        assert "—" not in text and "–" not in text


@pytest.mark.parametrize("gate, words", [
    ({"check": "min_years", "value": 20}, "needs at least 20 years of record"),
    ({"check": "max_return_period_factor", "value": 3, "return_period": 100},
     "the 100-year estimate may not exceed 3x the record length"),
    ({"check": "spread_within", "value": 0.25, "return_period": 100},
     "the fitted distributions must agree within 25% at the 100-year level"),
    ({"check": "fit_envelopes_max", "value": 0.25},
     "the fitted curve must reach the largest flood on record (within 25%)"),
    ({"check": "sampling_density", "value": "daily"}, "needs a record sampled about daily"),
    ({"check": "trend_on_series", "path": "ffa.amax_trend"},
     "the flood peaks must show no significant trend (at the 5% level)"),
    ({"check": "not_empty", "path": "current.spi"}, "must return an SPI value"),
    ({"check": "max_area_km2", "value": 10000}, "the catchment may be at most 10,000 km2 for a lumped model"),
    ({"check": "made_up", "value": 2}, "made_up 2"),
])
def test_plain_sentences(gate, words):
    assert plain(gate) == words


def test_plain_plan_lists_checks_and_controls_with_current_values():
    p = plain_plan(_study())
    s2 = next(s for s in p["steps"] if s["id"] == "s2")
    assert s2["checks"] == ["needs at least 20 years of record",
                            "the 100-year estimate may not exceed 3x the record length"]
    assert s2["gates"] == _study().step_by_id("s2").expects
    controls = {c["param"]: c for c in s2["controls"]}
    assert controls["return_period"]["value"] == 100 and controls["years"]["value"] is None
    assert controls["bootstrap_ci"]["value"] is False
    assert next(s for s in p["steps"] if s["id"] == "s1")["controls"] == []
    assert plain_plan(None) == {} and json.dumps(p)


# ── the Studio ──────────────────────────────────────────────────────────────


def test_studio_steer_reruns_one_step_and_rewrites_the_report(studio_factory):
    calls: list = []
    s, _ = studio_factory(tools=fake_tools(calls))
    s.say(PROBLEM)
    early = s.steer("s3", {"return_period": 50})
    assert early.kind == "answer" and "once the report is out" in early.text
    s.approve()
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency", "anywhere"]
    calls.clear()
    r = s.steer("s3", {"return_period": 50})
    assert r.kind == "report" and s.workspace.status == "done"
    assert [c[0] for c in calls] == ["flood_frequency", "anywhere"], "only s3 and its dependant ran"
    study = s.workspace.study
    assert study["plan"]["steering"][-1]["changes"] == {"return_period": {"from": 100, "to": 50}}
    assert study["problem"]["params"]["return_period"] == 50
    assert s.workspace.follow_ups[-1]["kind"] == "steer" and s.workspace.follow_ups[-1]["steps"] == ["s3", "s4"]
    assert any("50-year" in k["label"] for k in s.workspace.report["key_numbers"])
    assert "T = 50" in next(g["detail"] for g in s.workspace.run["gates"] if g["check"] == "max_return_period_factor")
    assert "steering:" in s.workspace.study_obj().to_yaml()


def test_studio_steer_refuses_a_bad_change_and_keeps_the_study(studio_factory):
    s, calls = studio_factory()
    s.say(PROBLEM)
    s.approve()
    n, study = len(calls), json.dumps(s.workspace.study, sort_keys=True)
    r = s.steer("s3", {"return_period": 7})
    assert r.kind == "answer" and "not accepted" in r.text and r.payload["errors"]
    assert len(calls) == n and json.dumps(s.workspace.study, sort_keys=True) == study
    assert s.workspace.messages[-1].text == r.text
