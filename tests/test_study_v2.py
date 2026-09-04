"""Study version 2: the plan and the receipt as one file, run with gates and fallbacks; version 1 still works."""

from __future__ import annotations

import json
from unittest.mock import patch

from aquascope.study import Step, Study, loads, parse_block_yaml, run_study, write_outputs

V1_TEXT = """# An AquaScope study: the steps behind an answer, so it can be run again.
title: "Sources"
question: "What sources are there?"
created: "2026-08-01T00:00:00+00:00"
aquascope_version: "0.11.0"
author: "hand"
steps:
  - tool: "list_sources"
    arguments: {}
  - tool: "probe"
    note: "a fake"
    arguments:
      years: 12
"""


def _fake_tools(calls: list | None = None):
    calls = calls if calls is not None else []

    def probe(**kw):
        calls.append(("probe", kw))
        return {"years": kw.get("years", 30), "unit": "m3/s", "points": [["2020-01-01", 1.0], ["2020-02-01", 2.0]],
                "methods": [{"name": "Probe", "text": "t", "citation": "c"}]}

    def donors(**kw):
        calls.append(("donors", kw))
        return {"k": kw.get("k", 1), "stations": [{"source": "x", "station_id": "1"}] * kw.get("k", 1)}

    def boom(**kw):
        calls.append(("boom", kw))
        raise RuntimeError("no service")

    def table(df=None, **kw):
        calls.append(("table", {"rows": int(len(df)), **kw}))
        return {"n": int(len(df)), "columns": list(df.columns)}

    return {"probe": probe, "donors": donors, "boom": boom, "table": table, "list_sources": lambda: {"sources": [1]}}


def test_a_version_1_study_loads_and_runs_as_before():
    study = loads(V1_TEXT)
    assert study.version == 1 and not study.is_v2 and [s.tool for s in study.steps] == ["list_sources", "probe"]
    assert study.steps[1].note == "a fake" and study.steps[1].id is None
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study)
    assert run.ok and [r["ok"] for r in run.results] == [True, True] and run.results[1]["id"] == "s2"
    assert study.results == {}, "a version-1 study is not rewritten into a plan"
    assert "\nversion: 2" not in study.to_yaml() and loads(study.to_yaml()).steps[1].arguments == {"years": 12}
    assert Study.from_dict(parse_block_yaml(V1_TEXT)).to_dict() == study.to_dict()


def _plan() -> Study:
    return Study(
        question="How big is the 100-year flood?", version=2,
        problem={"kind": "flood_risk", "site": {"lat": 1.0, "lon": 2.0}, "params": {"return_period": 100}},
        plan={"playbook": "flood_risk", "branch": "at_site", "rationale": "a: reason", "caveats": ["say so #1"]},
        steps=[
            Step(tool="probe", id="s1", rationale="fetch", arguments={"years": 40},
                 expects=[{"check": "min_years", "value": 20, "path": "years"}, {"check": "unit_present"}]),
            Step(tool="donors", id="s2", arguments={"k": 5}, depends_on=["s1"],
                 expects=[{"check": "min_donors", "value": 3, "path": "k"}]),
        ],
    )


def test_a_version_2_study_round_trips_through_yaml_with_and_without_pyyaml():
    study = _plan()
    text = study.to_yaml()
    assert text.splitlines()[2] == "version: 2" and "expects:" in text and "depends_on:" in text
    via_pyyaml = loads(text)
    via_subset = Study.from_dict(parse_block_yaml(text))
    assert via_pyyaml.to_dict() == via_subset.to_dict()
    assert via_subset.steps[0].expects == study.steps[0].expects
    assert via_subset.steps[1].depends_on == ["s1"] and via_subset.plan["caveats"] == ["say so #1"]
    assert via_subset.problem["site"] == {"lat": 1.0, "lon": 2.0}


def test_gates_are_evaluated_and_written_into_the_study():
    study = _plan()
    events: list[dict] = []
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study, on_event=events.append)
    assert run.ok and run.stop_reason is None
    assert [(g["step"], g["check"], g["passed"]) for g in run.gates] == [
        ("s1", "min_years", True), ("s1", "unit_present", True), ("s2", "min_donors", True)]
    assert study.results["s1"]["ok"] and study.results["s1"]["gates"][0]["passed"]
    assert study.results["s2"]["fallback_used"] is False and study.results["s1"]["sha256"]
    roles = {(e["role"], e["event"]) for e in events}
    assert {("runner", "start"), ("runner", "done"), ("reviewer", "gate")} <= roles
    assert all(set(e) >= {"role", "step", "event", "detail"} for e in events)
    md = run.to_markdown()
    assert "gate min_years: passed" in md and "## Caveats" in md and "say so #1" in md
    assert "results:" in study.to_yaml() and loads(study.to_yaml()).results["s2"]["ok"]


def test_a_failed_gate_runs_the_fallback_once_and_records_it():
    calls: list = []
    study = _plan()
    study.steps[1].arguments = {"k": 1}
    study.steps[1].fallback = {"step": {"tool": "donors", "arguments": {"k": 4},
                                        "expects": [{"check": "min_donors", "value": 3, "path": "k"}]}}
    with patch("aquascope.study._tools", return_value=_fake_tools(calls)):
        run = run_study(study)
    assert run.ok and run.stop_reason is None
    r = run.results[1]
    assert r["fallback_used"] and r["fallback"]["ok"] and r["fallback"]["gates"][0]["passed"]
    assert [c for c in calls if c[0] == "donors"] == [("donors", {"k": 1}), ("donors", {"k": 4})]
    assert study.results["s2"]["fallback_used"] and study.results["s2"]["fallback"]["tool"] == "donors"
    assert "fallback `donors(k=4)`: ok" in run.to_markdown()


def test_a_fallback_that_fails_its_own_gate_stops_the_study():
    study = _plan()
    study.steps[1].arguments = {"k": 1}
    study.steps[1].fallback = {"step": {"tool": "donors", "arguments": {"k": 2},
                                        "expects": [{"check": "min_donors", "value": 3, "path": "k"}]}}
    study.steps.append(Step(tool="probe", id="s3"))
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study)
    assert not run.ok and run.stopped_at == "s2" and "did not pass its own gates" in run.stop_reason
    assert [r["id"] for r in run.results] == ["s1", "s2"], "nothing after the stop runs"


def test_a_failed_gate_without_a_fallback_stops_with_the_reason():
    study = _plan()
    study.steps[1].arguments = {"k": 1}
    study.steps.append(Step(tool="probe", id="s3"))
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study)
    assert run.stopped_at == "s2" and "min_donors" in run.stop_reason and "1 donors, 3 needed" in run.stop_reason
    assert len(run.results) == 2 and run.replan is None
    assert "**Stopped at s2:**" in run.to_markdown()


def test_a_branch_fallback_is_handed_up_as_a_replan_request():
    study = _plan()
    study.steps[1].arguments = {"k": 1}
    study.steps[1].fallback = {"branch": "regional"}
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study)
    assert run.replan == {"step": "s2", "branch": "regional", "reason": run.replan["reason"]}
    assert "regional" in run.stop_reason


def test_a_step_whose_dependency_failed_is_skipped_not_run():
    calls: list = []
    study = Study(question="q", version=2, steps=[
        Step(tool="boom", id="a"), Step(tool="probe", id="b", depends_on=["a"]), Step(tool="probe", id="c")])
    with patch("aquascope.study._tools", return_value=_fake_tools(calls)):
        run = run_study(study)
    assert [r["ok"] for r in run.results] == [False, False, True] and run.results[1]["skipped"]
    assert "depends on a" in run.results[1]["error"] and [c[0] for c in calls] == ["boom", "probe"]
    assert not run.ok


def test_from_step_hands_a_previous_series_to_a_table_analysis():
    calls: list = []
    study = Study(question="q", version=2, steps=[
        Step(tool="probe", id="s1"), Step(tool="table", id="s2", arguments={"from_step": "s1", "alpha": 0.9})])
    with patch("aquascope.study._tools", return_value=_fake_tools(calls)):
        run = run_study(study)
    assert run.ok and run.results[1]["result"] == {"n": 2, "columns": ["datetime", "value"]}
    assert calls[1] == ("table", {"rows": 2, "alpha": 0.9})


def test_a_prior_run_is_reused_for_steps_that_passed():
    calls: list = []
    study = _plan()
    with patch("aquascope.study._tools", return_value=_fake_tools(calls)):
        first = run_study(study)
        study.steps[1].arguments = {"k": 6}
        events: list = []
        second = run_study(study, on_event=events.append, prior=first)
    assert second.ok and [c for c in calls if c[0] == "probe"] == [("probe", {"years": 40})]
    assert any(e["event"] == "reused" and e["step"] == "s1" for e in events)


def test_an_unknown_tool_fails_the_step_and_the_run_goes_on():
    study = Study(question="q", version=2, steps=[Step(tool="teleport", id="t"), Step(tool="list_sources", id="l")])
    run = run_study(study)
    assert [r["ok"] for r in run.results] == [False, True] and "unknown tool" in run.results[0]["error"]


def test_write_outputs_adds_the_study_file_for_a_plan(tmp_path):
    study = _plan()
    with patch("aquascope.study._tools", return_value=_fake_tools()):
        run = run_study(study)
    paths = write_outputs(run, tmp_path / "out")
    assert set(paths) == {"report.md", "manifest.json", "results.json", "study.yaml"}
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["steps"][0]["gates"][0]["check"] == "min_years" and manifest["stop_reason"] is None
    assert loads((tmp_path / "out" / "study.yaml").read_text()).results["s1"]["ok"]


def test_the_subset_parser_reads_block_scalars_flow_collections_and_comments():
    text = """# a comment
a: 1
b: [x, "y", 3]
c: {k: v, n: 2}
d: >-
  folded text
  on two lines
e: |
  literal
  lines
f:
  - g: 1
    h: [1, 2]
  - >-
    item text
  - plain item
i: 'single ''quoted'''
"""
    got = parse_block_yaml(text)
    assert got["a"] == 1 and got["b"] == ["x", "y", 3] and got["c"] == {"k": "v", "n": 2}
    assert got["d"] == "folded text on two lines" and got["e"] == "literal\nlines\n"
    assert got["f"] == [{"g": 1, "h": [1, 2]}, "item text", "plain item"] and got["i"] == "single 'quoted'"
