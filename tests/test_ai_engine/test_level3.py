"""The Analyst's level-3 parts: the sandbox, the checks and the study artifact.

These are the pieces that turn "a model answered" into "here is what ran, here
is what it does not establish, and here is how to run it again".
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aquascope.ai_engine import verify as verify_mod
from aquascope.ai_engine.sandbox import SandboxError, run_python
from aquascope.study import Step, Study, loads, run_study, study_from_calls, write_outputs

# ── the sandbox ─────────────────────────────────────────────────────────────


def test_a_snippet_gets_aquascope_pandas_and_numpy_without_importing() -> None:
    res = run_python("result = float(np.mean([1, 2, 3])) + len(pd.Series([1, 2]))")
    assert res.ok and res.result == pytest.approx(4.0)


def test_the_data_the_caller_passes_is_in_scope() -> None:
    df = pd.DataFrame({"q": [1.0, 2.0, 3.0, 4.0]})
    res = run_python("result = float(df['q'].max())", data={"df": df})
    assert res.ok and res.result == 4.0


def test_a_dataframe_result_comes_back_as_a_table() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res = run_python("result = df.head(2)", data={"df": df})
    assert res.ok
    assert res.result["type"] == "table"
    assert res.result["columns"] == ["a", "b"]
    assert res.result["n_rows"] == 2


def test_a_series_result_carries_its_index() -> None:
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]), name="q")
    res = run_python("result = s", data={"s": s})
    assert res.result["type"] == "series" and res.result["name"] == "q"
    assert res.result["index"][0].startswith("2020-01-01")


def test_printed_output_comes_back() -> None:
    res = run_python("print('hello from the snippet')\nresult = 1")
    assert "hello from the snippet" in res.stdout


def test_a_failing_snippet_returns_the_error_rather_than_raising() -> None:
    res = run_python("result = 1 / 0")
    assert not res.ok
    assert "ZeroDivisionError" in res.error


def test_the_result_is_json_serialisable() -> None:
    df = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=3), "q": [1.0, float("nan"), 3.0]})
    res = run_python("result = df", data={"df": df})
    json.dumps(res.to_dict(), allow_nan=False)     # NaN and Timestamps must already be handled


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from os import path",
    "import socket",
    "__import__('os')",
    "eval('1+1')",
    "exec('x = 1')",
    "open('/etc/passwd')",
    "result = (1).__class__.__bases__",
])
def test_the_obvious_ways_out_are_refused_before_anything_runs(code: str) -> None:
    with pytest.raises(SandboxError):
        run_python(code)


def test_a_snippet_may_use_the_scientific_stack_it_needs() -> None:
    res = run_python("import math\nfrom scipy import stats\nresult = round(math.sqrt(16) + stats.norm.cdf(0), 2)")
    assert res.ok and res.result == pytest.approx(4.5)


def test_syntax_errors_are_reported_as_a_refusal_not_a_crash() -> None:
    with pytest.raises(SandboxError, match="SyntaxError"):
        run_python("result = (")


def test_a_snippet_can_call_the_workbench() -> None:
    idx = pd.date_range("2015-01-01", periods=800, freq="D")
    df = pd.DataFrame({"date": idx, "discharge": [3.0 + (i % 30) / 10 for i in range(len(idx))]})
    res = run_python("result = workbench.flow_duration(df)['percentiles']['50']", data={"df": df})
    assert res.ok and isinstance(res.result, float)


# ── the checks ──────────────────────────────────────────────────────────────


def _result(name: str, payload: dict, ok: bool = True) -> dict:
    return {"name": name, "arguments": {}, "payload": payload, "ok": ok}


def test_an_answer_with_no_successful_tool_call_is_flagged() -> None:
    v = verify_mod.verify("The 100-year flood is about 500 m3/s.", [_result("analyze_station", {}, ok=False)])
    assert not v.ok
    assert any(c.name == "tools_were_used" for c in v.failed)


def test_a_number_that_is_not_in_any_tool_result_is_flagged() -> None:
    payload = {"stats": {"mean": 3.4}, "unit": "m3/s", "station_id": "USGS-1"}
    answer = "The mean is 3.4 m3/s at USGS-1, and the peak was 987.6 m3/s."
    v = verify_mod.verify(answer, [_result("analyze_station", payload)])
    failed = {c.name for c in v.failed}
    assert "numbers_come_from_tools" in failed


def test_numbers_that_match_after_rounding_are_accepted() -> None:
    payload = {"stats": {"mean": 3.4217}, "unit": "m3/s", "station_id": "USGS-1"}
    v = verify_mod.verify("The mean flow at USGS-1 is 3.42 m3/s.", [_result("analyze_station", payload)])
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_return_level_without_an_interval_is_flagged() -> None:
    payload = {"ffa": {"return_periods": [100], "fits": {"lp3": {"q": [500.0], "ci": [[440.0, 570.0]]}}},
               "unit": "m3/s", "station_id": "USGS-1"}
    bare = verify_mod.verify("The 100-year flood at USGS-1 is 500 m3/s.", [_result("flood_frequency", payload)])
    assert "flood_estimate_carries_uncertainty" in {c.name for c in bare.failed}

    with_ci = verify_mod.verify(
        "The 100-year flood at USGS-1 is 500 m3/s (90 % confidence interval 440 to 570).",
        [_result("flood_frequency", payload)],
    )
    assert "flood_estimate_carries_uncertainty" not in {c.name for c in with_ci.failed}


def test_a_significance_claim_that_contradicts_the_test_is_flagged() -> None:
    payload = {"trend": {"trend": "increasing", "p_value": 0.42, "n_years": 30}, "unit": "m3/s", "station_id": "S"}
    v = verify_mod.verify("Flows at S show a significant increasing trend.", [_result("analyze_station", payload)])
    assert "trend_matches_the_test" in {c.name for c in v.failed}


def test_a_significance_claim_that_agrees_passes() -> None:
    payload = {"trend": {"trend": "increasing", "p_value": 0.01, "n_years": 30}, "unit": "m3/s", "station_id": "S"}
    answer = "Flows at S show a significant increasing trend of 0.01."
    v = verify_mod.verify(answer, [_result("analyze_station", payload)])
    assert "trend_matches_the_test" not in {c.name for c in v.failed}


def test_an_answer_that_never_names_its_record_is_flagged() -> None:
    payload = {"station_id": "USGS-01013500", "name": "Fish River near Fort Kent", "unit": "m3/s",
               "stats": {"mean": 43.4}}
    v = verify_mod.verify("The mean flow is 43.4 m3/s.", [_result("analyze_station", payload)])
    assert "record_is_named" in {c.name for c in v.failed}


def test_the_unmet_checks_are_written_out_for_the_reader() -> None:
    v = verify_mod.verify("The flood is 999.9.", [_result("analyze_station", {"stats": {"mean": 1.0}, "unit": "m3/s"})])
    md = v.to_markdown()
    assert "does not establish" in md and "999.9" in md


# ── the study artifact ──────────────────────────────────────────────────────


def test_a_study_round_trips_through_its_own_yaml() -> None:
    study = Study(question="How big is the 100-year flood at Kingston?", steps=[
        Step(tool="find_stations", arguments={"query": "Kingston", "limit": 3}, note="find it first"),
        Step(tool="analyze_station", arguments={"source": "uk_ea", "station_id": "abc", "years": 40}),
    ])
    back = loads(study.to_yaml())
    assert back.question == study.question
    assert [s.tool for s in back.steps] == ["find_stations", "analyze_station"]
    assert back.steps[0].arguments == {"query": "Kingston", "limit": 3}
    assert back.steps[1].arguments["years"] == 40
    assert back.steps[0].note == "find it first"


def test_a_study_quotes_awkward_values_safely() -> None:
    study = Study(question='A question with: a colon, a #hash and "quotes"', steps=[
        Step(tool="find_stations", arguments={"query": "Le Rhône à Anthon: gauge #3"}),
    ])
    back = loads(study.to_yaml())
    assert back.question == study.question
    assert back.steps[0].arguments["query"] == "Le Rhône à Anthon: gauge #3"


def test_running_a_study_records_a_hash_per_step() -> None:
    run = run_study(Study(question="What sources are there?", steps=[Step(tool="list_sources")]))
    assert run.ok
    assert len(run.results) == 1
    assert run.results[0]["sha256"] and len(run.results[0]["sha256"]) == 16


def test_running_the_same_study_twice_gives_the_same_hashes() -> None:
    """A re-run that drifts should be visible, which is the point of the manifest."""
    study = Study(question="What sources are there?", steps=[Step(tool="list_sources")])
    first, second = run_study(study), run_study(study)
    assert first.results[0]["sha256"] == second.results[0]["sha256"]


def test_an_unknown_tool_fails_the_step_not_the_run() -> None:
    run = run_study(Study(question="?", steps=[Step(tool="list_sources"), Step(tool="teleport")]))
    assert not run.ok
    assert run.results[0]["ok"] and not run.results[1]["ok"]
    assert "unknown tool" in run.results[1]["error"]


def test_write_outputs_writes_a_report_and_a_manifest(tmp_path) -> None:
    run = run_study(Study(question="What sources are there?", steps=[Step(tool="list_sources")]))
    paths = write_outputs(run, tmp_path / "study-out")
    report = (tmp_path / "study-out" / "report.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "study-out" / "manifest.json").read_text(encoding="utf-8"))
    assert "list_sources" in report
    assert "No model was involved in this run" in report
    assert manifest["ok"] and manifest["steps"][0]["tool"] == "list_sources"
    assert set(paths) == {"report.md", "manifest.json", "results.json"}


def test_a_study_is_built_from_what_the_model_actually_ran() -> None:
    class Call:
        def __init__(self, name, arguments, ok=True):
            self.name, self.arguments, self.ok = name, arguments, ok

    calls = [
        Call("find_stations", {"query": "Thames"}),
        Call("describe_methods", {}),                 # housekeeping, not a step
        Call("analyze_station", {"source": "uk_ea", "station_id": "x"}),
        Call("analyze_station", {"source": "uk_ea", "station_id": "broken"}, ok=False),  # failed, not a step
    ]
    study = study_from_calls("Q?", calls, model="test-model")
    assert [s.tool for s in study.steps] == ["find_stations", "analyze_station"]
    assert study.model == "test-model" and study.author == "analyst"
