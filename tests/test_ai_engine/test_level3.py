"""The Analyst's level-3 parts: the sandbox, the checks and the study artifact.

These are the pieces that turn "a model answered" into "here is what ran, here
is what it does not establish, and here is how to run it again".
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from aquascope.ai_engine import analyst as analyst_mod
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


# ── the checks against real prose (#233) ────────────────────────────────────
#
# The first live recording scored 2/5, 1/5 and 0/1 on answers that were correct.
# Every one of those failures was the check's fault: a good model writes good
# typography, and the checks compared it against plain ASCII. A check that cries
# wolf on a correct answer is worse than no check, because it is printed to the
# reader as "what this answer does not establish".

KINGSTON_PAYLOAD = {
    "unit": "m3/s", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston",
    "agency": "Environment Agency", "years": 40.0, "n": 14555, "start": "1986-08-21", "end": "2026-08-19",
    "ffa": {"return_periods": [100], "fits": {"gev": {"q": [497.0], "ci": [[453.0, 525.0]]}}},
}

# Copied from explorer/showcase/kingston-flood.json: narrow no-break spaces,
# non-breaking hyphens, superscript unit, grouped digits, percentages.
KINGSTON_ANSWER = (
    "**100‑year flood at Kingston (River Thames)**\n\n"
    "Environment Agency (UK) | **Kingston** (station ID "
    "8496ce69‑482c‑406a‑a2f0‑ac418ef8f099) | "
    "1986‑08‑21 to 2026‑08‑19 (≈ 40 yr, 14 555 daily values) | "
    "**≈ 497 m³ s⁻¹** | **453 – 525 m³ s⁻¹** (90 % CI)\n\n"
    "The GEV‑bootstrap interval (≈ ± 7 % of the median) quantifies the sampling uncertainty."
)


def _one(payload: dict) -> list[dict]:
    return [{"name": "flood_frequency", "arguments": {}, "payload": payload, "ok": True}]


def test_a_well_typeset_answer_passes_every_check() -> None:
    v = verify_mod.verify(KINGSTON_ANSWER, _one(KINGSTON_PAYLOAD))
    assert v.ok, f"false failures on a correct answer: {[(c.name, c.detail) for c in v.failed]}"


def test_a_superscript_unit_counts_as_naming_the_unit() -> None:
    """m³ s⁻¹ is the unit. Reading it as 'no unit named' was the check's fault."""
    v = verify_mod.verify("Flow at Kingston is 497 m³ s⁻¹.", _one(KINGSTON_PAYLOAD))
    assert "units_are_named" not in {c.name for c in v.failed}


def test_a_unit_is_not_read_as_a_claimed_number() -> None:
    """'m3 s-1' contains 3 and -1; flagging those would discredit the check."""
    v = verify_mod.verify("Flow at Kingston is 497 m3 s-1.", _one(KINGSTON_PAYLOAD))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_non_breaking_hyphen_in_an_id_still_names_the_record() -> None:
    v = verify_mod.verify(
        "Station 8496ce69‑482c‑406a‑a2f0‑ac418ef8f099 gives 497 m3/s.",
        _one(KINGSTON_PAYLOAD),
    )
    assert "record_is_named" not in {c.name for c in v.failed}


def test_grouped_digits_are_one_number() -> None:
    v = verify_mod.verify("The record has 14 555 daily values, in m3/s at Kingston.",
                          _one(KINGSTON_PAYLOAD))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_date_is_not_a_claim() -> None:
    v = verify_mod.verify("Kingston's record runs 1986-08-21 to 2026-08-19, in m3/s.",
                          _one(KINGSTON_PAYLOAD))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_derived_percentage_is_not_a_fabricated_number() -> None:
    """"about 7 % of the median" is arithmetic over numbers that are in the result."""
    v = verify_mod.verify("At Kingston the interval is about 7 % of the 497 m3/s median.",
                          _one(KINGSTON_PAYLOAD))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_fabricated_number_is_still_caught_after_all_that_folding() -> None:
    """The point of the loosening is fewer false alarms, not a check that passes everything."""
    v = verify_mod.verify("At Kingston the 100-year flood is 497 m3/s (453 to 525), "
                          "and the 1908 peak reached 1234.5 m3/s.", _one(KINGSTON_PAYLOAD))
    assert "numbers_come_from_tools" in {c.name for c in v.failed}
    assert "1234.5" in next(c.detail for c in v.failed if c.name == "numbers_come_from_tools")


# ── fitting the context (#233) ──────────────────────────────────────────────
#
# Half the showcase questions failed with "Request too large ... Limit 8000,
# Requested 12073". Not a rate limit: the accumulated tool results were bigger
# than the whole per-minute window, so retrying could never help.

def _conversation(n_tools: int, size: int) -> list[dict]:
    msgs = [{"role": "system", "content": "you are the analyst"},
            {"role": "user", "content": "what is the 100-year flood?"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "analyze_station",
                     "content": f"result {i} " + "x" * size})
    return msgs


def test_a_conversation_that_fits_is_left_alone() -> None:
    msgs = _conversation(2, 100)
    assert analyst_mod.fit_context(msgs) == msgs


def test_the_oldest_results_are_cut_first_and_the_newest_survives() -> None:
    msgs = _conversation(4, 12_000)
    fitted = analyst_mod.fit_context(msgs, budget=20_000)
    tools = [m for m in fitted if m["role"] == "tool"]
    assert len(tools[0]["content"]) < 600, "the oldest result should be trimmed"
    assert tools[-1]["content"].startswith("result 3 "), "the newest result is what the answer needs"
    assert "trimmed" in tools[0]["content"], "it should say what was removed"


def test_it_does_not_touch_the_question_or_the_system_prompt() -> None:
    msgs = _conversation(3, 20_000)
    fitted = analyst_mod.fit_context(msgs, budget=5_000)
    assert fitted[0]["content"] == "you are the analyst"
    assert fitted[1]["content"] == "what is the 100-year flood?"


def test_one_huge_result_is_cut_down_too() -> None:
    """With a single tool call there is no older message to sacrifice."""
    msgs = _conversation(1, 60_000)
    fitted = analyst_mod.fit_context(msgs, budget=10_000)
    assert sum(len(str(m.get("content") or "")) for m in fitted) <= 10_000


def test_the_original_messages_are_not_mutated() -> None:
    msgs = _conversation(3, 12_000)
    before = [str(m.get("content")) for m in msgs]
    analyst_mod.fit_context(msgs, budget=5_000)
    assert [str(m.get("content")) for m in msgs] == before


def test_the_size_count_includes_what_the_model_wrote_not_just_the_text() -> None:
    """A run_python call carries its snippet in tool_calls, which used to be free."""
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c0", "type": "function",
         "function": {"name": "run_python", "arguments": '{"code": "' + "x" * 5000 + '"}'}}]}]
    assert analyst_mod._conversation_size(msgs) > 5000, (
        "counting only 'content' reads this conversation as empty, which is how a request "
        "budgeted at 6,000 tokens arrived as 9,300"
    )


def test_a_request_too_large_shrinks_the_context_and_tries_again() -> None:
    """413 means "this cannot fit my window", so the same request will never work.

    Three of the showcase questions died here: two records to compare, several
    tool results, and the accumulated conversation was bigger than the whole
    per-minute window. Retrying it unchanged was pointless; shrinking is the move.
    """
    from types import SimpleNamespace

    from aquascope.ai_engine.llm_transport import LLMHTTPError

    sizes: list[int] = []

    class TooLargeUntilSmall:
        """Asks for a tool, then refuses the follow-up until the result is trimmed."""

        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
            self.turn = 0

        def _create(self, **kwargs):
            size = analyst_mod._conversation_size(kwargs["messages"])
            sizes.append(size)
            self.turn += 1
            if self.turn == 1:
                call = SimpleNamespace(id="c0", function=SimpleNamespace(name="list_sources", arguments="{}"))
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[call]))])
            if size > 8_000:
                raise LLMHTTPError(413, '{"error":{"message":"Request too large"}}', "https://x/v1")
            msg = SimpleNamespace(content="There are sources.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    big = {"sources": [{"id": f"s{i}", "notes": "y" * 400} for i in range(60)]}
    with patch("aquascope.mcp_server.list_sources", return_value=big):
        result = analyst_mod.ask("What sources are there?", client=TooLargeUntilSmall(),
                                 model="test", max_steps=3)
    assert result.answer == "There are sources.", "it should answer once the context fits"
    assert len(sizes) >= 3, f"one call, one refusal, one retry: {sizes}"
    assert sizes[-1] < sizes[-2], f"the retry has to be smaller than what was refused: {sizes}"
    assert sizes[-1] < 8_000


def test_an_unparseable_tool_call_is_sent_back_to_the_model() -> None:
    """Groq rejects the whole request when the model's tool call is not valid JSON.

    It happened on a showcase question twice running: the model wrote raw Python
    where the JSON arguments object belongs. The model wrote it, so the model can
    write it again, rather than the question being lost.
    """
    from types import SimpleNamespace

    from aquascope.ai_engine.llm_transport import LLMHTTPError

    bad = ('{"error":{"message":"Failed to parse tool call arguments as JSON",'
           '"type":"invalid_request_error","code":"tool_use_failed"}}')
    seen: list[list[dict]] = []

    class MalformedOnce:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
            self.turn = 0

        def _create(self, **kwargs):
            seen.append(kwargs["messages"])
            self.turn += 1
            if self.turn == 1:
                raise LLMHTTPError(400, bad, "https://x/v1")
            msg = SimpleNamespace(content="Taipei is wetter than London.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    result = analyst_mod.ask("How wet is Taipei?", client=MalformedOnce(), model="test", max_steps=2)
    assert result.answer == "Taipei is wetter than London."
    assert len(seen) == 2, "it should have asked again"
    assert "single JSON object" in str(seen[1][-1]["content"]), "and said what was wrong"


def test_it_does_not_argue_with_the_model_forever() -> None:
    """Two nudges, then the failure surfaces: a model that cannot format a call will not learn."""
    from types import SimpleNamespace

    from aquascope.ai_engine.llm_transport import LLMHTTPError

    bad = '{"error":{"code":"tool_use_failed"}}'
    calls = {"n": 0}

    class AlwaysMalformed:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            calls["n"] += 1
            raise LLMHTTPError(400, bad, "https://x/v1")

    with pytest.raises(LLMHTTPError):
        analyst_mod.ask("How wet is Taipei?", client=AlwaysMalformed(), model="test", max_steps=2)
    assert calls["n"] == 3, f"the first try and two nudges, then stop: {calls['n']}"


# ── the second pass over the checks (#233) ──────────────────────────────────
#
# Reading all eight recorded traces showed three more ways a correct answer was
# marked down. Each is a limit of the check, not a defect in the answer.

def test_a_station_named_only_by_its_search_result_still_counts() -> None:
    """flood_frequency returns the id and no name; the search that found it has the name.

    So an answer saying "Kingston" was reported as not naming its record, because
    the only identifier the check looked at was a UUID the prose sensibly omits.
    """
    results = [
        {"name": "find_stations", "arguments": {"query": "Kingston"}, "ok": True,
         "payload": {"stations": [{"source": "uk_ea", "station_id": "8496ce69", "name": "Kingston"}]}},
        {"name": "flood_frequency", "arguments": {}, "ok": True,
         "payload": {"station_id": "8496ce69-482c-406a", "unit": "m3/s",
                     "ffa": {"return_periods": [100], "fits": {"gev": {"q": [497.0], "ci": [[453.0, 525.0]]}}}}},
    ]
    v = verify_mod.verify("The 100-year flood at Kingston is 497 m3/s (453 to 525).", results)
    assert "record_is_named" not in {c.name for c in v.failed}


def test_a_unit_with_an_exponent_is_not_a_claim_of_minus_one() -> None:
    """"mm yr-1" appeared five times in one answer and was read as five claims of -1."""
    payload = {"climate": {"precip_mm_yr": 1180.0, "pet_mm_yr": 900.0}, "station_id": "P1"}
    v = verify_mod.verify("At P1 rainfall is 1180 mm yr-1 and PET is 900 mm yr-1.",
                          [{"name": "anywhere", "arguments": {}, "payload": payload, "ok": True}])
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_coordinate_the_tool_accepted_is_not_invented() -> None:
    """The answer repeats back the point it was asked about, which is in the arguments."""
    results = [{"name": "describe_catchment", "arguments": {"lat": 47.0, "lon": -68.6},
                "payload": {"area_km2": 1420.0, "name": "Upper St John"}, "ok": True}]
    v = verify_mod.verify("The catchment upstream of 47.0 N, 68.6 W (Upper St John) covers 1420 km2.", results)
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_significance_threshold_is_a_convention_not_a_measurement() -> None:
    payload = {"trend": {"trend": "decreasing", "p_value": 0.0016, "n_years": 40}, "station_id": "S", "unit": "m3/s"}
    payload["stats"] = {"mean": 43.4}
    v = verify_mod.verify("At S the decreasing trend is significant (p = 0.0016 < 0.05), 43.4 m3/s.",
                          [{"name": "analyze_station", "arguments": {}, "payload": payload, "ok": True}])
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_claim_about_a_series_the_test_never_ran_on_is_not_a_contradiction() -> None:
    """The tool tests annual means. "No significant trend in low flow" is about something else.

    Marking that wrong taught the reader to distrust the checks, which is the
    opposite of the point.
    """
    payload = {"trend": {"trend": "increasing", "p_value": 0.0016, "n_years": 40},
               "station_id": "Seine", "unit": "m3/s"}
    answer = ("No significant trend in low flow is reported for either gauge at Seine "
              "(the annual mean does show a modest increasing trend).")
    v = verify_mod.verify(answer, [{"name": "analyze_station", "arguments": {}, "payload": payload, "ok": True}])
    assert "trend_matches_the_test" not in {c.name for c in v.failed}


def test_a_blanket_denial_that_contradicts_the_test_is_still_caught() -> None:
    payload = {"trend": {"trend": "increasing", "p_value": 0.0016, "n_years": 40},
               "station_id": "Seine", "unit": "m3/s"}
    v = verify_mod.verify("At Seine there is no significant trend.",
                          [{"name": "analyze_station", "arguments": {}, "payload": payload, "ok": True}])
    assert "trend_matches_the_test" in {c.name for c in v.failed}


def test_a_coordinate_in_the_wrong_hemisphere_is_caught() -> None:
    """Reading the sign off the compass letter is stricter than the original check.

    Before this the longitude was never verified at all; accepting "either sign"
    would have let a hemisphere error through, which in hydrology is a real error.
    """
    results = [{"name": "describe_catchment", "arguments": {"lat": 47.0, "lon": -68.6},
                "payload": {"area_km2": 1420.0, "name": "Upper St John", "unit": "km2"}, "ok": True}]
    good = verify_mod.verify("Upstream of 47.0 N, 68.6 W (Upper St John) covers 1420 km2.", results)
    bad = verify_mod.verify("Upstream of 47.0 N, 68.6 E (Upper St John) covers 1420 km2.", results)
    assert "numbers_come_from_tools" not in {c.name for c in good.failed}
    assert "numbers_come_from_tools" in {c.name for c in bad.failed}


def test_a_squared_unit_is_not_a_claim_of_two() -> None:
    results = [{"name": "describe_catchment", "arguments": {},
                "payload": {"area_km2": 1420.0, "name": "Upper St John"}, "ok": True}]
    v = verify_mod.verify("Upper St John covers 1420 km2.", results)
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}


def test_a_counted_thing_is_not_mistaken_for_a_unit() -> None:
    """"gauge 3" must keep its 3: only closed-up unit spellings are stripped."""
    results = [{"name": "find_stations", "arguments": {},
                "payload": {"stations": [{"name": "A"}], "n_returned": 1}, "ok": True}]
    v = verify_mod.verify("Station A is gauge 3 of the set.", results)
    assert "numbers_come_from_tools" in {c.name for c in v.failed}, "3 is a claim, not a unit"


# ── ranges, labels and years (#324) ──────────────────────────────────────────
#
# A live answer quoted "Q2 325 (297-348), Q10 433 (401-453)", en dashes in the
# ranges, and the check read 2325, -348, 10433 and -453 out of it: the label's digits were glued onto the
# next number by the grouping rule, and the dash of a range was taken for a
# sign. The same answer added "the 2014 winter floods" from memory; nothing
# checked that.


@pytest.mark.parametrize("text", ["453 – 525", "297–348", "447 - 604"])
def test_a_dash_between_two_numbers_is_a_range_not_a_sign(text) -> None:
    assert verify_mod._numbers(verify_mod.normalise(text)) == [float(text[:3]), float(text[-3:])]


@pytest.mark.parametrize("text, value", [("Q2 325", 325.0), ("Q10 433", 433.0), ("T100 495", 495.0)])
def test_a_label_is_not_glued_onto_the_number_after_it(text, value) -> None:
    assert verify_mod._numbers(verify_mod.normalise(text)) == [value]


def test_a_real_negative_survives() -> None:
    assert verify_mod._numbers("-0.864") == [-0.864]
    assert verify_mod._numbers("skew = -0.864") == [-0.864]
    assert verify_mod._numbers(verify_mod.normalise("tau = −0.14")) == [-0.14]


RETURN_LEVELS = {
    "unit": "m3/s", "station_id": "K",
    "ffa": {"return_periods": [2, 10, 50], "fits": {
        "gev_lmoments": {"q": [325.0, 433.0, 482.0]},
        "lp3": {"q": [325.0, 433.0, 482.0], "ci": [[297.0, 348.0], [401.0, 453.0], [443.0, 507.0]]},
    }},
}


def test_return_levels_with_their_intervals_are_not_reported_as_fabricated() -> None:
    answer = "At K: Q2 325 (297–348), Q10 433 (401–453), Q50 482 (443–507) m³/s."
    v = verify_mod.verify(answer, _one(RETURN_LEVELS))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}, [c.detail for c in v.failed]


def test_a_year_from_memory_is_listed_as_not_established() -> None:
    v = verify_mod.verify("Kingston's peak in the 2014 winter floods was 497 m3/s.", _one(KINGSTON_PAYLOAD))
    failed = {c.name: c.detail for c in v.failed}
    assert "years_traceable" in failed and "2014" in failed["years_traceable"]


def test_a_year_labelled_general_knowledge_is_allowed() -> None:
    v = verify_mod.verify(
        "Kingston flooded in the winter of 2014 (from general knowledge, not from the data). "
        "The 100-year flood is 497 m3/s (453 to 525).",
        _one(KINGSTON_PAYLOAD),
    )
    assert "years_traceable" not in {c.name for c in v.failed}


def test_a_year_in_a_result_date_or_period_is_traceable() -> None:
    v = verify_mod.verify("Kingston's record runs from 1986 to 2026, in m3/s.", _one(KINGSTON_PAYLOAD))
    assert "years_traceable" not in {c.name for c in v.failed}


def test_a_year_the_user_asked_about_is_not_the_models_memory() -> None:
    v = verify_mod.verify("Kingston has no data for 1947, in m3/s.", _one(KINGSTON_PAYLOAD),
                          question="What happened at Kingston in 1947?")
    assert "years_traceable" not in {c.name for c in v.failed}


def test_the_years_check_prints_with_the_others() -> None:
    v = verify_mod.verify("The 2014 floods at Kingston.", _one(KINGSTON_PAYLOAD))
    md = v.to_markdown()
    assert "does not establish" in md and "2014" in md and "general knowledge" in md


def test_prose_rounding_and_gate_words_are_not_invented_numbers():
    """A live run flagged tau -0.004 (result -0.0037), 29.0 and a gate's '20 needed' as invented."""
    from aquascope.ai_engine import verify as v

    assert v._close(-0.004, [-0.0037])
    assert v._close(29.0, [29.4])
    assert v._close(650.0, [652.5])
    assert not v._close(0.004, [0.0091])
    assert not v._close(1234.0, [1300.0])
    results = [
        {"name": "analyze_station", "arguments": {}, "ok": True,
         "payload": {"years": 142.9, "trend": {"tau": -0.0037}}},
        {"name": "gates", "arguments": {}, "ok": True,
         "payload": {"gates": [{"check": "min_years", "passed": True, "detail": "142.9 years of record, 20 needed"}]}},
    ]
    out = v.verify("Kendall tau = -0.004 over 142.9 years, against the 20 years the gate needs.", results)
    numbers = next(c for c in out.checks if "number" in c.name)
    assert numbers.passed, numbers.detail


def test_a_negative_slope_is_not_a_unit_exponent():
    """The keyless Kingston answer said "Sen's slope -0.0029 m3/s per year" and the check reported 29.0."""
    from aquascope.ai_engine import verify as v

    assert v._numbers("Sen's slope -0.0029 m3/s per year over 140 years", claims_only=True) == [-0.0029, 140.0]
    assert v._numbers("tau -0.14 and skew=-0.864", claims_only=True) == [-0.14, -0.864]
    assert v._numbers("mean flow 65.5 m s-1 and 12 kg-1", claims_only=True) == [65.5, 12.0]


# ── ranges and label-number pairs (#324) ────────────────────────────────────
#
# The first live run of the Anthropic provider reported six numbers as "not in
# any tool result" and none of them were real. They came from how the answer was
# written: an en dash between two bounds read as a negative, and a return-period
# label read as part of the following value.

KINGSTON_CURVE_PAYLOAD = {
    "unit": "m3/s", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston",
    "agency": "Environment Agency", "years": 40.0, "n": 14555,
    "ffa": {
        "return_periods": [2, 10, 100],
        "fits": {"gev": {
            "q": [325.0, 433.0, 497.0],
            "ci": [[297.0, 348.0], [400.0, 466.0], [453.0, 525.0]],
        }},
    },
}

KINGSTON_CURVE_ANSWER = (
    "**Flood frequency at Kingston (River Thames)**\n\n"
    "Environment Agency (UK) | **Kingston** (station ID "
    "8496ce69-482c-406a-a2f0-ac418ef8f099) | 40 yr, 14 555 daily values | "
    "Q2 325 (297–348) | Q10 433 (400–466) | Q100 497 m³ s⁻¹ "
    "(453 – 525 m³ s⁻¹, 90 % CI)"
)


def test_a_range_written_with_an_en_dash_is_two_positive_bounds() -> None:
    """`297–348` is an interval, not 297 and minus 348."""
    assert verify_mod._numbers(verify_mod.normalise("297–348")) == [297.0, 348.0]
    assert verify_mod._numbers(verify_mod.normalise("453 – 525")) == [453.0, 525.0]
    assert verify_mod._numbers(verify_mod.normalise("447 - 604")) == [447.0, 604.0]


def test_a_label_is_not_glued_onto_the_following_number() -> None:
    """`Q2 325` is a return-period label and its value, not 2325."""
    for prose, expected in (("Q2 325", 325.0), ("Q10 433", 433.0), ("T100 495", 495.0)):
        got = verify_mod._numbers(verify_mod.normalise(prose))
        assert expected in got, f"{prose!r} lost its value: {got}"
        assert not any(n > 1000 for n in got), f"{prose!r} invented {got}"


def test_digit_grouping_still_joins() -> None:
    """The label fix must not undo the grouping fix it sits next to."""
    assert verify_mod._numbers(verify_mod.normalise("14 555")) == [14555.0]
    assert verify_mod._numbers(verify_mod.normalise("1 234 567")) == [1234567.0]


def test_a_genuine_negative_survives() -> None:
    """Only a dash *between two digits* is a range separator."""
    assert verify_mod._numbers(verify_mod.normalise("skew=-0.864")) == [-0.864]
    assert verify_mod._numbers(verify_mod.normalise("τ = -0.14")) == [-0.14]
    assert -3.0 in verify_mod._numbers(verify_mod.normalise("values: 5, -3"))


def test_a_flood_curve_answer_produces_no_false_unestablished_line() -> None:
    """The reported case, end to end: six false numbers, none of them real."""
    v = verify_mod.verify(KINGSTON_CURVE_ANSWER, _one(KINGSTON_CURVE_PAYLOAD))
    assert "numbers_come_from_tools" not in {c.name for c in v.failed}, [
        (c.name, c.detail) for c in v.failed
    ]
