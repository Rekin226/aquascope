"""HydroGym report-quality scoring (#382): score_report over a synthetic workspace (a traced and an untraced
number, a failed gate named and unnamed in the limitations, a direction inversion, a findings block with a
resolving and a non-resolving basis) and the three references against their recorded studies. Nothing here
touches the network or a model."""

from __future__ import annotations

import json
import sys

import pytest

from aquascope import cli
from aquascope.gym import reports as gr

REFS = gr.load_references()
BY_ID = {r.id: r for r in REFS}


def _ws(*, answer: str, limitations: str = "", recommendations: str = "", extra_sections: list[dict] | None = None,
       s2_gate_passed: bool = True, s2_result: dict | None = None, findings: dict | None = None) -> dict:
    """A small, valid workspace dict: one trend step that establishes (s1, increasing, p = 0.01) and one anywhere
    step (s2) whose ``not_empty`` gate on 'flow' either passes or fails, so tests can flip one failure on and off."""
    s2_gates = [{"check": "not_empty", "passed": s2_gate_passed, "detail": "" if s2_gate_passed
                else "nothing at 'flow'"}]
    ws = {
        "site": {"lat": 10.0, "lon": 20.0},
        "brief": {"problem": "Is the flow trending, and what is it now?"},
        "study": {"steps": [{"id": "s1", "tool": "analyze_station", "method": "trend_mann_kendall"},
                            {"id": "s2", "tool": "anywhere", "method": "glofas_cross_check"}]},
        "run": {
            "results": [
                {"id": "s1", "tool": "analyze_station", "ok": True, "gates_passed": True,
                 "arguments": {"source": "usgs", "station_id": "X1"},
                 "result": {"trend": {"trend": "increasing", "p_value": 0.01, "tau": 0.5,
                                      "sens_slope_per_year": 0.2},
                           "years": 30.0, "unit": "m3/s"},
                 "gates": [{"check": "min_years", "passed": True, "detail": "30.0 years of record, 10 needed"}],
                 "methods": [{"name": "Mann-Kendall trend", "text": "...",
                             "citation": "Mann, H. B. (1945). Econometrica; Sen, P. K. (1968). J. Am. Stat. Assoc."}]},
                {"id": "s2", "tool": "anywhere", "ok": True, "gates_passed": s2_gate_passed,
                 "arguments": {"lat": 10.0, "lon": 20.0},
                 "result": s2_result if s2_result is not None else {"note": "no data"},
                 "gates": s2_gates, "fallback_used": False},
            ],
            "failed_gates": [] if s2_gate_passed else [{"step": "s2", "check": "not_empty", "passed": False,
                                                        "detail": "nothing at 'flow'"}],
            "stop_reason": None,
        },
        "critique": None,
        "report": {
            "answer": answer,
            "references": ["Mann, H. B. (1945). Nonparametric tests against trend. Econometrica.",
                          "Sen, P. K. (1968). J. Am. Stat. Assoc."],
            "sections": [{"id": "limitations", "text": limitations},
                        {"id": "recommendations", "text": recommendations}] + (extra_sections or []),
        },
    }
    if findings is not None:
        ws["findings"] = findings
    return ws


# ── numbers: traced and untraced ──


def test_traceability_scores_a_traced_and_an_untraced_number():
    ws = _ws(answer="The trend is 0.2 m3/s per year over 30.0 years. A separate estimate of 999.9 m3/s is "
                    "also quoted.",
            limitations="", recommendations="")
    out = gr.score_report(ws)
    assert out["dimensions"]["traceability"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["evidence"]["traceability"] == ["not in any tool result or its arguments: 999.9"]
    assert out["metrics"]["numbers_without_evidence"] == [999.9]
    assert out["metrics"]["grounding_rate"] == out["dimensions"]["traceability"]
    # arguments count too: the coordinates are given to a tool, not invented
    ws2 = _ws(answer="The site is at 10.0, 20.0.")
    assert gr.score_report(ws2)["dimensions"]["traceability"] == 1.0


# ── a failed gate: named and unnamed in the limitations ──


def test_not_established_completeness_rewards_naming_the_failed_gate():
    named = _ws(answer="The trend is 0.2 m3/s per year, increasing.", s2_gate_passed=False,
               limitations="The GloFAS cross-check (step s2) failed: nothing at 'flow' was returned.")
    out = gr.score_report(named)
    assert out["dimensions"]["not_established_completeness"] == 1.0
    assert out["evidence"]["not_established_completeness"] == []

    unnamed = _ws(answer="The trend is 0.2 m3/s per year, increasing.", s2_gate_passed=False,
                 limitations="No further caveats.")
    out2 = gr.score_report(unnamed)
    assert out2["dimensions"]["not_established_completeness"] == 0.0
    assert out2["evidence"]["not_established_completeness"]

    # nothing failed: the dimension has nothing to judge, so it is left out rather than scored perfect
    clean = _ws(answer="The trend is 0.2 m3/s per year, increasing.", s2_gate_passed=True)
    assert gr.score_report(clean)["dimensions"]["not_established_completeness"] is None


# ── a direction inversion ──


def test_a_direction_inversion_is_caught_and_lowers_no_filled_holes():
    inverted = _ws(answer="Mann-Kendall finds the trend is 0.2 m3/s per year. This confirms a declining trend "
                          "in the record.")
    out = gr.score_report(inverted)
    assert out["dimensions"]["no_filled_holes"] < 1.0
    inv = out["metrics"]["direction_inversions"]
    assert len(inv) == 1 and inv[0]["claimed"] == "down" and inv[0]["classifications"] == ["increasing"]
    assert "claims down where the test found increasing" in out["evidence"]["no_filled_holes"][0]

    # the honest direction, or a negated one, is not flagged
    honest = _ws(answer="Mann-Kendall finds an increasing trend of 0.2 m3/s per year, not a decline.")
    assert gr.score_report(honest)["metrics"]["direction_inversions"] == []
    assert gr.score_report(honest)["dimensions"]["no_filled_holes"] == 1.0


def test_a_number_quoted_near_a_failed_step_that_does_not_trace_is_a_filled_hole():
    ws = _ws(answer="The trend is 0.2 m3/s per year.", s2_gate_passed=False,
             limitations="The GloFAS cross-check gives a flow of 777.0 m3/s despite the missing 'flow' field.")
    out = gr.score_report(ws)
    assert out["dimensions"]["no_filled_holes"] < 1.0
    holes = [e for e in out["evidence"]["no_filled_holes"] if "777" in e]
    assert holes, out["evidence"]["no_filled_holes"]


# ── findings (#417): a resolving and a non-resolving basis ──


def test_findings_basis_resolution_and_wrong_but_valid():
    findings = {
        "findings": [
            {"id": "f1", "claim": "The trend is 0.2 m3/s per year.", "basis": ["s1.trend.sens_slope_per_year"],
             "grade": "established"},
            {"id": "f2", "claim": "The flow is 500 m3/s.", "basis": ["s2.flow"], "grade": "screening"},
            {"id": "f3", "claim": "The record is 999 years long.", "basis": ["s1.years"], "grade": "screening"},
        ],
        "decision": {"answer": "increasing", "value": 0.2, "unit": "m3/s/yr", "grade": "established"},
        "data_requests": [{"what": "abstraction records", "why": "to attribute cause"}],
    }
    ws = _ws(answer="The trend is 0.2 m3/s per year, increasing.", findings=findings)
    out = gr.score_report(ws)
    assert "findings" in out
    f = out["findings"]
    # f1's basis resolves (0.2, matching the claim); f2's does not (s2's payload has no 'flow'); f3's resolves
    # but to 30.0, nowhere near the claimed 999 years: wrong but valid.
    assert f["basis_resolution_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert f["wrong_but_valid_rate"] == pytest.approx(1 / 2, abs=1e-3)
    assert "s2.flow" in f["evidence"]["unresolved_basis"]
    assert any("s1.years" in w for w in f["evidence"]["wrong_but_valid"])

    # a workspace with no findings key carries none
    assert "findings" not in gr.score_report(_ws(answer="x"))


def test_findings_scored_against_a_reference():
    findings = {"findings": [], "decision": {"value": 0.19, "unit": "m3/s/yr", "grade": "indicative"},
               "data_requests": [{"what": "abstraction records", "why": "cause"}]}
    ws = _ws(answer="x", findings=findings)
    ref = gr.Reference(
        id="t", study="t",
        findings={"decision": {"value": 0.2, "tolerance": 0.1, "grade": "screening"},
                 "data_requests": ["abstraction records"]},
    )
    out = gr.score_report(ws, reference=ref)["findings"]
    # the value is within tolerance, but "indicative" is less cautious than the "screening" the reference
    # wants, so the decision does not agree
    assert out["decision_agreement"] == 0.0
    assert out["asked_when_reference_asks"] == 1.0


# ── the three references load and score ──


def test_the_three_references_load_and_validate():
    assert {r.id for r in REFS} == {"kingston-flood", "cambridge-groundwater", "toulouse-irrigation"}
    for ref in REFS:
        assert ref.study and (ref.must_say or ref.must_not_say)
        assert (gr.SHOWCASE_DIR / ref.study / "workspace.json").exists(), ref.study


def test_kingston_and_cambridge_score_perfectly_against_their_recorded_reports():
    for case_id in ("kingston-flood", "cambridge-groundwater"):
        ref = BY_ID[case_id]
        with (gr.SHOWCASE_DIR / ref.study / "workspace.json").open(encoding="utf-8") as fh:
            ws = json.load(fh)
        out = gr.score_report(ws, reference=ref)
        assert out["reference"]["score"] == 1.0, out["reference"]
        assert out["reference"]["must_say"]["failed"] == []
        assert out["reference"]["must_not_say"]["violated"] == []


def test_toulouse_states_the_significant_decreasing_trend_and_scores_clean():
    """The 2026-09-07 recording called a p = 0.302 trend "downward" and the scorer caught it (the synthetic
    direction-inversion test above keeps that metric honest). The 2026-09-15 recording on the 116-year record
    finds a significant decreasing trend (p < 0.001) and says so; the reference now requires the direction to
    be stated and forbids calling the trend insignificant."""
    ref = BY_ID["toulouse-irrigation"]
    with (gr.SHOWCASE_DIR / ref.study / "workspace.json").open(encoding="utf-8") as fh:
        ws = json.load(fh)
    out = gr.score_report(ws, reference=ref)
    assert out["reference"]["must_say"]["failed"] == [], out["reference"]
    assert out["reference"]["must_not_say"]["violated"] == []
    assert out["reference"]["score"] == 1.0 and out["dimensions"]["no_filled_holes"] == 1.0


# ── the CLI ──


def test_the_cli_reports_verbs(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "reports", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "kingston-flood" in out and "cambridge-groundwater" in out and "toulouse-irrigation" in out

    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "reports", "show", "kingston-flood"])
    cli.main()
    assert "glofas" in capsys.readouterr().out.lower()

    study_dir = gr.SHOWCASE_DIR / "kingston-flood"
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "reports", "score", str(study_dir), "--json"])
    cli.main()
    scored = json.loads(capsys.readouterr().out)
    assert scored["dimensions"]["traceability"] == 1.0

    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "reports", "score", str(study_dir), "--reference",
                                      "kingston-flood", "--json"])
    cli.main()
    scored2 = json.loads(capsys.readouterr().out)
    assert scored2["reference"]["score"] == 1.0

    out_file = tmp_path / "reports.jsonl"
    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "reports", "bench", "--study", "kingston-flood",
                                      "--study", "toulouse-irrigation", "--out", str(out_file), "--quiet"])
    cli.main()
    printed = capsys.readouterr().out
    assert "report-quality leaderboard" in printed and len(out_file.read_text(encoding="utf-8").splitlines()) == 2

    monkeypatch.setattr(sys, "argv", ["aquascope", "gym", "leaderboard", str(out_file)])
    cli.main()
    assert "report-quality leaderboard" in capsys.readouterr().out
