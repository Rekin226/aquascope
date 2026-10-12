"""`aquascope eval`: a study's scorecard, its trace and stats across studies, read from the bundle alone."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aquascope import evaluation as ev

SHOWCASE = Path(__file__).resolve().parents[1] / "explorer" / "showcase" / "studies"
KINGSTON = SHOWCASE / "kingston-flood"


def _workspace() -> dict:
    """A small finished keyless study: two steps, one gate skipped and one failed, a short timeline."""
    at = lambda s: f"2026-10-04T08:00:{s:02d}+00:00"  # noqa: E731
    status = lambda s, name: {"role": "coordinator", "step": None, "event": "status", "detail": name, "at": at(s)}  # noqa: E731
    return {
        "id": "abc123", "created": at(0), "status": "done", "model": None, "provider": None,
        "brief": {"problem": "What is the 100-year flood?", "playbook": "flood_risk"},
        "study": {"aquascope_version": "0.22.0", "plan": {"branch": "at_site"}},
        "run": {
            "started": at(10), "finished": at(40), "replans": 0,
            "summary": {"planned": 2, "ran": 2, "ok": 2, "failed": 0, "skipped": 0},
            "results": [
                {"id": "s1", "tool": "flood_frequency", "ok": True, "gates_passed": False, "gates": [
                    {"check": "min_years", "passed": True, "detail": "60 years of record, 20 needed"},
                    {"check": "spread_within", "passed": False, "detail": "spread 40% (25% allowed)"}]},
                {"id": "s2", "tool": "anywhere", "ok": True, "gates_passed": True, "fallback_used": True, "gates": [
                    {"check": "cross_check_ratio", "passed": True, "skipped": True,
                     "detail": "skipped: no comparable model cell verified"}]},
            ],
        },
        "critique": {"checks": [{"name": "a", "passed": True}, {"name": "b", "passed": False}], "issues": ["x"],
                     "not_established": ["y", "z"]},
        "report": {"grade": "indicative", "answer": "design flow: 100-year 412 m3/s (indicative). More text."},
        "ledger": {"author": {"calls": 2, "prompt_tokens": 1000, "completion_tokens": 500, "cost_usd": 0.01}},
        "events": [
            status(0, "scouting"), status(8, "planning"), status(9, "review"), status(10, "running"),
            {"role": "runner", "step": "s1", "event": "start", "detail": "flood_frequency()", "at": at(10)},
            {"role": "runner", "step": "s1", "event": "done", "detail": "", "at": at(22)},
            {"role": "runner", "step": "s2", "event": "start", "detail": "anywhere()", "at": at(22)},
            {"role": "runner", "step": "s2", "event": "done", "detail": "", "at": at(40)},
            status(40, "critique"), status(41, "authoring"), status(45, "done"),
        ],
    }


def test_the_trace_times_each_phase_and_step_and_lists_what_did_not_pass() -> None:
    t = ev.trace(_workspace())
    assert t["seconds"] == 45.0
    assert [(p["phase"], p["seconds"]) for p in t["phases"]] == [
        ("scouting", 8.0), ("planning", 1.0), ("review", 1.0), ("running", 30.0), ("critique", 1.0),
        ("authoring", 4.0)]
    assert t["slowest_phase"] == {"phase": "running", "seconds": 30.0}
    s1, s2 = t["steps"]
    assert s1["seconds"] == 12.0 and s1["gates"] == {"passed": 1, "failed": 1, "skipped": 0}
    assert s2["fallback_used"] and s2["gates"] == {"passed": 0, "failed": 0, "skipped": 1}
    assert s2["not_passed"] == [{"check": "cross_check_ratio", "state": "skipped",
                                 "detail": "no comparable model cell verified"}], "no doubled 'skipped: '"
    assert t["slowest_step"] == {"id": "s2", "tool": "anywhere", "seconds": 18.0}
    assert t["model_calls"]["author"]["calls"] == 2 and t["events"][0]["t"] == 0.0


def test_review_never_counts_as_the_slowest_phase() -> None:
    ws = _workspace()
    ws["events"][2]["at"] = "2026-10-04T07:00:00+00:00"   # a person who took an hour to approve
    ws["events"][0]["at"] = "2026-10-04T06:59:00+00:00"
    assert ev.trace(ws)["slowest_phase"]["phase"] != "review"


def test_the_scorecard_counts_gates_critic_and_cost(tmp_path) -> None:
    (tmp_path / "workspace.json").write_text(json.dumps(_workspace()), encoding="utf-8")
    card = ev.score_path(tmp_path)
    assert card["playbook"] == "flood_risk" and card["branch"] == "at_site" and card["date"] == "2026-10-04"
    assert card["outcome"]["grade"] == "indicative"
    assert card["outcome"]["headline"] == "design flow: 100-year 412 m3/s (indicative)"
    assert card["run"]["gates"] == {"passed": 1, "failed": 1, "skipped": 1} and card["run"]["fallbacks"] == 1
    assert card["critic"] == {"passed": 1, "total": 2, "issues": 1, "not_established": 2}
    assert card["cost"]["calls"] == 2 and card["cost"]["prompt_tokens"] == 1000 and card["cost"]["usd"] == 0.01
    assert "mean" in card["report"], "a partial workspace still gets a report row (scored or with its error)"


@pytest.mark.skipif(not KINGSTON.exists(), reason="the recorded showcase is not in this checkout")
def test_a_recorded_study_scores_its_report_against_its_reference_and_its_plan() -> None:
    card = ev.score_path(KINGSTON, plan_case="flood_at_site_potomac")
    assert card["model"] == "claude-sonnet-5" and card["cost"]["usd"] == pytest.approx(1.1231)
    assert card["report"]["mean"] is not None and card["report"]["reference"] is not None, (
        "kingston-flood has a report reference in the package")
    assert card["plan"]["case"] == "flood_at_site_potomac" and 0 <= card["plan"]["score"] <= 1
    assert card["cost"]["by_role"]["critic"]["calls"] >= 1


@pytest.mark.skipif(not SHOWCASE.exists(), reason="the recorded showcase is not in this checkout")
def test_stats_groups_every_recorded_study() -> None:
    dirs = ev.find_studies([SHOWCASE])
    assert len(dirs) == len(list(SHOWCASE.glob("*/workspace.json")))
    cards = [ev.score_path(d) for d in dirs]
    result = ev.stats(cards, by="model")
    assert sum(r["n"] for r in result["rows"]) == len(cards)
    for r in result["rows"]:
        assert r["gates_passed"] + r["gates_failed"] + r["gates_skipped"] == pytest.approx(1, abs=0.01)
    assert set(result["gate_problems"]) == {"failed", "skipped"}
    csv = ev.stats_rows_csv(result).splitlines()
    assert csv[0].startswith("model,n,grade_") and len(csv) == len(result["rows"]) + 1
    with pytest.raises(ValueError, match="group by"):
        ev.stats(cards, by="colour")


def test_find_studies_takes_dirs_files_and_trees(tmp_path) -> None:
    for name in ("a", "b/c"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "workspace.json").write_text("{}", encoding="utf-8")
    assert ev.find_studies([tmp_path]) == [tmp_path / "a", tmp_path / "b" / "c"]
    assert ev.find_studies([tmp_path / "a" / "workspace.json", tmp_path / "a"]) == [tmp_path / "a"]
    with pytest.raises(FileNotFoundError):
        ev.load_study(tmp_path / "missing")


def _cli(monkeypatch, capsys, *argv: str) -> str:
    from aquascope.cli import main

    monkeypatch.setattr(sys, "argv", ["aquascope", *argv])
    main()
    return capsys.readouterr().out


def test_the_cli_prints_a_scorecard_a_trace_and_stats(tmp_path, monkeypatch, capsys) -> None:
    study = tmp_path / "studies" / "one"
    study.mkdir(parents=True)
    (study / "workspace.json").write_text(json.dumps(_workspace()), encoding="utf-8")
    out = _cli(monkeypatch, capsys, "eval", "score", str(study))
    assert "grade indicative" in out and "gates 1 passed, 1 failed, 1 skipped" in out
    out = _cli(monkeypatch, capsys, "eval", "trace", str(study), "--events")
    assert "running 30 s" in out and "skipped cross_check_ratio: no comparable" in out and "events" in out
    out = _cli(monkeypatch, capsys, "eval", "stats", str(tmp_path / "studies"), "--by", "grade")
    assert "| indicative | 1 |" in out and "most often failed: spread_within (1 study)" in out
    data = json.loads(_cli(monkeypatch, capsys, "eval", "score", str(study), "--json"))
    assert data["run"]["gates"]["skipped"] == 1


def test_stats_on_a_folder_with_no_studies_says_so(tmp_path, monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _cli(monkeypatch, capsys, "eval", "stats", str(tmp_path))
    assert exit_info.value.code == 1
    assert "no studies" in capsys.readouterr().err
