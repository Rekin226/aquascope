"""`aquascope studio`: --yes writes the bundle, no terminal saves the workspace, --resume continues, the
interactive path answers, edits and follows up through input()."""

from __future__ import annotations

import json
import sys

import pytest

from aquascope import cli
from tests.test_studio.conftest import PROBLEM, patched


def _argv(monkeypatch, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", PROBLEM, "--lat", "51.415", "--lon", "-0.308", *extra])


def test_yes_runs_to_the_bundle(monkeypatch, capsys, tmp_path, no_deliverables):
    out = tmp_path / "bundle"
    _argv(monkeypatch, "--yes", "-q", "--out", str(out), "--intake", "return_period=50")
    with patched():
        cli.main()
    printed = capsys.readouterr().out
    assert "Plan (playbook, playbook flood_risk, branch at_site, 4 step(s))" in printed
    assert "The record at Kingston" in printed and "Bundle written to" in printed
    names = {p.name for p in out.iterdir()}
    assert {"report.md", "study.yaml", "report.json", "workspace.json"} <= names
    assert "T = 50 years" in (out / "report.md").read_text(encoding="utf-8")
    ws = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert ws["status"] == "done" and ws["brief"]["intake"]["return_period"] == 50
    # the study re-runs with no model
    monkeypatch.setattr(sys, "argv", ["aquascope", "run", str(out / "study.yaml"), "-q"])
    with patched():
        cli.main()
    assert "No model was involved" in capsys.readouterr().out


def test_without_a_terminal_the_plan_waits_and_the_workspace_resumes(monkeypatch, capsys, tmp_path, no_deliverables):
    out = tmp_path / "b"
    _argv(monkeypatch, "-q", "--out", str(out))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with patched():
        cli.main()
    captured = capsys.readouterr()
    assert "pass --yes" in captured.err and (out / "workspace.json").exists()
    assert json.loads((out / "workspace.json").read_text(encoding="utf-8"))["status"] == "review"
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "--resume", str(out / "workspace.json"), "--yes", "-q",
                                      "--out", str(out)])
    with patched():
        cli.main()
    printed = capsys.readouterr().out
    workspace = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert "Bundle written to" in printed and workspace["status"] == "done"


def test_interactive_answers_edits_and_follows_up(monkeypatch, capsys, tmp_path, no_deliverables):
    out = tmp_path / "c"
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "Can the river supply the town reliably?", "--lat",
                                      "51.415", "--lon", "-0.308", "-q", "--out", str(out)])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["2 m3/s, a licence", "e", "s2.years=20", "how reliable is it?", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    with patched():
        cli.main()
    printed = capsys.readouterr().out
    assert "1. Demand" in printed and "Plan (playbook, playbook supply_reliability" in printed
    assert "Bundle written to" in printed
    ws = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert ws["status"] == "done" and ws["brief"]["intake"]["demand_m3s"] == 2.0
    assert next(s for s in ws["study"]["steps"] if s["id"] == "s2")["arguments"]["years"] == 20
    assert ws["follow_ups"][0]["kind"] == "question"


def test_bad_arguments_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "x", "--lat", "1"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "x", "--lat", "1", "--lon", "2", "--intake", "broken"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "--resume", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit):
        cli.main()
    assert cli._parse_edits("s3.return_period=200, s2.k=8") == {"s3": {"arguments": {"return_period": 200}},
                                                                 "s2": {"arguments": {"k": 8}}}
    with pytest.raises(ValueError):
        cli._parse_edits("nodot=1")


def test_data_files_become_uploads(monkeypatch, capsys, tmp_path, no_deliverables):
    from tests.test_studio.conftest import SERIES_CSV

    csv = tmp_path / "flows.csv"
    csv.write_text(SERIES_CSV, encoding="utf-8")
    out = tmp_path / "d"
    _argv(monkeypatch, "--yes", "-q", "--out", str(out), "--data", str(csv))
    with patched():
        cli.main()
    ws = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert "upload:flows.csv" in ws["tables"]
    assert any(d["id"] == "upload:flows.csv" and d["variable"] == "discharge" for d in ws["inventory"]["datasets"])


# ── `aquascope studio` alone: it asks where and what ─────────────────────────

_CATALOG = [
    {"source": "usgs", "station_id": "USGS-01013500", "name": "Fish River near Fort Kent, Maine",
     "latitude": 47.2375, "longitude": -68.5828, "variables": ["discharge"]},
    {"source": "uk_ea", "station_id": "kingston", "name": "Kingston", "river": "Thames",
     "latitude": 51.415, "longitude": -0.308, "variables": ["discharge"]},
]


@pytest.fixture
def catalog():
    from aquascope.archive import catalog as cat

    cat.set_catalog(_CATALOG)
    yield
    cat.set_catalog(None)


def test_a_place_is_coordinates_a_station_id_or_words(catalog):
    assert cli._place_matches("47.2375, -68.5828")[0]["latitude"] == 47.2375
    assert cli._place_matches("usgs-01013500")[0]["station_id"] == "USGS-01013500"
    assert cli._place_matches("Thames Kingston")[0]["station_id"] == "kingston"
    assert cli._place_matches("nowhere at all") == []
    with pytest.raises(ValueError):
        cli._place_matches("95, 10")


def test_bare_studio_asks_where_and_what_then_runs(monkeypatch, capsys, tmp_path, catalog, no_deliverables):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "HF_TOKEN",
                "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    out = tmp_path / "bare"
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", "-q", "--out", str(out)])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def answer(prompt=""):
        if prompt.startswith("Where?"):
            return "Kingston"
        if prompt.startswith("\nWhat do you want"):
            return PROBLEM
        if prompt.startswith("Run this plan?"):
            return "y"
        return "done" if prompt.startswith("Follow-up") else "just go"

    monkeypatch.setattr("builtins.input", answer)
    with patched():
        cli.main()
    captured = capsys.readouterr()
    assert "Kingston (uk_ea kingston)" in captured.out and "Do you have an AI model key?" in captured.out
    ws = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert ws["status"] == "done"


def test_a_key_in_the_environment_is_offered_not_taken(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    args = cli.argparse.Namespace(lat=1.0, lon=2.0, query="q", provider=None, model=None, api_key=None,
                                  base_url=None, max_usd=None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli._studio_start(args) and args.provider is None
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert cli._studio_start(args) and args.provider == "groq" and args.max_usd == 1.0


def test_without_a_terminal_or_a_place_it_says_how(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "studio", PROBLEM])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit):
        cli.main()


def test_a_question_is_a_pick_list_by_number_without_questionary(monkeypatch, capsys):
    import builtins

    monkeypatch.setitem(sys.modules, "questionary", None)
    q = {"text": "Which period?", "why": "It changes the test.", "options": ["Whole", "Last 50"], "default": "Whole"}
    for typed, want in [("2", "Last 50"), ("", "Whole"), ("my own words", "my own words")]:
        monkeypatch.setattr(builtins, "input", lambda prompt="", typed=typed: typed)
        assert cli._choose(q) == want
    answers = iter(["3", "since 1990"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert cli._choose(q) == "since 1990", "the last row takes the answer in your own words"
    out = capsys.readouterr().out
    assert "(It changes the test.)" in out and "1. Whole  (default)" in out and "3. Other (type it)" in out


def test_not_now_at_the_plan_saves_and_says_how_to_resume(monkeypatch, capsys, tmp_path, no_deliverables):
    out = tmp_path / "later"
    _argv(monkeypatch, "-q", "--out", str(out), "--intake", "return_period=50")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["maybe", "3"])    # a reply that is no option is taken as a change and asked again; 3 is later

    def answer(prompt=""):
        if prompt.startswith("Pick"):
            return next(answers)
        return "done" if prompt.startswith("Follow-up") else "just go"

    monkeypatch.setattr("builtins.input", answer)
    with patched():
        cli.main()
    err = capsys.readouterr().err
    assert "Declined" not in err and "Saved. Pick it up any time: aquascope studio --resume" in err
    ws = json.loads((out / "workspace.json").read_text(encoding="utf-8"))
    assert ws["status"] == "review"
