"""`aquascope playbooks` and `aquascope solve --lat --lon`: the CLI face of Solve."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import aquascope.explore
from aquascope import cli
from tests.test_ai_engine.test_team import CATCHMENT, FLOW, RECON


def _tools():
    return {"describe_catchment": lambda **kw: CATCHMENT, "analyze_station": lambda **kw: FLOW,
            "flood_frequency": lambda **kw: FLOW}


def test_playbooks_list_and_show(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "playbooks"])
    cli.main()
    out = capsys.readouterr().out
    assert "flood_risk" in out and "ungauged_flow" in out and "groundwater_decline" in out and "7 playbook(s)" in out
    assert "drought_status" in out and "supply_reliability" in out and "irrigation_feasibility" in out
    assert "water_quality" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "playbooks", "show", "flood_risk"])
    cli.main()
    out = capsys.readouterr().out
    assert "at_site: when context.years_by_variable.discharge >= 20" in out
    assert "gates: min_years, max_return_period_factor, ci_finite, spread_within" in out
    assert "Wasko et al. 2024" in out and "declines:" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "playbooks", "show", "nope"])
    with pytest.raises(SystemExit):
        cli.main()


def test_solve_prints_the_plan_runs_with_yes_and_writes_the_study(monkeypatch, capsys, tmp_path):
    out = tmp_path / "report.md"
    study = tmp_path / "study.yaml"
    monkeypatch.setattr(sys, "argv", [
        "aquascope", "solve", "Design flow for a road crossing, 100-year return period", "--lat", "51.415",
        "--lon", "-0.308", "--playbook", "flood_risk", "--intake", "return_period=50", "--yes", "-q",
        "--out", str(out), "--study", str(study)])
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=RECON), \
         patch("aquascope.study._tools", return_value=_tools()):
        cli.main()
    printed = capsys.readouterr().out
    assert "Plan: playbook flood_risk, branch at_site, 3 step(s)" in printed
    assert "gate max_return_period_factor 3 on years" in printed and "Report saved to" in printed
    assert "## Steps and gates" in out.read_text() and "T = 50 years" in out.read_text()
    text = study.read_text()
    assert text.startswith("# An AquaScope study (version 2)") and '"return_period": 50' in text
    # and the study re-runs, gates and all, with no model
    monkeypatch.setattr(sys, "argv", ["aquascope", "run", str(study), "-q"])
    with patch("aquascope.study._tools", return_value=_tools()):
        cli.main()
    rerun = capsys.readouterr().out
    assert "gate spread_within: passed" in rerun and "No model was involved" in rerun


def test_solve_without_a_terminal_declines_unless_yes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "solve", "Design flow, 100-year return period",
                                      "--lat", "51.415", "--lon", "-0.308", "-q"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=RECON), \
         patch("aquascope.study._tools", return_value=_tools()):
        cli.main()
    captured = capsys.readouterr()
    assert "pass --yes" in captured.err and "Declined: The plan was declined at review." in captured.err


def test_solve_needs_both_coordinates(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aquascope", "solve", "x", "--lat", "1"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["aquascope", "solve", "x", "--lat", "1", "--lon", "2", "--intake", "broken"])
    with pytest.raises(SystemExit):
        cli.main()
