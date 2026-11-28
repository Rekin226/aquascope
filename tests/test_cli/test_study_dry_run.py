"""Offline tool validation for `aquascope run study.yaml --dry-run`."""

import json
import sys
from unittest.mock import Mock

import pytest

from aquascope import cli


def test_dry_run_rejects_unknown_step_and_fallback_tools(tmp_path, monkeypatch, caplog):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"version": 2, "steps": [
        {"tool": "flood_frequncy"},
        {"id": "gauge", "tool": "list_sources", "fallback": {"step": {"tool": "missing_fallback"}}},
    ]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aquascope", "run", str(path), "--dry-run"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "s1: unknown tool 'flood_frequncy'" in caplog.text
    assert "gauge.fallback: unknown tool 'missing_fallback'" in caplog.text


@pytest.mark.parametrize("plan", [
    {"steps": [{"tool": "list_sources"}]},
    {"version": 2, "steps": [
        {"tool": "list_sources", "fallback": {"step": {"tool": "list_sources"}}},
        {"tool": "list_sources", "fallback": {"branch": "regional"}},
        {"tool": "list_sources", "fallback": "stop"},
    ]},
])
def test_dry_run_lists_valid_steps_without_executing_them(plan, tmp_path, monkeypatch, capsys):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    run = Mock()
    monkeypatch.setattr("aquascope.study.run_study", run)
    monkeypatch.setattr(sys, "argv", ["aquascope", "run", str(path), "--dry-run"])

    cli.main()

    assert "1. list_sources()" in capsys.readouterr().out
    run.assert_not_called()
