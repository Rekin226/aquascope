"""The collector sweep keeps its promises: one row per source, one issue, no false red.

The script lives in ``.github/scripts/`` where the workflow calls it, so it is
loaded by path -- the same way ``job_health.py`` loads the health table. The
network is never touched here: ``run_probe`` takes its collector from
``aquascope.registry.build_collector``, which every test replaces.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "collector_health.py"


def load():
    spec = importlib.util.spec_from_file_location("collector_health", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def health():
    return load()


class _FakeCollector:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error

    def collect(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


def _patch(monkeypatch, collector):
    import aquascope.registry

    monkeypatch.setattr(aquascope.registry, "build_collector", lambda *a, **k: collector)


# ── the table has to match the registry ──────────────────────────────────────


def test_every_source_has_a_probe(health) -> None:
    from aquascope.registry import SOURCES

    assert set(health.PROBES) == set(SOURCES), (
        "PROBES and SOURCES drifted; --check in the workflow fails on this too"
    )


def test_check_mode_passes(health) -> None:
    assert health.main(["--check"]) == 0


def test_an_unknown_source_is_rejected(health, capsys) -> None:
    assert health.main(["--source", "not-a-source", "--no-issue"]) == 2
    assert "unknown source" in capsys.readouterr().err


# ── one outcome per call ─────────────────────────────────────────────────────


def test_a_call_with_records_is_ok(health, monkeypatch) -> None:
    _patch(monkeypatch, _FakeCollector([{"a": 1}, {"a": 2}]))
    outcome = health.run_probe("usgs", health.PROBES["usgs"])
    assert outcome.status == health.OK
    assert outcome.records == 2
    assert not outcome.failing


def test_a_call_without_records_is_empty_not_failed(health, monkeypatch) -> None:
    _patch(monkeypatch, _FakeCollector([]))
    outcome = health.run_probe("usgs", health.PROBES["usgs"])
    assert outcome.status == health.EMPTY
    assert not outcome.failing


def test_a_raise_is_a_failed_row_with_the_cause(health, monkeypatch) -> None:
    _patch(monkeypatch, _FakeCollector(error=RuntimeError("boom")))
    outcome = health.run_probe("usgs", health.PROBES["usgs"])
    assert outcome.status == health.FAIL
    assert "RuntimeError: boom" in outcome.detail
    assert outcome.failing


def test_a_slow_call_times_out(health, monkeypatch) -> None:
    import time

    def slow(**kwargs):
        time.sleep(2)
        return [{"a": 1}]

    class Slow:
        collect = staticmethod(slow)

    _patch(monkeypatch, Slow())
    probe = health.replace(health.PROBES["usgs"], timeout=1)
    outcome = health.run_probe("usgs", probe)
    assert outcome.status == health.TIMEOUT


def test_a_constructor_failure_is_a_row_not_a_crash(health, monkeypatch) -> None:
    import aquascope.registry

    def explode(*args, **kwargs):
        raise ValueError("bad constructor")

    monkeypatch.setattr(aquascope.registry, "build_collector", explode)
    outcome = health.run_probe("usgs", health.PROBES["usgs"])
    assert outcome.status == health.FAIL
    assert "constructor" in outcome.detail


# ── the summary and the issue ────────────────────────────────────────────────


def _outcomes(health):
    return [
        health.Outcome("usgs", health.FAIL, 1.0, 0, detail="404 gone"),
        health.Outcome("bom", health.OK, 2.0, 7),
        health.Outcome("taiwan_civil_iot", health.FAIL, 3.0, 0, detail="certificate", informational=True),
        health.Outcome("aquastat", health.EMPTY, 4.0, 0),
        health.Outcome("wapor", health.NEEDS_KEY, 0.0, 0, detail="missing KEY"),
    ]


def test_the_summary_puts_the_worst_first(health) -> None:
    text = health.format_summary(_outcomes(health))
    assert text.index("NEEDS_KEY") < text.index("FAIL") < text.index("EMPTY") < text.index("OK")


def test_the_summary_counts_every_status(health) -> None:
    text = health.format_summary(_outcomes(health))
    assert "5 sources" in text
    # taiwan_civil_iot fails too, but only usgs is counted as a real failure.
    assert "2 FAIL" in text and "1 OK" in text and "1 NEEDS_KEY" in text
    assert "**Failing:** `usgs`" in text


def test_the_issue_lists_only_real_failures(health) -> None:
    body = health.issue_body(_outcomes(health), run_url="https://example.test/run/1")
    assert "`usgs`" in body
    assert "`taiwan_civil_iot`" in body  # named as informational, not in the failure table
    assert "1 of 5 sources failed" in body
    assert "https://example.test/run/1" in body


def test_an_all_green_sweep_opens_nothing(health, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(health, "_gh", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setattr(health, "find_open_issue", lambda repo, dry_run=False: None)
    code = health.sync_issue([health.Outcome("usgs", health.OK, 1.0, 3)])
    assert code == 0
    assert not calls


def test_a_failure_opens_the_rolling_issue(health, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(health, "_gh", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setattr(health, "find_open_issue", lambda repo, dry_run=False: None)
    code = health.sync_issue(_outcomes(health))
    assert code == 1
    assert any(a[0] == "issue" and a[1] == "create" for a in calls)


def test_a_second_failure_comments_instead_of_opening(health, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(health, "_gh", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setattr(health, "find_open_issue", lambda repo, dry_run=False: 42)
    code = health.sync_issue(_outcomes(health))
    assert code == 1
    assert any(a[0] == "issue" and a[1] == "comment" and a[2] == "42" for a in calls)
    assert not any(a[1] == "create" for a in calls if a[0] == "issue")


def test_a_green_sweep_closes_the_issue(health, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(health, "_gh", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setattr(health, "find_open_issue", lambda repo, dry_run=False: 42)
    code = health.sync_issue([health.Outcome("usgs", health.OK, 1.0, 3)])
    assert code == 0
    assert any(a[0] == "issue" and a[1] == "close" for a in calls)


# ── what is allowed to be red ────────────────────────────────────────────────


def test_a_missing_secret_is_not_a_failure(health, monkeypatch, capsys) -> None:
    monkeypatch.delenv("TAIWAN_MOENV_API_KEY", raising=False)
    monkeypatch.setattr(health, "run_probe", lambda *a, **k: pytest.fail("must not call the network"))
    code = health.main(["--source", "taiwan_moenv", "--no-issue"])
    assert code == 0
    assert "NEEDS_KEY" in capsys.readouterr().out


def test_an_informational_failure_is_not_a_failure(health, monkeypatch, capsys) -> None:
    _patch(monkeypatch, _FakeCollector(error=RuntimeError("blocked")))
    code = health.main(["--source", "taiwan_civil_iot", "--no-issue"])
    assert code == 0
    assert "FAIL" in capsys.readouterr().out


def test_a_real_failure_exits_one(health, monkeypatch) -> None:
    _patch(monkeypatch, _FakeCollector(error=RuntimeError("gone")))
    assert health.main(["--source", "usgs", "--no-issue"]) == 1
