"""Job health records (#593, #449): the record, the index, the stale rule, the issue and the workflows' steps."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aquascope.archive import health_jobs as hj

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _run(key="forecast-archive", status="success", event="schedule", finished_at="2026-10-12T11:15:30Z",
         **kw) -> hj.Run:
    return hj.Run(key=key, status=status, event=event, started_at="2026-10-12T10:00:00Z", finished_at=finished_at,
                  **kw)


# ── judging a run ───────────────────────────────────────────────────────────────────────────────────────────


def test_a_successful_job_with_its_publish_is_a_success():
    assert hj.judge("success", {"publish": "success"}, event="schedule") == ("success", [])


def test_a_scheduled_run_that_did_not_publish_is_a_failure():
    """#449: the harvest ran, published nothing and passed. Now the promised publish decides."""
    status, reasons = hj.judge("success", {"publish": "skipped"}, event="schedule")
    assert status == "failure"
    assert reasons == ["publish did not run on a scheduled run"]
    assert hj.judge("success", {"publish": ""}, event="schedule")[0] == "failure"


def test_a_dispatched_run_is_not_held_to_the_publish_promise():
    assert hj.judge("success", {"publish": "skipped"}, event="workflow_dispatch") == ("success", [])


def test_cancelled_and_failed_jobs_fail_and_skipped_steps_record_nothing():
    assert hj.judge("cancelled")[1] == ["cancelled or timed out"]
    assert hj.judge("failure")[0] == "failure"
    assert hj.judge("skipped")[0] == "skipped"
    with pytest.raises(ValueError, match="skipped"):
        hj.make_record(_run(status="skipped"))


# ── the record ──────────────────────────────────────────────────────────────────────────────────────────────


def test_a_record_carries_the_run_its_duration_and_the_cadence():
    rec = hj.make_record(_run(counts={"gauges": 250}, data_date="2026-10-12", run_url="https://x/run/1"), now=NOW)
    assert rec["job"] == "forecast-archive" and rec["workflow"] == "forecast-archive.yml"
    assert rec["last_run"]["duration_s"] == 4530
    assert rec["last_run"]["counts"] == {"gauges": 250}
    assert rec["last_success"] == rec["last_run"] and rec["last_failure"] is None
    assert rec["stale_after_hours"] == 48 and rec["layers"] == ["forecast-points"]
    assert rec["green_streak"] == 1 and rec["consecutive_failures"] == 0


def test_a_failure_keeps_the_last_success_and_a_success_keeps_the_last_failure():
    ok = hj.make_record(_run(data_date="2026-10-11"), now=NOW)
    bad = hj.make_record(_run(status="failure", finished_at="2026-10-13T11:00:00Z"), ok, now=NOW)
    assert bad["last_success"]["data_date"] == "2026-10-11"
    assert bad["last_failure"]["status"] == "failure"
    assert bad["consecutive_failures"] == 1 and bad["green_streak"] == 0
    again = hj.make_record(_run(status="failure"), bad, now=NOW)
    assert again["consecutive_failures"] == 2
    good = hj.make_record(_run(), again, now=NOW)
    assert good["last_failure"] == again["last_failure"]
    assert good["consecutive_failures"] == 0 and good["green_streak"] == 1
    assert [h["status"] for h in good["history"]] == ["success", "failure", "failure", "success"]


def test_the_green_streak_counts_unattended_runs_only():
    rec = None
    for event in ("schedule", "schedule", "workflow_dispatch", "schedule"):
        rec = hj.make_record(_run(event=event), rec, now=NOW)
    assert rec["green_streak"] == 3


def test_the_history_is_capped():
    rec = None
    for _ in range(hj.HISTORY + 5):
        rec = hj.make_record(_run(), rec, now=NOW)
    assert len(rec["history"]) == hj.HISTORY


def test_a_previous_record_of_another_job_is_ignored():
    other = hj.make_record(_run(key="bulletin", status="failure"), now=NOW)
    rec = hj.make_record(_run(), other, now=NOW)
    assert rec["last_failure"] is None and len(rec["history"]) == 1


def test_times_with_a_z_and_epoch_seconds_parse_on_python_3_10():
    assert hj.parse_time("2026-10-12T10:00:00Z") == datetime(2026, 10, 12, 10, tzinfo=timezone.utc)
    assert hj.parse_time("1760263200") == datetime(2025, 10, 12, 10, tzinfo=timezone.utc)
    assert hj.parse_time("") is None and hj.parse_time("not a time") is None


# ── the stale rule (explorer/src/job-health-core.js reads the same fields the same way) ──────────────────────


def _entry(key="forecast-archive", hours_ago=10.0, status="success"):
    finished = NOW - timedelta(hours=hours_ago)
    rec = hj.make_record(_run(key=key, finished_at=finished.isoformat()), now=NOW)
    if status == "failure":
        rec = hj.make_record(_run(key=key, status="failure", finished_at=NOW.isoformat()), rec, now=NOW)
    return hj.index_entry(rec)


def test_fresh_stale_failing_unknown_and_on_demand():
    assert hj.assess(_entry(hours_ago=10), NOW)["state"] == "ok"
    stale = hj.assess(_entry(hours_ago=50), NOW)
    assert stale["state"] == "stale" and stale["age_hours"] == 50.0
    assert hj.assess(_entry(hours_ago=10, status="failure"), NOW)["state"] == "failing"
    assert hj.assess(_entry(hours_ago=60, status="failure"), NOW)["state"] == "stale"
    assert hj.assess(hj.blank_entry("harvest"), NOW)["state"] == "unknown"
    assert hj.assess(_entry(key="mirror-context", hours_ago=2000), NOW)["state"] == "on demand"


def test_a_job_that_only_ever_failed_is_stale():
    rec = hj.make_record(_run(status="failure"), now=NOW)
    assert hj.assess(hj.index_entry(rec), NOW)["state"] == "stale"


# ── the table and the index ─────────────────────────────────────────────────────────────────────────────────


def test_every_job_names_a_workflow_that_exists_and_every_scheduled_workflow_is_a_job():
    for j in hj.JOBS.values():
        assert (WORKFLOWS / j.workflow).exists(), j.workflow
        if j.stale_after_hours is None:
            assert j.key in ("mirror-context", "repair")  # dispatched, or only with a model key
        else:
            assert j.stale_after_hours > j.cadence_hours
    scheduled = {p.name for p in WORKFLOWS.glob("*.yml") if re.search(r"^\s+schedule:", p.read_text(), re.M)}
    assert scheduled <= {j.workflow for j in hj.JOBS.values()}


def test_the_layers_are_legend_rows_the_explorer_has():
    legend = (ROOT / "explorer" / "src" / "map-legend-core.js").read_text()
    order = re.search(r"ROW_ORDER = \[(.*?)\]", legend, re.S).group(1)
    rows = set(re.findall(r'"([a-z-]+)"', order))
    for j in hj.JOBS.values():
        assert set(j.layers) <= rows, j.key


def test_the_index_lists_every_job_with_the_cadence_from_the_table():
    rec = hj.make_record(_run(), now=NOW)
    rec["stale_after_hours"] = 999  # an old record's cadence is overridden by the table
    index = hj.build_index({"forecast-archive": rec, "bulletin": None}, now=NOW)
    keys = [e["job"] for e in index["jobs"]]
    assert keys == list(hj.JOBS)
    fa = index["jobs"][keys.index("forecast-archive")]
    assert fa["stale_after_hours"] == 48 and "history" not in fa and fa["last_success"]
    assert index["jobs"][keys.index("bulletin")]["last_run"] is None


def test_write_puts_the_record_and_the_index_under_health_jobs(tmp_path):
    rec = hj.make_record(_run(), now=NOW)
    path = hj.write(tmp_path, rec, hj.build_index({"forecast-archive": rec}, now=NOW))
    assert path == tmp_path / "health" / "jobs" / "forecast-archive.json"
    assert sorted(p.name for p in (tmp_path / "health" / "jobs").iterdir()) == ["forecast-archive.json", "index.json"]
    assert [p for p in tmp_path.rglob("*") if p.is_file() and "health/jobs" not in p.as_posix()] == []


def test_publish_uploads_only_the_record_and_the_index(tmp_path, monkeypatch):
    seen = {}

    def fake(folder, repo_id, *, token=None, allow_patterns=None, commit_message=None, **kw):
        seen["files"] = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
        seen["patterns"] = allow_patterns
        return "commit"

    import aquascope.archive.publish as pub

    monkeypatch.setattr(pub, "publish_folder", fake)
    rec = hj.make_record(_run(), now=NOW)
    hj.write(tmp_path, rec, hj.build_index({"forecast-archive": rec}, now=NOW))
    (tmp_path / "health" / "jobs" / "bulletin.json").write_text("{}")  # another job's file is never sent
    assert hj.publish(tmp_path, "forecast-archive") == "commit"
    assert seen["files"] == ["health/jobs/forecast-archive.json", "health/jobs/index.json"]
    assert seen["patterns"] == ["health/jobs/forecast-archive.json", "health/jobs/index.json"]


# ── what each job wrote ─────────────────────────────────────────────────────────────────────────────────────


def _dump(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_summaries_read_each_jobs_own_manifest(tmp_path):
    _dump(tmp_path / "health.json", {"run_at": "2026-10-05T10:40:00Z", "n_stations": 62000, "n_ok": 30,
                                     "sources": [{}] * 32})
    _dump(tmp_path / "obs" / "refresh_status.json", {"status": "partial"})
    s = hj.summarise("harvest", tmp_path)
    assert s["counts"] == {"stations": 62000, "sources_ok": 30, "sources": 32}
    assert s["data_date"] == "2026-10-05" and s["notes"] == ["observation refresh: partial"]

    _dump(tmp_path / "forecasts" / "manifest.json", {"status": {"n": 240}, "issues": [
        {"date": "2026-10-11", "n_gauges": 1, "n_rows": 2}, {"date": "2026-10-12", "n_gauges": 238, "n_rows": 9000}]})
    s = hj.summarise("forecast-archive", tmp_path)
    assert s["counts"] == {"gauges": 238, "rows": 9000, "status_gauges": 240} and s["data_date"] == "2026-10-12"

    _dump(tmp_path / "forecasts" / "warnings" / "manifest.json", {"n": 12, "checked": 9000, "issue_date": "2026-10-11",
                                                                  "smoke": True})
    s = hj.summarise("flood-warnings", tmp_path)
    assert s["counts"] == {"reaches": 12, "checked": 9000} and s["smoke"] is True

    _dump(tmp_path / "waterbalance" / "manifest.json", {"months": 430, "basins": 1300, "computed": ["2026-09"],
                                                        "last_month": "2026-09"})
    s = hj.summarise("water-balance", tmp_path)
    assert s["counts"] == {"months": 430, "basins": 1300, "computed": 1} and s["data_date"] == "2026-09"

    _dump(tmp_path / "bulletins" / "build.json", {"month": "2026-09"})
    _dump(tmp_path / "bulletins" / "2026-09" / "bulletin.json", {"coverage": {"classed": 1821, "considered": 3000}})
    assert hj.summarise("bulletin", tmp_path)["counts"] == {"classed": 1821, "considered": 3000}


def test_a_run_that_wrote_nothing_still_summarises(tmp_path):
    for key in hj.JOBS:
        s = hj.summarise(key, tmp_path / "missing")
        assert s["counts"] == {} and s.get("data_date") is None


# ── the issue ───────────────────────────────────────────────────────────────────────────────────────────────


def test_issue_actions():
    ok = hj.make_record(_run(), now=NOW)
    bad = hj.make_record(_run(status="failure"), now=NOW)
    assert hj.issue_action(bad, None) == "open"
    assert hj.issue_action(bad, 7) == "comment"
    assert hj.issue_action(ok, 7) == "close"
    assert hj.issue_action(ok, None) is None


class FakeGh:
    def __init__(self, open_issues=()):
        self.calls: list[tuple[str, ...]] = []
        self.open = list(open_issues)

    def __call__(self, *args, dry_run=False):
        self.calls.append(args)
        if args[:2] == ("issue", "list"):
            return json.dumps(self.open)
        return ""


def test_a_failed_run_opens_one_issue_labelled_job_health():
    gh = FakeGh()
    rec = hj.make_record(_run(status="success", requires={"publish": "skipped"}, run_url="https://run/1"), now=NOW)
    assert hj.sync_issue(rec, "o/r", gh=gh) == "open"
    create = next(c for c in gh.calls if c[:2] == ("issue", "create"))
    assert hj.issue_title("forecast-archive") in create and hj.LABEL in create
    body = create[create.index("--body") + 1]
    assert "publish did not run on a scheduled run" in body and "https://run/1" in body


def test_a_second_failure_comments_and_a_success_closes():
    gh = FakeGh([{"number": 41, "title": hj.issue_title("forecast-archive")},
                 {"number": 40, "title": hj.issue_title("bulletin")}])
    bad = hj.make_record(_run(status="failure"), now=NOW)
    assert hj.sync_issue(bad, "o/r", gh=gh) == "comment"
    assert any(c[:3] == ("issue", "comment", "41") for c in gh.calls)
    ok = hj.make_record(_run(), bad, now=NOW)
    assert hj.sync_issue(ok, "o/r", gh=gh) == "close"
    assert any(c[:3] == ("issue", "close", "41") for c in gh.calls)


# ── the command line, through the runner the workflows call ─────────────────────────────────────────────────


def test_the_runner_reports_offline_and_writes_the_summary(tmp_path):
    summary = tmp_path / "summary.md"
    _dump(tmp_path / "build" / "forecasts" / "manifest.json", {"issues": [{"date": "2026-10-12", "n_gauges": 3}]})
    out = subprocess.run(
        [sys.executable, "-I", str(ROOT / ".github" / "scripts" / "job_health.py"), "report", "forecast-archive",
         "--root", str(tmp_path / "build"), "--out", str(tmp_path / "h"), "--status", "success",
         "--event", "schedule", "--require", "publish=skipped", "--started", "2026-10-12T10:00:00Z"],
        capture_output=True, text=True, env={"GITHUB_STEP_SUMMARY": str(summary), "PATH": "/usr/bin:/bin"},
        cwd=tmp_path, check=True)
    assert "failure: publish did not run" in out.stdout
    rec = json.loads((tmp_path / "h" / "health" / "jobs" / "forecast-archive.json").read_text())
    assert rec["last_run"]["counts"] == {"gauges": 3}
    assert "Job health: forecast-archive" in summary.read_text()


def test_status_lines_say_when_nothing_is_published():
    assert hj.status_lines(None) == ["no job health published yet"]
    lines = hj.status_lines(hj.build_index({"forecast-archive": hj.make_record(_run(), now=NOW)}, now=NOW), NOW)
    assert len(lines) == len(hj.JOBS) and lines[1].startswith("forecast-archive")


# ── the workflows ───────────────────────────────────────────────────────────────────────────────────────────


def _health_runs(text: str) -> list[str]:
    return re.findall(r"job_health\.py report ([a-z-]+)", text)


@pytest.mark.parametrize("key", list(hj.JOBS))
def test_every_job_is_reported_by_its_workflow_whatever_the_outcome(key):
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load((WORKFLOWS / hj.JOBS[key].workflow).read_text())
    found = [(job, step) for job in doc["jobs"].values() for step in job.get("steps", [])
             if f"report {key} " in (step.get("run") or "")]
    assert found, f"{hj.JOBS[key].workflow} does not report {key}"
    job, step = found[0]
    cond = str(step.get("if") or job.get("if") or "")
    assert cond.startswith("always()"), f"{key}'s health step must run when the job failed"
    assert step.get("continue-on-error") or job.get("continue-on-error"), f"{key}'s health step must not turn it red"


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_a_workflow_that_files_job_health_issues_may_write_issues(path):
    text = path.read_text()
    if "job_health.py report" in text and "--issue false" not in text:
        assert re.search(r"^\s+issues: write", text, re.M), path.name


def test_the_harvest_fails_a_scheduled_run_that_published_nothing():
    text = (WORKFLOWS / "harvest.yml").read_text()
    step = text[text.index("Fail when a scheduled run published nothing"):]
    assert "steps.publish.outcome != 'success'" in step.split("\n")[1]
    assert "github.event_name == 'schedule'" in step.split("\n")[1]


# ── reading the published records: absent is not the same as unreadable ────────────────────────────────────


def _urlopen(code=None, body=b"{}"):
    import io
    import urllib.error

    def fake(url, timeout=None):
        if code:
            raise urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(b""))
        return io.BytesIO(body)

    return fake


def test_a_missing_record_is_none_and_an_unreadable_hub_raises(monkeypatch):
    monkeypatch.setattr(hj.urllib.request, "urlopen", _urlopen(404))
    assert hj.read_published("harvest") is None
    monkeypatch.setattr(hj.urllib.request, "urlopen", _urlopen(503))
    with pytest.raises(hj.ReadError):
        hj.read_published("harvest")
    monkeypatch.setattr(hj.urllib.request, "urlopen", _urlopen(body=b"not json"))
    assert hj.read_published("harvest") is None


def _gh_calls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="[]" if cmd[1:3] == ["issue", "list"] else "", stderr="")

    monkeypatch.setattr(hj.subprocess, "run", fake_run)
    return calls


def _report(tmp_path, *extra):
    return hj.main(["report", "forecast-archive", "--out", str(tmp_path), "--status", "success", "--event",
                    "schedule", "--require", "publish=failure", "--publish", "true", "--issue", "true",
                    "--gh-repo", "o/r", *extra])


def test_a_scheduled_run_that_failed_opens_its_issue_through_the_command_line(tmp_path, monkeypatch):
    calls = _gh_calls(monkeypatch)
    published = []
    monkeypatch.setattr(hj, "read_published", lambda key, repo=hj.DEFAULT_REPO: None)
    monkeypatch.setattr(hj, "publish", lambda out, key, **kw: published.append(key) or "commit")
    assert _report(tmp_path) == 0
    created = [c for c in calls if c[1:3] == ["issue", "create"]]
    assert len(created) == 1 and hj.LABEL in created[0] and hj.issue_title("forecast-archive") in created[0]
    assert published == ["forecast-archive"]


def test_a_hub_hiccup_never_publishes_over_the_record_but_still_files_the_issue(tmp_path, monkeypatch, capsys):
    calls = _gh_calls(monkeypatch)
    published = []

    def flaky(key, repo=hj.DEFAULT_REPO):
        if key == "forecast-archive":
            raise hj.ReadError("HTTP 503")
        return None

    monkeypatch.setattr(hj, "read_published", flaky)
    monkeypatch.setattr(hj, "publish", lambda out, key, **kw: published.append(key) or "commit")
    assert _report(tmp_path) == 0
    assert published == []
    assert "could not be read" in capsys.readouterr().out
    assert [c for c in calls if c[1:3] == ["issue", "create"]]
    assert (tmp_path / "health" / "jobs" / "forecast-archive.json").exists()
