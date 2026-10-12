"""Health records for the scheduled jobs (#593, #449).

Every scheduled workflow ends with one step, run whether the job passed or not, that writes a small record of
the run into the Archive and keeps one GitHub issue per job:

    health/jobs/<job>.json   the last run, the last success, the last failure, durations, row or file counts,
                             the date of the data, and the recent runs (for the green streak)
    health/jobs/index.json   every job in :data:`JOBS` with its expected cadence and its newest state, which
                             is what the Explorer reads (the legend's stale-data notes, Tools > Data status)

A job writes only under ``health/jobs/`` and only its own file plus the index. A failed scheduled run opens
(or comments on) the job's ``job-health`` issue; the next successful one closes it. A run counts as failed when
the job failed, was cancelled or timed out, or when a step it promises on a scheduled run (its publish) did not
succeed, which is the silent case #449 described: a harvest that ran but published nothing.

The expected cadence of every job lives in one table, :data:`JOBS`, and travels with the index, so the Explorer
compares a record's age with a number it is given rather than one of its own.

This file imports only the standard library at module level, so a workflow can still report a run in which the
install step failed (``.github/scripts/job_health.py`` loads it by path). Publishing needs the ``archive`` extra.

    python -m aquascope.archive.health_jobs status              the published index, one line per job
    python -m aquascope.archive.health_jobs report <job> --root build --status success --publish true ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FOLDER = "health/jobs"
INDEX = "index.json"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
#: Runs kept in a record's ``history``: enough to show two weeks of a daily job and a year of a monthly one.
HISTORY = 60
#: The label of the per-job issue.
LABEL = "job-health"
ISSUE_TITLE = "job-health: the {key} job's scheduled run failed"
#: Events that count as an unattended run: the cron, and a workflow chained to one (repair after the harvest).
UNATTENDED = ("schedule", "workflow_run")

DAY = 24
WEEK = 7 * DAY
MONTH = 31 * DAY


@dataclass(frozen=True)
class Job:
    """One scheduled job. ``stale_after_hours`` is how old its last success may be before the Explorer says
    the data is older than expected: the cadence plus the slack GitHub's cron delays need (scheduled runs have
    started up to seven hours late). ``None`` for a job that only runs when dispatched."""

    key: str
    label: str
    workflow: str
    cadence: str
    cadence_hours: int | None
    stale_after_hours: int | None
    #: The Explorer legend rows (map-legend ids) whose data this job writes.
    layers: tuple[str, ...] = ()
    #: The Archive prefixes the job publishes (``()`` for one that publishes nothing there).
    writes: tuple[str, ...] = ()
    what: str = ""


#: Every scheduled job, its cadence and what depends on it: the one table the Explorer's notes come from.
JOBS: dict[str, Job] = {j.key: j for j in (
    Job("harvest", "Weekly harvest", "harvest.yml", "weekly (Mondays)", WEEK, WEEK + 2 * DAY,
        layers=("gauges",), writes=("stations.parquet", "obs/", "basins/"),
        what="the station catalogue and the daily observations"),
    Job("forecast-archive", "Forecast archive", "forecast-archive.yml", "daily", DAY, 2 * DAY,
        layers=("forecast-points",), writes=("forecasts/", "feeds/"),
        what="each live gauge's status today and the forecasts as issued, and the per-gauge feeds"),
    Job("flood-warnings", "Floods ahead", "flood-warnings.yml", "daily", DAY, 2 * DAY,
        layers=("floods-ahead",), writes=("forecasts/warnings/",),
        what="the river reaches forecast to reach their 2-year flow"),
    Job("flow-status", "Flow against normal", "flood-warnings.yml", "daily", DAY, 2 * DAY,
        layers=("rivers",), writes=("flowstatus/",),
        what="every big reach's first-day flow against its normal for the month"),
    Job("scout", "Scout", "flood-warnings.yml", "daily", DAY, 2 * DAY, writes=("scout/",),
        what="what stands out on the map today"),
    Job("model-skill", "Model skill", "model-skill.yml", "monthly (the 3rd)", MONTH, MONTH + 4 * DAY,
        writes=("skill/",), what="every global model scored at the Archive gauges"),
    Job("bulletin", "Monthly bulletin", "bulletin.yml", "monthly (the 3rd)", MONTH, MONTH + 4 * DAY,
        writes=("bulletins/",), what="last month's state of the rivers"),
    Job("water-balance", "Water balance", "water-balance.yml", "monthly (the 9th)", MONTH, MONTH + 4 * DAY,
        layers=("water-balance",), writes=("waterbalance/",), what="the subbasins' monthly water balance"),
    Job("mirror-context", "Place-context mirrors", "mirror-context.yml", "when dispatched", None, None,
        layers=("floods-past",), writes=("context/",), what="flood events, dams and rain gauges, mirrored"),
    Job("showcase", "Showcase recordings", "showcase.yml", "weekly (Mondays)", WEEK, WEEK + 2 * DAY,
        what="the Analyst's worked examples, re-recorded as a pull request"),
    # Repair only runs when a model key is set, so a week without a record is not stale; a failed run still
    # opens its issue.
    Job("repair", "Collector repair", "repair.yml", "after each scheduled harvest, when a model key is set", WEEK,
        None, what="model-written patches for the collectors the harvest found failing"),
    Job("camels-benchmark", "CAMELS benchmark", "camels-benchmark.yml", "daily", DAY, 2 * DAY,
        what="the offline benchmark suite (a check, no data)"),
    Job("explorer-live-smoke", "Explorer live check", "explorer-live-smoke.yml", "daily", DAY, 2 * DAY,
        what="a browser records and exports US, UK and French gauges on the live Explorer (a check, no data)"),
    Job("pyodide-smoke", "Pyodide check", "pyodide-smoke.yml", "weekly (Mondays)", WEEK, WEEK + 2 * DAY,
        what="the wheel installs and runs under Pyodide (a check, no data)"),
)}


def job(key: str) -> Job:
    try:
        return JOBS[key]
    except KeyError:
        raise ValueError(f"unknown job {key!r}; the jobs are {', '.join(JOBS)}") from None


# ── what a run wrote: counts and the date of its data, read from each job's own manifest ──────────────────────


def _read(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _day(value: Any) -> str | None:
    text = str(value or "")[:10]
    return text or None


def _harvest(root: Path) -> dict[str, Any]:
    h = _read(root / "health.json")
    obs = _read(root / "obs" / "refresh_status.json")
    counts = {"stations": h.get("n_stations"), "sources_ok": h.get("n_ok"),
              "sources": len(h.get("sources") or []) or None}
    bundles = sorted(root.glob("obs/*/*.parquet"))
    if bundles:
        counts["bundles"] = len(bundles)
    return {"counts": counts, "data_date": _day(h.get("run_at")),
            "notes": [f"observation refresh: {obs['status']}"] if obs.get("status") else []}


def _forecasts(root: Path) -> dict[str, Any]:
    m = _read(root / "forecasts" / "manifest.json")
    issue = (m.get("issues") or [{}])[-1]
    feeds = sum(1 for _ in (root / "feeds").glob("*/*.xml")) if (root / "feeds").is_dir() else 0
    return {"counts": {"gauges": issue.get("n_gauges"), "rows": issue.get("n_rows"),
                       "status_gauges": (m.get("status") or {}).get("n"), "feeds": feeds or None},
            "data_date": _day(issue.get("date"))}


def _warnings(root: Path) -> dict[str, Any]:
    m = _read(root / "forecasts" / "warnings" / "manifest.json")
    return {"counts": {"reaches": m.get("n"), "checked": m.get("checked")}, "data_date": _day(m.get("issue_date")),
            "smoke": bool(m.get("smoke"))}


def _flowstatus(root: Path) -> dict[str, Any]:
    m = _read(root / "flowstatus" / "manifest.json")
    return {"counts": {"classed": m.get("n"), "reaches": m.get("reaches")}, "data_date": _day(m.get("issue_date")),
            "smoke": bool(m.get("smoke"))}


def _scout(root: Path) -> dict[str, Any]:
    m = _read(root / "scout" / "latest.json")
    return {"counts": {"findings": len(m["findings"]) if isinstance(m.get("findings"), list) else None},
            "data_date": _day(m.get("date"))}


def _skill(root: Path) -> dict[str, Any]:
    m = _read(root / "manifest.json")
    return {"counts": {"rows": m.get("n_rows"), "gauges": m.get("n_gauges"), "scored": m.get("scored_this_run")},
            "data_date": _day(m.get("built_at")), "smoke": bool(m.get("smoke"))}


def _bulletin(root: Path) -> dict[str, Any]:
    build = _read(root / "bulletins" / "build.json")
    month = build.get("month")
    b = _read(root / "bulletins" / str(month) / "bulletin.json") if month else {}
    cov = b.get("coverage") or {}
    return {"counts": {"classed": cov.get("classed"), "considered": cov.get("considered")},
            "data_date": str(month) if month else None, "smoke": bool(build.get("smoke"))}


def _waterbalance(root: Path) -> dict[str, Any]:
    m = _read(root / "waterbalance" / "manifest.json")
    return {"counts": {"months": m.get("months"), "basins": m.get("basins"),
                       "computed": len(m["computed"]) if isinstance(m.get("computed"), list) else None},
            "data_date": m.get("last_month"), "smoke": bool(m.get("smoke"))}


def _context(root: Path) -> dict[str, Any]:
    m = _read(root / "context" / "manifest.json")
    sets = m.get("datasets") or {}
    rows = [v.get("rows") for v in sets.values() if isinstance(v, dict) and isinstance(v.get("rows"), int)]
    return {"counts": {"datasets": len(sets) or None, "rows": sum(rows) if rows else None},
            "data_date": _day(m.get("updated"))}


def _showcase(root: Path) -> dict[str, Any]:
    n = sum(1 for _ in root.glob("*.json")) if root.is_dir() else 0
    return {"counts": {"recordings": n or None}}


#: How each job's own output says what it wrote. ``root`` is the folder the workflow passes (its --out).
SUMMARISE: dict[str, Callable[[Path], dict[str, Any]]] = {
    "harvest": _harvest, "forecast-archive": _forecasts, "flood-warnings": _warnings, "flow-status": _flowstatus,
    "scout": _scout, "model-skill": _skill, "bulletin": _bulletin, "water-balance": _waterbalance,
    "mirror-context": _context, "showcase": _showcase,
}


def summarise(key: str, root: str | Path | None) -> dict[str, Any]:
    """Counts, the data's date and notes for ``key``'s run, from the files it wrote under ``root``. Missing or
    unreadable files give empty counts, never an error: the record is written for a failed run too."""
    job(key)
    fn = SUMMARISE.get(key)
    if fn is None or root is None:
        return {"counts": {}, "data_date": None}
    out = fn(Path(root))
    out["counts"] = {k: v for k, v in (out.get("counts") or {}).items() if v is not None}
    return out


# ── the record ────────────────────────────────────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """An ISO time (a trailing ``Z`` included, which Python 3.10 cannot parse) or epoch seconds, as UTC."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if text.replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def judge(status: str, requires: dict[str, str] | None = None, *, event: str = "") -> tuple[str, list[str]]:
    """``("success" | "failure" | "skipped", reasons)`` for a run. ``status`` is the job's (or a step's)
    status as GitHub gives it: success, failure, cancelled or skipped. ``requires`` maps a promised step to its
    outcome; on an unattended run a promised step that did not succeed fails the run (#449: a scheduled harvest
    that published nothing used to pass)."""
    s = (status or "").strip().lower()
    if s == "skipped":
        return "skipped", ["the step did not run"]
    reasons: list[str] = []
    if s == "cancelled":
        reasons.append("cancelled or timed out")
    elif s != "success":
        reasons.append(f"the job ended with status {s or 'unknown'}")
    if event in UNATTENDED:
        for name, outcome in (requires or {}).items():
            o = (outcome or "").strip().lower() or "skipped"
            if o != "success":
                reasons.append(f"{name} {'did not run' if o == 'skipped' else o} on a scheduled run")
    return ("failure" if reasons else "success"), reasons


@dataclass
class Run:
    """One run of a job, as the workflow reports it."""

    key: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    event: str = ""
    run_url: str = ""
    requires: dict[str, str] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    data_date: str | None = None
    notes: list[str] = field(default_factory=list)
    smoke: bool = False


def green_streak(history: list[dict[str, Any]]) -> int:
    """Unattended runs in a row, newest first, that succeeded (the #593 two-week bar)."""
    n = 0
    for h in reversed(history):
        if h.get("event") not in UNATTENDED:
            continue
        if h.get("status") != "success":
            break
        n += 1
    return n


def make_record(run: Run, previous: dict[str, Any] | None = None, *, now: datetime | None = None) -> dict[str, Any]:
    """The job's record after ``run``, carrying the last success, the last failure and the history over from
    ``previous`` (the published record, or None for the first)."""
    j = job(run.key)
    now = now or _now()
    status, reasons = judge(run.status, run.requires, event=run.event)
    if status == "skipped":
        raise ValueError(f"{run.key}: the run was skipped, so there is nothing to record")
    finished = parse_time(run.finished_at) or now
    started = parse_time(run.started_at)
    duration = round((finished - started).total_seconds()) if started and finished >= started else None
    this = {"status": status, "started_at": _iso(started) if started else None, "finished_at": _iso(finished),
            "duration_s": duration, "event": run.event or None, "run_url": run.run_url or None,
            "counts": dict(run.counts), "data_date": run.data_date, "reasons": reasons, "notes": list(run.notes)}
    prev = previous if isinstance(previous, dict) and previous.get("job") == run.key else {}
    history = [h for h in (prev.get("history") or []) if isinstance(h, dict)]
    history.append({"finished_at": this["finished_at"], "status": status, "duration_s": duration,
                    "event": this["event"], "data_date": run.data_date})
    history = history[-HISTORY:]
    record = {
        "job": j.key, "label": j.label, "workflow": j.workflow, "what": j.what,
        "cadence": j.cadence, "cadence_hours": j.cadence_hours, "stale_after_hours": j.stale_after_hours,
        "layers": list(j.layers), "writes": list(j.writes),
        "updated_at": _iso(now),
        "last_run": this,
        "last_success": this if status == "success" else prev.get("last_success"),
        "last_failure": this if status == "failure" else prev.get("last_failure"),
        "consecutive_failures": 0 if status == "success" else int(prev.get("consecutive_failures") or 0) + 1,
        "history": history,
    }
    record["green_streak"] = green_streak(history)
    return record


# ── reading a record: the state the Explorer shows (explorer/src/job-health-core.js does the same) ─────────────


def assess(entry: dict[str, Any] | None, now: datetime | None = None) -> dict[str, Any]:
    """How a job reads now: ``state`` is "ok", "stale" (its last success is older than ``stale_after_hours``),
    "failing" (its last run failed, the data still within its cadence), "on demand" (no cadence) or "unknown"
    (no run recorded). ``age_hours`` is the age of the last success."""
    now = now or _now()
    entry = entry or {}
    ok = (entry.get("last_success") or {}).get("finished_at")
    last = entry.get("last_run") or {}
    t = parse_time(ok)
    age = round((now - t).total_seconds() / 3600, 1) if t else None
    limit = entry.get("stale_after_hours")
    if not last and not ok:
        state = "unknown"
    elif limit is not None and (age is None or age > float(limit)):
        state = "stale"
    elif last.get("status") == "failure":
        state = "failing"
    elif limit is None:
        state = "on demand"
    else:
        state = "ok"
    return {"job": entry.get("job"), "state": state, "age_hours": age,
            "data_date": (entry.get("last_success") or {}).get("data_date"),
            "last_status": last.get("status"), "green_streak": entry.get("green_streak", 0)}


def index_entry(record: dict[str, Any]) -> dict[str, Any]:
    """A record without its history, as the index carries it."""
    return {k: v for k, v in record.items() if k != "history"}


def blank_entry(key: str) -> dict[str, Any]:
    j = job(key)
    return {"job": j.key, "label": j.label, "workflow": j.workflow, "what": j.what, "cadence": j.cadence,
            "cadence_hours": j.cadence_hours, "stale_after_hours": j.stale_after_hours, "layers": list(j.layers),
            "writes": list(j.writes), "updated_at": None, "last_run": None, "last_success": None,
            "last_failure": None, "consecutive_failures": 0, "green_streak": 0}


def build_index(records: dict[str, dict[str, Any] | None], *, now: datetime | None = None) -> dict[str, Any]:
    """``health/jobs/index.json``: every job in :data:`JOBS`, in table order, with its newest record (or a blank
    one) and the cadence the Explorer compares against. The cadence always comes from the table, so a change to
    it reaches the Explorer on the next run of any job."""
    jobs = []
    for key, j in JOBS.items():
        rec = records.get(key)
        entry = index_entry(rec) if isinstance(rec, dict) and rec.get("job") == key else blank_entry(key)
        entry.update({"label": j.label, "workflow": j.workflow, "what": j.what, "cadence": j.cadence,
                      "cadence_hours": j.cadence_hours, "stale_after_hours": j.stale_after_hours,
                      "layers": list(j.layers), "writes": list(j.writes)})
        jobs.append(entry)
    return {"about": "Health of AquaScope's scheduled jobs (#593): each job's last run, last success and last "
                     "failure. A job is stale when its last success is older than stale_after_hours.",
            "updated_at": _iso(now or _now()), "jobs": jobs}


# ── files: local, published ───────────────────────────────────────────────────────────────────────────────────


def published_url(path: str, repo_id: str = DEFAULT_REPO) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


class ReadError(OSError):
    """A published record could not be read for a reason other than its absence (a timeout, a 5xx)."""


def _fetch_json(url: str, timeout: float = 30) -> dict[str, Any] | None:
    """The JSON at ``url``; None when it is not there (404) or is not a JSON object. Any other failure raises
    :class:`ReadError`, so a passing Hub hiccup is never mistaken for "no record yet" (which would publish a fresh
    record over the job's history)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - a fixed https host
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ReadError(f"{url}: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ReadError(f"{url}: {exc}") from exc
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        logger.info("%s is not JSON: %s", url, exc)
        return None
    return doc if isinstance(doc, dict) else None


def read_published(key: str, repo_id: str = DEFAULT_REPO) -> dict[str, Any] | None:
    """The published ``health/jobs/<key>.json``, or None before the job has published one. Raises
    :class:`ReadError` when the Hub could not be read."""
    return _fetch_json(published_url(f"{FOLDER}/{key}.json", repo_id))


def read_published_index(repo_id: str = DEFAULT_REPO) -> dict[str, Any] | None:
    return _fetch_json(published_url(f"{FOLDER}/{INDEX}", repo_id))


def write(out: str | Path, record: dict[str, Any], index: dict[str, Any]) -> Path:
    """Write ``<out>/health/jobs/<job>.json`` and ``<out>/health/jobs/index.json``; returns the record's path."""
    root = Path(out) / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record['job']}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (root / INDEX).write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return path


def _publish_with_hub(folder: Path, repo_id: str, *, token: str | None = None, allow_patterns: list[str],
                      commit_message: str) -> str:
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise PermissionError("No Hugging Face token: publishing needs HF_TOKEN")
    info = HfApi(token=token).upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
                                            allow_patterns=allow_patterns, commit_message=commit_message)
    return str(getattr(info, "commit_url", None) or info)


def publish(out: str | Path, key: str, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``health/jobs/<key>.json`` and ``health/jobs/index.json``, and nothing else, to the Archive dataset
    (needs HF_TOKEN with write access and huggingface_hub). A check job that never installs aquascope (the
    Explorer's browser check) still publishes, through huggingface_hub directly."""
    try:
        from aquascope.archive.publish import publish_folder
    except ImportError:
        publish_folder = _publish_with_hub

    src = Path(out) / FOLDER
    files = [src / f"{key}.json", src / INDEX]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing; write the record first")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        stage.mkdir(parents=True)
        for f in files:
            shutil.copy2(f, stage / f.name)
        return str(publish_folder(Path(tmp), repo_id, token=token,
                                  allow_patterns=[f"{FOLDER}/{key}.json", f"{FOLDER}/{INDEX}"],
                                  commit_message=f"health: {key}"))


# ── the job-health issue ──────────────────────────────────────────────────────────────────────────────────────


def issue_title(key: str) -> str:
    return ISSUE_TITLE.format(key=key)


def issue_action(record: dict[str, Any], open_issue: int | None) -> str | None:
    """"open", "comment", "close" or None for the record's newest run and the job's open issue, if any."""
    status = (record.get("last_run") or {}).get("status")
    if status == "failure":
        return "comment" if open_issue else "open"
    if status == "success" and open_issue:
        return "close"
    return None


def issue_body(record: dict[str, Any]) -> str:
    j = job(record["job"])
    last = record.get("last_run") or {}
    ok = record.get("last_success") or {}
    lines = [
        f"The scheduled run of **{j.label}** (`.github/workflows/{j.workflow}`) failed on "
        f"{last.get('finished_at') or '?'}.",
        "",
        *[f"- {r}" for r in last.get("reasons") or []],
        "",
        f"It writes {j.what}" + (f" (Archive: {', '.join(f'`{w}`' for w in j.writes)})." if j.writes else "."),
        f"Expected cadence: {j.cadence}. Failed runs in a row: {record.get('consecutive_failures', 1)}.",
        f"Last success: {ok.get('finished_at') or 'none recorded'}"
        + (f", data of {ok['data_date']}" if ok.get("data_date") else "") + ".",
        "",
        (f"Run: {last['run_url']}" if last.get("run_url") else ""),
        f"Record: {published_url(f'{FOLDER}/{j.key}.json')}",
        "",
        "_Opened automatically by the job's health step (#593); it closes itself on the next successful run._",
    ]
    return "\n".join(lines).replace("\n\n\n", "\n\n")


def _gh(*args: str, dry_run: bool = False) -> str:
    if dry_run:
        print("DRY RUN: gh", " ".join(args))
        return ""
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def find_open_issue(repo: str, key: str, *, gh: Callable[..., str] = _gh, dry_run: bool = False) -> int | None:
    if dry_run:
        return None
    raw = gh("issue", "list", "--repo", repo, "--label", LABEL, "--state", "open", "--limit", "100",
             "--json", "number,title")
    for item in json.loads(raw or "[]"):
        if item.get("title") == issue_title(key):
            return int(item["number"])
    return None


def sync_issue(record: dict[str, Any], repo: str, *, gh: Callable[..., str] = _gh, dry_run: bool = False) -> str:
    """Open, comment on or close the job's ``job-health`` issue for its newest run; returns what it did."""
    key = record["job"]
    try:
        gh("label", "create", LABEL, "--repo", repo, "--color", "d93f0b", "--force",
           "--description", "A scheduled job failed its run (#593)", dry_run=dry_run)
    except subprocess.CalledProcessError:
        pass  # exists already, or no permission: the issue still carries the label if it exists
    existing = find_open_issue(repo, key, gh=gh, dry_run=dry_run)
    action = issue_action(record, existing)
    last = record.get("last_run") or {}
    if action == "open":
        gh("issue", "create", "--repo", repo, "--title", issue_title(key), "--body", issue_body(record),
           "--label", LABEL, dry_run=dry_run)
    elif action == "comment":
        reasons = "; ".join(last.get("reasons") or []) or "failed"
        gh("issue", "comment", str(existing), "--repo", repo, "--body",
           f"Still failing on {last.get('finished_at')}: {reasons}. Failed runs in a row: "
           f"{record.get('consecutive_failures')}." + (f"\n\nRun: {last['run_url']}" if last.get("run_url") else ""),
           dry_run=dry_run)
    elif action == "close":
        gh("issue", "close", str(existing), "--repo", repo, "--comment",
           f"Healthy again: the run of {last.get('finished_at')} succeeded"
           + (f" ({last['run_url']})" if last.get("run_url") else "") + ". Closing automatically.",
           dry_run=dry_run)
    return action or "nothing"


# ── the command line ──────────────────────────────────────────────────────────────────────────────────────────


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _requires(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        name, _, outcome = item.partition("=")
        if name.strip():
            out[name.strip()] = outcome.strip()
    return out


def report(args: argparse.Namespace) -> int:
    """Record a run, write it, and (each independently, so one failing never hides another) publish it and
    keep the issue. Always exits 0: the job's own steps decide whether the run is red."""
    summary = summarise(args.job, args.root)
    run = Run(key=args.job, status=args.status, started_at=args.started, finished_at=args.finished, event=args.event,
              run_url=args.run_url, requires=_requires(args.require), counts=summary.get("counts") or {},
              data_date=summary.get("data_date"), notes=summary.get("notes") or [], smoke=bool(summary.get("smoke")))
    if judge(run.status)[0] == "skipped":
        print(f"[{args.job}] skipped, nothing recorded")
        return 0
    online = _truthy(args.publish) or _truthy(args.issue)
    unread: list[str] = []

    def published(key: str) -> dict[str, Any] | None:
        if not online:
            return None
        try:
            return read_published(key, args.repo)
        except ReadError as exc:
            unread.append(key)
            print(f"::warning::[{args.job}] could not read the published record of {key}: {exc}")
            return None

    previous = published(args.job)
    record = make_record(run, previous)
    others = {k: published(k) for k in JOBS if k != args.job}
    index = build_index({**others, args.job: record})
    path = write(args.out, record, index)
    last = record["last_run"]
    print(f"[{args.job}] {last['status']}" + (f": {'; '.join(last['reasons'])}" if last["reasons"] else "")
          + f" -> {path}")
    if _truthy(args.publish):
        if run.smoke:
            print(f"[{args.job}] a smoke run's record is never published")
        elif unread:
            # Publishing now would replace a record we could not read (its history, its last success) or blank
            # another job's row in the index. The next run publishes; this one stays in the job summary.
            print(f"::warning::[{args.job}] the health record was not published: the published records of "
                  f"{', '.join(unread)} could not be read")
        else:
            try:
                print(f"[{args.job}] published: {publish(args.out, args.job, repo_id=args.repo)}")
            except Exception as exc:  # noqa: BLE001 - reported, then the issue step still runs
                print(f"::warning::[{args.job}] the health record was not published: {exc}")
    if _truthy(args.issue):
        try:
            print(f"[{args.job}] issue: {sync_issue(record, args.gh_repo, dry_run=args.dry_run)}")
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            print(f"::warning::[{args.job}] the job-health issue was not updated: {exc}")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            counts = ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}" for k, v in last["counts"].items())
            fh.write(f"\n### Job health: {args.job}\n\n**{last['status']}**"
                     + (f" ({'; '.join(last['reasons'])})" if last["reasons"] else "")
                     + (f"; {counts}" if counts else "")
                     + (f"; data of {last['data_date']}" if last.get("data_date") else "")
                     + f"; scheduled runs green in a row: {record['green_streak']}.\n")
    return 0


def status_lines(index: dict[str, Any] | None, now: datetime | None = None) -> list[str]:
    if not index or not index.get("jobs"):
        return ["no job health published yet"]
    out = []
    for entry in index["jobs"]:
        a = assess(entry, now)
        age = f"{a['age_hours'] / 24:.1f} d ago" if a["age_hours"] is not None else "never"
        out.append(f"{entry['job']:<20} {a['state']:<9} last success {age:<12} streak {a['green_streak']:<3} "
                   f"{entry.get('cadence') or ''}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.health_jobs", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="record a run, write it, optionally publish it and keep the issue")
    r.add_argument("job", choices=list(JOBS))
    r.add_argument("--status", required=True, help="the job's status (success, failure, cancelled, skipped)")
    r.add_argument("--root", default=None, help="the folder the job wrote (its --out), for counts and data date")
    r.add_argument("--out", default="health-out", help="where to write health/jobs/")
    r.add_argument("--started", default=None, help="when the job started (ISO time or epoch seconds)")
    r.add_argument("--finished", default=None, help="when it finished (default: now)")
    r.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    r.add_argument("--run-url", default="")
    r.add_argument("--require", action="append", metavar="STEP=OUTCOME",
                   help="a step an unattended run promises, with its outcome (repeatable)")
    r.add_argument("--publish", default="false", help="true to upload health/jobs/<job>.json and the index")
    r.add_argument("--issue", default="false", help="true to open, update or close the job-health issue")
    r.add_argument("--repo", default=os.environ.get("HF_DATASET") or DEFAULT_REPO, help="the Archive dataset")
    r.add_argument("--gh-repo", default=os.environ.get("GITHUB_REPOSITORY") or "Rekin226/aquascope")
    r.add_argument("--dry-run", action="store_true", help="print the gh calls instead of making them")
    s = sub.add_parser("status", help="the published index, one line per job")
    s.add_argument("--repo", default=DEFAULT_REPO)
    s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "report":
        return report(a)
    try:
        index = read_published_index(a.repo)
    except ReadError as exc:
        print(f"could not read the published index: {exc}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps([{**assess(e), "label": e.get("label")} for e in (index or {}).get("jobs", [])], indent=1))
    else:
        print("\n".join(status_lines(index)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
