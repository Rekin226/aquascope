"""Sweep every collector's live fetch and keep one rolling issue about the result.

The weekly harvest only proves that a source's *station catalog* still answers.
A collector whose ``collect()`` path broke -- a retired endpoint, a column that
moved, a payload that stopped being JSON -- can still pass that check, because
the catalog and the observations travel different routes on the same site. This
script asks each source for the smallest slice of real data it can serve, one
row per source, so a broken data path gets its own row instead of hiding behind
a green harvest.

Two statuses never fail the job: ``NEEDS_KEY`` (a secret this runner does not
have) and the probes marked ``informational`` (the agency blocks non-local
runners, or its terms ask for a low crawl rate).

Usage::

    python .github/scripts/collector_health.py --check
    python .github/scripts/collector_health.py --source usgs --no-issue
    python .github/scripts/collector_health.py --issue          # what CI runs
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO = "Rekin226/aquascope"
LABEL = "collector-health"
TITLE = "collector-health: the weekly live collector sweep is failing"

OK = "OK"
EMPTY = "EMPTY"
FAIL = "FAIL"
TIMEOUT = "TIMEOUT"
NEEDS_KEY = "NEEDS_KEY"
STATUS_ORDER = (NEEDS_KEY, FAIL, TIMEOUT, EMPTY, OK)
FAILING = (FAIL, TIMEOUT, EMPTY)


@dataclass(frozen=True)
class Probe:
    """One source's smallest live call.

    ``kwargs`` goes to ``collect()``, ``ctor`` to the constructor. ``requires``
    names the environment variables without which the call cannot work; a
    missing one reports ``NEEDS_KEY`` instead of a failure. ``informational``
    marks a probe whose failure is not a code fault: an agency that blocks
    non-local runners, or a source whose terms ask for a slow crawl.
    """

    kwargs: dict = field(default_factory=dict)
    ctor: dict = field(default_factory=dict)
    timeout: int = 120
    informational: bool = False
    note: str = ""
    requires: tuple = ()


# One entry per key in aquascope.registry.SOURCES; --check fails when they drift.
PROBES: dict[str, Probe] = {
    # --- North America -------------------------------------------------------
    "usgs": Probe(
        # The keyless path refuses an unfiltered walk, so ask for one small bbox.
        kwargs={"collection": "daily", "bbox": "-77.1,38.8,-77.0,38.9", "days": 1, "max_items": 5},
        note="keyless DEMO_KEY path over a small Potomac bbox; set USGS_API_KEY to lift the rate limit",
    ),
    "noaa_nwps": Probe(
        kwargs={"lid": "AGSI4"},
        note="AHPS gauge on the Mississippi at St. Louis",
    ),
    "wqp": Probe(
        kwargs={"state_code": "RI", "max_results": 5},
        note="Water Quality Portal, Rhode Island",
    ),
    "camels_cl": Probe(
        kwargs={"station_ids": ["1001"], "start": "2010-01-01", "end": "2010-01-05"},
        timeout=300,
        informational=True,
        note="packaged CAMELS-CL dataset: a large download, not a live API",
    ),
    "camels_br": Probe(
        kwargs={"station_ids": ["1001"], "start": "2010-01-01", "end": "2010-01-05"},
        timeout=300,
        informational=True,
        note="packaged CAMELS-BR dataset: a large download, not a live API",
    ),
    # --- Africa --------------------------------------------------------------
    "south_africa_dws": Probe(
        kwargs={"station_id": "C1H001", "days": 7},
        informational=True,
        note="the DWS host answers 403 outside South Africa",
    ),
    # --- Americas ------------------------------------------------------------
    "colorado_cdss": Probe(
        kwargs={"abbrev": "PLAKERCO"},
        note="Colorado CDSS, Platte and Kenosha reach",
    ),
    "brazil_ana": Probe(
        kwargs={"days": 1},
        requires=("ANA_HIDROWEB_IDENTIFICADOR", "ANA_HIDROWEB_SENHA"),
        note="needs the ANA Hidroweb credentials",
    ),
    # --- Europe --------------------------------------------------------------
    "uk_ea": Probe(
        # `station` is the EA's own GUID, not the WISKI id; daily readings over
        # two months always have rows where a single 15-minute day may not.
        kwargs={
            "station": "8496ce69-482c-406a-a2f0-ac418ef8f099",
            "observed_property": "waterFlow",
            "collection": "daily",
            "days": 60,
        },
        note="Environment Agency gauge at Kingston, daily river flow",
    ),
    "hubeau_hydrometrie": Probe(
        kwargs={"code_station": "F700000103", "days": 1},
        note="Hub'Eau station F700000103",
    ),
    "pegelonline": Probe(
        kwargs={"station_id": "BONN", "days": 1},
        note="PEGELONLINE Rhine gauge at Bonn",
    ),
    "ireland_opw": Probe(
        kwargs={"max_stations": 1},
        note="one OPW station",
    ),
    "greece_hydroscope": Probe(
        # The first station the search returns may have no readings today.
        kwargs={"max_stations": 5},
        note="five Hydroscope stations, discharge",
    ),
    "greece_openhi": Probe(
        kwargs={"max_stations": 1, "latest_only": True},
        note="one OpenHi station, latest reading",
    ),
    "poland_imgw": Probe(
        kwargs={"max_stations": 1, "latest_only": True},
        note="one IMGW station, latest reading",
    ),
    "eu_wfd": Probe(
        kwargs={"country": "Germany", "year": 2020},
        informational=True,
        note="the EEA DiscoData host answers 403 outside the EU",
    ),
    # --- Taiwan --------------------------------------------------------------
    "taiwan_wra_level": Probe(kwargs={}, note="WRA river level feed"),
    "taiwan_wra_reservoir": Probe(kwargs={}, note="WRA reservoir storage feed"),
    "taiwan_wra_groundwater": Probe(kwargs={}, note="WRA groundwater feed"),
    "taiwan_wra_groundwater_daily": Probe(
        kwargs={},
        informational=True,
        note="the WRA groundwater query host answers 403 outside Taiwan",
    ),
    "taiwan_wra_fhy": Probe(
        kwargs={},
        informational=True,
        note="the FHY API answers 503 from outside Taiwan",
    ),
    "taiwan_wra_iot": Probe(kwargs={}, note="WRA IoT feed"),
    "taiwan_datagov": Probe(kwargs={"limit": 2}, note="data.gov.tw river level"),
    "taiwan_civil_iot": Probe(
        kwargs={"top": 2},
        informational=True,
        note="the station host ships a broken certificate chain",
    ),
    "taiwan_cwa": Probe(kwargs={}, note="CWA rainfall stations"),
    "taiwan_moenv": Probe(
        kwargs={"limit": 2},
        requires=("TAIWAN_MOENV_API_KEY",),
        note="needs a MOENV API key",
    ),
    # --- East Asia -----------------------------------------------------------
    "japan_mlit": Probe(
        kwargs={"prefecture": "Tokyo", "max_stations": 1},
        timeout=300,
        informational=True,
        note="refused by its own robots.txt once #737 lands; until then a timeout",
    ),
    "korea_wamis": Probe(
        kwargs={"station_id": "1018680", "parameter": "water_level"},
        informational=True,
        note="the WAMIS host drops connections from outside Korea",
    ),
    "india_wris": Probe(
        kwargs={
            "state_name": "Maharashtra",
            "district_name": "Pune",
            "agency_name": "Central Water Commission",
            "startdate": "2023-01-01",
            "enddate": "2023-01-02",
        },
        informational=True,
        note="the WRIS host is often unreachable from non-Indian runners",
    ),
    # --- Oceania -------------------------------------------------------------
    "bom": Probe(
        # The daily series lags the clock, so ask for a week rather than a day.
        kwargs={"station_id": "410001", "days": 7},
        note="BoM station 410001, water course discharge",
    ),
    # --- Global --------------------------------------------------------------
    "grdc": Probe(
        kwargs={"max_items": 5},
        timeout=300,
        note="GRDC bulk file, five readings",
    ),
    "openmeteo": Probe(
        # Weather mode wants an explicit window; forecast mode does not.
        kwargs={"latitude": 48.2082, "longitude": 16.3738, "start_date": "2024-01-01", "end_date": "2024-01-02"},
        note="Vienna, two days of archive weather",
    ),
    "copernicus": Probe(
        kwargs={"latitude": 48.2082, "longitude": 16.3738, "year": "2023", "month": "01"},
        requires=("COPERNICUS_CDS_API_KEY",),
        timeout=300,
        note="needs a Copernicus CDS API key",
    ),
    "gemstat": Probe(kwargs={"max_records": 5}, note="GEMStat, five records"),
    "sdg6": Probe(
        kwargs={"page_size": 5},
        timeout=300,
        informational=True,
        note="the UN statistics host is slow or unreachable from some regions",
    ),
    "aquastat": Probe(
        kwargs={"country_code": "Egypt", "variable_ids": [4253], "start_year": 2020, "end_year": 2021},
        note="AQUASTAT, Egypt total water withdrawal",
    ),
    "wapor": Probe(
        kwargs={"bbox": (0.0, 0.0, 0.1, 0.1), "start_date": "2020-01-01", "end_date": "2020-01-02"},
        timeout=300,
        note="WaPOR evapotranspiration, one tiny tile",
    ),
}


@dataclass
class Outcome:
    source: str
    status: str
    seconds: float = 0.0
    records: int = 0
    detail: str = ""
    note: str = ""
    informational: bool = False

    @property
    def failing(self) -> bool:
        return self.status in FAILING


class _TimeoutError(Exception):
    pass


def _alarm(signum, frame):  # noqa: ANN001 - signal handler signature
    raise _TimeoutError()


def _count(result) -> int:
    if result is None:
        return 0
    if isinstance(result, dict):
        return len(result)
    try:
        return len(result)
    except TypeError:
        return 1


def _error_text(exc: BaseException, limit: int = 1200) -> str:
    """``Type: message`` plus up to three chained causes, clipped to ``limit``."""
    text = f"{type(exc).__name__}: {exc}"
    cause, depth = exc.__cause__, 0
    while cause is not None and depth < 3:
        text += f" (from {type(cause).__name__}: {cause})"
        cause = cause.__cause__
        depth += 1
    return text[:limit]


def _call_with_timeout(fn, seconds: int):
    """Run ``fn`` and give up after ``seconds``.

    POSIX uses SIGALRM, which actually interrupts the call. Elsewhere (Windows
    dev boxes) a daemon thread is abandoned on timeout: the sweep keeps going,
    but the stuck thread lives until the process exits.
    """
    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(seconds)
        try:
            return fn()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    box: dict = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise _TimeoutError()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_probe(source: str, probe: Probe, *, api_key: str | None = None) -> Outcome:
    from aquascope.registry import build_collector

    started = time.monotonic()
    try:
        collector = build_collector(source, api_key=api_key, **probe.ctor)
    except Exception as exc:  # noqa: BLE001 - a constructor failure is a FAIL row, not a crash
        return Outcome(source, FAIL, time.monotonic() - started,
                       detail=f"constructor: {_error_text(exc)}", note=probe.note,
                       informational=probe.informational)

    try:
        result = _call_with_timeout(lambda: collector.collect(**probe.kwargs), probe.timeout)
    except _TimeoutError:
        return Outcome(source, TIMEOUT, time.monotonic() - started,
                       detail=f"no answer within {probe.timeout}s", note=probe.note,
                       informational=probe.informational)
    except Exception as exc:  # noqa: BLE001 - every collector error becomes a row
        return Outcome(source, FAIL, time.monotonic() - started, detail=_error_text(exc),
                       note=probe.note, informational=probe.informational)

    elapsed = time.monotonic() - started
    records = _count(result)
    return Outcome(source, OK if records else EMPTY, elapsed, records,
                   detail="" if records else "the call succeeded but returned nothing",
                   note=probe.note, informational=probe.informational)


def format_summary(outcomes: list[Outcome]) -> str:
    """The markdown matrix written to $GITHUB_STEP_SUMMARY."""
    ordered = sorted(outcomes, key=lambda o: (STATUS_ORDER.index(o.status), o.source))
    lines = [
        "## Collector health",
        "",
        "| source | status | records | seconds | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in ordered:
        mark = " _(informational)_" if o.informational and o.failing else ""
        detail = (o.detail or o.note).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{o.source}` | {o.status}{mark} | {o.records} | {o.seconds:.1f} | {detail} |")

    counts = Counter(o.status for o in outcomes)
    tally = ", ".join(f"{counts[s]} {s}" for s in STATUS_ORDER if counts[s])
    lines += ["", f"**{len(outcomes)} sources:** {tally}"]
    failures = [o for o in outcomes if o.failing and not o.informational]
    if failures:
        lines += ["", "**Failing:** " + ", ".join(f"`{o.source}`" for o in failures)]
    else:
        lines += ["", "**Failing:** none"]
    return "\n".join(lines) + "\n"


def issue_body(outcomes: list[Outcome], run_url: str = "") -> str:
    failures = [o for o in outcomes if o.failing and not o.informational]
    informational = [o for o in outcomes if o.failing and o.informational]
    parts = [
        "The weekly live sweep asks every collector for the smallest slice of real data it can serve, "
        "so a broken data path gets its own row instead of hiding behind a green station-catalog harvest.",
        "",
        f"**{len(failures)} of {len(outcomes)} sources failed the sweep.**",
        "",
    ]
    if failures:
        parts += ["| source | status | detail |", "| --- | --- | --- |"]
        for o in failures:
            detail = (o.detail or "").replace("|", "\\|").replace("\n", " ")[:400]
            parts.append(f"| `{o.source}` | {o.status} | {detail} |")
        parts += ["", "Reproduce one row locally:", "```bash"]
        parts.append(f"python .github/scripts/collector_health.py --source {failures[0].source} --no-issue")
        parts.append("```")
    if informational:
        parts += ["", "Informational failures (a missing secret, or an agency that blocks non-local runners): "
                  + ", ".join(f"`{o.source}`" for o in informational)]
    if run_url:
        parts += ["", f"Run: {run_url}"]
    parts += [
        "",
        "_Opened automatically by the collector-health workflow; it closes itself when every source answers._",
    ]
    return "\n".join(parts)


def _gh(*args: str, dry_run: bool = False) -> str:
    if dry_run:
        print("DRY RUN: gh", " ".join(args))
        return ""
    proc = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def find_open_issue(repo: str, *, dry_run: bool = False) -> int | None:
    if dry_run:
        return None
    raw = _gh("issue", "list", "--repo", repo, "--label", LABEL, "--state", "open",
              "--limit", "100", "--json", "number,title")
    for item in json.loads(raw or "[]"):
        if item["title"] == TITLE:
            return int(item["number"])
    return None


def sync_issue(outcomes: list[Outcome], *, repo: str = REPO, dry_run: bool = False,
               run_url: str = "") -> int:
    """Open, update or close the one rolling issue. Returns 1 when sources fail."""
    existing = find_open_issue(repo, dry_run=dry_run)
    failures = [o for o in outcomes if o.failing and not o.informational]
    if failures:
        body = issue_body(outcomes, run_url)
        if existing:
            _gh("issue", "comment", str(existing), "--repo", repo, "--body", body, dry_run=dry_run)
            print(f"still failing -> commented on #{existing}")
        else:
            _gh("issue", "create", "--repo", repo, "--title", TITLE, "--body", body,
                "--label", LABEL, dry_run=dry_run)
            print("failing -> opened the rolling issue")
        return 1
    if existing:
        _gh("issue", "close", str(existing), "--repo", repo, "--comment",
            "Every source answered this week; closing automatically.", dry_run=dry_run)
        print(f"all green -> closed #{existing}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", action="append", default=[], help="probe only these sources (repeatable)")
    ap.add_argument("--json", type=Path, help="write the outcomes here as JSON")
    ap.add_argument("--summary", type=Path, help="write the markdown matrix here (default: $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--issue", dest="issue", action="store_true", help="open, update or close the rolling issue")
    ap.add_argument("--no-issue", dest="issue", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="print the gh commands instead of running them")
    ap.add_argument("--timeout", type=int, default=0, help="override every probe's timeout")
    ap.add_argument("--check", action="store_true", help="only verify that every source has a probe")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--run-url", default="")
    ap.set_defaults(issue=False)
    args = ap.parse_args(argv)

    from aquascope.registry import SOURCES

    missing = sorted(set(SOURCES) - set(PROBES))
    extra = sorted(set(PROBES) - set(SOURCES))
    if missing or extra:
        print(f"PROBES out of sync with SOURCES: missing={missing} extra={extra}", file=sys.stderr)
        return 2
    if args.check:
        print(f"OK: {len(PROBES)} probes cover {len(SOURCES)} sources")
        return 0

    names = args.source or list(SOURCES)
    unknown = [n for n in names if n not in PROBES]
    if unknown:
        print(f"unknown source(s): {unknown}", file=sys.stderr)
        return 2

    outcomes: list[Outcome] = []
    for name in names:
        probe = replace(PROBES[name], timeout=args.timeout) if args.timeout else PROBES[name]
        absent = [key for key in probe.requires if not os.environ.get(key)]
        if absent:
            outcome = Outcome(name, NEEDS_KEY, detail="missing " + ", ".join(absent),
                              note=probe.note, informational=probe.informational)
            outcomes.append(outcome)
            print(f"[{name}] NEEDS_KEY: missing {', '.join(absent)}")
            continue
        outcome = run_probe(name, probe)
        outcomes.append(outcome)
        tail = f" -- {outcome.detail}" if outcome.detail else ""
        print(f"[{name}] {outcome.status} {outcome.records} records in {outcome.seconds:.1f}s{tail}")

    summary = format_summary(outcomes)
    print()
    print(summary)

    summary_path = args.summary
    if summary_path is None and os.environ.get("GITHUB_STEP_SUMMARY"):
        summary_path = Path(os.environ["GITHUB_STEP_SUMMARY"])
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")
        print(f"summary -> {summary_path}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(o) for o in outcomes], indent=2), encoding="utf-8")
        print(f"json -> {args.json}")

    exit_code = 1 if any(o.failing and not o.informational for o in outcomes) else 0
    if args.issue:
        exit_code = sync_issue(outcomes, repo=args.repo, dry_run=args.dry_run, run_url=args.run_url) or exit_code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
