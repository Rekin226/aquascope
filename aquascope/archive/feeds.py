"""Per-gauge Atom feeds for the Archive's gauges with a live record (#521).

Run once a day after the forecast archive (:mod:`aquascope.archive.forecasts`), from the files it just wrote: for
every gauge in today's status snapshot it keeps an Atom 1.0 feed (RFC 4287) that a feed reader can follow, with an
entry when the gauge's status class changes (today against normal) and an entry when the GEOGLOWS forecast
corrected to the gauge passes the gauge's own 2-year flow (or a rarer one) in the next 15 days.

Layout under ``<out>/feeds/`` (the only folder this job publishes):

    <source>/<station_id>.xml   one Atom feed per gauge (the station id with anything but A-Z a-z 0-9 . _ - as _)
    index.json                  every live gauge and its feed's path, and when the feeds were made
    state.parquet               the entries kept per gauge (the newest 20), so the next run continues them

Licences: the observations are each agency's, through the Archive; GEOGLOWS v2 output is CC BY 4.0. Forecast
entries are model output and say so; they are not flood warnings.

    python -m aquascope.archive.feeds run --out build
    python -m aquascope.archive.feeds publish --out build

Needs the ``archive`` extra (pyarrow, huggingface_hub).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

FOLDER = "feeds"
#: What the feeds publish uploads: the Atom files (the Archive's default allow-list has no ``*.xml``), the
#: index and the state.
PUBLISH_PATTERNS = ("*.xml", "*.json", "*.parquet")
#: Entries kept per feed.
MAX_ENTRIES = 20
#: A forecast alert is not repeated within this many days unless a rarer flow is passed.
ALERT_DAYS = 3
ABOUT = ("Atom feeds for AquaScope's Archive gauges with a live record (#521): an entry when today's flow moves "
         "to another class against normal, and when the GEOGLOWS v2 forecast corrected to the gauge (modelled, "
         "CC BY 4.0) passes the gauge's own 2-year flow or a rarer one in the next 15 days. Not flood warnings.")
RIGHTS = ("Observations: each gauge's agency, through the AquaScope Archive. Forecasts: GEOGLOWS v2 (GEOGloWS "
          "ECMWF Streamflow Service), CC BY 4.0, modelled.")


def safe_id(station_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(station_id)) or "_"


def feed_path(source: str, station_id: str) -> str:
    """The feed's path under ``feeds/``."""
    return f"{safe_id(source)}/{safe_id(station_id)}.xml"


def feed_url(source: str, station_id: str, repo_id: str | None = None) -> str:
    """Where the gauge's feed is published; also the feed's Atom id (entry ids add a fragment)."""
    from aquascope.archive.forecasts import DEFAULT_REPO, published_url

    return published_url(f"{FOLDER}/{feed_path(source, station_id)}", repo_id or DEFAULT_REPO)


def explorer_link(source: str, station_id: str) -> str:
    from aquascope.watch import EXPLORER_URL

    return f"{EXPLORER_URL}#s={quote(f'{source}/{station_id}', safe='/')}&tab=now"


def _d(x: Any) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


def _label(cls: str | None) -> str:
    from aquascope.nownext import STATUS_CLASSES

    return next((c["label"] for c in STATUS_CLASSES if c["id"] == cls), str(cls))


def _ordinal(n: int) -> str:
    from aquascope.nownext import _ordinal as o

    return o(n)


def _q(x: float | None) -> str:
    from aquascope.nownext import _fmt_q

    return f"{_fmt_q(x)} m³/s"


def _when(d: date) -> str:
    from aquascope.nownext import _day_month

    return _day_month(d, year=True)


def forecast_level(rows: list[dict[str, Any]], *, today: date, horizon: int = 15) -> dict[str, Any] | None:
    """The rarest gauge flow (``gauge_q<T>``) the corrected forecast's ensemble mean passes, or None."""
    from aquascope.archive.forecasts import GAUGE_RETURN_PERIODS

    days = [(r, _d(r.get("valid_date"))) for r in rows if r.get("model") == "geoglows" and r.get("mean_c") is not None]
    days = [(r, d) for r, d in days if d is not None and today <= d < today + timedelta(days=horizon)]
    if not days:
        return None
    peak_row, peak_day = max(days, key=lambda p: float(p[0]["mean_c"]))
    peak = float(peak_row["mean_c"])
    passed = [(t, float(peak_row[f"gauge_q{t}"])) for t in GAUGE_RETURN_PERIODS
              if peak_row.get(f"gauge_q{t}") is not None and peak >= float(peak_row[f"gauge_q{t}"])]
    if not passed:
        return None
    t, q = passed[-1]
    return {"return_period": t, "q": q, "peak": peak, "peak_date": peak_day.isoformat(),
            "issue_date": str(peak_row.get("issue_date"))[:10]}


def gauge_entries(status: dict[str, Any], issued: list[dict[str, Any]], previous: list[dict[str, Any]], *,
                  name: str, made: datetime, today: date, repo_id: str | None = None) -> list[dict[str, Any]]:
    """The gauge's entries after today's run: the earlier ones, a status entry when the class changed (or the
    first time), and a forecast entry when the corrected forecast passes a gauge flow. Newest
    :data:`MAX_ENTRIES` kept."""
    source, sid = str(status["source"]), str(status["station_id"])
    entries = sorted(previous, key=lambda e: str(e.get("updated")), reverse=True)
    link = explorer_link(source, sid)
    base = feed_url(source, sid, repo_id)
    stamp = made.strftime("%Y-%m-%dT%H:%M:%SZ")
    last_status = next((e for e in entries if e.get("kind") == "status"), None)
    cls, vdate = status.get("class"), _d(status.get("value_date"))
    if cls and vdate and (last_status is None or last_status.get("cls") != cls):
        pct = status.get("percentile")
        rank = f" ({_ordinal(int(round(float(pct))))} percentile of {int(status.get('n_years') or 0)} years)" \
            if pct is not None else ""
        was = f" It was {_label(last_status.get('cls'))} before." if last_status else ""
        entries.insert(0, {
            "source": source, "station_id": sid, "kind": "status", "cls": cls, "level": None,
            "entry_id": f"{base}#status-{vdate.isoformat()}", "updated": stamp, "link": link,
            "title": f"{name}: {_label(cls)}",
            "summary": f"Flow was {_label(cls)} on {_when(vdate)}{rank}.{was} Today against normal ranks the "
                       "day's flow against the same days of the year (7 either side) in every other year of the "
                       "record, in the USGS and WMO HydroSOS classes."})
    lvl = forecast_level(issued, today=today)
    if lvl:
        recent = [e for e in entries if e.get("kind") == "forecast"
                  and (_d(e.get("updated")) or date.min) > today - timedelta(days=ALERT_DAYS)]
        if not any(float(e.get("level") or 0) >= lvl["return_period"] for e in recent):
            t = lvl["return_period"]
            peak_on, issued_on = date.fromisoformat(lvl["peak_date"]), date.fromisoformat(lvl["issue_date"])
            entries.insert(0, {
                "source": source, "station_id": sid, "kind": "forecast", "cls": None, "level": float(t),
                "entry_id": f"{base}#forecast-{lvl['issue_date']}", "updated": stamp, "link": link,
                "title": f"{name}: forecast above the {t:g}-year flow",
                "summary": f"The GEOGLOWS v2 forecast corrected to this gauge (modelled) peaks at {_q(lvl['peak'])} "
                           f"on {_when(peak_on)}, above the {t:g}-year flow ({_q(lvl['q'])}) from the gauge's own "
                           f"record. Issued {_when(issued_on)}. "
                           "Model output, not a flood warning: check the agency and the national flood service."})
    # one entry per id (a re-run on the same day replaces, never duplicates)
    seen: set[str] = set()
    out = []
    for e in sorted(entries, key=lambda x: str(x.get("updated")), reverse=True):
        if e["entry_id"] in seen:
            continue
        seen.add(e["entry_id"])
        out.append(e)
    return out[:MAX_ENTRIES]


def write_feed(path: Path, *, source: str, station_id: str, name: str, entries: list[dict[str, Any]],
               made: datetime, repo_id: str) -> None:
    from aquascope.watch import atom_feed

    updated = max((str(e["updated"]) for e in entries), default=made.strftime("%Y-%m-%dT%H:%M:%SZ"))
    text = atom_feed(
        feed_id=feed_url(source, station_id, repo_id), title=f"{name}: AquaScope watch",
        subtitle="Today against normal and forecast alerts for this gauge. Forecasts are model output, not warnings.",
        updated=updated, self_url=feed_url(source, station_id, repo_id),
        link=explorer_link(source, station_id), rights=RIGHTS,
        entries=[{"id": e["entry_id"], "title": e["title"], "updated": e["updated"], "summary": e.get("summary"),
                  "link": e.get("link")} for e in entries])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _local_or_published(out: Path, rel: str, repo_id: str) -> list[dict[str, Any]]:
    from aquascope.archive.forecasts import read_published_rows

    local = out / rel
    if local.exists():
        import pyarrow.parquet as pq

        return pq.read_table(local).to_pylist()
    return read_published_rows(rel, repo_id)


def _issued(out: Path, repo_id: str, today: date) -> list[dict[str, Any]]:
    from aquascope.archive.forecasts import FOLDER as FC
    from aquascope.archive.forecasts import read_published_json

    local = out / FC / "issued" / f"{today.isoformat()}.parquet"
    if local.exists():
        return _local_or_published(out, f"{FC}/issued/{today.isoformat()}.parquet", repo_id)
    man = out / FC / "manifest.json"
    doc = json.loads(man.read_text()) if man.exists() else read_published_json(f"{FC}/manifest.json", repo_id)
    files = [i["file"] for i in doc.get("issues") or [] if i.get("file")]
    return _local_or_published(out, files[-1], repo_id) if files else []


def run(out: str | Path, *, repo_id: str | None = None, today: date | None = None, made: datetime | None = None,
        status: list[dict[str, Any]] | None = None, issued: list[dict[str, Any]] | None = None,
        state: list[dict[str, Any]] | None = None, catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write ``<out>/feeds/``: one Atom feed per gauge in today's status snapshot, the index and the state.

    The snapshot and the issue come from ``<out>/forecasts/`` when the forecast step has just written them,
    else from the Archive; the earlier entries come from the published ``feeds/state.parquet``.
    """
    from aquascope.archive.forecasts import DEFAULT_REPO, _write_rows, published_url, read_published_rows
    from aquascope.archive.forecasts import FOLDER as FC

    repo_id = repo_id or DEFAULT_REPO
    out = Path(out)
    made = made or datetime.now(timezone.utc)
    today = today or made.date()
    root = out / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    if status is None:
        status = _local_or_published(out, f"{FC}/status/latest.parquet", repo_id)
    if issued is None:
        issued = _issued(out, repo_id, today)
    if state is None:
        state = read_published_rows(f"{FOLDER}/state.parquet", repo_id)
    if catalog is None:
        from aquascope.archive.catalog import load_stations

        catalog = load_stations(repo_id=repo_id)
    names = {(str(r.get("source")), str(r.get("station_id"))): r.get("name") for r in catalog}
    by_gauge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in state:
        by_gauge[(str(e.get("source")), str(e.get("station_id")))].append(e)
    issued_by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in issued:
        issued_by[(str(r.get("source")), str(r.get("station_id")))].append(r)

    index: dict[str, str] = {}
    n_status = n_forecast = 0
    for row in sorted(status, key=lambda r: (str(r.get("source")), str(r.get("station_id")))):
        key = (str(row.get("source")), str(row.get("station_id")))
        name = names.get(key) or key[1]
        before = by_gauge.get(key, [])
        entries = gauge_entries(row, issued_by.get(key, []), before, name=name, made=made, today=today,
                                repo_id=repo_id)
        old_ids = {e.get("entry_id") for e in before}
        n_status += sum(1 for e in entries if e["kind"] == "status" and e["entry_id"] not in old_ids)
        n_forecast += sum(1 for e in entries if e["kind"] == "forecast" and e["entry_id"] not in old_ids)
        by_gauge[key] = entries
        write_feed(root / feed_path(*key), source=key[0], station_id=key[1], name=name, entries=entries, made=made,
                   repo_id=repo_id)
        index[f"{key[0]}/{key[1]}"] = feed_path(*key)

    rows = [e for key in sorted(by_gauge) for e in by_gauge[key]]
    _write_rows(rows, root / "state.parquet")
    doc = {"about": ABOUT, "updated": made.strftime("%Y-%m-%dT%H:%M:%SZ"), "date": today.isoformat(),
           "base": published_url(f"{FOLDER}/", repo_id), "n": len(index), "rights": RIGHTS,
           "new_entries": {"status": n_status, "forecast": n_forecast}, "feeds": index}
    (root / "index.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"feeds": len(index), "new_status_entries": n_status, "new_forecast_entries": n_forecast,
            "date": today.isoformat()}


def publish(out: str | Path, *, repo_id: str | None = None, token: str | None = None) -> str:
    """Upload ``<out>/feeds`` and nothing else to the Archive dataset (needs HF_TOKEN with write access)."""
    from aquascope.archive.forecasts import DEFAULT_REPO
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    if not (src / "index.json").exists():
        raise FileNotFoundError(f"{src}/index.json is missing; run the feeds step first")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        shutil.copytree(src, stage)
        day = json.loads((src / "index.json").read_text())["date"]
        return publish_folder(Path(tmp), repo_id or DEFAULT_REPO, token=token, commit_message=f"feeds: {day}",
                              allow_patterns=PUBLISH_PATTERNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.feeds", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--repo", default=os.environ.get("HF_DATASET") or None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    info = run(a.out, repo_id=a.repo) if a.cmd == "run" else {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
