"""The archive of issued forecasts and the daily status snapshot (#517).

Once a day, for the Archive's discharge gauges with a live record (a mirrored series of at least ten years, topped
up with the agency's newest days, and a GEOGLOWS river reach the gauge snaps to), this stores what the two global
forecasts said that day, raw and corrected to the gauge (:mod:`aquascope.nownext`), so forecast skill at each lead
time can be measured later against what the gauge then recorded. It also writes where each gauge's flow sits today
against normal, which the Explorer uses to colour the map.

Layout under ``<out>/forecasts/`` (the only folder this job publishes; the catalogue is never touched):

    manifest.json              what is published: each issue date, its gauges, what was dropped and why
    reaches.parquet            source, station_id, river_id, distance_m, lat, lon, glofas_lat, glofas_lon, snapped_at
    issued/<YYYY-MM-DD>.parquet  one row per gauge, model and valid day (raw and, for GEOGLOWS, corrected)
    status/latest.parquet      source, station_id, value_date, value, percentile, class, n_years
    status/latest.json         when the snapshot was made and which sources it covers
    status/<YYYY-MM-DD>.parquet  the same snapshot, kept by date

Licences: GEOGLOWS v2 output is CC BY 4.0; Open-Meteo serves GloFAS under CC BY 4.0. Both are model output and
labelled so. ``river_id`` is a TDX-Hydro reach identifier; no river geometry is stored.

Run by ``.github/workflows/forecast-archive.yml``:

    python -m aquascope.archive.forecasts run --out build [--max-gauges 250] [--max-items N]
    python -m aquascope.archive.forecasts publish --out build

Needs the ``archive`` extra (pyarrow, huggingface_hub).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FOLDER = "forecasts"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
#: Gauges forecast per day: GEOGLOWS takes 10 to 25 s per reach for the simulated record, so a few hundred keeps
#: the job inside half an hour of free runner time with a few workers.
DEFAULT_MAX_GAUGES = 250
#: A mirrored series whose last value is this recent is a candidate (the mirror is refreshed weekly).
CANDIDATE_DAYS = 21
#: A value this many days old or newer counts as today's, for the status snapshot and the forecasts.
FRESH_DAYS = 3
MIN_RECORD_YEARS = 10
#: Reaches that did not snap are tried again after this many days.
RESNAP_DAYS = 30

ABOUT = ("Issued forecasts and the daily status snapshot for AquaScope's Archive gauges (#517). Model output: "
         "GEOGLOWS v2 (GEOGloWS ECMWF Streamflow Service, CC BY 4.0) and GloFAS v4 via Open-Meteo (CC BY 4.0). "
         "Corrected values are flow-duration quantile mapping to each gauge's record (aquascope.nownext).")
LICENCE = {"geoglows": "CC BY 4.0", "glofas": "CC BY 4.0 (Open-Meteo API data; GloFAS by Copernicus EMS)"}
COLUMNS = {
    "issue_date": "the day the job ran (UTC)", "model": "geoglows or glofas", "valid_date": "the day forecast",
    "init_date": "the day the model run started (GEOGLOWS; GloFAS via Open-Meteo does not say, so empty)",
    "lead_day": "valid_date minus init_date (or issue_date when init_date is empty), in days",
    "mean/median/p25/p75/min/max": "ensemble statistics, m3/s",
    "high_res": "GEOGLOWS high-resolution run, m3/s", "*_c": "the same, corrected to the gauge (GEOGLOWS only)",
    "kge_raw/kge_corrected": "the correction's hindcast skill at this gauge",
    "by": "month or year flow-duration curves",
    "generated": "when the GEOGLOWS API answered (not when the run started; see init_date)",
    "reach_mean_ratio": "the reach's simulated mean flow over the gauge's, on the days they share (far from 1: "
    "the gauge may be on another river than its snapped reach)",
    "gauge_q2/gauge_q5/gauge_q10/gauge_q25/gauge_q50/gauge_q100": "the gauge's 2- to 100-year flows from its own "
    "annual maxima (m3/s), which the corrected values are compared with (the watch digest and the feeds, #521)",
}
#: The return periods whose gauge flows each issued row carries (``gauge_q<T>``).
GAUGE_RETURN_PERIODS = (2, 5, 10, 25, 50, 100)


# ── reading what is published ───────────────────────────────────────────────


def published_url(path: str, repo_id: str = DEFAULT_REPO) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


def _get(url: str) -> bytes | None:
    import httpx

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=120)
    except httpx.HTTPError as exc:
        logger.info("could not read %s: %s", url, exc)
        return None
    return resp.content if resp.status_code == 200 else None


def read_published_json(path: str, repo_id: str = DEFAULT_REPO) -> dict[str, Any]:
    data = _get(published_url(path, repo_id))
    try:
        return json.loads(data) if data else {}
    except ValueError:
        return {}


def read_published_rows(path: str, repo_id: str = DEFAULT_REPO) -> list[dict[str, Any]]:
    import io

    import pyarrow.parquet as pq

    data = _get(published_url(path, repo_id))
    return pq.read_table(io.BytesIO(data)).to_pylist() if data else []


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


# ── which gauges ────────────────────────────────────────────────────────────


def _d(x: Any) -> date | None:
    try:
        return date.fromisoformat(str(x)[:10])
    except (TypeError, ValueError):
        return None


def candidates(manifest: dict[str, Any], *, today: date, variable: str = "discharge",
               max_age_days: int = CANDIDATE_DAYS, min_years: float = MIN_RECORD_YEARS) -> list[dict[str, Any]]:
    """The mirrored series of ``variable`` that end within ``max_age_days`` of ``today`` and span ``min_years``."""
    out = []
    for key, entry in (manifest.get("sources") or {}).items():
        source, _, var = key.partition("/")
        if (entry.get("variable") or var) != variable:
            continue
        for sid, row in (entry.get("stations") or {}).items():
            first, last = _d(row.get("first")), _d(row.get("last"))
            if not first or not last or (today - last).days > max_age_days:
                continue
            years = (last - first).days / 365.25
            if years >= min_years:
                out.append({"source": entry.get("source") or source, "station_id": str(sid),
                            "first": first.isoformat(), "last": last.isoformat(), "years": round(years, 1)})
    out.sort(key=lambda r: (r["source"], -r["years"], r["station_id"]))
    return out


def round_robin(rows: list[dict[str, Any]], cap: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Take ``cap`` rows one source at a time (each source's longest records first), so a cap keeps every
    source in; returns (kept, dropped)."""
    if cap is None or cap >= len(rows):
        return list(rows), []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    kept: list[dict[str, Any]] = []
    queues = [list(v) for _, v in sorted(by_source.items())]
    while len(kept) < cap and any(queues):
        for q in queues:
            if q and len(kept) < cap:
                kept.append(q.pop(0))
    chosen = {(r["source"], r["station_id"]) for r in kept}
    return kept, [r for r in rows if (r["source"], r["station_id"]) not in chosen]


# ── records, status, reaches ────────────────────────────────────────────────


def gauge_record(source: str, station_id: str, *, today: date) -> tuple[Any, str]:
    """The mirrored daily discharge, topped up with the agency's newest days."""
    from aquascope.archive.observations import fetch_archived_series
    from aquascope.nownext import top_up

    s = fetch_archived_series(source, station_id, "discharge")
    if s is None or s.empty:
        return None, "no mirrored series"
    return top_up(s, source, station_id, variable="discharge", today=today)


def status_rows(records: dict[tuple[str, str], Any], *, today: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Today against normal for every record whose newest value is at most :data:`FRESH_DAYS` old."""
    from aquascope.nownext import METHODS, STATUS_CLASSES, flow_status

    rows = []
    for (source, sid), s in sorted(records.items()):
        if s is None or len(s) == 0:
            continue
        st = flow_status(s, today=today)
        if not st.get("class") or st.get("age_days") is None or st["age_days"] > FRESH_DAYS:
            continue
        rows.append({"source": source, "station_id": sid, "value_date": st["date"], "value": st["value"],
                     "percentile": st["percentile"], "class": st["class"], "n_years": st["n_years"]})
    counts = {c["id"]: sum(r["class"] == c["id"] for r in rows) for c in STATUS_CLASSES}
    meta = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "date": today.isoformat(),
            "sources": sorted({r["source"] for r in rows}), "n": len(rows), "counts": counts,
            "fresh_days": FRESH_DAYS, "variable": "discharge", "method": METHODS["status"]["text"],
            "classes": STATUS_CLASSES}
    return rows, meta


def snap_reaches(gauges: list[dict[str, Any]], known: dict[tuple[str, str], dict[str, Any]],
                 positions: dict[tuple[str, str], tuple[float, float]], *, today: date,
                 snapper: Callable[..., dict[str, Any]] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Snap each gauge to its GEOGLOWS reach once (kept in ``reaches.parquet``); a gauge that did not snap is
    tried again after :data:`RESNAP_DAYS`. A gauge's position is on its own river, so the nearest line is taken
    (a click's snap prefers the main channel; a tributary gauge beside a big river must not)."""
    if snapper is None:
        from functools import partial

        from aquascope.rivers import snap_to_river

        snapper = partial(snap_to_river, prefer="nearest")
    for g in gauges:
        key = (g["source"], g["station_id"])
        old = known.get(key)
        if old and (old.get("river_id") or (today - (_d(old.get("snapped_at")) or today)).days < RESNAP_DAYS):
            continue
        pos = positions.get(key)
        if not pos:
            continue
        try:
            sn = snapper(pos[0], pos[1])
        except Exception as exc:  # noqa: BLE001 - one gauge's network read must not stop the rest
            logger.info("snap failed for %s/%s: %s", *key, exc)
            continue
        known[key] = {"source": key[0], "station_id": key[1], "river_id": sn.get("river_id") if sn.get("snapped")
                      else None, "distance_m": sn.get("distance_m"), "lat": pos[0], "lon": pos[1],
                      "glofas_lat": None, "glofas_lon": None, "snapped_at": today.isoformat()}
    return known


# ── one gauge's forecast ────────────────────────────────────────────────────


def issue_one(gauge: dict[str, Any], series: Any, reach: dict[str, Any], *, today: date, days: int = 15
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows one gauge adds to today's issue, and its reach row (with the GloFAS cell, picked once)."""
    from aquascope.nownext import STAT_KEYS, forecast

    reach = dict(reach)
    if reach.get("glofas_lat") is None:
        from aquascope.explore import snap_glofas_cell

        try:
            cell = snap_glofas_cell(reach["lat"], reach["lon"], float(series.mean()), window=1)
        except Exception as exc:  # noqa: BLE001 - fall back to the gauge's own cell
            logger.info("GloFAS cell match failed for %s/%s: %s", gauge["source"], gauge["station_id"], exc)
            cell = {}
        ok = cell.get("lat") is not None and cell.get("flow_magnitude_matches")
        reach["glofas_lat"] = cell["lat"] if ok else reach["lat"]
        reach["glofas_lon"] = cell["lon"] if ok else reach["lon"]
    fc = forecast(river_id=reach["river_id"], obs=series, glofas_at=(reach["glofas_lat"], reach["glofas_lon"]),
                  days=days)
    skill = (fc.get("correction") or {}).get("skill") or {}
    common = {"issue_date": today.isoformat(), "source": gauge["source"], "station_id": gauge["station_id"],
              "river_id": int(reach["river_id"]), "by": skill.get("by"),
              "kge_raw": (skill.get("raw") or {}).get("kge"),
              "kge_corrected": (skill.get("corrected") or {}).get("kge"),
              "reach_mean_ratio": (fc.get("reach_check") or {}).get("ratio")}
    gthr = fc.get("gauge_thresholds") or {}
    gq = {float(t): q for t, q in zip(gthr.get("return_periods") or [], gthr.get("q") or [])}
    for t in GAUGE_RETURN_PERIODS:
        common[f"gauge_q{t}"] = gq.get(float(t))
    rows: list[dict[str, Any]] = []
    corrected = (fc.get("correction") or {}).get("forecast") or {}
    for model in ("geoglows", "glofas"):
        part = fc.get(model) or {}
        init = _d(part.get("initialized"))
        for i, day in enumerate(part.get("date") or []):
            row = {**common, "model": model, "init_date": init.isoformat() if init else None, "valid_date": day,
                   "lead_day": (date.fromisoformat(day) - (init or today)).days, "generated": part.get("generated")}
            for k in STAT_KEYS:
                vals = part.get(k)
                row[k] = vals[i] if isinstance(vals, list) and i < len(vals) else None
                cvals = corrected.get(k) if model == "geoglows" else None
                row[f"{k}_c"] = cvals[i] if isinstance(cvals, list) and i < len(cvals) else None
            rows.append(row)
    return rows, reach


# ── the run ─────────────────────────────────────────────────────────────────


def _positions(rows: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    for r in rows:
        lat, lon = r.get("latitude"), r.get("longitude")
        if lat is not None and lon is not None:
            out[(str(r["source"]), str(r["station_id"]))] = (float(lat), float(lon))
    return out


def _parallel(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def run(out: str | Path, *, repo_id: str = DEFAULT_REPO, max_gauges: int = DEFAULT_MAX_GAUGES,
        max_items: int | None = None, workers: int = 4, time_budget_s: float = 75 * 60, days: int = 15,
        today: date | None = None, manifest: dict[str, Any] | None = None,
        catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build today's status snapshot and forecast issue under ``<out>/forecasts/`` and return a summary.

    ``max_items`` caps the gauges looked at (a smoke run); ``max_gauges`` caps the ones forecast. Gauges past the
    caps or the time budget are counted under ``dropped`` in the manifest, by reason.
    """
    started = time.monotonic()
    today = today or datetime.now(timezone.utc).date()
    root = Path(out) / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = read_published_json("obs/manifest.json", repo_id)
    if catalog is None:
        from aquascope.archive.catalog import load_stations

        catalog = load_stations(repo_id=repo_id)
    dropped: dict[str, int] = defaultdict(int)

    cands = candidates(manifest, today=today)
    cands, extra = round_robin(cands, max_items)
    dropped["smoke cap"] += len(extra)
    logger.info("%d candidate gauges", len(cands))

    def record(g: dict[str, Any]) -> tuple[tuple[str, str], Any, str]:
        # Half the budget at most for reading records, so a slow agency leaves time for the forecasts.
        if time.monotonic() - started > time_budget_s / 2:
            return (g["source"], g["station_id"]), None, "time budget"
        try:
            s, note = gauge_record(g["source"], g["station_id"], today=today)
        except Exception as exc:  # noqa: BLE001
            return (g["source"], g["station_id"]), None, f"failed: {exc}"
        return (g["source"], g["station_id"]), s, note

    got = _parallel(record, cands, workers)
    records = {key: s for key, s, _ in got if s is not None}
    dropped["time budget (records)"] += sum(1 for _, s, why in got if s is None and why == "time budget")
    dropped["no mirrored series"] += sum(1 for _, s, why in got if s is None and why != "time budget")

    rows, meta = status_rows(records, today=today)
    _write_rows(rows, root / "status" / "latest.parquet")
    _write_rows(rows, root / "status" / f"{today.isoformat()}.parquet")
    (root / "status" / "latest.json").write_text(json.dumps(meta, indent=1))
    logger.info("status: %d gauges from %s", len(rows), ", ".join(meta["sources"]) or "no source")

    live = [g for g in cands if (g["source"], g["station_id"]) in {(r["source"], r["station_id"]) for r in rows}]
    dropped["no fresh value"] += len([g for g in cands if (g["source"], g["station_id"]) in records]) - len(live)
    known = {(r["source"], r["station_id"]): r for r in read_published_rows(f"{FOLDER}/reaches.parquet", repo_id)}
    known = snap_reaches(live, known, _positions(catalog), today=today)
    reachable = [g for g in live if (known.get((g["source"], g["station_id"])) or {}).get("river_id")]
    dropped["no river reach"] += len(live) - len(reachable)
    chosen, over = round_robin(reachable, max_gauges)
    dropped["daily cap"] += len(over)

    issued: list[dict[str, Any]] = []
    budget_hit = [0]

    def one(g: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
        key = (g["source"], g["station_id"])
        if time.monotonic() - started > time_budget_s:
            budget_hit[0] += 1
            return [], None, "time budget"
        try:
            rws, reach = issue_one(g, records[key], known[key], today=today, days=days)
            return rws, reach, ""
        except Exception as exc:  # noqa: BLE001 - one gauge failing must not stop the issue
            logger.info("forecast failed for %s/%s: %s", *key, exc)
            return [], None, "failed"

    for rws, reach, why in _parallel(one, chosen, workers):
        issued.extend(rws)
        if reach:
            known[(reach["source"], reach["station_id"])] = reach
        if why:
            dropped[why] += 1
    n_issued = len({(r["source"], r["station_id"]) for r in issued})
    if issued:
        _write_rows(issued, root / "issued" / f"{today.isoformat()}.parquet")
    _write_rows(sorted(known.values(), key=lambda r: (r["source"], r["station_id"])), root / "reaches.parquet")

    entry = {"date": today.isoformat(), "file": f"{FOLDER}/issued/{today.isoformat()}.parquet" if issued else None,
             "n_gauges": n_issued, "n_rows": len(issued), "sources": sorted({r["source"] for r in issued}),
             "dropped": {k: v for k, v in sorted(dropped.items()) if v},
             "seconds": round(time.monotonic() - started, 1)}
    published = read_published_json(f"{FOLDER}/manifest.json", repo_id)
    issues = [i for i in published.get("issues") or [] if i.get("date") != entry["date"]] + [entry]
    doc = {"about": ABOUT, "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "licence": LICENCE, "columns": COLUMNS, "status": {**meta, "file": f"{FOLDER}/status/latest.parquet"},
           "issues": sorted(issues, key=lambda i: i["date"])}
    (root / "manifest.json").write_text(json.dumps(doc, indent=1))
    return {"status_gauges": len(rows), "status_sources": meta["sources"], **entry}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/forecasts`` and nothing else to the Archive dataset (needs HF_TOKEN with write access)."""
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    if not (src / "manifest.json").exists():
        raise FileNotFoundError(f"{src}/manifest.json is missing; run the forecast step first")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        shutil.copytree(src, stage)
        day = json.loads((src / "manifest.json").read_text())["issues"][-1]["date"]
        return publish_folder(Path(tmp), repo_id, token=token, commit_message=f"forecasts: issue {day}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.forecasts", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    p.add_argument("--max-gauges", type=int, default=DEFAULT_MAX_GAUGES)
    p.add_argument("--max-items", type=int, default=None, help="gauges looked at, for a smoke run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--time-budget-min", type=float, default=75.0)
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "aquascope.hydrology", "aquascope.collectors.base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "run":
        info = run(a.out, repo_id=a.repo, max_gauges=a.max_gauges, max_items=a.max_items, workers=a.workers,
                   time_budget_s=a.time_budget_min * 60)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
