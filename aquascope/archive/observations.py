"""Archive Phases 1 and 2: harvest daily observations, one small file per station, incrementally.

Layout inside the archive folder / dataset::

    obs/<variable>/<source>/<station_id>.csv.gz   # columns: date,value (daily values, SI units)
    obs/<variable>/<source>.parquet               # Phase 2 bundle of the folder above (station_id, date, value)
    obs/manifest.json                             # per source/variable: station -> {first, last, n, unit, ...}

Why per-station CSV.gz and not only one big parquet: the two consumers that
read one station at a time (the Explorer in the browser, where pyarrow does
not run, and the MCP ``get_timeseries`` tool) want a single small GET; the
bundles (``aquascope.archive.bundles``) serve everyone who wants a whole
source in one ``read_parquet``.

Why a budget: Hub'Eau, UK EA and USGS together hold ~44k stations and each
one costs a few seconds at the agency, so a run harvests ``max_stations``
per source and variable and the manifest carries the cursor. Weekly runs
fill the archive over time; a station is refreshed once it is older than
``refresh_days``. Only sources whose registry entry says
``redistributable=True`` are ever harvested.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aquascope import __version__
from aquascope.registry import SOURCES

logger = logging.getLogger(__name__)

# Sources with long records reachable through aquascope.explore.fetch_series
# and terms that allow mirroring. Value: the variables harvested, first one is
# the default. Files are daily: sub-daily telemetry is folded per day (mean;
# rainfall summed, see _daily_agg). Units in the files: discharge m3/s,
# water_level m, groundwater_level m (UK EA: metres above Ordnance Datum),
# precipitation mm/day.
HARVESTABLE: dict[str, tuple[str, ...]] = {
    "usgs": ("discharge", "water_level"),
    "uk_ea": ("discharge", "water_level", "precipitation", "groundwater_level"),
    "hubeau_hydrometrie": ("discharge",),
    "taiwan_cwa": ("precipitation",),
    # 65 stations of 15-minute telemetry, CC BY-SA 4.0. Mirrored because the
    # browser cannot call system.openhi.net (CORS for localhost only, #408).
    "greece_openhi": ("discharge", "water_level", "precipitation"),
    # Daily from 1951 in hydrological-year zips the browser cannot fetch
    # (no CORS on the archive host); open re-use with attribution (#391).
    "poland_imgw": ("discharge", "water_level"),
}

ARCHIVE_UNITS = {"discharge": "m3/s", "water_level": "m", "groundwater_level": "m", "precipitation": "mm"}

MANIFEST_VERSION = 2


def harvestable_variables(source: str) -> tuple[str, ...]:
    """Variables the archive mirrors for ``source`` (empty when the source is not harvested)."""
    return HARVESTABLE.get(source, ())


def entry_key(source: str, variable: str) -> str:
    """Manifest key for one (source, variable) pair."""
    return f"{source}/{variable}"


@dataclass
class ObsHealth:
    source: str
    variable: str
    attempted: int = 0
    harvested: int = 0
    empty: int = 0
    failed: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    #: Why the source stopped before its budget of stations: "rate limited" or "time budget", else "".
    #: The stations it did not reach keep no manifest entry, so the next run picks them first.
    stopped: str = ""


@dataclass
class ObsReport:
    run_at: str
    aquascope_version: str
    sources: list[ObsHealth] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def obs_path(out_dir: Path, variable: str, source: str, station_id: str) -> Path:
    safe = station_id.replace("/", "_")
    return out_dir / "obs" / variable / source / f"{safe}.csv.gz"


def _migrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Version 1 keyed ``sources`` by source (one variable each); version 2 keys them by ``source/variable``."""
    if int(manifest.get("version") or 1) >= MANIFEST_VERSION:
        manifest.setdefault("bundles", {})
        return manifest
    sources = {}
    for key, entry in (manifest.get("sources") or {}).items():
        if "/" in key:
            sources[key] = entry
            continue
        variable = entry.get("variable") or (HARVESTABLE.get(key) or ("discharge",))[0]
        entry = dict(entry, source=key, variable=variable)
        sources[entry_key(key, variable)] = entry
    manifest["sources"] = sources
    manifest["version"] = MANIFEST_VERSION
    manifest.setdefault("bundles", {})
    return manifest


def load_manifest(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "obs" / "manifest.json"
    if p.exists():
        try:
            return _migrate_manifest(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError("manifest.json is unreadable; restore the last good manifest before harvesting") from exc
    return {"version": MANIFEST_VERSION, "sources": {}, "bundles": {}}


def save_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    p = out_dir / "obs" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for source in manifest.get("sources", {}).values():
        source["n_stations"] = sum(bool(row.get("n")) for row in source.get("stations", {}).values())
    temporary = p.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(p)
    return p


def _daily_agg(variable: str, s: pd.Series) -> str:
    """How sub-daily values fold into a day: rainfall totals sum, everything else averages.

    Rainfall that already arrives daily keeps the mean path so a missing day
    stays missing instead of becoming a zero total.
    """
    if variable != "precipitation" or len(s) < 2:
        return "mean"
    step = pd.Series(s.index).diff().dropna().median()
    return "sum" if step < pd.Timedelta(days=1) else "mean"

def sync_from_hub(out_dir: str | Path, repo_id: str, *, token: str | None = None) -> Path:
    """Sync observations and the last good catalog before incremental publication.

    A failed sync raises: publishing a fresh partial manifest over an existing
    archive would hide valid records even though their files still exist remotely.
    """
    from aquascope.utils.imports import require

    hub = require("huggingface_hub", feature="archive sync", group="archive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hub.snapshot_download(
        repo_id, repo_type="dataset", local_dir=str(out),
        allow_patterns=["obs/**", "stations.parquet", "stations.geojson", "health.json"], token=token,
    )
    load_manifest(out)  # a corrupt published manifest must fail the sync before any writes or publication
    return out


#: Stations the Explorer's daily live check reads (``.github/scripts/explorer_smoke.mjs``). Their mirror files are
#: harvested before anything else while they are new, stale or truncated, so the check does not depend on an agency
#: answering a browser (#501).
REFERENCE_STATIONS: frozenset[tuple[str, str]] = frozenset({
    ("usgs", "USGS-01013500"),                                  # Fish River near Fort Kent, from 1903
    ("uk_ea", "8496ce69-482c-406a-a2f0-ac418ef8f099"),           # Thames at Kingston, from 1883
    ("hubeau_hydrometrie", "F700000103"),                       # Seine at Paris-Austerlitz
})


def _pick_stations(
    catalog: list[dict[str, Any]],
    manifest: dict[str, Any],
    source: str,
    variable: str,
    max_stations: int,
    refresh_days: int,
    only: list[str] | None,
    years: int | None = None,
) -> list[dict[str, Any]]:
    done = manifest["sources"].get(entry_key(source, variable), {}).get("stations", {})
    cutoff = datetime.now(timezone.utc) - timedelta(days=refresh_days)
    # Only a capped harvest can skip a station: closed long before the window
    # it asks for, there is nothing to fetch. A full-record harvest (the
    # default, #270) mirrors closed stations too; their record is the point.
    closed_before = (
        (datetime.now(timezone.utc) - timedelta(days=int(years * 365.25))).date().isoformat() if years else None
    )
    fresh: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for row in catalog:
        if row["source"] != source or variable not in (row.get("variables") or []):
            continue
        if only and row["station_id"] not in only:
            continue
        end = row.get("period_end")
        if closed_before and end and str(end)[:10] < closed_before:
            continue  # closed long before the window we ask for: nothing to harvest
        entry = done.get(row["station_id"])
        if entry is None:
            fresh.append(row)
        else:
            try:
                stamp = (entry.get("last_attempt_at") if entry.get("last_attempt_status") in ("empty", "failed")
                         else entry.get("harvested_at"))
                when = datetime.fromisoformat(str(stamp or
                                                   "1970-01-01T00:00:00+00:00").replace("Z", "+00:00"))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
            except ValueError:
                when = datetime(1970, 1, 1, tzinfo=timezone.utc)
            if when < cutoff:
                stale.append(row)
    # A file that starts years after the catalog says the station does was harvested under the old 40-year cap
    # (#270); a full-record fetch now merges the earlier years in. Those go ahead of stations never harvested:
    # USGS alone catalogues ~25,000 of those, enough to fill every weekly budget, so behind them the truncated
    # files would wait years (#501). The reference stations go first of all.
    gap = {r["station_id"]: _missing_years(r, done.get(r["station_id"]) or {}) for r in stale}
    truncated = sorted((r for r in stale if gap[r["station_id"]] > 0), key=lambda r: -gap[r["station_id"]])
    rest = [r for r in stale if gap[r["station_id"]] == 0]
    ordered = truncated + fresh + rest
    reference = [r for r in ordered if (source, r["station_id"]) in REFERENCE_STATIONS]
    ordered = reference + [r for r in ordered if (source, r["station_id"]) not in REFERENCE_STATIONS]
    return ordered[:max_stations]


def _missing_years(row: dict[str, Any], entry: dict[str, Any]) -> float:
    """Years between the catalog's first date for a station and the first day its mirror file holds (0 when the
    file reaches back to the catalog's start, or either date is unknown)."""
    try:
        listed = datetime.fromisoformat(str(row.get("period_start") or "")[:10])
        held = datetime.fromisoformat(str(entry.get("first") or "")[:10])
    except ValueError:
        return 0.0
    gap = (held - listed).days / 365.25
    return gap if gap > 1.0 else 0.0


def _rate_limited(exc: BaseException) -> bool:
    """True when ``exc`` is, or was raised from, a 429 the HTTP client gave up on."""
    from aquascope.utils.http_client import RateLimitedError

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, RateLimitedError):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _harvest_one(
    out: Path,
    catalog: list[dict[str, Any]],
    manifest: dict[str, Any],
    key: str,
    var: str,
    fetch_series: Any,
    years: int | None,
    max_stations: int,
    refresh_days: int,
    only_stations: list[str] | None,
    max_seconds: float | None = None,
) -> ObsHealth:
    """Harvest one (source, variable) budget; updates ``manifest`` in place.

    Stops early, and says why in ``health.stopped``, when the agency rate-limits the run (its quota is spent,
    so every station after would fail too) or when ``max_seconds`` has passed, so one throttled agency costs
    its own rows and not the whole weekly run.
    """
    health = ObsHealth(source=key, variable=var)
    t0 = time.perf_counter()
    picked = _pick_stations(catalog, manifest, key, var, max_stations, refresh_days, only_stations, years)
    src_entry = manifest["sources"].setdefault(entry_key(key, var), {"source": key, "variable": var, "stations": {}})
    for row in picked:
        if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
            health.budget_exhausted = True
            health.stopped = "time budget"
            logger.warning("[%s/%s] time budget of %.0fs spent after %d stations; the rest wait for the next run",
                           key, var, max_seconds, health.attempted)
            break
        sid = row["station_id"]
        previous = src_entry["stations"].get(sid) or {}
        attempted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        health.attempted += 1
        try:
            extra = {"period_start": row["period_start"]} if row.get("period_start") else {}
            fetched = fetch_series(key, sid, years=years, prefer_archive=False, variable=var, **extra)
        except Exception as exc:  # noqa: BLE001 - one station must not sink the run
            health.failed += 1
            if len(health.errors) < 10:
                health.errors.append(f"{sid}: {type(exc).__name__}: {str(exc)[:160]}")
            logger.warning("[%s/%s] %s failed: %s", key, var, sid, exc)
            if _rate_limited(exc):
                health.stopped = "rate limited"
                logger.warning("[%s/%s] rate limited after %d stations; stopping this source for the run",
                               key, var, health.attempted)
                break
            src_entry["stations"][sid] = {**previous, "last_attempt_at": attempted_at,
                                          "last_attempt_status": "failed"}
            save_manifest(out, manifest)
            continue
        s = fetched["series"]
        if s is None or s.empty or fetched["variable"] != var:
            health.empty += 1
            src_entry["stations"][sid] = {**({"n": 0, "empty": True} if not previous else previous),
                                          "last_attempt_at": attempted_at, "last_attempt_status": "empty"}
            save_manifest(out, manifest)
            continue
        path = obs_path(out, var, key, sid)
        # Agencies sometimes return only the recent window. Preserve earlier data;
        # fresh values replace old values on overlapping days.
        if path.exists():
            old = read_csv_gz(path.read_bytes())
            fresh = read_csv_gz(series_to_csv_gz(s, agg=_daily_agg(var, s)))
            s = pd.concat([old, fresh])
            s = s[~s.index.duplicated(keep="last")].sort_index()
        payload = series_to_csv_gz(s, agg=_daily_agg(var, s))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".gz.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        daily_n = int(s.resample("D").mean().dropna().shape[0])
        entry = {
            "n": daily_n,
            "first": s.index.min().date().isoformat(),
            "last": s.index.max().date().isoformat(),
            "unit": fetched["unit"],
            "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_attempt_at": attempted_at, "last_attempt_status": "ok",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "file": str(path.relative_to(out)).replace("\\", "/"),
        }
        if fetched.get("note"):
            entry["note"] = str(fetched["note"])[:200]
        src_entry["stations"][sid] = entry
        health.harvested += 1
        save_manifest(out, manifest)
    health.seconds = round(time.perf_counter() - t0, 1)
    src_entry.update({
        "source": key,
        "variable": var,
        "unit": ARCHIVE_UNITS.get(var, ""),
        "license": SOURCES[key].license,
        "attribution": SOURCES[key].attribution,
        "n_stations": sum(1 for v in src_entry["stations"].values() if v.get("n")),
    })
    logger.info(
        "[%s/%s] %d harvested, %d empty, %d failed of %d picked in %.0fs%s",
        key, var, health.harvested, health.empty, health.failed, health.attempted, health.seconds,
        f" (stopped: {health.stopped})" if health.stopped else "",
    )
    return health


def harvest_observations(
    out_dir: str | Path,
    *,
    sources: list[str] | None = None,
    variable: str | None = None,
    years: int | None = None,
    max_stations: int = 100,
    refresh_days: int = 30,
    catalog: list[dict[str, Any]] | None = None,
    only_stations: list[str] | None = None,
    max_seconds: float | None = None,
) -> ObsReport:
    """Harvest up to ``max_stations`` stations per source and variable into ``out_dir/obs``.

    ``variable`` restricts the run to one variable (it must be harvestable for
    every source given); by default every variable in :data:`HARVESTABLE` is
    harvested for each source, each with its own budget and manifest cursor.
    ``years`` caps the record asked for to the last N years; by default the full
    record is requested, from the catalog's first date (#270). ``max_seconds``
    caps the time spent on each source and variable. ``catalog``
    defaults to the published station catalog. Failures are recorded per source
    and never raised. Returns an :class:`ObsReport`.
    """
    from aquascope.explore import fetch_series

    out = Path(out_dir)
    keys = sources or list(HARVESTABLE)
    for key in keys:
        if key not in HARVESTABLE:
            raise ValueError(f"{key} is not harvestable yet; choose from {list(HARVESTABLE)}")
        if not SOURCES[key].redistributable:
            raise ValueError(f"{key} is not marked redistributable in the registry; refusing to mirror it")
        if variable and variable not in HARVESTABLE[key]:
            raise ValueError(f"{key} does not harvest {variable!r}; it mirrors {list(HARVESTABLE[key])}")
    if catalog is None:
        from aquascope.archive.catalog import load_stations

        local_catalog = out / "stations.parquet"
        catalog = load_stations(path=local_catalog) if local_catalog.exists() else load_stations()

    manifest = load_manifest(out)
    report = ObsReport(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), aquascope_version=__version__)

    for key in keys:
        for var in (variable,) if variable else HARVESTABLE[key]:
            health = _harvest_one(out, catalog, manifest, key, var, fetch_series, years, max_stations,
                                  refresh_days, only_stations, max_seconds)
            report.sources.append(health)

    save_manifest(out, manifest)
    # last_run.json accumulates the sources of every invocation in a run
    # (the workflow calls harvest obs once per budget group).
    last_path = out / "obs" / "last_run.json"
    merged = report.to_dict()
    if last_path.exists():
        try:
            previous = json.loads(last_path.read_text(encoding="utf-8"))
            if previous.get("run_at", "")[:10] == merged["run_at"][:10]:
                mine = {(s["source"], s.get("variable")) for s in merged["sources"]}
                kept = [s for s in previous.get("sources", []) if (s["source"], s.get("variable")) not in mine]
                merged["sources"] = kept + merged["sources"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    temporary = last_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    temporary.replace(last_path)
    return report


def archive_series_url(source: str, station_id: str, variable: str, repo_id: str = "Rekin226/aquascope-gauges") -> str:
    safe = station_id.replace("/", "_")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/obs/{variable}/{source}/{safe}.csv.gz"


def fetch_archived_series(source: str, station_id: str, variable: str, *, timeout: float = 30) -> pd.Series | None:
    """Return the harvested daily series for a station, or None when the archive has none.

    Uses ``urllib`` on purpose: it works in CPython and, patched by
    ``pyodide_http``, inside the browser worker of the Explorer.
    """
    import urllib.error
    import urllib.request

    url = archive_series_url(source, station_id, variable)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            data = resp.read()
            revision = getattr(resp, "headers", {}).get("X-Repo-Commit")
        series = read_csv_gz(data)
        if isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
            series.attrs["archive_revision"] = revision
        return series
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            logger.info("archive read failed for %s/%s: HTTP %s", source, station_id, exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 - archive is a fast path, never a hard dependency
        logger.info("archive read failed for %s/%s: %s", source, station_id, exc)
        return None

def series_to_csv_gz(
        s: pd.Series,
        agg: str = "mean",
        quality: pd.Series | None = None,
    ) -> bytes:
        """Gzipped daily CSV. ``agg`` is "mean" (flows, levels) or "sum"
        (sub-daily rainfall totals).

        ``quality``, if given, must already be aligned to the SAME daily
        DatetimeIndex this function resamples ``s`` onto (i.e. already
        resolved to one quality value per calendar day) — this function
        does not itself decide how to collapse multiple same-day readings
        with different quality flags into one; that policy is the
        caller's job. Dates in ``s`` with no matching entry in ``quality``
        get "unknown".
        """
        daily = s.resample("D").sum(min_count=1).dropna() if agg == "sum" else s.resample("D").mean().dropna()
        buf = io.StringIO()
        if quality is not None:
            daily_quality = quality.reindex(daily.index)
            buf.write("date,value,quality\n")
            for d, v in daily.items():
                q = daily_quality.loc[d] if d in daily_quality.index else None
                # Quality is `str, Enum` — pull .value explicitly rather
                # than relying on str()/f-string on a live enum member
                # str() on a str, Enum member returns "Quality.APPROVED",
                if q is None or (isinstance(q, float) and pd.isna(q)):
                    q_str = "unknown"
                else:
                    q_str = q.value if hasattr(q, "value") else str(q)
                buf.write(f"{d.strftime('%Y-%m-%d')},{float(v):.6g},{q_str}\n")
        else:
            buf.write("date,value\n")
            for d, v in daily.items():
                buf.write(f"{d.strftime('%Y-%m-%d')},{float(v):.6g}\n")
        return gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)

def read_csv_gz(
    data: bytes,
    include_quality: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Read a gzipped daily CSV back to a value Series.

    With ``include_quality=True``, returns ``(value, quality)`` — the
    quality Series is all "unknown" when the file predates this
    column, so old archive files keep reading correctly either way.
    Default behaviour (a bare value Series) is UNCHANGED from before
    this column existed — every existing caller keeps working as-is.
    """
    df = pd.read_csv(io.BytesIO(data), compression="gzip", parse_dates=["date"])
    value = pd.Series(df["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name="value")
    if not include_quality:
        return value
    if "quality" in df.columns:
        quality = pd.Series(
            df["quality"].fillna("unknown").to_numpy(dtype=str),
            index=pd.DatetimeIndex(df["date"]),
            name="quality",
        )
    else:
        quality = pd.Series("unknown", index=value.index, name="quality")
    return value, quality
