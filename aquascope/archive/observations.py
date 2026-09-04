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

# Sources with long daily records reachable through aquascope.explore.fetch_series
# and terms that allow mirroring. Value: the variables harvested, first one is
# the default. Units in the files: discharge m3/s, water_level m,
# groundwater_level m (UK EA: metres above Ordnance Datum), precipitation mm/day.
HARVESTABLE: dict[str, tuple[str, ...]] = {
    "usgs": ("discharge", "water_level"),
    "uk_ea": ("discharge", "water_level", "precipitation", "groundwater_level"),
    "hubeau_hydrometrie": ("discharge",),
    "taiwan_cwa": ("precipitation",),
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
        except json.JSONDecodeError:
            logger.warning("manifest.json unreadable; starting a fresh one")
    return {"version": MANIFEST_VERSION, "sources": {}, "bundles": {}}


def save_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    p = out_dir / "obs" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


def series_to_csv_gz(s: pd.Series) -> bytes:
    daily = s.resample("D").mean().dropna()
    buf = io.StringIO()
    buf.write("date,value\n")
    for d, v in daily.items():
        buf.write(f"{d.strftime('%Y-%m-%d')},{float(v):.6g}\n")
    return gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)


def read_csv_gz(data: bytes) -> pd.Series:
    df = pd.read_csv(io.BytesIO(data), compression="gzip", parse_dates=["date"])
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name="value")


def sync_from_hub(out_dir: str | Path, repo_id: str, *, token: str | None = None) -> Path:
    """Download the current ``obs/`` tree (manifest + files) so a run can be incremental."""
    from aquascope.utils.imports import require

    hub = require("huggingface_hub", feature="archive sync", group="archive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        hub.snapshot_download(
            repo_id, repo_type="dataset", local_dir=str(out), allow_patterns=["obs/**"], token=token,
        )
    except Exception as exc:  # noqa: BLE001 - a fresh dataset has no obs/ yet
        logger.info("no existing observations to sync (%s)", exc)
    return out


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
                when = datetime.fromisoformat(entry.get("harvested_at", "1970-01-01T00:00:00+00:00"))
            except ValueError:
                when = datetime(1970, 1, 1, tzinfo=timezone.utc)
            if when < cutoff:
                stale.append(row)
    return (fresh + stale)[:max_stations]


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
) -> ObsHealth:
    """Harvest one (source, variable) budget; updates ``manifest`` in place."""
    health = ObsHealth(source=key, variable=var)
    t0 = time.perf_counter()
    picked = _pick_stations(catalog, manifest, key, var, max_stations, refresh_days, only_stations, years)
    src_entry = manifest["sources"].setdefault(entry_key(key, var), {"source": key, "variable": var, "stations": {}})
    for row in picked:
        sid = row["station_id"]
        health.attempted += 1
        try:
            fetched = fetch_series(key, sid, years=years, prefer_archive=False, variable=var)
        except Exception as exc:  # noqa: BLE001 - one station must not sink the run
            health.failed += 1
            if len(health.errors) < 10:
                health.errors.append(f"{sid}: {type(exc).__name__}: {str(exc)[:160]}")
            logger.warning("[%s/%s] %s failed: %s", key, var, sid, exc)
            continue
        s = fetched["series"]
        if s is None or s.empty or fetched["variable"] != var:
            health.empty += 1
            src_entry["stations"][sid] = {
                "n": 0, "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "empty": True,
            }
            continue
        payload = series_to_csv_gz(s)
        path = obs_path(out, var, key, sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        daily_n = int(s.resample("D").mean().dropna().shape[0])
        entry = {
            "n": daily_n,
            "first": s.index.min().date().isoformat(),
            "last": s.index.max().date().isoformat(),
            "unit": fetched["unit"],
            "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file": str(path.relative_to(out)).replace("\\", "/"),
        }
        if fetched.get("note"):
            entry["note"] = str(fetched["note"])[:200]
        src_entry["stations"][sid] = entry
        health.harvested += 1
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
        "[%s/%s] %d harvested, %d empty, %d failed of %d picked in %.0fs",
        key, var, health.harvested, health.empty, health.failed, health.attempted, health.seconds,
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
) -> ObsReport:
    """Harvest up to ``max_stations`` stations per source and variable into ``out_dir/obs``.

    ``variable`` restricts the run to one variable (it must be harvestable for
    every source given); by default every variable in :data:`HARVESTABLE` is
    harvested for each source, each with its own budget and manifest cursor.
    ``years`` caps the record asked for to the last N years; by default the full
    record is requested, from the catalog's first date (#270). ``catalog``
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

        catalog = load_stations()

    manifest = load_manifest(out)
    report = ObsReport(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), aquascope_version=__version__)

    for key in keys:
        for var in (variable,) if variable else HARVESTABLE[key]:
            health = _harvest_one(out, catalog, manifest, key, var, fetch_series, years, max_stations,
                                  refresh_days, only_stations)
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
    last_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
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
        return read_csv_gz(data)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            logger.info("archive read failed for %s/%s: HTTP %s", source, station_id, exc.code)
        return None
    except Exception as exc:  # noqa: BLE001 - archive is a fast path, never a hard dependency
        logger.info("archive read failed for %s/%s: %s", source, station_id, exc)
        return None

