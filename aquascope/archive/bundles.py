"""Archive Phase 2: one Parquet bundle per (variable, source) on top of the per-station files.

``obs/<variable>/<source>/*.csv.gz`` is what the Explorer and the MCP tools
read (one station, one GET). Everyone else wants a whole source in one
``read_parquet``: notebooks, DuckDB, R (arrow), the CAMELS-TW build. So after
each harvest the folder is rolled up into ``obs/<variable>/<source>.parquet``
with three columns::

    station_id  string   the catalog id (join key to stations.parquet)
    date        date32   calendar day, UTC
    value       double   daily value in the archive unit for the variable

The bundle is rebuilt from scratch every run (the folder is the source of
truth) and recorded in ``obs/manifest.json`` under ``bundles``. Needs
``pyarrow`` (the ``archive`` extra); reading a bundle needs only pandas plus
pyarrow or DuckDB.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aquascope.archive.observations import ARCHIVE_UNITS, entry_key, load_manifest, save_manifest
from aquascope.registry import SOURCES

logger = logging.getLogger(__name__)


@dataclass
class BundleInfo:
    variable: str
    source: str
    file: str
    n_stations: int
    n_rows: int
    bytes: int
    first: str | None
    last: str | None
    unit: str
    built_at: str
    seconds: float


def bundle_path(out_dir: Path, variable: str, source: str) -> Path:
    return out_dir / "obs" / variable / f"{source}.parquet"


def bundle_url(source: str, variable: str, repo_id: str = "Rekin226/aquascope-gauges") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/obs/{variable}/{source}.parquet"


def _read_station_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def build_bundle(out_dir: str | Path, variable: str, source: str) -> BundleInfo | None:
    """Roll ``obs/<variable>/<source>/*.csv.gz`` into ``obs/<variable>/<source>.parquet``.

    Returns None (and writes nothing) when the folder has no station files.
    """
    from aquascope.utils.imports import require

    pa = require("pyarrow", feature="archive bundles", group="archive")
    pq = require("pyarrow.parquet", feature="archive bundles", group="archive")

    out = Path(out_dir)
    folder = out / "obs" / variable / source
    files = sorted(folder.glob("*.csv.gz")) if folder.is_dir() else []
    if not files:
        return None
    t0 = time.perf_counter()
    frames = []
    for f in files:
        sid = f.name[: -len(".csv.gz")]
        try:
            df = _read_station_frame(f)
        except Exception as exc:
            # A partial replacement would silently delete this station's old
            # observations. The caller preserves this source's last good bundle
            # and continues building other sources.
            raise ValueError(f"Cannot replace bundle with unreadable station file {f.name}") from exc
        if df.empty:
            continue
        df.insert(0, "station_id", sid)
        frames.append(df)
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True).sort_values(["station_id", "date"], kind="stable")
    schema = pa.schema([
        pa.field("station_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("value", pa.float64()),
    ])
    table = pa.Table.from_pandas(big, schema=schema, preserve_index=False)
    meta = SOURCES.get(source)
    table = table.replace_schema_metadata({
        b"aquascope": json.dumps({
            "variable": variable, "source": source, "unit": ARCHIVE_UNITS.get(variable, ""),
            "license": meta.license if meta else "", "attribution": meta.attribution if meta else "",
            "layout": "one row per station and calendar day; join station_id to stations.parquet",
        }).encode("utf-8"),
    })
    path = bundle_path(out, variable, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="snappy", row_group_size=200_000)
    temporary.replace(path)
    info = BundleInfo(
        variable=variable,
        source=source,
        file=str(path.relative_to(out)).replace("\\", "/"),
        n_stations=int(big["station_id"].nunique()),
        n_rows=int(len(big)),
        bytes=path.stat().st_size,
        first=str(big["date"].min()),
        last=str(big["date"].max()),
        unit=ARCHIVE_UNITS.get(variable, ""),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seconds=round(time.perf_counter() - t0, 1),
    )
    logger.info("bundled %s/%s: %d stations, %d rows, %.1f MB", variable, source, info.n_stations, info.n_rows,
                info.bytes / 1e6)
    return info


def build_bundles(
    out_dir: str | Path, *, variables: list[str] | None = None, sources: list[str] | None = None
) -> list[BundleInfo]:
    """Build every bundle the ``obs/`` tree can produce and record them in the manifest."""
    out = Path(out_dir)
    root = out / "obs"
    infos: list[BundleInfo] = []
    if not root.is_dir():
        return infos
    manifest = load_manifest(out)
    for var_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if variables and var_dir.name not in variables:
            continue
        for src_dir in sorted(p for p in var_dir.iterdir() if p.is_dir()):
            if sources and src_dir.name not in sources:
                continue
            key = entry_key(src_dir.name, var_dir.name)
            attempt = {"attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            try:
                info = build_bundle(out, var_dir.name, src_dir.name)
            except Exception as exc:  # noqa: BLE001 - preserve this bundle and continue with independent sources
                logger.warning("bundle %s failed; keeping previous bundle: %s", key, exc)
                manifest.setdefault("bundle_status", {})[key] = {
                    **attempt, "status": "failed", "error": str(exc),
                    "retained_previous": key in manifest.get("bundles", {}),
                }
                save_manifest(out, manifest)
                continue
            if info is None:
                manifest.setdefault("bundle_status", {})[key] = {**attempt, "status": "empty",
                    "retained_previous": key in manifest.get("bundles", {})}
                save_manifest(out, manifest)
                continue
            infos.append(info)
            manifest.setdefault("bundles", {})[key] = asdict(info)
            manifest.setdefault("bundle_status", {})[key] = {**attempt, "status": "ok"}
            save_manifest(out, manifest)
    save_manifest(out, manifest)
    return infos


def read_bundle(source_or_bytes: str | Path | bytes, *, station_ids: list[str] | None = None) -> pd.DataFrame:
    """Read a bundle (path or bytes) into a DataFrame with ``station_id, date, value``."""
    if isinstance(source_or_bytes, bytes):
        df = pd.read_parquet(io.BytesIO(source_or_bytes))
    else:
        df = pd.read_parquet(source_or_bytes)
    if station_ids:
        df = df[df["station_id"].isin(set(station_ids))]
    df = df.reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_observations(
    source: str,
    variable: str | None = None,
    *,
    station_ids: list[str] | None = None,
    repo_id: str = "Rekin226/aquascope-gauges",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (and cache for a day) the bundle for ``source``/``variable`` from the Hub.

    ``variable`` defaults to the source's first harvestable variable. Returns
    an empty frame when the archive has no bundle yet for that pair.
    """
    import httpx

    from aquascope.archive.catalog import _download, cache_dir
    from aquascope.archive.observations import HARVESTABLE

    if variable is None:
        variable = (HARVESTABLE.get(source) or ("discharge",))[0]
    local = cache_dir() / f"{repo_id.replace('/', '__')}__{variable}__{source}.parquet"
    try:
        _download(bundle_url(source, variable, repo_id), local, refresh)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return pd.DataFrame(columns=["station_id", "date", "value"])
        raise
    return read_bundle(local, station_ids=station_ids)


def to_series(df: pd.DataFrame, station_id: str) -> pd.Series:
    """One station's daily series out of a bundle frame."""
    sub = df[df["station_id"] == station_id]
    return pd.Series(sub["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(sub["date"]), name="value")


__all__: list[Any] = [
    "BundleInfo", "build_bundle", "build_bundles", "bundle_path", "bundle_url", "load_observations",
    "read_bundle", "to_series",
]
