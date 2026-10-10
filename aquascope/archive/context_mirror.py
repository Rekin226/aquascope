"""Build the place-context mirrors in the Archive (#520): flood history, dams and the rain-gauge index.

Only datasets whose licence allows redistribution are mirrored:

* Groundsource flood events from news (Google, CC BY 4.0, Zenodo 18647054)
* the Microsoft AI for Good Sentinel-1 flood dataset 2014-2024 (MIT, Hugging Face), condensed to monthly
  detection counts on a 0.05 degree grid after the dataset's own recommended false-positive filters
* Global Dam Watch v1.0 barriers (CC BY 4.0, figshare 25988293)
* the GHCN-Daily stations that record precipitation (NOAA, CC0)

Global Water Watch is not mirrored: its data licence is not confirmed.

Layout under ``<out>/context/`` (the folder the Archive dataset publishes):

    manifest.json                               what is published, row counts, licences, the cells present
    floods/groundsource.parquet                 every event, sorted by 2-degree cell (bbox reads skip row groups)
    floods/groundsource/cells/<cell>.csv.gz     the same rows per cell, for the browser worker (no pyarrow there)
    floods/microsoft.parquet                    lat, lon, year, month, n (filtered 20 m detections per 0.05 deg cell)
    floods/microsoft/cells/<cell>.csv.gz
    dams/gdw_barriers.parquet
    dams/cells/<cell>.csv.gz
    ghcn/prcp_stations.csv.gz                   id, lat, lon, elev_m, name, first_year, last_year
    floods/monthly/                             both flood sources per half-degree cell and month (#547):
                                                grid.parquet, index.json, months/YYYY-MM.json.gz

Run by ``.github/workflows/mirror-context.yml``; every step is also a CLI subcommand:

    python -m aquascope.archive.context_mirror groundsource --out build
    python -m aquascope.archive.context_mirror microsoft-shard --out build --shard 0 --shards 16
    python -m aquascope.archive.context_mirror microsoft-merge --out build --parts parts/
    python -m aquascope.archive.context_mirror floods-monthly --out build
    python -m aquascope.archive.context_mirror dams --out build
    python -m aquascope.archive.context_mirror ghcn --out build
    python -m aquascope.archive.context_mirror manifest --out build
    python -m aquascope.archive.context_mirror publish --out build

Needs the ``archive`` extra (pyarrow, huggingface_hub) and, for Groundsource, shapely.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import struct
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope.context._common import CELL_DEG, MIRROR_FOLDER, cell_key, mirror_url
from aquascope.registry import CONTEXT_LAYERS

logger = logging.getLogger(__name__)

GROUNDSOURCE_URL = "https://zenodo.org/api/records/18647054/files/groundsource_2026.parquet/content"
GDW_SHP_URL = "https://ndownloader.figshare.com/files/47913754"
MS_REPO = "ai-for-good-lab/ai4g-flood-dataset"
GHCN_BASE = "https://noaa-ghcn-pds.s3.amazonaws.com"
#: The Microsoft dataset card's recommended filters for aggregating over long periods (its README).
MS_FILTERS = ("dem_metric_2 < 10", "soil_moisture_sca > 1", "soil_moisture_zscore > 1", "soil_moisture > 20",
              "temp > 0", "land_cover != 60", "edge_false_positives == 0")
MS_GRID = 0.05
ROW_GROUP = 50_000
#: What the context publish uploads: the Archive's usual files plus the monthly flood grid's gzipped JSON.
PUBLISH_PATTERNS = ("*.parquet", "*.json", "*.json.gz", "*.csv.gz", "README.md")

DATASET_LAYER = {"groundsource": "groundsource", "microsoft_floods": "microsoft_floods", "dams": "dams",
                 "ghcn": "rain_gauge"}


def _root(out: str | Path) -> Path:
    path = Path(out) / MIRROR_FOLDER
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path, *, retries: int = 5) -> Path:
    """Stream ``url`` to ``dest`` (following redirects), resuming on a dropped connection."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as resp:
                if resp.status_code == 416:
                    return dest
                resp.raise_for_status()
                mode = "ab" if have and resp.status_code == 206 else "wb"
                with dest.open(mode) as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
            return dest
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("download %s attempt %d failed: %s", url, attempt, exc)
            if attempt == retries:
                raise
    return dest


# ── writing ──────────────────────────────────────────────────────────────────


def _frame(rows: Any, columns: list[str]) -> Any:
    """A DataFrame of ``columns`` + ``cell``, sorted by cell then position (the order bbox reads rely on)."""
    import pandas as pd

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=columns + ["cell"])
    frame = frame[columns + ["cell"]]
    order = ["cell"] + [c for c in ("lat", "lon") if c in frame.columns]
    return frame.sort_values(order, kind="stable").reset_index(drop=True)


def _write_cells(frame: Any, folder: Path, columns: list[str]) -> list[str]:
    """One gzipped CSV per mirror cell; returns the cells written."""
    folder.mkdir(parents=True, exist_ok=True)
    cells = []
    for key, part in frame.groupby("cell", sort=True):
        text = part[columns].to_csv(index=False, lineterminator="\n")
        (folder / f"{key}.csv.gz").write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
        cells.append(str(key))
    return cells


def _write_parquet(frame: Any, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=ROW_GROUP,
                   compression="zstd")


def _info(name: str, frame: Any, cells: list[str], folder: str, **extra: Any) -> dict[str, Any]:
    meta = CONTEXT_LAYERS[DATASET_LAYER[name]]
    return {"rows": int(len(frame)), "cells": cells, "folder": folder, "cell_deg": CELL_DEG, "licence": meta.license,
            "attribution": meta.attribution, "source": meta.homepage,
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}


def _save_info(root: Path, name: str, info: dict[str, Any]) -> dict[str, Any]:
    (root / f"_{name}.json").write_text(json.dumps(info, indent=1))
    return info


# ── Groundsource ─────────────────────────────────────────────────────────────

GS_COLUMNS = ["uuid", "start_date", "end_date", "lat", "lon", "west", "south", "east", "north", "area_km2"]


def groundsource_frame(batch: Any) -> Any:
    """One Groundsource record batch as rows: the WKB footprint becomes its centre and its box."""
    import numpy as np
    import pandas as pd
    import shapely

    d = batch.to_pydict()
    geoms = shapely.from_wkb(d["geometry"])
    cents = shapely.centroid(geoms)
    bounds = shapely.bounds(geoms)
    frame = pd.DataFrame({
        "uuid": d["uuid"], "start_date": d["start_date"], "end_date": d["end_date"],
        "lat": np.round(shapely.get_y(cents), 5), "lon": np.round(shapely.get_x(cents), 5),
        "west": np.round(bounds[:, 0], 5), "south": np.round(bounds[:, 1], 5),
        "east": np.round(bounds[:, 2], 5), "north": np.round(bounds[:, 3], 5),
        "area_km2": pd.to_numeric(pd.Series(d["area_km2"], dtype="float64")).round(4),
    })
    frame = frame.dropna(subset=["lat", "lon"])
    frame["cell"] = [cell_key(la, lo) for la, lo in zip(frame["lat"], frame["lon"])]
    return frame


def build_groundsource(out: str | Path, *, source: str | Path | None = None, max_rows: int | None = None) -> dict:
    """Mirror Groundsource: download the Zenodo parquet (or read ``source``), write sorted parquet + cells."""
    import pandas as pd
    import pyarrow.parquet as pq

    root = _root(out)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(source) if source else download(GROUNDSOURCE_URL, Path(tmp) / "groundsource.parquet")
        pf = pq.ParquetFile(path)
        parts, n = [], 0
        for batch in pf.iter_batches(batch_size=200_000,
                                     columns=["uuid", "area_km2", "geometry", "start_date", "end_date"]):
            parts.append(groundsource_frame(batch))
            n += len(parts[-1])
            if max_rows and n >= max_rows:
                break
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=GS_COLUMNS + ["cell"])
    if max_rows:
        frame = frame.head(max_rows)
    frame = _frame(frame, GS_COLUMNS)
    _write_parquet(frame, root / "floods" / "groundsource.parquet")
    cells = _write_cells(frame, root / "floods" / "groundsource" / "cells", GS_COLUMNS)
    dates = frame["start_date"].dropna().astype(str)
    info = _info("groundsource", frame, cells, "floods/groundsource", parquet="floods/groundsource.parquet",
                 first=dates.min() if len(dates) else None, last=dates.max() if len(dates) else None,
                 note="lat/lon is the centre of the event's footprint; west..north its bounding box")
    return _save_info(root, "groundsource", info)


# ── Microsoft Sentinel-1 floods ──────────────────────────────────────────────

MS_COLUMNS = ["lat", "lon", "year", "month", "n"]
_MS_READ = ["year", "month", "lat", "lon", "dem_metric_2", "soil_moisture_sca", "soil_moisture_zscore", "soil_moisture",
            "temp", "land_cover", "edge_false_positives"]


def ms_files(api: Any = None) -> list[tuple[str, int]]:
    """(path, size) of every per-tile detection parquet in the Microsoft dataset."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    out = []
    for f in api.list_repo_tree(MS_REPO, repo_type="dataset", recursive=True):
        path = getattr(f, "path", "")
        if path.endswith("-post-processing.parquet"):
            out.append((path, int(getattr(f, "size", 0) or 0)))
    return sorted(out)


def shard_files(files: list[tuple[str, int]], shard: int, shards: int) -> list[str]:
    """Spread the files over ``shards`` by size (largest first onto the lightest shard)."""
    loads = [0] * shards
    owner: dict[str, int] = {}
    for path, size in sorted(files, key=lambda t: -t[1]):
        k = loads.index(min(loads))
        loads[k] += max(1, size)
        owner[path] = k
    return sorted(p for p, k in owner.items() if k == shard)


def ms_aggregate(frame: Any, grid: float = MS_GRID) -> Any:
    """Filter one batch of detections as the dataset card recommends and count them per grid cell and month."""
    import numpy as np
    import pandas as pd

    f = frame
    keep = ((f["dem_metric_2"] < 10) & (f["soil_moisture_sca"] > 1) & (f["soil_moisture_zscore"] > 1)
            & (f["soil_moisture"] > 20) & (f["temp"] > 0) & (f["land_cover"] != 60) & (f["edge_false_positives"] == 0))
    f = f[keep]
    if f.empty:
        return pd.DataFrame(columns=MS_COLUMNS)
    lat = np.floor(f["lat"].astype(float) / grid) * grid + grid / 2
    lon = np.floor(f["lon"].astype(float) / grid) * grid + grid / 2
    g = pd.DataFrame({"lat": lat.round(4), "lon": lon.round(4), "year": f["year"].astype(int),
                      "month": f["month"].astype(int)})
    return g.groupby(["lat", "lon", "year", "month"]).size().rename("n").reset_index()


def build_microsoft_shard(out: str | Path, *, shard: int, shards: int, max_files: int | None = None,
                          files: list[tuple[str, int]] | None = None) -> Path:
    """Aggregate one shard of the Microsoft tiles into ``<out>/ms_parts/part-<shard>.parquet``."""
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    files = files if files is not None else ms_files()
    mine = shard_files(files, shard, shards)[: max_files or None]
    parts = []
    for i, path in enumerate(mine, 1):
        with tempfile.TemporaryDirectory() as tmp:
            local = hf_hub_download(MS_REPO, path, repo_type="dataset", local_dir=tmp)
            pf = pq.ParquetFile(local)
            cols = [c for c in _MS_READ if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
                parts.append(ms_aggregate(batch.to_pandas()))
        if parts and len(parts) > 64:
            parts = [_resum(pd.concat(parts, ignore_index=True))]
        logger.info("microsoft shard %d: %d/%d %s", shard, i, len(mine), path)
    frame = _resum(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=MS_COLUMNS)
    dest = Path(out) / "ms_parts" / f"part-{shard:03d}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_to_arrow(frame), dest)
    return dest


def _resum(frame: Any) -> Any:
    if frame.empty:
        return frame
    return frame.groupby(["lat", "lon", "year", "month"], as_index=False)["n"].sum()


def _to_arrow(frame: Any) -> Any:
    import pyarrow as pa

    return pa.Table.from_pandas(frame[MS_COLUMNS].astype({"lat": float, "lon": float, "year": int, "month": int,
                                                          "n": int}), preserve_index=False)


def merge_microsoft(out: str | Path, parts: str | Path) -> dict:
    """Combine the shard parts into the mirror's parquet and cells."""
    import pandas as pd
    import pyarrow.parquet as pq

    root = _root(out)
    frames = [pq.read_table(p).to_pandas() for p in sorted(Path(parts).rglob("part-*.parquet"))]
    frame = _resum(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame(columns=MS_COLUMNS)
    frame = frame.astype({"lat": float, "lon": float, "year": int, "month": int, "n": int})
    frame["cell"] = [cell_key(la, lo) for la, lo in zip(frame["lat"], frame["lon"])]
    frame = _frame(frame, MS_COLUMNS)
    _write_parquet(frame, root / "floods" / "microsoft.parquet")
    cells = _write_cells(frame, root / "floods" / "microsoft" / "cells", MS_COLUMNS)
    info = _info("microsoft_floods", frame, cells, "floods/microsoft", parquet="floods/microsoft.parquet",
                 grid_deg=MS_GRID, filters=list(MS_FILTERS), shards=len(frames),
                 note="n = filtered 20 m flood detections in the 0.05 degree cell centred on lat/lon in that month")
    return _save_info(root, "microsoft_floods", info)


# ── Global Dam Watch ─────────────────────────────────────────────────────────

# The location is LAT_RIV/LONG_RIV: the point of the GIS layer, on the HydroSHEDS river, filled for every barrier.
# LAT_DAM/LONG_DAM is the surveyed dam position and is 0 for about 35,000 of the 41,145 barriers (checked
# 2026-10-08 against the shapefile's own point geometry, which equals LAT_RIV/LONG_RIV), so it is not used.
GDW_FIELDS = {"GDW_ID": "gdw_id", "DAM_NAME": "name", "RES_NAME": "reservoir", "RIVER": "river", "COUNTRY": "country",
              "YEAR_DAM": "year", "DAM_HGT_M": "height_m", "CAP_MCM": "capacity_mcm", "AREA_SKM": "area_km2",
              "MAIN_USE": "main_use", "DOR_PC": "dor_pc", "CATCH_SKM": "catchment_km2", "LAT_RIV": "lat",
              "LONG_RIV": "lon", "GRAND_ID": "grand_id"}
DAM_COLUMNS = list(GDW_FIELDS.values())


def read_dbf(data: bytes) -> Iterator[dict[str, Any]]:
    """Records of a dBASE III file (what a shapefile keeps its attributes in), numbers as floats."""
    nrec, hlen, reclen = struct.unpack("<IHH", data[4:12])
    fields = []
    pos = 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("ascii")
        fields.append((name, chr(data[pos + 11]), data[pos + 16]))
        pos += 32
    for i in range(nrec):
        rec = data[hlen + i * reclen: hlen + (i + 1) * reclen]
        if not rec or rec[:1] == b"*":
            continue
        off = 1
        row: dict[str, Any] = {}
        for name, kind, length in fields:
            raw = rec[off:off + length].decode("utf-8", errors="replace").strip()
            off += length
            if kind in ("N", "F"):
                try:
                    row[name] = float(raw) if raw else None
                except ValueError:
                    row[name] = None
            else:
                row[name] = raw or None
        yield row


def dam_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        r = {out: rec.get(src) for src, out in GDW_FIELDS.items()}
        lat, lon = r.get("lat"), r.get("lon")
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            continue
        for k in ("year", "height_m", "capacity_mcm", "area_km2", "dor_pc", "catchment_km2", "grand_id"):
            if isinstance(r.get(k), float) and r[k] < 0:  # GDW codes missing numbers as negative values
                r[k] = None
        if not r.get("grand_id"):  # 0 means "not in GRanD"
            r["grand_id"] = None
        for k in ("gdw_id", "year", "grand_id"):
            if isinstance(r.get(k), float):
                r[k] = int(r[k])
        r["cell"] = cell_key(lat, lon)
        rows.append(r)
    return rows


def build_dams(out: str | Path, *, source: str | Path | None = None) -> dict:
    """Mirror the Global Dam Watch v1.0 barriers from the shapefile zip's attribute table (LAT_RIV, LONG_RIV)."""
    root = _root(out)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(source) if source else download(GDW_SHP_URL, Path(tmp) / "GDW_v1_0_shp.zip")
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("GDW_barriers_v1_0.dbf"))
            frame = _frame(dam_rows(read_dbf(zf.read(name))), DAM_COLUMNS)
    frame = frame.astype({c: "Int64" for c in ("gdw_id", "year", "grand_id")})
    _write_parquet(frame, root / "dams" / "gdw_barriers.parquet")
    cells = _write_cells(frame, root / "dams" / "cells", DAM_COLUMNS)
    return _save_info(root, "dams", _info("dams", frame, cells, "dams", parquet="dams/gdw_barriers.parquet"))


# ── GHCN-Daily rain gauges ───────────────────────────────────────────────────

GHCN_COLUMNS = ["id", "lat", "lon", "elev_m", "name", "first_year", "last_year"]


def ghcn_prcp_stations(stations_txt: str, inventory_txt: str) -> list[dict[str, Any]]:
    """The GHCN-Daily stations with a PRCP record, with the first and last year of it."""
    from aquascope.context.gauges import parse_stations_txt

    span: dict[str, tuple[int, int]] = {}
    for line in inventory_txt.splitlines():
        if len(line) < 45 or line[31:35] != "PRCP":
            continue
        try:
            span[line[0:11].strip()] = (int(line[36:40]), int(line[41:45]))
        except ValueError:
            continue
    rows = []
    for st in parse_stations_txt(stations_txt):
        if st["id"] in span:
            st["first_year"], st["last_year"] = span[st["id"]]
            rows.append(st)
    return rows


def build_ghcn(out: str | Path, *, stations_txt: str | None = None, inventory_txt: str | None = None) -> dict:
    root = _root(out)
    if stations_txt is None or inventory_txt is None:
        with tempfile.TemporaryDirectory() as tmp:
            stations_txt = download(f"{GHCN_BASE}/ghcnd-stations.txt", Path(tmp) / "s.txt").read_text()
            inventory_txt = download(f"{GHCN_BASE}/ghcnd-inventory.txt", Path(tmp) / "i.txt").read_text()
    rows = ghcn_prcp_stations(stations_txt, inventory_txt)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(GHCN_COLUMNS)
    for r in rows:
        w.writerow(["" if r.get(c) is None else r.get(c) for c in GHCN_COLUMNS])
    (root / "ghcn").mkdir(parents=True, exist_ok=True)
    (root / "ghcn" / "prcp_stations.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
    return _save_info(root, "ghcn", _info("ghcn", rows, [], "ghcn", file="ghcn/prcp_stations.csv.gz"))


# ── Floods past: the monthly grid (#547) ─────────────────────────────────────


def _mirror_parquet(root: Path, rel: str, repo_id: str, tmp: Path, given: str | Path | None) -> Path:
    """A flood parquet: the one given, the one this run just built, or the published one."""
    if given:
        return Path(given)
    local = root / rel
    if local.exists():
        return local
    return download(mirror_url(rel, repo_id), tmp / Path(rel).name)


def build_floods_monthly(out: str | Path, *, repo_id: str = "Rekin226/aquascope-gauges",
                         groundsource: str | Path | None = None, microsoft: str | Path | None = None) -> dict:
    """Roll the two flood mirrors into the monthly half-degree grid under ``floods/monthly/`` (#547).

    Reads ``floods/groundsource.parquet`` and ``floods/microsoft.parquet`` from this run's build when they are
    there and from the published Archive otherwise, so it also runs on its own. Writes only under
    ``<out>/context/floods/monthly/``.
    """
    import shutil

    import pyarrow.parquet as pq

    from aquascope.context import floods_past as fp

    root = _root(out)
    with tempfile.TemporaryDirectory() as tmp:
        gs = _mirror_parquet(root, "floods/groundsource.parquet", repo_id, Path(tmp), groundsource)
        ms = _mirror_parquet(root, "floods/microsoft.parquet", repo_id, Path(tmp), microsoft)
        news = pq.read_table(gs, columns=["start_date", "lat", "lon"]).to_pandas()
        radar = pq.read_table(ms, columns=["lat", "lon", "year", "month", "n"]).to_pandas()
    frame = fp.grid_frame(news, radar)
    dates = news["start_date"].dropna().astype(str)
    del news, radar
    folder = root / "floods" / "monthly"
    if folder.exists():  # a month that has gone from the source must not linger
        shutil.rmtree(folder)
    (folder / "months").mkdir(parents=True)
    _write_parquet(frame, folder / "grid.parquet")
    payloads = fp.month_payloads(frame)
    for month, payload in payloads.items():
        (folder / "months" / f"{month}.json.gz").write_bytes(fp.encode_month(payload))
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = fp.index_payload(frame, news_first=dates.min() if len(dates) else None,
                             news_last=dates.max() if len(dates) else None, built=built)
    (folder / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    sizes = {"grid.parquet": (folder / "grid.parquet").stat().st_size,
             "index.json": (folder / "index.json").stat().st_size,
             "months": sum(p.stat().st_size for p in (folder / "months").glob("*.json.gz"))}
    gs_meta, ms_meta = CONTEXT_LAYERS["groundsource"], CONTEXT_LAYERS["microsoft_floods"]
    info = {"rows": int(len(frame)), "cells": [], "folder": "floods/monthly", "grid_deg": fp.GRID_DEG,
            "licence": f"{gs_meta.license} (news), {ms_meta.license} (radar)",
            "attribution": f"{gs_meta.attribution}; {ms_meta.attribution}",
            "source": gs_meta.homepage, "built": built, "parquet": "floods/monthly/grid.parquet",
            "index": "floods/monthly/index.json", "months": len(payloads),
            "first": index["first"], "last": index["last"], "bytes": sizes,
            "note": "news = Groundsource events that started in the month, centred in the cell; radar = filtered "
                    "Sentinel-1 20 m flood detections in the cell that month"}
    return _save_info(root, "floods_monthly", info)


# ── manifest and publish ─────────────────────────────────────────────────────


def _published_manifest(repo_id: str) -> dict[str, Any]:
    import httpx

    try:
        resp = httpx.get(mirror_url("manifest.json", repo_id), follow_redirects=True, timeout=60)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("no published context manifest: %s", exc)
    return {}


def write_manifest(out: str | Path, *, repo_id: str = "Rekin226/aquascope-gauges", merge: bool = True) -> dict:
    """``context/manifest.json`` from the datasets built here, kept together with those already published."""
    root = _root(out)
    manifest = _published_manifest(repo_id) if merge else {}
    datasets = dict(manifest.get("datasets") or {})
    for p in sorted(root.glob("_*.json")):
        datasets[p.stem[1:]] = json.loads(p.read_text())
    manifest = {
        "about": "Place-context mirrors for AquaScope (#520): only datasets whose licence allows redistribution.",
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cell_deg": CELL_DEG,
        "datasets": datasets,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def publish(out: str | Path, *, repo_id: str = "Rekin226/aquascope-gauges", token: str | None = None) -> str:
    """Upload ``<out>/context`` to the Archive dataset (needs HF_TOKEN with write access)."""
    import shutil

    from aquascope.archive.publish import publish_folder

    src = Path(out) / MIRROR_FOLDER
    for p in src.glob("_*.json"):
        p.unlink()
    names = ", ".join(sorted(json.loads((src / "manifest.json").read_text())["datasets"]))
    # Upload a folder holding only context/, so working files next to it (the Microsoft shard parts) stay local
    # and nothing lands outside context/ in the Archive.
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / MIRROR_FOLDER
        try:  # hard links: no second copy of a few hundred MB on the runner's disk
            shutil.copytree(src, stage, copy_function=os.link)
        except OSError:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.copytree(src, stage)
        return publish_folder(Path(tmp), repo_id, token=token, commit_message=f"context mirrors: {names}",
                              allow_patterns=PUBLISH_PATTERNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.context_mirror", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("groundsource", "dams", "ghcn", "manifest", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("groundsource", "dams"):
            p.add_argument("--source", default=None, help="a local copy instead of downloading")
        if name == "groundsource":
            p.add_argument("--max-rows", type=int, default=None)
        if name in ("manifest", "publish"):
            p.add_argument("--repo", default=os.environ.get("HF_DATASET", "Rekin226/aquascope-gauges"))
    p = sub.add_parser("microsoft-shard")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--max-files", type=int, default=None)
    p = sub.add_parser("microsoft-merge")
    p.add_argument("--out", required=True)
    p.add_argument("--parts", required=True)
    p = sub.add_parser("floods-monthly", help="the monthly half-degree flood grid (#547)")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", "Rekin226/aquascope-gauges"))
    p.add_argument("--groundsource", default=None, help="a local groundsource.parquet instead of the mirror's")
    p.add_argument("--microsoft", default=None, help="a local microsoft.parquet instead of the mirror's")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "groundsource":
        info = build_groundsource(a.out, source=a.source, max_rows=a.max_rows)
    elif a.cmd == "dams":
        info = build_dams(a.out, source=a.source)
    elif a.cmd == "ghcn":
        info = build_ghcn(a.out)
    elif a.cmd == "microsoft-shard":
        info = {"part": str(build_microsoft_shard(a.out, shard=a.shard, shards=a.shards, max_files=a.max_files))}
    elif a.cmd == "microsoft-merge":
        info = merge_microsoft(a.out, a.parts)
    elif a.cmd == "floods-monthly":
        info = build_floods_monthly(a.out, repo_id=a.repo, groundsource=a.groundsource, microsoft=a.microsoft)
    elif a.cmd == "manifest":
        info = write_manifest(a.out, repo_id=a.repo)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    summary = {k: v for k, v in info.items() if k != "cells"} if isinstance(info, dict) else info
    if isinstance(summary, dict) and "datasets" in summary:
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in summary["datasets"].items()}
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
