"""Flow against normal (#559): every big river reach classed by today's flow against its normal for the month.

The NOAA National Water Model map colours every US reach by flow against normal. This does it for the world with
GEOGLOWS v2, in two parts:

1. **The climatology, once.** GEOGLOWS publishes flow duration curves for all 6.8 million reaches
   (``retrospective/fdc.zarr``, variable ``daily_monthly``): the percentiles of daily flow in each calendar month of
   the retrospective simulation. Checked on 2026-10-11 against the daily record of one Rhine reach: they are the
   percentiles of 1950 to 2019 (linear interpolation) to within 1e-7. For the reaches of Strahler order
   :data:`DEFAULT_MIN_ORDER` and up (the same set Floods ahead checks), the 10th, 25th, 75th and 90th percentiles
   of each month are kept and published under ``flowstatus/climatology/``: ``month-01.parquet`` to
   ``month-12.parquet`` (river_id, p10, p25, p75, p90), ``reaches.parquet`` (river_id, lat, lon, strahler_order)
   and ``manifest.json``.
2. **Today, daily.** The day's GEOGLOWS forecast gives each reach's first-day flow (the ensemble mean of the
   forecast's first UTC day). The flood-warnings workflow reads those chunks anyway, so ``warnings run --day0``
   writes the flows as it goes and nothing is downloaded twice. Each reach is classed against its percentiles for
   the forecast's month in the five WMO HydroSOS classes the river status layer uses, and published under
   ``flowstatus/``: ``latest.bin.gz`` (the compact file the Explorer colours the river lines from),
   ``latest.parquet`` (river_id, flow_cms, class), ``<YYYY-MM-DD>.bin.gz`` and ``manifest.json``.

The browser file, before gzip: ``AQFS``, a version byte (1), the number of reaches n (uint32, little endian), the
sorted river ids as n unsigned LEB128 varints of the gap from the previous id (the first from 0), then n class bytes
(1 much below to 5 much above). Reaches with no class are left out.

What it is not: measured. The flow is the model's forecast for today and the normal is the same model's simulated
record, so a reach the model gets wrong is wrong in both, and the class says where today sits in the model's own
climate. Only reaches of Strahler order 5 and up are classed.

Run by ``.github/workflows/flow-climatology.yml`` (once) and ``flood-warnings.yml`` (daily):

    python -m aquascope.flowstatus climatology --out build [--max-chunks N] [--bbox W S E N]
    python -m aquascope.flowstatus publish-climatology --out build
    python -m aquascope.archive.warnings run --out build --day0 build/scratch/day0.parquet
    python -m aquascope.flowstatus daily --out build --day0 build/scratch/day0.parquet
    python -m aquascope.flowstatus publish --out build

Without ``--day0`` the daily step reads the forecast itself (``--max-chunks`` or ``--bbox`` for a smoke run). A smoke
run is marked so, and publish refuses it. Needs the ``archive`` extra (pyarrow, huggingface_hub) and a Blosc decoder.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from aquascope.archive.warnings import DEFAULT_MIN_ORDER, DEFAULT_REPO, GEOGLOWS_BUCKET

logger = logging.getLogger(__name__)

FOLDER = "flowstatus"
CLIM_FOLDER = "flowstatus/climatology"
FDC_ZARR = f"{GEOGLOWS_BUCKET}/retrospective/fdc.zarr"
FDC_VARIABLE = "daily_monthly"
#: The percentiles kept, as non-exceedance; the store indexes by exceedance probability (100 minus these).
PERCENTILES = (10, 25, 75, 90)
#: Measured on 2026-10-11 (see the module docstring): the years the store's percentiles are taken over.
BASE_PERIOD = "1950 to 2019"
MAGIC = b"AQFS"
VERSION = 1

#: Codes 1 to 5 in the browser file, with the ids, words and bounds of the river status layer (map_layers.py).
CLASSES = [
    {"code": 1, "id": "much_below", "label": "much below normal", "range": "below the 10th percentile"},
    {"code": 2, "id": "below", "label": "below normal", "range": "10th to 25th percentile"},
    {"code": 3, "id": "normal", "label": "normal", "range": "25th to 75th percentile"},
    {"code": 4, "id": "above", "label": "above normal", "range": "75th to 90th percentile"},
    {"code": 5, "id": "much_above", "label": "much above normal", "range": "above the 90th percentile"},
]
CODE_OF = {c["id"]: c["code"] for c in CLASSES}
ID_OF = {c["code"]: c["id"] for c in CLASSES}

METHOD = (
    "For every GEOGLOWS v2 river reach of Strahler order {order} and up, the day's flow is the ensemble mean (51 "
    "members, the high-resolution run left out) of the global forecast's first UTC day. It is compared with the "
    "reach's 10th, 25th, 75th and 90th percentiles of daily flow in the same calendar month of the GEOGLOWS "
    "retrospective simulation, {period} (GEOGLOWS's retrospective/fdc.zarr): below the 10th is much below normal, "
    "10th to 25th below, 25th to 75th normal, 75th to 90th above and above the 90th much above. A flow equal to a "
    "bound counts towards normal. A reach whose 90th percentile is zero (dry nine days in ten that month) has no "
    "normal to compare with and is not classed."
)
NOT = (
    "Modelled, not measured: today's flow is the model's forecast and the normal is the same model's simulated record, "
    "so a reach the model gets wrong is wrong in both. Small rivers (below Strahler order {order}) are not classed. "
    "For the state of a river, follow your national hydrological service."
)
LICENCE = {
    "flow": "GEOGLOWS v2 forecast (GEOGloWS ECMWF Streamflow Service), CC BY 4.0",
    "normal": "GEOGLOWS v2 retrospective flow duration curves (retrospective/fdc.zarr), CC BY 4.0 (the AWS Open Data "
              "registry entry; the store carries no licence of its own)",
    "this_file": "CC BY 4.0",
    "geometry": "none stored: reaches are keyed by riverId, drawn on the GEOGLOWS stream tiles",
}


# ── the pure part ───────────────────────────────────────────────────────────


def classify(flow: float | None, p10: float | None, p25: float | None, p75: float | None,
             p90: float | None) -> int:
    """The class code (1 much below to 5 much above) of one flow against a month's percentiles, or 0 when it cannot
    be classed: a missing flow or percentile, or a channel whose 90th percentile is zero."""
    import math

    try:
        q, a, b, c, d = (float(x) for x in (flow, p10, p25, p75, p90))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if not all(math.isfinite(x) for x in (q, a, b, c, d)) or d <= 0:
        return 0
    if q < a:
        return 1
    if q < b:
        return 2
    if q > d:
        return 5
    if q > c:
        return 4
    return 3


def classify_array(flow: Any, p: Any) -> Any:
    """:func:`classify` for many reaches: ``flow`` (n,) against ``p`` (4, n) in :data:`PERCENTILES` order."""
    import numpy as np

    q = np.asarray(flow, dtype="float64")
    p = np.asarray(p, dtype="float64")
    out = np.zeros(q.shape, dtype="uint8")
    valid = np.isfinite(q) & np.isfinite(p).all(axis=0) & (p[3] > 0)
    with np.errstate(invalid="ignore"):
        out[valid] = 3
        out[valid & (q > p[2])] = 4
        out[valid & (q > p[3])] = 5
        out[valid & (q < p[1])] = 2
        out[valid & (q < p[0])] = 1
    return out


def _varints(values: Any) -> bytes:
    """Unsigned LEB128 for non-negative integers, vectorised: 7 bits a byte, the high bit set while more follow."""
    import numpy as np

    v = np.asarray(values, dtype="uint64")
    if not len(v):
        return b""
    nbytes = np.ones(len(v), dtype="int64")
    for k in range(1, 10):
        nbytes += v >= (np.uint64(1) << np.uint64(7 * k))
    out = np.zeros(int(nbytes.sum()), dtype="uint8")
    start = np.concatenate([[0], np.cumsum(nbytes)[:-1]])
    for k in range(int(nbytes.max())):
        has = nbytes > k
        byte = (v[has] >> np.uint64(7 * k)) & np.uint64(0x7F)
        more = (nbytes[has] > k + 1).astype("uint64") << np.uint64(7)
        out[start[has] + k] = (byte | more).astype("uint8")
    return out.tobytes()


def encode(ids: Any, classes: Any) -> bytes:
    """The browser file (gzip) for reaches ``ids`` with class codes ``classes``; class 0 is left out."""
    import numpy as np

    ids = np.asarray(ids, dtype="int64")
    classes = np.asarray(classes, dtype="uint8")
    keep = classes > 0
    ids, classes = ids[keep], classes[keep]
    order = np.argsort(ids, kind="stable")
    ids, classes = ids[order], classes[order]
    if len(ids) and (ids[0] < 0 or (np.diff(ids) <= 0).any()):
        raise ValueError("river ids must be non-negative and unique")
    gaps = np.diff(np.concatenate([[0], ids])) if len(ids) else ids
    body = MAGIC + bytes([VERSION]) + int(len(ids)).to_bytes(4, "little") + _varints(gaps) + classes.tobytes()
    return gzip.compress(body, compresslevel=9, mtime=0)


def decode(data: bytes) -> tuple[Any, Any]:
    """The reaches and class codes of a browser file (gzipped or not): (ids int64, classes uint8)."""
    import numpy as np

    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    if raw[:4] != MAGIC or raw[4] != VERSION:
        raise ValueError("not a flow status file (AQFS version 1)")
    n = int.from_bytes(raw[5:9], "little")
    ids = np.zeros(n, dtype="int64")
    pos, acc, i = 9, 0, 0
    while i < n:
        shift = val = 0
        while True:
            b = raw[pos]
            pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        acc += val
        ids[i] = acc
        i += 1
    classes = np.frombuffer(raw[pos:pos + n], dtype="uint8").copy()
    if len(classes) != n:
        raise ValueError("flow status file is cut short")
    return ids, classes


def counts_by_class(classes: Any) -> dict[str, int]:
    import numpy as np

    c = np.asarray(classes)
    return {k["id"]: int((c == k["code"]).sum()) for k in CLASSES}


def sentence(manifest: dict[str, Any], counts: dict[str, int] | None = None, where: str = "") -> str:
    """One plain line: the day, how many reaches, the share above and below normal. Every number from the counts."""
    if not manifest or not manifest.get("issue_date"):
        return "No Flow against normal issue is published yet."
    counts = counts if counts is not None else (manifest.get("counts") or {})
    n = sum(counts.values())
    if not n:
        return f"No river reach{where} is classed in the GEOGLOWS forecast of {manifest['issue_date']}."
    hi = counts.get("above", 0) + counts.get("much_above", 0)
    lo = counts.get("below", 0) + counts.get("much_below", 0)
    pct = lambda k: round(100 * k / n)  # noqa: E731
    return (f"GEOGLOWS forecast of {manifest['issue_date']}: of {n:,} river reaches{where}, {pct(hi)}% run above "
            f"normal for the month ({pct(counts.get('much_above', 0))}% much above) and {pct(lo)}% below "
            f"({pct(counts.get('much_below', 0))}% much below). Modelled, not measured.")


def in_bbox(lat: Any, lon: Any, bbox: list[float] | tuple[float, ...]) -> Any:
    """A mask of the points inside ``bbox`` (west, south, east, north), across the antimeridian when west > east."""
    import numpy as np

    if len(bbox) != 4:
        raise ValueError("bbox is west, south, east, north")
    west, south, east, north = (float(x) for x in bbox)
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    ok = (lat >= south) & (lat <= north)
    return ok & ((lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east))


# ── the climatology, once ───────────────────────────────────────────────────


def fdc_layout(meta: dict[str, Any]) -> dict[str, Any]:
    """Check the flow duration store is (month, p_exceed, river_id) with whole months and probabilities per chunk."""
    arr = meta.get(f"{FDC_VARIABLE}/.zarray")
    dims = (meta.get(f"{FDC_VARIABLE}/.zattrs") or {}).get("_ARRAY_DIMENSIONS")
    if not arr or dims != ["month", "p_exceed", "river_id"]:
        raise RuntimeError(f"unexpected flow duration layout: {FDC_VARIABLE} dims {dims}")
    shape, chunks = arr["shape"], arr["chunks"]
    if chunks[0] != shape[0] or chunks[1] != shape[1] or shape[0] != 12:
        raise RuntimeError(f"flow duration chunks {chunks} split months or probabilities; this reader wants them whole")
    return {"reaches": shape[2], "chunk": chunks[2], "zarray": arr}


def build_climatology(out: str | Path, *, min_order: int = DEFAULT_MIN_ORDER, max_chunks: int | None = None,
                      bbox: list[float] | None = None, workers: int = 8,
                      fetch: Callable[[str], bytes | None] | None = None,
                      tables: Callable[[Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write each month's percentiles for every reach of order ``min_order`` and up to ``<out>/flowstatus/climatology``.

    ``max_chunks`` (that many store chunks spread over the globe) or ``bbox`` makes a smoke run, marked so and refused
    by :func:`publish_climatology`.
    """
    import numpy as np

    from aquascope.archive import warnings as fw

    started = time.monotonic()
    fetch = fetch or fw.Fetcher()
    meta = fw._meta(fetch, FDC_ZARR)
    lay = fdc_layout(meta)
    p_exceed = [int(x) for x in fw._read_1d(fetch, FDC_ZARR, "p_exceed", meta)]
    months = [int(x) for x in fw._read_1d(fetch, FDC_ZARR, "month", meta)]
    if months != list(range(1, 13)):
        raise RuntimeError(f"unexpected months {months}")
    try:
        rows_at = [p_exceed.index(100 - p) for p in PERCENTILES]
    except ValueError as exc:
        raise RuntimeError(f"the store lacks an exceedance probability for {PERCENTILES}") from exc
    rid = fw._read_1d(fetch, FDC_ZARR, "river_id", meta).astype("int64")
    tab = (tables or fw.reach_tables)(rid)
    selected = np.asarray(tab["order"]) >= min_order
    if bbox is not None:
        selected &= in_bbox(tab["lat"], tab["lon"], bbox)
    c = lay["chunk"]
    sel_idx = np.nonzero(selected)[0]
    needed = [int(x) for x in np.unique(sel_idx // c)]
    chosen = fw.evenly(needed, max_chunks)
    logger.info("%d reaches (order >= %d) in %d chunks of %d; reading %d", len(sel_idx), min_order, len(needed), c,
                len(chosen))
    slot = np.full(len(rid), -1, dtype="int64")
    slot[sel_idx] = np.arange(len(sel_idx))
    pct: Any = np.full((12, len(PERCENTILES), len(sel_idx)), np.nan, dtype="float32")
    tally = {"read": 0, "failed": 0}
    lock = threading.Lock()

    def one(ch: int) -> None:
        try:
            data = fetch(f"{FDC_ZARR}/{FDC_VARIABLE}/0.0.{ch}")
            if data is None:
                raise OSError("missing chunk")
            block = fw.decode_chunk(data, lay["zarray"])
        except Exception as exc:  # noqa: BLE001 - one chunk failing must not stop the rest; it is counted
            logger.info("chunk %d failed: %s", ch, exc)
            with lock:
                tally["failed"] += 1
            return
        lo = ch * c
        local = np.nonzero(selected[lo:lo + c])[0]
        local = local[lo + local < len(rid)]
        pct[:, :, slot[lo + local]] = block[:, rows_at][:, :, local]
        with lock:
            tally["read"] += 1
            if tally["read"] % 500 == 0:
                logger.info("%d of %d chunks, %.0f s", tally["read"], len(chosen), time.monotonic() - started)

    if workers <= 1:
        for ch in chosen:
            one(ch)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, chosen))

    have = np.isfinite(pct).all(axis=(0, 1))
    ids = rid[sel_idx][have]
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    pct = pct[:, :, have][:, :, order]
    g = sel_idx[have][order]
    root = Path(out) / CLIM_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    for m in range(12):
        cols = {"river_id": ids, **{f"p{p}": pct[m, i] for i, p in enumerate(PERCENTILES)}}
        pq.write_table(pa.table(cols), root / f"month-{m + 1:02d}.parquet", compression="zstd")
    pq.write_table(pa.table({
        "river_id": ids,
        "lat": np.round(np.asarray(tab["lat"], dtype="float64")[g], 4).astype("float32"),
        "lon": np.round(np.asarray(tab["lon"], dtype="float64")[g], 4).astype("float32"),
        "strahler_order": np.asarray(tab["order"])[g].astype("int8"),
    }), root / "reaches.parquet", compression="zstd")
    smoke = max_chunks is not None or bbox is not None
    sizes = {p.name: p.stat().st_size for p in sorted(root.glob("*.parquet"))}
    doc = {
        "about": "Flow against normal (#559): each reach's 10th, 25th, 75th and 90th percentiles of daily flow by "
                 f"calendar month, for the reaches of Strahler order {min_order} and up.",
        "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"{FDC_ZARR}/", "variable": FDC_VARIABLE, "percentiles": list(PERCENTILES),
        "base_period": BASE_PERIOD, "min_strahler_order": min_order, "n": int(len(ids)),
        "no_percentiles": int((~have).sum()),
        "chunks": {"needed": len(needed), "read": tally["read"], "failed": tally["failed"], "planned": len(chosen),
                   "width": c},
        "bytes_read": getattr(fetch, "bytes", None), "seconds": round(time.monotonic() - started, 1),
        "smoke": smoke, "bbox": list(bbox) if bbox is not None else None, "sizes": sizes,
        "files": {"months": [f"{CLIM_FOLDER}/month-{m:02d}.parquet" for m in range(1, 13)],
                  "reaches": f"{CLIM_FOLDER}/reaches.parquet"},
        "licence": LICENCE["normal"],
    }
    (root / "manifest.json").write_text(json.dumps(doc, indent=1))
    keys = ("n", "no_percentiles", "chunks", "bytes_read", "seconds", "smoke", "sizes")
    return {k: doc[k] for k in keys}


def _publish(src: Path, folder: str, patterns: list[str], message: str, repo_id: str, token: str | None) -> str:
    from aquascope.archive.publish import publish_folder

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / folder
        stage.mkdir(parents=True)
        for p in src.iterdir():
            if p.is_file():
                shutil.copy2(p, stage / p.name)
        return str(publish_folder(Path(tmp), repo_id, token=token, allow_patterns=patterns, commit_message=message))


def publish_climatology(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/flowstatus/climatology`` and nothing else. A smoke run is refused."""
    src = Path(out) / CLIM_FOLDER
    manifest = src / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run the climatology step first")
    doc = json.loads(manifest.read_text())
    if doc.get("smoke"):
        raise ValueError("this is a smoke run (--max-chunks or --bbox); it is never published")
    return _publish(src, CLIM_FOLDER, [f"{CLIM_FOLDER}/*.parquet", f"{CLIM_FOLDER}/manifest.json"],
                    f"flowstatus/climatology: {doc['n']:,} reaches", repo_id, token)


# ── today, daily ────────────────────────────────────────────────────────────


def _climatology_file(name: str, climatology: str | Path | None, repo_id: str) -> bytes | None:
    if climatology:
        root = Path(climatology)
        root = root / CLIM_FOLDER if (root / CLIM_FOLDER).exists() else root
        p = root / name
        return p.read_bytes() if p.exists() else None
    from aquascope.archive.forecasts import _get, published_url

    return _get(published_url(f"{CLIM_FOLDER}/{name}", repo_id))


def read_month(month: int, *, climatology: str | Path | None = None, repo_id: str = DEFAULT_REPO) -> Any:
    """A month's percentiles as a DataFrame (river_id, p10, p25, p75, p90), sorted by river_id, or None."""
    import pyarrow.parquet as pq

    data = _climatology_file(f"month-{int(month):02d}.parquet", climatology, repo_id)
    if data is None:
        return None
    return pq.read_table(io.BytesIO(data)).to_pandas().sort_values("river_id", kind="stable").reset_index(drop=True)


def read_day0(*, ids: Any = None, max_chunks: int | None = None, run_name: str | None = None, workers: int = 8,
              fetch: Callable[[str], bytes | None] | None = None,
              today: date | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """The first-day flow of the forecast for ``ids`` (all reaches when None), read with the Floods ahead reader:
    (river_id, flow_cms, info). ``max_chunks`` reads that many chunks spread over the globe (a smoke run)."""
    import numpy as np

    from aquascope.archive import warnings as fw

    fetch = fetch or fw.Fetcher()
    name = run_name or fw.latest_run(fetch, today=today)
    base = f"{fw.FORECAST_BUCKET}/{name}.zarr"
    meta = fw._meta(fetch, base)
    lay = fw.forecast_layout(meta)
    rivid = fw._read_1d(fetch, base, "rivid", meta).astype("int64")
    seconds = fw._read_1d(fetch, base, "time", meta)
    ensemble = fw._read_1d(fetch, base, "ensemble", meta)
    members = np.array([i for i, e in enumerate(ensemble) if int(e) != fw.HIGH_RES_MEMBER])
    wanted = np.isin(rivid, np.asarray(ids, dtype="int64")) if ids is not None else np.ones(len(rivid), bool)
    c = lay["chunk"]
    needed = [int(x) for x in np.unique(np.nonzero(wanted)[0] // c)]
    chosen = fw.evenly(needed, max_chunks)
    flow: Any = np.full(len(rivid), np.nan, dtype="float32")
    tally = {"read": 0, "failed": 0}
    lock = threading.Lock()

    def one(ch: int) -> None:
        try:
            data = fetch(f"{base}/Qout/0.0.{ch}")
            if data is None:
                raise OSError("missing chunk")
            block = fw.decode_chunk(data, lay["zarray"])
        except Exception as exc:  # noqa: BLE001 - counted, and the reaches stay unclassed
            logger.info("chunk %d failed: %s", ch, exc)
            with lock:
                tally["failed"] += 1
            return
        lo = ch * c
        local = np.nonzero(wanted[lo:lo + c])[0]
        flow[lo + local] = fw.day0_flow(block[members][:, :, local], seconds)
        with lock:
            tally["read"] += 1

    if workers <= 1:
        for ch in chosen:
            one(ch)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, chosen))
    keep = np.isfinite(flow)
    info = {"run": name, "issue_date": lay["start"], "forecast": f"{base}/", "smoke": max_chunks is not None,
            "chunks": {"needed": len(needed), "read": tally["read"], "failed": tally["failed"],
                       "planned": len(chosen)},
            "bytes_read": getattr(fetch, "bytes", None)}
    return rivid[keep], flow[keep], info


def daily(out: str | Path, *, day0: str | Path | None = None, climatology: str | Path | None = None,
          repo_id: str = DEFAULT_REPO, max_chunks: int | None = None, bbox: list[float] | None = None,
          run_name: str | None = None, workers: int = 8, fetch: Callable[[str], bytes | None] | None = None,
          today: date | None = None) -> dict[str, Any]:
    """Class today's flow of every climatology reach and write ``<out>/flowstatus/``; return a summary.

    ``day0`` is the parquet ``warnings run --day0`` wrote; without it the forecast is read here. ``climatology`` is a
    folder written by :func:`build_climatology` (default: the published one).
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    clim_doc_raw = _climatology_file("manifest.json", climatology, repo_id)
    clim_doc = json.loads(clim_doc_raw) if clim_doc_raw else {}
    if not clim_doc:
        raise FileNotFoundError("no flowstatus/climatology is published yet: run flow-climatology.yml once "
                                "(python -m aquascope.flowstatus climatology, then publish-climatology)")
    if day0:
        tab = pq.read_table(day0).to_pandas()
        info_path = Path(day0).with_suffix(".json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        ids, flow = tab["river_id"].to_numpy("int64"), tab["flow_cms"].to_numpy("float32")
        bytes_read = None
    else:
        want = None
        if bbox is not None:
            rdata = _climatology_file("reaches.parquet", climatology, repo_id)
            if rdata is None:
                raise FileNotFoundError("the climatology has no reaches.parquet")
            r = pq.read_table(io.BytesIO(rdata)).to_pandas()
            want = r["river_id"].to_numpy("int64")[in_bbox(r["lat"], r["lon"], bbox)]
        ids, flow, info = read_day0(ids=want, max_chunks=max_chunks, run_name=run_name, workers=workers, fetch=fetch,
                                    today=today)
        info["smoke"] = bool(info.get("smoke")) or bbox is not None
        bytes_read = info.get("bytes_read")
    issue = info.get("issue_date") or ""
    if not issue:
        raise ValueError("the day-0 flows carry no issue date")
    month = int(issue[5:7])
    pct = read_month(month, climatology=climatology, repo_id=repo_id)
    if pct is None:
        raise FileNotFoundError(f"the climatology has no month-{month:02d}.parquet")
    flows = pd.Series(flow, index=pd.Index(ids, name="river_id"))
    flows = flows[~flows.index.duplicated()]
    q = flows.reindex(pct["river_id"].to_numpy("int64")).to_numpy("float32")
    p = np.vstack([pct[f"p{x}"].to_numpy("float32") for x in PERCENTILES])
    cls = classify_array(q, p)
    has_flow = np.isfinite(q)
    dry = has_flow & np.isfinite(p).all(axis=0) & ~(p[3] > 0)
    counts = counts_by_class(cls)
    smoke = bool(info.get("smoke")) or max_chunks is not None or bool(clim_doc.get("smoke"))
    root = Path(out) / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    blob = encode(pct["river_id"].to_numpy("int64"), cls)
    (root / "latest.bin.gz").write_bytes(blob)
    (root / f"{issue}.bin.gz").write_bytes(blob)
    keep = has_flow
    pq.write_table(pa.table({"river_id": pct["river_id"].to_numpy("int64")[keep], "flow_cms": q[keep],
                             "class": cls[keep]}), root / "latest.parquet", compression="zstd")
    history: list[dict[str, Any]] = []
    if not smoke:
        from aquascope.archive.forecasts import read_published_json

        published = read_published_json(f"{FOLDER}/manifest.json", repo_id)
        history = [h for h in published.get("history") or [] if h.get("issue_date") != issue]
    order = int(clim_doc.get("min_strahler_order") or DEFAULT_MIN_ORDER)
    doc = {
        "about": f"Flow against normal (#559): every GEOGLOWS v2 reach of Strahler order {order} and up classed by "
                 "its first-day forecast flow against its percentiles for the month. Built daily by AquaScope's "
                 "flood-warnings workflow.",
        "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue_date": issue, "run": info.get("run"), "forecast": info.get("forecast"), "month": month,
        "n": int((cls > 0).sum()), "counts": counts, "reaches": int(len(pct)),
        "no_flow": int((~has_flow).sum()), "dry": int(dry.sum()),
        "min_strahler_order": order, "classes": CLASSES, "percentiles": list(PERCENTILES),
        "base_period": clim_doc.get("base_period", BASE_PERIOD), "climatology": f"{CLIM_FOLDER}/manifest.json",
        "smoke": smoke, "chunks": info.get("chunks"), "bytes_read": bytes_read,
        "seconds": round(time.monotonic() - started, 1),
        "sizes": {"latest.bin.gz": len(blob), "latest.parquet": (root / "latest.parquet").stat().st_size},
        "format": "gzip of: AQFS, version byte 1, n (uint32 LE), n LEB128 varints (gap from the previous sorted "
                  "river id), n class bytes (1 much below .. 5 much above)",
        "method": METHOD.format(order=order, period=clim_doc.get("base_period", BASE_PERIOD)),
        "not": NOT.format(order=order), "licence": LICENCE,
        "files": {"latest": f"{FOLDER}/latest.bin.gz", "table": f"{FOLDER}/latest.parquet",
                  "dated": f"{FOLDER}/{issue}.bin.gz"},
        "history": sorted(history + [{"issue_date": issue, "file": f"{FOLDER}/{issue}.bin.gz", "counts": counts}],
                          key=lambda h: h["issue_date"]),
    }
    doc["sentence"] = sentence(doc)
    (root / "manifest.json").write_text(json.dumps(doc, indent=1))
    keys = ("issue_date", "run", "n", "counts", "reaches", "no_flow", "dry", "smoke", "sizes", "seconds", "sentence")
    return {k: doc[k] for k in keys}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload today's ``<out>/flowstatus`` files (not the climatology). A smoke run is refused."""
    src = Path(out) / FOLDER
    manifest = src / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run the daily step first")
    doc = json.loads(manifest.read_text())
    if doc.get("smoke"):
        raise ValueError("this is a smoke run; it is never published")
    return _publish(src, FOLDER, [f"{FOLDER}/*.bin.gz", f"{FOLDER}/latest.parquet", f"{FOLDER}/manifest.json"],
                    f"flowstatus: {doc['issue_date']}", repo_id, token)


# ── reading what is published (the MCP tool and the CLI) ───────────────────


def flow_vs_normal(bbox: list[float] | tuple[float, ...] | None = None, *, river_id: int | None = None,
                   repo_id: str = DEFAULT_REPO, local: str | Path | None = None) -> dict[str, Any]:
    """The published Flow against normal issue: when, counts by class (inside ``bbox``, west, south, east, north,
    when given) and a plain sentence; with ``river_id``, that reach's flow, class and percentiles for the month.
    ``local`` reads folders written by :func:`build_climatology` and :func:`daily` instead of the Archive."""
    import numpy as np
    import pyarrow.parquet as pq

    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox is west, south, east, north")
    if local:
        root = Path(local) / FOLDER
        mpath = root / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        blob = (root / "latest.bin.gz").read_bytes() if (root / "latest.bin.gz").exists() else None
    else:
        from aquascope.archive.forecasts import _get, published_url, read_published_json

        manifest = read_published_json(f"{FOLDER}/manifest.json", repo_id)
        blob = _get(published_url(f"{FOLDER}/latest.bin.gz", repo_id)) if manifest else None
    if not manifest or blob is None:
        return {"available": False, "sentence": "No Flow against normal issue is published yet; the daily workflow "
                "(flood-warnings.yml) writes flowstatus/ in the Archive dataset once the climatology is there."}
    ids, cls = decode(blob)
    where = ""
    if bbox is not None:
        rdata = _climatology_file("reaches.parquet", local, repo_id)
        if rdata is None:
            return {"available": False, "sentence": "The climatology's reach positions are not published yet."}
        r = pq.read_table(io.BytesIO(rdata), columns=["river_id", "lat", "lon"]).to_pandas()
        inside = r["river_id"].to_numpy("int64")[in_bbox(r["lat"], r["lon"], bbox)]
        m = np.isin(ids, inside)
        ids, cls = ids[m], cls[m]
        where = " in this box"
    counts = counts_by_class(cls)
    res: dict[str, Any] = {
        "available": True, "issue_date": manifest.get("issue_date"), "run": manifest.get("run"),
        "month": manifest.get("month"), "bbox": list(bbox) if bbox is not None else None,
        "n": int(len(ids)), "counts": counts, "sentence": sentence(manifest, counts, where),
        "min_strahler_order": manifest.get("min_strahler_order"), "base_period": manifest.get("base_period"),
        "method": manifest.get("method"), "not": manifest.get("not"), "licence": manifest.get("licence"),
        "smoke": manifest.get("smoke", False),
    }
    if river_id is not None:
        rid = int(river_id)
        i = int(np.searchsorted(ids, rid))
        code = int(cls[i]) if i < len(ids) and ids[i] == rid else 0
        reach: dict[str, Any] = {"river_id": rid, "class": ID_OF.get(code), "classed": bool(code)}
        month = manifest.get("month")
        pct = read_month(int(month), climatology=local, repo_id=repo_id) if month else None
        if pct is not None:
            row = pct[pct["river_id"] == rid]
            if len(row):
                reach.update({f"p{p}": float(f"{float(row.iloc[0][f'p{p}']):.4g}") for p in PERCENTILES})
        if local:
            t = Path(local) / FOLDER / "latest.parquet"
            table = pq.read_table(t).to_pandas() if t.exists() else None
        else:
            from aquascope.archive.forecasts import _get, published_url

            tdata = _get(published_url(f"{FOLDER}/latest.parquet", repo_id))
            table = pq.read_table(io.BytesIO(tdata)).to_pandas() if tdata else None
        if table is not None:
            row = table[table["river_id"] == rid]
            if len(row):
                reach["flow_cms"] = float(f"{float(row.iloc[0]['flow_cms']):.4g}")
        if not code:
            reach["why"] = ("not classed: below Strahler order {} or a channel dry most of the month"
                            .format(manifest.get("min_strahler_order")))
        res["reach"] = reach
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.flowstatus", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("climatology", help="the monthly percentiles, once")
    p.add_argument("--out", required=True)
    p.add_argument("--min-order", type=int, default=DEFAULT_MIN_ORDER)
    p.add_argument("--max-chunks", type=int, default=None, help="store chunks read, for a smoke run")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("W", "S", "E", "N"),
                   help="only the reaches in this box, for a smoke run")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("publish-climatology")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    p = sub.add_parser("daily", help="today's classes")
    p.add_argument("--out", required=True)
    p.add_argument("--day0", default=None, help="the parquet `warnings run --day0` wrote (else the forecast is read)")
    p.add_argument("--climatology", default=None, help="a folder with flowstatus/climatology (default: published)")
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    p.add_argument("--max-chunks", type=int, default=None, help="forecast chunks read, for a smoke run")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("W", "S", "E", "N"))
    p.add_argument("--run", default=None, help="a forecast run (YYYYMMDD00) instead of the newest")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "climatology":
        info = build_climatology(a.out, min_order=a.min_order, max_chunks=a.max_chunks, bbox=a.bbox,
                                 workers=a.workers)
    elif a.cmd == "publish-climatology":
        info = {"commit": publish_climatology(a.out, repo_id=a.repo)}
    elif a.cmd == "daily":
        info = daily(a.out, day0=a.day0, climatology=a.climatology, repo_id=a.repo, max_chunks=a.max_chunks,
                     bbox=a.bbox, run_name=a.run, workers=a.workers)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
