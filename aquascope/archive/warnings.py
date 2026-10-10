"""Floods ahead (#546): the river reaches GEOGLOWS expects to pass their 2-year flow in the next 15 days.

Once a day this reads the whole global GEOGLOWS v2 forecast (``s3://geoglows-v2-forecasts/YYYYMMDD00.zarr``, 51
ensemble members, 15 days, 6.8 million reaches), keeps the reaches of Strahler order :data:`DEFAULT_MIN_ORDER` and up
(rivers draining a few thousand km2 and more) and every reach an Archive gauge sits on, and compares each one's
forecast with its own 2-, 5-, 10-, 25-, 50- and 100-year flows from the GEOGLOWS retrospective simulation. The reaches
whose ensemble-mean daily flow is expected to reach the 2-year flow are published, with the peak day, the
return-period class, the share of members that agree and the class on each of the 15 days.

What it is not: an official warning. It is model output (no gauge correction, no local knowledge, no rainfall
nowcast), and the thresholds come from the same model's simulated record, not from what any river really did.

Layout under ``<out>/forecasts/warnings/`` (the only folder this job writes or publishes):

    manifest.json          the issue date, counts by class, the method, the thresholds, what it is not, licences
    latest.parquet         one row per reach expected to pass its 2-year flow (every column, see COLUMNS)
    latest.geojson         the same reaches as points, slim, for the Explorer (at most GEOJSON_MAX)
    <YYYY-MM-DD>.parquet   the same table kept by the forecast's start date

Cost, measured on 2026-10-10 against the 2026-10-09 run: the forecast's Qout array is chunked 686 reaches wide with
every member and time step in a chunk (about 16 MB compressed each, 9,970 chunks, about 150 GB in all). The reaches
are stored so that the bigger rivers sit together: order 5 and up (842,581 reaches) is 3,401 chunks (about 54 GB),
order 6 and up is 1,654; the gauges' reaches add 57. A 300-chunk smoke run took 9.7 minutes on a home line.
Chunks are streamed, reduced and dropped, so disk use is nil and memory is a few hundred MB.

Run by ``.github/workflows/flood-warnings.yml``:

    python -m aquascope.archive.warnings run --out build [--min-order 5] [--max-chunks N]
    python -m aquascope.archive.warnings publish --out build

``--max-chunks`` is a smoke run: it reads N chunks spread over the globe and is marked so, and publish refuses it.
Needs the ``archive`` extra (pyarrow, huggingface_hub) and a Blosc decoder (``numcodecs`` or ``blosc2``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import warnings as _warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FOLDER = "forecasts/warnings"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
FORECAST_BUCKET = "https://geoglows-v2-forecasts.s3.us-west-2.amazonaws.com"
GEOGLOWS_BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com"
RETURN_PERIODS_ZARR = f"{GEOGLOWS_BUCKET}/retrospective/return-periods.zarr"
#: Return periods fitted to annual maxima of DAILY flow, compared with daily means of the forecast (like with like).
THRESHOLD_VARIABLE = "gumbel_daily"
#: Read in place with anonymous S3 range requests (pyarrow), only the columns used.
MODEL_TABLE = "geoglows-v2/tables/v2-model-table.parquet"
POSITIONS_TABLE = "geoglows-v2/tables/package-metadata-table.parquet"
RETURN_PERIODS = (2, 5, 10, 25, 50, 100)
#: Strahler order 5 is a median upstream area of about 3,200 km2 (order 6: 14,000 km2) in the GEOGLOWS model table.
DEFAULT_MIN_ORDER = 5
DAYS = 15
#: GEOGLOWS's ensemble coordinate runs 1 to 52; 52 is the high-resolution run, kept out of the ensemble statistics.
HIGH_RES_MEMBER = 52
#: A reach whose 2-year flow is below this (m3/s) is not classed, unless a gauge sits on it: at order 5 and up these
#: are mostly desert wadis, where a trickle of a few hundred litres a second "reaches the 2-year flow" (seen in the
#: 2026-10-09 run).
MIN_Q2_CMS = 5.0
#: The Explorer's file stays small (about 2 MB): past this many reaches it keeps the highest classes first, then the
#: largest peaks against the 2-year flow. The parquet keeps them all.
GEOJSON_MAX = 10000
#: A fresh run is written over a few hours; the newest one is used only once its metadata is there.
SEARCH_DAYS = 40

CLASSES = [
    {"id": 2, "label": "2-year flow", "text": "at or above the 2-year flow"},
    {"id": 5, "label": "5-year flow", "text": "at or above the 5-year flow"},
    {"id": 10, "label": "10-year flow", "text": "at or above the 10-year flow"},
    {"id": 25, "label": "25-year flow", "text": "at or above the 25-year flow"},
    {"id": 50, "label": "50-year flow", "text": "at or above the 50-year flow"},
    {"id": 100, "label": "100-year flow", "text": "at or above the 100-year flow"},
]
#: One character per day in ``daily``: the index into this tuple (0 = below the 2-year flow).
DAILY_CODES = (0, 2, 5, 10, 25, 50, 100)

METHOD = (
    "For every GEOGLOWS v2 river reach of Strahler order {order} and up, and every reach an Archive gauge sits on, "
    "the 51 ensemble members of the day's global "
    "forecast (the high-resolution run left out) are averaged over each UTC day of the 15 from the run's start; the "
    "ensemble mean's highest daily flow is the peak, and its class is the largest return period whose flow it reaches "
    "(2, 5, 10, 25, 50 or 100 years). The return-period flows are GEOGLOWS's own: a Gumbel distribution fitted by "
    "the method of moments to the annual maxima of daily flow in the retrospective simulation since 1940. Reaches "
    "whose 2-year flow is under {floor:g} m3/s (mostly dry desert channels) are not classed unless a gauge is on "
    "them. 'share' is the fraction of the 51 members whose own daily peak reaches the 2-year flow."
)
NOT = (
    "Model output, not an official warning: no gauge correction, no local knowledge, no forecaster. The thresholds "
    "are the model's own simulated floods, not observed ones, so a reach the model gets wrong is wrong in both. "
    "Small rivers (below Strahler order {order}) are not checked unless an Archive gauge is on them. For warnings, "
    "follow your national hydrological or meteorological service."
)
ABOUT = ("Floods ahead (#546): river reaches the GEOGLOWS v2 global forecast expects to reach their 2-year flow in "
         "the next 15 days. Built daily by AquaScope's flood-warnings workflow.")
LICENCE = {
    "forecast": "GEOGLOWS v2 forecast (GEOGloWS ECMWF Streamflow Service), CC BY 4.0",
    "thresholds": "GEOGLOWS v2 retrospective return periods; the dataset's own metadata says CC BY-NC-SA 4.0",
    "this_file": "CC BY-NC-SA 4.0, because it carries the return-period classes (non-commercial, share alike)",
    "geometry": "positions only (the reach's point in the GEOGLOWS tables); no TDX-Hydro line geometry is stored",
}
COLUMNS = {
    "river_id": "GEOGLOWS v2 (TDX-Hydro) reach id, the riverId of the stream network",
    "lat/lon": "the reach's point in the GEOGLOWS metadata table",
    "strahler_order": "the reach's Strahler order", "area_km2": "upstream area, km2",
    "peak_cms": "highest daily ensemble-mean flow in the 15 days, m3/s", "peak_date": "the day of that peak (UTC)",
    "rp": "the largest return period (years) whose flow the peak reaches",
    "share": "fraction of the 51 members whose own daily peak reaches the 2-year flow",
    "q2/q5/q10/q25/q50/q100": "the reach's return-period flows, m3/s",
    "daily": "one character per day from the run's start: 0 below the 2-year flow, 1 to 6 for 2, 5, 10, 25, 50, "
             "100 years",
    "gauges": "Archive gauges snapped to this reach (forecasts/reaches.parquet), as source/station_id",
}


# ── the pure part ───────────────────────────────────────────────────────────


def classify(peak: float | None, thresholds: dict[int, float] | list[float] | tuple[float, ...]) -> int:
    """The largest return period (years) whose flow ``peak`` reaches, or 0. ``thresholds`` maps return periods to
    flows, or lists the flows for :data:`RETURN_PERIODS` in order. A missing or non-positive 2-year flow gives 0."""
    import math

    if isinstance(thresholds, dict):
        q = {int(k): v for k, v in thresholds.items()}
    else:
        q = dict(zip(RETURN_PERIODS, thresholds))
    try:
        p = float(peak)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    q2 = q.get(2)
    if math.isnan(p) or q2 is None or not math.isfinite(float(q2)) or float(q2) <= 0:
        return 0
    best = 0
    for t in RETURN_PERIODS:
        v = q.get(t)
        if v is not None and math.isfinite(float(v)) and p >= float(v):
            best = t
    return best


def classify_array(peaks: Any, q: Any) -> Any:
    """:func:`classify` for many reaches: ``peaks`` (n,) against ``q`` (6, n) in :data:`RETURN_PERIODS` order."""
    import numpy as np

    peaks = np.asarray(peaks, dtype="float64")
    q = np.asarray(q, dtype="float64")
    out = np.zeros(peaks.shape, dtype="int16")
    valid = np.isfinite(peaks) & np.isfinite(q[0]) & (q[0] > 0)
    for i, t in enumerate(RETURN_PERIODS):
        with np.errstate(invalid="ignore"):
            out[valid & np.isfinite(q[i]) & (peaks >= q[i])] = t
    return out


def daily_means(values: Any, seconds: Any, days: int = DAYS) -> Any:
    """Average (members, steps, reaches) values over each UTC day from the run's start: (members, days, reaches).
    The members are 3-hourly and the steps in between are empty (NaN), so a plain NaN-mean per day is the day's
    mean; a day with no value at all stays NaN."""
    import numpy as np

    values = np.asarray(values)
    day_of = (np.asarray(seconds, dtype="int64") // 86400).astype(int)
    out = np.full((values.shape[0], days, values.shape[2]), np.nan, dtype="float32")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices are expected
        for d in range(days):
            steps = np.nonzero(day_of == d)[0]
            if len(steps):
                out[:, d, :] = np.nanmean(values[:, steps, :], axis=1)
    return out


def summarise(values: Any, seconds: Any, q: Any, days: int = DAYS) -> dict[str, Any]:
    """The 15-day picture for a block of reaches. ``values`` is (members, steps, reaches) of ensemble members only,
    ``q`` (6, reaches). Returns arrays over reaches: peak (ensemble-mean daily peak), peak_day (0-based), rp (class),
    share (members reaching the 2-year flow) and daily (days, reaches) classes."""
    import numpy as np

    daily = daily_means(values, seconds, days)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(daily, axis=0)                       # (days, reaches)
        member_peak = np.nanmax(daily, axis=1)                  # (members, reaches)
    has = np.isfinite(mean).any(axis=0)
    peak = np.where(has, np.nanmax(np.where(np.isfinite(mean), mean, -np.inf), axis=0), np.nan)
    peak_day = np.where(has, np.argmax(np.where(np.isfinite(mean), mean, -np.inf), axis=0), -1)
    q = np.asarray(q, dtype="float64")
    with np.errstate(invalid="ignore"):
        reach2 = member_peak >= q[0][None, :]
        n_members = np.isfinite(member_peak).sum(axis=0)
        share = np.where(n_members > 0, reach2.sum(axis=0) / np.maximum(n_members, 1), np.nan)
    daily_rp = np.stack([classify_array(mean[d], q) for d in range(mean.shape[0])])
    return {"peak": peak, "peak_day": peak_day, "rp": classify_array(peak, q), "share": share, "daily": daily_rp}


def daily_string(classes: Any) -> str:
    """One character per day: the index of each day's class in :data:`DAILY_CODES`."""
    index = {c: i for i, c in enumerate(DAILY_CODES)}
    return "".join(str(index.get(int(c), 0)) for c in classes)


def _sig(x: Any, digits: int = 3) -> float | None:
    import math

    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{digits}g}")


def evenly(items: list[Any], n: int | None) -> list[Any]:
    """``n`` items spread over the list (all of them when ``n`` is None or not smaller), so a smoke run sees every
    region rather than the first one."""
    if n is None or n >= len(items):
        return list(items)
    if n <= 0:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def counts_by_class(rps: list[int]) -> dict[str, int]:
    return {str(c["id"]): sum(1 for r in rps if r == c["id"]) for c in CLASSES}


def sentence(manifest: dict[str, Any], n: int | None = None, where: str = "") -> str:
    """One plain line about an issue: when, how many reaches, how big the biggest class is."""
    if not manifest or not manifest.get("issue_date"):
        return "No Floods ahead issue is published yet."
    counts = manifest.get("counts") or {}
    n = sum(counts.values()) if n is None else n
    start = manifest["issue_date"]
    if not n:
        return (f"The GEOGLOWS forecast from {start} expects no checked river reach{where} to reach its 2-year flow "
                "in the next 15 days.")
    top = max((int(k) for k, v in counts.items() if v), default=2)
    big = sum(v for k, v in counts.items() if int(k) >= 10)
    tail = f", {big:,} of them the 10-year flow or more" if big else ""
    return (f"The GEOGLOWS forecast from {start} expects {n:,} river reach{'es' if n != 1 else ''}{where} to reach "
            f"the 2-year flow within 15 days{tail}; the largest class is the {top}-year flow. Model output, not an "
            "official warning.")


# ── reading the forecast ────────────────────────────────────────────────────


def _blosc_decompress(buf: bytes) -> bytes:
    try:
        from numcodecs import blosc

        return bytes(blosc.decompress(buf))
    except ImportError:
        pass
    try:
        import blosc2

        return bytes(blosc2.decompress(buf))
    except ImportError as exc:
        raise ImportError("Reading the GEOGLOWS Zarr needs a Blosc decoder: pip install numcodecs (or blosc2)") from exc


def decode_chunk(buf: bytes, meta: dict[str, Any]) -> Any:
    """A Zarr v2 chunk as a numpy array of the chunk's full shape (edge chunks are padded in v2)."""
    import numpy as np

    comp = meta.get("compressor") or {}
    if comp and comp.get("id") != "blosc":
        raise ValueError(f"unexpected Zarr compressor {comp.get('id')!r}")
    raw = _blosc_decompress(buf) if comp else buf
    return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["chunks"], order=meta.get("order", "C"))


class Fetcher:
    """GET with a shared connection pool, a few retries and a byte count."""

    def __init__(self, timeout: float = 120.0, retries: int = 3) -> None:
        import httpx

        self._client = httpx.Client(timeout=timeout, follow_redirects=True,
                                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32))
        self.retries = retries
        self.bytes = 0
        self._lock = threading.Lock()

    def __call__(self, url: str) -> bytes | None:
        import httpx

        for attempt in range(self.retries):
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                logger.info("GET %s failed (%s), attempt %d", url, exc, attempt + 1)
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                with self._lock:
                    self.bytes += len(resp.content)
                return bytes(resp.content)
            logger.info("GET %s answered %d, attempt %d", url, resp.status_code, attempt + 1)
            time.sleep(2 * (attempt + 1))
        raise OSError(f"{url} did not answer after {self.retries} tries")


def latest_run(fetch: Callable[[str], bytes | None], *, today: date | None = None) -> str:
    """The newest ``YYYYMMDD00`` forecast run whose consolidated metadata is published."""
    today = today or datetime.now(timezone.utc).date()
    after = (today - timedelta(days=SEARCH_DAYS)).strftime("%Y%m%d")
    xml = fetch(f"{FORECAST_BUCKET}/?list-type=2&delimiter=/&start-after={after}")
    names = sorted(set(re.findall(r"<Prefix>(\d{10})\.zarr/</Prefix>", (xml or b"").decode("utf-8", "replace"))))
    for name in reversed(names):
        if fetch(f"{FORECAST_BUCKET}/{name}.zarr/.zmetadata"):
            return str(name)
    raise RuntimeError(f"no GEOGLOWS forecast run with metadata found since {after}")


def _meta(fetch: Callable[[str], bytes | None], base: str) -> dict[str, Any]:
    data = fetch(f"{base}/.zmetadata")
    if not data:
        raise RuntimeError(f"{base}/.zmetadata is missing")
    meta: dict[str, Any] = json.loads(data)["metadata"]
    return meta


def _read_1d(fetch: Callable[[str], bytes | None], base: str, var: str, meta: dict[str, Any]) -> Any:
    import numpy as np

    arr = meta[f"{var}/.zarray"]
    n, c = arr["shape"][0], arr["chunks"][0]
    parts = []
    for i in range(-(-n // c)):
        data = fetch(f"{base}/{var}/{i}")
        if data is None:
            raise RuntimeError(f"{base}/{var}/{i} is missing")
        parts.append(decode_chunk(data, arr).reshape(-1))
    return np.concatenate(parts)[:n]


def forecast_layout(meta: dict[str, Any]) -> dict[str, Any]:
    """Check the forecast's Qout is (ensemble, time, rivid) with whole members and steps in each chunk."""
    q = meta.get("Qout/.zarray")
    dims = (meta.get("Qout/.zattrs") or {}).get("_ARRAY_DIMENSIONS")
    if not q or dims != ["ensemble", "time", "rivid"]:
        raise RuntimeError(f"unexpected forecast layout: Qout dims {dims}")
    shape, chunks = q["shape"], q["chunks"]
    if chunks[0] != shape[0] or chunks[1] != shape[1]:
        raise RuntimeError(f"forecast chunks {chunks} split members or steps; this reader expects them whole")
    units = (meta.get("time/.zattrs") or {}).get("units", "")
    m = re.match(r"seconds since (\d{4}-\d{2}-\d{2})", units)
    if not m:
        raise RuntimeError(f"unexpected time units {units!r}")
    return {"members": shape[0], "steps": shape[1], "reaches": shape[2], "chunk": chunks[2], "start": m.group(1),
            "zarray": q}


def _s3_columns(path: str, columns: list[str]) -> Any:
    from pyarrow import fs as pafs
    from pyarrow import parquet as pq

    s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    return pq.read_table(path, filesystem=s3, columns=columns).to_pandas()


def reach_tables(rivid: Any) -> dict[str, Any]:
    """Strahler order, upstream area and position for every forecast reach, aligned to ``rivid``."""
    import numpy as np
    import pandas as pd

    model = _s3_columns(MODEL_TABLE, ["LINKNO", "strmOrder", "DSContArea"]).set_index("LINKNO")
    pos = _s3_columns(POSITIONS_TABLE, ["LINKNO", "lat", "lon"]).set_index("LINKNO")
    idx = pd.Index(np.asarray(rivid, dtype="int64"))
    m = model.reindex(idx)
    p = pos.reindex(idx)
    return {"order": m["strmOrder"].fillna(0).to_numpy(dtype="int16"),
            "area_km2": (m["DSContArea"] / 1e6).to_numpy(dtype="float64"),
            "lat": p["lat"].to_numpy(dtype="float64"), "lon": p["lon"].to_numpy(dtype="float64")}


def read_thresholds(fetch: Callable[[str], bytes | None], rivid: Any, wanted: Any) -> Any:
    """GEOGLOWS's return-period flows (6, n) for the forecast's reaches, NaN except where ``wanted``."""
    import numpy as np

    meta = _meta(fetch, RETURN_PERIODS_ZARR)
    rp = _read_1d(fetch, RETURN_PERIODS_ZARR, "return_period", meta)
    if [int(x) for x in rp] != list(RETURN_PERIODS):
        raise RuntimeError(f"unexpected return periods {list(rp)}")
    river = _read_1d(fetch, RETURN_PERIODS_ZARR, "river_id", meta)
    arr = meta[f"{THRESHOLD_VARIABLE}/.zarray"]
    c = arr["chunks"][1]
    out: Any = np.full((len(RETURN_PERIODS), len(rivid)), np.nan, dtype="float32")
    if len(river) == len(rivid) and np.array_equal(river, rivid):
        pos = np.arange(len(rivid))
    else:  # same reaches in another order: map by id
        lookup = {int(r): i for i, r in enumerate(river)}
        pos = np.array([lookup.get(int(r), -1) for r in rivid])
    want = np.nonzero(np.asarray(wanted))[0]
    src = pos[want]
    keep = src >= 0
    want, src = want[keep], src[keep]
    for ch in np.unique(src // c):
        data = fetch(f"{RETURN_PERIODS_ZARR}/{THRESHOLD_VARIABLE}/0.{int(ch)}")
        if data is None:
            continue
        block = decode_chunk(data, arr)
        sel = (src // c) == ch
        out[:, want[sel]] = block[:, src[sel] - int(ch) * c]
    return out


def gauges_by_reach(repo_id: str = DEFAULT_REPO) -> dict[int, list[str]]:
    """The Archive gauges the forecast job snapped to a reach (forecasts/reaches.parquet), by river_id."""
    from aquascope.archive.forecasts import read_published_rows

    out: dict[int, list[str]] = {}
    try:
        rows = read_published_rows("forecasts/reaches.parquet", repo_id)
    except Exception as exc:  # noqa: BLE001 - the gauges are a nicety, never the failure
        logger.info("no gauge reaches: %s", exc)
        return out
    for r in rows:
        if r.get("river_id") is not None:
            out.setdefault(int(r["river_id"]), []).append(f"{r['source']}/{r['station_id']}")
    return out


# ── the run ─────────────────────────────────────────────────────────────────


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("river_id", pa.int64()), ("lat", pa.float64()), ("lon", pa.float64()),
                        ("strahler_order", pa.int16()), ("area_km2", pa.float64()), ("peak_cms", pa.float64()),
                        ("peak_date", pa.string()), ("rp", pa.int16()), ("share", pa.float64()),
                        *[(f"q{t}", pa.float64()) for t in RETURN_PERIODS], ("daily", pa.string()),
                        ("gauges", pa.list_(pa.string()))])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def to_geojson(rows: list[dict[str, Any]], cap: int = GEOJSON_MAX) -> tuple[dict[str, Any], bool]:
    """The reaches as slim points for the browser, highest class (then the largest peak over 2-year flow) first."""
    import math

    # A reach with no position cannot be drawn, and a NaN would make the file invalid JSON for the browser.
    placed = [r for r in rows if math.isfinite(r.get("lat", math.nan)) and math.isfinite(r.get("lon", math.nan))]
    ranked = sorted(placed, key=lambda r: (-r["rp"], -(r["peak_cms"] or 0) / max(r["q2"] or 1e-9, 1e-9)))
    feats = []
    for r in ranked[:cap]:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(r["lon"], 4),
                                                                                       round(r["lat"], 4)]},
                      "properties": {"river_id": r["river_id"], "rp": r["rp"], "peak": r["peak_cms"], "q2": r["q2"],
                                     "day": r["peak_date"], "share": r["share"], "order": r["strahler_order"],
                                     "daily": r["daily"], "gauges": ";".join(r.get("gauges") or [])}})
    return {"type": "FeatureCollection", "features": feats}, len(placed) > cap


def run(out: str | Path, *, repo_id: str = DEFAULT_REPO, min_order: int = DEFAULT_MIN_ORDER,
        min_q2: float = MIN_Q2_CMS, max_chunks: int | None = None, workers: int = 8,
        time_budget_s: float = 4.5 * 3600,
        run_name: str | None = None, fetch: Callable[[str], bytes | None] | None = None,
        tables: Callable[[Any], dict[str, Any]] | None = None,
        gauges: dict[int, list[str]] | None = None, today: date | None = None) -> dict[str, Any]:
    """Build today's Floods ahead issue under ``<out>/forecasts/warnings/`` and return a summary.

    ``max_chunks`` makes a smoke run: that many forecast chunks, spread over the globe, and a manifest marked
    ``smoke`` that :func:`publish` refuses. Chunks past the time budget are skipped and counted.
    """
    import numpy as np

    started = time.monotonic()
    fetch = fetch or Fetcher()
    name = run_name or latest_run(fetch, today=today)
    base = f"{FORECAST_BUCKET}/{name}.zarr"
    meta = _meta(fetch, base)
    lay = forecast_layout(meta)
    rivid = _read_1d(fetch, base, "rivid", meta)
    seconds = _read_1d(fetch, base, "time", meta)
    ensemble = _read_1d(fetch, base, "ensemble", meta)
    members = np.array([i for i, e in enumerate(ensemble) if int(e) != HIGH_RES_MEMBER])
    start = date.fromisoformat(lay["start"])
    logger.info("forecast %s: %d members (%d in the ensemble), %d steps, %d reaches", name, len(ensemble),
                len(members), len(seconds), len(rivid))

    tab = (tables or reach_tables)(rivid)
    # Every reach an Archive gauge sits on is checked too, whatever its order, so the gauge can say so
    # (805 reaches on 2026-10-10, 57 more chunks).
    by_reach = gauges if gauges is not None else gauges_by_reach(repo_id)
    on_gauge = np.isin(np.asarray(rivid, dtype="int64"), np.array(sorted(by_reach), dtype="int64"))
    selected = (tab["order"] >= min_order) | on_gauge
    c = lay["chunk"]
    sel_idx = np.nonzero(selected)[0]
    needed = [int(x) for x in np.unique(sel_idx // c)]
    chosen = evenly(needed, max_chunks)
    logger.info("%d reaches (order >= %d, or with a gauge) in %d chunks; reading %d", len(sel_idx), min_order,
                len(needed), len(chosen))
    in_chosen: Any = np.zeros(len(rivid), dtype=bool)
    for ch in chosen:
        in_chosen[ch * c:(ch + 1) * c] = True
    q = read_thresholds(fetch, rivid, selected & in_chosen)

    rows: list[dict[str, Any]] = []
    tally = {"checked": 0, "no_threshold": 0, "below_floor": 0, "failed": 0, "skipped_time": 0, "read": 0}
    lock = threading.Lock()
    zarr = lay["zarray"]

    def one(ch: int) -> None:
        if time.monotonic() - started > time_budget_s:
            with lock:
                tally["skipped_time"] += 1
            return
        try:
            data = fetch(f"{base}/Qout/0.0.{ch}")
            if data is None:
                raise OSError("missing chunk")
            block = decode_chunk(data, zarr)
        except Exception as exc:  # noqa: BLE001 - one chunk failing must not stop the issue
            logger.info("chunk %d failed: %s", ch, exc)
            with lock:
                tally["failed"] += 1
            return
        lo = ch * c
        local = np.nonzero(selected[lo:lo + c])[0]
        glob = lo + local
        qq = q[:, glob]
        has_q = np.isfinite(qq[0]) & (qq[0] > 0)
        ok = has_q & ((qq[0] >= min_q2) | on_gauge[glob])   # a gauged river is no desert wadi
        s = summarise(block[members][:, :, local[ok]], seconds, qq[:, ok])
        found: list[dict[str, Any]] = []
        for j, g in enumerate(glob[ok]):
            rp = int(s["rp"][j])
            if rp < 2:
                continue
            g = int(g)
            peak_day = int(s["peak_day"][j])
            found.append({
                "river_id": int(rivid[g]), "lat": float(tab["lat"][g]), "lon": float(tab["lon"][g]),
                "strahler_order": int(tab["order"][g]), "area_km2": _sig(tab["area_km2"][g], 4),
                "peak_cms": _sig(s["peak"][j]), "peak_date": (start + timedelta(days=peak_day)).isoformat(),
                "rp": rp, "share": round(float(s["share"][j]), 2),
                **{f"q{t}": _sig(qq[i, ok][j]) for i, t in enumerate(RETURN_PERIODS)},
                "daily": daily_string(s["daily"][:, j]), "gauges": [],
            })
        with lock:
            tally["read"] += 1
            tally["checked"] += int(ok.sum())
            tally["no_threshold"] += int((~has_q).sum())
            tally["below_floor"] += int((has_q & ~ok).sum())
            rows.extend(found)
            if tally["read"] % 200 == 0:
                logger.info("%d of %d chunks, %d reaches flagged, %.0f s", tally["read"], len(chosen), len(rows),
                            time.monotonic() - started)

    if workers <= 1:
        for ch in chosen:
            one(ch)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, chosen))

    for r in rows:
        r["gauges"] = sorted(by_reach.get(r["river_id"], []))
    rows.sort(key=lambda r: (-r["rp"], r["river_id"]))

    root = Path(out) / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    day = start.isoformat()
    _write_parquet(rows, root / "latest.parquet")
    _write_parquet(rows, root / f"{day}.parquet")
    fc, truncated = to_geojson(rows)
    (root / "latest.geojson").write_text(json.dumps(fc, separators=(",", ":"), allow_nan=False))
    counts = counts_by_class([r["rp"] for r in rows])
    smoke = max_chunks is not None
    entry = {"issue_date": day, "file": f"{FOLDER}/{day}.parquet", "n": len(rows), "counts": counts}
    history: list[dict[str, Any]] = []
    if not smoke:
        from aquascope.archive.forecasts import read_published_json

        published = read_published_json(f"{FOLDER}/manifest.json", repo_id)
        history = [h for h in published.get("history") or [] if h.get("issue_date") != day]
    doc = {
        "about": ABOUT, "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue_date": day, "run": name, "forecast": f"{base}/", "valid_from": day,
        "valid_to": (start + timedelta(days=DAYS - 1)).isoformat(), "days": DAYS,
        "n": len(rows), "counts": counts, "checked": tally["checked"], "no_threshold": tally["no_threshold"],
        "below_floor": tally["below_floor"], "min_strahler_order": min_order, "min_q2_cms": min_q2,
        "gauge_reaches": int(on_gauge.sum()),
        "members": int(len(members)),
        "chunks": {"needed": len(needed), "read": tally["read"], "failed": tally["failed"],
                   "skipped_time": tally["skipped_time"], "planned": len(chosen)},
        "bytes_read": getattr(fetch, "bytes", None), "seconds": round(time.monotonic() - started, 1),
        "smoke": smoke, "geojson_features": len(fc["features"]), "geojson_truncated": truncated,
        "method": METHOD.format(order=min_order, floor=min_q2),
        "thresholds": {"source": f"{RETURN_PERIODS_ZARR}/", "variable": THRESHOLD_VARIABLE,
                       "return_periods": list(RETURN_PERIODS),
                       "fit": "Gumbel (method of moments) on annual maxima of daily flow, GEOGLOWS v2 retrospective "
                              "simulation since 1940"},
        "classes": CLASSES, "daily_codes": list(DAILY_CODES), "not": NOT.format(order=min_order),
        "licence": LICENCE, "columns": COLUMNS,
        "files": {"latest": f"{FOLDER}/latest.parquet", "geojson": f"{FOLDER}/latest.geojson",
                  "dated": f"{FOLDER}/{day}.parquet"},
        "history": sorted(history + [entry], key=lambda h: h["issue_date"]),
    }
    (root / "manifest.json").write_text(json.dumps(doc, indent=1))
    keys = ("issue_date", "run", "n", "counts", "checked", "no_threshold", "below_floor", "chunks", "bytes_read",
            "seconds", "smoke", "geojson_features")
    return {k: doc[k] for k in keys}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/forecasts/warnings`` and nothing else to the Archive dataset (needs HF_TOKEN with write
    access). A smoke run is refused: its few chunks would replace the day's issue."""
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    manifest = src / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run the warnings step first")
    doc = json.loads(manifest.read_text())
    if doc.get("smoke"):
        raise ValueError("this is a smoke run (--max-chunks); it is never published")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        shutil.copytree(src, stage)
        return str(publish_folder(Path(tmp), repo_id, token=token, allow_patterns=[f"{FOLDER}/*"],
                                  commit_message=f"forecasts/warnings: issue {doc['issue_date']}"))


# ── reading what is published (the MCP tool and the CLI) ───────────────────


def _in_bbox(lat: float, lon: float, bbox: list[float] | tuple[float, ...]) -> bool:
    west, south, east, north = (float(x) for x in bbox)
    if not south <= lat <= north:
        return False
    return west <= lon <= east if west <= east else (lon >= west or lon <= east)  # across the antimeridian


def flood_warnings(bbox: list[float] | tuple[float, ...] | None = None, *, min_rp: int = 2, limit: int = 50,
                   repo_id: str = DEFAULT_REPO, local: str | Path | None = None) -> dict[str, Any]:
    """The published Floods ahead issue, optionally inside ``bbox`` (west, south, east, north in degrees): when it
    was issued, counts by return-period class, the reaches (highest class first, at most ``limit``), the method and
    what it is not. ``local`` reads a folder written by :func:`run` instead of the Archive."""
    import io

    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox is west, south, east, north")
    if local:
        root = Path(local) / FOLDER if (Path(local) / FOLDER).exists() else Path(local)
        mpath = root / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        data = (root / "latest.parquet").read_bytes() if (root / "latest.parquet").exists() else None
    else:
        from aquascope.archive.forecasts import _get, published_url, read_published_json

        manifest = read_published_json(f"{FOLDER}/manifest.json", repo_id)
        data = _get(published_url(f"{FOLDER}/latest.parquet", repo_id)) if manifest else None
    if not manifest or data is None:
        return {"available": False, "sentence": "No Floods ahead issue is published yet; the daily workflow "
                "(flood-warnings.yml) writes forecasts/warnings/ in the Archive dataset."}
    import pyarrow.parquet as pq

    rows = pq.read_table(io.BytesIO(data)).to_pylist()
    if bbox is not None:
        rows = [r for r in rows if _in_bbox(r["lat"], r["lon"], bbox)]
    rows = [r for r in rows if (r.get("rp") or 0) >= min_rp]
    counts = counts_by_class([r["rp"] for r in rows])
    rows.sort(key=lambda r: (-r["rp"], -(r["peak_cms"] or 0) / max(r["q2"] or 1e-9, 1e-9)))
    where = " in this box" if bbox is not None else ""
    return {
        "available": True, "issue_date": manifest.get("issue_date"), "valid_to": manifest.get("valid_to"),
        "run": manifest.get("run"), "made": manifest.get("made"), "bbox": list(bbox) if bbox is not None else None,
        "n": len(rows), "counts": counts, "sentence": sentence({**manifest, "counts": counts}, len(rows), where),
        "reaches": rows[:max(0, int(limit))], "truncated": len(rows) > limit,
        "min_strahler_order": manifest.get("min_strahler_order"), "method": manifest.get("method"),
        "not": manifest.get("not"), "licence": manifest.get("licence"), "smoke": manifest.get("smoke", False),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.warnings", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    p.add_argument("--min-order", type=int, default=DEFAULT_MIN_ORDER)
    p.add_argument("--min-q2", type=float, default=MIN_Q2_CMS, help="smallest 2-year flow classed, m3/s")
    p.add_argument("--max-chunks", type=int, default=None, help="forecast chunks read, for a smoke run")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time-budget-min", type=float, default=270.0)
    p.add_argument("--run", default=None, help="a forecast run (YYYYMMDD00) instead of the newest")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "run":
        info = run(a.out, repo_id=a.repo, min_order=a.min_order, min_q2=a.min_q2, max_chunks=a.max_chunks,
                   workers=a.workers,
                   time_budget_s=a.time_budget_min * 60, run_name=a.run)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
