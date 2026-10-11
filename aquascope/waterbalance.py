"""Subbasins by monthly water balance (#557): the HEC-HMS and QSWAT basin map, for the whole world, keyless.

Every HydroBASINS level-4 subbasin (1,342 of them, about 100,000 km2 on average) gets, for every month since
January 1990:

``p_mm``        precipitation, ERA5 total precipitation, area-weighted over the subbasin
``et_mm``       actual evapotranspiration, ERA5 total evaporation (sign flipped), area-weighted the same way
``q_mm``        the runoff the subbasin itself makes: the GEOGLOWS v2 monthly flow at its outlet reach(es) minus
                the flow coming in from the level-4 subbasins upstream, spread over its own area; withheld (with
                why) where that difference cannot be trusted (:func:`runoff_checks`)
``closure_mm``  P minus ET minus Q. Shown, never hidden: it holds the change in storage (soil, snow, groundwater,
                lakes and reservoirs) and the errors of three products that were not made to agree. Month by month
                it swings with the seasons; over whole years a steady basin's should be small.

and each value's anomaly against the subbasin's own 1991-2020 mean for that calendar month.

ET is ERA5's, not WaPOR's: WaPOR covers Africa and the Near East from 2018 only, and this map is the world since
1990. GEOGLOWS v2 is itself driven by ERA5 runoff, so Q is not independent of P and ET; it is routed, and it is
the same model the Explorer's rivers show.

Data, all read anonymously over HTTPS:

* ERA5 hourly single levels, ARCO-ERA5 on Google Cloud (``ar/full_37-1h-0p25deg-chunk-1.zarr-v3``): CC BY 4.0,
  Copernicus Climate Change Service / ECMWF (Hersbach et al. 2020).
* GEOGLOWS v2 retrospective monthly flow (``retrospective/monthly-timeseries.zarr``) and the HydroSOS
  pairing of level-4 subbasins with their outlet reaches (``hydrosos/hybas_linkno_pairs.parquet``), CC BY 4.0.
* The level-4 subbasin outlines (``hydrosos/hydrobasins_level_4.parquet``, the HydroBASINS v1 polygons, which
  BasinATLAS v1.0 publishes under CC BY 4.0, Linke et al. 2019) and the Archive's BasinATLAS level-12 routing
  (``basins/lev12_topology.parquet``), which gives each level-4 subbasin its area and its upstream neighbours.

Cost, measured on 2026-10-11 (``docs/explorer.md`` has the table): ERA5 is stored one hour of the whole globe per
chunk, on average 1.83 MB of precipitation and 3.01 MB of evaporation, so a month is about 1,460 chunks and 3.5 GB,
and 1990 to September 2026 is 441 months and about 1.56 TB. Reading is all there is to it: decoding and summing a
chunk over the subbasins takes about 2 ms. GitHub runners read 74 to 186 MB/s, so the backfill is 2.3 to 5.9 hours
of reading. GEOGLOWS's whole monthly record for the 943 outlet reaches is about 0.5 GB. A run reads the published
table first and computes only the months that are missing (or were preliminary ERA5T), newest first, until its
time budget runs out, then publishes what it has: one or two runs for the backfill, a few minutes a month after.

Layout under ``<out>/waterbalance/`` (the only folder this job writes or publishes):

    manifest.json       months covered, counts, method, what closure means, licences, the run's cost
    monthly.parquet     one row per subbasin and month: every variable, its anomaly, the outlet flows
    normals.parquet     the 1991-2020 mean per subbasin and calendar month
    basins.parquet      the subbasins: area, outlet reaches, downstream and upstream neighbours, flags
    basins.geojson      simplified outlines for the Explorer (feature ``k`` = the row in the year files)
    normals.json        the normals, for the Explorer's card
    years/<YYYY>.json   one year of the map, every variable and anomaly, for the Explorer's time bar

Run by ``.github/workflows/water-balance.yml``:

    python -m aquascope.waterbalance run --out build [--smoke-hours N] [--time-budget-min M]
    python -m aquascope.waterbalance publish --out build

Needs the ``archive`` and ``basins`` extras and a Blosc decoder (``numcodecs`` or ``blosc2``) to build; reading
what is published (:func:`water_balance`) needs only ``pyarrow``.
"""

from __future__ import annotations

import argparse
import calendar
import io
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FOLDER = "waterbalance"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
ERA5_ZARR = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_VARS = {"p": "total_precipitation", "e": "evaporation"}
GEOGLOWS_BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com"
MONTHLY_Q_ZARR = f"{GEOGLOWS_BUCKET}/retrospective/monthly-timeseries.zarr"
BASINS_URL = f"{GEOGLOWS_BUCKET}/hydrosos/hydrobasins_level_4.parquet"
PAIRS_URL = f"{GEOGLOWS_BUCKET}/hydrosos/hybas_linkno_pairs.parquet"
TOPOLOGY_PATH = "basins/lev12_topology.parquet"
START = "1990-01"
NORMAL_YEARS = (1991, 2020)
MIN_NORMAL_YEARS = 20
VARIABLES = ("p", "et", "q", "closure")
LABELS = {"p": "rain", "et": "evaporation", "q": "runoff", "closure": "closure error"}
UNITS = "mm per month"
GEOJSON_TOLERANCE = 0.05     # degrees: a level-4 subbasin drawn on the globe, not a survey
GEOJSON_MIN_PART = 0.02      # square degrees: islands and holes smaller than this are left out of the outline
# Runoff that cannot be trusted is withheld per subbasin (see :func:`runoff_checks`), never painted as a number:
OUTLET_SHARE = 0.5           # the outlet carries less than half the inflow, over the record: not the main river
THROUGH_MM = 1000.0          # the inflow is more than a metre of water a month over the subbasin, on average
RATIO_MONTHS = 12            # months needed before "more runoff than rain over the record" is judged
RUNOFF_CHECKS = {
    "outlet": "the GEOGLOWS reach paired with its outlet carries less than half of what flows in, so it is not "
              "the river leaving it (a delta or a coast with several outlets, or another river)",
    "through": "the river brings in more than a metre of water a month over it, so its own runoff is lost in the "
               "difference of two large flows (5 % of the inflow is already 50 mm)",
    "ratio": "over the record its outlet carries more water than falls on it as rain, so the paired reach drains "
             "more than this subbasin",
    "upstream": "its inflow comes from a subbasin whose outlet reach fails one of these checks",
}

METHOD = (
    "For every HydroBASINS level-4 subbasin and month: P and ET are ERA5's hourly total precipitation and total "
    "evaporation (sign flipped) summed over the month and averaged over the 0.25-degree cells whose centres fall "
    "in the subbasin, weighted by cell area (the nearest cell for a subbasin smaller than one). Q is the GEOGLOWS "
    "v2 retrospective monthly mean flow at the subbasin's outlet reach(es), minus the outlet flow of the level-4 "
    "subbasins draining into it, times the seconds in the month, over the subbasin's area (BasinATLAS). The "
    "closure error is P - ET - Q. Anomalies are against the subbasin's 1991-2020 mean for the calendar month."
)
CLOSURE = (
    "P - ET - Q is shown as it comes out. It holds the change in storage (soil moisture, snow, groundwater, lakes "
    "and reservoirs), which swings with the seasons, and the errors of three products made separately. Over whole "
    "years it should be small where nothing is stored or lost; where it is not, one of the three is off."
)
NOT = (
    "Modelled, not measured: a reanalysis and a routed model, not gauges. Q is missing where GEOGLOWS has no outlet "
    "reach for the subbasin, or where an upstream subbasin has none (its inflow cannot be taken off). A negative Q "
    "means the river leaves the subbasin smaller than it came in (evaporation from the channel, irrigation, "
    "losses to the ground), or that the model's routing moves water across the month boundary. Q is also withheld "
    "(the flows are kept in the table) where it cannot be trusted: where the paired outlet reach carries less than "
    "half the inflow, where the inflow is more than a metre of water a month over the subbasin, where the outlet "
    "carries more than the rain over the record, or downstream of a subbasin whose outlet fails those checks."
)
LICENCE = {
    "era5": "ERA5 hourly single levels (ARCO-ERA5 copy on Google Cloud): CC BY 4.0, Copernicus Climate Change "
            "Service / ECMWF. Hersbach et al. (2020), https://doi.org/10.1002/qj.3803. Contains modified "
            "Copernicus Climate Change Service information.",
    "geoglows": "GEOGLOWS v2 retrospective simulation and HydroSOS tables: CC BY 4.0.",
    "basins": "HydroBASINS level-4 subbasins as published in BasinATLAS v1.0, CC BY 4.0. Linke et al. (2019), "
              "https://doi.org/10.1038/s41597-019-0300-6.",
    "this_file": "CC BY 4.0, with the attributions above.",
}
SOURCES = {
    "era5": ERA5_ZARR, "geoglows_monthly": MONTHLY_Q_ZARR, "basins": BASINS_URL, "pairs": PAIRS_URL,
    "topology": f"Archive {TOPOLOGY_PATH}",
}


# ── months ──────────────────────────────────────────────────────────────────


def month_range(start: str, end: str) -> list[str]:
    """Every ``YYYY-MM`` from ``start`` to ``end``, both included."""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_hours(month: str) -> int:
    y, m = (int(x) for x in month.split("-"))
    return calendar.monthrange(y, m)[1] * 24


def month_seconds(month: str) -> float:
    return month_hours(month) * 3600.0


def next_month(month: str) -> str:
    y, m = (int(x) for x in month.split("-"))
    return f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"


def last_complete_month(stop: str) -> str:
    """The newest month whose every hour is at or before ``stop`` (a date): ERA5's last hour of a month is the
    first hour of the next (hourly accumulations end at their time stamp), so the month before ``stop``'s."""
    d = date.fromisoformat(str(stop)[:10])
    y, m = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    return f"{y:04d}-{m:02d}"


# ── the subbasins ───────────────────────────────────────────────────────────


@dataclass
class Basins:
    """The level-4 subbasins and how the ERA5 grid and GEOGLOWS reaches map onto them.

    ``cells``, ``cell_basin`` and ``cell_weight`` are parallel: a flat ERA5 cell index (row * 1440 + column), the
    subbasin it belongs to, and its weight (cos latitude). ``outlets`` are each subbasin's GEOGLOWS reaches,
    ``down`` the index of the subbasin it drains into (-1 for the sea or a sink), ``up`` the ones draining into it.
    """

    ids: list[int]
    area_km2: list[float]
    lat: list[float]
    lon: list[float]
    outlets: list[list[int]]
    down: list[int]
    cells: Any
    cell_basin: Any
    cell_weight: Any
    geometry: list[Any] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def up(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in self.ids]
        for i, d in enumerate(self.down):
            if d >= 0:
                out[d].append(i)
        return out

    @property
    def linked(self) -> list[bool]:
        return [bool(o) for o in self.outlets]

    @property
    def runoff_ok(self) -> list[bool]:
        """Q can be worked out: the subbasin has an outlet reach and so does every subbasin draining into it."""
        linked = self.linked
        return [linked[i] and all(linked[u] for u in ups) for i, ups in enumerate(self.up)]


def _read_parquet_bytes(data: bytes | None, what: str) -> Any:
    import pandas as pd

    if not data:
        raise RuntimeError(f"{what} could not be read")
    return pd.read_parquet(io.BytesIO(data))


def era5_grid() -> tuple[Any, Any]:
    """ERA5's 0.25-degree grid: latitudes 90 to -90 (721), longitudes 0 to 359.75 (1440)."""
    import numpy as np

    return np.arange(721) * -0.25 + 90.0, np.arange(1440) * 0.25


def build_basins(fetch: Callable[[str], bytes | None], *, repo_id: str = DEFAULT_REPO,
                 topology: Any = None) -> Basins:
    """Read the subbasins, their outlet reaches and the level-12 routing, and map the ERA5 grid onto them.

    The level-12 sub-basins nest in the level-4 ones, so each level-12 centre is placed in its level-4 subbasin:
    a level-4 subbasin's area is the sum of its level-12 areas, and it drains into the level-4 subbasin its
    outlet's level-12 ``next_down`` lies in.
    """
    import numpy as np
    import shapely

    poly = _read_parquet_bytes(fetch(BASINS_URL), BASINS_URL)
    pairs = _read_parquet_bytes(fetch(PAIRS_URL), PAIRS_URL)
    if topology is None:
        from aquascope.archive.forecasts import published_url

        topology = _read_parquet_bytes(fetch(published_url(TOPOLOGY_PATH, repo_id)), TOPOLOGY_PATH)
    ids = [int(x) for x in poly["HYBAS_ID"]]
    geoms = shapely.make_valid(shapely.from_wkb(poly["geometry"].values))
    tree = shapely.STRtree(geoms)

    # level 12 into level 4: area and routing
    pts = shapely.points(topology["lon"].to_numpy(float), topology["lat"].to_numpy(float))
    pi, bi = tree.query(pts, predicate="within")
    l4 = np.full(len(topology), -1, dtype=np.int64)
    l4[pi] = bi
    area = np.bincount(l4[l4 >= 0], weights=topology["sub_area"].to_numpy(float)[l4 >= 0], minlength=len(ids))
    where = dict(zip(topology["hybas_id"].to_numpy(np.int64).tolist(), l4.tolist()))
    nd = np.array([where.get(int(x), -1) for x in topology["next_down"].to_numpy(np.int64)], dtype=np.int64)
    down = [-1] * len(ids)
    votes: dict[int, dict[int, float]] = {}
    for a, b, w in zip(l4.tolist(), nd.tolist(), topology["up_area"].to_numpy(float).tolist()):
        if a >= 0 and b >= 0 and a != b:
            votes.setdefault(a, {})
            votes[a][b] = max(votes[a].get(b, 0.0), w)
    for a, v in votes.items():
        down[a] = max(v, key=v.get)    # where the most area leaves (the outlet), in the rare split case

    # outlets
    outlets: list[list[int]] = [[] for _ in ids]
    pos = {h: i for i, h in enumerate(ids)}
    for h, link in zip(pairs["HYBAS_ID"].tolist(), pairs["LINKNO"].tolist()):
        if int(h) in pos:
            outlets[pos[int(h)]].append(int(link))

    # the ERA5 grid
    lat, lon = era5_grid()
    lon180 = np.where(lon >= 180, lon - 360, lon)
    lo, la = np.meshgrid(lon180, lat)
    ci, cb = tree.query(shapely.points(lo.ravel(), la.ravel()), predicate="within")
    order = np.argsort(ci, kind="stable")
    ci, cb = ci[order], cb[order]
    keep = np.concatenate([[True], ci[1:] != ci[:-1]]) if len(ci) else np.zeros(0, bool)  # a cell counts once
    ci, cb = ci[keep], cb[keep]
    rep = shapely.point_on_surface(geoms)
    rlat, rlon = shapely.get_y(rep), shapely.get_x(rep)
    missing = sorted(set(range(len(ids))) - set(cb.tolist()))
    extra_c, extra_b = [], []
    for b in missing:                  # smaller than a cell: the nearest cell to a point inside it
        r = int(round((90.0 - rlat[b]) / 0.25))
        c = int(round((rlon[b] % 360) / 0.25)) % 1440
        extra_c.append(r * 1440 + c)
        extra_b.append(b)
    cells = np.concatenate([ci, np.array(extra_c, dtype=ci.dtype)]).astype(np.int64)
    cell_basin = np.concatenate([cb, np.array(extra_b, dtype=cb.dtype)]).astype(np.int64)
    cell_weight = np.cos(np.deg2rad(lat))[cells // 1440]
    logger.info("subbasins: %d, %d with an outlet reach, %d drain into another, %d smaller than a cell",
                len(ids), sum(1 for o in outlets if o), sum(1 for d in down if d >= 0), len(missing))
    return Basins(ids=ids, area_km2=[float(a) for a in area], lat=[float(x) for x in rlat],
                  lon=[float(x) for x in rlon], outlets=outlets, down=down, cells=cells, cell_basin=cell_basin,
                  cell_weight=cell_weight, geometry=list(geoms))


# ── ERA5 ────────────────────────────────────────────────────────────────────


def era5_layout(fetch: Callable[[str], bytes | None]) -> dict[str, Any]:
    """The store's metadata, its hour index and the last day of final ERA5 and of preliminary ERA5T."""
    import numpy as np

    from aquascope.archive.warnings import _meta, _read_1d

    meta = _meta(fetch, ERA5_ZARR)
    attrs = meta.get(".zattrs", {})
    for v in ERA5_VARS.values():
        arr = meta.get(f"{v}/.zarray") or {}
        if list(arr.get("chunks", []))[1:] != [721, 1440]:
            raise RuntimeError(f"ERA5 {v} is not chunked one whole-globe hour at a time any more: {arr.get('chunks')}")
    units = meta.get("time/.zattrs", {}).get("units", "")
    if not units.startswith("hours since 1900-01-01"):
        raise RuntimeError(f"unexpected ERA5 time units {units!r}")
    hours = _read_1d(fetch, ERA5_ZARR, "time", meta).astype(np.int64)
    return {"meta": meta, "hours": hours, "final_until": attrs.get("valid_time_stop"),
            "era5t_until": attrs.get("valid_time_stop_era5t") or attrs.get("valid_time_stop")}


def hour_index(hours: Any, when: datetime) -> int:
    """The position of ``when`` in the store's hour index."""
    import numpy as np

    h = int((when - datetime(1900, 1, 1)).total_seconds() // 3600)
    i = int(np.searchsorted(hours, h))
    if i >= len(hours) or int(hours[i]) != h:
        raise RuntimeError(f"ERA5 has no hour {when.isoformat()}")
    return i


def month_hour_indexes(hours: Any, month: str, max_hours: int | None = None) -> list[int]:
    """The hours whose accumulations make up ``month``: 01:00 on the 1st to 00:00 on the 1st of the next month."""
    y, m = (int(x) for x in month.split("-"))
    first = hour_index(hours, datetime(y, m, 1)) + 1
    n = month_hours(month)
    if max_hours:
        n = min(n, int(max_hours))
    return list(range(first, first + n))


def era5_month(fetch: Callable[[str], bytes | None], layout: dict[str, Any], basins: Basins, month: str, *,
               workers: int = 32, max_hours: int | None = None) -> dict[str, Any]:
    """ERA5 P and ET for every subbasin in ``month`` (mm), with how many hours were read and missing.

    Each hourly chunk is decoded and summed over the subbasins in its worker, so memory stays at a few chunks.
    A subbasin whose month has a missing or NaN hour gets no value: an incomplete month is not a month.
    """
    import numpy as np

    from aquascope.archive.warnings import decode_chunk

    meta = layout["meta"]
    idx = month_hour_indexes(layout["hours"], month, max_hours)
    n = basins.n
    cells, cb, w = basins.cells, basins.cell_basin, basins.cell_weight
    wsum = np.bincount(cb, weights=w, minlength=n)
    lock = threading.Lock()
    acc = {k: np.zeros(n) for k in ERA5_VARS}
    bad = {k: np.zeros(n, dtype=bool) for k in ERA5_VARS}
    missing = {k: 0 for k in ERA5_VARS}

    def one(job: tuple[str, int]) -> None:
        key, i = job
        var = ERA5_VARS[key]
        arr = meta[f"{var}/.zarray"]
        data = fetch(f"{ERA5_ZARR}/{var}/{i}.0.0")
        if data is None:
            with lock:
                missing[key] += 1
                bad[key][:] = True
            return
        vals = decode_chunk(data, arr).reshape(-1)[cells].astype(np.float64)
        nan = np.isnan(vals)
        if nan.any():
            hit = np.bincount(cb[nan], minlength=n) > 0
            vals = np.where(nan, 0.0, vals)
        else:
            hit = None
        s = np.bincount(cb, weights=vals * w, minlength=n)
        with lock:
            acc[key] += s
            if hit is not None:
                bad[key] |= hit

    jobs = [(k, i) for i in idx for k in ERA5_VARS]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, jobs))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = acc["p"] / wsum * 1000.0
        et = -acc["e"] / wsum * 1000.0
    p[bad["p"] | (wsum == 0)] = np.nan
    et[bad["e"] | (wsum == 0)] = np.nan
    return {"month": month, "p": p, "et": et, "hours": len(idx), "missing": missing}


# ── GEOGLOWS ────────────────────────────────────────────────────────────────


def geoglows_flows(fetch: Callable[[str], bytes | None], reaches: list[int], months: list[str]) -> dict[str, Any]:
    """GEOGLOWS v2 monthly mean flow (m3/s) for ``reaches`` in ``months``: {month: {reach: flow}}.

    The store is chunked 250 reaches by 1,020 months, so the whole record of one reach is one or two chunks; the
    943 outlet reaches sit in about 570 column chunks (about 0.5 GB for every month since 1940).
    """
    import numpy as np

    from aquascope.archive.warnings import _meta, _read_1d, decode_chunk

    meta = _meta(fetch, MONTHLY_Q_ZARR)
    qarr = meta["Q/.zarray"]
    if meta.get("Q/.zattrs", {}).get("_ARRAY_DIMENSIONS") != ["time", "river_id"]:
        raise RuntimeError("GEOGLOWS monthly Q is not (time, river_id) any more")
    rid = _read_1d(fetch, MONTHLY_Q_ZARR, "river_id", meta).astype(np.int64)
    days = _read_1d(fetch, MONTHLY_Q_ZARR, "time", meta).astype(np.int64)
    base = date(1940, 1, 1)
    labels = [(date.fromordinal(base.toordinal() + int(d))).strftime("%Y-%m") for d in days]
    tpos = {m: i for i, m in enumerate(labels)}
    want_t = sorted({tpos[m] for m in months if m in tpos})
    pos = {int(r): i for i, r in enumerate(rid.tolist())}
    cols = {r: pos[r] for r in reaches if r in pos}
    tc, rc = qarr["chunks"]
    out: dict[str, dict[int, float]] = {m: {} for m in months}
    by_chunk: dict[int, list[int]] = {}
    for r, p in cols.items():
        by_chunk.setdefault(p // rc, []).append(r)
    jobs = [(t, c) for c in by_chunk for t in sorted({ti // tc for ti in want_t})]
    lock = threading.Lock()

    def one(job: tuple[int, int]) -> None:
        t, c = job
        data = fetch(f"{MONTHLY_Q_ZARR}/Q/{t}.{c}")
        if data is None:
            return
        block = decode_chunk(data, qarr)
        for r in by_chunk[c]:
            col = cols[r] - c * rc
            for ti in want_t:
                if ti // tc != t:
                    continue
                v = float(block[ti - t * tc, col])
                if math.isfinite(v):
                    with lock:
                        out[labels[ti]][r] = v

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, jobs))
    return {"flows": out, "last_month": labels[-1] if labels else None, "found": len(cols), "asked": len(reaches),
            "chunks": len(jobs)}


# ── the balance ─────────────────────────────────────────────────────────────


def outlet_flow(outlets: list[list[int]], flows: dict[int, float]) -> list[float | None]:
    """Each subbasin's outlet flow: the sum over its outlet reaches, or None when one is missing or none exist."""
    out: list[float | None] = []
    for reaches in outlets:
        if not reaches or any(r not in flows for r in reaches):
            out.append(None)
        else:
            out.append(float(sum(flows[r] for r in reaches)))
    return out


def runoff_checks(rows: list[dict[str, Any]], basins: Basins) -> dict[int, str]:
    """The subbasins whose runoff is withheld, with why (a key of :data:`RUNOFF_CHECKS`).

    Q is a difference of two routed flows spread over the subbasin, and the HydroSOS pairing gives one outlet
    reach per subbasin. Judged on the whole table's means (months with a runoff value), a subbasin fails when its
    outlet carries less than ``OUTLET_SHARE`` of its inflow (``outlet``), when the inflow is more than
    ``THROUGH_MM`` a month over it (``through``), or when, over at least ``RATIO_MONTHS`` months, it runs off more
    than it receives as rain (``ratio``). A subbasin downstream of an ``outlet`` or ``ratio`` failure takes that
    outlet's flow as its inflow, so it fails too (``upstream``).
    """
    pos = {h: i for i, h in enumerate(basins.ids)}
    sums: dict[int, list[float]] = {}   # months, outlet and inflow m3/s, inflow mm; runoff, rain (mm), their months
    for r in rows:
        i = pos.get(int(r["hybas_id"]))
        if i is None or r.get("q_mm") is None or r.get("q_out_cms") is None or r.get("q_in_cms") is None:
            continue
        s = sums.setdefault(int(r["hybas_id"]), [0.0] * 7)
        s[0] += 1
        s[1] += float(r["q_out_cms"])
        s[2] += float(r["q_in_cms"])
        area = basins.area_km2[i]
        s[3] += float(r["q_in_cms"]) * month_seconds(r["month"]) / (area * 1e6) * 1000.0 if area > 0 else 0.0
        if r.get("p_mm") is not None:     # runoff against rain over the same months
            s[4] += float(r["q_mm"])
            s[5] += float(r["p_mm"])
            s[6] += 1
    out: dict[int, str] = {}
    for h, (n, qout, qin, in_mm, q, p, n_p) in sums.items():
        if qin > 0 and qout < OUTLET_SHARE * qin:
            out[h] = "outlet"
        elif in_mm / n > THROUGH_MM:
            out[h] = "through"
        elif n_p >= RATIO_MONTHS and q > p:
            out[h] = "ratio"
    bad = {h for h, why in out.items() if why in ("outlet", "ratio")}
    for i, ups in enumerate(basins.up):
        h = basins.ids[i]
        if h in sums and h not in out and any(basins.ids[u] in bad for u in ups):
            out[h] = "upstream"
    return out


def withhold_runoff(rows: list[dict[str, Any]], checks: dict[int, str]) -> None:
    """Blank ``q_mm`` and ``closure_mm`` of the subbasins in ``checks``; the flows stay, so anyone can redo it."""
    for r in rows:
        if int(r["hybas_id"]) in checks:
            r["q_mm"] = None
            r["closure_mm"] = None


def balance_month(month: str, p: list[float | None], et: list[float | None], q_out: list[float | None],
                  basins: Basins) -> list[dict[str, Any]]:
    """One month's rows: P, ET, the subbasin's own runoff (outlet minus inflow, in mm) and the closure error."""
    secs = month_seconds(month)
    up = basins.up
    rows = []
    for i, hid in enumerate(basins.ids):
        qo = q_out[i]
        qin: float | None = 0.0
        for u in up[i]:
            if q_out[u] is None:
                qin = None
                break
            qin = (qin or 0.0) + float(q_out[u])
        q_mm = None
        area = basins.area_km2[i]
        if qo is not None and qin is not None and area > 0:
            q_mm = (qo - qin) * secs / (area * 1e6) * 1000.0
        pi, ei = _num(p[i]), _num(et[i])
        closure = pi - ei - q_mm if pi is not None and ei is not None and q_mm is not None else None
        rows.append({"hybas_id": hid, "month": month, "p_mm": pi, "et_mm": ei, "q_mm": q_mm, "closure_mm": closure,
                     "q_out_cms": qo, "q_in_cms": qin if up[i] else (0.0 if qo is not None else None)})
    return rows


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normals(rows: list[dict[str, Any]], years: tuple[int, int] = NORMAL_YEARS,
            min_years: int = MIN_NORMAL_YEARS) -> dict[tuple[int, int], dict[str, float | None]]:
    """The mean of each variable per (subbasin, calendar month) over ``years``, where at least ``min_years`` of
    them have a value; None otherwise."""
    sums: dict[tuple[int, int], dict[str, list[float]]] = {}
    for r in rows:
        y, m = (int(x) for x in r["month"].split("-"))
        if not years[0] <= y <= years[1]:
            continue
        slot = sums.setdefault((r["hybas_id"], m), {v: [] for v in VARIABLES})
        for v in VARIABLES:
            x = r.get(f"{v}_mm")
            if x is not None:
                slot[v].append(float(x))
    out: dict[tuple[int, int], dict[str, float | None]] = {}
    for key, slot in sums.items():
        out[key] = {v: (sum(xs) / len(xs) if len(xs) >= min_years else None) for v, xs in slot.items()}
    return out


def add_anomalies(rows: list[dict[str, Any]], norms: dict[tuple[int, int], dict[str, float | None]]) -> None:
    """``<v>_anom``: the value minus the subbasin's normal for the calendar month (None without a normal)."""
    for r in rows:
        n = norms.get((r["hybas_id"], int(r["month"][5:7]))) or {}
        for v in VARIABLES:
            x, nv = r.get(f"{v}_mm"), n.get(v)
            r[f"{v}_anom"] = (x - nv) if x is not None and nv is not None else None


def basin_sentence(row: dict[str, Any], *, area_km2: float | None = None) -> str:
    """A month of one subbasin in words, every number from the row (never typed by a model)."""
    label = month_label(row["month"])
    parts = []
    if row.get("p_mm") is not None:
        parts.append(f"{row['p_mm']:.0f} mm of rain")
    if row.get("et_mm") is not None:
        parts.append(f"{row['et_mm']:.0f} mm evaporated")
    if row.get("q_mm") is not None:
        q = row["q_mm"]
        parts.append(f"{q:.0f} mm ran off" if q >= 0 else f"the river lost {abs(q):.0f} mm crossing it")
    if not parts:
        return f"{label}: no water balance for this subbasin."
    text = f"{label}: " + ", ".join(parts)
    c = row.get("closure_mm")
    if c is not None:
        text += f"; P - ET - Q leaves {c:+.0f} mm"
    elif row.get("q_mm") is None:
        text += "; no runoff worked out here, so no closure"
    pa = row.get("p_anom")
    if pa is not None:
        text += f". Rain was {abs(pa):.0f} mm {'above' if pa >= 0 else 'below'} the 1991-2020 normal for the month"
    return text + "."


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def month_label(month: str) -> str:
    return f"{MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"


# ── writing ─────────────────────────────────────────────────────────────────

COLUMNS = {
    "hybas_id": "HydroBASINS level-4 id",
    "month": "YYYY-MM",
    "p_mm": "precipitation, mm in the month (ERA5)",
    "et_mm": "actual evapotranspiration, mm in the month (ERA5 total evaporation, sign flipped)",
    "q_mm": "the subbasin's own runoff, mm in the month (GEOGLOWS outlet minus inflow, over its area)",
    "closure_mm": "P - ET - Q, mm: storage change and the products' errors together",
    "q_out_cms": "GEOGLOWS monthly mean flow at the outlet reach(es), m3/s",
    "q_in_cms": "GEOGLOWS monthly mean flow coming in from the level-4 subbasins upstream, m3/s",
    "era5t": "true while ERA5 for the month is the preliminary ERA5T (it is recomputed when final)",
    "p_anom/et_anom/q_anom/closure_anom": "the value minus the 1991-2020 mean for the calendar month, mm",
}


def _r1(x: Any) -> float | None:
    v = _num(x)
    return None if v is None else round(v, 1)


def _write_table(rows: list[dict[str, Any]], path: Path, schema: list[tuple[str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {"int64": pa.int64(), "float32": pa.float32(), "float64": pa.float64(), "string": pa.string(),
             "bool": pa.bool_(), "list<int64>": pa.list_(pa.int64())}
    sch = pa.schema([(n, types[t]) for n, t in schema])
    table = pa.Table.from_pylist([{n: r.get(n) for n, _ in schema} for r in rows], schema=sch)
    table = table.replace_schema_metadata({"licence": json.dumps(LICENCE), "method": METHOD})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=50_000)


MONTHLY_SCHEMA = [("hybas_id", "int64"), ("month", "string"), ("p_mm", "float32"), ("et_mm", "float32"),
                  ("q_mm", "float32"), ("closure_mm", "float32"), ("q_out_cms", "float32"), ("q_in_cms", "float32"),
                  ("era5t", "bool"), ("p_anom", "float32"), ("et_anom", "float32"), ("q_anom", "float32"),
                  ("closure_anom", "float32")]


def basins_rows(basins: Basins, checks: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """The subbasins table. ``runoff_ok``: a runoff value is published (an outlet reach here and upstream, and the
    checks passed); ``runoff_check``: why it is withheld when the reaches exist (a key of :data:`RUNOFF_CHECKS`)."""
    import numpy as np

    up, ok, checks = basins.up, basins.runoff_ok, checks or {}

    ncell = np.bincount(basins.cell_basin, minlength=basins.n)
    return [{"hybas_id": h, "k": i, "area_km2": round(basins.area_km2[i], 1), "lat": round(basins.lat[i], 4),
             "lon": round(basins.lon[i], 4), "outlets": basins.outlets[i],
             "down_id": basins.ids[basins.down[i]] if basins.down[i] >= 0 else None,
             "up_ids": [basins.ids[u] for u in up[i]], "linked": bool(basins.outlets[i]),
             "runoff_ok": ok[i] and h not in checks, "runoff_check": checks.get(h) if ok[i] else None,
             "era5_cells": int(ncell[i])} for i, h in enumerate(basins.ids)]


def outline(geom: Any) -> Any:
    """A subbasin's outline for the globe: simplified, rounded to 0.01 degree, without its smallest islands and
    holes (the biggest part is always kept)."""
    import numpy as np
    import shapely

    polys = [p for p in shapely.get_parts(geom) if p.geom_type == "Polygon" and not p.is_empty]
    if not polys:
        return None
    big = [p for p in polys if p.area >= GEOJSON_MIN_PART] or [max(polys, key=lambda p: p.area)]
    out = []
    for p in big:
        holes = [h for h in p.interiors if shapely.Polygon(h).area >= GEOJSON_MIN_PART]
        q = shapely.simplify(shapely.Polygon(p.exterior, holes), GEOJSON_TOLERANCE, preserve_topology=True)
        q = shapely.transform(q, lambda c: np.round(c, 2))
        if not q.is_empty:
            out.append(q)
    if not out:
        return None
    return out[0] if len(out) == 1 else shapely.MultiPolygon(out)


def basins_geojson(basins: Basins, checks: dict[int, str] | None = None) -> dict[str, Any]:
    """The outlines: ``k`` the row in the year files, ``a`` km2, ``r`` 1 where runoff is published, and ``c`` why
    it is withheld where the reaches exist but a check failed."""
    import shapely

    feats = []
    ok, checks = basins.runoff_ok, checks or {}
    for i, g in enumerate(basins.geometry):
        o = outline(g) if g is not None else None
        if o is None:
            continue
        props = {"k": i, "id": str(basins.ids[i]), "a": round(basins.area_km2[i]),
                 "r": int(ok[i] and basins.ids[i] not in checks)}
        if ok[i] and basins.ids[i] in checks:
            props["c"] = checks[basins.ids[i]]
        feats.append({"type": "Feature", "properties": props, "geometry": json.loads(shapely.to_geojson(o))})
    return {"type": "FeatureCollection", "features": feats}


def year_files(rows: list[dict[str, Any]], basins: Basins) -> dict[str, dict[str, Any]]:
    """The Explorer's per-year files: for each month, every variable and anomaly in subbasin order (``k``)."""
    k = {h: i for i, h in enumerate(basins.ids)}
    by_month: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_month.setdefault(r["month"], []).append(r)
    out: dict[str, dict[str, Any]] = {}
    keys = [f"{v}_mm" for v in VARIABLES] + [f"{v}_anom" for v in VARIABLES]
    for month in sorted(by_month):
        doc = out.setdefault(month[:4], {"year": int(month[:4]), "months": [], "era5t": [], "n": basins.n,
                                         **{key: [] for key in keys}})
        grid = {key: [None] * basins.n for key in keys}
        t = False
        for r in by_month[month]:
            i = k.get(r["hybas_id"])
            if i is None:
                continue
            for key in keys:
                grid[key][i] = _r1(r.get(key))
            t = t or bool(r.get("era5t"))
        doc["months"].append(month)
        doc["era5t"].append(t)
        for key in keys:
            doc[key].append(grid[key])
    return out


def normals_doc(norms: dict[tuple[int, int], dict[str, float | None]], basins: Basins) -> dict[str, Any]:
    doc: dict[str, Any] = {"period": f"{NORMAL_YEARS[0]}-{NORMAL_YEARS[1]}", "min_years": MIN_NORMAL_YEARS,
                           "n": basins.n}
    for v in VARIABLES:
        doc[v] = [[_r1((norms.get((h, m)) or {}).get(v)) for h in basins.ids] for m in range(1, 13)]
    return doc


# ── the run ─────────────────────────────────────────────────────────────────


def _previous(local: str | Path | None, repo_id: str, fetch: Callable[[str], bytes | None]) -> list[dict[str, Any]]:
    """The table published last time (or one in a local folder), to compute only what is missing."""
    try:
        if local:
            p = Path(local) / FOLDER / "monthly.parquet"
            if not p.exists():
                p = Path(local) / "monthly.parquet"
            data = p.read_bytes() if p.exists() else None
        else:
            from aquascope.archive.forecasts import published_url

            data = fetch(published_url(f"{FOLDER}/monthly.parquet", repo_id))
        if not data:
            return []
        import pyarrow.parquet as pq

        return list(pq.read_table(io.BytesIO(data)).to_pylist())
    except (OSError, ValueError) as exc:
        logger.info("no previous water balance: %s", exc)
        return []


def run(out: str | Path, *, repo_id: str = DEFAULT_REPO, start: str = START, end: str | None = None,
        workers: int = 32, time_budget_s: float = 270 * 60, smoke_hours: int | None = None,
        previous: str | Path | None = None, fetch: Callable[[str], bytes | None] | None = None,
        basins: Basins | None = None, refresh_months: int = 3) -> dict[str, Any]:
    """Compute the missing months (newest first, until the time budget) and write ``<out>/waterbalance``.

    ``smoke_hours`` reads only that many hours of the newest month, marks the result a smoke run (it is not a
    month's total) and is never published. ``previous`` reads last time's table from a folder instead of the
    Archive. ``refresh_months``: the newest months are always recomputed (ERA5T becomes final, GEOGLOWS fills in).
    """
    from aquascope.archive.warnings import Fetcher

    t0 = time.monotonic()
    get: Callable[[str], bytes | None] = fetch or Fetcher()
    if basins is None:
        basins = build_basins(get, repo_id=repo_id)
    layout = era5_layout(get)
    last = last_complete_month(layout["era5t_until"])
    final_last = last_complete_month(layout["final_until"]) if layout.get("final_until") else last
    if end and end < last:
        last = end
    wanted = month_range(start, last)
    smoke = bool(smoke_hours)
    prev = [] if smoke else _previous(previous, repo_id, get)
    have: dict[str, dict[int, dict[str, Any]]] = {}
    for r in prev:
        have.setdefault(r["month"], {})[int(r["hybas_id"])] = r
    stale = {m for m, rows in have.items() if any(r.get("era5t") for r in rows.values())}
    fresh = set(wanted[-refresh_months:]) if refresh_months else set()
    todo = [m for m in reversed(wanted) if m not in have or m in stale or m in fresh]
    if smoke:
        todo = todo[:1]
    era5: dict[str, dict[str, Any]] = {}
    skipped = []
    missing_hours = 0

    def counted() -> int | None:  # the Fetcher counts what it read; an injected fetch may not
        return getattr(get, "bytes", None)

    b0, te = counted(), time.monotonic()
    for month in todo:
        if time.monotonic() - t0 > time_budget_s:
            skipped.append(month)
            continue
        t1 = time.monotonic()
        res = era5_month(get, layout, basins, month, workers=workers, max_hours=smoke_hours)
        missing_hours += sum(res["missing"].values())
        era5[month] = res
        logger.info("%s: %d hours in %.0f s (%.1f GB read so far)", month, res["hours"], time.monotonic() - t1,
                    (counted() or 0) / 1e9)
    era5_seconds = time.monotonic() - te
    b1 = counted()
    era5_bytes = b1 - b0 if b0 is not None and b1 is not None else None
    # the table: this run's months and the ones kept from last time
    months = sorted(set(era5) | {m for m in have if m in set(wanted)})
    reaches = sorted({r for o in basins.outlets for r in o})
    t2 = time.monotonic()
    q = geoglows_flows(get, reaches, months)
    q_seconds = time.monotonic() - t2
    rows: list[dict[str, Any]] = []
    for month in months:
        if month in era5:
            p, et = list(era5[month]["p"]), list(era5[month]["et"])
            era5t = month > final_last
        else:
            old = have[month]
            p = [old.get(h, {}).get("p_mm") for h in basins.ids]
            et = [old.get(h, {}).get("et_mm") for h in basins.ids]
            era5t = any(r.get("era5t") for r in old.values())
        q_out = outlet_flow(basins.outlets, q["flows"].get(month, {}))
        for r in balance_month(month, p, et, q_out, basins):
            r["era5t"] = era5t
            rows.append(r)
    checks = runoff_checks(rows, basins)
    withhold_runoff(rows, checks)
    norms = normals(rows)
    add_anomalies(rows, norms)
    root = Path(out) / FOLDER
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _write_table(rows, root / "monthly.parquet", MONTHLY_SCHEMA)
    nrows = [{"hybas_id": h, "month_of_year": m, **{f"{v}_mm": (norms.get((h, m)) or {}).get(v) for v in VARIABLES}}
             for h in basins.ids for m in range(1, 13)]
    _write_table(nrows, root / "normals.parquet", [("hybas_id", "int64"), ("month_of_year", "int64")] +
                 [(f"{v}_mm", "float32") for v in VARIABLES])
    _write_table(basins_rows(basins, checks), root / "basins.parquet", [
        ("hybas_id", "int64"), ("k", "int64"), ("area_km2", "float64"), ("lat", "float64"), ("lon", "float64"),
        ("outlets", "list<int64>"), ("down_id", "int64"), ("up_ids", "list<int64>"), ("linked", "bool"),
        ("runoff_ok", "bool"), ("runoff_check", "string"), ("era5_cells", "int64")])
    if basins.geometry:
        (root / "basins.geojson").write_text(json.dumps(basins_geojson(basins, checks), separators=(",", ":")))
    (root / "normals.json").write_text(json.dumps(normals_doc(norms, basins), separators=(",", ":"), allow_nan=False))
    years = year_files(rows, basins)
    (root / "years").mkdir()
    for y, doc in years.items():
        (root / "years" / f"{y}.json").write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    counts = {v: sum(1 for r in rows if r.get(f"{v}_mm") is not None) for v in VARIABLES}
    seconds = round(time.monotonic() - t0, 1)
    manifest = {
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_month": months[0] if months else None, "last_month": months[-1] if months else None,
        "months": len(months), "years": sorted(years), "basins": basins.n,
        "linked": sum(basins.linked), "runoff_ok": sum(basins.runoff_ok) - len(checks),
        "runoff_withheld": {why: sum(1 for x in checks.values() if x == why) for why in RUNOFF_CHECKS},
        "runoff_checks": RUNOFF_CHECKS, "rows": len(rows), "values": counts,
        "computed": sorted(era5), "skipped_for_time": sorted(skipped), "missing_hours": missing_hours,
        "era5_final_until": layout.get("final_until"), "era5t_until": layout.get("era5t_until"),
        "geoglows_last_month": q["last_month"], "normal_period": f"{NORMAL_YEARS[0]}-{NORMAL_YEARS[1]}",
        "units": UNITS, "variables": LABELS, "columns": COLUMNS, "method": METHOD, "closure": CLOSURE, "not": NOT,
        "licence": LICENCE, "sources": SOURCES, "smoke": smoke, "smoke_hours": smoke_hours or None,
        "cost": {"seconds": seconds, "bytes_read": counted(), "era5_bytes": era5_bytes,
                 "era5_seconds": round(era5_seconds, 1), "era5_months": len(era5),
                 "era5_chunks": sum(2 * e["hours"] for e in era5.values()), "geoglows_chunks": q["chunks"],
                 "geoglows_seconds": round(q_seconds, 1), "workers": workers},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, allow_nan=False))
    return {k: manifest[k] for k in ("first_month", "last_month", "months", "basins", "linked", "runoff_ok",
                                     "runoff_withheld", "computed", "skipped_for_time", "missing_hours", "smoke",
                                     "cost")}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/waterbalance`` and nothing else to the Archive dataset. A smoke run is refused."""
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    manifest = src / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run the water balance first")
    doc = json.loads(manifest.read_text())
    if doc.get("smoke"):
        raise ValueError("this is a smoke run (--smoke-hours); it is never published")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        shutil.copytree(src, stage)
        return str(publish_folder(Path(tmp), repo_id, token=token,
                                  allow_patterns=[f"{FOLDER}/*", f"{FOLDER}/years/*"],
                                  commit_message=f"waterbalance: {doc['first_month']} to {doc['last_month']}"))


# ── reading what is published (the MCP tool and the CLI) ───────────────────


def _load(local: str | Path | None, repo_id: str, name: str) -> bytes | None:
    if local:
        root = Path(local) / FOLDER if (Path(local) / FOLDER).exists() else Path(local)
        p = root / name
        return p.read_bytes() if p.exists() else None
    from aquascope.archive.forecasts import _get, published_url

    return _get(published_url(f"{FOLDER}/{name}", repo_id))


def _ring_contains(ring: list[list[float]], x: float, y: float) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi:
            inside = not inside
        j = i
    return inside


def feature_contains(geometry: dict[str, Any], lon: float, lat: float) -> bool:
    """Whether a GeoJSON Polygon or MultiPolygon holds the point (holes respected), in plain Python."""
    polys = geometry.get("coordinates") or []
    if geometry.get("type") == "Polygon":
        polys = [polys]
    for rings in polys:
        if rings and _ring_contains(rings[0], lon, lat) and not any(_ring_contains(h, lon, lat) for h in rings[1:]):
            return True
    return False


def basin_at(lat: float, lon: float, *, repo_id: str = DEFAULT_REPO, local: str | Path | None = None) -> int | None:
    """The level-4 subbasin holding the point, from the published outlines."""
    data = _load(local, repo_id, "basins.geojson")
    if not data:
        return None
    for f in json.loads(data).get("features", []):
        if feature_contains(f["geometry"], lon, lat):
            return int(f["properties"]["id"])
    return None


def water_balance(hybas_id: int | None = None, lat: float | None = None, lon: float | None = None,
                  month: str | None = None, *, months: int = 24, bbox: list[float] | tuple[float, ...] | None = None,
                  variable: str = "p", limit: int = 20, repo_id: str = DEFAULT_REPO,
                  local: str | Path | None = None) -> dict[str, Any]:
    """The published water balance: one subbasin's months (by ``hybas_id`` or a point), or a month's overview.

    With a subbasin: its area and flags, the ``months`` months up to ``month`` (default the newest) with P, ET,
    Q, the closure error and their anomalies, its 1991-2020 normals, and the month in words. Without one: for
    ``month`` (default the newest), in ``bbox`` if given (west, south, east, north), how many subbasins have each
    variable, their medians, and the ``limit`` subbasins furthest from normal in ``variable`` (p, et, q, closure).
    """
    if variable not in VARIABLES:
        raise ValueError(f"variable is one of {', '.join(VARIABLES)}")
    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox is west, south, east, north")
    if month is not None and (len(month) != 7 or month[4] != "-"):
        raise ValueError("month is YYYY-MM")
    raw = _load(local, repo_id, "manifest.json")
    manifest = json.loads(raw) if raw else {}
    data = _load(local, repo_id, "monthly.parquet") if manifest else None
    if not manifest or data is None:
        return {"available": False, "sentence": "No water balance is published yet; the monthly workflow "
                "(water-balance.yml) writes waterbalance/ in the Archive dataset."}
    import pyarrow.parquet as pq

    month = month or manifest.get("last_month")
    base = {"available": True, "month": month, "units": UNITS, "method": manifest.get("method"),
            "closure": manifest.get("closure"), "not": manifest.get("not"), "licence": manifest.get("licence"),
            "first_month": manifest.get("first_month"), "last_month": manifest.get("last_month"),
            "normal_period": manifest.get("normal_period")}
    braw = _load(local, repo_id, "basins.parquet")
    binfo = {int(r["hybas_id"]): r for r in pq.read_table(io.BytesIO(braw)).to_pylist()} if braw else {}
    if hybas_id is None and lat is not None and lon is not None:
        hybas_id = basin_at(float(lat), float(lon), repo_id=repo_id, local=local)
        if hybas_id is None:
            return {**base, "found": False, "sentence": "No level-4 subbasin holds that point (sea, or a coast "
                    "left out of the simplified outlines)."}
    table = pq.read_table(io.BytesIO(data))
    if hybas_id is not None:
        import pyarrow.compute as pc

        rows = table.filter(pc.equal(table["hybas_id"], int(hybas_id))).to_pylist()
        rows = sorted((r for r in rows if r["month"] <= (month or "9999-99")), key=lambda r: r["month"])
        if not rows:
            return {**base, "found": False, "hybas_id": hybas_id, "sentence": f"No water balance for subbasin "
                    f"{hybas_id} up to {month}."}
        rows = rows[-max(1, int(months)):]
        last = {k: (_r1(v) if isinstance(v, float) else v) for k, v in rows[-1].items()}
        nraw = _load(local, repo_id, "normals.parquet")
        norm = []
        if nraw:
            import pyarrow.compute as pc

            nt = pq.read_table(io.BytesIO(nraw))
            norm = sorted(nt.filter(pc.equal(nt["hybas_id"], int(hybas_id))).to_pylist(),
                          key=lambda r: r["month_of_year"])
        info = binfo.get(int(hybas_id), {})
        series = [{k: (_r1(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
        return {**base, "found": True, "hybas_id": int(hybas_id), "area_km2": info.get("area_km2"),
                "runoff_ok": info.get("runoff_ok"), "runoff_check": info.get("runoff_check"),
                "runoff_withheld_because": RUNOFF_CHECKS.get(info.get("runoff_check") or ""),
                "outlets": info.get("outlets"), "down_id": info.get("down_id"),
                "up_ids": info.get("up_ids"), "month": last["month"], "this_month": last,
                "sentence": basin_sentence(last, area_km2=info.get("area_km2")), "series": series,
                "normals": [{k: (_r1(v) if isinstance(v, float) else v) for k, v in r.items()} for r in norm]}
    import pyarrow.compute as pc

    rows = table.filter(pc.equal(table["month"], month)).to_pylist()
    if bbox is not None:
        from aquascope.archive.warnings import _in_bbox

        rows = [r for r in rows if r["hybas_id"] in binfo and
                _in_bbox(binfo[r["hybas_id"]]["lat"], binfo[r["hybas_id"]]["lon"], bbox)]
    if not rows:
        return {**base, "n": 0, "sentence": f"No water balance for {month}" + (" in this box." if bbox else ".")}

    def median(v: str) -> float | None:
        xs = sorted(r[v] for r in rows if r.get(v) is not None)
        if not xs:
            return None
        mid = len(xs) // 2
        return _r1(xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2)

    counts = {v: sum(1 for r in rows if r.get(f"{v}_mm") is not None) for v in VARIABLES}
    medians = {v: median(f"{v}_mm") for v in VARIABLES}
    key = f"{variable}_anom"
    ranked = sorted((r for r in rows if r.get(key) is not None), key=lambda r: -abs(r[key]))[:max(0, int(limit))]
    where = " in this box" if bbox is not None else ""
    sentence = (f"{month_label(month)}{where}: {len(rows):,} subbasins; median rain {medians['p']} mm, evaporation "
                f"{medians['et']} mm, runoff {medians['q']} mm ({counts['q']:,} with runoff), closure error "
                f"{medians['closure']} mm.")
    return {**base, "n": len(rows), "bbox": list(bbox) if bbox is not None else None, "counts": counts,
            "medians": medians, "variable": variable, "sentence": sentence,
            "furthest_from_normal": [{k: (_r1(v) if isinstance(v, float) else v) for k, v in r.items()}
                                     for r in ranked]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.waterbalance", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=None, help="the newest month to compute (YYYY-MM)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--time-budget-min", type=float, default=270.0)
    p.add_argument("--smoke-hours", type=int, default=None, help="hours of the newest month read, for a smoke run")
    p.add_argument("--previous", default=None, help="a folder with last time's waterbalance/ instead of the Archive")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "run":
        info = run(a.out, repo_id=a.repo, start=a.start, end=a.end, workers=a.workers,
                   time_budget_s=a.time_budget_min * 60, smoke_hours=a.smoke_hours, previous=a.previous)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
