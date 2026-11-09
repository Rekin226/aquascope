"""Groundwater on the map (#558): a head surface from the Archive's wells, and your MODFLOW heads.

Two things ModelMuse, FloPy and GMS show, made keyless:

1. **A groundwater-level surface and its change** (:func:`groundwater_surface`, :func:`surface_from_readings`). The
   Archive mirrors the Environment Agency's groundwater levels for England (``obs/groundwater_level/uk_ea``, about
   380 wells, Open Government Licence v3.0). Their values are **heads in metres above Ordnance Datum** (mAOD), so the
   datum is known and common to every well: the level surface is a head surface, not a depth to water (that would need
   each well's ground level, which the Archive does not hold). The change between two months needs no datum at all.

   How a month's value is chosen per well: the mean of its readings in that month; with none, the reading nearest the
   15th within :data:`NEAREST_DAYS` days. When the month asked for has fewer than :data:`COVERAGE_SHARE` of the
   wells the twelve months before it had (the Archive is a few weeks behind), the latest month that does is used,
   and the result says so.

   How it is interpolated: inverse distance weighting (power :data:`IDW_POWER`) on a grid regular in Web Mercator
   (so it lies exactly on the map), with weights that fade to zero at twice :data:`RADIUS_KM`.

   **The density mask**, the honest part: a cell is drawn only where at least :data:`MIN_WELLS` wells lie within
   :data:`RADIUS_KM` km of it. Cells with one or two wells that close are returned as "too few wells" (the map greys
   them); cells with none are left empty. Wells in different aquifers are mixed by the interpolation, so this is a
   picture of the measurements, not a groundwater model.

2. **Your MODFLOW heads** (:func:`read_modflow_heads`). A binary head (or drawdown) file from MODFLOW-2005 or
   MODFLOW 6, read by :func:`read_heads`, a small pure-numpy reader of the standard layout (a 44-byte header per layer
   record in single precision, 52 in double, then ``nrow * ncol`` values), placed on its grid from a ``.dis`` (MODFLOW 6
   or 2005 text), a ``.disv`` (MODFLOW 6 text), a ``.dis.grb``/``.disv.grb`` (MODFLOW 6 binary grid file), or an origin,
   rotation, cell size and CRS you type in. Cells come back as polygons in lon/lat, heads per layer and time.

   Why not FloPy: FloPy 3.11.0 (CC0) is a pure-Python wheel and it does install with micropip in Pyodide 314 and read
   a ``.hds`` with ``flopy.utils.HeadFile`` (checked 2026-10-11 on the Freyberg examples). But ``import flopy``
   imports matplotlib, so the browser would download about 10 MB more (matplotlib 7.0 MB, fonttools 1.2 MB, Pillow
   1.0 MB, the FloPy wheel 1.5 MB and a few small ones) to read a file whose layout fits in a page. The reader here
   gives the same numbers (a test compares it with FloPy where FloPy is installed) and imports on a bare install.

Nothing leaves the machine: the head file is read where it is (in the Explorer, in your browser's worker). Projecting
a grid from a projected CRS needs pyproj (``aquascope[spatial]``; the Explorer loads Pyodide's build when needed).
"""

from __future__ import annotations

import math
import re
import struct
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

#: A cell of the surface is drawn when at least MIN_WELLS wells lie within RADIUS_KM km of its centre.
RADIUS_KM = 20.0
MIN_WELLS = 3
#: Inverse distance weighting power, and the distance (in units of RADIUS_KM) where weights reach zero.
IDW_POWER = 2.0
TAPER = 2.0
#: Cells on the long side of the surface grid.
MAX_GRID = 220
#: With no reading in the month, the reading nearest the 15th within this many days stands in.
NEAREST_DAYS = 31
#: A month counts as covered when at least this share of the wells reporting in the year before it report in it.
COVERAGE_SHARE = 0.6
#: The change is shown against the same month this many months earlier, unless another month is asked for.
CHANGE_MONTHS = 12
#: Grids with more cells per layer than this are drawn in blocks (MODFLOW) or refused (DISV).
MAX_CELLS = 60_000
EARTH_R_KM = 6371.0088
WEB_MERCATOR_R = 6378137.0
MAX_MERCATOR_LAT = 85.05112878

#: What the Archive holds per source: the datum the values are measured from, or None when it is not common.
SOURCES: dict[str, dict[str, Any]] = {
    "uk_ea": {
        "label": "Environment Agency groundwater levels (England)",
        "datum": "mAOD (metres above Ordnance Datum Newlyn)",
        "kind": "head",
        "licence": "OGL-UK-3.0",
        "attribution": "Environment Agency, Open Government Licence v3.0",
    },
}

RULE = (f"Drawn where at least {MIN_WELLS} wells lie within {RADIUS_KM:g} km; grey where only one or two do; "
        f"empty where none do. Inverse distance weighting (power {IDW_POWER:g}).")
NOT = ("A picture of the wells' measurements, interpolated: not a groundwater model. Wells in different aquifers "
       "are mixed, and between wells the surface is a guess.")
GRACE_HINT = ("For the regional picture, the GRACE water storage anomaly (Layers, Water storage anomaly) shows the "
              "same change from space, at about 300 km.")

# ── months and dates ────────────────────────────────────────────────────────


def _month_of(value: Any) -> tuple[int, int]:
    """(year, month) from 'YYYY-MM', 'YYYY-MM-DD', a date or a datetime."""
    if isinstance(value, (date, datetime)):
        return value.year, value.month
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    m = re.match(r"^\s*(\d{4})-(\d{1,2})", str(value or ""))
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError(f"not a month: {value!r} (use YYYY-MM)")
    return int(m.group(1)), int(m.group(2))


def _month_str(ym: tuple[int, int]) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def _add_months(ym: tuple[int, int], n: int) -> tuple[int, int]:
    k = ym[0] * 12 + (ym[1] - 1) + n
    return k // 12, k % 12 + 1


def _month_label(ym: tuple[int, int]) -> str:
    return date(ym[0], ym[1], 1).strftime("%B %Y")


def _days(dates: Any) -> np.ndarray:
    """Dates (ISO strings, dates or datetime64) as integer days since 1970-01-01."""
    arr = np.asarray(dates)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[D]").astype(np.int64)
    return np.array([np.datetime64(str(d)[:10], "D") for d in arr], dtype="datetime64[D]").astype(np.int64)


def _day_of(ym: tuple[int, int], day: int = 1) -> int:
    return int(np.datetime64(f"{ym[0]:04d}-{ym[1]:02d}-{day:02d}", "D").astype(np.int64))


def _month_keys(days: np.ndarray) -> np.ndarray:
    """Month index (year * 12 + month - 1) of integer days."""
    d = days.astype("datetime64[D]")
    years = d.astype("datetime64[Y]").astype(int) + 1970
    months = d.astype("datetime64[M]").astype(int) % 12
    return years * 12 + months


# ── the wells' values for a month ───────────────────────────────────────────


def _readings_arrays(readings: Any, n_wells: int, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(well index, day, value) arrays from {"well"|"id", "date", "value"} columns or [id, date, value] rows."""
    if isinstance(readings, dict):
        dates = readings.get("date") or []
        values = readings.get("value") or []
        if "well" in readings:
            well = np.asarray(readings["well"], dtype=np.int64)
        else:
            pos = {w: i for i, w in enumerate(ids)}
            well = np.array([pos.get(str(i), -1) for i in readings.get("id") or []], dtype=np.int64)
    else:
        rows = list(readings or [])
        pos = {w: i for i, w in enumerate(ids)}
        well = np.array([pos.get(str(r[0]), -1) for r in rows], dtype=np.int64)
        dates = [r[1] for r in rows]
        values = [r[2] for r in rows]
    if not (len(well) == len(dates) == len(values)):
        raise ValueError("readings: well, date and value must have the same length")
    if not len(well):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, float)
    day = _days(dates)
    val = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    ok = (well >= 0) & (well < n_wells) & np.isfinite(val)
    return well[ok], day[ok], val[ok]


def month_values(well: np.ndarray, day: np.ndarray, value: np.ndarray, month: Any, n_wells: int) -> list[dict | None]:
    """Each well's value for a month: the mean of its readings in it, else the one nearest the 15th within
    NEAREST_DAYS days, else None. Items: {value, how ("month mean" or "nearest"), n, date (the nearest's)}."""
    ym = _month_of(month)
    lo, hi = _day_of(ym), _day_of(_add_months(ym, 1))
    mid = _day_of(ym, 15)
    out: list[dict | None] = [None] * n_wells
    inside = (day >= lo) & (day < hi)
    if inside.any():
        sums = np.bincount(well[inside], weights=value[inside], minlength=n_wells)
        counts = np.bincount(well[inside], minlength=n_wells)
        for i in np.nonzero(counts)[0]:
            out[int(i)] = {"value": float(sums[i] / counts[i]), "how": "month mean", "n": int(counts[i])}
    near = (~inside) & (np.abs(day - mid) <= NEAREST_DAYS)
    if near.any():
        order = np.argsort(np.abs(day[near] - mid), kind="stable")
        w, d, v = well[near][order], day[near][order], value[near][order]
        for i, dd, vv in zip(w, d, v):
            if out[int(i)] is None:
                out[int(i)] = {"value": float(vv), "how": "nearest", "n": 1,
                               "date": str(np.datetime64(int(dd), "D"))}
    return out


def pick_month(well: np.ndarray, day: np.ndarray, wanted: Any) -> tuple[tuple[int, int], str | None]:
    """The month to draw: the one asked for when enough wells report in it, else the latest month before it that
    has COVERAGE_SHARE of the wells the year before it had. Returns (month, a note or None)."""
    ym = _month_of(wanted)
    if not len(day):
        return ym, None
    keys = _month_keys(day)
    k0 = ym[0] * 12 + ym[1] - 1
    window = (keys > k0 - 13) & (keys <= k0)
    per_month = {}
    for k in np.unique(keys[window]):
        per_month[int(k)] = len(np.unique(well[window][keys[window] == k]))
    typical = float(np.median(list(per_month.values()))) if per_month else 0.0
    if not typical:
        return ym, None
    need = COVERAGE_SHARE * typical
    for k in range(k0, k0 - 13, -1):
        if per_month.get(k, 0) >= need:
            got = (k // 12, k % 12 + 1)
            if got == ym:
                return ym, None
            n = per_month.get(k0, 0)
            return got, (f"{_month_label(ym)} has {n} well{'' if n == 1 else 's'} so far, so this is "
                         f"{_month_label(got)}, the latest month most wells reported in.")
    return ym, f"Few wells report around {_month_label(ym)}."


# ── geometry ────────────────────────────────────────────────────────────────


def _merc_y(lat: Any) -> np.ndarray:
    lat = np.clip(np.asarray(lat, dtype=float), -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT)
    return WEB_MERCATOR_R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _merc_lat(y: Any) -> np.ndarray:
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y, dtype=float) / WEB_MERCATOR_R)) - np.pi / 2)


def _dist_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between every point of 1 (rows) and 2 (columns)."""
    p1, p2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dl = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    w, s, e, n = (float(v) for v in bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError("bbox must be [west, south, east, north] in degrees, west < east and south < north")
    return w, s, e, n


def _grow(bbox: tuple[float, float, float, float], km: float) -> tuple[float, float, float, float]:
    w, s, e, n = bbox
    dlat = km / 111.32
    dlon = km / (111.32 * max(0.05, math.cos(math.radians(max(abs(s), abs(n))))))
    return max(-180.0, w - dlon), max(-90.0, s - dlat), min(180.0, e + dlon), min(90.0, n + dlat)


def interpolate(lons: Any, lats: Any, values: Any, bbox: Any = None, *, radius_km: float = RADIUS_KM,
                min_wells: int = MIN_WELLS, power: float = IDW_POWER, max_grid: int = MAX_GRID) -> dict[str, Any]:
    """Inverse distance weighting of point values on a grid regular in Web Mercator, with the density mask.

    The grid covers ``bbox`` cut to the wells' extent grown by ``radius_km``. Returns {west, south, east, north,
    nx, ny, values (ny x nx, north row first, NaN where not drawn), status (ny x nx: 2 drawn, 1 too few wells,
    0 no well near), cell_km} or None when there is nothing to draw."""
    lon = np.asarray(lons, dtype=float)
    lat = np.asarray(lats, dtype=float)
    val = np.asarray(values, dtype=float)
    ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon, lat, val = lon[ok], lat[ok], val[ok]
    if not len(val):
        return None
    reach = _grow((float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())), radius_km)
    box = _bbox(bbox) or reach
    w, s, e, n = max(box[0], reach[0]), max(box[1], reach[1]), min(box[2], reach[2]), min(box[3], reach[3])
    if not (w < e and s < n):
        return None
    x0, x1 = np.radians(w) * WEB_MERCATOR_R, np.radians(e) * WEB_MERCATOR_R
    y0, y1 = float(_merc_y(s)), float(_merc_y(n))
    step = max(x1 - x0, y1 - y0) / max(8, int(max_grid))
    nx, ny = max(1, int(round((x1 - x0) / step))), max(1, int(round((y1 - y0) / step)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny
    clon = np.degrees(xs / WEB_MERCATOR_R)
    clat = _merc_lat(ys)
    glon, glat = np.meshgrid(clon, clat)
    glon, glat = glon.ravel(), glat.ravel()
    out = np.full(glon.size, np.nan)
    status = np.zeros(glon.size, dtype=np.int8)
    far = TAPER * radius_km
    chunk = max(1, 2_000_000 // max(1, len(val)))
    for a in range(0, glon.size, chunk):
        d = _dist_km(glon[a:a + chunk], glat[a:a + chunk], lon, lat)
        near = (d <= radius_km).sum(axis=1)
        wgt = np.where(d < far, (1.0 / np.maximum(d, 0.05) ** power) * (1 - (d / far) ** 2) ** 2, 0.0)
        tot = wgt.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = (wgt * val[None, :]).sum(axis=1) / tot
        st = np.where(near >= min_wells, 2, np.where(near >= 1, 1, 0)).astype(np.int8)
        out[a:a + chunk] = np.where(st == 2, est, np.nan)
        status[a:a + chunk] = st
    lat_mid = math.radians((s + n) / 2)
    return {
        "west": w, "south": s, "east": e, "north": n, "nx": nx, "ny": ny,
        "values": out.reshape(ny, nx), "status": status.reshape(ny, nx),
        "cell_km": round(step / 1000 * math.cos(lat_mid), 3),
    }


def nice_step(span: float, target: int = 8) -> float:
    """A round contour interval giving about ``target`` lines over ``span``."""
    if not span or not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 2.5, 5, 10):
        if raw <= k * mag:
            return k * mag
    return 10 * mag


def contour_lines(grid: dict[str, Any], step: float | None = None, *, target: int = 8) -> dict[str, Any]:
    """Contour segments of a surface grid by marching squares over the drawn cells' centres.
    Returns {step, levels: [{level, segments: [[[lon, lat], [lon, lat]], ...]}]} (segments are not joined)."""
    v = np.asarray(grid["values"], dtype=float)
    ny, nx = v.shape
    finite = v[np.isfinite(v)]
    if finite.size < 4 or nx < 2 or ny < 2:
        return {"step": None, "levels": []}
    lo, hi = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    step = step or nice_step(hi - lo, target)
    x0 = math.radians(grid["west"]) * WEB_MERCATOR_R
    x1 = math.radians(grid["east"]) * WEB_MERCATOR_R
    y0, y1 = float(_merc_y(grid["south"])), float(_merc_y(grid["north"]))
    cx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    cy = y1 - (np.arange(ny) + 0.5) * (y1 - y0) / ny

    def at(i: float, j: float) -> list[float]:
        x = float(np.interp(j, np.arange(nx), cx))
        y = float(np.interp(i, np.arange(ny), cy))
        return [round(math.degrees(x / WEB_MERCATOR_R), 5), round(float(_merc_lat(y)), 5)]

    levels = []
    first = math.ceil(lo / step) * step
    k = 0
    while first + k * step <= hi and k < 60:
        level = first + k * step
        k += 1
        segs = []
        a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]
        okq = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
        code = ((a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8) * okq
        for i, j in zip(*np.nonzero((code > 0) & (code < 15))):
            q = [float(a[i, j]), float(b[i, j]), float(c[i, j]), float(d[i, j])]
            corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            pts = []
            for e in range(4):
                p, r = q[e], q[(e + 1) % 4]
                if (p > level) != (r > level):
                    t = (level - p) / (r - p)
                    (ia, ja), (ib, jb) = corners[e], corners[(e + 1) % 4]
                    pts.append(at(ia + t * (ib - ia), ja + t * (jb - ja)))
            for m in range(0, len(pts) - 1, 2):
                segs.append([pts[m], pts[m + 1]])
        if segs:
            levels.append({"level": round(level, 6), "segments": segs})
    return {"step": step, "levels": levels}


# ── the surface ─────────────────────────────────────────────────────────────


def _rounded(a: np.ndarray, digits: int = 3) -> list:
    return [None if not math.isfinite(x) else round(float(x), digits) for x in np.asarray(a, dtype=float).ravel()]


def _robust_range(vals: np.ndarray, symmetric: bool) -> list[float] | None:
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return None
    if symmetric:
        # A change saturates at the 90th percentile of its size, rounded up to a round number, so the
        # ordinary rises and falls show and the few large ones do not wash the rest out to white.
        m = max(float(np.percentile(np.abs(vals), 90)), 0.1)
        m = nice_step(m, 1)
        return [-round(m, 3), round(m, 3)]
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    if hi - lo < 0.1:
        lo, hi = lo - 0.05, hi + 0.05
    return [round(lo, 3), round(hi, 3)]


def surface_from_readings(
    wells: list[dict[str, Any]],
    readings: Any,
    *,
    month: Any = None,
    mode: str = "level",
    compare: Any = None,
    bbox: Any = None,
    source: str = "uk_ea",
    datum: str | None = None,
    radius_km: float = RADIUS_KM,
    min_wells: int = MIN_WELLS,
    power: float = IDW_POWER,
    max_grid: int = MAX_GRID,
    contours: bool = True,
) -> dict[str, Any]:
    """The surface from wells and their readings, with no network: the engine behind :func:`groundwater_surface`
    and the Explorer's Groundwater layer (the page reads the Archive's readings with DuckDB and hands them here).

    ``wells``: [{id, lon, lat, name?}]. ``readings``: {"well": [index into wells], "date": [...], "value": [...]}
    (or "id" in place of "well"), or rows [id, date, value]. ``month``: the month to draw (YYYY-MM or a date; default
    the newest in the readings). ``mode``: "level" (the head surface; needs a common datum) or "change" (this month
    minus ``compare``, default CHANGE_MONTHS months earlier). ``bbox``: [west, south, east, north], the view."""
    if mode not in ("level", "change"):
        raise ValueError("mode must be 'level' or 'change'")
    meta = SOURCES.get(source, {})
    datum = datum if datum is not None else meta.get("datum")
    ids = [str(w.get("id")) for w in wells]
    wlon = np.array([float(w.get("lon", np.nan)) for w in wells], dtype=float)
    wlat = np.array([float(w.get("lat", np.nan)) for w in wells], dtype=float)
    well, day, value = _readings_arrays(readings, len(wells), ids)
    box = _bbox(bbox)
    base = {
        "kind": "groundwater_surface", "mode": mode, "source": source, "unit": "m", "datum": datum,
        "rule": RULE, "radius_km": radius_km, "min_wells": min_wells, "not": NOT, "grace": GRACE_HINT,
        "attribution": meta.get("attribution"), "licence": meta.get("licence"), "bbox": list(box) if box else None,
        "notes": [],
    }
    if mode == "level" and not datum:
        return {**base, "error": "These wells have no common datum, so only the change between two months can be "
                                 "drawn (mode='change')."}
    if not len(day):
        return {**base, "error": "No readings from these wells."}
    wanted = _month_of(month) if month else _month_of(str(np.datetime64(int(day.max()), "D")))
    ym, note = pick_month(well, day, wanted)
    if note:
        base["notes"].append(note)
    now = month_values(well, day, value, ym, len(wells))
    cmp_ym = None
    then: list[dict | None] = [None] * len(wells)
    if mode == "change":
        cmp_ym = _month_of(compare) if compare else _add_months(ym, -CHANGE_MONTHS)
        if cmp_ym >= ym:
            return {**base, "error": "compare must be a month before the month drawn"}
        then = month_values(well, day, value, cmp_ym, len(wells))
    vals = np.full(len(wells), np.nan)
    for i, rec in enumerate(now):
        if rec is None:
            continue
        if mode == "level":
            vals[i] = rec["value"]
        elif then[i] is not None:
            vals[i] = rec["value"] - then[i]["value"]
    have = np.isfinite(vals) & np.isfinite(wlon) & np.isfinite(wlat)
    reach = _grow(box, TAPER * radius_km) if box else None
    if reach:
        have &= (wlon >= reach[0]) & (wlon <= reach[2]) & (wlat >= reach[1]) & (wlat <= reach[3])
    grid = interpolate(wlon[have], wlat[have], vals[have], box, radius_km=radius_km, min_wells=min_wells,
                       power=power, max_grid=max_grid) if have.any() else None
    in_view = have.copy()
    if box:
        in_view &= (wlon >= box[0]) & (wlon <= box[2]) & (wlat >= box[1]) & (wlat <= box[3])
    n_in_view = int(in_view.sum())
    # Every well the surface used, the ones just outside the view too: a cell at the view's edge is drawn or
    # masked by the wells within RADIUS_KM of it, and the map card counts them from this list.
    out_wells = []
    for i in np.nonzero(have)[0]:
        rec = now[i]
        item = {"id": ids[i], "name": wells[i].get("name"), "lon": round(float(wlon[i]), 5),
                "lat": round(float(wlat[i]), 5), "value": round(float(vals[i]), 3), "how": rec["how"]}
        if not in_view[i]:
            item["in_view"] = False
        if rec.get("date"):
            item["date"] = rec["date"]
        if mode == "change":
            item["level"] = round(rec["value"], 3)
            item["level_before"] = round(then[i]["value"], 3)
            if then[i].get("date"):
                item["date_before"] = then[i]["date"]
        out_wells.append(item)
    reported = sum(1 for r in now if r is not None)
    label = _month_label(ym) if mode == "level" else f"{_month_label(cmp_ym)} to {_month_label(ym)}"
    res = {
        **base, "month": _month_str(ym), "asked": _month_str(wanted),
        "compare": _month_str(cmp_ym) if cmp_ym else None, "label": label,
        "quantity": ("groundwater head" if mode == "level" else "change in groundwater level"),
        "wells": out_wells, "n_wells_reporting": reported, "n_wells_used": int(have.sum()),
        "n_wells_in_view": n_in_view,
    }
    if grid is None:
        res["grid"] = None
        res["summary"] = (f"Not enough wells with a value for {label} here to draw a surface "
                          f"({n_in_view} in view).")
        return res
    v, st = grid["values"], grid["status"]
    drawn = int((st == 2).sum())
    reached = int((st > 0).sum())
    rng = _robust_range(v[st == 2], symmetric=(mode == "change"))
    res["grid"] = {k: grid[k] for k in ("west", "south", "east", "north", "nx", "ny", "cell_km")}
    res["grid"].update({"values": _rounded(v, 3), "status": [int(x) for x in st.ravel()]})
    res["range"] = rng
    res["drawn_share"] = round(drawn / reached, 3) if reached else 0.0
    vv = v[st == 2]
    if mode == "change" and vv.size:
        res["rising_share"] = round(float((vv > 0.05).mean()), 3)
        res["falling_share"] = round(float((vv < -0.05).mean()), 3)
    if contours and mode == "level" and drawn >= 4:
        res["contours"] = contour_lines(grid)
    shown = f"{res['drawn_share']:.0%} of the area near a well" if reached else "nothing"
    if mode == "level":
        res["summary"] = (f"Groundwater head, {label}, from {n_in_view} wells in view ({datum}); "
                          f"drawn over {shown}.")
    else:
        rise, fall = res.get("rising_share", 0), res.get("falling_share", 0)
        res["summary"] = (f"Change in groundwater level from {label}, {n_in_view} wells in view: rising "
                          f"(over 5 cm) on {rise:.0%} of the drawn area, falling on {fall:.0%}; drawn over {shown}.")
    return res


def _archive_wells(source: str, repo_id: str) -> list[dict[str, Any]]:
    from aquascope.archive.catalog import load_stations

    rows = load_stations(repo_id=repo_id)
    return [{"id": str(r["station_id"]), "name": r.get("name"), "lon": r.get("longitude"), "lat": r.get("latitude")}
            for r in rows if r.get("source") == source and "groundwater_level" in (r.get("variables") or [])]


def groundwater_surface(
    bbox: Any = None,
    date: Any = None,
    *,
    month: Any = None,
    mode: str = "level",
    compare: Any = None,
    source: str = "uk_ea",
    repo_id: str = "Rekin226/aquascope-gauges",
    **kw: Any,
) -> dict[str, Any]:
    """A groundwater-level surface for a box from the Archive's wells (the ModelMuse view of the measurements).

    ``bbox``: [west, south, east, north] (default: every well). ``date`` or ``month``: the month to draw (default the
    newest well-covered month). ``mode``: "level" (head in mAOD for the Environment Agency's wells) or "change"
    (against ``compare``, default 12 months earlier). The density mask, the method and what it is not are in the
    result (``rule``, ``not``). Reads the Archive's catalogue and its groundwater bundle (a few MB, cached a day)."""
    from aquascope.archive.bundles import load_observations

    if source not in SOURCES:
        raise ValueError(f"the Archive mirrors groundwater levels for {', '.join(SOURCES)} only")
    wells = _archive_wells(source, repo_id)
    if not wells:
        return {"kind": "groundwater_surface", "error": "No groundwater wells in the Archive catalogue."}
    box = _bbox(bbox)
    if box:
        reach = _grow(box, TAPER * kw.get("radius_km", RADIUS_KM))
        wells = [w for w in wells if w["lon"] is not None and w["lat"] is not None
                 and reach[0] <= float(w["lon"]) <= reach[2] and reach[1] <= float(w["lat"]) <= reach[3]]
    df = load_observations(source, "groundwater_level", station_ids=[w["id"] for w in wells], repo_id=repo_id)
    when = month or date
    if when:
        ym = _month_of(when)
        first = _add_months(ym, -(CHANGE_MONTHS + 14))
        if compare:
            first = min(first, _add_months(_month_of(compare), -2))
        lo, hi = f"{_month_str(first)}-01", f"{_month_str(_add_months(ym, 2))}-01"
        df = df[(df["date"] >= lo) & (df["date"] < hi)]
    readings = {"id": df["station_id"].astype(str).tolist(), "date": df["date"].astype(str).str[:10].tolist(),
                "value": df["value"].astype(float).tolist()}
    return surface_from_readings(wells, readings, month=when, mode=mode, compare=compare, bbox=box, source=source,
                                 **kw)


def surface_summary(res: dict[str, Any]) -> dict[str, Any]:
    """The surface without its grid values and contours (for an MCP answer or a terminal)."""
    out = {k: v for k, v in res.items() if k not in ("grid", "contours")}
    g = res.get("grid")
    if g:
        out["grid"] = {k: g[k] for k in ("west", "south", "east", "north", "nx", "ny", "cell_km")}
    if res.get("contours"):
        out["contour_step_m"] = res["contours"].get("step")
    return out


def surface_geojson(res: dict[str, Any]) -> dict[str, Any]:
    """The drawn cells as polygons with their value, the wells as points and the contours as lines."""
    feats: list[dict[str, Any]] = []
    g = res.get("grid")
    if g:
        nx, ny = g["nx"], g["ny"]
        xs = np.linspace(math.radians(g["west"]), math.radians(g["east"]), nx + 1)
        lons = np.degrees(xs)
        lats = _merc_lat(np.linspace(float(_merc_y(g["north"])), float(_merc_y(g["south"])), ny + 1))
        for k, (val, st) in enumerate(zip(g["values"], g["status"])):
            if st != 2 or val is None:
                continue
            i, j = divmod(k, nx)
            ring = [[lons[j], lats[i]], [lons[j + 1], lats[i]], [lons[j + 1], lats[i + 1]], [lons[j], lats[i + 1]]]
            ring = [[round(float(a), 5), round(float(b), 5)] for a, b in ring]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                          "properties": {"layer": "surface", "value": val}})
    for lv in (res.get("contours") or {}).get("levels", []):
        feats.append({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": lv["segments"]},
                      "properties": {"layer": "contour", "value": lv["level"]}})
    for w in res.get("wells") or []:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [w["lon"], w["lat"]]},
                      "properties": {"layer": "well", **{k: v for k, v in w.items() if k not in ("lon", "lat")}}})
    return {"type": "FeatureCollection", "features": feats,
            "properties": {k: res.get(k) for k in ("mode", "label", "unit", "datum", "rule", "attribution", "licence")}}


# ── MODFLOW: the binary head file ───────────────────────────────────────────

#: Values MODFLOW writes for inactive (HNOFLO) and dry (HDRY) cells that are read as "no head".
NO_HEAD = (-999.99, -999.0, -9999.0, -99999.0, 999.99, 9999.0)
_TEXT_OK = re.compile(rb"^[ A-Za-z0-9_\-\.]{16}$")


def _as_bytes(data: Any) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    return Path(data).read_bytes()


def _header(buf: bytes, pos: int, prec: str) -> tuple[int, int, float, float, str, int, int, int] | None:
    f = "f" if prec == "single" else "d"
    size = 44 if prec == "single" else 52
    if pos + size > len(buf):
        return None
    kstp, kper, pertim, totim = struct.unpack_from(f"<ii{f}{f}", buf, pos)
    off = pos + (16 if prec == "single" else 24)
    text = buf[off:off + 16]
    ncol, nrow, ilay = struct.unpack_from("<iii", buf, off + 16)
    if not _TEXT_OK.match(text) or not text.strip() or not 0 < ncol < 10**8 or not 0 < nrow < 10**8:
        return None
    if not (math.isfinite(totim) and math.isfinite(pertim)):
        return None
    return kstp, kper, pertim, totim, text.decode("ascii").strip(), ncol, nrow, ilay


def heads_precision(buf: bytes) -> str:
    """'single' or 'double', from the first record's header (the text must read as text and the sizes fit)."""
    for prec in ("single", "double"):
        h = _header(buf, 0, prec)
        if h is None:
            continue
        size = (44 if prec == "single" else 52) + h[5] * h[6] * (4 if prec == "single" else 8)
        if size <= len(buf) and (size == len(buf) or _header(buf, size, prec) is not None):
            return prec
    raise ValueError("not a MODFLOW binary head file (no readable record header; is it a formatted text file?)")


def read_heads(data: Any) -> dict[str, Any]:
    """Read a MODFLOW binary head (or drawdown) file: MODFLOW-2005 or MODFLOW 6, single or double precision.

    Returns {text, precision, nrow, ncol, layers (1-based), times (totim), kstpkper, pertim, heads}: ``heads`` is a
    float array (time, layer, nrow * ncol) with NaN for inactive and dry cells (|h| >= 1e20 or a NO_HEAD value) and
    for layers a time does not write."""
    buf = _as_bytes(data)
    prec = heads_precision(buf)
    hsize, vsize = (44, 4) if prec == "single" else (52, 8)
    dtype = np.dtype("<f4" if prec == "single" else "<f8")
    pos = 0
    records = []
    shape = None
    while pos < len(buf):
        h = _header(buf, pos, prec)
        if h is None:
            raise ValueError(f"unreadable record header at byte {pos} (a truncated file?)")
        kstp, kper, pertim, totim, text, ncol, nrow, ilay = h
        n = ncol * nrow
        if shape is None:
            shape = (nrow, ncol)
        elif (nrow, ncol) != shape:
            raise ValueError("records of different sizes in one file (an unstructured DISU file?) are not supported")
        if pos + hsize + n * vsize > len(buf):
            raise ValueError("the file ends inside a record (a truncated file?)")
        arr = np.frombuffer(buf, dtype=dtype, count=n, offset=pos + hsize)
        records.append((kstp, kper, pertim, totim, text, abs(ilay) or 1, arr))
        pos += hsize + n * vsize
    if not records:
        raise ValueError("no records in the head file")
    texts = sorted({r[4] for r in records})
    keys: list[tuple[int, int, float]] = []
    seen: dict[tuple[int, int, float], int] = {}
    for r in records:
        k = (r[0], r[1], r[3])
        if k not in seen:
            seen[k] = len(keys)
            keys.append(k)
    layers = sorted({r[5] for r in records})
    lpos = {lay: i for i, lay in enumerate(layers)}
    heads = np.full((len(keys), len(layers), shape[0] * shape[1]), np.nan)
    pertim = [0.0] * len(keys)
    for kstp, kper, pt, totim, _text, lay, arr in records:
        t = seen[(kstp, kper, totim)]
        heads[t, lpos[lay]] = arr.astype(float)
        pertim[t] = float(pt)
    bad = ~np.isfinite(heads) | (np.abs(heads) >= 1e20)
    for v in NO_HEAD:
        bad |= np.isclose(heads, v, rtol=0, atol=1e-3)
    heads[bad] = np.nan
    return {
        "text": " / ".join(texts), "precision": prec, "nrow": shape[0], "ncol": shape[1], "layers": layers,
        "times": [float(k[2]) for k in keys], "kstpkper": [[int(k[0]), int(k[1])] for k in keys],
        "pertim": pertim, "heads": heads,
    }


# ── MODFLOW: the grid ───────────────────────────────────────────────────────

LENGTH_UNITS_M = {"meters": 1.0, "metres": 1.0, "m": 1.0, "feet": 0.3048, "ft": 0.3048, "centimeters": 0.01,
                  "cm": 0.01}
#: MODFLOW-2005 LENUNI codes.
LENUNI = {1: "feet", 2: "meters", 3: "centimeters"}


def _strip_comment(line: str) -> str:
    for mark in ("#", "!", "//"):
        k = line.find(mark)
        if k >= 0:
            line = line[:k]
    return line.strip()


def _numbers(lines: list[str], start: int, n: int) -> tuple[list[float], int]:
    out: list[float] = []
    i = start
    while len(out) < n and i < len(lines):
        for tok in _strip_comment(lines[i]).replace(",", " ").split():
            if "*" in tok:  # Fortran repeat count: 20*250.0
                k, v = tok.split("*", 1)
                out.extend([float(v)] * int(k))
            else:
                out.append(float(tok.replace("D", "E").replace("d", "e")))
        i += 1
    if len(out) < n:
        raise ValueError(f"expected {n} values, found {len(out)}")
    return out[:n], i


def _array(lines: list[str], i: int, n: int, *, mf6: bool) -> tuple[np.ndarray, int]:
    """One array from its control line at ``lines[i]``: CONSTANT, INTERNAL or (2005) a fixed LOCAT record."""
    toks = _strip_comment(lines[i]).split()
    if not toks:
        raise ValueError("an empty array control line")
    word = toks[0].upper()
    if word == "CONSTANT":
        return np.full(n, float(toks[1].replace("D", "E"))), i + 1
    if word == "INTERNAL":
        factor = 1.0
        if mf6:
            up = [t.upper() for t in toks]
            if "FACTOR" in up:
                factor = float(toks[up.index("FACTOR") + 1])
        elif len(toks) > 1:
            m = re.match(r"^[-+0-9.eEdD]+", toks[1])
            factor = float(m.group(0).replace("D", "E")) if m else 1.0
        vals, j = _numbers(lines, i + 1, n)
        return np.asarray(vals) * (factor or 1.0), j
    if word in ("OPEN/CLOSE", "EXTERNAL"):
        raise ValueError(f"the grid file reads an array from another file ({word}); drop the MODFLOW 6 binary grid "
                         "file (.dis.grb) instead, or type the cell size in")
    m = re.match(r"^\s*(-?\d+)\s+([-+0-9.eEdD]+)", lines[i])
    if not m:
        raise ValueError(f"cannot read the array control line: {lines[i].strip()!r}")
    locat, cnst = int(m.group(1)), float(m.group(2).replace("D", "E"))
    if locat == 0:
        return np.full(n, cnst), i + 1
    vals, j = _numbers(lines, i + 1, n)
    return np.asarray(vals) * (cnst or 1.0), j


def _blocks(text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        toks = line.split()
        if toks[0].upper() == "BEGIN" and len(toks) > 1:
            cur = toks[1].upper()
            blocks.setdefault(cur, [])
            continue
        if toks[0].upper() == "END":
            cur = None
            continue
        if cur:
            blocks[cur].append(line)
    return blocks


def _options(lines: list[str]) -> dict[str, str]:
    out = {}
    for line in lines:
        toks = line.split(None, 1)
        out[toks[0].upper()] = toks[1].strip() if len(toks) > 1 else ""
    return out


def _dims(lines: list[str]) -> dict[str, int]:
    out = {}
    for line in lines:
        toks = line.split()
        if len(toks) >= 2:
            out[toks[0].upper()] = int(toks[1])
    return out


def _georef(opts: dict[str, str]) -> dict[str, Any]:
    num = lambda k: float(opts[k].split()[0].replace("D", "E")) if k in opts and opts[k] else None  # noqa: E731
    crs = opts.get("CRS") or None
    return {"xorigin": num("XORIGIN"), "yorigin": num("YORIGIN"), "angrot": num("ANGROT"),
            "crs": crs.strip().strip("'\"") if crs else None,
            "length_units": (opts.get("LENGTH_UNITS") or "").lower() or None}


def _mf6_dis(text: str) -> dict[str, Any]:
    b = _blocks(text)
    d = _dims(b.get("DIMENSIONS", []))
    if "NCPL" in d:
        return _mf6_disv(b, d)
    if not {"NROW", "NCOL"} <= set(d):
        raise ValueError("a MODFLOW 6 DIS file needs NROW and NCOL in its DIMENSIONS block")
    nrow, ncol = d["NROW"], d["NCOL"]
    lines = b.get("GRIDDATA", [])
    got: dict[str, np.ndarray] = {}
    i = 0
    while i < len(lines):
        key = lines[i].split()[0].upper()
        if key in ("DELR", "DELC") and len(lines[i].split()) == 1:
            got[key], i = _array(lines, i + 1, ncol if key == "DELR" else nrow, mf6=True)
        else:
            i += 1
    if "DELR" not in got or "DELC" not in got:
        raise ValueError("the DIS file has no DELR or DELC array")
    return {"type": "dis", "nlay": d.get("NLAY", 1), "nrow": nrow, "ncol": ncol, "delr": got["DELR"],
            "delc": got["DELC"], **_georef(_options(b.get("OPTIONS", []))), "from": "MODFLOW 6 DIS file"}


def _mf6_disv(b: dict[str, list[str]], d: dict[str, int]) -> dict[str, Any]:
    ncpl, nvert = d["NCPL"], d.get("NVERT", 0)
    verts = np.zeros((nvert, 2))
    for line in b.get("VERTICES", []):
        t = line.split()
        verts[int(t[0]) - 1] = (float(t[1]), float(t[2]))
    cells: list[list[int]] = [[] for _ in range(ncpl)]
    for line in b.get("CELL2D", []):
        t = line.split()
        k = int(t[3])
        cells[int(t[0]) - 1] = [int(v) - 1 for v in t[4:4 + k]]
    if any(not c for c in cells):
        raise ValueError("the DISV file's CELL2D block does not list every cell")
    offsets = np.cumsum([0] + [len(c) for c in cells])
    return {"type": "disv", "nlay": d.get("NLAY", 1), "ncpl": ncpl, "vertices": verts,
            "cell_vertices": np.array([v for c in cells for v in c], dtype=np.int64), "cell_offsets": offsets,
            **_georef(_options(b.get("OPTIONS", []))), "from": "MODFLOW 6 DISV file"}


def _flopy_comment_georef(text: str) -> dict[str, Any]:
    """The georeference FloPy writes into a MODFLOW-2005 file's first comment (xul/yul or xll/yll, rotation,
    proj4_str or epsg, start_datetime), when it is there."""
    found: dict[str, str] = {}
    for line in text.splitlines()[:20]:
        if not line.lstrip().startswith("#"):
            break
        for part in line.lstrip("# ").split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                found[k.strip().lower()] = v.strip()
    return found


ITMUNI = {1: "seconds", 2: "minutes", 3: "hours", 4: "days", 5: "years"}


def _apply_flopy_georef(g: dict[str, Any], head: dict[str, str]) -> dict[str, Any]:
    """Fill a grid's origin, rotation, CRS and start date from a FloPy comment (only what the grid lacks)."""
    num = lambda k: float(head[k]) if head.get(k) not in (None, "", "None") else None  # noqa: E731
    try:
        rot = num("rotation") or 0.0
        if g.get("xorigin") is None or g.get("yorigin") is None:
            if num("xll") is not None and num("yll") is not None:
                g.update(xorigin=num("xll"), yorigin=num("yll"), angrot=rot)
            elif num("xul") is not None and num("yul") is not None and g.get("type") == "dis":
                height = float(np.sum(g["delc"]))
                a = math.radians(rot)
                g.update(xorigin=num("xul") + height * math.sin(a), yorigin=num("yul") - height * math.cos(a),
                         angrot=rot)
    except ValueError:
        pass
    if not g.get("crs"):
        if head.get("epsg") not in (None, "", "None"):
            g["crs"] = f"EPSG:{head['epsg']}"
        elif head.get("proj4_str") not in (None, "", "None"):
            g["crs"] = head["proj4_str"]
    if head.get("start_datetime") and not g.get("start_datetime"):
        g["start_datetime"] = head["start_datetime"]
    return g


def read_sidecar(data: Any) -> dict[str, str]:
    """What a name file (.nam, with FloPy's georeference comment) or a MODFLOW 6 TDIS file says about where and when:
    xul/yul or xll/yll, rotation, proj4_str or epsg, start_datetime, time_units."""
    text = _as_bytes(data).decode("utf-8", "replace")
    out = dict(_flopy_comment_georef(text))
    if re.search(r"^\s*BEGIN\s+OPTIONS", text, re.IGNORECASE | re.MULTILINE):
        opts = _options(_blocks(text).get("OPTIONS", []))
        if opts.get("TIME_UNITS"):
            out["time_units"] = opts["TIME_UNITS"].lower()
        if opts.get("START_DATE_TIME"):
            out["start_datetime"] = opts["START_DATE_TIME"]
    return out


def _mf2005_dis(text: str) -> dict[str, Any]:
    head = _flopy_comment_georef(text)
    lines = [ln for ln in text.splitlines() if _strip_comment(ln)]
    if not lines:
        raise ValueError("an empty DIS file")
    t = _strip_comment(lines[0]).split()
    nlay, nrow, ncol = int(t[0]), int(t[1]), int(t[2])
    itmuni = int(t[4]) if len(t) > 4 else 0
    lenuni = int(t[5]) if len(t) > 5 else 0
    _, i = _numbers(lines, 1, nlay)  # line 2 is LAYCBD
    delr, i = _array(lines, i, ncol, mf6=False)
    delc, i = _array(lines, i, nrow, mf6=False)
    g: dict[str, Any] = {"type": "dis", "nlay": nlay, "nrow": nrow, "ncol": ncol, "delr": delr, "delc": delc,
                         "xorigin": None, "yorigin": None, "angrot": None, "crs": None,
                         "length_units": LENUNI.get(lenuni), "time_units": ITMUNI.get(itmuni),
                         "from": "MODFLOW-2005 DIS file"}
    _apply_flopy_georef(g, head)
    return g


def _grb(buf: bytes) -> dict[str, Any]:
    """A MODFLOW 6 binary grid file (.dis.grb or .disv.grb)."""
    def line(k: int, size: int, at: int) -> str:
        return buf[at + k * size: at + (k + 1) * size].decode("ascii", "replace").strip()

    kind = line(0, 50, 0).split()[1].upper()
    ntxt = int(line(2, 50, 0).split()[1])
    lentxt = int(line(3, 50, 0).split()[1])
    pos = 200
    defs = []
    for k in range(ntxt):
        txt = line(k, lentxt, pos).split("#")[0].split()
        name, typ, nd = txt[0].upper(), txt[1].upper(), int(txt[3])
        dims = [int(x) for x in txt[4:4 + nd]]
        defs.append((name, typ, dims))
    pos += ntxt * lentxt
    vals: dict[str, Any] = {}
    for name, typ, dims in defs:
        count = int(np.prod(dims)) if dims else 1
        if typ == "INTEGER":
            arr = np.frombuffer(buf, "<i4", count, pos)
            pos += 4 * count
        elif typ == "DOUBLE":
            arr = np.frombuffer(buf, "<f8", count, pos)
            pos += 8 * count
        elif typ == "CHARACTER":
            vals[name] = buf[pos:pos + count].decode("ascii", "replace").strip()
            pos += count
            continue
        else:
            raise ValueError(f"unknown type {typ} in the binary grid file")
        vals[name] = arr[0] if not dims else (arr.reshape(dims[::-1]) if len(dims) > 1 else arr)
    geo = {"xorigin": float(vals.get("XORIGIN", 0.0)), "yorigin": float(vals.get("YORIGIN", 0.0)),
           "angrot": float(vals.get("ANGROT", 0.0)), "crs": vals.get("CRS") or None, "length_units": None}
    if kind == "DIS":
        return {"type": "dis", "nlay": int(vals["NLAY"]), "nrow": int(vals["NROW"]), "ncol": int(vals["NCOL"]),
                "delr": np.asarray(vals["DELR"], float), "delc": np.asarray(vals["DELC"], float), **geo,
                "from": "MODFLOW 6 binary grid file"}
    if kind == "DISV":
        verts = np.asarray(vals["VERTICES"], float).reshape(-1, 2)
        ia = np.asarray(vals["IAVERT"], np.int64) - 1
        ja = np.asarray(vals["JAVERT"], np.int64) - 1
        cells = [ja[ia[k]:ia[k + 1]] for k in range(len(ia) - 1)]
        cells = [c[:-1] if len(c) > 1 and c[0] == c[-1] else c for c in cells]  # MODFLOW closes each ring
        offsets = np.cumsum([0] + [len(c) for c in cells])
        return {"type": "disv", "nlay": int(vals["NLAY"]), "ncpl": int(vals["NCPL"]), "vertices": verts,
                "cell_vertices": np.concatenate(cells) if cells else np.zeros(0, np.int64), "cell_offsets": offsets,
                **geo, "from": "MODFLOW 6 binary grid file"}
    raise ValueError(f"a {kind} binary grid file is not supported (DIS and DISV are)")


def read_grid(data: Any) -> dict[str, Any]:
    """Read a MODFLOW grid: a MODFLOW 6 ``.dis`` or ``.disv``, a MODFLOW-2005 ``.dis``, or a MODFLOW 6 binary grid
    file. Returns the grid's type, sizes, cell widths (DIS) or vertices (DISV), and its origin, rotation, CRS and
    length unit where the file says them (None where it does not)."""
    buf = _as_bytes(data)
    if buf[:5] == b"GRID ":
        return _grb(buf)
    text = buf.decode("utf-8", "replace")
    if re.search(r"^\s*BEGIN\s+", text, re.IGNORECASE | re.MULTILINE):
        return _mf6_dis(text)
    return _mf2005_dis(text)


def grid_from_cell_size(nrow: int, ncol: int, cell_size: Any) -> dict[str, Any]:
    """A regular DIS grid from the head file's size and a cell size (one number, or [width, height])."""
    if isinstance(cell_size, (list, tuple)):
        dx, dy = float(cell_size[0]), float(cell_size[1] if len(cell_size) > 1 else cell_size[0])
    else:
        dx = dy = float(cell_size)
    if not (dx > 0 and dy > 0):
        raise ValueError("the cell size must be positive")
    return {"type": "dis", "nlay": 1, "nrow": nrow, "ncol": ncol, "delr": np.full(ncol, dx), "delc": np.full(nrow, dy),
            "xorigin": None, "yorigin": None, "angrot": None, "crs": None, "length_units": None,
            "from": "the cell size you gave"}


# ── MODFLOW: placing the grid on the map ────────────────────────────────────


def _crs_is_lonlat(crs: str | None) -> bool:
    return str(crs or "").strip().upper().replace(" ", "") in ("EPSG:4326", "4326", "WGS84", "OGC:CRS84", "CRS84")


def _normal_crs(crs: Any) -> str | None:
    if crs is None or str(crs).strip() == "":
        return None
    s = str(crs).strip()
    return f"EPSG:{s}" if s.isdigit() else s


def _to_lonlat(x: np.ndarray, y: np.ndarray, crs: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Project to lon/lat. Returns (lon, lat, metres per CRS unit)."""
    if _crs_is_lonlat(crs):
        return x, y, float("nan")
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:  # pragma: no cover - pyproj is in aquascope[spatial] and the test install
        raise ImportError("placing a projected grid needs pyproj (pip install pyproj)") from exc
    src = CRS.from_user_input(crs)
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    unit = 1.0
    try:
        unit = float(src.axis_info[0].unit_conversion_factor or 1.0)
    except (IndexError, AttributeError, TypeError):
        pass
    return np.asarray(lon, float), np.asarray(lat, float), unit


def _place(local_x: np.ndarray, local_y: np.ndarray, origin: tuple[float, float], angrot: float, scale: float,
           crs: str) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(angrot or 0.0)
    x = origin[0] + scale * (local_x * math.cos(a) - local_y * math.sin(a))
    y = origin[1] + scale * (local_x * math.sin(a) + local_y * math.cos(a))
    lon, lat, _ = _to_lonlat(x, y, crs)
    return lon, lat


#: The Earth's mean radius (IUGG), for placing a model without a georeference near a point.
EARTH_RADIUS_M = 6_371_008.8


def _place_near(local_x: np.ndarray, local_y: np.ndarray, centre: tuple[float, float], scale_m: float,
                middle: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """An approximate placement: the model's middle at ``centre`` (lon, lat), north up, lengths true to the model
    (``scale_m`` metres per model unit), on a local equirectangular plane. Good for looking at a model, not for
    overlaying it on anything: only its size and shape are the model's own."""
    lon0, lat0 = float(centre[0]), float(centre[1])
    x = (np.asarray(local_x, float) - middle[0]) * scale_m
    y = (np.asarray(local_y, float) - middle[1]) * scale_m
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * max(math.cos(math.radians(lat0)), 0.01)))
    return lon, lat


def _crs_unit_m(crs: str) -> float | None:
    if _crs_is_lonlat(crs):
        return None
    try:
        from pyproj import CRS

        return float(CRS.from_user_input(crs).axis_info[0].unit_conversion_factor or 1.0)
    except Exception:  # noqa: BLE001 - an unknown CRS fails later with pyproj's own message
        return None


def _block_size(nrow: int, ncol: int, max_cells: int) -> int:
    return max(1, math.ceil(math.sqrt(nrow * ncol / max(1, max_cells))))


def _block_heads(heads: np.ndarray, nrow: int, ncol: int, k: int) -> np.ndarray:
    """Mean of k x k blocks of cells (NaN-aware), heads shaped (time, layer, nrow * ncol)."""
    nt, nl = heads.shape[:2]
    br, bc = math.ceil(nrow / k), math.ceil(ncol / k)
    pad = np.full((nt, nl, br * k, bc * k), np.nan)
    pad[:, :, :nrow, :ncol] = heads.reshape(nt, nl, nrow, ncol)
    blocks = pad.reshape(nt, nl, br, k, bc, k)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(blocks).sum(axis=(3, 5))
        tot = np.nansum(blocks, axis=(3, 5))
        out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out.reshape(nt, nl, br * bc)


TIME_UNITS_DAYS = {"days": 1.0, "day": 1.0, "d": 1.0, "hours": 1 / 24, "h": 1 / 24, "minutes": 1 / 1440,
                   "seconds": 1 / 86400, "s": 1 / 86400, "years": 365.25, "y": 365.25, "unknown": 1.0}


def read_modflow_heads(
    hds: Any,
    grid: Any = None,
    *,
    origin: Any = None,
    rotation: float | None = None,
    crs: Any = None,
    cell_size: Any = None,
    length_units: str | None = None,
    start_date: Any = None,
    time_units: str | None = None,
    sidecars: Any = None,
    max_cells: int = MAX_CELLS,
    approximate_at: Any = None,
) -> dict[str, Any]:
    """Your MODFLOW heads on the map: read a binary head file and place its cells in lon/lat.

    ``hds``: the head (or drawdown) file, as bytes or a path. ``grid``: a ``.dis``, ``.disv`` or ``.dis.grb``
    (bytes or a path), or None with ``cell_size`` for a regular grid. ``origin`` [x, y] (the grid's lower-left corner,
    in the CRS), ``rotation`` (degrees counter-clockwise) and ``crs`` (EPSG code or any pyproj string) override what
    the grid file says; a grid with no CRS needs one. ``start_date`` (YYYY-MM-DD) turns model times (in
    ``time_units``, default the grid's or days) into dates. ``sidecars``: other files of the model that say where or
    when it is (a ``.nam`` with FloPy's georeference comment, a MODFLOW 6 ``.tdis``), as bytes or paths.
    ``approximate_at`` [lon, lat]: for a model whose files do not say where it is, draw it there instead, its middle
    on that point, north up, at its true size in its own length unit (metres when the files do not name one). Its
    place and orientation are then not the model's, and the result says so (``grid.approximate``).

    Returns a dict: the grid (type, sizes, how it was placed), ``times``, ``dates``, ``layers``, ``kstpkper``, the
    value ``range``, per-layer statistics at the last time, notes, and two arrays for drawing: ``rings`` (every cell's
    corners in lon/lat, flat, with ``ring_offsets``) and ``heads`` (time x layer x cell, NaN for no head). On a grid
    bigger than ``max_cells`` cells per layer a DIS grid is drawn in k x k blocks (``block``) and a DISV grid is
    refused. Use :func:`modflow_frame` for one time and layer, :func:`modflow_geojson` for a GeoJSON."""
    h = read_heads(hds)
    notes: list[str] = []
    if grid is not None:
        g = read_grid(grid)
    elif cell_size is not None:
        g = grid_from_cell_size(h["nrow"], h["ncol"], cell_size)
    else:
        return {"kind": "modflow_heads", "error": "Where is this model? Add its grid file (.dis, .disv or .dis.grb) "
                "or type a cell size, origin and CRS.", "needs": ["grid", "origin", "crs"],
                "nrow": h["nrow"], "ncol": h["ncol"], "layers": h["layers"], "n_times": len(h["times"])}
    if g["type"] == "dis":
        if (g["nrow"], g["ncol"]) != (h["nrow"], h["ncol"]):
            raise ValueError(f"the head file is {h['nrow']} rows x {h['ncol']} columns but the grid is "
                             f"{g['nrow']} x {g['ncol']}: are they from the same model?")
        ncell = g["nrow"] * g["ncol"]
    else:
        if h["nrow"] * h["ncol"] != g["ncpl"]:
            raise ValueError(f"the head file has {h['nrow'] * h['ncol']} cells per layer but the grid has "
                             f"{g['ncpl']}: are they from the same model?")
        ncell = g["ncpl"]
    side: dict[str, str] = {}
    for item in sidecars or []:
        side.update({k: v for k, v in read_sidecar(item).items() if v})
    placed_by = "the grid file"
    if side:
        if g.get("xorigin") == 0 and g.get("yorigin") == 0 and (side.get("xul") or side.get("xll")):
            g["xorigin"] = g["yorigin"] = None  # a binary grid file's 0, 0 means "not set"
        before = (g.get("xorigin"), g.get("crs"))
        _apply_flopy_georef(g, side)
        if (g.get("xorigin"), g.get("crs")) != before:
            placed_by = "the model's files"
        if side.get("time_units") and not g.get("time_units"):
            g["time_units"] = side["time_units"]
    xo = g.get("xorigin")
    yo = g.get("yorigin")
    if origin is not None:
        xo, yo = float(origin[0]), float(origin[1])
        placed_by = "you"
    rot = float(rotation) if rotation is not None else float(g.get("angrot") or 0.0)
    the_crs = _normal_crs(crs) or _normal_crs(g.get("crs"))
    if crs is not None:
        placed_by = "you"
    approx = approximate_at is not None
    if approx:
        try:
            centre = (float(approximate_at[0]), float(approximate_at[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("approximate_at is a point: [lon, lat]") from exc
        if not (-180 <= centre[0] <= 180 and -85 <= centre[1] <= 85):
            raise ValueError("approximate_at must be a lon, lat on the map (latitude within 85 degrees)")
        placed_by = "approximation, at the map's centre"
        rot = 0.0
        the_crs = None
    elif the_crs is None or xo is None or yo is None:
        return {"kind": "modflow_heads", "error": "This grid does not say where it is on the Earth. Give the "
                "lower-left corner (origin), the rotation and the CRS (an EPSG code).",
                "needs": [k for k, missing in (("origin", xo is None or yo is None), ("crs", the_crs is None)) if missing],
                "grid_from": g["from"], "nrow": h["nrow"], "ncol": h["ncol"], "layers": h["layers"],
                "n_times": len(h["times"]), "xorigin": g.get("xorigin"), "yorigin": g.get("yorigin"),
                "angrot": g.get("angrot"), "length_units": g.get("length_units")}
    if xo == 0 and yo == 0 and origin is None and not approx:
        notes.append("The grid file puts the model at 0, 0 in its CRS, which is rarely where a model is: "
                     "check the origin.")
    units = (length_units or g.get("length_units") or "").lower() or None
    crs_m = None if approx else _crs_unit_m(the_crs)
    scale = 1.0
    if approx:
        scale = LENGTH_UNITS_M.get(units or "", 1.0)
        notes.append("Placed approximately: the model's files do not say where it is, so its middle is put at the "
                     "centre of the map, north up. Its size and shape are the model's"
                     f"{f' (in {units})' if units in LENGTH_UNITS_M else ''}; its place and orientation are not.")
        if units not in LENGTH_UNITS_M:
            notes.append("The grid does not name its length unit; read as metres.")
    elif units in LENGTH_UNITS_M and crs_m:
        scale = LENGTH_UNITS_M[units] / crs_m
        if abs(scale - 1) > 1e-9:
            notes.append(f"Model lengths in {units} scaled to the CRS's units.")
    elif not units and not _crs_is_lonlat(the_crs):
        notes.append("The grid does not name its length unit; read as the CRS's unit.")
    heads = h["heads"]
    block = 1
    if g["type"] == "dis":
        nrow, ncol = g["nrow"], g["ncol"]
        delr, delc = np.asarray(g["delr"], float), np.asarray(g["delc"], float)
        if ncell > max_cells:
            block = _block_size(nrow, ncol, max_cells)
            heads = _block_heads(heads, nrow, ncol, block)
            delr = np.add.reduceat(delr, np.arange(0, ncol, block))
            delc = np.add.reduceat(delc, np.arange(0, nrow, block))
            nrow, ncol = len(delc), len(delr)
            notes.append(f"{ncell:,} cells per layer is more than the map draws smoothly, so it shows blocks of "
                         f"{block} x {block} cells (their mean head).")
        xe = np.concatenate([[0.0], np.cumsum(delr)])
        ye = float(np.sum(delc)) - np.concatenate([[0.0], np.cumsum(delc)])
        gx, gy = np.meshgrid(xe, ye)
        if approx:
            lon, lat = _place_near(gx.ravel(), gy.ravel(), centre, scale, (float(xe[-1]) / 2, float(ye[0]) / 2))
        else:
            lon, lat = _place(gx.ravel(), gy.ravel(), (xo, yo), rot, scale, the_crs)
        lon, lat = lon.reshape(nrow + 1, ncol + 1), lat.reshape(nrow + 1, ncol + 1)
        ii, jj = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        corners = [(ii, jj), (ii, jj + 1), (ii + 1, jj + 1), (ii + 1, jj)]
        rings = np.stack([np.stack([lon[a, b], lat[a, b]], axis=-1) for a, b in corners], axis=1).reshape(-1, 2)
        offsets = np.arange(0, 4 * nrow * ncol + 1, 4)
        shape = {"nrow": g["nrow"], "ncol": g["ncol"], "drawn_nrow": nrow, "drawn_ncol": ncol}
    else:
        if ncell > max_cells:
            raise ValueError(f"{ncell:,} cells per layer is more than the map can draw (up to {max_cells:,})")
        v = np.asarray(g["vertices"], float)
        if approx:
            middle = ((float(v[:, 0].min()) + float(v[:, 0].max())) / 2, (float(v[:, 1].min()) + float(v[:, 1].max())) / 2)
            lon, lat = _place_near(v[:, 0], v[:, 1], centre, scale, middle)
        else:
            lon, lat = _place(v[:, 0], v[:, 1], (xo, yo), rot, scale, the_crs)
        cv = np.asarray(g["cell_vertices"], np.int64)
        rings = np.stack([lon[cv], lat[cv]], axis=-1)
        offsets = np.asarray(g["cell_offsets"], np.int64)
        shape = {"ncpl": g["ncpl"]}
    if not (np.all(np.abs(rings[:, 1]) <= 90) and np.all(np.abs(rings[:, 0]) <= 360)):
        raise ValueError("the placed grid falls outside the Earth: check the origin and the CRS")
    finite = heads[np.isfinite(heads)]
    if finite.size > 2_000_000:
        finite = finite[:: finite.size // 2_000_000 + 1]
    rng = None
    if finite.size:
        lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, hi + 0.5
        rng = [round(lo, 4), round(hi, 4)]
    if not (time_units or g.get("time_units")):
        notes.append("The files do not name the time unit; model times read as days.")
    time_units = (time_units or g.get("time_units") or "days").lower()
    days_per = TIME_UNITS_DAYS.get(time_units)
    if days_per is None:
        raise ValueError(f"time_units must be one of {', '.join(sorted(set(TIME_UNITS_DAYS)))}")
    start = start_date or (g.get("start_datetime") if g.get("start_datetime") else None)
    dates = None
    if start:
        try:
            d0 = _parse_start(start)
            stamps = [d0 + timedelta(days=t * days_per) for t in h["times"]]
            dates = [d.date().isoformat() if (d.hour, d.minute, d.second) == (0, 0, 0)
                     else d.isoformat(timespec="minutes") for d in stamps]
        except ValueError:
            notes.append(f"Could not read the start date {start!r}; times are model times.")
    last = heads[-1]
    stats = []
    for li, lay in enumerate(h["layers"]):
        v = last[li][np.isfinite(last[li])]
        stats.append({"layer": lay, "cells_with_head": int(v.size),
                      "min": round(float(v.min()), 4) if v.size else None,
                      "mean": round(float(v.mean()), 4) if v.size else None,
                      "max": round(float(v.max()), 4) if v.size else None})
    lon_all, lat_all = rings[:, 0], rings[:, 1]
    return {
        "kind": "modflow_heads", "text": h["text"], "precision": h["precision"],
        "grid": {"type": g["type"], "from": g["from"], "nlay": g.get("nlay"), **shape, "cells": ncell,
                 "crs": the_crs, "origin": None if approx else [xo, yo], "rotation": rot, "length_units": units,
                 "placed_by": placed_by, "block": block, "approximate": approx},
        "layers": h["layers"], "times": h["times"], "kstpkper": h["kstpkper"], "dates": dates,
        "time_units": time_units, "range": rng, "last_time_stats": stats,
        "bbox": [round(float(lon_all.min()), 6), round(float(lat_all.min()), 6), round(float(lon_all.max()), 6),
                 round(float(lat_all.max()), 6)],
        "notes": notes,
        "summary": (f"{h['text']} from {len(h['times'])} time step(s) on {len(h['layers'])} layer(s), "
                    f"{ncell:,} cells per layer, "
                    f"{'placed approximately at the map centre' if approx else f'placed by {placed_by} ({the_crs})'}."),
        "rings": rings, "ring_offsets": offsets, "heads": heads,
    }


def _parse_start(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:19] if "T" in s or " " in s else s, fmt)
        except ValueError:
            continue
    raise ValueError(f"not a date: {value!r}")


def modflow_frame(res: dict[str, Any], time_index: int = -1, layer: int | None = None) -> np.ndarray:
    """One time and layer's heads (cell order of ``rings``). ``layer`` is 1-based (default the first)."""
    heads = res["heads"]
    t = int(time_index)
    if not -len(res["times"]) <= t < len(res["times"]):
        raise ValueError(f"time_index out of range (0 to {len(res['times']) - 1})")
    lay = res["layers"][0] if layer is None else int(layer)
    if lay not in res["layers"]:
        raise ValueError(f"layer {lay} is not in the file (layers {res['layers']})")
    return heads[t, res["layers"].index(lay)]


def modflow_summary(res: dict[str, Any], time_index: int = -1, layer: int | None = None) -> dict[str, Any]:
    """The result without its arrays, plus one frame's statistics: what an MCP answer or a terminal shows."""
    out = {k: v for k, v in res.items() if k not in ("rings", "ring_offsets", "heads")}
    if res.get("error"):
        return out
    f = modflow_frame(res, time_index, layer)
    v = f[np.isfinite(f)]
    t = int(time_index) % len(res["times"])
    out["frame"] = {"time_index": t, "time": res["times"][t], "date": (res.get("dates") or [None] * (t + 1))[t],
                    "layer": res["layers"][0] if layer is None else int(layer), "cells_with_head": int(v.size),
                    "min": round(float(v.min()), 4) if v.size else None,
                    "max": round(float(v.max()), 4) if v.size else None}
    if len(out["times"]) > 40:
        out["times"] = out["times"][:20] + ["..."] + out["times"][-20:]
        out["kstpkper"] = out["kstpkper"][:20] + ["..."] + out["kstpkper"][-20:]
        if out.get("dates"):
            out["dates"] = out["dates"][:20] + ["..."] + out["dates"][-20:]
    return out


def modflow_geojson(res: dict[str, Any], time_index: int = -1, layer: int | None = None) -> dict[str, Any]:
    """One time and layer as a GeoJSON FeatureCollection of cell polygons with their head (cells with no head
    are left out)."""
    f = modflow_frame(res, time_index, layer)
    rings, off = res["rings"], res["ring_offsets"]
    feats = []
    for c in range(len(off) - 1):
        if not math.isfinite(f[c]):
            continue
        ring = [[round(float(x), 6), round(float(y), 6)] for x, y in rings[off[c]:off[c + 1]]]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                      "properties": {"cell": c + 1, "head": round(float(f[c]), 4)}})
    t = int(time_index) % len(res["times"])
    return {"type": "FeatureCollection", "features": feats,
            "properties": {"time": res["times"][t], "date": (res.get("dates") or [None] * (t + 1))[t],
                           "layer": res["layers"][0] if layer is None else int(layer), "crs_in": res["grid"]["crs"],
                           "text": res["text"]}}


__all__ = [
    "MIN_WELLS", "RADIUS_KM", "SOURCES", "contour_lines", "groundwater_surface", "interpolate", "modflow_frame",
    "modflow_geojson", "modflow_summary", "month_values", "pick_month", "read_grid", "read_heads",
    "read_modflow_heads", "read_sidecar", "surface_from_readings", "surface_geojson", "surface_summary",
]
