"""What a river passes: the dams on its way to the sea, the dams upstream of a reach, the countries it crosses.

The companions of :func:`aquascope.rivers.trace_downstream` (#516), kept apart so ``rivers`` stays about the
network itself:

* :func:`dams_along` lists the Global Dam Watch dams within a small buffer of a traced path, in the order the
  water reaches them, with name, storage capacity, main use and the degree of regulation where GDW gives it.
* :func:`upstream_dams` finds the dams upstream of a reach: the candidates in the basin's bounding box from the
  same mirror, each matched to its river reach in the stream tiles and kept when that reach drains to this one.
  With ``with_flow`` it adds the degree of regulation at the reach: the storage upstream as a share of a year's
  mean flow (GEOGLOWS v2 annual averages, modelled).
* :func:`countries_along` names the countries a path crosses, from Natural Earth's 1:50m admin-0 boundaries.

Data and licences. The dams come from the Archive's Global Dam Watch v1.0 mirror (CC BY 4.0), cell-sorted in
2-degree cells and read with :mod:`aquascope.context`'s readers; until that mirror is published every function
says so and the rest of the answer stands. The boundaries are Natural Earth (public domain), read from the
``world-atlas`` 2.0.2 TopoJSON on jsDelivr (CORS open, checked 2026-10-08). They show borders as Natural Earth
draws them, which is not a position on any dispute. Everything here is plain Python, so the Explorer's Pyodide
worker runs it unchanged.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from aquascope import rivers

logger = logging.getLogger(__name__)

__all__ = [
    "COUNTRIES_URL",
    "countries_along",
    "dams_along",
    "parse_countries_topojson",
    "path_cells",
    "upstream_dams",
]

#: Natural Earth 1:50m admin-0 countries as TopoJSON (world-atlas 2.0.2, ISC; the data is public domain).
COUNTRIES_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json"
#: The zoom of the stream tiles the upstream dams are matched at: every reach is there, about 40 m a unit.
MATCH_ZOOM = 8
#: How far a dam may sit from the line of its reach in the stream tiles and still be matched to it.
MATCH_M = 1500.0
#: A dam nearest the first point of the path and further off it than this (km) lies behind the start.
START_KM = 0.05
#: Million m3 a year per m3/s of mean flow (365.25 days).
MCM_PER_CMS_YEAR = 365.25 * 86_400 / 1e6

DAMS_METHOD = ("Global Dam Watch v1.0 dams within the buffer of the drawn path, placed along it by their nearest "
               "point on the line. A dam whose own catchment (GDW) is under half the area draining to the start "
               "of the path sits on a side stream, and one behind the start is upstream of it: both are left out.")
UPSTREAM_METHOD = ("Global Dam Watch v1.0 dams in the basin's bounding box, each matched to the nearest river reach "
                   f"in the GEOGLOWS v2 stream tiles (within {MATCH_M:,.0f} m), kept when that reach drains to this "
                   "one through the GEOGLOWS routing table. Degree of regulation: the storage upstream as a share of "
                   "the river's mean annual flow here (GEOGLOWS v2 annual averages, 1940 onward, modelled).")
UPSTREAM_NOTE = ("Approximate: a dam at a confluence can be matched to the wrong branch, and a dam on the same reach "
                 "as the site may sit just below it. Small farm dams are not in Global Dam Watch.")


def _dams_meta() -> dict[str, str]:
    from aquascope.registry import CONTEXT_LAYERS

    m = CONTEXT_LAYERS["dams"]
    return {"label": m.label, "licence": m.license, "attribution": m.attribution, "homepage": m.homepage}


def _missing_note() -> str:
    from aquascope.context._common import MIRROR_MISSING

    return str(MIRROR_MISSING.format(what="Global Dam Watch"))


# ── seams (tests replace these) ──────────────────────────────────────────────


def _dam_rows(keys: list[str]) -> list[dict[str, str]] | None:
    """The mirrored GDW rows in these 2-degree cells, or None when the mirror is not published."""
    from aquascope.context._common import read_cells

    rows: list[dict[str, str]] | None = read_cells("dams", keys)
    return rows


_COUNTRIES: dict[str, list[_Country]] = {}


def _countries() -> list[_Country]:
    """The admin-0 countries, read once per process (about 750 kB, cached a month on disk in CPython)."""
    if COUNTRIES_URL not in _COUNTRIES:
        _COUNTRIES[COUNTRIES_URL] = parse_countries_topojson(json.loads(rivers._fetch_text(COUNTRIES_URL)))
    return _COUNTRIES[COUNTRIES_URL]


# ── along a path ─────────────────────────────────────────────────────────────


def path_cells(coords: list[list[float]], pad_km: float) -> list[str]:
    """The 2-degree mirror cells a path and a ``pad_km`` buffer around it touch, in the order the path meets them."""
    from aquascope.context._common import cell_key

    keys: list[str] = []
    seen: set[str] = set()
    pad_lat = max(0.0, float(pad_km)) / 111.0
    pts: list[tuple[float, float]] = []
    for a, b in zip(coords, coords[1:] or coords):
        pts.append((a[0], a[1]))
        pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    if coords:
        pts.append((coords[-1][0], coords[-1][1]))
    for lon, lat in pts:
        pad_lon = pad_lat / max(0.05, math.cos(math.radians(lat)))
        for dy in (-pad_lat, 0.0, pad_lat):
            for dx in (-pad_lon, 0.0, pad_lon):
                y = max(-89.999, min(89.999, lat + dy))
                x = ((lon + dx + 180.0) % 360.0) - 180.0
                key = cell_key(y, x)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def _dam_record(r: dict[str, str]) -> dict[str, Any]:
    from aquascope.context._common import num
    from aquascope.context.events import _dam

    d: dict[str, Any] = _dam(r)
    d["purpose"] = d.pop("main_use", None)
    d["catchment_km2"] = num(r.get("catchment_km2"))
    d["dor_source"] = "Global Dam Watch" if d.get("dor_pc") is not None else None
    return d


def _label(d: dict[str, Any]) -> str:
    return str(d["name"]) if d.get("name") and d["name"] != "unnamed" else "an unnamed dam"


def _unplaced(d: dict[str, Any]) -> bool:
    """No position, or 0, 0: a mirror built from GDW's LAT_DAM/LONG_DAM (before it read LAT_RIV/LONG_RIV) parks
    some 35,000 barriers there, and no dam stands at 0, 0."""
    return d["lat"] is None or d["lon"] is None or (d["lat"] == 0 and d["lon"] == 0)


def dams_along(coords: list[list[float]], *, reach_of_segment: list[int] | None = None,
               dam_km: float = 2.0, min_catchment_km2: float | None = None, max_cells: int = 120,
               rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """The Global Dam Watch dams within ``dam_km`` of a path, in the order the water reaches them.

    ``coords`` is the path as ``[lon, lat]`` from upstream to the outlet. Each dam carries its name, reservoir,
    river, country, year, height, storage capacity (million m3), purpose (GDW's main use), the degree of
    regulation where GDW gives it (``dor_pc``, its storage as a share of the mean annual flow at the dam), how
    far off the path it is (``distance_km``) and how far along (``along_km``). ``reach_of_segment`` maps each
    segment to a river reach id, so each dam names the reach it sits by. ``min_catchment_km2`` leaves out dams
    whose own catchment is smaller (they sit on a side stream). ``rows`` replaces the mirror read.
    """
    meta = _dams_meta()
    base: dict[str, Any] = {"dam_km": float(dam_km), "source": meta, "method": DAMS_METHOD}
    if len(coords) < 2:
        return {**base, "available": True, "dams": [], "n_dams": 0, "total_capacity_mcm": 0.0,
                "summary": "No path to look along."}
    keys = path_cells(coords, dam_km)
    truncated_cells = len(keys) > max_cells
    if rows is None:
        try:
            rows = _dam_rows(keys[:max_cells])
        except Exception as exc:  # noqa: BLE001 - the path stands without its dams
            return {**base, "available": False, "dams": [], "n_dams": None, "error": f"{type(exc).__name__}: {exc}",
                    "summary": f"Dams on the path are unavailable ({exc})."}
    if rows is None:
        return {**base, "available": False, "dams": [], "n_dams": None, "note": _missing_note(),
                "summary": "Dams on the path are not available yet (the Global Dam Watch mirror is not published)."}
    index = rivers.PathIndex(coords, dam_km)
    found: list[dict[str, Any]] = []
    side = 0
    behind = 0
    seen: set[str] = set()
    for r in rows:
        d = _dam_record(r)
        if _unplaced(d):
            continue
        ident = str(d.get("gdw_id") or f"{d['lat']:.5f},{d['lon']:.5f}")
        if ident in seen:
            continue
        loc = index.locate(d["lat"], d["lon"])
        if loc is None:
            continue
        if loc[1] <= 0.0 and loc[0] > START_KM:
            # nearest to the very first point and off it: the dam is behind the start (upstream), not on the way
            behind += 1
            continue
        if (min_catchment_km2 and d.get("catchment_km2") is not None
                and d["catchment_km2"] < 0.5 * float(min_catchment_km2)):
            side += 1
            continue
        seen.add(ident)
        d["distance_km"] = round(loc[0], 2)
        d["along_km"] = round(loc[1], 1)
        if reach_of_segment is not None and 0 <= loc[2] < len(reach_of_segment):
            d["river_id"] = reach_of_segment[loc[2]]
        found.append(d)
    found.sort(key=lambda d: (d["along_km"], d["distance_km"]))
    total = sum(d.get("capacity_mcm") or 0.0 for d in found)
    out = {**base, "available": True, "dams": found, "n_dams": len(found), "total_capacity_mcm": round(total, 1),
           "side_streams_left_out": side, "behind_the_start": behind, "complete": not truncated_cells}
    n = len(found)
    if n:
        big = max(found, key=lambda d: d.get("capacity_mcm") or 0.0)
        out["summary"] = (f"{n} dam{'s' if n != 1 else ''} within {dam_km:g} km of the path"
                          + (f", holding about {total:,.0f} million m3" if total else "")
                          + (f"; the largest is {_label(big)} at km {big['along_km']:,.0f}"
                             if big.get("capacity_mcm") else "") + ".")
    else:
        out["summary"] = f"No dams in Global Dam Watch within {dam_km:g} km of the path."
    if truncated_cells:
        out["note"] = f"Only the first {max_cells} cells of the path were searched; the river goes on past them."
    return out


# ── countries ────────────────────────────────────────────────────────────────


class _Country:
    """One admin-0 country: its rings, its bounding box and, built on first use, its edges by 1-degree band."""

    __slots__ = ("_bands", "bbox", "code", "name", "rings")

    def __init__(self, name: str, code: str | None, rings: list[list[tuple[float, float]]]):
        self.name = name
        self.code = code
        self.rings = [r for r in rings if len(r) >= 3]
        xs = [p[0] for r in self.rings for p in r] or [0.0]
        ys = [p[1] for r in self.rings for p in r] or [0.0]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self._bands: dict[int, list[tuple[float, float, float, float]]] | None = None

    def _build(self) -> dict[int, list[tuple[float, float, float, float]]]:
        bands: dict[int, list[tuple[float, float, float, float]]] = {}
        for ring in self.rings:
            for a, b in zip(ring, ring[1:] + ring[:1]):
                if a[1] == b[1]:
                    continue
                for band in range(int(math.floor(min(a[1], b[1]))), int(math.floor(max(a[1], b[1]))) + 1):
                    bands.setdefault(band, []).append((a[0], a[1], b[0], b[1]))
        return bands

    def contains(self, lon: float, lat: float) -> bool:
        w, s, e, n = self.bbox
        if not (w <= lon <= e and s <= lat <= n):
            return False
        if self._bands is None:
            self._bands = self._build()
        inside = False
        for x1, y1, x2, y2 in self._bands.get(int(math.floor(lat)), ()):
            if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside


def parse_countries_topojson(topo: dict[str, Any], obj: str = "countries") -> list[_Country]:
    """The polygons of a TopoJSON geometry collection (quantised and delta-encoded or not) as countries."""
    tf = topo.get("transform") or None
    sx, sy = (tf["scale"] if tf else (1.0, 1.0))
    tx, ty = (tf["translate"] if tf else (0.0, 0.0))
    arcs: list[list[tuple[float, float]]] = []
    for arc in topo.get("arcs") or []:
        pts: list[tuple[float, float]] = []
        x = y = 0.0
        for p in arc:
            if tf:
                x, y = x + p[0], y + p[1]
                pts.append((x * sx + tx, y * sy + ty))
            else:
                pts.append((float(p[0]), float(p[1])))
        arcs.append(pts)

    def ring(idx: list[int]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for a in idx:
            pts = arcs[a] if a >= 0 else arcs[~a][::-1]
            out.extend(pts if not out else pts[1:])
        return out

    countries: list[_Country] = []
    for g in ((topo.get("objects") or {}).get(obj) or {}).get("geometries") or []:
        kind = g.get("type")
        if kind == "Polygon":
            polys = [g.get("arcs") or []]
        elif kind == "MultiPolygon":
            polys = g.get("arcs") or []
        else:
            continue
        name = (g.get("properties") or {}).get("name") or str(g.get("id") or "unnamed")
        countries.append(_Country(name, str(g["id"]) if g.get("id") is not None else None,
                                  [ring(r) for poly in polys for r in poly]))
    return countries


def _country_at(countries: list[_Country], lon: float, lat: float, hint: _Country | None) -> _Country | None:
    if hint is not None and hint.contains(lon, lat):
        return hint
    for c in countries:
        if c is not hint and c.contains(lon, lat):
            return c
    return None


def countries_along(coords: list[list[float]], *, samples: int = 400,
                    countries: list[_Country] | None = None) -> dict[str, Any]:
    """The countries a path crosses, in the order the water enters them, with the km in each.

    The path is sampled at about ``samples`` even steps and each sample placed in a Natural Earth 1:50m
    admin-0 country. Steps over the sea, or off the coarse coastline near the mouth, count for no country. A
    river along a border wanders between both sides at this scale, so both are listed.
    """
    from aquascope.registry import CONTEXT_LAYERS

    meta = CONTEXT_LAYERS["natural_earth"]
    base = {"source": meta.attribution, "licence": f"{meta.license} ({meta.short})", "url": COUNTRIES_URL}
    if len(coords) < 2:
        return {**base, "available": True, "countries": [], "summary": ""}
    if countries is None:
        try:
            countries = _countries()
        except Exception as exc:  # noqa: BLE001 - the path stands without its countries
            logger.info("country boundaries unreadable: %s", exc)
            return {**base, "available": False, "countries": [], "error": f"{type(exc).__name__}: {exc}",
                    "summary": "The countries could not be read."}
    index = rivers.PathIndex(coords, 1.0)
    total = index.length_km
    step = max(total / max(1, int(samples)), 0.5)
    stats: dict[str, dict[str, Any]] = {}
    hint: _Country | None = None
    k = 0
    at = step / 2
    while at < total:
        while k < len(index.cum) - 2 and index.cum[k + 1] < at:
            k += 1
        span = index.cum[k + 1] - index.cum[k]
        t = (at - index.cum[k]) / span if span > 0 else 0.0
        a, b = coords[k], coords[k + 1]
        lon, lat = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        c = _country_at(countries, lon, lat, hint)
        if c is not None:
            hint = c
            s = stats.setdefault(c.name, {"name": c.name, "code": c.code, "first_km": round(max(0.0, at - step / 2), 1),
                                          "km": 0.0})
            s["km"] += min(step, total - (at - step / 2))
        at += step
    listed = sorted(stats.values(), key=lambda s: s["first_km"])
    for s in listed:
        s["km"] = round(s["km"], 1)
    names = [s["name"] for s in listed]
    summary = ("Crosses " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) + "."
               if names else "No country boundary found along the path (open water at this scale).")
    return {**base, "available": True, "countries": listed, "summary": summary}


# ── upstream of a reach ──────────────────────────────────────────────────────


def _match_reaches(dams: list[dict[str, Any]], max_tiles: int) -> tuple[dict[int, int], int]:
    """The river reach each dam sits on (index in ``dams`` -> river_id) from the zoom-8 stream tiles, and how
    many dams were left unchecked because their tile was past ``max_tiles``."""
    z = MATCH_ZOOM
    by_tile: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dams):
        gx, gy = rivers._world(d["lon"], d["lat"], z)
        by_tile.setdefault((int(gx // 4096), int(gy // 4096)), []).append(i)
    # the tiles with the most candidates first, so a cap drops the sparse ones
    order = sorted(by_tile, key=lambda t: -len(by_tile[t]))
    matched: dict[int, int] = {}
    unchecked = 0
    for n, tile in enumerate(order):
        if n >= max_tiles:
            unchecked += len(by_tile[tile])
            continue
        reaches = rivers._tile_reaches(z, tile[0], tile[1])
        lines = []
        for rid, reach in reaches.items():
            for line in reach["lines"]:
                xs = [p[0] for p in line]
                ys = [p[1] for p in line]
                lines.append((rid, line, min(xs), min(ys), max(xs), max(ys)))
        for i in by_tile[tile]:
            d = dams[i]
            gx, gy = rivers._world(d["lon"], d["lat"], z)
            tol = MATCH_M / rivers._metres_per_unit(d["lat"], z)
            best: tuple[float, int] | None = None
            for rid, line, x0, y0, x1, y1 in lines:
                if gx < x0 - tol or gx > x1 + tol or gy < y0 - tol or gy > y1 + tol:
                    continue
                for a, b in zip(line, line[1:]):
                    dist = rivers._segment_distance(gx, gy, a[0], a[1], b[0], b[1])[0]
                    if dist <= tol and (best is None or dist < best[0]):
                        best = (dist, rid)
            if best is not None:
                matched[i] = best[1]
    return matched, unchecked


def _mean_flow(rid: int) -> float | None:
    """The mean of GEOGLOWS v2's annual average flows for a reach, m3/s (modelled), or None."""
    data = rivers._fetch_json(f"{rivers.GEOGLOWS_API}/annualaverages/{rid}", {"format": "json"})
    values = data.get(str(rid)) if isinstance(data, dict) else None
    vals = [float(v) for v in (values or []) if v is not None and math.isfinite(float(v)) and float(v) >= 0]
    return sum(vals) / len(vals) if vals else None


def upstream_dams(river_id: int | str | None = None, *, lat: float | None = None, lon: float | None = None,
                  max_distance_m: float = rivers.DEFAULT_MAX_DISTANCE_M, with_flow: bool = True, limit: int = 10,
                  max_cells: int = 48, max_tiles: int = 40) -> dict[str, Any]:
    """Is the river regulated upstream of this reach? The Global Dam Watch dams that drain to it, their storage,
    and (``with_flow``) the degree of regulation: that storage as a share of a year's mean flow here.

    Give ``river_id``, or ``lat``/``lon`` to snap first. ``dams`` lists up to ``limit`` of them, largest storage
    first. A basin whose bounding box spans more than ``max_cells`` 2-degree cells is not searched (the answer
    says so); dams in stream tiles past ``max_tiles`` are counted as ``unchecked``. Approximate by construction:
    see ``note``.
    """
    from aquascope.context._common import cells_for_bbox

    meta = _dams_meta()
    snap: dict[str, Any] | None = None
    if river_id is None:
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon to snap to the nearest reach")
        snap = rivers.snap_to_river(lat, lon, max_distance_m=max_distance_m)
        if not snap["snapped"]:
            return {"river_id": None, "snap": snap, "available": False, "regulated": None,
                    "summary": snap["message"], "source": meta}
        river_id = snap["river_id"]
    rid = rivers._river_id(river_id)
    net = rivers._network_for(rid, lat, lon)
    i = net.index(rid)
    assert i is not None
    ups = rivers._upstream_of(net, i)
    area_km2 = float(net.area_m2[ups].sum()) / 1e6
    base: dict[str, Any] = {"river_id": rid, "upstream_area_km2": round(area_km2, 1), "source": meta,
                            "method": UPSTREAM_METHOD, "note": UPSTREAM_NOTE, "attribution": meta["attribution"]}
    if snap is not None:
        base["snap"] = snap
    lons = [float(x) for x in net.lon[ups] if math.isfinite(float(x))]
    lats = [float(x) for x in net.lat[ups] if math.isfinite(float(x))]
    if lat is not None and lon is not None:
        lons.append(float(lon))
        lats.append(float(lat))
    if not lons:
        return {**base, "available": False, "regulated": None, "summary": "The basin has no location to search."}
    # the reach centres are ERA5-cell weighted (0.25 degree), so the box gets a generous margin
    pad = 0.35
    bbox = (max(-180.0, min(lons) - pad), max(-90.0, min(lats) - pad),
            min(180.0, max(lons) + pad), min(90.0, max(lats) + pad))
    keys = cells_for_bbox(*bbox)
    if len(keys) > max_cells:
        return {**base, "available": True, "complete": False, "regulated": None, "n_dams": None,
                "bbox": [round(v, 3) for v in bbox],
                "summary": (f"The basin is too large to list its dams here ({len(keys)} cells of 2 degrees); "
                            "BasinATLAS's degree of regulation covers it.")}
    try:
        rows = _dam_rows(keys)
    except Exception as exc:  # noqa: BLE001
        return {**base, "available": False, "regulated": None, "error": f"{type(exc).__name__}: {exc}",
                "summary": f"Dams upstream are unavailable ({exc})."}
    if rows is None:
        return {**base, "available": False, "regulated": None, "note": _missing_note(),
                "summary": "Dams upstream are not available yet (the Global Dam Watch mirror is not published)."}
    # where the basin is, on a 0.25 degree grid (the reach centres are ERA5-cell weighted): a candidate dam must
    # be in or next to one of its cells, which keeps the stream-tile reads to the basin itself
    near: set[tuple[int, int]] = set()
    for x, y in zip(lons, lats):
        cx, cy = int(math.floor(x / 0.25)), int(math.floor(y / 0.25))
        near.update((cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    cands: list[dict[str, Any]] = []
    for r in rows:
        d = _dam_record(r)
        if _unplaced(d):
            continue
        if (int(math.floor(d["lon"] / 0.25)), int(math.floor(d["lat"] / 0.25))) not in near:
            continue
        # a dam that drains more than the reach does is below it, whatever its position says
        if d.get("catchment_km2") is not None and d["catchment_km2"] > 1.25 * area_km2:
            continue
        cands.append(d)
    matched, unchecked = _match_reaches(cands, max_tiles)
    up_ids = {int(x) for x in net.ids[ups]}
    found = []
    for k, d in enumerate(cands):
        r_id = matched.get(k)
        if r_id is None or r_id not in up_ids:
            continue
        d["river_id"] = r_id
        d["on_this_reach"] = r_id == rid
        found.append(d)
    found.sort(key=lambda d: d.get("capacity_mcm") or 0.0, reverse=True)
    total = sum(d.get("capacity_mcm") or 0.0 for d in found)
    out: dict[str, Any] = {**base, "available": True, "complete": unchecked == 0, "bbox": [round(v, 3) for v in bbox],
                           "n_dams": len(found), "total_capacity_mcm": round(total, 1),
                           "unchecked": unchecked, "dams": found[:max(0, int(limit))],
                           # none found among the dams checked is not "unregulated" while some went unchecked
                           "regulated": True if found else (False if unchecked == 0 else None)}
    if with_flow and found and total > 0:
        try:
            q = _mean_flow(rid)
        except Exception as exc:  # noqa: BLE001 - the dams stand without the flow
            q = None
            out["flow_error"] = f"{type(exc).__name__}: {exc}"
        if q:
            out["mean_flow_cms"] = round(q, 3)
            out["mean_flow_source"] = "GEOGLOWS v2 annual averages (modelled), CC BY 4.0"
            out["degree_of_regulation_pct"] = round(100.0 * total / (q * MCM_PER_CMS_YEAR), 1)
    n = len(found)
    if n:
        text = (f"Regulated upstream: {n} dam{'s' if n != 1 else ''} in Global Dam Watch drain to this reach"
                + (f", holding about {total:,.0f} million m3" if total else ""))
        if out.get("degree_of_regulation_pct") is not None:
            text += (f", about {out['degree_of_regulation_pct']:g} % of a year's mean flow here "
                     "(GEOGLOWS, modelled)")
        big = found[0]
        if big.get("capacity_mcm"):
            text += f"; the largest is {_label(big)}"
        out["summary"] = text + "."
    elif unchecked:
        out["summary"] = "No dams in Global Dam Watch found upstream of this reach among those checked."
    else:
        out["summary"] = "No dams in Global Dam Watch upstream of this reach."
    if unchecked:
        out["summary"] += f" {unchecked} more dam{'s' if unchecked != 1 else ''} in the basin's box were not checked."
    return out
