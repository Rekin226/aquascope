"""A cross-section and a hydrograph through a flood model's results (#621). Experimental.

What RAS Mapper's profile line and time series plot show, from a :class:`aquascope.io.mesh.Mesh`:

* :func:`section`: draw a line across the flood and read, along it, the ground and the water level at every
  model step (and the highest water level of the run). The ground is the cell's bed level: HEC-RAS's
  ``Cells Minimum Elevation`` (the lowest terrain point in the cell, not the terrain under the line), or a
  UGRID file's bed level. A file without bed levels gives the water surface only, and says so.
* :func:`hydrograph`: the water level and the depth at one cell over the run, with the arrival time, the peak
  and the duration marked. They follow the definitions of HEC-RAS Mapper's Arrival Time, Duration and
  Recession maps (HEC-RAS Mapper User's Manual, "Creating Results Maps",
  https://www.hec.usace.army.mil/confluence/rasdocs/rmum/6.3/mapping-results/creating-results-maps): arrival
  is when the depth first exceeds a threshold depth, recession when it falls back below it for the last time,
  and the duration runs from one to the other (like RAS, a second peak does not start a new duration). The
  threshold defaults to the readers' own dry depth (:data:`aquascope.io.mesh.DRY_DEPTH`), and every time is to
  the model's output interval, not its computation interval.

numpy and the standard library only, so it runs in the Explorer's mesh worker beside the reader; the same
functions answer ``aquascope mesh`` and the MCP tools. Every number a reader sees is computed here.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from aquascope.io.mesh import DRY_DEPTH, Mesh, MeshError

__all__ = ["HEC_MAPS", "MeshIndex", "cell_at", "cells_along", "hydrograph", "section"]

#: The definitions the hydrograph marks follow.
HEC_MAPS = ("https://www.hec.usace.army.mil/confluence/rasdocs/rmum/6.3/mapping-results/creating-results-maps")

#: How many points a section reads along its line at most, and at least.
MAX_SAMPLES = 400
MIN_SAMPLES = 40
#: Steps read from the file at a time.
BLOCK = 48
EARTH_M = 6371008.8


# ── small helpers ───────────────────────────────────────────────────────────


def _r(x: Any, nd: int = 3) -> float | None:
    """A float for JSON: rounded, None for NaN or inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _plain(a: Any, nd: int = 3) -> list[float | None]:
    return [_r(x, nd) for x in np.asarray(a, dtype="f8").reshape(-1)]


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(str(t).replace("Z", "+00:00")[:19])


def hours_since_start(times: list[str]) -> np.ndarray:
    """Each model time in hours after the first."""
    if not times:
        return np.zeros(0)
    t0 = _parse(times[0])
    return np.array([(_parse(t) - t0).total_seconds() / 3600.0 for t in times])


def _bed(m: Mesh) -> np.ndarray | None:
    bed = getattr(m._source, "bed", None)
    if bed is None or m.variable != "depth":
        return None
    return np.asarray(bed, dtype="f8")


def _length_unit(m: Mesh) -> str | None:
    return m.units.get("length") or m.units.get(m.variable) or m.units.get("depth")


def _ground_words(m: Mesh) -> str:
    if _bed(m) is None:
        return ""
    if m.format.startswith("hec") or "RAS" in (m.model or ""):
        return "the lowest ground in each cell (HEC-RAS Cells Minimum Elevation)"
    return "the bed level of each cell, as the file gives it"


def _boxes(m: Mesh) -> np.ndarray:
    off = np.asarray(m.ring_offsets).astype(np.intp)
    counts = np.diff(off)
    boxes = np.full((m.n_cells, 4), np.nan)
    ok = counts > 0
    starts = off[:-1][ok]
    xs, ys = np.asarray(m.ring_lon), np.asarray(m.ring_lat)
    boxes[ok, 0] = np.minimum.reduceat(xs, starts)
    boxes[ok, 1] = np.minimum.reduceat(ys, starts)
    boxes[ok, 2] = np.maximum.reduceat(xs, starts)
    boxes[ok, 3] = np.maximum.reduceat(ys, starts)
    return boxes


class MeshIndex:
    """Which cell holds a point, for many points at once: the cells' boxes binned on a regular grid, then an
    even-odd test of each point against the few cells of its bin, all in numpy."""

    def __init__(self, m: Mesh):
        self.m = m
        boxes = _boxes(m)
        self.boxes = boxes
        ok = np.isfinite(boxes).all(axis=1)
        self.ok = ok
        if not ok.any():
            self.nb, self.cells, self.start = 0, np.zeros(0, np.intp), np.zeros(1, np.intp)
            return
        self.w, self.s = float(np.min(boxes[ok, 0])), float(np.min(boxes[ok, 1]))
        e, n = float(np.max(boxes[ok, 2])), float(np.max(boxes[ok, 3]))
        nb = max(1, int(math.sqrt(int(ok.sum()))))
        self.nb = nb
        self.dx = max((e - self.w) / nb, 1e-12)
        self.dy = max((n - self.s) / nb, 1e-12)
        idx = np.nonzero(ok)[0].astype(np.intp)
        b = boxes[idx]
        x0, x1 = self._bin(b[:, 0], self.w, self.dx), self._bin(b[:, 2], self.w, self.dx)
        y0, y1 = self._bin(b[:, 1], self.s, self.dy), self._bin(b[:, 3], self.s, self.dy)
        wx = x1 - x0 + 1
        counts = wx * (y1 - y0 + 1)
        total = int(counts.sum())
        first = np.repeat(np.cumsum(counts) - counts, counts)
        local = np.arange(total, dtype=np.intp) - first
        wxr = np.repeat(wx, counts)
        bins = (np.repeat(y0, counts) + local // wxr) * nb + np.repeat(x0, counts) + local % wxr
        cell = np.repeat(idx, counts)
        order = np.argsort(bins, kind="stable")
        self.cells = cell[order]
        self.start = np.searchsorted(bins[order], np.arange(nb * nb + 1)).astype(np.intp)

    def _bin(self, v: np.ndarray, lo: float, step: float) -> np.ndarray:
        return np.clip(np.floor((v - lo) / step), 0, self.nb - 1).astype(np.intp)

    def locate(self, lon: Any, lat: Any) -> np.ndarray:
        """The cell holding each point, -1 outside the mesh."""
        lon = np.atleast_1d(np.asarray(lon, dtype="f8"))
        lat = np.atleast_1d(np.asarray(lat, dtype="f8"))
        out = np.full(lon.size, -1, dtype=np.intp)
        if not lon.size or not self.nb:
            return out
        fin = np.isfinite(lon) & np.isfinite(lat)
        bx = np.floor((lon - self.w) / self.dx)
        by = np.floor((lat - self.s) / self.dy)
        inside = fin & (bx >= 0) & (bx < self.nb) & (by >= 0) & (by < self.nb)
        pts = np.nonzero(inside)[0].astype(np.intp)
        if not pts.size:
            return out
        b = (by[pts] * self.nb + bx[pts]).astype(np.intp)
        n_c = self.start[b + 1] - self.start[b]
        has = n_c > 0
        pts, b, n_c = pts[has], b[has], n_c[has]
        if not pts.size:
            return out
        # every (point, candidate cell) pair
        first = np.repeat(np.cumsum(n_c) - n_c, n_c)
        local = np.arange(int(n_c.sum()), dtype=np.intp) - first
        pp = np.repeat(pts, n_c)
        cc = self.cells[np.repeat(self.start[b], n_c) + local]
        bx_ = self.boxes[cc]
        x, y = lon[pp], lat[pp]
        keep = (bx_[:, 0] <= x) & (x <= bx_[:, 2]) & (bx_[:, 1] <= y) & (y <= bx_[:, 3])
        pp, cc, x, y = pp[keep], cc[keep], x[keep], y[keep]
        if not pp.size:
            return out
        off = np.asarray(self.m.ring_offsets).astype(np.intp)
        ne = off[cc + 1] - off[cc]
        pair = np.repeat(np.arange(pp.size, dtype=np.intp), ne)
        first = np.repeat(np.cumsum(ne) - ne, ne)
        local = np.arange(int(ne.sum()), dtype=np.intp) - first
        k = np.repeat(off[cc], ne) + local
        last = local == np.repeat(ne - 1, ne)
        k2 = np.where(last, np.repeat(off[cc], ne), k + 1)
        xs, ys = np.asarray(self.m.ring_lon), np.asarray(self.m.ring_lat)
        px, py, qx, qy = xs[k], ys[k], xs[k2], ys[k2]
        xp, yp = x[pair], y[pair]
        cross = (py > yp) != (qy > yp)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = px + (yp - py) * (qx - px) / (qy - py)
        hit = cross & (xp < xi)
        parity = np.bincount(pair, weights=hit.astype("f8"), minlength=pp.size).astype(np.intp) % 2 == 1
        if parity.any():
            ip, ic = pp[parity], cc[parity]
            # one cell per point: the first in the mesh's own order, as a click would find
            order = np.lexsort((ic, ip))
            ip, ic = ip[order], ic[order]
            u, first_i = np.unique(ip, return_index=True)
            out[u] = ic[first_i]
        return out


def cells_along(m: Mesh, lon: Any, lat: Any, index: MeshIndex | None = None) -> np.ndarray:
    """The cell holding each point (its outline contains it), -1 outside the mesh."""
    return (index or MeshIndex(m)).locate(lon, lat)


def cell_at(m: Mesh, lon: float, lat: float) -> int:
    """The cell holding one point, -1 outside the mesh."""
    return int(cells_along(m, [lon], [lat])[0])


def _typical_cell_m(m: Mesh) -> float:
    area = m.cell_areas()
    area = area[np.isfinite(area) & (area > 0)]
    return float(math.sqrt(np.median(area))) if area.size else 50.0


def _along(line: list[list[float]], spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Points every ``spacing`` metres along a polyline (its ends included): lon, lat and distance."""
    pts = np.asarray(line, dtype="f8")
    lat0 = math.radians(float(np.mean(pts[:, 1])))
    x = np.radians(pts[:, 0]) * math.cos(lat0) * EARTH_M
    y = np.radians(pts[:, 1]) * EARTH_M
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    n = int(min(MAX_SAMPLES, max(MIN_SAMPLES, math.ceil(total / spacing) + 1))) if total > 0 else 1
    d = np.linspace(0.0, total, n)
    lon = np.interp(d, cum, pts[:, 0])
    lat = np.interp(d, cum, pts[:, 1])
    return lon, lat, d, total


def _check_line(line: Any) -> list[list[float]]:
    try:
        pts = [[float(p[0]), float(p[1])] for p in line]
    except (TypeError, ValueError, IndexError) as exc:
        raise MeshError("A section needs a line: two or more [longitude, latitude] points.") from exc
    pts = [p for p in pts if math.isfinite(p[0]) and math.isfinite(p[1])]
    if len(pts) < 2:
        raise MeshError("A section needs a line: two or more [longitude, latitude] points.")
    if all(p == pts[0] for p in pts):
        raise MeshError("The two ends of the section are the same point: draw a line across the flood.")
    return pts


# ── the section ─────────────────────────────────────────────────────────────


def section(m: Mesh, line: Any, *, spacing_m: float | None = None) -> dict[str, Any]:
    """The ground and the water level along ``line`` ([[lon, lat], ...]) at every step of the run.

    Returns ``distance_m`` (from the line's start), ``lon``/``lat`` and ``cell`` of each point read (-1 off
    the mesh), ``ground`` (None without bed levels), ``levels`` (one list per model step, None where dry or off
    the mesh), ``max_level`` (the highest water level of the run at each point), ``times``, the
    units and a few facts: the length, the share of the line on the mesh, the widest wet width and the deepest
    water along the line, each with its step.
    """
    pts = _check_line(line)
    cell_m = _typical_cell_m(m)
    spacing = float(spacing_m) if spacing_m else max(cell_m / 4.0, 1.0)
    lon, lat, dist, total = _along(pts, spacing)
    cells = cells_along(m, lon, lat)
    on = cells >= 0
    if not on.any():
        raise MeshError("The line does not cross the model's mesh: draw it across the flood.")
    bed = _bed(m)
    ground = np.full(lon.size, np.nan)
    if bed is not None:
        ground[on] = bed[cells[on]]
    uniq, inv = np.unique(cells[on], return_inverse=True)
    n_steps = len(m.times)
    levels = np.full((n_steps, lon.size), np.nan)
    depth = np.full((n_steps, lon.size), np.nan)
    for i in range(0, n_steps, BLOCK):
        block = m.values_block(i, min(i + BLOCK, n_steps))[:, uniq]   # depth (NaN dry) or water surface
        vals = block[:, inv]
        if bed is not None:
            depth[i:i + vals.shape[0], on] = vals
            levels[i:i + vals.shape[0], on] = vals + ground[on]
        else:
            levels[i:i + vals.shape[0], on] = vals
    max_level = np.full(lon.size, np.nan)
    mv = np.asarray(m.max_value, dtype="f8")[cells[on]]
    if bed is not None:
        mv = np.where(mv > DRY_DEPTH, mv + ground[on], np.nan)
    max_level[on] = mv
    step_len = np.diff(dist).mean() if dist.size > 1 else 0.0
    facts: dict[str, Any] = {"length_m": _r(total, 1), "on_mesh_pct": _r(100.0 * on.mean(), 1),
                             "spacing_m": _r(step_len, 2), "typical_cell_m": _r(cell_m, 1)}
    if n_steps and bed is not None:
        wet = np.isfinite(depth)
        width = wet.sum(axis=1) * step_len
        k = int(np.argmax(width))
        facts["widest_wet_m"] = _r(width[k], 1)
        facts["widest_step"] = k
        if np.isfinite(depth).any():
            s, j = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(depth), depth, -np.inf))), depth.shape)
            facts["deepest"] = _r(depth[s, j])
            facts["deepest_at_m"] = _r(dist[j], 1)
            facts["deepest_step"] = int(s)
    notes = []
    if bed is None:
        notes.append("No bed levels in this file, so the section shows the water surface only, no ground.")
    if not on.all():
        notes.append(f"{100.0 * (1 - on.mean()):.0f}% of the line lies off the mesh.")
    return {
        "line": [[_r(p[0], 6), _r(p[1], 6)] for p in pts],
        "distance_m": _plain(dist, 2), "lon": _plain(lon, 6), "lat": _plain(lat, 6),
        "cell": [int(c) for c in cells],
        "ground": _plain(ground) if bed is not None else None, "ground_is": _ground_words(m),
        "levels": [_plain(row) for row in levels], "max_level": _plain(max_level),
        "times": list(m.times),
        "units": {"length": _length_unit(m), "distance": "m"},
        "variable": m.variable, "facts": facts, "notes": notes, "name": m.name,
    }


# ── the hydrograph ──────────────────────────────────────────────────────────


def _when(times: list[str], k: int | None) -> str | None:
    return None if k is None else times[k]


def hydrograph(m: Mesh, cell: int | None = None, *, lon: float | None = None, lat: float | None = None,
               threshold: float | None = None) -> dict[str, Any]:
    """The water level and the depth at one cell (``cell``, or the cell under ``lon``/``lat``) over the run,
    with the arrival, the peak, the recession and the duration marked.

    ``threshold`` is the depth that counts as flooded (in the file's length unit; default
    :data:`~aquascope.io.mesh.DRY_DEPTH`). Times are hours after the run's first step and ISO strings;
    ``still_wet`` says the water had not receded by the end of the run (the duration is then a lower bound).
    """
    if cell is None:
        if lon is None or lat is None:
            raise MeshError("Say which cell: its number, or a longitude and latitude inside the mesh.")
        cell = cell_at(m, float(lon), float(lat))
        if cell < 0:
            raise MeshError("That point is outside the model's mesh.")
    cell = int(cell)
    if not m.times:
        raise MeshError("This file has no time steps, only its maxima, so there is no hydrograph.")
    thr = DRY_DEPTH if threshold is None else float(threshold)
    s = m.series(cell)
    value = np.asarray(s["value"], dtype="f8")
    hours = hours_since_start(m.times)
    bed = _bed(m)
    ground = float(bed[cell]) if bed is not None else None
    if bed is not None:
        depth = value
        level = np.where(np.isfinite(depth), depth + ground, np.nan)
    else:
        depth = None
        level = value
    out: dict[str, Any] = {
        "cell": cell, "lon": _r(m.lon[cell], 6), "lat": _r(m.lat[cell], 6), "times": list(m.times),
        "hours": _plain(hours, 4), "level": _plain(level), "depth": _plain(depth) if depth is not None else None,
        "ground": _r(ground), "ground_is": _ground_words(m), "threshold": thr,
        "units": {"length": _length_unit(m), "time": "h"}, "variable": m.variable, "source": HEC_MAPS,
    }
    if depth is None:
        k = int(np.nanargmax(level)) if np.isfinite(level).any() else None
        out.update({"wet": k is not None, "peak": None if k is None else
                    {"step": k, "time": m.times[k], "hours": _r(hours[k], 4), "level": _r(level[k])},
                    "arrival": None, "recession": None, "duration_h": None, "still_wet": None,
                    "note": "No bed levels in this file, so no depth: no arrival time or duration, only the "
                            "water surface and its highest point."})
        return out
    wet = np.isfinite(depth) & (depth > thr)
    if not wet.any():
        out.update({"wet": False, "peak": None, "arrival": None, "recession": None, "duration_h": None,
                    "still_wet": False, "note": f"The water never got deeper than {thr:g} here."})
        return out
    idx = np.nonzero(wet)[0]
    a, r = int(idx[0]), int(idx[-1])
    k = int(np.nanargmax(np.where(wet, depth, -np.inf)))
    still = bool(r == len(depth) - 1)
    # Recession: the first step after the last wet one (the water is back under the threshold there).
    rec = None if still else r + 1
    end_h = hours[r] if still else hours[rec]
    gaps = int(np.count_nonzero(~wet[a:r + 1]))
    out.update({
        "wet": True,
        "arrival": {"step": a, "time": m.times[a], "hours": _r(hours[a], 4), "depth": _r(depth[a]),
                    "level": _r(level[a])},
        "peak": {"step": k, "time": m.times[k], "hours": _r(hours[k], 4), "depth": _r(depth[k]),
                 "level": _r(level[k])},
        "recession": None if rec is None else {"step": rec, "time": m.times[rec], "hours": _r(hours[rec], 4)},
        "duration_h": _r(end_h - hours[a], 3), "still_wet": still,
        "dry_spells": gaps,
        "note": ("Still wet at the end of the run: the duration is at least this long." if still else
                 ("The depth dipped under the threshold and rose again; like HEC-RAS, the duration runs on to the "
                  "last recession." if gaps else "")),
    })
    return out
