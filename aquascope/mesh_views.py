"""Flood model results the way an engineer reads them (#619, #620): smooth depth, contours and lenses.

Experimental (``docs/stability.md``): the names and the shapes here may change in a minor release.

A mesh result (:class:`aquascope.io.mesh.Mesh`) has one value per cell. Drawn as it is, it is a mosaic of
squares. RAS Mapper and Crayfish draw it smooth, with depth contours, and they offer lenses on the whole event:
the flood hazard, when each place first floods, the largest depth and speed. This module computes them; the
Explorer (``explorer/src/mesh-views.js``), the CLI (``aquascope mesh lenses FILE``) and the MCP tool
``mesh_lenses`` only show them.

* :func:`field_grid` and :func:`smooth_field`: the cell values on a regular Web Mercator grid, interpolated
  linearly between cell centres on their Delaunay triangulation (scipy). Depth takes a dry cell as zero, so the
  wet edge tapers between the last wet and the first dry centre instead of stepping at a cell face. There is no
  sub-grid terrain in a result file, so this is an interpolation of the cell values, not RAS Mapper's
  terrain-based depth.
* :func:`contour_levels` and :func:`contour_lines`: lines of equal depth (or arrival time) at a round interval,
  traced on that grid with contourpy, as GeoJSON in longitude and latitude.
* :func:`hazard_class` and :func:`mesh_lenses`: the combined flood hazard curves H1 to H6 of the Australian
  Disaster Resilience Guideline 7-3 (AIDR 2014, Tables 1 and 2; background in Smith et al. 2014), assessed at every
  output step as the guideline asks ("at all stages of the flood hydrograph"), the arrival time (the first
  output step with water at least :data:`ARRIVAL_DEPTH_M` deep), and the largest depth and speed with the step
  they were reached.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from aquascope.io.mesh import DRY_DEPTH, Mesh, MeshError, read_mesh

# ── the flood hazard curves ─────────────────────────────────────────────────

HAZARD_SOURCE = {
    "name": "Australian Disaster Resilience Guideline 7-3: Technical flood risk management guideline: Flood hazard",
    "short": "AIDR Guideline 7-3 (2014), Tables 1 and 2",
    "publisher": "Australian Institute for Disaster Resilience",
    "year": 2014,
    "tables": "Table 1 (vulnerability thresholds) and Table 2 (classification limits, which the guideline says are "
              "modelled after Smith and Wasko 2012); background to the curves in Smith, Davey and Cox (2014), Flood "
              "hazard, UNSW Water Research Laboratory Technical Report 2014/07",
    "url": "https://knowledge.aidr.org.au/media/1891/guideline-7-3-technical-flood-risk-management.pdf",
    "licence": "CC BY-NC 4.0 (the guideline's own terms)",
}

# Table 2's classification limits: a place is in the lowest class whose three limits it meets (D x V in m2/s,
# still-water depth D in m, velocity V in m/s, each inclusive); beyond H5 it is H6. Words from Table 1.
HAZARD_CLASSES: tuple[dict[str, Any], ...] = (
    {"code": "H1", "dv": 0.3, "depth": 0.3, "velocity": 2.0,
     "words": "Generally safe for vehicles, people and buildings."},
    {"code": "H2", "dv": 0.6, "depth": 0.5, "velocity": 2.0, "words": "Unsafe for small vehicles."},
    {"code": "H3", "dv": 0.6, "depth": 1.2, "velocity": 2.0, "words": "Unsafe for vehicles, children and the elderly."},
    {"code": "H4", "dv": 1.0, "depth": 2.0, "velocity": 2.0, "words": "Unsafe for vehicles and people."},
    {"code": "H5", "dv": 4.0, "depth": 4.0, "velocity": 4.0,
     "words": "Unsafe for vehicles and people. All building types vulnerable to structural damage. Some less "
              "robust building types vulnerable to failure."},
    {"code": "H6", "dv": None, "depth": None, "velocity": None,
     "words": "Unsafe for vehicles and people. All building types considered vulnerable to failure."},
)

# The arrival time is the first output step with at least this much water. AquaScope's choice, stated wherever
# the lens is shown: deeper than the 1 cm that reads as wet, so a film does not count as arrival.
ARRIVAL_DEPTH_M = 0.1

# Above this many cell values (steps x cells), the hazard is taken from the largest depth and the largest speed
# over the run (which may not happen together, so it can only overstate) instead of step by step.
HAZARD_BUDGET = 40_000_000

R_MERC = 6378137.0   # the Web Mercator sphere


def metres_per(unit: str | None) -> float | None:
    """Metres in one of a file's length (or speed) units: 1 for metres, 0.3048 for feet, None when unstated."""
    u = str(unit or "").strip().lower()
    if not u:
        return None
    if u.startswith(("ft", "feet", "foot", "us survey foot")):
        return 0.3048
    if u.startswith(("m", "meter", "metre")):
        return 1.0
    return None


def hazard_class(depth_m: np.ndarray, velocity_ms: np.ndarray) -> np.ndarray:
    """The hazard class (1 to 6 for H1 to H6) of each depth and velocity pair, in metres and metres a second;
    0 where it is dry (depth below 1 cm, or not a number) or the velocity is missing."""
    d = np.asarray(depth_m, dtype="f8")
    v = np.abs(np.asarray(velocity_ms, dtype="f8"))
    with np.errstate(invalid="ignore"):
        cls = np.full(d.shape, 6, dtype="i1")
        for k in range(len(HAZARD_CLASSES) - 2, -1, -1):   # H5 down to H1: the limits are nested
            c = HAZARD_CLASSES[k]
            ok = (d * v <= c["dv"]) & (d <= c["depth"]) & (v <= c["velocity"])
            cls[ok] = k + 1
        cls[~(d >= DRY_DEPTH) | ~np.isfinite(v)] = 0
    return cls


def _hours(times: list[str]) -> np.ndarray:
    if not times:
        return np.zeros(0)
    t0 = datetime.fromisoformat(times[0])
    return np.array([(datetime.fromisoformat(t) - t0).total_seconds() / 3600.0 for t in times])


@dataclass
class Lenses:
    """The event-wide lenses of one mesh result, one value per cell (NaN, or -1 for a step, where none)."""

    hazard: np.ndarray | None        # int8, 0 dry, 1..6 for H1..H6, the highest class over the run
    hazard_dv: np.ndarray | None     # D x V (m2/s) at the step that class was reached with the most D x V
    hazard_depth: np.ndarray | None  # depth (m) at that step
    hazard_speed: np.ndarray | None  # speed (m/s) at that step
    hazard_step: np.ndarray | None   # int32
    arrival_h: np.ndarray | None     # hours after the first output step; 0 for a cell wet from the start
    arrival_step: np.ndarray | None  # int32
    max_depth: np.ndarray            # the reader's largest depth (or water surface), file units
    max_depth_step: np.ndarray       # int32, the output step with the most (-1 without time steps)
    max_speed: np.ndarray | None     # the reader's largest speed, file units
    max_speed_step: np.ndarray | None
    meta: dict[str, Any]

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for name in ("hazard", "hazard_dv", "hazard_depth", "hazard_speed", "hazard_step", "arrival_h",
                     "arrival_step", "max_depth", "max_depth_step", "max_speed", "max_speed_step"):
            a = getattr(self, name)
            if a is not None:
                out[name] = a
        return out


def mesh_lenses(m: Mesh, *, arrival_depth_m: float = ARRIVAL_DEPTH_M, budget: int = HAZARD_BUDGET) -> Lenses:
    """The four lenses of a flood result: hazard class, arrival time, largest depth and largest speed.

    The hazard is classified at every output step from the depth and the speed there (AIDR Guideline 7-3, Tables
    1 and 2) and the highest class over the run is kept, with the step and the numbers that put the cell there.
    Above ``budget`` cell values it is classified once from the largest depth and the largest speed, and a note
    says so. Depth and speed are converted to metres (a file in feet is converted; a file with no units is read
    as metres and a note says so). The arrival time is the first output step with at least ``arrival_depth_m``
    of water, in hours after the run's first output step. A file without bed levels (water surface only) has
    no hazard and no arrival time; a file without velocities has no hazard.
    """
    n, steps = m.n_cells, len(m.times)
    notes: list[str] = []
    depth_file = m.variable == "depth"
    length_unit = m.units.get("depth") if depth_file else m.units.get("water_surface")
    to_m = metres_per(length_unit)
    v_to_m = metres_per(str(m.units.get("velocity") or "").split("/")[0])
    if depth_file and to_m is None:
        notes.append("The file does not state its length unit: depths are read as metres.")
        to_m = 1.0
    if m.has_velocity and v_to_m is None:
        notes.append("The file does not state its velocity unit: speeds are read as metres a second.")
        v_to_m = 1.0
    hours = _hours(m.times)

    max_depth = np.asarray(m.max_value, dtype="f4")
    max_depth_step = np.full(n, -1, dtype="i4")
    max_speed = None if m.max_speed is None else np.asarray(m.max_speed, dtype="f4")
    max_speed_step = None if m.max_speed is None else np.full(n, -1, dtype="i4")
    arrival_step = arrival_h = None
    hazard = hazard_dv = hazard_depth = hazard_speed = hazard_step = None
    available = ["depth", "max_depth"]
    if max_speed is not None:
        available.append("max_speed")

    if steps:
        thr = arrival_depth_m / to_m if (depth_file and to_m) else None
        best = np.full(n, -np.inf)
        first = np.full(n, -1, dtype="i4")
        for i in range(0, steps, 32):
            block = m.values_block(i, min(i + 32, steps))
            with np.errstate(invalid="ignore"):
                vals = np.where(np.isfinite(block), block, -np.inf)
                k = np.argmax(vals, axis=0)
                top = vals[k, np.arange(n)]
                better = top > best
                best[better] = top[better]
                max_depth_step[better] = i + k[better]
                if thr is not None:
                    reached = block >= thr
                    anyr = reached.any(axis=0) & (first < 0)
                    first[anyr] = i + np.argmax(reached[:, anyr], axis=0)
        if thr is not None:
            arrival_step = first
            arrival_h = np.where(first >= 0, hours[np.maximum(first, 0)], np.nan).astype("f4")
            available.append("arrival")
    if depth_file and m.has_velocity and steps:
        hazard = np.zeros(n, dtype="i1")
        hazard_dv = np.full(n, np.nan, dtype="f4")
        hazard_depth = np.full(n, np.nan, dtype="f4")
        hazard_speed = np.full(n, np.nan, dtype="f4")
        hazard_step = np.full(n, -1, dtype="i4")
        if steps * n <= budget:
            sp_best = np.full(n, -np.inf)
            for k in range(steps):
                f = m.frame(k)
                d = f["value"] * to_m
                sp = np.hypot(f["u"], f["v"]) if f["u"] is not None else np.full(n, np.nan)
                spm = sp * v_to_m
                cls = hazard_class(d, spm)
                with np.errstate(invalid="ignore"):
                    dv = d * spm
                    up = (cls > hazard) | ((cls == hazard) & (cls > 0) & (dv > np.nan_to_num(hazard_dv, nan=-1)))
                    hazard[up] = cls[up]
                    hazard_dv[up] = dv[up]
                    hazard_depth[up] = d[up]
                    hazard_speed[up] = spm[up]
                    hazard_step[up] = k
                    if max_speed_step is not None:
                        fast = np.isfinite(sp) & (sp > sp_best)
                        sp_best[fast] = sp[fast]
                        max_speed_step[fast] = k
            how = "step"
        else:
            d = max_depth.astype("f8") * to_m
            spm = max_speed.astype("f8") * v_to_m
            hazard = hazard_class(d, spm)
            with np.errstate(invalid="ignore"):
                hazard_dv = (d * spm).astype("f4")
            hazard_depth, hazard_speed = d.astype("f4"), spm.astype("f4")
            hazard_dv[hazard == 0] = np.nan
            how = "largest"
            notes.append("The run is too long to classify every step here: the hazard comes from the largest depth "
                         "and the largest speed in each cell, which may not happen together, so it can overstate.")
        available.insert(1, "hazard")
    elif depth_file and not m.has_velocity:
        notes.append("The file has no velocities, so the flood hazard (which needs depth and speed) is not shown.")
        how = None
    else:
        how = None
    if not depth_file:
        notes.append("The file has no bed levels, only the water surface: no depth, so no hazard and no arrival time.")

    area = m.cell_areas()
    meta: dict[str, Any] = {
        "available": available,
        "variable": m.variable,
        "units": {"length": length_unit, "velocity": m.units.get("velocity")},
        "hours": [round(float(h), 4) for h in hours],
        "arrival": {"depth_m": arrival_depth_m,
                    "depth": None if not (depth_file and to_m) else round(arrival_depth_m / to_m, 4),
                    "rule": f"the first output step with at least {arrival_depth_m:g} m of water"},
        "hazard": None,
        "notes": notes,
    }
    if hazard is not None:
        counts = np.bincount(hazard.astype(np.intp), minlength=7)
        areas = np.bincount(hazard.astype(np.intp), weights=area, minlength=7)
        meta["hazard"] = {
            "classes": [{**c, "cells": int(counts[k + 1]), "area_m2": round(float(areas[k + 1]), 1)}
                        for k, c in enumerate(HAZARD_CLASSES)],
            "source": HAZARD_SOURCE, "from": how, "units": {"depth": "m", "velocity": "m/s", "dv": "m2/s"},
            "rule": "the highest class reached at any output step" if how == "step"
            else "from the largest depth and the largest speed",
        }
    if arrival_h is not None and np.isfinite(arrival_h).any():
        fin = arrival_h[np.isfinite(arrival_h)]
        meta["arrival"].update({"first_h": round(float(fin.min()), 3), "last_h": round(float(fin.max()), 3),
                                "wet_at_start": int((arrival_step == 0).sum()),
                                "reached": int((arrival_step >= 0).sum())})
    return Lenses(hazard, hazard_dv, hazard_depth, hazard_speed, hazard_step, arrival_h, arrival_step, max_depth,
                  max_depth_step, max_speed, max_speed_step, meta)


# ── the smooth field: cell values on a regular grid ─────────────────────────


def _merc_y(lat: np.ndarray) -> np.ndarray:
    return R_MERC * np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85, 85)) / 2))


def _lat_of(y: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R_MERC)) - np.pi / 2)


@dataclass
class FieldGrid:
    """A regular Web Mercator grid over a mesh, with each pixel inside the mesh tied to three cell centres.

    The image runs from the north-west corner, row by row (``height`` rows of ``width`` pixels). ``pix`` are the
    flat indices of the pixels inside the mesh, ``tri`` their three cells and ``w`` the linear weights. The
    corners are (west, north), (east, north), (east, south), (west, south) in degrees.
    """

    west: float
    east: float
    south: float
    north: float
    width: int
    height: int
    pix: np.ndarray
    tri: np.ndarray
    w: np.ndarray
    n_cells: int
    px_per_cell: float = 5.0

    def frame(self) -> dict[str, Any]:
        return {"west": self.west, "east": self.east, "south": self.south, "north": self.north,
                "width": self.width, "height": self.height,
                "corners": [[self.west, self.north], [self.east, self.north], [self.east, self.south],
                            [self.west, self.south]]}

    def lonlat(self, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (0 at the north-west corner of the image, in pixels) as longitude and latitude."""
        lon = self.west + (np.asarray(col) / self.width) * (self.east - self.west)
        yn, ys = _merc_y(np.array(self.north)), _merc_y(np.array(self.south))
        lat = _lat_of(yn - (np.asarray(row) / self.height) * (yn - ys))
        return lon, lat


def _local_xy(lon: np.ndarray, lat: np.ndarray, lat0: float) -> np.ndarray:
    k = math.pi / 180 * 6371008.8
    return np.stack([np.asarray(lon) * k * math.cos(math.radians(lat0)), np.asarray(lat) * k], 1)


def _inside(points: np.ndarray, cand: np.ndarray, m: Mesh, lat0: float, chunk: int = 250_000) -> np.ndarray:
    """For each point, whether it lies inside any of its candidate cells (even-odd rule on the outlines)."""
    off = np.asarray(m.ring_offsets)
    counts = np.diff(off)
    nv = int(counts.max()) if counts.size else 0
    ring = _local_xy(m.ring_lon, m.ring_lat, lat0)
    # Outlines padded to nv points by repeating their last point (a zero-length edge crosses nothing).
    k = np.minimum(np.arange(nv)[None, :], (counts - 1)[:, None])
    idx = off[:-1, None] + k
    rx, ry = ring[idx, 0], ring[idx, 1]
    out = np.zeros(points.shape[0], dtype=bool)
    size = max(1, chunk // max(1, cand.shape[1]))
    for s in range(0, points.shape[0], size):
        p = points[s:s + size]
        c = cand[s:s + size]
        hit = np.zeros(p.shape[0], dtype=bool)
        for j in range(c.shape[1]):
            xs, ys = rx[c[:, j]], ry[c[:, j]]          # (P, nv)
            xa, ya = xs, ys
            xb, yb = np.roll(xs, -1, axis=1), np.roll(ys, -1, axis=1)
            py, px = p[:, 1:2], p[:, 0:1]
            with np.errstate(invalid="ignore", divide="ignore"):
                cross = ((ya > py) != (yb > py)) & (px < xa + (py - ya) * (xb - xa) / (yb - ya))
            hit |= (cross.sum(axis=1) % 2) == 1
        out[s:s + size] = hit
    return out


def field_grid(m: Mesh, *, px_per_cell: float = 5.0, max_pixels: int = 1_000_000, max_side: int = 2048) -> FieldGrid:
    """The grid the smooth field is drawn on: about ``px_per_cell`` pixels across a typical cell, at most
    ``max_pixels`` pixels and ``max_side`` on the long side. Built once per mesh; each step is then a weighted
    sum (:func:`smooth_field`)."""
    from scipy.spatial import Delaunay, QhullError, cKDTree

    n = m.n_cells
    if n == 0:
        raise MeshError("This mesh has no cells.")
    w0, s0, e0, n0 = m.bbox()
    lat0 = (s0 + n0) / 2
    area = m.cell_areas()
    cell_m = float(np.sqrt(np.median(area[area > 0]))) if (area > 0).any() else 1.0
    pad = max(e0 - w0, n0 - s0) * 0.002
    west, east, south, north = w0 - pad, e0 + pad, s0 - pad, n0 + pad
    xm0, xm1 = math.radians(west) * R_MERC, math.radians(east) * R_MERC
    ym0, ym1 = float(_merc_y(np.array(south))), float(_merc_y(np.array(north)))
    px_m = cell_m / px_per_cell / math.cos(math.radians(lat0))   # one pixel, in Web Mercator metres
    width, height = max(2, math.ceil((xm1 - xm0) / px_m)), max(2, math.ceil((ym1 - ym0) / px_m))
    scale = max(1.0, math.sqrt(width * height / max_pixels), max(width, height) / max_side)
    width, height = max(2, int(width / scale)), max(2, int(height / scale))
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    lon = west + cols.ravel() / width * (east - west)
    lat = _lat_of(ym1 - rows.ravel() / height * (ym1 - ym0))
    pts = _local_xy(lon, lat, lat0)
    centres = _local_xy(m.lon, m.lat, lat0)
    tree = cKDTree(centres)
    # Only points no further from a centre than that cell's farthest corner can be inside a cell.
    off = np.asarray(m.ring_offsets)
    corner = _local_xy(m.ring_lon, m.ring_lat, lat0)
    owner = np.repeat(np.arange(n), np.diff(off))
    reach = np.zeros(n)
    np.maximum.at(reach, owner, np.hypot(*(corner - centres[owner]).T))
    kk = min(4, n)
    dist, cand = tree.query(pts, k=kk)
    if kk == 1:
        dist, cand = dist[:, None], cand[:, None]
    near = dist[:, 0] <= reach.max() * 1.0001
    sel = np.flatnonzero(near)
    ins = _inside(pts[sel], cand[sel].astype(np.intp), m, lat0)
    pix = sel[ins]
    p = pts[pix]
    nearest = cand[pix, 0].astype(np.intp)
    tri = np.repeat(nearest[:, None], 3, axis=1)
    w = np.zeros((pix.size, 3), dtype="f4")
    w[:, 0] = 1.0
    if n >= 3:
        try:
            dl = Delaunay(centres)
            s = dl.find_simplex(p)
            ok = s >= 0
            tr = dl.transform[s[ok]]
            b = np.einsum("ijk,ik->ij", tr[:, :2, :], p[ok] - tr[:, 2, :])
            bw = np.column_stack([b, 1 - b.sum(axis=1)])
            tri[ok] = dl.simplices[s[ok]]
            w[ok] = np.clip(bw, 0, 1).astype("f4")
        except QhullError:   # centres all on a line: each pixel takes its own cell's value
            pass
    return FieldGrid(west, east, south, north, int(width), int(height), pix.astype("i4"), tri.astype("i4"), w, n,
                     px_per_cell / scale)


def smooth_field(grid: FieldGrid, values: np.ndarray, *, kind: str = "depth",
                 depth: np.ndarray | None = None, raw: bool = False) -> np.ndarray:
    """A per-cell array on the grid, as a (height, width) float32 image, NaN where nothing is drawn.

    ``kind`` says how a missing (NaN) cell counts: ``"depth"`` takes it as zero (a dry cell) and draws only where
    the interpolated depth is at least 1 cm; ``"speed"`` takes it as zero and draws where ``depth`` (the matching
    per-cell depth) would be drawn; ``"strict"`` (water surface, arrival time) interpolates where all three cells
    have a value and elsewhere takes the value of the pixel's own cell (NaN when it has none). ``raw`` returns the
    depth before the 1 cm cut, NaN only outside the mesh (for the wet edge)."""
    v = np.asarray(values, dtype="f4")
    if v.size != grid.n_cells:
        raise MeshError(f"{v.size} values for a mesh of {grid.n_cells} cells.")
    g = v[grid.tri]
    filled = np.where(np.isfinite(g), g, 0.0)
    out = (grid.w * filled).sum(axis=1)
    if kind == "depth":
        if not raw:
            out[~(out >= DRY_DEPTH)] = np.nan
    elif kind == "speed":
        if depth is None:
            raise MeshError("A speed field needs the depth it goes with.")
        dg = np.asarray(depth, dtype="f4")[grid.tri]
        dd = (grid.w * np.where(np.isfinite(dg), dg, 0.0)).sum(axis=1)
        out[~(dd >= DRY_DEPTH)] = np.nan
    elif kind == "strict":
        # Next to a cell without a value, a pixel takes its own cell's (the nearest centre of its three), so the gap
        # keeps the cell's shape instead of eating every triangle that touches it.
        bad = (~np.isfinite(g) & (grid.w > 1e-6)).any(axis=1)
        own = g[np.arange(g.shape[0]), np.argmax(grid.w, axis=1)]
        out[bad] = own[bad]
    else:
        raise MeshError(f"Unknown field kind {kind!r}.")
    img = np.full(grid.width * grid.height, np.nan, dtype="f4")
    img[grid.pix] = out
    return img.reshape(grid.height, grid.width)


# ── contour lines ───────────────────────────────────────────────────────────

_NICE = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
_NICE_HOURS = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0, 168.0)


def contour_levels(lo: float, hi: float, *, most: int = 6, hours: bool = False) -> list[float]:
    """Round levels strictly between ``lo`` and ``hi``: the smallest interval from 1, 2, 2.5, 5 times a power of
    ten (or 0.25 h, 0.5 h, 1, 2, 3, 6, 12 h, a day ... for times) that gives at most ``most`` lines."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    for step in (_NICE_HOURS if hours else _NICE):
        first = math.floor(lo / step + 1e-9) + 1
        levels = [round(k * step, 6) for k in range(first, int(math.ceil(hi / step - 1e-9)))]
        if len(levels) <= most:
            return [x for x in levels if lo < x < hi]
    step = 10 ** math.ceil(math.log10((hi - lo) / most))
    return [round(k * step, 6) for k in range(math.floor(lo / step) + 1, int(math.ceil(hi / step)))]


def _label(level: float, unit: str | None) -> str:
    txt = f"{level:g}"
    return f"{txt} {unit}" if unit else txt


def blur(image: np.ndarray, radius: int) -> np.ndarray:
    """A box blur of ``radius`` pixels that leaves NaN where the image is NaN and does not spread it (a mean of the
    values around each pixel that have one). For tracing lines on a field that steps cell by cell."""
    if radius < 1:
        return image
    v = np.asarray(image, dtype="f8")
    ok = np.isfinite(v)
    acc, cnt = np.where(ok, v, 0.0), ok.astype("f8")
    k = 2 * radius + 1
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        ca, cc = np.cumsum(np.pad(acc, pad), axis=axis), np.cumsum(np.pad(cnt, pad), axis=axis)
        sl_hi = [slice(None), slice(None)]
        sl_lo = [slice(None), slice(None)]
        sl_hi[axis], sl_lo[axis] = slice(k, None), slice(0, -k)
        acc, cnt = ca[tuple(sl_hi)] - ca[tuple(sl_lo)], cc[tuple(sl_hi)] - cc[tuple(sl_lo)]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = acc / cnt
    out[~ok] = np.nan
    return out


def contour_lines(grid: FieldGrid, image: np.ndarray, levels: list[float], *, unit: str | None = None,
                  kind: str = "level", digits: int = 6, trace_at: list[float] | None = None,
                  min_px: float = 0.0) -> dict[str, Any]:
    """Lines of equal value on a grid image (from :func:`smooth_field`) as a GeoJSON FeatureCollection in degrees:
    one MultiLineString per level, with ``level``, ``label`` (the level and its unit) and ``kind``. ``trace_at``
    (one value per level) traces each line at another value than it is labelled with; pieces shorter than
    ``min_px`` pixels are left out. Needs contourpy (with matplotlib, or in the browser); without it, no lines
    and ``note`` says why."""
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": []}
    if not levels:
        return fc
    try:
        import contourpy
    except ImportError:
        fc["note"] = "Contour lines need contourpy (pip install contourpy)."
        return fc
    z = np.ma.masked_invalid(np.asarray(image, dtype="f8"))
    if z.mask.all():
        return fc
    rows, cols = image.shape
    gen = contourpy.contour_generator(x=np.arange(cols) + 0.5, y=np.arange(rows) + 0.5, z=z,
                                      line_type=contourpy.LineType.Separate)
    for n_level, level in enumerate(levels):
        parts = []
        for seg in gen.lines(trace_at[n_level] if trace_at else level):
            if len(seg) < 2 or (min_px and float(np.hypot(*np.diff(seg, axis=0).T).sum()) < min_px):
                continue
            lon, lat = grid.lonlat(seg[:, 0], seg[:, 1])
            parts.append(np.round(np.column_stack([lon, lat]), digits).tolist())
        if parts:
            fc["features"].append({"type": "Feature",
                                   "properties": {"level": level, "label": _label(level, unit), "kind": kind},
                                   "geometry": {"type": "MultiLineString", "coordinates": parts}})
    return fc


def wet_edge(grid: FieldGrid, values: np.ndarray, *, digits: int = 6) -> dict[str, Any]:
    """The edge of the water (depth 1 cm on the smooth field) as a GeoJSON FeatureCollection, ``kind`` "edge"."""
    raw = smooth_field(grid, values, kind="depth", raw=True)
    fc = contour_lines(grid, raw, [DRY_DEPTH], kind="edge", digits=digits)
    for f in fc["features"]:
        f["properties"]["label"] = ""
    return fc


# ── for the Explorer (mesh-worker.js) ───────────────────────────────────────


def depth_levels(m: Mesh, lenses: Lenses | None = None) -> list[float]:
    """The depth contour levels of a mesh, the same at every step: a round interval chosen so the depths most of the
    mesh reaches (up to their 90th percentile) get about five lines, carried on up to the deepest water but never
    more than ten lines, so a deep channel does not bury a shallow floodplain's lines or stack its own."""
    mv = np.asarray(m.max_value, dtype="f8")
    fin = mv[np.isfinite(mv)]
    if not fin.size or m.variable != "depth":
        return []
    bulk = contour_levels(0.0, float(np.quantile(fin, 0.9)), most=5)
    top = float(np.quantile(fin, 0.995))
    if len(bulk) < 2:
        return contour_levels(0.0, top, most=6)
    step = bulk[1] - bulk[0]
    return [round(k * step, 6) for k in range(1, 11) if k * step < top]


def views_payload(m: Mesh, lenses: Lenses, grid: FieldGrid) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The lenses for the Explorer: their metadata (with the grid's frame and the contour levels) and their
    per-cell arrays as float32/int32/int8, ready to become typed arrays."""
    meta = json.loads(json.dumps(lenses.meta))
    meta["grid"] = grid.frame()
    meta["depth_levels"] = depth_levels(m, lenses)
    if lenses.arrival_h is not None and np.isfinite(lenses.arrival_h).any():
        a = lenses.arrival_h[np.isfinite(lenses.arrival_h)]
        meta["arrival_levels"] = contour_levels(float(a.min()), float(a.max()), most=8, hours=True)
    else:
        meta["arrival_levels"] = []
    arrays = {}
    for k, a in lenses.arrays().items():
        arrays[k] = a.astype("i1" if k == "hazard" else "i4" if k.endswith("_step") else "f4")
    return meta, arrays


def field_payload(m: Mesh, grid: FieldGrid, lenses: Lenses, what: str = "depth", step: int | None = None
                  ) -> dict[str, Any]:
    """One smooth picture for the Explorer: ``image`` (a float32 array of height x width, NaN where nothing) and
    ``lines`` (GeoJSON text: the contour lines, and for depth the wet edge). ``what`` is ``"depth"`` (at
    ``step``, or the largest over the run when step is None), ``"max_depth"``, ``"max_speed"`` or ``"arrival"``."""
    unit = m.units.get(m.variable)
    # Linear pieces on a triangulation crease along the triangles' edges; a blur of about half a cell (which does
    # not move the wet edge) takes the creases out of the picture and the lines.
    soft = max(1, round(grid.px_per_cell / 2.5))
    if what in ("depth", "max_depth"):
        vals = m.frame(step)["value"] if (what == "depth" and step is not None) else m.max_value
        if m.variable == "depth":
            img = blur(smooth_field(grid, vals, kind="depth"), soft)
            lines = contour_lines(grid, img, depth_levels(m, lenses), unit=unit, kind="depth")
            lines["features"] += wet_edge(grid, vals)["features"]
        else:
            img = smooth_field(grid, vals, kind="strict")
            lines = {"type": "FeatureCollection", "features": []}
    elif what == "max_speed":
        if lenses.max_speed is None:
            raise MeshError("This file has no velocities.")
        img = blur(smooth_field(grid, lenses.max_speed, kind="speed", depth=m.max_value) if m.variable == "depth"
                   else smooth_field(grid, lenses.max_speed, kind="strict"), soft)
        lines = {"type": "FeatureCollection", "features": []}
    elif what == "arrival":
        if lenses.arrival_h is None:
            raise MeshError("This file has no arrival time (it needs depths and time steps).")
        img = smooth_field(grid, lenses.arrival_h, kind="strict")
        a = lenses.arrival_h[np.isfinite(lenses.arrival_h)]
        levels = contour_levels(float(a.min()), float(a.max()), most=8, hours=True) if a.size else []
        # A line no place reaches since the line before would lie on that one (the water jumps from 0 h to 8 h
        # across a bank, say, and 2, 4 and 6 h would stack there): only the levels some cell arrives by are drawn.
        if len(levels) > 1:
            gap = levels[1] - levels[0]
            levels = [lv for lv in levels if bool(((a > lv - gap) & (a <= lv)).any())]
        # Arrival comes in output steps, so the field is flat between them: the line "flooded by 2 h" is traced
        # halfway to the next output time, on the field softened by about a cell, and crumbs are left out.
        hours = _hours(m.times)
        half = float(np.median(np.diff(hours))) / 2 if hours.size > 1 else 0.0
        px = max(1, round(grid.px_per_cell / 2))
        lines = contour_lines(grid, blur(img, px), levels, unit="h", kind="arrival",
                              trace_at=[lv + half for lv in levels], min_px=4 * grid.px_per_cell)
    else:
        raise MeshError(f"Unknown picture {what!r}.")
    return {"image": img.astype("f4").ravel(), "lines": json.dumps(lines, separators=(",", ":"))}


# ── for the CLI and the MCP tool ────────────────────────────────────────────


def lens_summary(path: str, crs: str | int | None = None, geometry: str | None = None,
                 arrival_depth_m: float = ARRIVAL_DEPTH_M) -> dict[str, Any]:
    """The lenses of a flood result on this machine, in numbers: the area and cells in each hazard class (AIDR
    Guideline 7-3) with its source, the arrival times (first and last, and how many cells were wet from the
    start), and the largest depth and speed with the model time and place of each. ok is false with a plain
    error for a file the readers cannot open."""
    try:
        m = read_mesh(path, crs=crs, geometry=geometry)
    except MeshError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        lz = mesh_lenses(m, arrival_depth_m=arrival_depth_m)
        out: dict[str, Any] = {"ok": True, "name": m.name, "model": m.model, "cells": m.n_cells,
                               "time_steps": len(m.times), "variable": m.variable,
                               "units": lz.meta["units"], "available": lz.meta["available"], "notes": lz.meta["notes"]}
        if lz.meta["hazard"]:
            h = lz.meta["hazard"]
            out["hazard"] = {"source": h["source"]["short"], "url": h["source"]["url"], "rule": h["rule"],
                             "classes": [{"code": c["code"], "words": c["words"], "cells": c["cells"],
                                          "area_km2": round(c["area_m2"] / 1e6, 4)} for c in h["classes"]]}
        else:
            out["hazard"] = None
        arr = lz.meta["arrival"]
        out["arrival"] = {k: arr.get(k) for k in ("rule", "first_h", "last_h", "wet_at_start", "reached")}

        def where(values: np.ndarray | None, steps: np.ndarray | None, unit: str | None) -> dict[str, Any] | None:
            if values is None or not np.isfinite(values).any():
                return None
            i = int(np.nanargmax(values))
            k = int(steps[i]) if steps is not None and steps[i] >= 0 else None
            return {"value": round(float(values[i]), 3), "unit": unit, "cell": i,
                    "lon": round(float(m.lon[i]), 5), "lat": round(float(m.lat[i]), 5),
                    "time": m.times[k] if k is not None else None}

        out["max_depth"] = where(lz.max_depth, lz.max_depth_step, m.units.get(m.variable))
        out["max_speed"] = where(lz.max_speed, lz.max_speed_step, m.units.get("velocity"))
        return out
    finally:
        m.close()
