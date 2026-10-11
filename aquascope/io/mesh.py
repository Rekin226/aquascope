"""Model results on a mesh: HEC-RAS 2D plan files and UGRID NetCDF (#555).

The Crayfish view of a model run, read without a GIS: the cells of the mesh in
longitude and latitude, the model's own times, the depth (or the water surface
where the file has no bed levels), the velocity where the file has it, and the
largest depth and speed reached in each cell.

Two readers return the same slim structure, :class:`Mesh`:

* :func:`read_ras_plan` reads a HEC-RAS plan HDF5 file (``*.p01.hdf``) with
  h5py: the 2D flow areas' cells, the water surface and face velocity time
  series, and the summary maxima. Cell velocities are rebuilt from the face
  normal velocities by least squares, as RAS Mapper does. A stripped plan
  file that holds the results but not the mesh (a ``*.p01.timeseries.hdf``)
  is read with the geometry file it was run with (``geometry=``, a
  ``*.g01.hdf``); without one, the error says what is missing and that.
* :func:`read_ugrid` reads a UGRID 2D mesh (D-Flow FM, ADH, ADCIRC and others):
  NetCDF-4 files with h5py, NetCDF-3 classic files with scipy.

Both read lazily: the geometry and the maxima when the file is opened, one time
step (:meth:`Mesh.frame`) or one cell (:meth:`Mesh.series`) when asked. That is
what lets the Explorer open a dropped file of hundreds of MB in the browser
without copying it into memory (the worker mounts it with WORKERFS).

Coordinates are reprojected to longitude and latitude with pyproj from the
coordinate system the file states (RAS Mapper's projection, a UGRID grid
mapping or EPSG code). A file that states none needs ``crs=`` (an EPSG code or
WKT); a UGRID file in degrees needs nothing.

:func:`mesh_summary` is the small answer for the MCP server and the CLI
(``aquascope mesh FILE``). Every number in it is computed here.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# The largest file the readers accept, and the largest mesh they draw. Reads are lazy, so the file size costs
# little; what costs memory is the mesh itself (its cells, their rings and two arrays per time step).
MAX_FILE_BYTES = 2 * 1024**3
MAX_CELLS = 250_000
# Index arrays are np.intp throughout: 32 bits in the browser (Pyodide is wasm32), where numpy refuses to cast
# int64 counts or indices down for np.repeat and np.bincount.
# A cell is dry below this depth, in the file's own length unit.
DRY_DEPTH = 0.01
# Above this many face values (steps x faces), the RAS cell speed maximum is taken from the summary output instead
# of rebuilt step by step.
SPEED_BUDGET = 60_000_000

RAS_TS = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
RAS_SUMMARY = "Results/Unsteady/Output/Output Blocks/Base Output/Summary Output"
RAS_GEOM = "Geometry/2D Flow Areas"

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class MeshError(ValueError):
    """A file the mesh readers cannot show, with a message meant for the person who dropped it."""


@dataclass
class Mesh:
    """A model result on a mesh, in longitude and latitude.

    ``lon``/``lat`` are the cell centres; the cell outlines are rings packed in ``ring_lon``/``ring_lat``,
    cell ``i`` running from ``ring_offsets[i]`` to ``ring_offsets[i + 1]`` (not closed). ``variable`` is
    ``"depth"`` or ``"water_surface"`` (a file without bed levels); ``max_value`` is its largest value in each
    cell over the run, ``max_speed`` the largest speed (None without velocities). ``times`` are ISO strings
    as the file gives them (no time zone).
    """

    format: str
    name: str
    size_bytes: int
    model: str
    title: str
    crs: dict[str, Any]
    units: dict[str, str | None]
    areas: list[dict[str, Any]]
    lon: np.ndarray
    lat: np.ndarray
    ring_offsets: np.ndarray
    ring_lon: np.ndarray
    ring_lat: np.ndarray
    times: list[str]
    variable: str
    max_value: np.ndarray
    max_speed: np.ndarray | None
    notes: list[str] = field(default_factory=list)
    _source: Any = field(default=None, repr=False)

    @property
    def n_cells(self) -> int:
        return int(self.lon.size)

    @property
    def has_velocity(self) -> bool:
        return self.max_speed is not None

    def frame(self, step: int) -> dict[str, np.ndarray | None]:
        """One time step: ``value`` (depth or water surface per cell, NaN where dry) and ``u``/``v`` (the
        eastward and northward velocity per cell, None without velocities)."""
        if not self.times:
            raise MeshError("This file has no time steps, only its maxima.")
        step = int(step)
        if not 0 <= step < len(self.times):
            raise MeshError(f"Time step {step} is outside the model's {len(self.times)} steps.")
        return self._source.frame(step)

    def series(self, cell: int) -> dict[str, Any]:
        """The time series at one cell: ``value`` (depth or water surface) and ``speed`` (None without
        velocities), one value per time step, NaN where dry."""
        cell = int(cell)
        if not 0 <= cell < self.n_cells:
            raise MeshError(f"Cell {cell} is not in this mesh ({self.n_cells} cells).")
        out = self._source.series(cell) if self.times else {"value": np.array([]), "speed": None}
        return {"times": list(self.times), **out}

    def close(self) -> None:
        if self._source is not None:
            self._source.close()

    def cell_areas(self) -> np.ndarray:
        """The plan area of each cell in square metres, from its outline (shoelace on a local flat projection)."""
        off = self.ring_offsets
        lat0 = np.radians(np.repeat(self.lat, np.diff(off)))
        x = np.radians(self.ring_lon) * np.cos(lat0) * 6371008.8
        y = np.radians(self.ring_lat) * 6371008.8
        nxt = np.arange(x.size) + 1
        nxt[off[1:] - 1] = off[:-1]
        cross = x * y[nxt] - x[nxt] * y
        cell = np.repeat(np.arange(self.n_cells), np.diff(off))
        return np.abs(np.bincount(cell, cross, self.n_cells)) / 2

    def peak_step(self) -> int | None:
        """The time step with the most water on the mesh: the largest depth times cell area summed over the cells
        (for a file without bed levels, the highest water surface over the wet area). None without time steps."""
        if not self.times:
            return None
        area = self.cell_areas()
        best, best_i = -np.inf, 0
        n = len(self.times)
        for i in range(0, n, 32):
            block = self._source.values_block(slice(i, min(i + 32, n)))
            if self.variable == "water_surface":
                block = block - np.nanmin(self.max_value)
            total = np.nansum(np.where(np.isfinite(block), block, 0.0) * area, axis=1)
            k = int(np.argmax(total))
            if total[k] > best:
                best, best_i = float(total[k]), i + k
        return best_i

    def values_block(self, start: int, stop: int, *, raw: bool = False) -> np.ndarray:
        """Steps ``start`` to ``stop`` (not included) as a (steps, cells) array: depth (or water surface) with the
        dry cells NaN, or with ``raw`` the values as the file gives them (a negative or tiny depth kept), which is
        what the checks in :mod:`aquascope.mesh_review` read."""
        if not self.times:
            raise MeshError("This file has no time steps, only its maxima.")
        start, stop = max(0, int(start)), min(len(self.times), int(stop))
        return np.atleast_2d(self._source.values_block(slice(start, stop), raw=raw))

    def bbox(self) -> list[float]:
        """[west, south, east, north] of the cell outlines, in degrees."""
        return [float(np.nanmin(self.ring_lon)), float(np.nanmin(self.ring_lat)),
                float(np.nanmax(self.ring_lon)), float(np.nanmax(self.ring_lat))]

    def meta(self) -> dict[str, Any]:
        """Everything about the mesh but its arrays, as plain JSON types."""
        return {
            "format": self.format, "name": self.name, "size_bytes": self.size_bytes, "model": self.model,
            "title": self.title, "crs": self.crs, "units": self.units, "areas": self.areas,
            "cells": self.n_cells, "times": self.times, "variable": self.variable,
            "has_velocity": self.has_velocity, "bbox": self.bbox(), "notes": list(self.notes),
        }


# ── shared helpers ──────────────────────────────────────────────────────────


def _text(value: Any) -> str:
    """An HDF5 or NetCDF attribute as text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip("\x00 ")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _text(value.reshape(-1)[0])
        return " ".join(_text(v) for v in value.reshape(-1))
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", "replace").strip("\x00 ")
    return str(value).strip()


def _check_size(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        raise MeshError(f"No such file: {p}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise MeshError(f"{p.name} is {size / 1024**3:.1f} GB; the mesh viewer reads files up to "
                        f"{MAX_FILE_BYTES // 1024**3} GB.")
    return int(size)


def _sniff(path: str | Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head == _HDF5_MAGIC:
        return "hdf5"
    if head[:3] == b"CDF":
        return "netcdf3"
    return "other"


def _h5py():
    from aquascope.utils.imports import require

    return require("h5py", feature="reading HEC-RAS and NetCDF-4 mesh results")


def _transformer(crs_in: Any):
    from aquascope.utils.imports import require

    pyproj = require("pyproj", feature="placing a model mesh on the globe")
    try:
        src = pyproj.CRS.from_user_input(crs_in)
    except Exception as exc:  # pyproj raises CRSError, a subclass of several bases
        raise MeshError(f"The coordinate system could not be read ({exc}).") from exc
    return src, pyproj.Transformer.from_crs(src, "EPSG:4326", always_xy=True)


def _given_crs(crs: str | int | None) -> Any:
    if crs is None or crs == "":
        return None
    text = str(crs).strip()
    if re.fullmatch(r"\d{4,6}", text):
        return f"EPSG:{text}"
    return text


def _crs_label(src: Any) -> str:
    epsg = None
    try:
        epsg = src.to_epsg(min_confidence=70)
    except Exception:  # an unusual WKT pyproj cannot match is still usable
        epsg = None
    name = getattr(src, "name", "") or "unnamed"
    return f"{name} (EPSG:{epsg})" if epsg else name


def _to_lonlat(x: np.ndarray, y: np.ndarray, crs_in: Any, origin: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reproject arrays to longitude and latitude; returns them with the CRS description."""
    src, tr = _transformer(crs_in)
    lon, lat = tr.transform(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    lon, lat = np.asarray(lon, dtype="f8"), np.asarray(lat, dtype="f8")
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any() or np.nanmax(np.abs(lat[ok])) > 90:
        raise MeshError("The mesh does not land on the globe in the coordinate system given. Check the projection.")
    return lon, lat, {"label": _crs_label(src), "from": origin, "geographic": bool(src.is_geographic),
                      "wkt": src.to_wkt()}


def _east_north_angle(x: np.ndarray, y: np.ndarray, crs_in: Any, lat: np.ndarray) -> np.ndarray:
    """Per point, the angle (radians, counter-clockwise from east) of the projection's +x axis: grid convergence,
    so velocities in projected x and y can be turned into eastward and northward ones."""
    _src, tr = _transformer(crs_in)
    span = np.maximum(np.abs(np.asarray(x, dtype="f8")) * 1e-7, 1.0)
    lon0, lat0 = tr.transform(np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8"))
    lon1, lat1 = tr.transform(np.asarray(x, dtype="f8") + span, np.asarray(y, dtype="f8"))
    de = (np.asarray(lon1) - np.asarray(lon0)) * np.cos(np.radians(lat))
    dn = np.asarray(lat1) - np.asarray(lat0)
    return np.arctan2(dn, de)


def _rotate(u: np.ndarray, v: np.ndarray, angle: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if angle is None:
        return u, v
    c, s = np.cos(angle), np.sin(angle)
    return u * c - v * s, u * s + v * c


def _nanmax(a: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax without the warning on an all-NaN column (a cell dry all along)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=axis)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _units_of(system: str) -> dict[str, str | None]:
    s = system.lower()
    if "us" in s or "english" in s or "customary" in s:
        return {"length": "ft", "velocity": "ft/s"}
    if "si" in s or "metric" in s:
        return {"length": "m", "velocity": "m/s"}
    return {"length": None, "velocity": None}


# ── HEC-RAS ─────────────────────────────────────────────────────────────────

_RAS_STAMP = re.compile(r"(\d{1,2})([A-Za-z]{3})(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?(?::(\d{1,3}))?")
_MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                            "NOV", "DEC"])}


def parse_ras_time(text: str) -> datetime:
    """A RAS time stamp ("01JAN1999 12:00:00:000", "01Jan1999 24:00") as a datetime; 24:00 is the next day."""
    m = _RAS_STAMP.search(_text(text))
    if not m:
        raise MeshError(f"Unreadable HEC-RAS time stamp: {text!r}")
    day, mon, year, hh, mm, ss, ms = m.groups()
    month = _MONTHS.get(mon.upper())
    if not month:
        raise MeshError(f"Unreadable HEC-RAS time stamp: {text!r}")
    base = datetime(int(year), month, int(day))
    return base + timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss or 0), milliseconds=int(ms or 0))


class _RasSource:
    """Lazy reads of one RAS plan file: the time series of every 2D area, stitched into one cell numbering."""

    def __init__(self, fh, areas: list[dict[str, Any]], bed: np.ndarray | None, angle: np.ndarray | None):
        self.fh = fh
        self.areas = areas
        self.bed = bed
        self.angle = angle
        self.n_steps = 0

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:  # already closed
            pass

    def _ws(self, area: dict, sel) -> np.ndarray | None:
        ds = self.fh.get(f"{RAS_TS}/2D Flow Areas/{area['name']}/Water Surface")
        if ds is None:
            return None
        return np.asarray(ds[sel], dtype="f8")

    def _value(self, ws: np.ndarray, bed: np.ndarray | None) -> np.ndarray:
        if bed is None:
            return ws
        depth = ws - bed
        depth[~(depth > DRY_DEPTH)] = np.nan
        return depth

    def frame(self, step: int) -> dict[str, np.ndarray | None]:
        values, us, vs = [], [], []
        for a in self.areas:
            n = a["cells"]
            ws = self._ws(a, (step, slice(0, n)))
            ws = np.full(n, np.nan) if ws is None else ws
            bed = None if self.bed is None else self.bed[a["start"]:a["start"] + n]
            values.append(self._value(ws, bed))
            fv = self.fh.get(f"{RAS_TS}/2D Flow Areas/{a['name']}/Face Velocity")
            if fv is not None and a.get("lsq") is not None:
                u, v = a["lsq"].solve(np.asarray(fv[step, :], dtype="f8")[None, :])
                us.append(u[0])
                vs.append(v[0])
            else:
                us.append(np.full(n, np.nan))
                vs.append(np.full(n, np.nan))
        value = np.concatenate(values)
        if not any(a.get("lsq") is not None for a in self.areas):
            return {"value": value, "u": None, "v": None}
        u, v = _rotate(np.concatenate(us), np.concatenate(vs), self.angle)
        dry = ~np.isfinite(value) if self.bed is not None else np.zeros(value.size, bool)
        u[dry] = np.nan
        v[dry] = np.nan
        return {"value": value, "u": u, "v": v}

    def values_block(self, sl: slice, raw: bool = False) -> np.ndarray:
        """Depth (or water surface) for a block of steps: (steps, cells). ``raw`` keeps the depth as the file
        gives it (water surface less bed, negative or tiny included), without the dry cells set to NaN."""
        parts = []
        for a in self.areas:
            n = a["cells"]
            ws = self._ws(a, (sl, slice(0, n)))
            if ws is None:
                ws = np.full((len(range(*sl.indices(self.n_steps))), n), np.nan)   # an area without output
            bed = None if self.bed is None else self.bed[a["start"]:a["start"] + n]
            parts.append((ws if bed is None else ws - bed) if raw else self._value(ws, bed))
        return np.concatenate(parts, axis=1)

    def series(self, cell: int) -> dict[str, Any]:
        a = next(a for a in self.areas if a["start"] <= cell < a["start"] + a["cells"])
        i = cell - a["start"]
        ws = self._ws(a, (slice(None), i))
        if ws is None:
            return {"value": np.array([]), "speed": None}
        bed = None if self.bed is None else np.array([self.bed[cell]])
        value = self._value(ws.copy(), bed)
        speed = None
        fv = self.fh.get(f"{RAS_TS}/2D Flow Areas/{a['name']}/Face Velocity")
        if fv is not None and a.get("lsq") is not None:
            faces = a["lsq"].faces_of(i)
            if faces.size:
                vals = np.asarray(fv[:, faces.tolist()], dtype="f8")
                u, v = a["lsq"].solve_cell(i, faces, vals)
                speed = np.hypot(u, v)
                if self.bed is not None:
                    speed[~np.isfinite(value)] = np.nan
        return {"value": value, "speed": speed}


class _FaceLsq:
    """Cell velocity from face normal velocities: u_c minimises sum_f L_f (n_f . u_c - v_f)^2 over the cell's
    faces (weights L_f, the face lengths)."""

    def __init__(self, n_cells: int, cell_idx: np.ndarray, face_idx: np.ndarray, normals: np.ndarray,
                 lengths: np.ndarray):
        self.n = n_cells
        self.cell_idx = cell_idx
        self.face_idx = face_idx
        nx, ny = normals[face_idx, 0], normals[face_idx, 1]
        w = lengths[face_idx]
        self.wx, self.wy = w * nx, w * ny
        a = np.bincount(cell_idx, w * nx * nx, n_cells)
        b = np.bincount(cell_idx, w * nx * ny, n_cells)
        d = np.bincount(cell_idx, w * ny * ny, n_cells)
        det = a * d - b * b
        bad = ~(np.abs(det) > 1e-12 * np.maximum(a * d, 1e-30))
        det[bad] = np.nan
        self.ia, self.ib, self.id = d / det, -b / det, a / det
        order = np.argsort(cell_idx, kind="stable")
        self._sorted_faces = face_idx[order]
        self._starts = np.searchsorted(cell_idx[order], np.arange(n_cells + 1))
        self.normals, self.lengths = normals, lengths

    def solve(self, vn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """vn: (steps, faces) -> u, v: (steps, cells)."""
        steps = vn.shape[0]
        u = np.empty((steps, self.n))
        v = np.empty((steps, self.n))
        for t in range(steps):
            f = vn[t, self.face_idx]
            f = np.where(np.isfinite(f), f, 0.0)
            rx = np.bincount(self.cell_idx, self.wx * f, self.n)
            ry = np.bincount(self.cell_idx, self.wy * f, self.n)
            u[t] = self.ia * rx + self.ib * ry
            v[t] = self.ib * rx + self.id * ry
        return u, v

    def faces_of(self, cell: int) -> np.ndarray:
        return np.unique(self._sorted_faces[self._starts[cell]:self._starts[cell + 1]])

    def solve_cell(self, cell: int, faces: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """vals: (steps, len(faces)) face velocities of this cell's faces -> u, v per step."""
        n = self.normals[faces]
        w = self.lengths[faces]
        f = np.where(np.isfinite(vals), vals, 0.0)
        rx = (f * (w * n[:, 0])).sum(axis=1)
        ry = (f * (w * n[:, 1])).sum(axis=1)
        return self.ia[cell] * rx + self.ib[cell] * ry, self.ib[cell] * rx + self.id[cell] * ry


def _ras_rings(g, n: int, centers: np.ndarray, pairs_c: np.ndarray, pairs_f: np.ndarray,
               fp_xy: np.ndarray, face_fp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell outlines: RAS's own ordered face points where the file has them, else the cell's face points sorted
    by angle around its centre (RAS cells are convex)."""
    if "Cells FacePoint Indexes" in g:
        idx = np.asarray(g["Cells FacePoint Indexes"][:n])
        valid = idx >= 0
        counts = valid.sum(axis=1)
        flat = idx[valid]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)
        return offsets, fp_xy[flat, 0], fp_xy[flat, 1]
    pc = np.concatenate([pairs_c, pairs_c])
    pp = np.concatenate([face_fp[pairs_f, 0], face_fp[pairs_f, 1]])
    key = np.unique(pc.astype(np.int64) * (len(fp_xy) + 1) + pp)   # int64: cells x points passes 2**31
    pc, pp = (key // (len(fp_xy) + 1)).astype(np.intp), (key % (len(fp_xy) + 1)).astype(np.intp)
    ang = np.arctan2(fp_xy[pp, 1] - centers[pc, 1], fp_xy[pp, 0] - centers[pc, 0])
    order = np.lexsort((ang, pc))
    pc, pp = pc[order], pp[order]
    offsets = np.concatenate([[0], np.cumsum(np.bincount(pc, minlength=n))]).astype(np.intp)
    return offsets, fp_xy[pp, 0], fp_xy[pp, 1]


def _ras_area(fh, name: str, count: int, cell_points: np.ndarray | None, cell_info: np.ndarray | None,
              index: int) -> dict[str, Any]:
    g = fh[f"{RAS_GEOM}/{name}"]
    fp_xy = np.asarray(g["FacePoints Coordinate"][:], dtype="f8")
    face_fp = np.asarray(g["Faces FacePoint Indexes"][:], dtype=np.intp)
    info = np.asarray(g["Cells Face and Orientation Info"][:count], dtype=np.intp)
    vals = np.asarray(g["Cells Face and Orientation Values"][:], dtype=np.intp)
    counts = info[:, 1]
    pairs_c = np.repeat(np.arange(count), counts)
    first = np.cumsum(counts) - counts
    pairs_f = vals[np.repeat(info[:, 0] - first, counts) + np.arange(int(counts.sum()), dtype=np.intp), 0]
    if "Cells Center Coordinate" in g:
        centers = np.asarray(g["Cells Center Coordinate"][:count], dtype="f8")
    elif cell_points is not None and cell_info is not None:
        start, n = (int(x) for x in cell_info[index])
        centers = cell_points[start:start + min(n, count)]
    else:
        centers = np.stack([np.bincount(pairs_c, fp_xy[face_fp[pairs_f, 0], 0], count) / np.maximum(counts, 1),
                            np.bincount(pairs_c, fp_xy[face_fp[pairs_f, 0], 1], count) / np.maximum(counts, 1)], 1)
    offsets, rx, ry = _ras_rings(g, count, centers, pairs_c, pairs_f, fp_xy, face_fp)
    if "Faces NormalUnitVector and Length" in g:
        nl = np.asarray(g["Faces NormalUnitVector and Length"][:], dtype="f8")
        normals, lengths = nl[:, :2], nl[:, 2]
    else:  # RAS's normal is the face A -> B turned a quarter clockwise (checked against its own normals)
        ab = fp_xy[face_fp[:, 1]] - fp_xy[face_fp[:, 0]]
        lengths = np.hypot(ab[:, 0], ab[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            normals = np.stack([ab[:, 1], -ab[:, 0]], 1) / lengths[:, None]
    bed = np.asarray(g["Cells Minimum Elevation"][:count], dtype="f8") if "Cells Minimum Elevation" in g else None
    lsq = _FaceLsq(count, pairs_c, pairs_f, normals, lengths) if pairs_f.size else None
    return {"name": name, "cells": int(count), "centers": centers, "offsets": offsets, "rx": rx, "ry": ry,
            "bed": bed, "lsq": lsq, "faces": int(face_fp.shape[0])}


# What the reader needs of each 2D flow area's geometry (Geometry/2D Flow Areas/<name>/...).
_RAS_AREA_NEEDS = ("FacePoints Coordinate", "Faces FacePoint Indexes", "Cells Face and Orientation Info",
                   "Cells Face and Orientation Values")


def _ras_has_results(fh) -> bool:
    return "Results/Unsteady" in fh or "Results/Steady" in fh


def _ras_missing_geometry(fh) -> str | None:
    """The first part of the 2D mesh a RAS plan file lacks, as its HDF5 path; None when the mesh is all there.
    Only a file with results is asked: a file with neither is not a plan result at all."""
    if f"{RAS_GEOM}/Attributes" not in fh:
        return RAS_GEOM if "Geometry" not in fh or _ras_2d_results(fh) else None
    attrs = fh[f"{RAS_GEOM}/Attributes"][:]
    names = [_text(x) for x in attrs["Name"]] if "Name" in (attrs.dtype.names or ()) else []
    if not names:
        return f"{RAS_GEOM}/Attributes (the names of the 2D areas)"
    for name in names:
        if f"{RAS_GEOM}/{name}" not in fh:
            return f"{RAS_GEOM}/{name}"
        need = list(_RAS_AREA_NEEDS)
        if "Cell Count" not in attrs.dtype.names:
            need.append("Cells Center Coordinate")
        for part in need:
            if f"{RAS_GEOM}/{name}/{part}" not in fh:
                return f"{RAS_GEOM}/{name}/{part}"
    return None


def _ras_2d_results(fh) -> bool:
    return f"{RAS_TS}/2D Flow Areas" in fh or f"{RAS_SUMMARY}/2D Flow Areas" in fh


def _no_geometry(name: str, missing: str) -> MeshError:
    """The error for a plan file with results but not the mesh they sit on (a stripped ``*.p01.timeseries.hdf``)."""
    return MeshError(f"{name} holds model results but not the mesh they sit on: {missing} is missing. It looks like "
                     "a stripped plan file (a .p01.timeseries.hdf, or one saved without its geometry). Open the "
                     "full plan file (.p01.hdf), or give the geometry file the plan was run with (.g01.hdf) too.")


class _RasWithGeometry:
    """A stripped plan file read with its geometry file: the Geometry group from the geometry file, everything
    else (results, plan data, the file's own attributes) from the plan, as one h5py-like file."""

    def __init__(self, plan, geom):
        self.plan, self.geom = plan, geom
        self.attrs = _LayeredAttrs(plan.attrs, geom.attrs)

    def _of(self, key: str):
        return self.geom if key.lstrip("/").split("/", 1)[0] == "Geometry" else self.plan

    def __contains__(self, key: str) -> bool:
        return key in self._of(key)

    def __getitem__(self, key: str):
        return self._of(key)[key]

    def get(self, key: str, default: Any = None):
        return self._of(key).get(key, default)

    def close(self) -> None:
        for f in (self.plan, self.geom):
            try:
                f.close()
            except Exception:  # already closed
                pass


class _LayeredAttrs:
    """The plan's attributes; for RAS Mapper's projection and the units, the geometry file's when the plan lacks
    them. The rest (the file type, the version) stay the plan's."""

    SHARED = ("Projection", "Units System")

    def __init__(self, first, second):
        self.first, self.second = first, second

    def get(self, key: str, default: Any = None):
        value = self.first.get(key)
        if value is None and key in self.SHARED:
            value = self.second.get(key)
        return default if value is None else value


def _open_ras_geometry(h5py, plan, geometry: str | Path):
    """Pair a stripped plan file with its geometry file, after checking the geometry fits the plan's results."""
    gname = Path(geometry).name
    _check_size(geometry)
    if _sniff(geometry) != "hdf5":
        raise MeshError(f"{gname} is not an HDF5 file, so not a HEC-RAS geometry file (.g01.hdf).")
    geom = h5py.File(geometry, "r")
    try:
        missing = _ras_missing_geometry(geom) if f"{RAS_GEOM}/Attributes" in geom else RAS_GEOM
        if missing:
            raise MeshError(f"{gname} has no 2D mesh to give the plan: {missing} is missing. Give the geometry "
                            "file (.g01.hdf) the plan was run with.")
        attrs = geom[f"{RAS_GEOM}/Attributes"][:]
        names = [_text(x) for x in attrs["Name"]]
        counts = [int(x) for x in attrs["Cell Count"]] if "Cell Count" in attrs.dtype.names else \
            [int(geom[f"{RAS_GEOM}/{n}/Cells Center Coordinate"].shape[0]) for n in names]
        for n, count in zip(names, counts):
            ws = plan.get(f"{RAS_TS}/2D Flow Areas/{n}/Water Surface")
            if ws is None:
                ws = plan.get(f"{RAS_SUMMARY}/2D Flow Areas/{n}/Maximum Water Surface")
            if ws is None:
                raise MeshError(f"{gname} has a 2D area {n!r} that the plan has no results for. Give the geometry "
                                "file (.g01.hdf) the plan was run with.")
            if ws.shape[-1] < count:
                raise MeshError(f"{gname}'s 2D area {n!r} has {count:,} cells, more than the plan's results hold "
                                f"({ws.shape[-1]:,}). Give the geometry file (.g01.hdf) the plan was run with.")
    except Exception:
        geom.close()
        raise
    return _RasWithGeometry(plan, geom)


def read_ras_plan(path: str | Path, *, crs: str | int | None = None, max_cells: int = MAX_CELLS,
                  geometry: str | Path | None = None) -> Mesh:
    """Read a HEC-RAS 2D plan HDF5 file (``*.p##.hdf``) into a :class:`Mesh`.

    crs overrides (or supplies) the coordinate system: an EPSG code or WKT. The file's own is RAS Mapper's
    projection, stored with the results.

    geometry is the geometry HDF5 file (``*.g##.hdf``) the plan was run with, for a stripped plan file that
    holds the results but not the mesh (a ``*.p##.timeseries.hdf``). Without it such a file raises a
    :class:`MeshError` naming the missing part; it is ignored for a full plan file.
    """
    size = _check_size(path)
    name = Path(path).name
    if _sniff(path) != "hdf5":
        raise MeshError(f"{name} is not an HDF5 file, so not a HEC-RAS plan result.")
    h5py = _h5py()
    fh = h5py.File(path, "r")
    try:
        if "geometry" not in _text(fh.attrs.get("File Type")).lower() and _ras_has_results(fh):
            missing = _ras_missing_geometry(fh)
            if missing:
                if geometry is None:
                    raise _no_geometry(name, missing)
                fh = _open_ras_geometry(h5py, fh, geometry)
        return _read_ras(fh, name, size, crs, max_cells)
    except KeyError as exc:   # a part of the file the reader needs is not there: say so, not a traceback
        fh.close()
        raise MeshError(f"{name} lacks a part of a HEC-RAS plan the reader needs ({exc}). Open the full plan "
                        "file (.p01.hdf).") from exc
    except Exception:
        fh.close()
        raise


def _read_ras(fh, name: str, size: int, crs: str | int | None, max_cells: int) -> Mesh:
    kind = _text(fh.attrs.get("File Type"))
    if "geometry" in kind.lower():
        raise MeshError(f"{name} is a HEC-RAS geometry file, with no results in it. Open the plan's result file "
                        "(*.p01.hdf) instead.")
    if f"{RAS_GEOM}/Attributes" not in fh:
        raise MeshError(f"{name} has no 2D flow areas" + (f" (it is a {kind} file)." if kind else "."))
    attrs = fh[f"{RAS_GEOM}/Attributes"][:]
    names = [_text(x) for x in attrs["Name"]]
    counts = [int(x) for x in attrs["Cell Count"]] if "Cell Count" in attrs.dtype.names else None
    if counts is None:
        counts = [int(fh[f"{RAS_GEOM}/{n}/Cells Center Coordinate"].shape[0]) for n in names]
    total = sum(counts)
    if total > max_cells:
        raise MeshError(f"{name} has {total:,} cells; the mesh viewer draws up to {max_cells:,}.")
    cell_points = np.asarray(fh[f"{RAS_GEOM}/Cell Points"][:], "f8") if f"{RAS_GEOM}/Cell Points" in fh else None
    cell_info = np.asarray(fh[f"{RAS_GEOM}/Cell Info"][:], np.intp) if f"{RAS_GEOM}/Cell Info" in fh else None
    areas, start = [], 0
    for i, (n_name, count) in enumerate(zip(names, counts)):
        a = _ras_area(fh, n_name, count, cell_points, cell_info, i)
        a["start"] = start
        start += count
        areas.append(a)

    given = _given_crs(crs)
    wkt = _text(fh.attrs.get("Projection"))
    if given is None and not wkt:
        raise MeshError(f"{name} does not say its coordinate system (no RAS Mapper projection is stored). "
                        "Give it as an EPSG code.")
    crs_in, origin = (given, "given") if given is not None else (wkt, "file")
    cx = np.concatenate([a["centers"][:, 0] for a in areas])
    cy = np.concatenate([a["centers"][:, 1] for a in areas])
    lon, lat, crs_info = _to_lonlat(cx, cy, crs_in, origin)
    rlon, rlat, _ = _to_lonlat(np.concatenate([a["rx"] for a in areas]), np.concatenate([a["ry"] for a in areas]),
                               crs_in, origin)
    offsets = [np.asarray([0], np.intp)]
    base = 0
    for a in areas:
        offsets.append(a["offsets"][1:] + base)
        base += int(a["offsets"][-1])
    ring_offsets = np.concatenate(offsets)

    units = _units_of(_text(fh.attrs.get("Units System")))
    bed = np.concatenate([a["bed"] for a in areas]) if all(a["bed"] is not None for a in areas) else None
    notes: list[str] = []
    variable = "depth" if bed is not None else "water_surface"
    if bed is None:
        notes.append("No bed levels in this file (Cells Minimum Elevation), so no depth: this is the water surface.")

    times: list[str] = []
    if f"{RAS_TS}/Time Date Stamp (ms)" in fh or f"{RAS_TS}/Time Date Stamp" in fh:
        key = "Time Date Stamp (ms)" if f"{RAS_TS}/Time Date Stamp (ms)" in fh else "Time Date Stamp"
        stamps = fh[f"{RAS_TS}/{key}"]
        times = [_iso(parse_ras_time(s)) for s in stamps[:]]
    has_ts = bool(times) and any(f"{RAS_TS}/2D Flow Areas/{a['name']}/Water Surface" in fh for a in areas)
    if not has_ts:
        times = []
        notes.append("This file has no time series output, only the maxima.")
    has_v = any(f"{RAS_TS}/2D Flow Areas/{a['name']}/Face Velocity" in fh for a in areas)
    if not has_v:
        for a in areas:
            a["lsq"] = None

    if times and not any(f"{RAS_SUMMARY}/2D Flow Areas/{a['name']}/Maximum Water Surface" in fh for a in areas):
        notes.append("No summary output: the largest is over the saved time steps.")
    angle = _east_north_angle(cx, cy, crs_in, lat) if has_v else None
    src = _RasSource(fh, areas, bed, angle)
    src.n_steps = len(times)
    max_value = _ras_max_value(fh, areas, bed, src, times)
    max_speed = _ras_max_speed(fh, areas, src, times, notes) if has_v else None
    plan = fh.get("Plan Data/Plan Information")
    pattrs = dict(plan.attrs) if plan is not None else {}
    title = _text(pattrs.get("Plan Title") or pattrs.get("Plan Name")) or Path(name).stem
    project = _text(pattrs.get("Project Title"))
    model = _text(fh.attrs.get("File Version")) or "HEC-RAS"
    return Mesh(
        format="hec-ras", name=name, size_bytes=size, model=model,
        title=f"{title} ({project})" if project and project not in title else title,
        crs=crs_info, units={"depth": units["length"], "water_surface": units["length"], "velocity": units["velocity"]},
        areas=[{"name": a["name"], "cells": a["cells"], "start": a["start"]} for a in areas],
        lon=lon, lat=lat, ring_offsets=ring_offsets, ring_lon=rlon, ring_lat=rlat, times=times,
        variable=variable, max_value=max_value, max_speed=max_speed, notes=notes, _source=src)


def _ras_max_value(fh, areas, bed, src: _RasSource, times: list[str]) -> np.ndarray:
    parts = []
    for a in areas:
        n = a["cells"]
        mx = fh.get(f"{RAS_SUMMARY}/2D Flow Areas/{a['name']}/Maximum Water Surface")
        if mx is not None:
            ws = np.asarray(mx[0, :n], dtype="f8")
        elif times:
            ws = _block_max(fh[f"{RAS_TS}/2D Flow Areas/{a['name']}/Water Surface"], n)
        else:
            ws = np.full(n, np.nan)
        b = None if bed is None else bed[a["start"]:a["start"] + n]
        parts.append(src._value(ws, b))
    return np.concatenate(parts)


def _block_max(ds, n: int, block: int = 64) -> np.ndarray:
    out = np.full(n, -np.inf)
    for i in range(0, ds.shape[0], block):
        out = np.fmax(out, _nanmax(np.asarray(ds[i:i + block, :n], dtype="f8"), axis=0))
    out[~np.isfinite(out)] = np.nan
    return out


def _ras_max_speed(fh, areas, src: _RasSource, times: list[str], notes: list[str]) -> np.ndarray:
    parts = []
    proxy = False
    for a in areas:
        n = a["cells"]
        fv = fh.get(f"{RAS_TS}/2D Flow Areas/{a['name']}/Face Velocity")
        lsq = a.get("lsq")
        if fv is None or lsq is None:
            parts.append(np.full(n, np.nan))
            continue
        steps = fv.shape[0]
        if steps * fv.shape[1] <= SPEED_BUDGET:
            best = np.zeros(n)
            for i in range(0, steps, 64):
                u, v = lsq.solve(np.asarray(fv[i:i + 64, :], dtype="f8"))
                best = np.fmax(best, _nanmax(np.hypot(u, v), axis=0))
        else:   # too long a run to rebuild every step: the largest face speed around each cell
            proxy = True
            fmax = fh.get(f"{RAS_SUMMARY}/2D Flow Areas/{a['name']}/Maximum Face Velocity")
            face = np.abs(np.asarray(fmax[0, :], "f8")) if fmax is not None else np.zeros(fv.shape[1])
            best = np.zeros(n)
            np.maximum.at(best, lsq.cell_idx, face[lsq.face_idx])
        if src.bed is not None:
            seg = src.bed[a["start"]:a["start"] + n]
            mx = fh.get(f"{RAS_SUMMARY}/2D Flow Areas/{a['name']}/Maximum Water Surface")
            if mx is not None:
                best[~(np.asarray(mx[0, :n], "f8") - seg > DRY_DEPTH)] = np.nan
        parts.append(best)
    if proxy:
        notes.append("The largest speed per cell is the largest face velocity around it (the run is too long to "
                     "rebuild every step here).")
    return np.concatenate(parts)


# ── UGRID ───────────────────────────────────────────────────────────────────


class _H5Nc:
    """NetCDF-4 through h5py: variables, attributes, shapes and reads with the CF packing undone."""

    def __init__(self, path):
        self.fh = _h5py().File(path, "r")

    def names(self) -> list[str]:
        return [k for k, v in self.fh.items() if hasattr(v, "shape")]

    def attrs(self, name: str | None = None) -> dict[str, Any]:
        obj = self.fh if name is None else self.fh[name]
        return {k: v for k, v in obj.attrs.items() if k not in ("DIMENSION_LIST", "REFERENCE_LIST", "_Netcdf4Dimid",
                                                                "CLASS", "NAME", "_Netcdf4Coordinates")}

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.fh[name].shape)

    def raw(self, name: str, sel=()) -> np.ndarray:
        return np.asarray(self.fh[name][sel])

    def close(self) -> None:
        self.fh.close()


class _Nc3:
    """NetCDF-3 classic through scipy (the package's base install), read on demand."""

    def __init__(self, path):
        from scipy.io import netcdf_file

        self.fh = netcdf_file(path, "r", mmap=False, maskandscale=False)

    def names(self) -> list[str]:
        return list(self.fh.variables)

    def attrs(self, name: str | None = None) -> dict[str, Any]:
        obj = self.fh if name is None else self.fh.variables[name]
        return dict(obj._attributes)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.fh.variables[name].shape)

    def raw(self, name: str, sel=()) -> np.ndarray:
        return np.asarray(self.fh.variables[name][sel])

    def close(self) -> None:
        self.fh.close()


def _read(nc, name: str, sel=()) -> np.ndarray:
    """A variable with _FillValue / missing_value as NaN and scale_factor / add_offset applied."""
    a = nc.attrs(name)
    data = nc.raw(name, sel)
    if data.dtype.kind in "iuf":
        out = data.astype("f8")
        for key in ("_FillValue", "missing_value"):
            if key in a:
                fill = np.asarray(a[key]).reshape(-1)
                for f in fill:
                    out[data == f] = np.nan
        if "scale_factor" in a:
            out = out * float(np.asarray(a["scale_factor"]).reshape(-1)[0])
        if "add_offset" in a:
            out = out + float(np.asarray(a["add_offset"]).reshape(-1)[0])
        if data.dtype.kind == "f":
            out[np.abs(out) > 9e36] = np.nan
        return out
    return data


_TIME_UNITS = re.compile(r"^\s*(\w+)\s+since\s+(.+?)\s*$", re.I)
_UNIT_SECONDS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1, "minute": 60, "minutes": 60, "min": 60,
                 "mins": 60, "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600, "day": 86400,
                 "days": 86400, "d": 86400}


_CF_REF = re.compile(r"^(\d{1,4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{1,2})(?::(\d{1,2})(?:\.(\d+))?)?)?"
                     r"\s*(Z|UTC|GMT|[+-]\d{1,2}(?::?\d{2})?)?$", re.I)


def _cf_times(values: np.ndarray, units: str) -> list[str]:
    """CF times ("hours since 2021-02-03 06:00 +01:00", standard calendar) as ISO strings in UTC."""
    m = _TIME_UNITS.match(units or "")
    if not m or m.group(1).lower() not in _UNIT_SECONDS:
        raise MeshError(f"Unreadable time units: {units!r}")
    r = _CF_REF.match(m.group(2).strip())
    if not r:
        raise MeshError(f"Unreadable time reference: {units!r}")
    y, mo, d, hh, mm, ss, frac, tz = r.groups()
    base = datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0),
                    int(((frac or "0") + "000000")[:6]))
    if tz and tz[0] in "+-":
        hh_mm = tz[1:].replace(":", "")
        hours, minutes = (int(hh_mm[:-2]), int(hh_mm[-2:])) if len(hh_mm) > 2 else (int(hh_mm), 0)
        offset = timedelta(hours=hours, minutes=minutes)
        base = base - offset if tz[0] == "+" else base + offset
    sec = _UNIT_SECONDS[m.group(1).lower()]
    return [_iso(base + timedelta(seconds=float(v) * sec)) for v in np.asarray(values, "f8").reshape(-1)]


_DEPTH_NAMES = re.compile(r"(^|_)(waterdepth|water_depth|depth|hs|wd)$", re.I)
_LEVEL_NAMES = re.compile(r"(^|_)(s1|zeta|waterlevel|water_level|wse|wl|ssh|eta)$", re.I)
_BED_NAMES = re.compile(r"(^|_)(flowelem_bl|bl|node_z|bedlevel|bed_level|zb|elevation|bathymetry|depth_b)$", re.I)
_U_NAMES = re.compile(r"(^|_)(ucx|ucxa|u|vel_x|velocity_x|uc_x|ucmag_x)$", re.I)
_V_NAMES = re.compile(r"(^|_)(ucy|ucya|v|vel_y|velocity_y|uc_y|ucmag_y)$", re.I)


class _UgridSource:
    def __init__(self, nc, value_var: str | None, level_var: str | None, bed: np.ndarray | None, loc: dict,
                 u_var: str | None, v_var: str | None, angle: np.ndarray | None, faces: np.ndarray,
                 n_faces: int, time_axis: dict[str, int]):
        self.nc, self.value_var, self.level_var, self.bed = nc, value_var, level_var, bed
        self.loc, self.u_var, self.v_var, self.angle = loc, u_var, v_var, angle
        self.faces, self.n_faces, self.time_axis = faces, n_faces, time_axis

    def close(self) -> None:
        try:
            self.nc.close()
        except Exception:  # already closed
            pass

    def _to_faces(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Node values -> face values (the mean of a face's nodes); face values pass through."""
        if self.loc.get(name) != "node":
            return arr
        idx = self.faces
        ok = idx >= 0
        vals = np.where(ok, arr[..., np.where(ok, idx, 0)], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(vals, axis=-1)

    def _get(self, name: str, step) -> np.ndarray:
        shape = self.nc.shape(name)
        sel = [slice(None)] * len(shape)
        sel[self.time_axis[name]] = step
        return self._to_faces(name, _read(self.nc, name, tuple(sel)))

    def _get_cell(self, name: str, cell: int) -> np.ndarray:
        """All time steps at one face, reading only that face's column (or its nodes' columns)."""
        shape = self.nc.shape(name)
        axis = self.time_axis[name]
        if self.loc.get(name) == "node":
            nodes = sorted({int(i) for i in self.faces[cell] if i >= 0})
            cols = [_read(self.nc, name, (slice(None), k) if axis == 0 else (k, slice(None))) for k in nodes]
            if not cols:
                return np.full(shape[axis], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(np.stack(cols, axis=-1), axis=-1)
        return _read(self.nc, name, (slice(None), cell) if axis == 0 else (cell, slice(None)))

    def _value(self, step) -> np.ndarray:
        if self.value_var:
            d = self._get(self.value_var, step)
        else:
            d = self._get(self.level_var, step)
            if self.bed is not None:
                d = d - self.bed
        if self.value_var or self.bed is not None:
            d = np.where(d > DRY_DEPTH, d, np.nan)
        return d

    def frame(self, step: int) -> dict[str, np.ndarray | None]:
        value = self._value(step)
        if not (self.u_var and self.v_var):
            return {"value": value, "u": None, "v": None}
        u, v = _rotate(self._get(self.u_var, step), self._get(self.v_var, step), self.angle)
        dry = ~np.isfinite(value)
        u, v = np.where(dry, np.nan, u), np.where(dry, np.nan, v)
        return {"value": value, "u": u, "v": v}

    def series(self, cell: int) -> dict[str, Any]:
        if self.value_var:
            value = self._get_cell(self.value_var, cell)
        else:
            value = self._get_cell(self.level_var, cell)
            if self.bed is not None:
                value = value - self.bed[cell]
        if self.value_var or self.bed is not None:
            value = np.where(value > DRY_DEPTH, value, np.nan)
        speed = None
        if self.u_var and self.v_var:
            speed = np.hypot(self._get_cell(self.u_var, cell), self._get_cell(self.v_var, cell))
            speed[~np.isfinite(value)] = np.nan
        return {"value": value, "speed": speed}

    def values_block(self, sl: slice, raw: bool = False) -> np.ndarray:
        if not raw:
            return self._value(sl)
        if self.value_var:
            return self._get(self.value_var, sl)
        d = self._get(self.level_var, sl)
        return d - self.bed if self.bed is not None else d

    def block_max(self, which: str, n_steps: int, block: int = 32) -> np.ndarray:
        best = np.full(self.n_faces, np.nan)
        for i in range(0, n_steps, block):
            sl = slice(i, min(i + block, n_steps))
            if which == "value":
                cur = self._value(sl)
            else:
                cur = np.hypot(self._get(self.u_var, sl), self._get(self.v_var, sl))
                cur = np.where(np.isfinite(self._value(sl)), cur, np.nan)
            best = np.fmax(best, _nanmax(cur, axis=0) if cur.ndim > 1 else cur)
        return best


def _ugrid_crs(nc, x_name: str, y_name: str, x: np.ndarray, y: np.ndarray, given: Any) -> tuple[Any, str]:
    """The mesh's coordinate system: the one given, degrees when the coordinates say so, else the file's grid
    mapping (WKT or EPSG code). None when nothing says."""
    if given is not None:
        return given, "given"
    xa, ya = nc.attrs(x_name), nc.attrs(y_name)
    units = (_text(xa.get("units")) + " " + _text(ya.get("units"))).lower()
    std = (_text(xa.get("standard_name")) + " " + _text(ya.get("standard_name"))).lower()
    in_range = np.nanmin(y) >= -90 and np.nanmax(y) <= 90 and np.nanmin(x) >= -360 and np.nanmax(x) <= 360
    if in_range and ("degree" in units or "longitude" in std):
        return "EPSG:4326", "file"
    candidates = [n for n in nc.names() if any(k in nc.attrs(n) for k in ("grid_mapping_name", "crs_wkt",
                                                                            "spatial_ref", "epsg", "EPSG_code"))]
    for n in candidates:
        a = nc.attrs(n)
        for key in ("crs_wkt", "spatial_ref"):
            wkt = _text(a.get(key))
            if wkt.startswith(("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS", "GEODCRS", "COMPOUNDCRS")):
                return wkt, "file"
        for key in ("epsg", "EPSG_code"):
            m = re.search(r"(\d{4,6})", _text(a.get(key)))
            if m and m.group(1) != "0":
                return f"EPSG:{m.group(1)}", "file"
    return None, ""


def _pick(nc, names: list[str], pattern: re.Pattern, std_names: tuple[str, ...], mesh: str) -> str | None:
    """A data variable on the mesh, by standard name first, then by its name."""
    on_mesh = [n for n in names if _text(nc.attrs(n).get("mesh")) in ("", mesh)]
    for n in on_mesh:
        if _text(nc.attrs(n).get("standard_name")).lower() in std_names:
            return n
    for n in on_mesh:
        if pattern.search(n):
            return n
    return None


def read_ugrid(path: str | Path, *, crs: str | int | None = None, max_cells: int = MAX_CELLS) -> Mesh:
    """Read a UGRID 2D mesh result (NetCDF-4 or NetCDF-3) into a :class:`Mesh`: the faces as cells, the depth
    (or the water level, less the bed where the file has one) and the velocity where present.

    crs overrides (or supplies) the coordinate system: an EPSG code or WKT.
    """
    size = _check_size(path)
    kind = _sniff(path)
    if kind == "hdf5":
        nc = _H5Nc(path)
    elif kind == "netcdf3":
        nc = _Nc3(path)
    else:
        raise MeshError(f"{Path(path).name} is not a NetCDF file.")
    try:
        return _read_ugrid(nc, Path(path).name, size, crs, max_cells)
    except Exception:
        nc.close()
        raise


def _read_ugrid(nc, name: str, size: int, crs: str | int | None, max_cells: int) -> Mesh:
    names = nc.names()
    topo = [n for n in names if _text(nc.attrs(n).get("cf_role")) == "mesh_topology"
            and int(np.asarray(nc.attrs(n).get("topology_dimension", 2)).reshape(-1)[0]) == 2]
    if not topo:
        raise MeshError(f"{name} has no UGRID 2D mesh (no variable with cf_role = mesh_topology).")
    mesh = topo[0]
    ta = nc.attrs(mesh)
    node_xy = _text(ta.get("node_coordinates")).split()
    fnc = _text(ta.get("face_node_connectivity"))
    if len(node_xy) != 2 or not fnc or fnc not in names:
        raise MeshError(f"{name}: the mesh {mesh!r} names no node coordinates or face-node connectivity.")
    x_name, y_name = node_xy
    nx_, ny_ = _read(nc, x_name), _read(nc, y_name)
    conn_raw = _read(nc, fnc)
    ca = nc.attrs(fnc)
    start = int(np.asarray(ca.get("start_index", 0)).reshape(-1)[0])
    conn = np.where(np.isfinite(conn_raw), conn_raw, -1).astype(np.intp)
    conn = np.where(conn >= start, conn - start, -1)
    if conn.ndim == 2 and conn.shape[0] < conn.shape[1] and conn.shape[0] <= 8 and conn.shape[1] > 8:
        conn = conn.T   # stored nodes-first
    n_faces = conn.shape[0]
    if n_faces > max_cells:
        raise MeshError(f"{name} has {n_faces:,} cells; the mesh viewer draws up to {max_cells:,}.")
    conn[conn >= nx_.size] = -1

    given = _given_crs(crs)
    crs_in, origin = _ugrid_crs(nc, x_name, y_name, nx_, ny_, given)
    if crs_in is None:
        raise MeshError(f"{name} does not say its coordinate system (no grid mapping, EPSG code or degrees). "
                        "Give it as an EPSG code.")
    node_lon, node_lat, crs_info = _to_lonlat(nx_, ny_, crs_in, origin)
    valid = conn >= 0
    counts = valid.sum(axis=1)
    ring_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)
    flat = conn[valid]
    ring_lon, ring_lat = node_lon[flat], node_lat[flat]
    face_x = np.nanmean(np.where(valid, nx_[np.where(valid, conn, 0)], np.nan), axis=1)
    face_y = np.nanmean(np.where(valid, ny_[np.where(valid, conn, 0)], np.nan), axis=1)
    fxy = _text(ta.get("face_coordinates")).split()
    if len(fxy) == 2 and all(n in names for n in fxy):
        fx, fy = _read(nc, fxy[0]), _read(nc, fxy[1])
        if fx.size == n_faces and np.isfinite(fx).all():
            face_x, face_y = fx, fy
    lon, lat, _ = _to_lonlat(face_x, face_y, crs_in, origin)

    # time
    time_name = next((n for n in names if n.lower() == "time"), None) or next(
        (n for n in names if "since" in _text(nc.attrs(n).get("units")) and len(nc.shape(n)) == 1), None)
    times: list[str] = []
    n_t = 0
    if time_name:
        tv = _read(nc, time_name)
        times = _cf_times(tv, _text(nc.attrs(time_name).get("units")))
        n_t = len(times)
    n_nodes = nx_.size
    loc: dict[str, str] = {}
    time_axis: dict[str, int] = {}
    data_vars = []
    for n in names:
        shp = nc.shape(n)
        if not n_t or len(shp) < 2 or n_t not in shp:
            continue
        rest = [s for i, s in enumerate(shp) if i != shp.index(n_t)]
        if len(rest) != 1:
            continue   # 3D (layered) variables are left out
        if rest[0] == n_faces:
            loc[n] = "face"
        elif rest[0] == n_nodes:
            loc[n] = "node"
        else:
            continue
        time_axis[n] = shp.index(n_t)
        data_vars.append(n)
    value_var = _pick(nc, data_vars, _DEPTH_NAMES, ("sea_floor_depth_below_sea_surface", "water_depth",
                                                    "depth"), mesh)
    level_var = None if value_var else _pick(nc, data_vars, _LEVEL_NAMES, (
        "sea_surface_height", "sea_surface_height_above_geoid", "water_surface_height_above_reference_datum",
        "sea_surface_elevation"), mesh)
    if not value_var and not level_var:
        raise MeshError(f"{name} has no depth or water level on its mesh over time.")
    notes: list[str] = []
    bed = None
    variable = "depth"
    if level_var:
        static = [n for n in names if nc.shape(n) in ((n_faces,), (n_nodes,)) and n not in (x_name, y_name)]
        bed_var = _pick(nc, static, _BED_NAMES, ("altitude", "bedrock_altitude", "sea_floor_depth_below_geoid"), mesh)
        if bed_var:
            b = _read(nc, bed_var)
            if b.size == n_nodes:
                with np.errstate(invalid="ignore"):
                    b = np.nanmean(np.where(valid, b[np.where(valid, conn, 0)], np.nan), axis=1)
            if "depth" in _text(nc.attrs(bed_var).get("standard_name")):
                b = -b   # a depth below the datum is a negative bed level
            bed = b
        else:
            variable = "water_surface"
            notes.append("No bed level in this file, so no depth: this is the water level.")
    u_var = _pick(nc, data_vars, _U_NAMES, ("sea_water_x_velocity", "eastward_sea_water_velocity",
                                            "eastward_water_velocity"), mesh)
    v_var = _pick(nc, data_vars, _V_NAMES, ("sea_water_y_velocity", "northward_sea_water_velocity",
                                            "northward_water_velocity"), mesh)
    if bool(u_var) != bool(v_var):
        u_var = v_var = None
    angle = _east_north_angle(face_x, face_y, crs_in, lat) if u_var and not crs_info["geographic"] else None
    src = _UgridSource(nc, value_var, level_var, bed, loc, u_var, v_var, angle, conn, n_faces, time_axis)
    max_value = src.block_max("value", n_t) if n_t else np.full(n_faces, np.nan)
    max_speed = src.block_max("speed", n_t) if (n_t and u_var) else None
    vunits = _text(nc.attrs(value_var or level_var).get("units")) or None
    uunits = _text(nc.attrs(u_var).get("units")) or None if u_var else None
    if vunits is None:
        notes.append("The file does not state the units of its values.")
    gattrs = nc.attrs()
    model = _text(gattrs.get("source")) or _text(gattrs.get("institution")) or "UGRID"
    title = _text(gattrs.get("title")) or Path(name).stem
    return Mesh(
        format="ugrid", name=name, size_bytes=size, model=model, title=title, crs=crs_info,
        units={"depth": vunits if variable == "depth" else None,
               "water_surface": vunits if variable != "depth" else None, "velocity": uunits},
        areas=[{"name": mesh, "cells": int(n_faces), "start": 0}], lon=lon, lat=lat, ring_offsets=ring_offsets,
        ring_lon=ring_lon, ring_lat=ring_lat, times=times, variable=variable, max_value=max_value,
        max_speed=max_speed, notes=notes, _source=src)


# ── one door for both ───────────────────────────────────────────────────────


def read_mesh(path: str | Path, *, crs: str | int | None = None, max_cells: int = MAX_CELLS,
              geometry: str | Path | None = None) -> Mesh:
    """Open a model result whatever it is: a HEC-RAS plan HDF5 file or a UGRID NetCDF file. geometry is the
    HEC-RAS geometry file (``*.g01.hdf``) for a stripped plan file (see :func:`read_ras_plan`)."""
    _check_size(path)
    kind = _sniff(path)
    if kind == "hdf5":
        h5py = _h5py()
        with h5py.File(path, "r") as fh:
            is_ras = (f"{RAS_GEOM}/Attributes" in fh or "HEC-RAS" in _text(fh.attrs.get("File Type"))
                      or _ras_2d_results(fh) or _ras_has_results(fh))
        if is_ras:
            return read_ras_plan(path, crs=crs, max_cells=max_cells, geometry=geometry)
        return read_ugrid(path, crs=crs, max_cells=max_cells)
    if kind == "netcdf3":
        return read_ugrid(path, crs=crs, max_cells=max_cells)
    raise MeshError(f"{Path(path).name} is neither a HEC-RAS plan (HDF5) nor a UGRID NetCDF file.")


def _num(x: float | None, nd: int = 2) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def mesh_summary(path: str | Path, crs: str | int | None = None,
                 geometry: str | Path | None = None) -> dict[str, Any]:
    """What a model result holds, without its arrays: format, model, coordinate system, extent, cells, the
    model's time range, units, and the largest depth (or water surface) and speed with where they are.
    Every number is computed from the file. geometry is the HEC-RAS geometry file for a stripped plan file."""
    try:
        m = read_mesh(path, crs=crs, geometry=geometry)
    except MeshError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        out: dict[str, Any] = {"ok": True, **{k: v for k, v in m.meta().items() if k != "times"}}
        out["crs"] = {k: v for k, v in m.crs.items() if k != "wkt"}
        out["bbox"] = [round(v, 5) for v in m.bbox()]
        out["time_steps"] = len(m.times)
        out["start"] = m.times[0] if m.times else None
        out["end"] = m.times[-1] if m.times else None
        unit = m.units.get(m.variable)
        mv = m.max_value
        if np.isfinite(mv).any():
            i = int(np.nanargmax(mv))
            out["max"] = {"variable": m.variable, "value": _num(mv[i]), "unit": unit, "cell": i,
                          "lon": _num(m.lon[i], 5), "lat": _num(m.lat[i], 5)}
            if m.variable == "depth":
                out["wet_cells"] = int(np.isfinite(mv).sum())
        else:
            out["max"] = None
        if m.max_speed is not None and np.isfinite(m.max_speed).any():
            j = int(np.nanargmax(m.max_speed))
            out["max_speed"] = {"value": _num(m.max_speed[j]), "unit": m.units.get("velocity"), "cell": j,
                                "lon": _num(m.lon[j], 5), "lat": _num(m.lat[j], 5)}
        else:
            out["max_speed"] = None
        return out
    finally:
        m.close()


def browser_payload(m: Mesh) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The mesh for the Explorer: its metadata and its arrays as float32/int32, ready to become typed arrays."""
    meta = m.meta()
    finite = m.max_value[np.isfinite(m.max_value)]
    meta["value_range"] = [float(finite.min()), float(finite.max())] if finite.size else None
    meta["peak_step"] = m.peak_step()
    if m.max_speed is not None and np.isfinite(m.max_speed).any():
        meta["speed_max"] = float(np.nanmax(m.max_speed))
    arrays = {
        "lon": m.lon.astype("f8"), "lat": m.lat.astype("f8"),
        "ring_offsets": m.ring_offsets.astype("i4"), "ring_lon": m.ring_lon.astype("f8"),
        "ring_lat": m.ring_lat.astype("f8"), "max_value": m.max_value.astype("f4"),
    }
    if m.max_speed is not None:
        arrays["max_speed"] = m.max_speed.astype("f4")
    return meta, arrays


def frame_payload(m: Mesh, step: int) -> dict[str, np.ndarray]:
    """One time step as float32 arrays: value, and u and v when the file has velocities."""
    f = m.frame(step)
    out = {"value": f["value"].astype("f4")}
    if f["u"] is not None:
        out["u"] = f["u"].astype("f4")
        out["v"] = f["v"].astype("f4")
    return out


def series_payload(m: Mesh, cell: int) -> dict[str, Any]:
    """The time series at one cell as plain lists (None where dry), for the map card."""
    s = m.series(cell)

    def plain(a: np.ndarray) -> list[float | None]:
        return [None if not math.isfinite(float(x)) else round(float(x), 4) for x in a]

    return {"times": s["times"], "value": plain(s["value"]),
            "speed": plain(s["speed"]) if s["speed"] is not None else None, "variable": m.variable, "units": m.units}
