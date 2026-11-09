"""What a hydrogeologist reads off MODFLOW heads (#623, #624): contours, flow arrows, drawdown, pathlines to wells,
capture zones. Experimental (``docs/stability.md``).

Everything here works on one time step and one layer of a structured (DIS) grid read by
:func:`aquascope.groundwater.surface.read_modflow_heads`, and returns geometry in lon/lat for the map:

* **Head contours** (:func:`head_contours`): marching squares over the cell centres, at one round interval for the
  whole run (so the lines do not jump between steps), joined into labelled polylines.
* **Flow arrows** (:func:`flow_field`, :func:`flow_arrows`): the direction groundwater moves, down the head gradient
  (Darcy's law, ``q = -K grad h``, with K taken as the same in every direction, because a head file does not hold K).
  The gradient is a central difference between neighbouring cell centres (one-sided next to inactive cells), turned
  into map directions with the grid's own geometry, so a rotated or approximately placed grid points the right way.
* **Drawdown** (:func:`drawdown`): the head at a reference step (the first, or one you choose) minus the head now.
* **Wells** (:func:`read_wells`, :func:`find_wells`): from a dropped ``.wel`` file (MODFLOW-2005 or MODFLOW 6), or,
  without one, the cells whose drawdown is a local maximum. The drawdown used to find them is each cell's fall from
  the highest head it has in the run to the lowest, so a cone shows whether the run deepens it or lets it recover; a
  candidate also has to be a head minimum at its lowest (water flows into it). With a single step there is no
  drawdown, and the local head minima are used. The result says which.
* **Pathlines and capture zones** (:func:`pathlines`, :func:`capture_zones`): particles tracked through the flow
  field of one step, as if those heads held still (a steady-state reading of each step). Forward tracking ends in a
  pumping well's cell; backward tracking from a ring around each well traces where its water comes from, and the
  area those backward paths sweep is the capture zone, outlined.

What this is not, and the result says so: a head file holds no hydraulic conductivity, porosity or vertical flow, so
the lines are streamlines of the head gradient in one layer, not travel times, and water that leaves the layer up or
down is not followed. Wells in another layer are shown but do not capture water in this one. For travel times and
3D paths, run MODPATH on the model.

Pure numpy and scipy (both in the base install and in the Explorer's Pyodide worker).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from aquascope.groundwater.surface import LENGTH_UNITS_M, _as_bytes, nice_step

#: The Earth's mean radius (IUGG), for the local plane the geometry is measured on.
EARTH_RADIUS_M = 6_371_008.8
#: Particle step, in cells.
STEP_CELLS = 0.25
#: Arrows across the longer side of the model.
ARROWS_ACROSS = 14
#: Backward paths started around each well, before the fan is filled in where neighbouring paths drift apart.
RING_PATHS = 24
#: The most backward paths per well.
MAX_PATHS = 480
#: Pathlines drawn per well (a readable subset of the backward fan).
PATHS_SHOWN = 9
#: The most wells found without a .wel file.
MAX_FOUND = 12

METHOD = ("Arrows point down the head gradient (Darcy's law, the same conductivity in every direction). Particles "
          "follow that gradient within the layer, the step's heads held still: forward tracking ends at a pumping "
          "well, and the capture zone is the area that backward tracking from around the well sweeps.")
LIMITS = ("A head file has no conductivity, porosity or vertical flow: these are streamlines of one step in one layer, "
          "not travel times, and water leaving the layer is not followed. For travel times and 3D paths, run MODPATH.")

_COMPASS = ("north", "north-north-east", "north-east", "east-north-east", "east", "east-south-east", "south-east",
            "south-south-east", "south", "south-south-west", "south-west", "west-south-west", "west",
            "west-north-west", "north-west", "north-north-west")


def compass(bearing: float) -> str:
    """'east-south-east' for 112.5 degrees (clockwise from north)."""
    return _COMPASS[int(((bearing % 360) + 11.25) // 22.5) % 16]


# ── the grid's geometry ─────────────────────────────────────────────────────


def lattice_from_rings(rings: Any, nrow: int, ncol: int) -> tuple[np.ndarray, np.ndarray]:
    """The grid's corner lattice, (nrow + 1, ncol + 1) lon and lat arrays, from a DIS grid's cell rings (4 corners per
    cell, row-major, each top-left, top-right, bottom-right, bottom-left), as :func:`read_modflow_heads` gives them."""
    r = np.asarray(rings, float).reshape(nrow, ncol, 4, 2)
    lon = np.empty((nrow + 1, ncol + 1))
    lat = np.empty((nrow + 1, ncol + 1))
    lon[:nrow, :ncol], lat[:nrow, :ncol] = r[:, :, 0, 0], r[:, :, 0, 1]
    lon[:nrow, ncol], lat[:nrow, ncol] = r[:, -1, 1, 0], r[:, -1, 1, 1]
    lon[nrow, :ncol], lat[nrow, :ncol] = r[-1, :, 3, 0], r[-1, :, 3, 1]
    lon[nrow, ncol], lat[nrow, ncol] = r[-1, -1, 2, 0], r[-1, -1, 2, 1]
    return lon, lat


class Plane:
    """A structured grid on the map: its corner lattice in lon/lat and on a local plane in metres (east, north), and
    the conversions between fractional cell indexes (row ``fi``, column ``fj``; the centre of cell (i, j) is (i, j))
    and places."""

    def __init__(self, lon: Any, lat: Any) -> None:
        self.lon = np.asarray(lon, float)
        self.lat = np.asarray(lat, float)
        self.nrow, self.ncol = self.lon.shape[0] - 1, self.lon.shape[1] - 1
        self.lon0 = float(np.mean(self.lon))
        self.lat0 = float(np.mean(self.lat))
        k = math.cos(math.radians(self.lat0))
        self.x = np.radians(self.lon - self.lon0) * EARTH_RADIUS_M * k
        self.y = np.radians(self.lat - self.lat0) * EARTH_RADIUS_M
        x, y = self.x, self.y
        # the Jacobian of (east, north) against (column, row), per cell, from its corners
        self.x_j = ((x[:-1, 1:] + x[1:, 1:]) - (x[:-1, :-1] + x[1:, :-1])) / 2
        self.y_j = ((y[:-1, 1:] + y[1:, 1:]) - (y[:-1, :-1] + y[1:, :-1])) / 2
        self.x_i = ((x[1:, :-1] + x[1:, 1:]) - (x[:-1, :-1] + x[:-1, 1:])) / 2
        self.y_i = ((y[1:, :-1] + y[1:, 1:]) - (y[:-1, :-1] + y[:-1, 1:])) / 2
        self.area = np.abs(self.x_j * self.y_i - self.x_i * self.y_j)

    def _lattice_at(self, arr: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
        # corner-lattice coordinates of a cell-centre index: (fi + 0.5, fj + 0.5)
        u = np.clip(np.asarray(fi, float) + 0.5, 0, self.nrow)
        v = np.clip(np.asarray(fj, float) + 0.5, 0, self.ncol)
        i0 = np.minimum(np.floor(u).astype(int), self.nrow - 1)
        j0 = np.minimum(np.floor(v).astype(int), self.ncol - 1)
        a, b = u - i0, v - j0
        return ((1 - a) * (1 - b) * arr[i0, j0] + (1 - a) * b * arr[i0, j0 + 1] + a * (1 - b) * arr[i0 + 1, j0]
                + a * b * arr[i0 + 1, j0 + 1])

    def lonlat(self, fi: Any, fj: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._lattice_at(self.lon, fi, fj), self._lattice_at(self.lat, fi, fj)

    def xy(self, fi: Any, fj: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._lattice_at(self.x, fi, fj), self._lattice_at(self.y, fi, fj)


def _bearing(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Bearing (degrees clockwise from north) from point 1 to point 2, on a local plane (short distances)."""
    dx = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    return np.degrees(np.arctan2(dx, lat2 - lat1)) % 360


# ── the flow field ──────────────────────────────────────────────────────────


def _index_diff(h: np.ndarray, axis: int) -> np.ndarray:
    """dh per index step along an axis: central where both neighbours have a head, one-sided where one does."""
    n = h.shape[axis]
    out = np.zeros_like(h)
    if n < 2:
        return out
    hp = np.full_like(h, np.nan)
    hm = np.full_like(h, np.nan)
    sl = [slice(None)] * 2
    sl2 = [slice(None)] * 2
    sl[axis], sl2[axis] = slice(0, n - 1), slice(1, n)
    hp[tuple(sl)] = h[tuple(sl2)]   # the next cell's head
    hm[tuple(sl2)] = h[tuple(sl)]   # the previous cell's head
    both = np.isfinite(hp) & np.isfinite(hm)
    out[both] = (hp[both] - hm[both]) / 2
    fwd = ~both & np.isfinite(hp)
    out[fwd] = hp[fwd] - h[fwd]
    back = ~both & ~fwd & np.isfinite(hm)
    out[back] = h[back] - hm[back]
    out[~np.isfinite(h)] = np.nan
    return out


def flow_field(head: Any, plane: Plane, *, length_factor: float = 1.0) -> dict[str, np.ndarray]:
    """The head gradient and the direction water moves, per cell.

    ``head`` (nrow, ncol), NaN where there is none. ``length_factor``: metres per model length unit (heads are in it).
    Returns ``gx``, ``gy`` (the gradient east and north, dimensionless), ``vx``, ``vy`` (the Darcy direction,
    ``-grad h``, same units) and ``dj``, ``di`` (that direction in cell indexes per unit of length, for tracking)."""
    h = np.asarray(head, float) * float(length_factor)
    h_j, h_i = _index_diff(h, 1), _index_diff(h, 0)
    det = plane.x_j * plane.y_i - plane.x_i * plane.y_j
    det = np.where(np.abs(det) > 0, det, np.nan)
    # [h_j, h_i] = [[x_j, y_j], [x_i, y_i]] [h_x, h_y]
    gx = (h_j * plane.y_i - h_i * plane.y_j) / det
    gy = (h_i * plane.x_j - h_j * plane.x_i) / det
    vx, vy = -gx, -gy
    # [vx, vy] = [[x_j, x_i], [y_j, y_i]] [dj, di]
    dj = (vx * plane.y_i - vy * plane.x_i) / det
    di = (vy * plane.x_j - vx * plane.y_j) / det
    return {"gx": gx, "gy": gy, "vx": vx, "vy": vy, "dj": dj, "di": di}


def flow_arrows(field: dict[str, np.ndarray], plane: Plane, *, across: int = ARROWS_ACROSS) -> dict[str, Any]:
    """Arrows on a thinned grid of cells: where, which way (bearing, degrees clockwise from north), and how steep
    (``size`` 0.45 to 1, the square root of the gradient against the step's 95th percentile). Plus the step's typical
    gradient and direction, for the legend."""
    gx, gy = field["gx"], field["gy"]
    mag = np.hypot(gx, gy)
    ok = np.isfinite(mag)
    if not ok.any():
        return {"arrows": [], "gradient": None, "bearing": None, "towards": None}
    nrow, ncol = mag.shape
    k = max(1, round(max(nrow, ncol) / max(1, across)))
    top = float(np.nanpercentile(mag[ok], 95)) or 1.0
    ii, jj = np.meshgrid(np.arange(k // 2, nrow, k), np.arange(k // 2, ncol, k), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    keep = ok[ii, jj] & (mag[ii, jj] > top * 0.02)
    ii, jj = ii[keep], jj[keep]
    sj, si = field["dj"][ii, jj], field["di"][ii, jj]
    norm = np.hypot(sj, si)
    norm[norm == 0] = 1
    lon, lat = plane.lonlat(ii.astype(float), jj.astype(float))
    lon2, lat2 = plane.lonlat(ii + 0.3 * si / norm, jj + 0.3 * sj / norm)
    bear = _bearing(lon, lat, lon2, lat2)
    size = np.clip(np.sqrt(mag[ii, jj] / top), 0.45, 1.0)
    arrows = [{"lon": round(float(a), 6), "lat": round(float(b), 6), "bearing": round(float(c), 1),
               "size": round(float(s), 2)} for a, b, c, s in zip(lon, lat, bear, size)]
    # the typical direction: the mean flow vector's bearing; the typical gradient: the median magnitude
    mvx, mvy = float(np.nanmean(field["vx"][ok])), float(np.nanmean(field["vy"][ok]))
    typical = float(np.nanmedian(mag[ok]))
    b = math.degrees(math.atan2(mvx, mvy)) % 360 if math.hypot(mvx, mvy) > 0 else None
    return {"arrows": arrows, "gradient": typical, "bearing": None if b is None else round(b, 1),
            "towards": None if b is None else compass(b), "every": k}


# ── contours ────────────────────────────────────────────────────────────────


def _segments(v: np.ndarray, level: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    a, b, c, d = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]
    okq = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    code = ((a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8) * okq
    segs = []
    for i, j in zip(*np.nonzero((code > 0) & (code < 15))):
        q = [float(a[i, j]), float(b[i, j]), float(c[i, j]), float(d[i, j])]
        corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
        pts = []
        for e in range(4):
            p, r = q[e], q[(e + 1) % 4]
            if (p > level) != (r > level):
                t = (level - p) / (r - p)
                (ia, ja), (ib, jb) = corners[e], corners[(e + 1) % 4]
                pts.append((round(ia + t * (ib - ia), 7), round(ja + t * (jb - ja), 7)))
        if len(pts) == 4:
            # a saddle: pair the crossings by the quad's mean, the usual disambiguation
            mid = sum(q) / 4
            if (mid > level) == (q[0] > level):
                pts = [pts[0], pts[3], pts[1], pts[2]]
        for m in range(0, len(pts) - 1, 2):
            if pts[m] != pts[m + 1]:  # a level through a corner gives a point, not a segment
                segs.append((pts[m], pts[m + 1]))
    return segs


def _join(segs: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Segments into polylines (closed ones end where they start)."""
    ends: dict[tuple[float, float], list[int]] = {}
    for k, (p, q) in enumerate(segs):
        ends.setdefault(p, []).append(k)
        ends.setdefault(q, []).append(k)
    used = [False] * len(segs)
    lines = []

    def walk(start_seg: int, frm: tuple[float, float]) -> list[tuple[float, float]]:
        out = [frm]
        k, at = start_seg, frm
        while True:
            used[k] = True
            p, q = segs[k]
            nxt = q if p == at else p
            out.append(nxt)
            at = nxt
            cand = [m for m in ends.get(at, []) if not used[m]]
            if not cand:
                return out
            k = cand[0]

    # open lines first, from an end that only one segment touches
    for pt, ks in ends.items():
        if len(ks) == 1 and not used[ks[0]]:
            lines.append(walk(ks[0], pt))
    for k in range(len(segs)):
        if not used[k]:
            lines.append(walk(k, segs[k][0]))
    return lines


def head_contours(values: Any, plane: Plane, *, step: float | None = None, levels_from: Any = None,
                  target: int = 10, skip_zero: bool = False) -> dict[str, Any]:
    """Contour polylines of cell-centre values, in lon/lat: {step, levels: [{level, lines: [[[lon, lat], ...]]}]}.

    ``step`` defaults to a round interval for about ``target`` lines over ``levels_from`` (a [lo, hi] range, by
    default the values' own 1st to 99th percentile). Use the run's range so the interval holds from step to step.
    ``skip_zero`` leaves out the zero line (for drawdown, where it is mostly noise)."""
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    if finite.size < 4 or min(v.shape) < 2:
        return {"step": None, "levels": []}
    if levels_from is not None:
        lo, hi = float(levels_from[0]), float(levels_from[1])
    else:
        lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    step = float(step) if step else nice_step(hi - lo, target)
    vlo, vhi = float(finite.min()), float(finite.max())
    first = math.ceil(max(lo, vlo) / step) * step
    out = []
    k = 0
    while first + k * step <= min(hi, vhi) + 1e-12 and k < 80:
        level = round(first + k * step, 9)
        k += 1
        if skip_zero and abs(level) < step * 1e-6:
            continue
        lines = []
        for line in _join(_segments(v, level)):
            if len(line) < 2:
                continue
            fi = np.array([p[0] for p in line])
            fj = np.array([p[1] for p in line])
            lon, lat = plane.lonlat(fi, fj)
            lines.append([[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)])
        if lines:
            out.append({"level": level, "lines": lines})
    return {"step": step, "levels": out}


# ── drawdown and wells ──────────────────────────────────────────────────────


def drawdown(heads: Any, reference: Any) -> np.ndarray:
    """Drawdown: the reference head minus the head (positive where the water table fell). NaN where either is."""
    return np.asarray(reference, float) - np.asarray(heads, float)


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _num(tok: str) -> float:
    return float(tok.replace("d", "e").replace("D", "e"))


def read_wells(data: Any) -> dict[str, Any]:
    """Read a MODFLOW well file: MODFLOW-2005 (and NWT) WEL or MODFLOW 6 WEL, as bytes or a path.

    Returns {format, periods, wells: [{layer, row, col (1-based; ``cell`` instead of row and column on a DISV grid),
    q: [rate per stress period]}], notes}. Rates are in the model's units (length cubed per time); negative pumps
    water out. Lists in external files (OPEN/CLOSE) and parameters are not read, and the notes say so."""
    text = _as_bytes(data).decode("utf-8", "replace")
    if re.search(r"^\s*BEGIN\s+PERIOD", text, re.IGNORECASE | re.MULTILINE):
        return _mf6_wells(text)
    return _mf2005_wells(text)


def _strip(line: str) -> str:
    return re.split(r"[#!]", line, maxsplit=1)[0].strip()


def _mf2005_wells(text: str) -> dict[str, Any]:
    lines = [_strip(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    notes: list[str] = []
    if not lines:
        raise ValueError("an empty well file")
    i = 0
    if lines[0].upper().startswith("PARAMETER"):
        if int(lines[0].split()[1]) > 0:
            notes.append("The file defines well parameters; only the listed wells are read.")
        i += 1
    i += 1  # MXACTW IWELCB [options]
    while i < len(lines) and not _NUM.match(lines[i].split()[0]):
        i += 1  # an options line (SPECIFY, AUX ...)
    periods: list[dict[tuple[int, int, int], float]] = []
    while i < len(lines):
        tok = lines[i].split()
        if not _NUM.match(tok[0]):
            raise ValueError(f"cannot read the well file at {lines[i]!r}")
        itmp = int(_num(tok[0]))
        i += 1
        if itmp < 0:
            periods.append(dict(periods[-1]) if periods else {})
            continue
        cur: dict[tuple[int, int, int], float] = {}
        for _ in range(itmp):
            if i >= len(lines):
                break
            t = lines[i].split()
            i += 1
            if t[0].upper() in ("OPEN/CLOSE", "EXTERNAL"):
                notes.append("A stress period's list is in another file (OPEN/CLOSE), which was not read.")
                break
            key = (int(_num(t[0])), int(_num(t[1])), int(_num(t[2])))
            cur[key] = cur.get(key, 0.0) + _num(t[3])
        periods.append(cur)
    return _wells_out("MODFLOW-2005 WEL", periods, notes, disv=False)


def _mf6_wells(text: str) -> dict[str, Any]:
    notes: list[str] = []
    found: dict[int, dict[tuple, float]] = {}
    disv = None
    block: int | None = None
    for raw in text.splitlines():
        line = _strip(raw)
        if not line:
            continue
        m = re.match(r"BEGIN\s+PERIOD\s+(\d+)", line, re.IGNORECASE)
        if m:
            block = int(m.group(1))
            found[block] = {}
            continue
        if re.match(r"END\s+PERIOD", line, re.IGNORECASE):
            block = None
            continue
        if block is None:
            continue
        t = line.split()
        if t[0].upper() in ("OPEN/CLOSE", "EXTERNAL"):
            notes.append(f"Period {block}'s list is in another file (OPEN/CLOSE), which was not read.")
            continue
        n_int = 0
        while n_int < len(t) and re.match(r"^[-+]?\d+$", t[n_int]):
            n_int += 1
        # cellid then q: DIS has three integers (layer, row, column), DISV two (layer, cell)
        if disv is None:
            disv = n_int == 2 or (n_int == 3 and len(t) == 3)
        n_id = 2 if disv else 3
        if len(t) <= n_id:
            continue
        key = tuple(int(x) for x in t[:n_id])
        try:
            q = _num(t[n_id])
        except ValueError:
            notes.append("A rate given by a time series name was not read.")
            continue
        found[block][key] = found[block].get(key, 0.0) + q
    if not found:
        raise ValueError("no PERIOD blocks in the MODFLOW 6 well file")
    periods: list[dict[tuple, float]] = []
    last: dict[tuple, float] = {}
    for p in range(1, max(found) + 1):
        if p in found:
            last = found[p]
        periods.append(dict(last))
    return _wells_out("MODFLOW 6 WEL", periods, notes, disv=bool(disv))


def _wells_out(fmt: str, periods: list[dict[tuple, float]], notes: list[str], *, disv: bool) -> dict[str, Any]:
    keys = sorted({k for p in periods for k in p})
    wells = []
    for k in keys:
        w: dict[str, Any] = {"layer": k[0]}
        if disv:
            w["cell"] = k[1]
        else:
            w["row"], w["col"] = k[1], k[2]
        w["q"] = [round(p.get(k, 0.0), 6) for p in periods]
        wells.append(w)
    return {"format": fmt, "periods": len(periods), "wells": wells, "notes": notes}


def _neighbour_stack(a: np.ndarray, fill: float) -> np.ndarray:
    p = np.pad(a, 1, constant_values=fill)
    n, m = a.shape
    return np.stack([p[1 + di:1 + di + n, 1 + dj:1 + dj + m] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                     if (di, dj) != (0, 0)])


def find_wells(heads: Any, *, max_wells: int = MAX_FOUND) -> dict[str, Any]:
    """Pumping wells found in heads alone, for a model dropped without its .wel file.

    ``heads``: (time, nrow, ncol) for one layer. With more than one step, a candidate's drawdown (its fall from the
    highest head it has in the run to the lowest) is a local maximum, clearly above the layer's typical fall, and the
    cell is a head minimum among its neighbours at its lowest. With one step, the local head minima. Returns
    {wells: [{row, col (1-based), drawdown, lowest_step}], how}."""
    h = np.asarray(heads, float)
    if h.ndim == 2:
        h = h[None]
    nt = h.shape[0]
    with np.errstate(invalid="ignore"), np.errstate(all="ignore"):
        if nt > 1 and np.isfinite(h).any():
            hi = np.nanmax(np.where(np.isfinite(h), h, -np.inf), axis=0)
            lo = np.nanmin(np.where(np.isfinite(h), h, np.inf), axis=0)
            fall = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, np.nan)
        else:
            fall = None
    out = []
    if fall is not None and np.isfinite(fall).sum() > 8:
        f = np.where(np.isfinite(fall), fall, -np.inf)
        nb = _neighbour_stack(f, -np.inf)
        typical = float(np.nanmedian(fall))
        prom = fall - typical
        top = float(np.nanmax(prom))
        if top > 0:
            cand = np.isfinite(fall) & (f >= nb.max(axis=0)) & (prom >= max(0.25 * top, 1e-6))
            for i, j in zip(*np.nonzero(cand)):
                series = h[:, i, j]
                t_low = int(np.nanargmin(series))
                frame = h[t_low]
                nbh = _neighbour_stack(np.where(np.isfinite(frame), frame, np.inf), np.inf)[:, i, j]
                if frame[i, j] <= np.min(nbh) + 1e-9:
                    out.append({"row": int(i) + 1, "col": int(j) + 1, "drawdown": round(float(prom[i, j]), 4),
                                "lowest_step": t_low})
        how = ("found in the heads: cells whose drawdown (the fall from their highest head in the run to their "
               "lowest) is a local maximum, and where water flows in at their lowest. Add the .wel file for the "
               "model's own wells.")
    else:
        frame = h[-1]
        g = np.where(np.isfinite(frame), frame, np.inf)
        nb = _neighbour_stack(g, np.inf)
        interior = np.isfinite(frame) & np.all(np.isfinite(np.where(np.isinf(nb), np.nan, nb)), axis=0)
        cand = interior & (g < nb.min(axis=0))
        for i, j in zip(*np.nonzero(cand)):
            depth = float(np.mean(nb[:, i, j]) - frame[i, j])
            out.append({"row": int(i) + 1, "col": int(j) + 1, "drawdown": round(depth, 4), "lowest_step": nt - 1})
        how = ("found in the heads: one step has no drawdown, so these are the cells whose head is lower than all "
               "their neighbours' (where a pumping well would be). Add the .wel file for the model's own wells.")
    out.sort(key=lambda w: -w["drawdown"])
    return {"wells": out[:max_wells], "how": how}


# ── particle tracking ───────────────────────────────────────────────────────


def _sample(field: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    """Bilinear value of a cell-centre field at fractional indexes, NaN-aware (missing corners drop out)."""
    nrow, ncol = field.shape
    fi = np.clip(fi, 0, nrow - 1)
    fj = np.clip(fj, 0, ncol - 1)
    i0 = np.minimum(np.floor(fi).astype(int), max(0, nrow - 2))
    j0 = np.minimum(np.floor(fj).astype(int), max(0, ncol - 2))
    i1, j1 = np.minimum(i0 + 1, nrow - 1), np.minimum(j0 + 1, ncol - 1)
    a, b = fi - i0, fj - j0
    tot = np.zeros_like(fi)
    wsum = np.zeros_like(fi)
    for ii, jj, w in ((i0, j0, (1 - a) * (1 - b)), (i0, j1, (1 - a) * b), (i1, j0, a * (1 - b)), (i1, j1, a * b)):
        v = field[ii, jj]
        ok = np.isfinite(v)
        tot += np.where(ok, v * w, 0)
        wsum += np.where(ok, w, 0)
    return np.where(wsum > 1e-9, tot / np.maximum(wsum, 1e-12), np.nan)


def track(field: dict[str, np.ndarray], starts: Any, *, backward: bool = False, sinks: Any = None,
          step: float = STEP_CELLS, max_steps: int | None = None) -> dict[str, Any]:
    """Track particles through a flow field (from :func:`flow_field`), all at once, in cell indexes.

    ``starts``: (n, 2) fractional (row, column), 0-based. ``backward`` follows the flow upstream. ``sinks``: a
    (nrow, ncol) integer array, the well number (1, 2, ...) in a pumping well's cell and 0 elsewhere; forward tracking
    ends on entering one. Each particle moves ``step`` cells at a time (midpoint rule) until it leaves the active
    cells, reaches a sink, stalls (a head minimum or maximum that is not a well), or runs out of steps.

    Returns {paths: [array (k, 2) of (row, column)], end: ["well", "edge", "stalled", "steps"], well: [number or 0]}."""
    dj, di = field["dj"], field["di"]
    nrow, ncol = dj.shape
    sgn = -1.0 if backward else 1.0
    active = np.isfinite(dj) & np.isfinite(di)
    p = np.asarray(starts, float).reshape(-1, 2).copy()
    n = len(p)
    max_steps = max_steps or int(4 * (nrow + ncol) / step)
    sk = np.zeros((nrow, ncol), int) if sinks is None else np.asarray(sinks, int)
    speed = np.hypot(dj, di)
    tiny = (float(np.nanmedian(speed[active])) if active.any() else 1.0) * 1e-6
    hist = [p.copy()]
    alive = np.ones(n, bool)
    end = np.array(["steps"] * n, dtype=object)
    well = np.zeros(n, int)
    anchor = p.copy()

    def vel(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = _sample(di, q[:, 0], q[:, 1]), _sample(dj, q[:, 0], q[:, 1])
        s = np.hypot(a, b)
        bad = ~np.isfinite(s) | (s < tiny)
        s = np.where(bad, 1.0, s)
        return np.where(bad, np.nan, sgn * a / s), np.where(bad, np.nan, sgn * b / s)

    for it in range(1, max_steps + 1):
        idx = np.nonzero(alive)[0]
        if not idx.size:
            break
        q = p[idx]
        vi, vj = vel(q)
        mid = q + 0.5 * step * np.stack([vi, vj], axis=1)
        vi2, vj2 = vel(mid)
        vi2 = np.where(np.isfinite(vi2), vi2, vi)
        vj2 = np.where(np.isfinite(vj2), vj2, vj)
        stuck = ~np.isfinite(vi2) | ~np.isfinite(vj2)
        new = q + step * np.stack([np.nan_to_num(vi2), np.nan_to_num(vj2)], axis=1)
        inside = (new[:, 0] >= -0.5) & (new[:, 0] <= nrow - 0.5) & (new[:, 1] >= -0.5) & (new[:, 1] <= ncol - 0.5)
        ci = np.clip(np.rint(new[:, 0]).astype(int), 0, nrow - 1)
        cj = np.clip(np.rint(new[:, 1]).astype(int), 0, ncol - 1)
        out = ~stuck & (~inside | ~active[ci, cj])
        sink = np.zeros(len(idx), int) if backward else np.where(stuck | out, 0, sk[ci, cj])
        # a particle leaving the grid stops on its edge; one entering an inactive cell, half a step in
        if out.any():
            d = new - q
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.stack([np.where(d[:, 0] > 0, (nrow - 0.5 - q[:, 0]) / d[:, 0],
                                         np.where(d[:, 0] < 0, (-0.5 - q[:, 0]) / d[:, 0], np.inf)),
                                np.where(d[:, 1] > 0, (ncol - 0.5 - q[:, 1]) / d[:, 1],
                                         np.where(d[:, 1] < 0, (-0.5 - q[:, 1]) / d[:, 1], np.inf))], axis=1)
            tcut = np.where(inside, 0.5, np.clip(lim.min(axis=1), 0, 1))
            new = np.where(out[:, None], q + tcut[:, None] * d, new)
        moved = ~stuck
        p[idx[moved]] = new[moved]
        at_well = sink > 0
        p[idx[at_well]] = np.stack([ci[at_well], cj[at_well]], axis=1)  # end in the well's centre
        row = np.full((n, 2), np.nan)
        row[idx[moved]] = p[idx[moved]]
        hist.append(row)
        alive[idx[stuck]] = False
        end[idx[stuck]] = "stalled"
        alive[idx[out]] = False
        end[idx[out]] = "edge"
        alive[idx[at_well]] = False
        end[idx[at_well]] = "well"
        well[idx[at_well]] = sink[at_well]
        if it % 40 == 0:
            slow = alive & (np.hypot(*(p - anchor).T) < 1.0)
            alive[slow] = False
            end[slow] = "stalled"
            anchor = p.copy()
    h = np.stack(hist)  # (steps + 1, n, 2)
    paths = [h[np.isfinite(h[:, k, 0]), k] for k in range(n)]
    return {"paths": paths, "end": list(end), "well": [int(x) for x in well]}


def _ring(i: int, j: int, angles: np.ndarray, radius: float = 0.45) -> np.ndarray:
    return np.stack([i - radius * np.cos(angles), j + radius * np.sin(angles)], axis=1)


def _fan(field: dict[str, np.ndarray], i: int, j: int, *, max_paths: int = MAX_PATHS, gap: float = 0.7) -> tuple[
        np.ndarray, list[np.ndarray]]:
    """Backward paths from a ring around cell (i, j), filled in between neighbours that drift more than ``gap``
    cells apart. Returns (angles, paths), sorted by angle."""
    angles = np.linspace(0, 2 * np.pi, RING_PATHS, endpoint=False)
    paths = track(field, _ring(i, j, angles), backward=True)["paths"]
    for _ in range(6):
        order = np.argsort(angles)
        angles = angles[order]
        paths = [paths[k] for k in order]
        new = []
        for k in range(len(angles)):
            a, b = paths[k], paths[(k + 1) % len(angles)]
            m = min(len(a), len(b))
            if m < 2:
                continue
            if float(np.max(np.hypot(*(a[:m] - b[:m]).T))) > gap:
                a2 = angles[(k + 1) % len(angles)] + (2 * np.pi if k + 1 == len(angles) else 0)
                new.append((angles[k] + a2) / 2 % (2 * np.pi))
        if not new or len(angles) + len(new) > max_paths:
            break
        new_a = np.array(new)
        angles = np.concatenate([angles, new_a])
        paths = paths + track(field, _ring(i, j, new_a), backward=True)["paths"]
    order = np.argsort(angles)
    return angles[order], [paths[k] for k in order]


# ── outlines ────────────────────────────────────────────────────────────────


def _outline_rings(mask: np.ndarray, scale: int) -> list[list[tuple[float, float]]]:
    """Closed outlines of a mask (0/1, or a smoothed 0 to 1) on a grid ``scale`` times finer than the cells, at
    0.5, in cell indexes."""
    m = np.pad(np.asarray(mask, float), 1)
    lines = _join(_segments(m, 0.5))
    rings = []
    for line in lines:
        if len(line) < 4:
            continue
        # back to cell indexes: sub-cell centre a (of the padded grid) is cell row (a - 1 + 0.5) / scale - 0.5
        rings.append([((a - 1 + 0.5) / scale - 0.5, (b - 1 + 0.5) / scale - 0.5) for a, b in line])
    return rings


def _rasterise(paths: list[np.ndarray], nrow: int, ncol: int, scale: int) -> np.ndarray:
    m = np.zeros((nrow * scale, ncol * scale), bool)
    for pth in paths:
        if not len(pth):
            continue
        # densify so a step never skips a sub-cell
        d = np.hypot(*np.diff(pth, axis=0).T) if len(pth) > 1 else np.zeros(0)
        pts = [pth[:1]]
        for k, dist in enumerate(d):
            nseg = max(1, int(math.ceil(dist * scale * 2)))
            t = np.linspace(0, 1, nseg + 1)[1:, None]
            pts.append(pth[k] + t * (pth[k + 1] - pth[k]))
        q = np.concatenate(pts)
        a = np.clip(np.floor((q[:, 0] + 0.5) * scale).astype(int), 0, nrow * scale - 1)
        b = np.clip(np.floor((q[:, 1] + 0.5) * scale).astype(int), 0, ncol * scale - 1)
        m[a, b] = True
    return m


def capture_zones(field: dict[str, np.ndarray], wells: list[tuple[int, int]], plane: Plane, *,
                  scale: int = 3) -> list[dict[str, Any]]:
    """Each pumping well's capture zone, from backward tracking: the area its backward paths sweep, outlined.

    ``wells``: 0-based (row, column) of the pumping wells in this layer. Returns per well {rings (lon/lat), area_km2,
    paths (a readable subset of the backward paths, reversed to run toward the well, in lon/lat), mask (cells,
    bool)}."""
    from scipy import ndimage

    nrow, ncol = field["dj"].shape
    active = np.isfinite(field["dj"])
    act_sub = np.kron(active, np.ones((scale, scale), bool))
    sub_area = np.kron(plane.area, np.ones((scale, scale))) / scale ** 2
    out = []
    fans = []
    for i, j in wells:
        _angles, paths = _fan(field, i, j)
        raw = _rasterise(paths, nrow, ncol, scale)
        raw[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = True
        fans.append((paths, raw))
    masks = []
    for raw in (r for _p, r in fans):
        m = ndimage.binary_closing(raw, structure=np.ones((3, 3), bool), iterations=2) | raw
        # fill the small gaps the paths leave, never a hole the size of another well's zone
        holes, nh = ndimage.label(ndimage.binary_fill_holes(m) & ~m)
        if nh:
            sizes = ndimage.sum(np.ones_like(holes), holes, index=np.arange(1, nh + 1))
            m |= np.isin(holes, 1 + np.nonzero(sizes <= 2 * scale * scale)[0])
        masks.append(m & act_sub)
    if len(masks) > 1:
        # water goes to one well: where two wells' swept areas overlap (their paths hug a divide), a particle
        # tracked forward from that spot says which well it reaches
        stack = np.stack(masks)
        clash = stack.sum(axis=0) > 1
        if clash.any():
            a, b = np.nonzero(clash)
            sinks = np.zeros((nrow, ncol), int)
            for n, (wi, wj) in enumerate(wells, start=1):
                sinks[wi, wj] = n
            dest = np.array(track(field, np.stack([(a + 0.5) / scale - 0.5, (b + 0.5) / scale - 0.5], axis=1),
                                  sinks=sinks)["well"])
            for n in range(len(masks)):
                masks[n][a, b] = dest == n + 1
    for k, (i, j) in enumerate(wells):
        paths = fans[k][0]
        m = masks[k]
        lab, _n = ndimage.label(m)
        keep = lab[i * scale + scale // 2, j * scale + scale // 2]
        if keep:
            m = lab == keep
        rings = []
        # the outline of the swept area, softened by a third of a cell so it reads as a zone, not as pixels
        soft = ndimage.gaussian_filter(m.astype(float), sigma=scale / 3)
        for ring in _outline_rings(soft, scale):
            fi = np.array([p[0] for p in ring])
            fj = np.array([p[1] for p in ring])
            lon, lat = plane.lonlat(fi, fj)
            rings.append([[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)])
        step_k = max(1, len(paths) // PATHS_SHOWN)
        shown = []
        for pth in paths[::step_k][:PATHS_SHOWN * 2]:
            if len(pth) < 3:
                continue
            rev = pth[::-1]
            rev = rev[:: max(1, len(rev) // 160)]
            lon, lat = plane.lonlat(rev[:, 0], rev[:, 1])
            wlon, wlat = plane.lonlat(np.array([float(i)]), np.array([float(j)]))
            coords = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]
            coords.append([round(float(wlon[0]), 6), round(float(wlat[0]), 6)])
            shown.append(coords)
        cells = m.reshape(nrow, scale, ncol, scale).any(axis=(1, 3))
        out.append({"rings": rings, "area_km2": round(float(sub_area[m].sum()) / 1e6, 4), "paths": shown,
                    "mask": cells})
    return out


def pathlines(field: dict[str, np.ndarray], wells: list[tuple[int, int]], *, every: int = 3) -> dict[str, Any]:
    """Forward pathlines from a regular set of cell centres (every ``every`` cells), ending at the pumping wells
    (0-based row, column) when they reach one. Returns {paths: [array (k, 2) of (row, column)], well: [number or 0],
    end: [...]}: the same tracking as :func:`capture_zones`, the other way."""
    nrow, ncol = field["dj"].shape
    sinks = np.zeros((nrow, ncol), int)
    for n, (i, j) in enumerate(wells, start=1):
        sinks[i, j] = n
    ii, jj = np.meshgrid(np.arange(every // 2, nrow, every), np.arange(every // 2, ncol, every), indexing="ij")
    starts = np.stack([ii.ravel(), jj.ravel()], axis=1).astype(float)
    ok = np.isfinite(field["dj"][starts[:, 0].astype(int), starts[:, 1].astype(int)])
    return track(field, starts[ok], sinks=sinks)


# ── one step of a model ─────────────────────────────────────────────────────


def _layer_heads(res: dict[str, Any], layer: int | None) -> tuple[int, int, np.ndarray]:
    lay = res["layers"][0] if layer is None else int(layer)
    if lay not in res["layers"]:
        raise ValueError(f"layer {lay} is not in the file (layers {res['layers']})")
    li = res["layers"].index(lay)
    g = res["grid"]
    nrow, ncol = g["drawn_nrow"], g["drawn_ncol"]
    return lay, li, np.asarray(res["heads"], float)[:, li].reshape(len(res["times"]), nrow, ncol)


def _run_range(a: np.ndarray, symmetric: bool = False) -> list[float] | None:
    v = a[np.isfinite(a)]
    if v.size > 2_000_000:
        v = v[:: v.size // 2_000_000 + 1]
    if not v.size:
        return None
    if symmetric:
        m = float(np.percentile(np.abs(v), 99.5))
        m = m if m > 1e-6 else 1.0
        return [round(-m, 4), round(m, 4)]
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return [round(lo, 4), round(hi, 4)]


def model_wells(res: dict[str, Any], layer: int | None = None, wel: Any = None) -> dict[str, Any]:
    """The wells to show for a layer: the .wel file's (bytes, a path, or what :func:`read_wells` returned) when there
    is one, else :func:`find_wells` on the layer's heads. Rows and columns are the drawn grid's (1-based)."""
    lay, _li, cube = _layer_heads(res, layer)
    g = res["grid"]
    block = int(g.get("block") or 1)
    if wel is not None:
        parsed = wel if isinstance(wel, dict) and "wells" in wel else read_wells(wel)
        wells = []
        for n, w in enumerate(parsed["wells"], start=1):
            if "row" not in w:
                continue
            r, c = (w["row"] - 1) // block + 1, (w["col"] - 1) // block + 1
            wells.append({"id": f"W{n}", "row": r, "col": c, "layer": w["layer"], "model_row": w["row"],
                          "model_col": w["col"], "q": w["q"], "from": "wel"})
        how = f"from the {parsed['format']} file ({len(wells)} well{'s' if len(wells) != 1 else ''})"
        return {"wells": wells, "how": how, "from": "wel", "notes": parsed.get("notes", [])}
    found = find_wells(cube)
    wells = [{"id": f"W{n}", "row": w["row"], "col": w["col"], "layer": lay, "model_row": (w["row"] - 1) * block + 1,
              "model_col": (w["col"] - 1) * block + 1, "q": None, "found_drawdown": w["drawdown"],
              "lowest_step": w["lowest_step"], "from": "found"} for n, w in enumerate(found["wells"], start=1)]
    return {"wells": wells, "how": found["how"], "from": "found", "notes": []}


def _period_of(res: dict[str, Any], t: int) -> int:
    ksp = res.get("kstpkper") or []
    if 0 <= t < len(ksp) and isinstance(ksp[t], (list, tuple)):
        return int(ksp[t][1])
    return t + 1


def model_views(
    res: dict[str, Any],
    time_index: int = -1,
    layer: int | None = None,
    *,
    reference: int | str | None = 0,
    wells: Any = None,
    show: Any = None,
) -> dict[str, Any]:
    """Contours, flow arrows, drawdown, wells, pathlines and capture zones for one step and layer of
    :func:`aquascope.groundwater.surface.read_modflow_heads`'s result, in lon/lat.

    ``reference``: the step drawdown is measured from (default the first; ``"highest"`` for the step at which the
    wells' heads are highest, the least pumped). ``wells``: a ``.wel`` file (bytes or a path), what
    :func:`model_wells` returned, or None to find them in the heads. ``show``: which of "head", "arrows",
    "drawdown", "capture" to compute (default all).

    Returns {kind, time_index, layer, length_unit, head: {step, range, levels}, flow: {arrows, gradient, bearing,
    towards}, drawdown: {reference, range, levels, suggested_reference}, wells: [...], wells_how, capture: [{well,
    rings, area_km2, paths}], method, limits, notes}. A DISV grid returns only an error: these views need rows and
    columns."""
    g = res.get("grid") or {}
    if res.get("error"):
        return {"kind": "modflow_views", "error": res["error"]}
    if g.get("type") != "dis":
        return {"kind": "modflow_views", "error": "Contours, arrows and pathlines need a structured (DIS) grid; this "
                "one is unstructured (DISV), so the cells are drawn as they are."}
    show = set(show or ("head", "arrows", "drawdown", "capture"))
    nt = len(res["times"])
    t = int(time_index) % nt
    lay, _li, cube = _layer_heads(res, layer)
    nrow, ncol = g["drawn_nrow"], g["drawn_ncol"]
    plane = Plane(*lattice_from_rings(res["rings"], nrow, ncol))
    units = (g.get("length_units") or "").lower()
    factor = LENGTH_UNITS_M.get(units, 1.0)
    frame = cube[t]
    notes: list[str] = []
    out: dict[str, Any] = {"kind": "modflow_views", "time_index": t, "layer": lay, "length_unit": units or None,
                           "method": METHOD, "limits": LIMITS, "notes": notes}
    head_range = _run_range(cube)
    if "head" in show:
        c = head_contours(frame, plane, levels_from=head_range, target=10)
        out["head"] = {"range": head_range, **c}
    field = flow_field(frame, plane, length_factor=factor)
    if "arrows" in show:
        out["flow"] = flow_arrows(field, plane)
    w = wells if isinstance(wells, dict) and "how" in wells else model_wells(res, lay, wells)
    period = _period_of(res, t)
    here = []
    for x in w["wells"]:
        i, j = x["row"] - 1, x["col"] - 1
        if not (0 <= i < nrow and 0 <= j < ncol):
            continue
        lon, lat = plane.lonlat(np.array([float(i)]), np.array([float(j)]))
        q = None
        if x.get("q"):
            q = x["q"][min(len(x["q"]), period) - 1]
        if x["from"] == "wel":
            pumping = bool(q is not None and q < 0 and x["layer"] == lay)
        else:
            nb = _neighbour_stack(np.where(np.isfinite(frame), frame, np.inf), np.inf)[:, i, j]
            pumping = bool(np.isfinite(frame[i, j]) and frame[i, j] <= float(np.min(nb)) + 1e-9)
        here.append({**{k: v for k, v in x.items() if k != "q"}, "lon": round(float(lon[0]), 6),
                     "lat": round(float(lat[0]), 6), "q": None if q is None else round(float(q), 4),
                     "pumping": pumping, "in_layer": x["layer"] == lay,
                     "head": None if not np.isfinite(frame[i, j]) else round(float(frame[i, j]), 4)})
    out["wells_how"] = w["how"]
    out["wells_from"] = w["from"]
    notes.extend(w.get("notes") or [])
    if "drawdown" in show:
        ref = _reference_step(cube, here, reference)
        dd_cube = drawdown(cube, cube[ref])
        dd = dd_cube[t]
        rng = _run_range(dd_cube, symmetric=True)
        step = nice_step(rng[1], 6) if rng else None
        c = head_contours(dd, plane, step=step, levels_from=rng, skip_zero=True)
        best = _reference_step(cube, here, "highest") if here else ref
        out["drawdown"] = {"reference": ref, "range": rng, "step": c["step"], "levels": c["levels"],
                           "suggested_reference": best if best != ref else None,
                           "max": None if not np.isfinite(dd).any() else round(float(np.nanmax(dd)), 4)}
        for x in here:
            v = dd[x["row"] - 1, x["col"] - 1]
            x["drawdown"] = None if not np.isfinite(v) else round(float(v), 4)
    out["wells"] = here
    if "capture" in show:
        pumped = [(x["row"] - 1, x["col"] - 1) for x in here if x["pumping"]]
        zones = capture_zones(field, pumped, plane) if pumped else []
        ids = [x["id"] for x in here if x["pumping"]]
        out["capture"] = [{"well": wid, "rings": z["rings"], "area_km2": z["area_km2"], "paths": z["paths"]}
                          for wid, z in zip(ids, zones)]
        for x in here:
            z = next((c for c in out["capture"] if c["well"] == x["id"]), None)
            x["capture_km2"] = z["area_km2"] if z else None
        if not pumped:
            notes.append("No well pumps in this layer at this step, so there is no capture zone to draw.")
    return out


def _reference_step(cube: np.ndarray, wells: list[dict[str, Any]], reference: int | str | None) -> int:
    nt = cube.shape[0]
    if reference == "highest":
        cells = [(x["row"] - 1, x["col"] - 1) for x in wells if x.get("in_layer", True)]
        if not cells:
            return 0
        with np.errstate(invalid="ignore"):
            s = np.nanmean(np.stack([cube[:, i, j] for i, j in cells], axis=1), axis=1)
        return int(np.nanargmax(s)) if np.isfinite(s).any() else 0
    r = 0 if reference is None else int(reference)
    if not -nt <= r < nt:
        raise ValueError(f"the reference step is out of range (0 to {nt - 1})")
    return r % nt


def views_summary(v: dict[str, Any]) -> dict[str, Any]:
    """The views without their geometry, for an MCP answer or a terminal."""
    if v.get("error"):
        return v
    out: dict[str, Any] = {k: v[k] for k in ("kind", "time_index", "layer", "length_unit", "wells_how", "wells_from", "method",
                             "limits", "notes") if k in v}
    if "head" in v:
        out["head"] = {"range": v["head"]["range"], "step": v["head"]["step"], "levels": len(v["head"]["levels"])}
    if "flow" in v:
        out["flow"] = {k: v["flow"].get(k) for k in ("gradient", "bearing", "towards")}
    if "drawdown" in v:
        out["drawdown"] = {k: v["drawdown"].get(k) for k in ("reference", "range", "step", "max",
                                                             "suggested_reference")}
    out["wells"] = v.get("wells", [])
    if "capture" in v:
        out["capture"] = [{"well": c["well"], "area_km2": c["area_km2"]} for c in v["capture"]]
    return out


__all__ = [
    "LIMITS", "METHOD", "Plane", "capture_zones", "compass", "drawdown", "find_wells", "flow_arrows", "flow_field",
    "head_contours", "lattice_from_rings", "model_views", "model_wells", "pathlines", "read_wells", "track",
    "views_summary",
]
