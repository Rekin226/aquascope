"""Drop a pumping well on the map and see the drawdown cone (#660). Experimental.

The Theis (1935) solution for a fully penetrating well pumped at a constant rate ``Q`` in a confined, homogeneous,
isotropic aquifer of infinite extent::

    s(r, t) = Q / (4 pi T) * W(u),    u = r**2 S / (4 T t)

with ``T`` the transmissivity, ``S`` the storativity and ``W(u) = E1(u)`` the well function. Several wells add by
superposition: the drawdown at a point is the sum of each well's own drawdown there. The well function and the
single-well drawdown are :mod:`aquascope.groundwater.aquifer`'s (the engine behind ``aquifer_drawdown`` and
``aquascope groundwater --analysis theis``); this module adds the superposition over a map, the contours, the
defaults and the check.

* :func:`drawdown_map`: the cone of one or several wells as contour lines in lon/lat at round levels (0.05, 0.1,
  0.2, 0.5, 1, 2, 5 m ...), a coarse image for the fill, each well's drawdown at its face, and optionally the cone
  at a series of times for the time bar.
* :func:`drawdown_at`: the drawdown at any point, with each well's share.
* :func:`check`: the textbook checks (Fetter's worked example and the USGS table of ``W(u)``) and superposition.

Units: metres, days, m3/day, m2/day, storativity dimensionless. The defaults (:data:`DEFAULTS`) are a municipal well
in a confined sand aquifer; :data:`RANGES` gives the typical ranges and :data:`RANGES_SOURCE` where they come from.
Lengths on the map are measured on a local plane around the wells (equirectangular, a few metres in 100 km).

References: Theis, C.V. (1935), Trans. AGU 16, 519-524. Fetter, C.W. (2001), Applied Hydrogeology, 4th ed.,
Prentice Hall. Lohman, S.W. (1972), Ground-water hydraulics, USGS Professional Paper 708. Heath, R.C. (1983), Basic
ground-water hydrology, USGS Water-Supply Paper 2220.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from aquascope.groundwater.aquifer import _well_function, theis_drawdown
from aquascope.groundwater.views import _join, _segments

__all__ = [
    "DEFAULTS", "EDGE_M", "FRAME_DAYS", "LEVELS", "LIMITS", "MAX_WELLS", "RANGES", "RANGES_SOURCE",
    "check", "drawdown_at", "drawdown_map", "wells_from",
]

#: The Earth's mean radius (IUGG), for the local plane around the wells.
EARTH_RADIUS_M = 6_371_008.8

#: The defaults: a municipal well (1000 m3/day, about 11.6 L/s) in a confined sand aquifer, read after 30 days, the
#: drawdown at the well's face taken at a radius of 0.15 m (a 0.3 m borehole, no well loss).
DEFAULTS: dict[str, float] = {
    "rate_m3_day": 1000.0,
    "transmissivity_m2_day": 500.0,
    "storativity": 1e-4,
    "days": 30.0,
    "well_radius_m": 0.15,
}

#: Typical ranges, from Heath (1983).
RANGES: dict[str, tuple[float, float]] = {
    "transmissivity_m2_day": (1.0, 100_000.0),
    "storativity": (1e-5, 1e-3),
}
RANGES_SOURCE = ("Heath, R.C. (1983), Basic ground-water hydrology, USGS Water-Supply Paper 2220: the transmissivity "
                 "of principal aquifers ranges from less than 1 to 100,000 m2/d (p. 26), and the storage coefficient "
                 "of most confined aquifers from about 1e-5 to 1e-3 (p. 28). https://pubs.usgs.gov/wsp/2220/report.pdf")

#: What the package accepts.
LIMITS: dict[str, tuple[float, float]] = {
    "rate_m3_day": (0.1, 1_000_000.0),
    "transmissivity_m2_day": (0.01, 1_000_000.0),
    "storativity": (1e-7, 0.5),
    "days": (0.001, 36_500.0),
    "well_radius_m": (0.01, 5.0),
}
MAX_WELLS = 10

#: The contour levels, m: a 1-2-5 series, so a cone reads at a glance on a log scale like a type curve.
LEVELS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
#: The cone's edge: the smallest level drawn, m.
EDGE_M = LEVELS[0]
#: The most contour lines drawn (the deepest, tightest circles go first).
MOST_LEVELS = 8
#: The cone is cut at this distance from the wells, m (a confined aquifer is never infinite in practice).
MAX_REACH_M = 100_000.0
#: The times the cone is drawn at for the time bar, days (those up to the time asked, and that time).
FRAME_DAYS: tuple[float, ...] = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730,
                                 1095, 1825, 3650, 7300, 10950, 18250, 36500)
#: The fill image's size, cells.
IMAGE_CELLS = 72

METHOD = ("Theis (1935): s = Q/(4 pi T) W(u), u = r2 S/(4 T t), W(u) = E1(u); several wells add by superposition. "
          "The drawdown at a well is read at its radius (0.15 m by default), with no well loss.")
ASSUMPTIONS = ("A confined aquifer, homogeneous, isotropic and of infinite extent; wells that fully penetrate it and "
               "pump at a constant rate; water released from storage at once. No recharge, no rivers or boundaries, "
               "no leakage, no delayed yield of a water table. A screening answer, not a well-field design.")


# ── inputs ──────────────────────────────────────────────────────────────────


def _num(name: str, value: Any, limits: tuple[float, float] | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if limits and not limits[0] <= x <= limits[1]:
        raise ValueError(f"{name} must be from {limits[0]:g} to {limits[1]:g}, got {x:g}")
    return x


def wells_from(wells: Iterable[Any]) -> list[dict[str, Any]]:
    """Check a list of wells: each a mapping with ``lon``, ``lat`` and an optional ``rate_m3_day`` (or ``rate``,
    m3/day; :data:`DEFAULTS` otherwise) and ``name``, or a ``[lon, lat]`` / ``[lon, lat, rate]`` list. Returns
    ``[{n, name, lon, lat, rate_m3_day}]``; raises ``ValueError`` on a bad one."""
    out: list[dict[str, Any]] = []
    for k, w in enumerate(wells or [], start=1):
        if isinstance(w, Mapping):
            lon, lat = w.get("lon"), w.get("lat")
            rate = w.get("rate_m3_day", w.get("rate", DEFAULTS["rate_m3_day"]))
            name = str(w.get("name") or f"Well {k}")[:60]
        elif isinstance(w, (list, tuple)) and len(w) in (2, 3):
            lon, lat = w[0], w[1]
            rate = w[2] if len(w) == 3 else DEFAULTS["rate_m3_day"]
            name = f"Well {k}"
        else:
            raise ValueError(f"well {k} must be {{lon, lat, rate_m3_day}} or [lon, lat, rate]")
        lon_f = _num(f"well {k}'s lon", lon, (-180.0, 180.0))
        lat_f = _num(f"well {k}'s lat", lat, (-89.0, 89.0))
        out.append({"n": k, "name": name, "lon": lon_f, "lat": lat_f,
                    "rate_m3_day": _num(f"well {k}'s rate (m3/day)", rate, LIMITS["rate_m3_day"])})
    if not out:
        raise ValueError("drop at least one well")
    if len(out) > MAX_WELLS:
        raise ValueError(f"at most {MAX_WELLS} wells")
    return out


def _aquifer(transmissivity: Any, storativity: Any, days: Any, well_radius: Any) -> tuple[float, float, float, float]:
    return (_num("transmissivity (m2/day)", transmissivity, LIMITS["transmissivity_m2_day"]),
            _num("storativity", storativity, LIMITS["storativity"]),
            _num("days", days, LIMITS["days"]),
            _num("the well radius (m)", well_radius, LIMITS["well_radius_m"]))


# ── the local plane ─────────────────────────────────────────────────────────


class _Plane:
    """Metres east and north of the wells' centre, and back (equirectangular)."""

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0, self.lat0 = lon0, lat0
        self.k = math.cos(math.radians(lat0))

    def xy(self, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return (np.radians(lon - self.lon0) * EARTH_RADIUS_M * self.k, np.radians(lat - self.lat0) * EARTH_RADIUS_M)

    def lonlat(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (self.lon0 + np.degrees(x / (EARTH_RADIUS_M * self.k)), self.lat0 + np.degrees(y / EARTH_RADIUS_M))


def _plane(wells: list[dict[str, Any]]) -> _Plane:
    return _Plane(float(np.mean([w["lon"] for w in wells])), float(np.mean([w["lat"] for w in wells])))


# ── the drawdown ────────────────────────────────────────────────────────────


def _theis(q: float, r: np.ndarray, t: float, s: float, days: float) -> np.ndarray:
    """One well's drawdown at distances ``r`` (already clipped to the well's radius): the aquifer module's W(u)."""
    u = r * r * s / (4.0 * t * days)
    return q / (4.0 * math.pi * t) * _well_function(u)


def _field(wells: list[dict[str, Any]], xw: np.ndarray, yw: np.ndarray, x: np.ndarray, y: np.ndarray, t: float,
           s: float, days: float, rw: float) -> np.ndarray:
    """The total drawdown at points (x, y): every well's Theis drawdown, added."""
    total = np.zeros(np.broadcast(x, y).shape)
    for w, a, b in zip(wells, xw, yw):
        r = np.maximum(np.hypot(x - a, y - b), rw)
        total = total + _theis(w["rate_m3_day"], r, t, s, days)
    return total


def _reach(q: float, t: float, s: float, days: float, level: float) -> float:
    """The distance at which one well pumping ``q`` draws the water down by ``level`` (m), by bisection on u."""
    target = level * 4.0 * math.pi * t / q
    lo, hi = 1e-30, 60.0   # W(1e-30) is about 68.5, W(60) about 1.5e-28
    if float(_well_function(np.array(lo))) < target:
        return 0.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if float(_well_function(np.array(mid))) > target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-12:
            break
    return math.sqrt(4.0 * t * days * lo / s)


def _axis(centres: list[float], reach: float) -> np.ndarray:
    """Grid lines along one axis: even across the whole cone, and closing in on every well geometrically, so the
    tight circles near a well are traced as finely as the wide ones far out."""
    lo, hi = min(centres) - reach, max(centres) + reach
    n_geo = 30 if len(centres) <= 3 else 18
    parts = [np.linspace(lo, hi, 61)]
    ring = np.geomspace(0.5, reach, n_geo)
    for c in centres:
        parts += [c - ring, c + ring, np.array([c])]
    ax = np.unique(np.clip(np.concatenate(parts), lo, hi))
    keep = np.concatenate([[True], np.diff(ax) > 0.05])
    return ax[keep]


def _round_ll(lon: np.ndarray, lat: np.ndarray) -> list[list[float]]:
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]


def _thin(x: np.ndarray, y: np.ndarray, parts: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Keep the points of a line at least 1/``parts`` of its length apart (the tensor grid crowds them where it
    closes in on a well), and its ends: the same line, a fraction of the points."""
    seg = np.hypot(np.diff(x), np.diff(y))
    total = float(seg.sum())
    if len(x) <= 24 or total <= 0:
        return x, y
    step = total / parts
    keep = [0]
    run = 0.0
    for k, ds in enumerate(seg, start=1):
        run += ds
        if run >= step:
            keep.append(k)
            run = 0.0
    if keep[-1] != len(x) - 1:
        keep.append(len(x) - 1)
    return x[keep], y[keep]


def _contours(v: np.ndarray, xs: np.ndarray, ys: np.ndarray, plane: _Plane,
              levels: list[float]) -> list[dict[str, Any]]:
    out = []
    ii, jj = np.arange(len(ys), dtype=float), np.arange(len(xs), dtype=float)
    for level in levels:
        lines = []
        for line in _join(_segments(v, level)):
            if len(line) < 3:
                continue
            fi = np.array([p[0] for p in line])
            fj = np.array([p[1] for p in line])
            x, y = np.interp(fj, jj, xs), np.interp(fi, ii, ys)
            x, y = _thin(x, y)
            lon, lat = plane.lonlat(x, y)
            lines.append(_round_ll(lon, lat))
        if lines:
            out.append({"level": level, "label": f"{level:g} m", "lines": lines})
    return out


def _levels(top: float) -> list[float]:
    levels = [x for x in LEVELS if x < top]
    return levels[:MOST_LEVELS]


def drawdown_at(lon: float, lat: float, wells: Iterable[Any], *, days: float = DEFAULTS["days"],
                transmissivity: float = DEFAULTS["transmissivity_m2_day"],
                storativity: float = DEFAULTS["storativity"],
                well_radius: float = DEFAULTS["well_radius_m"]) -> dict[str, Any]:
    """The drawdown at a point (lon, lat) from the wells, m, with each well's share and distance."""
    ws = wells_from(wells)
    t, s, d, rw = _aquifer(transmissivity, storativity, days, well_radius)
    lon_f, lat_f = _num("lon", lon, (-180.0, 180.0)), _num("lat", lat, (-89.0, 89.0))
    plane = _plane(ws)
    x, y = plane.xy(lon_f, lat_f)
    rows, total = [], 0.0
    for w in ws:
        a, b = plane.xy(w["lon"], w["lat"])
        r = max(float(np.hypot(x - a, y - b)), rw)
        sw = float(_theis(w["rate_m3_day"], np.array(r), t, s, d))
        total += sw
        rows.append({"n": w["n"], "name": w["name"], "distance_m": round(r, 1), "drawdown_m": round(sw, 4)})
    return {"lon": lon_f, "lat": lat_f, "days": d, "drawdown_m": round(total, 4), "wells": rows}


def drawdown_map(wells: Iterable[Any], *, days: float = DEFAULTS["days"],
                 transmissivity: float = DEFAULTS["transmissivity_m2_day"],
                 storativity: float = DEFAULTS["storativity"], well_radius: float = DEFAULTS["well_radius_m"],
                 frames: bool | Iterable[float] = False, image: bool = True) -> dict[str, Any]:
    """The drawdown cone of one or several pumping wells after ``days`` of pumping, for the map.

    ``wells``: see :func:`wells_from`. Returns a JSON-safe dict: ``wells`` (each with ``at_well_m``, its drawdown at
    its face, ``own_m`` its own share and ``others_m`` what the other wells add there), ``contours`` (``[{level,
    label, lines}]``, lines as [[lon, lat], ...], at :data:`LEVELS` from 5 cm up), ``bbox``, ``reach_m`` (how far the
    5 cm line reaches from the wells, the largest well alone), ``image`` (a coarse grid of the drawdown over the bbox,
    north row first, for the fill), the aquifer, the method, the assumptions, the defaults and their ranges with
    their source. ``frames`` (True, or a list of days) adds the cone at those times (by default :data:`FRAME_DAYS` up
    to ``days``), for the time bar, each with its own image when ``image``. Numbers from code."""
    ws = wells_from(wells)
    t, s, d, rw = _aquifer(transmissivity, storativity, days, well_radius)
    plane = _plane(ws)
    xw, yw = plane.xy([w["lon"] for w in ws], [w["lat"] for w in ws])
    q_max = max(w["rate_m3_day"] for w in ws)
    q_sum = sum(w["rate_m3_day"] for w in ws)
    reach = _reach(q_max, t, s, d, EDGE_M)
    # The grid reaches past the largest well's 5 cm line by the rest of the wells' rate too (their sum is the bound).
    span = min(max(_reach(q_sum, t, s, d, EDGE_M) * 1.15, 200.0), MAX_REACH_M)
    xs, ys = _axis(list(xw), span), _axis(list(yw), span)
    gx, gy = np.meshgrid(xs, ys)
    v = _field(ws, xw, yw, gx, gy, t, s, d, rw)
    # each well's own drawdown at its face, and what the others add there
    rows = []
    for k, w in enumerate(ws):
        own = float(_theis(w["rate_m3_day"], np.array(rw), t, s, d))
        total = float(_field(ws, xw, yw, np.array(xw[k]), np.array(yw[k]), t, s, d, rw))
        rows.append({**w, "rate_l_s": round(w["rate_m3_day"] / 86.4, 2), "at_well_m": round(total, 3),
                     "own_m": round(own, 3), "others_m": round(total - own, 3),
                     "reach_m": round(_reach(w["rate_m3_day"], t, s, d, EDGE_M), 1)})
    levels = _levels(float(np.nanmax(v)))
    lon_lo, lat_lo = plane.lonlat(xs[0], ys[0])
    lon_hi, lat_hi = plane.lonlat(xs[-1], ys[-1])
    bbox = [round(float(lon_lo), 6), round(float(lat_lo), 6), round(float(lon_hi), 6), round(float(lat_hi), 6)]
    out: dict[str, Any] = {
        "ok": True, "wells": rows, "days": d, "transmissivity_m2_day": t, "storativity": s, "well_radius_m": rw,
        "levels": levels, "contours": _contours(v, xs, ys, plane, levels), "bbox": bbox,
        "reach_m": round(reach, 1), "cut_at_m": MAX_REACH_M if span >= MAX_REACH_M else None,
        "max_m": round(max(r["at_well_m"] for r in rows), 3),
        "method": METHOD, "assumptions": ASSUMPTIONS, "defaults": dict(DEFAULTS),
        "ranges": {k: list(v_) for k, v_ in RANGES.items()}, "ranges_source": RANGES_SOURCE,
        "check": CHECK_LINE,
    }
    gxs = np.linspace(xs[0], xs[-1], IMAGE_CELLS)
    gys = np.linspace(ys[-1], ys[0], IMAGE_CELLS)   # north row first, as an image
    ix, iy = np.meshgrid(gxs, gys)

    def picture(days_: float) -> dict[str, Any]:
        img = _field(ws, xw, yw, ix, iy, t, s, days_, rw)
        return {"bbox": bbox, "nx": IMAGE_CELLS, "ny": IMAGE_CELLS, "values": [round(float(x), 3) for x in img.ravel()]}

    if image:
        out["image"] = picture(d)
    if frames:
        times = sorted({float(x) for x in (FRAME_DAYS if frames is True else frames) if 0 < float(x) <= d} | {d})
        out["frames"] = []
        for tt in times:
            fv = _field(ws, xw, yw, gx, gy, t, s, tt, rw)
            out["frames"].append({"days": tt, "contours": _contours(fv, xs, ys, plane, levels),
                                  "max_m": round(float(max(_field(ws, xw, yw, np.array(a), np.array(b), t, s, tt, rw)
                                                           for a, b in zip(xw, yw))), 3),
                                  "reach_m": round(_reach(q_max, t, s, tt, EDGE_M), 1),
                                  **({"image": picture(tt)} if image else {})})
    out["summary"] = summary(out)
    return out


def _dist(m: float) -> str:
    return f"{m / 1000:,.1f} km" if m >= 1000 else f"{m:,.0f} m"


def summary(res: Mapping[str, Any]) -> str:
    """One plain sentence about a :func:`drawdown_map` result."""
    ws = res["wells"]
    deep = max(ws, key=lambda w: w["at_well_m"])
    n = len(ws)
    who = "the well" if n == 1 else f"{deep['name']}, the deepest of {n} wells,"
    return (f"After {res['days']:g} days, {who} draws the water down {deep['at_well_m']:.2f} m at its face; the "
            f"5 cm line reaches {_dist(res['reach_m'])} out (T {res['transmissivity_m2_day']:g} m2/day, "
            f"S {res['storativity']:g}).")


# ── the check ───────────────────────────────────────────────────────────────

#: Fetter (2001), Applied Hydrogeology, 4th ed., the worked example of the Theis equation (section on the Theis
#: solution, just before "Flow in a leaky, confined aquifer"): K 14.9 m/day over 20.1 m, so T = 299 m2/day as the
#: book rounds it, S 0.0051, Q 2725 m3/day, r 7.0 m, t 1 day. The book prints u = 0.00021, W(u) = 7.94 and 5.7 m.
FETTER = {"rate_m3_day": 2725.0, "transmissivity_m2_day": 299.0, "storativity": 0.0051, "radius_m": 7.0,
          "days": 1.0, "printed_u": 0.00021, "printed_w": 7.94, "printed_m": 5.7}
#: Lohman (1972), USGS Professional Paper 708, Table 5 (p. 16, from Ferris, Knowles, Brown and Stallman, 1962,
#: p. 96): W(u) for u = N x 10^k, the rows N = 1.0 and N = 2.0, k from -15 to 0, as printed.
PP708_TABLE5 = {
    1.0: (33.9616, 31.6590, 29.3564, 27.0538, 24.7512, 22.4486, 20.1460, 17.8435, 15.5409, 13.2383, 10.9357, 8.6332,
          6.3315, 4.0379, 1.8229, 0.2194),
    2.0: (33.2684, 30.9658, 28.6632, 26.3607, 24.0581, 21.7555, 19.4529, 17.1503, 14.8477, 12.5451, 10.2426, 7.9402,
          5.6394, 3.3547, 1.2227, 0.04890),
}
CHECK_LINE = ("Checked against Fetter (2001), Applied Hydrogeology, 4th ed., the Theis worked example (5.7 m at 7 m "
              "after a day; the code gives 5.73 m), and USGS Professional Paper 708, Table 5 (W(u), 32 values to "
              "four decimals).")


def check() -> dict[str, Any]:
    """The checks, recomputed: Fetter's worked example, the USGS table of W(u), and superposition (two equal wells
    give the same drawdown at mirror points; the drawdown at one well due to the other is the same both ways; the
    sum of wells pumped together equals the wells pumped alone, added). Returns ``{ok, rows, superposition}``."""
    f = FETTER
    u = f["radius_m"] ** 2 * f["storativity"] / (4 * f["transmissivity_m2_day"] * f["days"])
    s = float(theis_drawdown(f["transmissivity_m2_day"], f["storativity"], f["rate_m3_day"], f["radius_m"],
                             f["days"])[0])
    rows = [{"what": "Fetter (2001), Theis worked example: drawdown at 7.0 m after 1 day", "printed": f["printed_m"],
             "computed": round(s, 4), "unit": "m", "ok": round(s, 1) == f["printed_m"],
             "note": f"u = {u:.3g} (printed 0.00021), W(u) = {float(_well_function(np.array(u))):.3f} (printed 7.94 "
                     "for u = 2e-4, read from the table)"}]
    worst = 0.0
    for n, row in PP708_TABLE5.items():
        for k, printed in zip(range(-15, 1), row):
            w = float(_well_function(np.array(n * 10.0 ** k)))
            worst = max(worst, abs(round(w, 4 if printed >= 0.1 else 5) - printed))
    rows.append({"what": "USGS PP 708, Table 5: W(u) for u = 1e-15 to 2, 32 values", "printed": "4 decimals",
                 "computed": f"largest difference {worst:.1e}", "unit": "", "ok": worst < 1.5e-4})
    # superposition: two wells 1 km apart, equal rates
    a = {"lon": 0.0, "lat": 45.0, "rate_m3_day": 1500.0}
    b = {"lon": 0.0127, "lat": 45.0, "rate_m3_day": 1500.0}
    kw = {"days": 10.0}
    left = drawdown_at(-0.005, 45.003, [a, b], **kw)["drawdown_m"]
    right = drawdown_at(0.0177, 45.003, [a, b], **kw)["drawdown_m"]
    ab = drawdown_at(b["lon"], b["lat"], [a], **kw)["drawdown_m"]
    ba = drawdown_at(a["lon"], a["lat"], [b], **kw)["drawdown_m"]
    both = drawdown_at(0.004, 44.998, [a, b], **kw)["drawdown_m"]
    alone = (drawdown_at(0.004, 44.998, [a], **kw)["drawdown_m"]
             + drawdown_at(0.004, 44.998, [b], **kw)["drawdown_m"])
    sup = {"mirror_m": [left, right], "reciprocal_m": [ab, ba], "together_m": both, "alone_added_m": round(alone, 4),
           "ok": abs(left - right) < 1e-3 and abs(ab - ba) < 1e-9 and abs(both - alone) < 1e-3}
    rows.append({"what": "Superposition: mirror points, reciprocity, together = alone added", "printed": "equal",
                 "computed": f"{left:.4f} / {right:.4f} m; {ab:.4f} / {ba:.4f} m; {both:.4f} / {alone:.4f} m",
                 "unit": "m", "ok": sup["ok"]})
    return {"ok": all(r["ok"] for r in rows), "rows": rows, "superposition": sup,
            "sources": [
                "Fetter, C.W. (2001). Applied Hydrogeology, 4th edition. Prentice Hall.",
                "Lohman, S.W. (1972). Ground-water hydraulics. USGS Professional Paper 708, Table 5, p. 16. "
                "https://pubs.usgs.gov/pp/0708/report.pdf",
            ]}
