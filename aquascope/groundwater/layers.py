"""A MODFLOW model's layers, a cross-section through them, and a cell's heads over time (#625, #626). Experimental.

What ModelMuse and Groundwater Vistas show beside the plan view, computed here so the Explorer, the CLI and an
assistant get the same numbers:

- :func:`read_layer_elevations`: each layer's top and bottom from the grid file (TOP and BOTM of a MODFLOW 6
  ``.dis``/``.disv`` or ``.dis.grb``, or a MODFLOW-2005 ``.dis``, where confining beds under a layer are skipped).
- :func:`model_layers`: those elevations on the cells as :func:`~aquascope.groundwater.surface.read_modflow_heads`
  draws them (block means on a big grid), with the model's base and crest, and a vertical exaggeration for a 3D
  view, chosen so the stack stands about :data:`STACK_SHARE` of the model's width and rounded to 1, 2 or 5 times a
  power of ten. The exaggeration is a drawing choice, said wherever the stack is drawn.
- :func:`cross_section`: a line drawn over the model, cut through every layer: the cells it crosses (the edges
  found by bisection to :data:`EDGE_TOL_M`), their layer tops and bottoms, every layer's head at every saved time,
  and the water table, read as the head of the uppermost layer that has one (a dry or inactive layer is skipped).
- :func:`cell_hydrograph`: one cell's head over time in every layer, with the layer's top and bottom.

Distances along a line are on a local equirectangular plane around it, which over a model's few tens of kilometres
differs from the geodesic distance by well under one percent. Elevations and heads are in the model's length unit.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from aquascope.groundwater.surface import (
    EARTH_RADIUS_M,
    LENGTH_UNITS_M,
    _array,
    _as_bytes,
    _blocks,
    _dims,
    _numbers,
    _strip_comment,
)

#: The 3D stack is drawn about this share of the model's horizontal extent tall.
STACK_SHARE = 0.12
#: A cross-section's cell edges are found to within this many metres along the line.
EDGE_TOL_M = 0.5
#: At most this many points are tested along a cross-section's line.
MAX_SAMPLES = 4000


# ── the layers' tops and bottoms ────────────────────────────────────────────


def _grb_values(buf: bytes) -> dict[str, Any]:
    """Every array of a MODFLOW 6 binary grid file, by name (the same layout surface._grb reads)."""
    def line(k: int, size: int, at: int) -> str:
        return buf[at + k * size: at + (k + 1) * size].decode("ascii", "replace").strip()

    ntxt = int(line(2, 50, 0).split()[1])
    lentxt = int(line(3, 50, 0).split()[1])
    pos = 200
    defs = []
    for k in range(ntxt):
        txt = line(k, lentxt, pos).split("#")[0].split()
        defs.append((txt[0].upper(), txt[1].upper(), [int(x) for x in txt[4:4 + int(txt[3])]]))
    pos += ntxt * lentxt
    vals: dict[str, Any] = {}
    for name, typ, dims in defs:
        count = int(np.prod(dims)) if dims else 1
        if typ == "CHARACTER":
            vals[name] = buf[pos:pos + count].decode("ascii", "replace").strip()
            pos += count
            continue
        dt, size = ("<i4", 4) if typ == "INTEGER" else ("<f8", 8)
        vals[name] = np.frombuffer(buf, dt, count, pos)
        pos += size * count
    vals["_KIND"] = line(0, 50, 0).split()[1].upper()
    return vals


def _mf6_griddata(text: str) -> dict[str, Any]:
    b = _blocks(text)
    d = _dims(b.get("DIMENSIONS", []))
    nlay = d.get("NLAY", 1)
    ncpl = d["NCPL"] if "NCPL" in d else d.get("NROW", 0) * d.get("NCOL", 0)
    if not ncpl:
        raise ValueError("the grid file has no NROW and NCOL (or NCPL) in its DIMENSIONS block")
    lines = b.get("GRIDDATA", [])
    got: dict[str, np.ndarray] = {}
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        key = toks[0].upper()
        if key in ("TOP", "BOTM", "IDOMAIN"):
            n = ncpl if key == "TOP" else ncpl * nlay
            if key != "TOP" and len(toks) > 1 and toks[1].upper() == "LAYERED":
                parts, j = [], i + 1
                for _ in range(nlay):
                    a, j = _array(lines, j, ncpl, mf6=True)
                    parts.append(a)
                got[key], i = np.concatenate(parts), j
            else:
                got[key], i = _array(lines, i + 1, n, mf6=True)
        else:
            i += 1
    if "TOP" not in got or "BOTM" not in got:
        raise ValueError("the grid file has no TOP or BOTM array")
    botm = got["BOTM"].reshape(nlay, ncpl)
    return {"nlay": nlay, "ncpl": ncpl, "top": got["TOP"], "botm": botm, "tops": _stack_tops(got["TOP"], botm),
            "idomain": got["IDOMAIN"].reshape(nlay, ncpl).astype(int) if "IDOMAIN" in got else None,
            "from": "MODFLOW 6 grid file"}


def _mf2005_griddata(text: str) -> dict[str, Any]:
    lines = [ln for ln in text.splitlines() if _strip_comment(ln)]
    t = _strip_comment(lines[0]).split()
    nlay, nrow, ncol = int(t[0]), int(t[1]), int(t[2])
    laycbd, i = _numbers(lines, 1, nlay)
    _, i = _array(lines, i, ncol, mf6=False)  # DELR
    _, i = _array(lines, i, nrow, mf6=False)  # DELC
    n = nrow * ncol
    top, i = _array(lines, i, n, mf6=False)
    botm, tops = [], [top]
    for k in range(nlay):
        b, i = _array(lines, i, n, mf6=False)
        botm.append(b)
        if laycbd[k]:
            # The confining bed under layer k is not a model layer: the next layer starts at its bottom.
            cb, i = _array(lines, i, n, mf6=False)
            tops.append(cb)
        else:
            tops.append(b)
    return {"nlay": nlay, "ncpl": n, "top": top, "botm": np.vstack(botm), "tops": np.vstack(tops[:nlay]),
            "idomain": None, "laycbd": [int(x) for x in laycbd], "from": "MODFLOW-2005 DIS file"}


def _stack_tops(top: np.ndarray, botm: np.ndarray) -> np.ndarray:
    """Each layer's top: the model top, then the bottom of the layer above."""
    return np.vstack([np.asarray(top, float)[None, :], np.asarray(botm, float)[:-1]])


def read_layer_elevations(grid: Any) -> dict[str, Any]:
    """Each layer's top and bottom from a MODFLOW grid file (bytes or a path).

    Returns {nlay, ncpl, top (ncpl), botm (nlay x ncpl), tops (nlay x ncpl: each layer's own top, which is the
    bottom of a MODFLOW-2005 confining bed where the layer above has one), idomain (or None), from}, in the
    model's length unit and the cell order of the head file."""
    buf = _as_bytes(grid)
    if buf[:5] == b"GRID ":
        v = _grb_values(buf)
        nlay = int(v["NLAY"][0])
        ncpl = int(v["NCPL"][0]) if "NCPL" in v else int(v["NROW"][0]) * int(v["NCOL"][0])
        if "TOP" not in v or "BOTM" not in v:
            raise ValueError("the binary grid file has no TOP or BOTM")
        idom = v.get("IDOMAIN")
        top = np.asarray(v["TOP"], float)[:ncpl]
        botm = np.asarray(v["BOTM"], float).reshape(nlay, ncpl)
        return {"nlay": nlay, "ncpl": ncpl, "top": top, "botm": botm, "tops": _stack_tops(top, botm),
                "idomain": np.asarray(idom, int).reshape(nlay, ncpl) if idom is not None and len(idom) == nlay * ncpl
                else None, "from": "MODFLOW 6 binary grid file"}
    text = buf.decode("utf-8", "replace")
    if re.search(r"^\s*BEGIN\s+", text, re.IGNORECASE | re.MULTILINE):
        return _mf6_griddata(text)
    return _mf2005_griddata(text)


def _block_mean(a: np.ndarray, nrow: int, ncol: int, k: int) -> np.ndarray:
    br, bc = math.ceil(nrow / k), math.ceil(ncol / k)
    pad = np.full((br * k, bc * k), np.nan)
    pad[:nrow, :ncol] = a.reshape(nrow, ncol)
    with np.errstate(invalid="ignore"):
        return np.nanmean(pad.reshape(br, k, bc, k), axis=(1, 3)).ravel()


def nice_exaggeration(raw: float) -> int:
    """The largest of 1, 2, 5, 10, 20, 50, ... not above ``raw`` (at least 1)."""
    if not (raw > 1):
        return 1
    best = 1
    p = 1
    while p <= raw:
        for m in (1, 2, 5):
            if m * p <= raw:
                best = m * p
        p *= 10
    return int(best)


def _local_xy(lon: np.ndarray, lat: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.radians(np.asarray(lon, float) - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat, float) - lat0) * EARTH_RADIUS_M
    return x, y


def model_layers(res: dict[str, Any], grid: Any) -> dict[str, Any]:
    """The layers of a model read by :func:`~aquascope.groundwater.surface.read_modflow_heads`, on its drawn cells.

    ``grid``: the same grid file (bytes or a path), or :func:`read_layer_elevations`'s result. Returns {layers
    (1-based), top (cells), botm (layers x cells), base, crest, thickness, metres_per_unit, extent_m, exaggeration,
    note, from}; ``top``/``botm`` are NaN where a layer has no cell (IDOMAIN 0 or below)."""
    if res.get("error"):
        raise ValueError(res["error"])
    el = grid if isinstance(grid, dict) and "botm" in grid else read_layer_elevations(grid)
    g = res["grid"]
    ncell = int(g["cells"])
    if int(el["ncpl"]) != ncell:
        raise ValueError(f"the grid file has {el['ncpl']} cells per layer but the heads {ncell}: the same model?")
    top = np.asarray(el["top"], float).copy()
    botm = np.asarray(el["botm"], float).copy()
    tops = (np.asarray(el["tops"], float) if el.get("tops") is not None else _stack_tops(top, botm)).copy()
    if el.get("idomain") is not None:
        off = np.asarray(el["idomain"]) <= 0
        botm[off] = np.nan
        tops[off] = np.nan
        if off.all(axis=0).any():
            top[off.all(axis=0)] = np.nan
    block = int(g.get("block") or 1)
    if block > 1:
        nrow, ncol = int(g["nrow"]), int(g["ncol"])
        top = _block_mean(top, nrow, ncol, block)
        botm = np.vstack([_block_mean(b, nrow, ncol, block) for b in botm])
        tops = np.vstack([_block_mean(b, nrow, ncol, block) for b in tops])
    finite = np.concatenate([top[np.isfinite(top)], botm[np.isfinite(botm)]])
    if not finite.size:
        raise ValueError("the grid file has no layer elevations")
    base, crest = float(finite.min()), float(finite.max())
    units = (g.get("length_units") or "").lower()
    m_per = LENGTH_UNITS_M.get(units, 1.0)
    rings = np.asarray(res["rings"], float)
    lat0 = float(rings[:, 1].mean())
    x, y = _local_xy(rings[:, 0], rings[:, 1], lat0, float(rings[:, 0].mean()))
    extent = float(math.hypot(x.max() - x.min(), y.max() - y.min()))
    thick_m = (crest - base) * m_per
    ve = nice_exaggeration(STACK_SHARE * extent / thick_m) if thick_m > 0 else 1
    return {
        "kind": "modflow_layers", "layers": list(range(1, int(el["nlay"]) + 1)), "top": top, "botm": botm,
        "tops": tops,
        "base": round(base, 4), "crest": round(crest, 4), "thickness": round(crest - base, 4),
        "metres_per_unit": m_per, "length_units": units or None, "extent_m": round(extent, 1), "exaggeration": ve,
        "note": (f"Vertical exaggeration x{ve}: the layers are drawn {ve} times taller than they are, from the "
                 f"model's base ({base:.1f}{' ' + units if units else ''})."),
        "from": el["from"], "block": block,
    }


def layers_summary(lay: dict[str, Any]) -> dict[str, Any]:
    """:func:`model_layers` without its arrays, with each layer's mean top, bottom and thickness."""
    out = {k: v for k, v in lay.items() if k not in ("top", "botm", "tops")}
    rows = []
    for i, k in enumerate(lay["layers"]):
        upper = layer_top(lay, i)
        th = upper - lay["botm"][i]
        ok = np.isfinite(th)
        rows.append({"layer": k, "mean_top": round(float(np.nanmean(upper)), 3) if ok.any() else None,
                     "mean_bottom": round(float(np.nanmean(lay["botm"][i])), 3) if ok.any() else None,
                     "mean_thickness": round(float(th[ok].mean()), 3) if ok.any() else None})
    out["per_layer"] = rows
    return out


def layer_top(lay: dict[str, Any], i: int) -> np.ndarray:
    """Layer ``i``'s top (0-based): the model top for the first, the bottom of the one above (or of its confining
    bed, in a MODFLOW-2005 model with LAYCBD) for the others."""
    if lay.get("tops") is not None:
        return lay["tops"][i]
    return lay["top"] if i == 0 else lay["botm"][i - 1]


# ── a cross-section along a line ────────────────────────────────────────────


def _cell_index(res: dict[str, Any], lat0: float, lon0: float) -> dict[str, Any]:
    rings = np.asarray(res["rings"], float)
    offs = np.asarray(res["ring_offsets"]).astype(np.intp)  # intp: reduceat refuses int64 on 32-bit (Pyodide)
    x, y = _local_xy(rings[:, 0], rings[:, 1], lat0, lon0)
    starts = offs[:-1]
    xmin, xmax = np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts)
    ymin, ymax = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    size = float(np.median(np.sqrt(np.maximum((xmax - xmin) * (ymax - ymin), 1e-6))))
    bucket = max(size * 2, 1.0)
    index: dict[tuple[int, int], list[int]] = {}
    i0, i1 = np.floor(xmin / bucket).astype(int), np.floor(xmax / bucket).astype(int)
    j0, j1 = np.floor(ymin / bucket).astype(int), np.floor(ymax / bucket).astype(int)
    for c in range(len(starts)):
        for bi in range(i0[c], i1[c] + 1):
            for bj in range(j0[c], j1[c] + 1):
                index.setdefault((bi, bj), []).append(c)
    return {"x": x, "y": y, "offs": offs, "bucket": bucket, "index": index, "size": size,
            "box": (xmin, xmax, ymin, ymax)}


def _inside(ix: dict[str, Any], c: int, px: float, py: float) -> bool:
    a, b = ix["offs"][c], ix["offs"][c + 1]
    xs, ys = ix["x"][a:b], ix["y"][a:b]
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > py) != (ys[j] > py) and px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i] + 1e-300) + xs[i]:
            inside = not inside
        j = i
    return inside


def _find(ix: dict[str, Any], px: float, py: float, hint: int = -1) -> int:
    if hint >= 0 and _inside(ix, hint, px, py):
        return hint
    key = (int(math.floor(px / ix["bucket"])), int(math.floor(py / ix["bucket"])))
    for c in ix["index"].get(key, ()):
        xmin, xmax, ymin, ymax = (b[c] for b in ix["box"])
        if xmin <= px <= xmax and ymin <= py <= ymax and _inside(ix, c, px, py):
            return c
    return -1


def cross_section(res: dict[str, Any], layers: dict[str, Any], line: Any, *,
                  max_samples: int = MAX_SAMPLES) -> dict[str, Any]:
    """Cut the model along ``line`` ([[lon, lat], ...], two points or more) through every layer.

    ``res``: :func:`~aquascope.groundwater.surface.read_modflow_heads`'s result; ``layers``: :func:`model_layers`'s.
    Returns a dict: ``segments`` (each {cell, row, col, from_m, to_m}: the cells crossed, in order, with where the
    line enters and leaves each), ``top`` and ``bottom`` (layer x segment), ``heads`` (time x layer x segment, NaN for
    no head; layers are the grid's, a layer the head file does not write is all NaN), ``water_table`` (time x
    segment), ``length_m``, ``vertices_m`` (where the line bends), ``times``, ``dates``, ``layers``, ``unit`` and
    notes. Parts of the line outside the model are left out of ``segments``."""
    pts = np.asarray(line, float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("the line is a list of two points or more, [[lon, lat], ...]")
    lat0, lon0 = float(pts[:, 1].mean()), float(pts[:, 0].mean())
    ix = _cell_index(res, lat0, lon0)
    lx, ly = _local_xy(pts[:, 0], pts[:, 1], lat0, lon0)
    seglen = np.hypot(np.diff(lx), np.diff(ly))
    total = float(seglen.sum())
    if total <= 0:
        raise ValueError("the line has no length")
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    step = max(ix["size"] / 4.0, total / max(10, int(max_samples)))
    d = np.unique(np.concatenate([np.arange(0.0, total, step), cum, [total]]))

    def at(dist: float) -> tuple[float, float]:
        k = min(int(np.searchsorted(cum, dist, side="right")) - 1, len(seglen) - 1)
        f = (dist - cum[k]) / seglen[k] if seglen[k] > 0 else 0.0
        return float(lx[k] + f * (lx[k + 1] - lx[k])), float(ly[k] + f * (ly[k + 1] - ly[k]))

    cells = []
    hint = -1
    for dist in d:
        hint = _find(ix, *at(float(dist)), hint)
        cells.append(hint)

    def edge(d_a: float, d_b: float, c_a: int) -> float:
        lo, hi = d_a, d_b
        while hi - lo > EDGE_TOL_M:
            mid = (lo + hi) / 2
            if c_a >= 0 and _inside(ix, c_a, *at(mid)):
                lo = mid
            elif c_a < 0 and _find(ix, *at(mid)) < 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    segs: list[list[float]] = []
    start = 0.0
    for k in range(1, len(d)):
        if cells[k] != cells[k - 1]:
            e = edge(float(d[k - 1]), float(d[k]), cells[k - 1])
            if cells[k - 1] >= 0:
                segs.append([cells[k - 1], start, e])
            start = e
    if cells[-1] >= 0:
        segs.append([cells[-1], start, total])
    if not segs:
        return {"kind": "modflow_section", "error": "The line does not cross the model.", "length_m": round(total, 1)}
    idx = np.array([int(s[0]) for s in segs])
    g = res["grid"]
    nl = len(layers["layers"])
    top = np.vstack([layer_top(layers, i)[idx] for i in range(nl)])
    bottom = np.vstack([layers["botm"][i][idx] for i in range(nl)])
    heads_in = np.asarray(res["heads"], float)
    nt = heads_in.shape[0]
    heads = np.full((nt, nl, len(idx)), np.nan)
    for hi, lay in enumerate(res["layers"]):
        if 1 <= lay <= nl:
            heads[:, lay - 1, :] = heads_in[:, hi, :][:, idx]
    wt = np.full((nt, len(idx)), np.nan)
    for i in range(nl - 1, -1, -1):  # the uppermost layer with a head wins
        h = heads[:, i, :]
        wt = np.where(np.isfinite(h), h, wt)
    structured = g.get("type") == "dis" and int(g.get("block") or 1) == 1
    ncol = int(g.get("drawn_ncol") or g.get("ncol") or 1)
    segments = [{"cell": int(c), "row": int(c // ncol) + 1 if structured else None,
                 "col": int(c % ncol) + 1 if structured else None,
                 "from_m": round(float(a), 2), "to_m": round(float(b), 2)} for c, a, b in segs]
    m_per = float(layers.get("metres_per_unit") or 1.0)
    notes = []
    if any(s["from_m"] > 0.01 for s in segments[:1]) or segments[-1]["to_m"] < total - 0.01:
        notes.append("Parts of the line outside the model are left blank.")
    if int(g.get("block") or 1) > 1:
        notes.append(f"Cut through blocks of {g['block']} x {g['block']} cells (their mean), as the map draws them.")
    with np.errstate(invalid="ignore"):
        above = np.isfinite(wt) & np.isfinite(top[0]) & (wt > top[0][None, :])
    if above.any():
        notes.append("Where the line stands above the top of the model, the uppermost layer's head is above that layer's "
                     "top: the layer is confined there (or its top is not the ground), so the line is its head, a "
                     "potentiometric surface, not a free water table.")
    return {
        "kind": "modflow_section", "segments": segments, "top": top, "bottom": bottom, "heads": heads,
        "water_table": wt, "length_m": round(total, 1), "vertices_m": [round(float(v), 1) for v in cum],
        "layers": layers["layers"], "times": res.get("times"), "dates": res.get("dates"),
        "unit": layers.get("length_units"), "metres_per_unit": m_per, "notes": notes,
        "water_table_rule": "The water table is the head of the uppermost layer that has one (dry and inactive cells "
                            "have none).",
        "summary": f"A {total / 1000:.2f} km section across {len(segments)} cells and {nl} layer(s).",
    }


def section_summary(sec: dict[str, Any], time_index: int = -1) -> dict[str, Any]:
    """A cross-section without its arrays, with one time's heads and water table per segment (rounded)."""
    out = {k: v for k, v in sec.items() if k not in ("top", "bottom", "heads", "water_table")}
    if sec.get("error"):
        return out
    t = int(time_index) % sec["heads"].shape[0]
    r = lambda a: [None if not math.isfinite(v) else round(float(v), 3) for v in a]  # noqa: E731
    out["time_index"] = t
    out["top"] = [r(a) for a in sec["top"]]
    out["bottom"] = [r(a) for a in sec["bottom"]]
    out["heads_at_time"] = [r(a) for a in sec["heads"][t]]
    out["water_table_at_time"] = r(sec["water_table"][t])
    return out


# ── a cell's heads over time ────────────────────────────────────────────────


def cell_hydrograph(res: dict[str, Any], cell: int, layers: dict[str, Any] | None = None) -> dict[str, Any]:
    """One drawn cell's head over time in every layer the head file writes.

    ``cell`` is 0-based in the order of ``res["rings"]``. Returns {cell, row, col, times, dates, series (each {layer,
    head (list, None for no head), min, max, first, last, change, min_at, top, bottom}), unit, summary}."""
    heads = np.asarray(res["heads"], float)
    ncell = heads.shape[2]
    c = int(cell)
    if not 0 <= c < ncell:
        raise ValueError(f"cell out of range (0 to {ncell - 1})")
    g = res["grid"]
    structured = g.get("type") == "dis" and int(g.get("block") or 1) == 1
    ncol = int(g.get("drawn_ncol") or g.get("ncol") or 1)
    dates = res.get("dates")
    times = res.get("times") or list(range(heads.shape[0]))
    series = []
    for li, lay in enumerate(res["layers"]):
        h = heads[:, li, c]
        ok = np.isfinite(h)
        s: dict[str, Any] = {"layer": int(lay), "head": [round(float(v), 4) if math.isfinite(v) else None for v in h],
                             "min": None, "max": None, "first": None, "last": None, "change": None, "min_at": None,
                             "top": None, "bottom": None}
        if ok.any():
            v = h[ok]
            k = int(np.nanargmin(np.where(ok, h, np.inf)))
            s.update(min=round(float(v.min()), 4), max=round(float(v.max()), 4), first=round(float(v[0]), 4),
                     last=round(float(v[-1]), 4), change=round(float(v[-1] - v[0]), 4),
                     min_at=(dates[k] if dates else times[k]))
        if layers is not None and 1 <= lay <= len(layers["layers"]):
            tp, bt = layer_top(layers, lay - 1)[c], layers["botm"][lay - 1][c]
            s["top"] = round(float(tp), 4) if math.isfinite(tp) else None
            s["bottom"] = round(float(bt), 4) if math.isfinite(bt) else None
        series.append(s)
    with_head = [s for s in series if s["min"] is not None]
    unit = (g.get("length_units") or "") or None
    words = "; ".join(f"layer {s['layer']} {s['first']:.2f} to {s['last']:.2f}" for s in with_head)
    return {
        "kind": "modflow_hydrograph", "cell": c, "row": int(c // ncol) + 1 if structured else None,
        "col": int(c % ncol) + 1 if structured else None, "times": list(times), "dates": dates, "series": series,
        "unit": unit,
        "summary": (f"Head over {len(times)} saved time(s): {words}." if with_head
                    else "No head in any layer at this cell (inactive or dry)."),
    }


__all__ = [
    "EDGE_TOL_M", "MAX_SAMPLES", "STACK_SHARE", "cell_hydrograph", "cross_section", "layer_top", "layers_summary",
    "model_layers", "nice_exaggeration", "read_layer_elevations", "section_summary",
]
