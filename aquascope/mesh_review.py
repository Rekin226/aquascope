"""Review my model results (#564): checks on a model run on a mesh, each a finding with a place, a value and a source.

A modeller's first question about a run is "does it look right?". This module answers with checks made by code
on a :class:`aquascope.io.mesh.Mesh` (a HEC-RAS 2D plan or a UGRID result), the same for the Explorer (the
file stays in the browser), ``aquascope mesh review FILE`` and the MCP tool ``mesh_review``:

* ``deepest``: the deepest cells, with their depth and the model time it was reached (for a file without bed
  levels, the largest rise of the water surface instead, said so).
* ``negative``: cells whose depth goes below zero (the water surface under the bed) at some step.
* ``tiny``: cells wetted only by a film, a largest depth under :data:`TINY_DEPTH_M` (rain on the grid, a
  terrain or tolerance problem).
* ``isolated``: cells wet at their largest while every neighbour stays dry.
* ``flicker``: cells that go wet and dry again and again (:data:`FLICKER_TRANSITIONS` changes or more).
* ``mass_balance``: the volume error the model wrote itself (HEC-RAS volume accounting), where it exists.
* ``floods_radar`` and ``floods_news``: the modelled largest extent against floods seen by Sentinel-1 radar
  (Microsoft AI for Good, October 2014 to September 2024) and floods in the news (Groundsource), from the
  Archive's context mirror. When the run's months are outside a record, the comparison is made with every
  month on record instead, and every finding says so: then it tells where floods have been seen, not
  whether this event was right.
* ``gauges``: the modelled peak against the observed peak at each Archive gauge inside the mesh (timing only:
  a gauge measures flow or stage, the mesh depth), or a line saying there is none.

Every number is formatted here into a finding's ``facts``. The keyless titles and reasons are templates over
those facts. A model (the reader's own key, or the on-device model) may only order the findings and word them,
writing each number as a placeholder (``{f3}``) that code fills in; :func:`aquascope.map_explain.claim_lock`
(Explain's claim lock, #562) refuses wording with any number its facts do not hold, and a ranking word the
rules wording does not use is refused too.

numpy and the standard library only, so it runs in the Explorer's mesh worker beside the reader.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from aquascope.io.mesh import DRY_DEPTH, Mesh, MeshError

__all__ = [
    "CHECKS", "SEVERITIES", "apply_words", "cells_at", "check_volume", "gather_context", "gather_floods",
    "gauges_inside", "model_words", "review", "review_file", "review_prompt",
]

#: The checks, in the order the review lists them, with the words a reader sees for each.
CHECKS: dict[str, str] = {
    "negative": "Negative depths",
    "mass_balance": "Volume balance",
    "isolated": "Isolated wet cells",
    "flicker": "Wet and dry flicker",
    "tiny": "Very shallow water",
    "floods_radar": "Floods seen by radar",
    "floods_news": "Floods in the news",
    "gauges": "Gauges in the mesh",
    "deepest": "Deepest water",
}
SEVERITIES = ("problem", "warning", "info")

#: Thresholds, in metres (turned into feet for a model in feet).
TINY_DEPTH_M = 0.03       # a largest depth under this is a film
TINY_FLOOR_M = 0.001      # under this a cell is dry, not a film (rounding in the file)
NEGATIVE_M = 0.01         # a depth below minus this is negative (the readers' own dry depth)
FLICKER_TRANSITIONS = 4   # wet to dry or back, this many times or more
FLICKER_MIN_STEPS = 5
VOLUME_WARN_PCT = 1.0     # a common rule of thumb for a sound run (ras-commander checks the same 1%)
VOLUME_PROBLEM_PCT = 5.0
GAUGE_DAYS = 1            # daily records: a peak a day apart is on time
TOP_DEEPEST = 5
PINS_PER_CHECK = 3
MAX_CELLS_FLAGGED = 5000
MAX_GAUGES = 5
RADAR_DEG = 0.05

FT_PER_M = 1 / 0.3048

SOURCES = {
    "model": "Your model file, read by aquascope.io.mesh",
    "radar": "Sentinel-1 flood detections: Microsoft AI for Good Lab (2014-2024), MIT",
    "news": "Flood events from news: Groundsource (Mayo et al. 2026, Google), CC BY 4.0",
    "gauges": "Archive gauges (each agency's terms), daily records",
}
RADAR_PERIOD = ("2014-10", "2024-09")

ABOUT = ("Review my model results (#564): checks made by code on your model run, each with its place, value and "
         "source. A check that flags something says where to look, not that the model is wrong.")

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── formatting: every number a reader sees is made here ─────────────────────


def _sig(x: float, digits: int = 3) -> float:
    if not x or not math.isfinite(x):
        return 0.0 if not x else x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def fmt(x: float | None, digits: int = 3) -> str:
    """12.3456 -> '12.3', 0.01234 -> '0.0123', 41234 -> '41,200', -0.5 -> '-0.5' (no trailing zeros)."""
    if x is None or not math.isfinite(float(x)):
        return "unknown"
    v = _sig(float(x), digits)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(v))))) if v else 0
    out = f"{v:,.{decimals}f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def _count(n: int) -> str:
    return f"{int(n):,}"


def _when(iso: str | None) -> str:
    """'1999-01-02T06:00:00' -> '2 Jan 1999 06:00'."""
    if not iso:
        return ""
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)
    return f"{t.day} {_MONTHS[t.month - 1]} {t.year} {t:%H:%M}"


def _day(iso: str | None) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
    except (TypeError, ValueError):
        return str(iso or "")
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _month(m: str) -> str:
    return f"{_MONTHS[int(m[5:7]) - 1]} {m[:4]}"


def _span(a: str, b: str) -> str:
    return _month(a) if a == b else f"{_month(a)} to {_month(b)}"


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{_count(n)} {word if n == 1 else (many or word + 's')}"


# ── the mesh: neighbours and point lookup ───────────────────────────────────


def neighbours(m: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of cells that share an edge (two outline points in common), as two index arrays."""
    # Indices are np.intp (32 bits in the browser, where numpy will not cast int64 down for repeat or bincount);
    # only the edge keys, a product of two point numbers, need 64 bits.
    off = np.asarray(m.ring_offsets).astype(np.intp)
    counts = np.diff(off)
    n_pts = int(off[-1])
    if n_pts == 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp)
    # Points shared by two cells come from the same file coordinates through the same projection: they match
    # to well under a millimetre once rounded.
    key = np.round(np.stack([np.asarray(m.ring_lon), np.asarray(m.ring_lat)], 1) * 1e7).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1).astype(np.int64)
    nxt = np.arange(n_pts, dtype=np.intp) + 1
    ends = off[1:] - 1
    nxt[ends[counts > 0]] = off[:-1][counts > 0]
    a, b = np.minimum(vid, vid[nxt]), np.maximum(vid, vid[nxt])
    ekey = a * (int(vid.max()) + 1) + b
    cell = np.repeat(np.arange(m.n_cells, dtype=np.intp), counts)
    order = np.argsort(ekey, kind="stable")
    ek, ce = ekey[order], cell[order]
    same = np.nonzero(ek[1:] == ek[:-1])[0]
    i, j = ce[same], ce[same + 1]
    keep = i != j
    return i[keep].astype(np.intp), j[keep].astype(np.intp)


def _cell_boxes(m: Mesh) -> np.ndarray:
    off = np.asarray(m.ring_offsets).astype(np.intp)
    counts = np.diff(off)
    boxes = np.full((m.n_cells, 4), np.nan)
    ok = counts > 0
    starts = off[:-1][ok]
    boxes[ok, 0] = np.minimum.reduceat(np.asarray(m.ring_lon), starts)
    boxes[ok, 1] = np.minimum.reduceat(np.asarray(m.ring_lat), starts)
    boxes[ok, 2] = np.maximum.reduceat(np.asarray(m.ring_lon), starts)
    boxes[ok, 3] = np.maximum.reduceat(np.asarray(m.ring_lat), starts)
    # reduceat over a run that ends at the next start: right because the empty rings were left out
    return boxes


def cells_at(m: Mesh, lon: Any, lat: Any) -> np.ndarray:
    """The cell holding each point (its outline contains it), -1 outside the mesh."""
    lon = np.atleast_1d(np.asarray(lon, dtype="f8"))
    lat = np.atleast_1d(np.asarray(lat, dtype="f8"))
    boxes = _cell_boxes(m)
    off = np.asarray(m.ring_offsets).astype(np.intp)
    xs, ys = np.asarray(m.ring_lon), np.asarray(m.ring_lat)
    out = np.full(lon.size, -1, dtype=np.intp)
    for k in range(lon.size):
        x, y = lon[k], lat[k]
        cand = np.nonzero((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3]))[0]
        for c in cand:
            px, py = xs[off[c]:off[c + 1]], ys[off[c]:off[c + 1]]
            qx, qy = np.roll(px, -1), np.roll(py, -1)
            cross = (py > y) != (qy > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = px + (y - py) * (qx - px) / (qy - py)
            if int(np.count_nonzero(cross & (x < xi))) % 2 == 1:
                out[k] = c
                break
    return out


# ── one pass over the run ───────────────────────────────────────────────────


def _scale(m: Mesh) -> float:
    """Metres to the file's length unit."""
    return FT_PER_M if (m.units.get(m.variable) or m.units.get("depth") or "").lower().startswith("ft") else 1.0


def scan(m: Mesh, block: int = 32) -> dict[str, Any]:
    """What the checks need from every step, read in blocks: per cell the lowest and highest raw value, the step
    of the highest, whether it is ever wet and how many times it changes between wet and dry."""
    n = m.n_cells
    steps = len(m.times)
    out: dict[str, Any] = {"steps": steps}
    depth = m.variable == "depth"
    if not steps:
        mv = np.asarray(m.max_value, dtype="f8")
        out.update(lo=None, hi=mv, peak_step=None, wet=np.isfinite(mv) if depth else np.zeros(n, bool),
                   transitions=None)
        return out
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    peak = np.zeros(n, dtype=np.intp)
    for i in range(0, steps, block):
        b = m.values_block(i, min(i + block, steps), raw=True)
        with np.errstate(invalid="ignore"):
            lo = np.fmin(lo, np.nanmin(np.where(np.isfinite(b), b, np.inf), axis=0))
            bmax = np.nanmax(np.where(np.isfinite(b), b, -np.inf), axis=0)
        k = np.argmax(np.where(np.isfinite(b), b, -np.inf), axis=0)
        better = bmax > hi
        peak[better] = i + k[better]
        hi = np.where(better, bmax, hi)
    lo[~np.isfinite(lo)] = np.nan
    hi[~np.isfinite(hi)] = np.nan
    # Wet: a depth over the readers' dry depth; for a file without bed levels, a water surface over its own
    # lowest value in the run (a dry RAS cell holds the water surface at the bed).
    base = None if depth else lo
    trans = np.zeros(n, dtype=np.intp)
    prev = None
    wet_ever = np.zeros(n, bool)
    for i in range(0, steps, block):
        b = m.values_block(i, min(i + block, steps), raw=True)
        with np.errstate(invalid="ignore"):
            w = (b > DRY_DEPTH) if depth else (b - base > DRY_DEPTH)
        wet_ever |= w.any(axis=0)
        seq = w if prev is None else np.vstack([prev[None, :], w])
        trans += np.count_nonzero(seq[1:] != seq[:-1], axis=0)
        prev = w[-1]
    out.update(lo=lo, hi=hi, peak_step=peak, wet=wet_ever, transitions=trans)
    return out


# ── findings ────────────────────────────────────────────────────────────────


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"f{self.n}"


def _fact(ids: _Ids, label: str, value: Any, unit: str | None, shown: str, *, when: str = "",
          lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    out = {"id": ids(), "label": label, "value": value, "unit": unit or "", "shown": shown, "when": when}
    if lat is not None and lon is not None:
        out["place"] = {"lat": round(float(lat), 5), "lon": round(float(lon), 5)}
    return out


def _finding(check: str, severity: str, lat: float, lon: float, title: str, reason: str, facts: list[dict[str, Any]],
             source: str, *, cells: Any = (), when: str = "", period: str = "") -> dict[str, Any]:
    cells = [int(c) for c in list(cells)[:MAX_CELLS_FLAGGED]]
    head = facts[0] if facts else {}
    out = {"check": check, "label": CHECKS[check], "severity": severity, "lat": round(float(lat), 5),
           "lon": round(float(lon), 5), "title": title, "reason": reason, "facts": facts, "source": source,
           "cells": cells, "when": when, "by": "rules", "rules_title": title, "rules_reason": reason,
           "value": head.get("value"), "unit": head.get("unit") or None, "value_label": head.get("label")}
    if period:
        out["period"] = period
    return out


def _centre(m: Mesh, cells: Any = None) -> tuple[float, float]:
    idx = np.asarray(list(cells) if cells is not None else [], dtype=np.intp)
    if idx.size:
        return float(np.mean(m.lat[idx])), float(np.mean(m.lon[idx]))
    w, s, e, n = m.bbox()
    return (s + n) / 2, (w + e) / 2


def _check(checks: list[dict[str, Any]], check: str, status: str, line: str, n: int = 0) -> None:
    checks.append({"check": check, "label": CHECKS[check], "status": status, "line": line, "flagged": int(n)})


def _deepest(m: Mesh, sc: dict[str, Any], ids: _Ids, top: int, unit: str | None) -> tuple[list, str]:
    depth = m.variable == "depth"
    if depth:
        val = np.asarray(m.max_value, dtype="f8") if sc["hi"] is None else np.where(sc["hi"] > DRY_DEPTH, sc["hi"],
                                                                                      np.nan)
        label = "Largest depth"
    else:
        if sc["lo"] is None:
            return [], "This file has no time steps and no bed levels, so neither depth nor rise can be read."
        val = sc["hi"] - sc["lo"]
        val = np.where(val > DRY_DEPTH, val, np.nan)
        label = "Rise of the water surface"
    if not np.isfinite(val).any():
        return [], "No cell holds water at any step."
    # The deepest places, not five cells side by side: each pin at least a twentieth of the mesh's width from the
    # ones before it.
    ranked = [int(i) for i in np.argsort(np.where(np.isfinite(val), -val, np.inf), kind="stable")
              if np.isfinite(val[i])]
    w, s_, e, n = m.bbox()
    kx = math.cos(math.radians((s_ + n) / 2))
    gap = math.hypot((e - w) * kx, n - s_) / 20
    order: list[int] = []
    for i in ranked:
        if all(math.hypot((m.lon[i] - m.lon[j]) * kx, m.lat[i] - m.lat[j]) >= gap for j in order):
            order.append(i)
        if len(order) >= top:
            break
    out = []
    for rank, i in enumerate(order, 1):
        t = m.times[int(sc["peak_step"][i])] if sc["peak_step"] is not None else None
        f = _fact(ids, label, round(float(val[i]), 4), unit, fmt(val[i]), when=_when(t), lat=m.lat[i], lon=m.lon[i])
        u = f" {unit}" if unit else ""
        if depth:
            title = f"Deepest water: {f['shown']}{u}"
            reason = (f"The deepest water in this part of the mesh: {f['shown']}{u}"
                      + (f" at {f['when']}" if f["when"] else "") +
                      ". Worth a look if it is not a channel or a pit you expect.")
        else:
            title = f"Largest rise: {f['shown']}{u}"
            reason = (f"The file has no bed levels, so no depth: the water surface here rose {f['shown']}{u} over "
                      "the run" + (f", highest at {f['when']}" if f["when"] else "") + ", the largest rise in this "
                      "part of the mesh.")
        out.append(_finding("deepest", "info", m.lat[i], m.lon[i], title, reason, [f], SOURCES["model"], cells=[i],
                            when=t or ""))
    line = (f"The deepest water in {_plural(len(out), 'part')} of the mesh is pinned." if depth else
            f"No bed levels in this file: the largest rises of the water surface, in {_plural(len(out), 'part')} of "
            "the mesh, are pinned instead.")
    return out, line


def _negative(m: Mesh, sc: dict[str, Any], ids: _Ids, unit: str | None, k: float) -> tuple[list, str, str]:
    if m.variable != "depth" or sc["lo"] is None:
        return [], "skipped", ("No depth in this file (no bed levels), so negative depths cannot be read."
                               if m.variable != "depth" else "No time steps in this file, only the maxima.")
    bad = np.nonzero(sc["lo"] < -NEGATIVE_M * k)[0]
    if not bad.size:
        return [], "ok", "No depth goes below zero at any step."
    worst = int(bad[np.argmin(sc["lo"][bad])])
    u = f" {unit}" if unit else ""
    f1 = _fact(ids, "Lowest depth", round(float(sc["lo"][worst]), 4), unit, fmt(sc["lo"][worst]),
               lat=m.lat[worst], lon=m.lon[worst])
    f2 = _fact(ids, "Cells with a negative depth", int(bad.size), "cells", _count(bad.size))
    title = f"Negative depth: {f1['shown']}{u}"
    reason = (f"The water surface falls below the bed here, to {f1['shown']}{u}; {f2['shown']} "
              f"{'cell goes' if bad.size == 1 else 'cells go'} below zero in all. A model fault to fix before "
              "reading the depths: check the terrain, the initial conditions and the time step.")
    return [_finding("negative", "problem", m.lat[worst], m.lon[worst], title, reason, [f1, f2], SOURCES["model"],
                     cells=bad)], "flagged", f"{_plural(int(bad.size), 'cell')} with a negative depth."


def _tiny(m: Mesh, sc: dict[str, Any], ids: _Ids, unit: str | None, k: float) -> tuple[list, str, str]:
    if m.variable != "depth" or sc["hi"] is None:
        return [], "skipped", "No depth in this file (no bed levels), so shallow water cannot be read."
    hi = sc["hi"]
    with np.errstate(invalid="ignore"):
        film = np.nonzero((hi > TINY_FLOOR_M * k) & (hi < TINY_DEPTH_M * k))[0]
    limit = fmt(TINY_DEPTH_M * k)
    u = f" {unit}" if unit else ""
    if not film.size:
        return [], "ok", f"No cell is wetted only by a film (under {limit}{u})."
    wet = int(np.count_nonzero(hi > TINY_FLOOR_M * k))
    # One pin for them all: a film is spread over many cells, each flagged on the map.
    pick = film[np.argsort(-hi[film])][:1]
    out = []
    f2 = None
    for i in pick:
        i = int(i)
        f1 = _fact(ids, "Largest depth", round(float(hi[i]), 4), unit, fmt(hi[i]), lat=m.lat[i], lon=m.lon[i])
        if f2 is None:
            f2 = _fact(ids, "Cells wetted only by a film", int(film.size), "cells", _count(film.size))
            share = 100 * film.size / max(wet, 1)
            f3 = _fact(ids, "Share of the cells that get wet", round(film.size / max(wet, 1), 4), "",
                       f"{fmt(share, 2) if share < 1 else round(share)}%")
            facts = [f1, f2, f3]
            reason = (f"This cell never gets deeper than {f1['shown']}{u}, under {limit}{u}. "
                      f"{f2['shown']} {'cell in all is' if film.size == 1 else 'cells in all are'} like it, "
                      f"{f3['shown']} of those that get wet: rain on the grid, "
                      "a terrain or a tolerance problem.")
            cells = film
        else:
            facts = [f1]
            reason = f"Another cell wetted only by a film: never deeper than {f1['shown']}{u}."
            cells = [i]
        out.append(_finding("tiny", "warning", m.lat[i], m.lon[i], f"Only a film of water: {f1['shown']}{u}", reason,
                            facts, SOURCES["model"], cells=cells))
    return out, "flagged", f"{_plural(int(film.size), 'cell')} wetted only by a film (under {limit}{u})."


def _isolated(m: Mesh, sc: dict[str, Any], ids: _Ids, unit: str | None, nb: tuple[np.ndarray, np.ndarray]
              ) -> tuple[list, str, str]:
    wet = np.asarray(sc["wet"], bool)
    if not wet.any():
        return [], "skipped", "No cell gets wet, so there is nothing to be isolated."
    i, j = nb
    n = m.n_cells
    deg = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    wet_nb = np.bincount(i, weights=wet[j], minlength=n) + np.bincount(j, weights=wet[i], minlength=n)
    alone = np.nonzero(wet & (deg > 0) & (wet_nb == 0))[0]
    proxy = "" if m.variable == "depth" else " (wet here: a water surface that rose during the run)"
    if not alone.size:
        return [], "ok", f"Every wet cell has a wet neighbour{proxy}."
    size = sc["hi"] - (sc["lo"] if m.variable != "depth" else 0) if sc["hi"] is not None else np.asarray(m.max_value)
    pick = alone[np.argsort(-np.nan_to_num(size[alone], nan=-np.inf))][:PINS_PER_CHECK]
    out = []
    for r, c in enumerate(pick):
        c = int(c)
        f1 = _fact(ids, "Largest depth" if m.variable == "depth" else "Rise of the water surface",
                   round(float(size[c]), 4), unit, fmt(size[c]), lat=m.lat[c], lon=m.lon[c])
        u = f" {unit}" if unit else ""
        facts = [f1]
        if r == 0:
            f2 = _fact(ids, "Isolated wet cells", int(alone.size), "cells", _count(alone.size))
            facts.append(f2)
            reason = (f"This cell holds water ({f1['shown']}{u}) while every neighbour stays dry{proxy}; "
                      f"{f2['shown']} {'such cell' if alone.size == 1 else 'such cells'} in all. A pit in the terrain, "
                      "a source term or a wet start "
                      "left behind.")
        else:
            reason = f"Another wet cell with only dry neighbours ({f1['shown']}{u})."
        out.append(_finding("isolated", "warning", m.lat[c], m.lon[c], "Wet cell with dry neighbours", reason, facts,
                            SOURCES["model"], cells=alone if r == 0 else [c]))
    return out, "flagged", f"{_plural(int(alone.size), 'wet cell')} with only dry neighbours{proxy}."


def _flicker(m: Mesh, sc: dict[str, Any], ids: _Ids) -> tuple[list, str, str]:
    if sc["transitions"] is None or sc["steps"] < FLICKER_MIN_STEPS:
        return [], "skipped", f"Fewer than {FLICKER_MIN_STEPS} time steps, so flicker cannot be told."
    if m.variable != "depth":
        return [], "skipped", "No depth in this file (no bed levels), so wet and dry cannot be told step by step."
    tr = sc["transitions"]
    bad = np.nonzero(tr >= FLICKER_TRANSITIONS)[0]
    if not bad.size:
        return [], "ok", f"No cell changes between wet and dry {FLICKER_TRANSITIONS} times or more."
    pick = bad[np.argsort(-tr[bad], kind="stable")][:PINS_PER_CHECK]
    out = []
    for r, c in enumerate(pick):
        c = int(c)
        f1 = _fact(ids, "Changes between wet and dry", int(tr[c]), "times", _count(tr[c]), lat=m.lat[c], lon=m.lon[c])
        f2 = _fact(ids, "Time steps", sc["steps"], "steps", _count(sc["steps"]))
        facts = [f1, f2]
        if r == 0:
            f3 = _fact(ids, "Cells that flicker", int(bad.size), "cells", _count(bad.size))
            facts.append(f3)
            reason = (f"This cell goes from wet to dry or back {f1['shown']} times in {f2['shown']} steps; "
                      f"{f3['shown']} {'cell flickers' if bad.size == 1 else 'cells flicker'} in all. Often a sign "
                      "of a time step too long for the mesh "
                      "or of a shallow, flat area at the edge of the flow.")
        else:
            reason = f"Another cell that flickers: {f1['shown']} changes in {f2['shown']} steps."
        out.append(_finding("flicker", "warning", m.lat[c], m.lon[c], f"Wet and dry {f1['shown']} times", reason,
                            facts, SOURCES["model"], cells=bad if r == 0 else [c]))
    return out, "flagged", (f"{_plural(int(bad.size), 'cell')} {'flickers' if bad.size == 1 else 'flicker'} "
                            "between wet and dry.")


# ── mass balance: the model's own volume accounting ─────────────────────────

_VA = "Results/Unsteady/Summary/Volume Accounting"


def check_volume(m: Mesh) -> dict[str, Any] | None:
    """The volume accounting a HEC-RAS plan file holds (``Results/Unsteady/Summary/Volume Accounting``): the error
    in percent, its volume and unit, the start and end volumes. None for a file without one (a UGRID map file
    holds none).

    HEC-RAS 6.5 writes the plan's totals on the group itself (``Error Percent``, ``Error``, ``Volume Starting``,
    ``Volume Ending``, ``Total Boundary Flux of Water In`` / ``Out``, ``Vol Accounting in``) and each 2D area's
    under ``Volume Accounting 2D/<area>`` with other names (``Vol Starting``, ``Vol Ending``, ``Cum Inflow``,
    ``Cum Outflow``); both are read (checked on a real 6.5 plan, docs/assessments/hec-ras.md)."""
    fh = getattr(getattr(m, "_source", None), "fh", None)
    if m.format != "hec-ras" or fh is None or _VA not in fh:
        return None
    found = []

    def read(name: str, obj: Any) -> None:
        attrs = {str(k): v for k, v in obj.attrs.items()}
        pct = next((v for k, v in attrs.items() if re.fullmatch(r"(?i)error\s*percent|percent\s*error", k)), None)
        if pct is None:
            return
        try:
            pct = float(np.asarray(pct).reshape(-1)[0])
        except (TypeError, ValueError):
            return
        if not math.isfinite(pct):
            return

        def num(*keys: str) -> float | None:
            v = next((attrs[k] for k in keys if k in attrs), None)
            try:
                v = float(np.asarray(v).reshape(-1)[0])
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None

        unit = attrs.get("Vol Accounting in")
        unit = unit.decode("utf-8", "replace").strip("\x00 ") if isinstance(unit, (bytes, np.bytes_)) else unit
        found.append({"where": name or "plan", "error_percent": pct, "error": num("Error"),
                      "unit": str(unit).strip() if unit else None,
                      # the plan's totals and each 2D area's own (Volume Accounting 2D/<area>) name them apart
                      "start": num("Volume Starting", "Vol Starting"), "end": num("Volume Ending", "Vol Ending"),
                      "in": num("Total Boundary Flux of Water In", "Cum Inflow"),
                      "out": num("Total Boundary Flux of Water Out", "Cum Outflow")})

    grp = fh[_VA]
    read("", grp)
    grp.visititems(lambda n, o: read(n, o) if hasattr(o, "attrs") and not hasattr(o, "shape") else None)
    if not found:
        return None
    found.sort(key=lambda x: -abs(x["error_percent"]))
    return {"worst": found[0], "parts": found}


def _mass(m: Mesh, ids: _Ids, wet: np.ndarray) -> tuple[list, str, str]:
    va = check_volume(m)
    if va is None:
        why = ("This file holds no volume accounting (HEC-RAS writes it under Results, Summary)."
               if m.format == "hec-ras" else "A mesh map file holds no volume accounting, so the balance cannot "
                                             "be checked here.")
        return [], "skipped", why
    w = va["worst"]
    pct = w["error_percent"]
    lat, lon = _centre(m, np.nonzero(wet)[0][:20000])
    f1 = _fact(ids, "Volume error", round(pct, 4), "%", f"{fmt(pct, 2)}%", lat=lat, lon=lon)
    facts = [f1]
    if w["error"] is not None:
        facts.append(_fact(ids, "Volume error", round(w["error"], 4), w["unit"] or "", fmt(w["error"])))
    where = "" if w["where"] in ("plan", "") else f" ({w['where'].split('/')[-1]})"
    if abs(pct) > VOLUME_PROBLEM_PCT:
        sev = "problem"
    elif abs(pct) > VOLUME_WARN_PCT:
        sev = "warning"
    else:
        return [], "ok", f"The model's own volume error is {f1['shown']}{where}, within {fmt(VOLUME_WARN_PCT, 1)}%."
    vu = (w["unit"] or "").strip()
    vu = f" × {vu}" if vu[:1].isdigit() else (f" {vu}" if vu else "")   # "1000 m3" is a unit of a thousand
    extra = f", {facts[1]['shown']}{vu}" if len(facts) > 1 else ""
    reason = (f"The model's own volume accounting{where} says {f1['shown']} of the water is not accounted "
              f"for{extra}. Over {fmt(VOLUME_WARN_PCT, 1)}% is worth fixing before the depths are used: time step, "
              "mesh size near steep terrain, boundary conditions.")
    return [_finding("mass_balance", sev, lat, lon, f"Volume error {f1['shown']}", reason, facts,
                     SOURCES["model"])], "flagged", f"Volume error {f1['shown']}{where}."


# ── observed floods ─────────────────────────────────────────────────────────


def _run_months(m: Mesh) -> tuple[str, str] | None:
    if not m.times:
        return None
    return m.times[0][:7], m.times[-1][:7]


def _overlap(run: tuple[str, str] | None, period: tuple[str, str] | None) -> bool:
    return bool(run and period and run[0] <= period[1] and run[1] >= period[0])


def _radar(m: Mesh, wet: np.ndarray, floods: dict[str, Any] | None, ids: _Ids) -> tuple[list, str, str]:
    radar = (floods or {}).get("radar")
    if floods is None:
        return [], "skipped", "The floods seen by radar were not looked up."
    if not radar or radar.get("rows") is None:
        return [], "skipped", "The radar floods could not be read" + (
            f" ({radar.get('error')})." if radar and radar.get("error") else ".")
    period = tuple(radar.get("period") or RADAR_PERIOD)
    run = _run_months(m)
    same = _overlap(run, period)
    rows = [r for r in radar["rows"] if not same or run[0] <= r[2] <= run[1]]
    # The 0.05 degree radar cell of each mesh cell's centre.
    rr = np.floor((np.asarray(m.lat) + 90) / RADAR_DEG + 1e-9).astype(np.int64)
    rc = np.floor((np.asarray(m.lon) + 180) / RADAR_DEG + 1e-9).astype(np.int64)
    key = rr * 100000 + rc
    seen: dict[int, list[Any]] = {}
    for lat, lon, month, n in rows:
        k = int(math.floor((lat + 90) / RADAR_DEG + 1e-9)) * 100000 + int(math.floor((lon + 180) / RADAR_DEG + 1e-9))
        acc = seen.setdefault(k, [lat, lon, 0, set()])
        acc[2] += int(n)
        acc[3].add(month)
    covered = np.unique(key)
    when = (f"during the run's months ({_span(*run)})" if same else
            f"in any month on record ({_span(*period)}; the run" + (f", {_span(*run)}," if run else "") +
            " is outside the radar's record)")
    if not covered.size:
        return [], "skipped", "The mesh covers no radar cell."
    hit_wet = hit_dry = 0
    dry_cells = []
    for k in covered:
        if int(k) not in seen:
            continue
        inside = key == k
        if wet[inside].any():
            hit_wet += 1
        else:
            hit_dry += 1
            dry_cells.append((int(k), np.nonzero(inside)[0]))
    n_seen = hit_wet + hit_dry
    line = (f"Radar saw flooding in {_count(n_seen)} of the {_count(covered.size)} radar squares (0.05°) the mesh "
            f"covers, {when}; the model wets cells in {_count(hit_wet)} of them.")
    if not n_seen:
        return [], "ok", f"Radar saw no flooding over the mesh {when}."
    out = []
    sev = "warning" if same else "info"
    dry_cells.sort(key=lambda x: -seen[x[0]][2])
    for k, cells in dry_cells[:PINS_PER_CHECK]:
        lat, lon, n, months = seen[k]
        f1 = _fact(ids, "Radar flood detections", n, "20 m pixels", _count(n), lat=lat, lon=lon)
        f2 = _fact(ids, "Months with detections", len(months), "months", _count(len(months)),
                   when=_span(min(months), max(months)))
        f3 = _fact(ids, "Mesh cells in this radar cell", int(cells.size), "cells", _count(cells.size))
        if same:
            reason = (f"Sentinel-1 radar saw flooding here during the run ({f1['shown']} detections, "
                      f"{f2['when']}), but the model keeps all {f3['shown']} of its cells in this 0.05° square dry.")
        else:
            reason = (f"Not this event: radar saw flooding here in {f2['shown']} months of {_span(*period)} "
                      f"({f1['shown']} detections), and the model keeps its {f3['shown']} cells in this 0.05° "
                      "square dry. Worth a look if the run is meant to flood like those.")
        out.append(_finding("floods_radar", sev, lat, lon, "Radar saw flooding, the model is dry", reason,
                            [f1, f2, f3], SOURCES["radar"], cells=cells, period="same" if same else "record"))
    status = "flagged" if (out and same) else "ok"
    return out, status, line


def _news(m: Mesh, wet: np.ndarray, floods: dict[str, Any] | None, ids: _Ids) -> tuple[list, str, str]:
    news = (floods or {}).get("news")
    if floods is None:
        return [], "skipped", "The floods in the news were not looked up."
    if not news or news.get("events") is None:
        return [], "skipped", "The news floods could not be read" + (
            f" ({news.get('error')})." if news and news.get("error") else ".")
    period = tuple(news.get("period") or ("2000-01", "2026-02"))
    run = _run_months(m)
    same = _overlap(run, period)
    lat, lon = np.asarray(m.lat), np.asarray(m.lon)
    hits = []
    for e in news["events"]:
        if same:
            # The days the event ran, a day either side (a report is dated, a flood takes time).
            try:
                a = date.fromisoformat(str(e["start"])[:10]) - timedelta(days=1)
                b = date.fromisoformat(str(e.get("end") or e["start"])[:10]) + timedelta(days=1)
            except ValueError:
                continue
            if not (a.isoformat() <= m.times[-1][:10] and b.isoformat() >= m.times[0][:10]):
                continue
        box = (e["west"], e["south"], e["east"], e["north"])
        under = np.nonzero((lon >= box[0]) & (lon <= box[2]) & (lat >= box[1]) & (lat <= box[3]))[0]
        if not under.size:
            continue
        hits.append((e, under))
    when = (f"during the run ({_span(*run)})" if same else
            f"on record ({_span(*period)}; the run" + (f", {_span(*run)}," if run else "") +
            " is outside the news record)")
    if not hits:
        return [], "ok", f"No flood in the news over the mesh {when}."
    dry = [(e, u) for e, u in hits if not wet[u].any()]
    line = (f"{_plural(len(hits), 'flood')} in the news over the mesh {when}; the model wets cells under "
            f"{_count(len(hits) - len(dry))} of them.")
    out = []
    sev = "warning" if same else "info"
    dry.sort(key=lambda x: float(x[0].get("area_km2") or 1e9))   # the most precise footprints first
    spots: set[tuple[float, float]] = set()
    picked = []
    for e, under in dry:
        spot = (round(float(e["lat"]), 3), round(float(e["lon"]), 3))
        if spot not in spots:
            spots.add(spot)
            picked.append((e, under))
    for e, under in picked[:PINS_PER_CHECK]:
        f1 = _fact(ids, "Flood in the news", _day(e["start"]), "", _day(e["start"]), when=_day(e["start"]),
                   lat=e["lat"], lon=e["lon"])
        f2 = _fact(ids, "Mesh cells under its footprint", int(under.size), "cells", _count(under.size))
        area = e.get("area_km2")
        facts = [f1, f2]
        if area is not None:
            facts.append(_fact(ids, "Footprint", round(float(area), 3), "km²", fmt(float(area))))
        if same:
            reason = (f"A flood here was in the news on {f1['shown']}, during the run, but the model keeps all "
                      f"{f2['shown']} of its cells under the report's footprint dry.")
        else:
            reason = (f"Not this event: a flood here was in the news on {f1['shown']}, and the model keeps its "
                      f"{f2['shown']} cells under that report's footprint dry. A news report says a flood "
                      "happened, not how far it reached.")
        elat, elon = float(e["lat"]), float(e["lon"])
        if not (m.bbox()[0] <= elon <= m.bbox()[2] and m.bbox()[1] <= elat <= m.bbox()[3]):
            elat, elon = float(np.mean(lat[under])), float(np.mean(lon[under]))
        out.append(_finding("floods_news", sev, elat, elon, "In the news, dry in the model", reason, facts,
                            SOURCES["news"], cells=under, when=str(e["start"])[:10],
                            period="same" if same else "record"))
    status = "flagged" if (out and same) else "ok"
    return out, status, line


# ── gauges inside the mesh ──────────────────────────────────────────────────


def gauges_inside(m: Mesh, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The gauges (``{source, station_id, name, lat, lon}``) that stand inside a cell of the mesh, each with its
    ``cell`` (all of them: the gauge check reads the first :data:`MAX_GAUGES` and counts the rest)."""
    rows = [r for r in rows or [] if r.get("lat") is not None and r.get("lon") is not None]
    if not rows:
        return []
    w, s, e, n = m.bbox()
    rows = [r for r in rows if w <= float(r["lon"]) <= e and s <= float(r["lat"]) <= n]
    if not rows:
        return []
    cells = cells_at(m, [float(r["lon"]) for r in rows], [float(r["lat"]) for r in rows])
    return [{**r, "cell": int(c)} for r, c in zip(rows, cells) if c >= 0]


def _gauges(m: Mesh, gauges: list[dict[str, Any]] | None, ids: _Ids, unit: str | None) -> tuple[list, str, str]:
    if gauges is None:
        return [], "skipped", "The Archive's gauges were not looked up."
    inside = [g for g in gauges if g.get("cell") is not None and int(g["cell"]) >= 0]
    if not inside:
        return [], "ok", "No Archive gauge stands inside the mesh, so no peak can be set against one."
    if not m.times:
        return [], "skipped", "No time steps in this file, so no modelled peak to set against the gauges."
    total = len(inside)
    inside = inside[:MAX_GAUGES]
    out = []
    flagged = 0
    for g in inside:
        c = int(g["cell"])
        name = str(g.get("name") or g.get("station_id"))
        s = m.series(c)
        vals = np.asarray(s["value"], dtype="f8")
        src = f"{g.get('source')} gauge {g.get('station_id')} ({SOURCES['gauges']}); your model file"
        series = [(str(d)[:10], v) for d, v in (g.get("series") or []) if v is not None and math.isfinite(float(v))]
        if not np.isfinite(vals).any():
            f1 = _fact(ids, "Gauge", name, "", name, lat=g["lat"], lon=g["lon"])
            out.append(_finding("gauges", "info", g["lat"], g["lon"], f"Gauge {name}: dry in the model",
                                f"The gauge {f1['shown']} stands in a cell the model keeps dry all run.", [f1], src,
                                cells=[c]))
            continue
        k = int(np.nanargmax(vals))
        t_model = s["times"][k]
        fm = _fact(ids, "Modelled peak" if m.variable == "depth" else "Modelled highest water surface",
                   round(float(vals[k]), 4), unit, fmt(vals[k]), when=_when(t_model), lat=g["lat"], lon=g["lon"])
        if not series:
            note = g.get("note") or "it has no observations during the run"
            f1 = _fact(ids, "Gauge", name, "", name)
            out.append(_finding("gauges", "info", g["lat"], g["lon"], f"Gauge {name}: no record for the run",
                                f"The gauge {f1['shown']} stands in the mesh, but {note}. The model peaks there at "
                                f"{fm['when']}.", [fm, f1], src, cells=[c], when=t_model))
            continue
        gunit = g.get("unit") or ""
        what = str(g.get("variable") or "flow").replace("_", " ")
        model_what = "depth" if m.variable == "depth" else "water surface"
        fn = _fact(ids, "Gauge", name, "", name)
        series.sort()
        j = max(range(len(series)), key=lambda x: float(series[x][1]))
        d_obs, v_obs = series[j]
        fo = _fact(ids, f"Observed highest ({what})", round(float(v_obs), 4), gunit, fmt(float(v_obs)),
                   when=_day(d_obs))
        if len(series) > 2 and j in (0, len(series) - 1):
            # Highest on the first or last day: the gauge was falling, or still rising, so there is no peak to time.
            way = "falling from the start of the run" if j == 0 else "still rising at its end"
            reason = (f"At {fn['shown']} the gauge was highest on {fo['when']} ({fo['shown']} {gunit}), {way}, so "
                      f"it shows no peak to set against the model's at {fm['when']}.")
            out.append(_finding("gauges", "info", g["lat"], g["lon"], f"No observed peak in the run at {name}", reason,
                                [fo, fm, fn], src, cells=[c], when=t_model))
            continue
        lag = (date.fromisoformat(d_obs) - date.fromisoformat(t_model[:10])).days
        fl = _fact(ids, "Days between the peaks", abs(lag), "days", _count(abs(lag)))
        if abs(lag) > GAUGE_DAYS:
            flagged += 1
            way = "after" if lag > 0 else "before"
            reason = (f"At {fn['shown']} the gauge peaked on {fo['when']} ({fo['shown']} {gunit}), "
                      f"{fl['shown']} days {way} the model's peak at {fm['when']}. Timing only: the gauge measures "
                      f"{what}, the model {model_what}.")
            out.append(_finding("gauges", "warning", g["lat"], g["lon"], f"Peak {fl['shown']} days off at {name}",
                                reason, [fl, fo, fm, fn], src, cells=[c], when=t_model))
        else:
            reason = (f"At {fn['shown']} the gauge peaked on {fo['when']} ({fo['shown']} {gunit}) and the model at "
                      f"{fm['when']}: within a day, as close as a daily record can tell. Timing only: the gauge "
                      f"measures {what}, the model {model_what}.")
            out.append(_finding("gauges", "info", g["lat"], g["lon"], f"Peak on time at {name}", reason,
                                [fo, fm, fn, fl], src, cells=[c], when=t_model))
    off = (f"{_count(flagged)} observed peak{'' if flagged == 1 else 's'} more than {_plural(GAUGE_DAYS, 'day')} "
           "off the model's" if flagged else f"no observed peak more than {_plural(GAUGE_DAYS, 'day')} off the model's")
    some = f", the first {_count(len(inside))} checked" if total > len(inside) else ""
    line = f"{_plural(total, 'gauge')} inside the mesh{some}; {off}."
    return out, "flagged" if flagged else "ok", line


# ── the review ──────────────────────────────────────────────────────────────

_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def review(m: Mesh, *, context: dict[str, Any] | None = None, top: int = TOP_DEEPEST) -> dict[str, Any]:
    """Run every check on a mesh. ``context`` holds what lies outside the file: ``floods`` (from
    :func:`gather_floods`) and ``gauges`` (rows with their observed ``series``, from :func:`gather_context`);
    without it the observed-flood and gauge checks say they were skipped.

    Returns ``{"ok", "mesh", "findings", "checks", "sources", "about"}``: the findings most severe first, each
    ``{id, rank, check, label, severity, lat, lon, value, unit, title, reason, facts, source, cells, when}``,
    and one line per check (``status`` flagged, ok or skipped)."""
    context = context or {}
    unit = m.units.get(m.variable)
    k = _scale(m)
    ids = _Ids()
    sc = scan(m)
    wet = np.asarray(sc["wet"], bool)
    nb = neighbours(m)
    findings: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for check, run in (
        ("negative", lambda: _negative(m, sc, ids, unit, k)),
        ("mass_balance", lambda: _mass(m, ids, wet)),
        ("isolated", lambda: _isolated(m, sc, ids, unit, nb)),
        ("flicker", lambda: _flicker(m, sc, ids)),
        ("tiny", lambda: _tiny(m, sc, ids, unit, k)),
        ("floods_radar", lambda: _radar(m, wet, context.get("floods"), ids)),
        ("floods_news", lambda: _news(m, wet, context.get("floods"), ids)),
        ("gauges", lambda: _gauges(m, context.get("gauges"), ids, unit)),
    ):
        got, status, line = run()
        findings.extend(got)
        _check(checks, check, status, line, sum(1 for f in got if f["severity"] != "info"))
    deep, line = _deepest(m, sc, ids, top, unit)
    findings.extend(deep)
    _check(checks, "deepest", "ok" if deep else "skipped", line)

    findings.sort(key=lambda f: (_ORDER[f["severity"]], list(CHECKS).index(f["check"])))
    for rank, f in enumerate(findings, 1):
        f["id"] = f"r{rank}"
        f["rank"] = rank
    order = {c: i for i, c in enumerate(CHECKS)}
    checks.sort(key=lambda c: order[c["check"]])
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
    return {
        "ok": True, "about": ABOUT,
        "mesh": {"name": m.name, "format": m.format, "cells": m.n_cells, "variable": m.variable, "unit": unit,
                 "steps": len(m.times), "start": m.times[0] if m.times else None,
                 "end": m.times[-1] if m.times else None, "wet_cells": int(wet.sum()), "bbox": m.bbox()},
        "findings": findings, "checks": checks, "counts": counts,
        "summary": _summary(counts, checks),
        "sources": {k2: v for k2, v in SOURCES.items()},
        "thresholds": {"tiny_depth_m": TINY_DEPTH_M, "negative_m": NEGATIVE_M, "dry_depth": DRY_DEPTH,
                       "flicker_transitions": FLICKER_TRANSITIONS, "volume_warn_pct": VOLUME_WARN_PCT,
                       "volume_problem_pct": VOLUME_PROBLEM_PCT, "gauge_days": GAUGE_DAYS},
    }


def _summary(counts: dict[str, int], checks: list[dict[str, Any]]) -> str:
    ran = [c for c in checks if c["status"] != "skipped"]
    bits = []
    if counts["problem"]:
        bits.append(_plural(counts["problem"], "problem"))
    if counts["warning"]:
        bits.append(_plural(counts["warning"], "thing", "things") + " worth a look")
    head = ", ".join(bits) if bits else "Nothing flagged"
    return f"{head}; {_count(len(ran))} of {_count(len(checks))} checks ran."


# ── the model path: it orders and words, code holds the numbers ─────────────

_RANKING = re.compile(r"\b(deepest|highest|lowest|largest|smallest|biggest|worst|best|most|least|record|"
                      r"unprecedented|catastrophic|severe|dangerous|safe|wrong|correct)\b", re.I)


def review_prompt(findings: list[dict[str, Any]], context: str = "") -> dict[str, Any]:
    """The system prompt, the user message and the reply schema for a model that orders and words a review."""
    items = []
    for f in findings:
        items.append({"id": f["id"], "check": f["label"], "severity": f["severity"], "rules_title": f["rules_title"],
                      "rules_reason": f["rules_reason"],
                      "facts": [{"placeholder": "{" + x["id"] + "}", "label": x["label"], "value": x["shown"],
                                 **({"unit": x["unit"]} if x.get("unit") else {}),
                                 **({"when": x["when"], "when_placeholder": "{" + x["id"] + ".when}"}
                                    if x.get("when") else {})} for x in f["facts"]]})
    system = "\n".join([
        "You help a hydraulic modeller review a model run shown on a map. Below are findings made by code from the "
        "model file and from observed floods and gauges, each with an id, its severity, its rules wording and its "
        "facts.",
        "Order the findings, most worth the modeller's attention first (a problem before a warning before "
        "information), and write for each a short title (at most 60 characters) and a reason (one or two plain "
        "sentences, at most 260 characters).",
        "Never write a number yourself, in digits or in words. Write each value as its fact's placeholder, "
        "exactly as given ({f3}, or {f3.when} for its time); code fills them in, and wording with any other "
        "number is thrown away. Use only the placeholders of that finding's own facts.",
        "Do not judge the model right or wrong, and do not add causes or advice the rules wording does not hold. "
        "Do not use a ranking word (deepest, largest, worst...) unless the rules wording uses it.",
        'Reply with ONE JSON object: {"order": [ids], "notes": [{"id", "title", "reason"}]}.',
    ])
    if context:
        system += f"\n\nWhat the modeller is looking at: {context}"
    schema = {"type": "object", "required": ["order", "notes"], "properties": {
        "order": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "object", "required": ["id", "title", "reason"],
                                             "properties": {"id": {"type": "string"}, "title": {"type": "string"},
                                                            "reason": {"type": "string"}}}}}}
    return {"system": system, "prompt": json.dumps({"findings": items}, ensure_ascii=False), "schema": schema}


def _reply_json(reply: Any) -> Any:
    if not isinstance(reply, str):
        return reply
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    start = raw.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return data
    except ValueError:
        return None


def _locked(f: dict[str, Any], title: str, reason: str) -> tuple[tuple[str, str] | None, str | None]:
    """A model's title and reason for one finding, held to its facts: filled text, or None and why not."""
    from aquascope.map_explain import _PLACEHOLDER, _fill, claim_lock, placeholder_touches

    title = re.sub(r"\s+", " ", str(title or "")).strip()
    reason = re.sub(r"\s+", " ", str(reason or "")).strip()
    if not title or not reason:
        return None, "empty"
    if len(title) > 80 or len(reason) > 400:
        return None, "too long"
    own = {x["id"]: x for x in f["facts"]}
    words = f"{title} {reason}"
    for mt in _PLACEHOLDER.finditer(words):
        if mt.group(1) not in own:
            return None, f"{{{mt.group(1)}}} is not one of its facts"
        if mt.group(2) == "when" and not own[mt.group(1)].get("when"):
            return None, f"{{{mt.group(1)}.when}} has no time"
        if mt.group(2) == "place":
            return None, "a place placeholder (findings carry coordinates, not names)"
    # A sign or a digit against a placeholder changes the number code fills in ("-{f1}" turns 3.2% into -3.2%):
    # the shared check Explain and Scout make too.
    touch = placeholder_touches(words)
    if touch:
        return None, f"a sign or a digit against a placeholder ({', '.join(touch)})"
    # A fact's unit is code's too ("m3/s", "1000 m3", "km²"): its digits are not a claim.
    stray = claim_lock(words, [{**x, "label": f"{x['label']} {x.get('unit') or ''}"} for x in own.values()])
    if stray:
        return None, f"a number not in its facts ({', '.join(stray)})"
    said = f"{f['rules_title']} {f['rules_reason']}".lower()
    rank = [w for w in _RANKING.findall(words) if w.lower() not in said]
    if rank:
        return None, f"a ranking or judging word ({rank[0]})"
    if "{" in _PLACEHOLDER.sub("", words) or "}" in _PLACEHOLDER.sub("", words):
        return None, "a stray brace"
    return (_fill(title, own), _fill(reason, own)), None


def apply_words(findings: list[dict[str, Any]], reply: Any, *, by: str = "model") -> dict[str, Any]:
    """Hold a model's reply to the claim lock: ``{"findings", "refused", "ordered"}``. The model's order is kept
    for the ids it knows (the rules order follows for the rest); each title and reason that passes replaces
    the template, the others keep it and are listed in ``refused``. Values, places, severities and the flagged
    cells are never the model's."""
    data = _reply_json(reply)
    byid = {f["id"]: f for f in findings}
    if not isinstance(data, dict):
        return {"findings": [dict(f) for f in findings], "refused": ["the model did not reply with JSON"],
                "ordered": False}
    order = [i for i in dict.fromkeys(str(x) for x in data.get("order") or []) if i in byid]
    ordered = bool(order)
    order += [f["id"] for f in findings if f["id"] not in order]
    notes = {str(n.get("id")): n for n in data.get("notes") or [] if isinstance(n, dict)}
    out, refused = [], []
    for rank, fid in enumerate(order, 1):
        f = {**byid[fid], "rank": rank}
        n = notes.get(fid)
        if n:
            got, why = _locked(f, n.get("title"), n.get("reason"))
            if got:
                f.update(title=got[0], reason=got[1], by=by)
            else:
                refused.append(f"{fid}: {why}")
        out.append(f)
    return {"findings": out, "refused": refused, "ordered": ordered}


def model_words(findings: list[dict[str, Any]], *, provider: str | None = None, model: str | None = None,
                api_key: str | None = None, base_url: str | None = None, context: str = "",
                client: Any = None) -> dict[str, Any]:
    """Ask the reader's own model (their key, the provider registry) to order and word the findings, and hold its
    reply to the claim lock. No key of ours is ever used."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        model, provider = cfg["model"], cfg["provider"]
    p = review_prompt(findings, context)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=2000,
        messages=[{"role": "system", "content": p["system"]}, {"role": "user", "content": p["prompt"]}],
    )
    res = apply_words(findings, resp.choices[0].message.content or "", by="key")
    res["model"] = f"{model} via {provider}" if provider else str(model)
    return res


# ── what lies outside the file: observed floods and gauges ──────────────────


def gather_floods(bbox: list[float], pad: float = RADAR_DEG) -> dict[str, Any]:
    """The radar detections (Sentinel-1, per 0.05 degree cell and month) and the news flood events (Groundsource)
    around a mesh, from the Archive's context mirror: ``{"radar": {"rows": [[lat, lon, "YYYY-MM", n]], "period"},
    "news": {"events": [...], "period"}}``. A source that cannot be read carries an ``error`` instead."""
    from aquascope.context._common import cells_for_bbox, dataset_info, num, read_cells

    w, s, e, n = (float(x) for x in bbox)
    box = (w - pad, s - pad, e + pad, n + pad)
    keys = cells_for_bbox(*box)
    out: dict[str, Any] = {"bbox": [round(x, 5) for x in box]}

    def inside(la: float | None, lo: float | None) -> bool:
        return la is not None and lo is not None and box[1] <= la <= box[3] and box[0] <= lo <= box[2]

    try:
        rows = read_cells("microsoft_floods", keys)
        if rows is None:
            out["radar"] = {"rows": None, "error": "the radar mirror is not published"}
        else:
            got = []
            for r in rows:
                la, lo, y, mo, k = (num(r.get(x)) for x in ("lat", "lon", "year", "month", "n"))
                if None in (y, mo, k) or not inside(la, lo):
                    continue
                got.append([la, lo, f"{int(y):04d}-{int(mo):02d}", int(k)])
            out["radar"] = {"rows": got, "period": list(RADAR_PERIOD)}
    except Exception as exc:  # noqa: BLE001 - a source that cannot be read is said, the other checks run
        out["radar"] = {"rows": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        rows = read_cells("groundsource", keys)
        if rows is None:
            out["news"] = {"events": None, "error": "the news mirror is not published"}
        else:
            info = dataset_info("groundsource") or {}
            events = []
            for r in rows:
                la, lo = num(r.get("lat")), num(r.get("lon"))
                bw, bs, be, bn = (num(r.get(x)) for x in ("west", "south", "east", "north"))
                if la is None or lo is None or not r.get("start_date"):
                    continue
                bw, bs, be, bn = (bw if bw is not None else lo, bs if bs is not None else la,
                                  be if be is not None else lo, bn if bn is not None else la)
                if be < box[0] or bw > box[2] or bn < box[1] or bs > box[3]:
                    continue
                events.append({"start": r["start_date"][:10], "end": (r.get("end_date") or "")[:10] or None,
                               "lat": la, "lon": lo, "west": bw, "south": bs, "east": be, "north": bn,
                               "area_km2": num(r.get("area_km2"))})
            first, last = str(info.get("first") or "2000-01")[:7], str(info.get("last") or "")[:7]
            out["news"] = {"events": events, "period": [first, last or first]}
    except Exception as exc:  # noqa: BLE001
        out["news"] = {"events": None, "error": f"{type(exc).__name__}: {exc}"}
    return out


def gauge_series(gauges: list[dict[str, Any]], start: str, end: str, days: int = 0) -> list[dict[str, Any]]:
    """Each gauge's observed daily record over the run's days (``days`` more either side), by
    :func:`aquascope.explore.fetch_series` (the Archive first, else the agency): the rows come back with
    ``series`` ([[date, value]]), ``variable``, ``unit`` and ``note``."""
    from aquascope.explore import fetch_series

    a = (date.fromisoformat(start[:10]) - timedelta(days=days)).isoformat()
    b = (date.fromisoformat(end[:10]) + timedelta(days=days)).isoformat()
    out = []
    for n_g, g in enumerate(gauges):
        row = dict(g)
        if n_g >= MAX_GAUGES:   # counted by the gauge check, not read
            out.append(row)
            continue
        try:
            got = fetch_series(g["source"], g["station_id"])
            s = got.get("series")
            if s is None or s.empty:
                row.update(series=[], note="it has no record in the Archive or at the agency")
            else:
                cut = s[(s.index >= a) & (s.index <= b + "T23:59:59")]
                row.update(series=[[str(i)[:10], float(v)] for i, v in cut.items() if v == v],
                           variable=got.get("variable"), unit=got.get("unit"),
                           note="")
                if cut.empty:
                    row["note"] = (f"its record has no day in the run (it runs {_day(str(s.index.min()))} to "
                                   f"{_day(str(s.index.max()))})")
        except Exception as exc:  # noqa: BLE001 - one gauge that cannot be read leaves the others
            row.update(series=[], note=f"its record could not be read ({type(exc).__name__})")
        out.append(row)
    return out


def gather_context(m: Mesh, *, floods: bool = True, gauges: bool = True,
                   stations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Observed floods around a mesh and the Archive gauges inside it with their records over the run, for
    :func:`review`. ``stations`` are catalog rows (``source, station_id, name, lat, lon``); without them the
    Archive's catalog is read."""
    out: dict[str, Any] = {}
    if floods:
        out["floods"] = gather_floods(m.bbox())
    if gauges:
        if stations is None:
            from aquascope.archive.catalog import load_stations

            w, s, e, n = m.bbox()
            stations = [{"source": r["source"], "station_id": r["station_id"], "name": r.get("name"),
                         "lat": r["latitude"], "lon": r["longitude"]} for r in load_stations()
                        if r.get("latitude") is not None and r.get("longitude") is not None
                        and s <= r["latitude"] <= n and w <= r["longitude"] <= e]
        inside = gauges_inside(m, stations)
        out["gauges"] = gauge_series(inside, m.times[0], m.times[-1]) if (inside and m.times) else inside
    return out


def review_file(path: str, *, crs: str | int | None = None, context: bool = True, top: int = TOP_DEEPEST,
                llm: dict[str, Any] | None = None, geometry: str | None = None) -> dict[str, Any]:
    """Review a model result on this machine (``aquascope mesh review FILE`` and the MCP tool ``mesh_review``):
    read it, gather the observed floods and gauges (unless ``context`` is False), run the checks and, with
    ``llm`` = ``{provider, model, api_key, base_url}``, have the reader's own model order and word them,
    claim-locked. ``geometry`` is the HEC-RAS geometry file (``*.g01.hdf``) for a stripped plan file. ``ok`` is
    false with a plain error for a file the reader cannot open."""
    from aquascope.io.mesh import read_mesh

    try:
        m = read_mesh(path, crs=crs, geometry=geometry)
    except MeshError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        errors = []
        ctx: dict[str, Any] = {}
        if context:
            try:
                ctx = gather_context(m)
            except Exception as exc:  # noqa: BLE001 - the checks on the file still run
                errors.append(f"context: {type(exc).__name__}: {exc}")
        res = review(m, context=ctx, top=top)
        res["by"] = "rules"
        if llm and res["findings"]:
            try:
                got = model_words(res["findings"], **llm)
                res.update(findings=got["findings"], refused=got["refused"], by=got["model"])
            except (RuntimeError, ValueError) as exc:
                errors.append(f"model: {exc}")
        if errors:
            res["errors"] = errors
        return res
    finally:
        m.close()
