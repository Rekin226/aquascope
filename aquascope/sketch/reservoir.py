"""Draw a reservoir on a river and see it flatten a flood (#662): level-pool routing. Experimental.

A reservoir whose outflow depends only on its water level is routed with the storage-indication method (the
level-pool or Modified Puls method): over each step ``dt`` the continuity equation, written as

    2 S[j+1] / dt + O[j+1] = I[j] + I[j+1] + 2 S[j] / dt - O[j]

has everything on the right known, and the left is a function of the outflow alone once the storage-outflow
relation is known; :func:`level_pool` reads ``O[j+1]`` off that working curve by linear interpolation, as the
textbooks do with their working tables. The check (:data:`CHECK`, docs/sketchpad.md) is the USDA NRCS National
Engineering Handbook, Part 630, Chapter 17 (Amend. 69, April 2014), Example 17-2: its outflow table (Table 17-12)
is reproduced within 0.51 ft3/s, the peak 364 ft3/s on day 5.3 as in the book.

On the map the reservoir is a drawn area on a river:

* :func:`plan` finds the river: every GEOGLOWS reach drawn inside the area (the stream tiles at zoom 12), the
  biggest of them, and the dam at the downstream end of the most downstream one; and the terrain window to read.
* :func:`stage_storage` turns the Terrarium terrain inside the area into the water's area and volume at each level
  above the lowest ground: the stage-area and stage-storage curves of the area as drawn (a level higher than the
  ground at the area's edge would spread past it; :func:`stage_storage` says along how much of the edge).
* :func:`spillway` rates an uncontrolled spillway as a broad-crested weir, ``Q = C b H^1.5``, with
  :data:`aquascope.floodsim.WEIR_C` (critical depth over the crest, about 1.705 in SI) unless another is given.
* :func:`route_reservoir` routes a flood through it and on down the river with Route it
  (:mod:`aquascope.routing`): the T-year flow or a peak entering at the dam's reach, a flood routed from a reach
  upstream, or a drawn storm (#670). The run is made twice, with the reservoir and without, so the card can say
  how much it flattens the flood at the dam and downstream.

Everything is numpy over what the page or the CLI reads, so the Explorer's Pyodide worker runs it unchanged.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "CHECK", "DEFAULTS", "METHOD", "NEH_17_2", "level_pool", "neh_check", "plan", "route_reservoir", "spillway",
    "stage_storage", "terrain_for",
]

METHOD = {
    "name": "Level-pool (storage-indication) routing",
    "text": ("2S2/dt + O2 = I1 + I2 + 2S1/dt - O1 each step, the outflow read off the working curve (2S/dt + O "
             "against O) by linear interpolation; the storage from the terrain inside the drawn area, the outflow "
             "from a broad-crested spillway, Q = C b H^1.5."),
    "citation": ("USDA NRCS (2014), National Engineering Handbook, Part 630 Hydrology, Chapter 17, Flood Routing "
                 "(210-VI-NEH, Amend. 69), storage-indication method, Example 17-2; Chow, Maidment and Mays (1988), "
                 "Applied Hydrology, section 8.2 (level pool routing)."),
}

#: NEH Part 630, Chapter 17 (Amend. 69, April 2014), Example 17-2 (pp. 17-18 to 17-24), in the book's own units:
#: the working curve of Table 17-11 (discharge in ft3/s, storage above the first-stage crest at 580 ft in
#: ft3/s-days), and the operations table, Table 17-12 (time in days, inflow and outflow in ft3/s; the 0.5-day step
#: before 4.5 days and after 6.0 days, 0.1 day between, as the book changes its step).
NEH_17_2 = {
    "source": ("USDA NRCS (2014), National Engineering Handbook, Part 630 Hydrology, Chapter 17 Flood Routing, "
               "210-VI-NEH, Amend. 69, April 2014, Example 17-2, Tables 17-11 and 17-12, pp. 17-22 and 17-24"),
    "url": "https://directives.nrcs.usda.gov/sites/default/files2/1712930622/21209.pdf",
    "elevation_ft": [580.0, 581.2, 582.2, 583.2, 584.2, 585.0, 587.0, 587.5, 588.0, 588.5, 590.0, 592.0, 595.0],
    "outflow_cfs": [0.0, 11.6, 32.8, 60.3, 92.8, 114.6, 149.0, 203.6, 289.0, 353.0, 365.0, 382.0, 401.0],
    "storage_cfs_days": [0.0, 64.5, 118.3, 172.1, 225.9, 268.9, 423.8, 462.5, 501.2, 539.9, 656.1, 863.5, 1174.5],
    "time_days": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4,
                  5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    "inflow_cfs": [0, 70, 79, 84, 88, 99, 110, 128, 156, 245, 269, 308, 380, 522, 2002, 1049, 577, 393, 312, 267,
                   217, 200, 184, 174, 164, 138, 118, 106, 94, 88, 82, 75, 67],
    "outflow_book_cfs": [0, 3, 9, 19, 31, 45, 59, 74, 92, 115, 119, 122, 127, 134, 212, 358, 363, 364, 364, 363,
                         362, 360, 359, 357, 355, 249, 174, 147, 142, 137, 132, 126, 120],
}

#: The check, run by :func:`level_pool` on :data:`NEH_17_2` (tests/test_sketch_reservoir.py repeats it).
CHECK = {
    "reference": NEH_17_2["source"],
    "how": ("Table 17-12's inflow routed through Table 17-11's working curve with level_pool, the book's steps of "
            "0.5 and 0.1 day, from an empty pool at the first-stage crest; every outflow compared with the book's, "
            "which the book rounds to whole ft3/s at each step."),
    "tolerance": "every outflow within 1 ft3/s of the book's; the peak and its day the book's",
    "max_abs_diff_cfs": 0.51,
    "peak_cfs": 364.2, "peak_day": 5.3, "book_peak_cfs": 364, "book_peak_days": [5.3, 5.4],
    "summary": ("Checked against NRCS NEH 630 Chapter 17, Example 17-2: its outflow table is reproduced within "
                "0.51 ft3/s, the peak 364 ft3/s on day 5.3 as in the book."),
}

#: The dam when none is given (illustrative, for the reader to set): the spillway's crest 10 m above the lowest ground,
#: the top of the dam 5 m above the crest, the spillway 30 m wide.
DEFAULTS = {"crest_m": 10.0, "rise_m": 5.0, "width_m": 30.0}

#: The terrain tiles (the Explorer reads the same ones for Flood it).
TERRAIN_CREDIT = ("Terrain: Terrarium tiles (Mapzen/Tilezen joerd on AWS Open Data: SRTM, GMTED2010, ETOPO1, national "
                  "lidar and others)")
#: The most terrain cells read for an area, and the finest and coarsest zooms used.
MAX_CELLS = 1024 * 1024
ZOOMS = (14, 9)
#: The levels the stage-storage curve is tabulated at, at most.
N_LEVELS = 60
NOTE_TERRAIN = ("The storage is the terrain's: where a lake is already there, the tiles hold its surface, not its "
                "bed, so only the storage above today's water is counted.")
NOTE_LEVEL = ("Level pool: the water surface is flat, and the outflow depends only on the level over the spillway's "
              "crest. No gates, no outlet works, the pool starting at the crest.")


# ── the method ──────────────────────────────────────────────────────────────


def level_pool(inflow: Any, storage: Any, outflow: Any, *, dt: float | None = None, times: Any = None,
               s0: float = 0.0, o0: float | None = None) -> dict[str, Any]:
    """Storage-indication (level-pool) routing of ``inflow`` through a reservoir.

    ``storage`` and ``outflow`` are the storage-outflow relation (increasing, the same length), in units that agree
    with the time step: storage in flow units times time units. ``dt`` is a constant step, or ``times`` the times
    of the inflow values (the step may change, as in NEH 630 Example 17-2). ``s0`` is the storage at the start and
    ``o0`` the outflow (by default the relation's outflow at ``s0``). Each step solves
    ``2 S2/dt + O2 = I1 + I2 + 2 S1/dt - O1`` for ``O2`` by linear interpolation on the working curve
    ``2S/dt + O`` against ``O``; above the relation's last point the working curve goes on along its last segment.

    Returns ``outflow`` and ``storage`` at each inflow time.
    """
    import numpy as np

    q_in = np.asarray(inflow, dtype=float)
    s_tab = np.asarray(storage, dtype=float)
    o_tab = np.asarray(outflow, dtype=float)
    if q_in.ndim != 1 or len(q_in) < 2:
        raise ValueError("inflow needs two values or more")
    if s_tab.shape != o_tab.shape or len(s_tab) < 2:
        raise ValueError("storage and outflow are two lists of the same length, two points or more")
    if np.any(np.diff(s_tab) <= 0) or np.any(np.diff(o_tab) < 0):
        raise ValueError("storage must rise and outflow must not fall along the relation")
    if times is None:
        if not dt or dt <= 0:
            raise ValueError("give dt (a constant step) or times")
        t = np.arange(len(q_in)) * float(dt)
    else:
        t = np.asarray(times, dtype=float)
        if t.shape != q_in.shape or np.any(np.diff(t) <= 0):
            raise ValueError("times must rise and match the inflow")
    out = np.empty_like(q_in)
    sto = np.empty_like(q_in)
    sto[0] = float(s0)
    out[0] = float(np.interp(s0, s_tab, o_tab)) if o0 is None else float(o0)
    # The last segment of the relation, carried on above it (a level above the highest point tabulated).
    ds, do = s_tab[-1] - s_tab[-2], o_tab[-1] - o_tab[-2]
    for j in range(len(q_in) - 1):
        step = t[j + 1] - t[j]
        w = q_in[j] + q_in[j + 1] + 2.0 * sto[j] / step - out[j]
        curve = 2.0 * s_tab / step + o_tab
        if w <= curve[-1]:
            o2 = float(np.interp(w, curve, o_tab))
        else:
            slope = do / (2.0 * ds / step + do)
            o2 = float(o_tab[-1] + (w - curve[-1]) * slope)
        out[j + 1] = o2
        sto[j + 1] = (w - o2) * step / 2.0
    return {"outflow": out, "storage": sto, "time": t}


def spillway(levels: Any, crest: float, width_m: float, coefficient: float | None = None) -> Any:
    """An uncontrolled spillway's outflow (m3/s) at each water level (m): ``C b H^1.5`` with ``H`` the level over the
    crest, a broad-crested weir (:data:`aquascope.floodsim.WEIR_C`, critical depth over the crest, unless
    ``coefficient`` is given), and none below the crest."""
    import numpy as np

    from aquascope.floodsim import WEIR_C

    c = WEIR_C if coefficient is None else float(coefficient)
    if not (width_m > 0 and c > 0):
        raise ValueError("the spillway's width and coefficient must be above 0")
    head = np.clip(np.asarray(levels, dtype=float) - float(crest), 0.0, None)
    return c * float(width_m) * head ** 1.5


# ── the terrain: stage, area and storage of the area drawn ──────────────────


def _ring(polygon: Any) -> list[list[float]]:
    ring = [[float(p[0]), float(p[1])] for p in polygon or []]
    if len(ring) > 3 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3 or any(abs(x) > 180 or abs(y) > 85 for x, y in ring):
        raise ValueError("a reservoir needs an area of three or more [lon, lat] points")
    return ring


def terrain_for(polygon: Any) -> dict[str, Any]:
    """The Terrarium window that covers the area: the finest zoom (14 at most) at which it has at most
    :data:`MAX_CELLS` cells. Returns the window (``x0``, ``y0``, ``nx``, ``ny``, ``zoom``, ``dx``, ``bounds``) for
    the page or :func:`aquascope.floodsim.terrain_window` to read."""
    from aquascope import floodsim as fs

    ring = _ring(polygon)
    w, e = min(p[0] for p in ring), max(p[0] for p in ring)
    s, n = min(p[1] for p in ring), max(p[1] for p in ring)
    for zoom in range(ZOOMS[0], ZOOMS[1] - 1, -1):
        x0, y0 = fs._world_px(w, n, zoom)
        x1, y1 = fs._world_px(e, s, zoom)
        x0, y0 = int(math.floor(x0)) - 2, int(math.floor(y0)) - 2
        nx, ny = int(math.ceil(x1)) + 2 - x0, int(math.ceil(y1)) + 2 - y0
        if nx * ny <= MAX_CELLS:
            return fs.window(x0, y0, nx, ny, zoom)
    raise ValueError("the area is too large for a reservoir; draw it smaller")


def stage_storage(z: Any, win: dict[str, Any], polygon: Any, *, top: float | None = None,
                  above_bed: float | None = None, n_levels: int = N_LEVELS) -> dict[str, Any]:
    """The stage-area and stage-storage curves of the drawn area from its terrain.

    ``z`` is the window's terrain (m, rows north to south, :func:`terrain_for`'s window), ``polygon`` the area.
    The cells whose centres are inside count; each covers the window's pixel area at its latitude. At a level
    ``h`` the water's area is the cells below ``h`` and its volume ``sum(h - z)`` over them, from the lowest ground
    (the bed at the dam, as the terrain has it) up to ``top`` (or ``above_bed`` m above the lowest ground; by
    default the highest ground inside), at
    ``n_levels`` levels. ``edge_below_pct`` says, for each level, how much of the area's edge has ground lower than
    it: water there would spread past the area drawn."""
    import numpy as np

    from aquascope import floodsim as fs
    from aquascope.routing import _inside

    ring = _ring(polygon)
    zz = np.asarray(z, dtype=float).reshape(int(win["ny"]), int(win["nx"]))
    rows, cols = np.mgrid[0:zz.shape[0], 0:zz.shape[1]]
    lon, lat = fs.cell_lonlat(win, rows.ravel(), cols.ravel())
    inside = _inside(lon, lat, ring).reshape(zz.shape)
    if inside.sum() < 9:
        raise ValueError("the area covers fewer than 9 terrain cells; draw it larger")
    cell_m = fs.pixel_size_m(float(np.mean(lat.reshape(zz.shape)[inside])), int(win["zoom"]))
    cell_a = cell_m * cell_m
    ground = zz[inside]
    bed = float(ground.min())
    hi = (bed + float(above_bed)) if above_bed is not None else float(ground.max()) if top is None else float(top)
    if hi <= bed:
        raise ValueError(f"the top ({hi:.1f} m) must be above the lowest ground in the area ({bed:.1f} m)")
    levels = np.linspace(bed, hi, int(n_levels))
    area = np.array([float(np.sum(ground < h)) * cell_a for h in levels])
    volume = np.array([float(np.sum(np.clip(h - ground, 0.0, None))) * cell_a for h in levels])
    # The edge: the inside cells with an outside neighbour (four-connected).
    pad = np.pad(inside, 1, constant_values=False)
    edge = inside & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    edge_z = zz[edge]
    edge_pct = np.array([100.0 * float(np.mean(edge_z < h)) for h in levels])
    return {"levels_m": levels, "area_m2": area, "storage_m3": volume, "edge_below_pct": edge_pct, "bed_m": bed,
            "top_m": hi, "cell_m": cell_m, "n_cells": int(inside.sum()), "area_drawn_m2": float(inside.sum()) * cell_a,
            "edge_ground_min_m": float(edge_z.min()) if edge_z.size else None}


# ── the river: where the dam is ─────────────────────────────────────────────


def plan(polygon: Any, *, reaches: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The river under a drawn reservoir and the terrain to read for it.

    Every GEOGLOWS reach whose line has a point inside the area (the stream tiles at zoom 12, every reach drawn;
    ``reaches`` stands in for them in tests) is in the lake. The river is the biggest of them (the highest Strahler
    order), and the dam is at the downstream end of its most downstream reach in the lake: ``dam`` (its
    ``river_id``, order, and the point of its line inside the area nearest the reach below). Returns that with the
    terrain window (:func:`terrain_for`)."""
    import numpy as np

    from aquascope import floodsim as fs
    from aquascope import rivers
    from aquascope.routing import _inside

    ring = _ring(polygon)
    w, e = min(p[0] for p in ring), max(p[0] for p in ring)
    s, n = min(p[1] for p in ring), max(p[1] for p in ring)
    found = reaches if reaches is not None else fs.stream_reaches([w, s, e, n])
    lake: dict[int, dict[str, Any]] = {}
    for rid, item in found.items():
        pts = [p for line in item["lines"] for p in line]
        if not pts:
            continue
        arr = np.asarray(pts, dtype=float)
        hit = _inside(arr[:, 0], arr[:, 1], ring)
        if hit.any():
            lake[int(rid)] = {"order": int(item.get("order") or 0), "inside": arr[hit].tolist(), "all": arr}
    if not lake:
        return {"ok": False, "error": "No river runs through the area drawn: draw the reservoir across a river."}
    top = max(r["order"] for r in lake.values())
    main = [rid for rid, r in lake.items() if r["order"] == top]
    clat = sum(p[1] for p in ring) / len(ring)
    clon = sum(p[0] for p in ring) / len(ring)
    net = rivers._network_for(main[0], clat, clon)
    # Each reach in the lake drains, through the lake, to a reach whose water leaves it: the dam is on the biggest
    # river's exit that the most reaches of the lake drain to (the river through the lake, not a side stream).
    members = set(lake)
    drains: dict[int, int] = {}
    for rid in lake:
        j, seen = rid, set()
        while True:
            i = net.index(j)
            ds = int(net.ds[i]) if i is not None else -1
            if ds not in members or ds in seen:
                break
            seen.add(ds)
            j = ds
        drains[j] = drains.get(j, 0) + 1
    exits = [rid for rid in drains if lake[rid]["order"] == top] or list(drains)
    dam_id = max(exits, key=lambda rid: (drains[rid], lake[rid]["order"]))
    below = net.index(dam_id)
    ds_id = int(net.ds[below]) if below is not None else -1
    inside = np.asarray(lake[dam_id]["inside"], dtype=float)
    if ds_id in found and found[ds_id]["lines"]:
        ref = np.asarray([p for line in found[ds_id]["lines"] for p in line], dtype=float)
        d = np.min(np.hypot(inside[:, None, 0] - ref[None, :, 0], inside[:, None, 1] - ref[None, :, 1]), axis=1)
        dam_pt = inside[int(np.argmin(d))]
    else:
        dam_pt = inside[-1]
    return {
        "ok": True, "polygon": [[round(x, 6), round(y, 6)] for x, y in ring],
        "dam": {"river_id": int(dam_id), "order": int(top), "lon": round(float(dam_pt[0]), 6),
                "lat": round(float(dam_pt[1]), 6), "downstream_id": ds_id},
        "lake_reaches": sorted(lake), "river_reaches": sorted(main), "window": terrain_for(ring),
        "terrain_source": TERRAIN_CREDIT,
    }


# ── the run: a flood through the reservoir and on down the river ────────────


def _series(q: Any) -> list[float]:
    from aquascope.routing import _sig

    return [_sig(float(v)) for v in q]


def route_reservoir(polygon: Any, *, z: Any = None, window: dict[str, Any] | None = None,
                    dam: dict[str, Any] | None = None,
                    crest_m: float = DEFAULTS["crest_m"], top_m: float | None = None,
                    width_m: float = DEFAULTS["width_m"],
                    coefficient: float | None = None, return_period: float | None = None,
                    peak_cms: float | None = None, duration_h: float = 24.0, from_river_id: int | None = None,
                    storm: dict[str, Any] | None = None, hours: int | None = None,
                    frames: bool = True) -> dict[str, Any]:
    """A flood through a drawn reservoir and on down the river: level-pool routing inside Route it.

    ``polygon`` is the reservoir as drawn, ``z`` the terrain of ``window`` and ``dam`` :func:`plan`'s dam (all three
    found and read here when not given: the page reads the terrain tiles itself and passes them). The levels are in
    m above the lowest ground in the area: the spillway's crest ``crest_m`` (the pool starts there) and the top of
    the dam ``top_m`` (by default the crest plus 5 m); the spillway is ``width_m`` wide with ``coefficient`` (a
    broad-crested weir by default).

    What comes in: ``storm`` (route_storm's arguments: ``polygon``, ``rain_mm``, ``duration_h``,
    ``runoff_coefficient``; the storm's runoff routed down to the dam), or a flood of ``peak_cms`` or the
    ``return_period``-year flow over ``duration_h`` entering at ``from_river_id`` (a reach upstream, routed down to the
    dam) or at the dam's own reach. The run is :func:`aquascope.routing.route` or ``route_storm`` with the reservoir
    in the dam's reach, so the outflow goes on down the river; it is run without the reservoir too.

    Returns the routed result (``reaches``, ``frames`` and the rest, as Route it's) with ``reservoir``: the inflow
    and outflow hydrographs at the dam, the peaks and the peak reduction, the highest level and storage, whether
    the dam is overtopped, the stage-storage table and its source, and ``without`` (the same run with no
    reservoir: its peak at the dam and furthest down).
    """
    import numpy as np

    from aquascope import routing

    if dam is None or window is None:
        p = plan(polygon)
        if not p["ok"]:
            return {**p, "attribution": routing.ATTRIBUTION}
        dam, window = p["dam"], p["window"]
    if z is None:
        from aquascope import floodsim as fs

        z = fs.terrain_window(window["x0"], window["y0"], window["nx"], window["ny"], window["zoom"])["z"]
    crest = float(crest_m)
    top = crest + DEFAULTS["rise_m"] if top_m is None else float(top_m)
    if not (0 < crest < top):
        raise ValueError("the crest must be above the lowest ground (0 m) and below the top of the dam")
    # Tabulated a little above the top of the dam, so a flood that overtops it is still read off the terrain.
    curve = stage_storage(z, window, polygon, above_bed=top + max(2.0, 0.5 * top))
    bed = curve["bed_m"]
    levels, vol = curve["levels_m"], curve["storage_m3"]
    # The relation from the crest up: storage above the crest's and the spillway's outflow at each level.
    lv = np.concatenate([[bed + crest], levels[levels > bed + crest]])
    st = np.interp(lv, levels, vol)
    keep = np.concatenate([[True], np.diff(st) > 0])
    lv, st = lv[keep], st[keep]
    if len(lv) < 2:
        raise ValueError("the area holds no water between the crest and the top of the dam; draw it larger or "
                         "lower the crest")
    q = spillway(lv, bed + crest, width_m, coefficient)
    s_crest = float(st[0])
    rec: dict[str, Any] = {}

    def through(inflow: Any) -> Any:
        res = level_pool(inflow, st - s_crest, q, dt=routing.DT_S, s0=0.0, o0=0.0)
        rec.update(inflow=np.asarray(inflow, dtype=float), outflow=res["outflow"], storage=res["storage"])
        return res["outflow"]

    dam_id = int(dam["river_id"])
    common = {"hours": hours}
    # The flood as given, so a page can send the same one through a changed reservoir.
    flood: dict[str, Any] = ({"storm": storm} if storm is not None else
                             {k: v for k, v in (("from_river_id", from_river_id), ("return_period", return_period),
                                                ("peak_cms", peak_cms), ("duration_h", duration_h)) if v is not None})
    if storm is not None:
        args = {k: storm[k] for k in ("rain_mm", "duration_h", "runoff_coefficient") if k in storm}
        run = routing.route_storm(storm["polygon"], frames=frames, through={dam_id: through}, **args, **common)
        base = routing.route_storm(storm["polygon"], frames=False, **args, **common)
        what = (f"a storm of {float(storm['rain_mm']):g} mm in {float(storm.get('duration_h', 6.0)):g} h "
                "over the area drawn")
    else:
        start = dam_id if from_river_id is None else int(from_river_id)
        kw = {"peak_cms": peak_cms, "return_period": return_period, "duration_h": duration_h, "along": True,
              "frames": frames}
        run = routing.route(start, through={dam_id: through}, **kw, **common)
        base = routing.route(start, **{**kw, "frames": False, "along": False}, **common)
        what = (f"{run['inflow']['source']} over {duration_h:g} h" + ("" if start == dam_id
                                                                     else f" from reach {start}"))
    if not run.get("ok"):
        return run
    if "inflow" not in rec:
        return {"ok": False, "error": (f"The flood does not reach the dam's reach ({dam_id}): route it from a reach "
                                       "upstream of the reservoir, or draw the storm over its catchment."),
                "attribution": routing.ATTRIBUTION}
    q_in, q_out, s_run = rec["inflow"], rec["outflow"], rec["storage"]
    peak_in, peak_out = float(q_in.max()), float(q_out.max())
    s_max = float(s_run.max()) + s_crest
    level_max = float(np.interp(s_max, vol, levels)) if s_max <= vol[-1] else float(levels[-1])
    over = bool(s_max > float(np.interp(bed + top, levels, vol)) + 1e-6)
    t_in, t_out = int(np.argmax(q_in)), int(np.argmax(q_out))
    ref = [r for r in base.get("reaches") or [] if int(r["river_id"]) == dam_id]
    far = run.get("furthest") or {}
    if storm is not None and run.get("outlets"):
        # A storm has no single path: "furthest down" is the outlet the dam drains to.
        from aquascope import rivers

        net = rivers._network_for(dam_id)
        outs = {int(r["river_id"]): r for r in run["outlets"]}
        j, steps = dam_id, 0
        while j not in outs and steps < 100_000:
            i = net.index(j)
            if i is None or int(net.ds[i]) < 0:
                break
            j, steps = int(net.ds[i]), steps + 1
        out = outs.get(j)
        far = {"river_id": out["river_id"], "peak_cms": out["peak_cms"]} if out else {}
        base_rows = base.get("outlets") or []
    else:
        base_rows = base.get("reaches") or []
    far_id = int(far.get("river_id", -1)) if far else -1
    far_base = next((r for r in base_rows if int(r["river_id"]) == far_id),
                    None)
    i_top = int(np.searchsorted(levels, bed + top))
    edge_top = float(curve["edge_below_pct"][min(i_top, len(levels) - 1)])
    res_block = {
        "dam": dam, "crest_m": crest, "top_m": top, "width_m": float(width_m),
        "coefficient": float(coefficient) if coefficient is not None else round(float(_weir_c()), 4),
        "bed_m": round(bed, 2), "crest_level_m": round(bed + crest, 2), "top_level_m": round(bed + top, 2),
        "inflow_cms": _series(q_in), "outflow_cms": _series(q_out), "dt_h": routing.DT_S / 3600.0,
        "peak_in_cms": routing._sig(peak_in), "peak_out_cms": routing._sig(peak_out),
        "peak_in_h": t_in * routing.DT_S / 3600.0, "peak_out_h": t_out * routing.DT_S / 3600.0,
        "reduction_pct": round(100.0 * (1.0 - peak_out / peak_in), 1) if peak_in > 0 else None,
        "delay_h": (t_out - t_in) * routing.DT_S / 3600.0,
        "flood_storage_m3": routing._sig(float(s_run.max())), "stored_end_m3": routing._sig(float(s_run[-1])),
        "max_level_m": round(level_max, 2),
        "max_depth_m": round(level_max - bed, 2), "overtopped": over,
        "storage_at_crest_m3": routing._sig(s_crest),
        "edge_below_top_pct": round(edge_top, 1),
        "curve": {"level_m": [round(float(v), 2) for v in levels], "area_m2": [routing._sig(float(v)) for v in
                                                                               curve["area_m2"]],
                  "storage_m3": [routing._sig(float(v)) for v in vol]},
        "cells": curve["n_cells"], "cell_m": round(curve["cell_m"], 1), "flood": flood,
        "area_drawn_km2": round(curve["area_drawn_m2"] / 1e6, 3), "what": what,
        "terrain_source": TERRAIN_CREDIT, "method": METHOD, "check": CHECK["summary"],
        "without": {"peak_at_dam_cms": ref[0]["peak_cms"] if ref else None,
                    "furthest_river_id": far.get("river_id"),
                    "furthest_peak_cms": far_base["peak_cms"] if far_base else None,
                    "furthest_peak_with_cms": far.get("peak_cms")},
    }
    notes = list(run.get("notes") or []) + [NOTE_LEVEL, NOTE_TERRAIN]
    if over:
        notes.append("The water rises above the top of the dam: it would be overtopped. The spillway alone is "
                     "routed above the top; raise the dam, widen the spillway or lower its crest.")
    if edge_top > 0:
        notes.append(f"Along {edge_top:.0f} % of the area's edge the ground is lower than the top of the dam: a lake "
                     "that full would spread past the area drawn, which this storage leaves out.")
    run.update(reservoir=res_block, notes=notes)
    run["message"] = (f"A reservoir of {res_block['area_drawn_km2']:g} km² drawn on reach {dam_id}, spillway "
                      f"{float(width_m):g} m wide at {crest:g} m: {what} peaks at {peak_in:,.0f} m³/s at the dam and "
                      f"leaves at {peak_out:,.0f} m³/s, {res_block['reduction_pct']:g} % lower and "
                      f"{res_block['delay_h']:g} h later. The water rises to {res_block['max_depth_m']:g} m above "
                      "the lowest ground" + (", over the top of the dam." if over else "."))
    return run


def neh_check() -> dict[str, Any]:
    """The check, run now: NEH 630 Example 17-2's inflow through its working curve with :func:`level_pool`, and the
    book's outflow beside each value (``rows``: day, inflow, book, here, difference, all ft3/s)."""
    import numpy as np

    ex = NEH_17_2
    got = level_pool(ex["inflow_cfs"], ex["storage_cfs_days"], ex["outflow_cfs"], times=ex["time_days"])["outflow"]
    rows = [{"day": t, "inflow_cfs": i, "book_cfs": b, "here_cfs": round(float(g), 1),
             "diff_cfs": round(float(g) - b, 2)}
            for t, i, b, g in zip(ex["time_days"], ex["inflow_cfs"], ex["outflow_book_cfs"], got)]
    j = int(np.argmax(got))
    return {"reference": ex["source"], "url": ex["url"], "rows": rows,
            "max_abs_diff_cfs": round(float(np.max(np.abs(got - np.asarray(ex["outflow_book_cfs"])))), 2),
            "peak_cfs": round(float(got[j]), 1), "peak_day": ex["time_days"][j], "summary": CHECK["summary"]}


def _weir_c() -> float:
    from aquascope.floodsim import WEIR_C

    return WEIR_C
