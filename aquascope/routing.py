"""Route it (#656): a flood wave down GEOGLOWS's own river network, Muskingum routing reach by reach.

GEOGLOWS v2 publishes the routing configuration of its global model per processing unit (VPU) on S3, readable
from any page (``Access-Control-Allow-Origin: *``): ``rapid_connect.csv`` (each reach and the reach it drains
to, -1 at an outlet), ``k.csv`` (each reach's Muskingum travel time, in seconds) and ``x.csv`` (its weight, 0.36
in the four units checked), in the same row order, which is topological: every reach comes after every
reach that drains to it. This module routes water over that network with the same equation the model uses:

* :func:`plan` gives each reach a number of sub-reaches and of sub-steps so that every Muskingum coefficient
  is non-negative (``2kx <= dt <= 2k(1 - x)`` for each sub-reach and sub-step), as river-route, the GEOGLOWS
  team's router, does: a reach too short for the step takes sub-steps of its own with its inflow interpolated
  linearly across them; a reach too long for it is split into sub-reaches in series.
* :func:`route_network` routes any set of reaches, upstream to downstream, each reach's whole series at once
  (the recurrence ``Q[t+1] = C3 Q[t] + b[t]`` is a first-order linear filter).
* :func:`route` is the map's question: water entering at a reach (a number for a duration, a release, or the
  reach's T-year flow), routed down its path to the sea; per reach the peak, the hour of the peak, the wave's
  front and how much the peak has flattened, with the gauges and the towns on the way.
* :func:`route_storm` turns a drawn storm into runoff on the reaches under it (rain depth times a runoff
  coefficient times each unit catchment's area, spread evenly over the storm) and routes it.
* :func:`retro_check` is the check against GEOGLOWS's own retrospective simulation: it routes the hourly
  retrospective flow of an upstream reach, the tributaries that join on the way and the runoff of the path's own
  catchments to a reach downstream, and compares the result with the retrospective there (:data:`CHECK` has
  the numbers, docs/explorer.md how they were made).

Muskingum is linear, so the wave added at a reach routes on its own: what :func:`route` returns is the flow
that water adds to whatever the river already carries, not the river's total flow.

Everything is numpy and scipy over plain CSV reads (no pyarrow), so the Explorer's Pyodide worker runs it
unchanged. GEOGLOWS v2 is CC BY 4.0; the river network TDX-Hydro (NGA), CC BY-SA 4.0.
"""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from typing import Any

__all__ = [
    "CHECK", "DT_S", "MAX_HOURS", "METHOD", "SHAPES", "hydrograph", "places_on_path", "plan", "retro_check",
    "route", "route_network", "route_storm",
]

#: The routing step, in seconds: GEOGLOWS v2 routes hourly ERA5 runoff, so its step is an hour.
DT_S = 3600
#: The most hours one run routes (30 days).
MAX_HOURS = 720
#: The most reaches followed down from the start.
MAX_PATH = 4000
#: The most reaches a storm may fall on, and the most reaches a storm's run routes.
MAX_STORM_REACHES = 20_000
MAX_STORM_NETWORK = 60_000
#: The most frames returned for the map's animation, and the most reaches each frame carries.
MAX_FRAMES = 120
MAX_FRAME_REACHES = 4000
#: Below this share of the peak that entered, a reach's wave is too small to report.
MIN_SHARE = 0.01
#: Inflow shapes: a constant flow for the duration (a release), or a triangle that peaks at a third of it.
SHAPES = ("block", "triangle")
#: How close a gauge or a town must be to the path to be given an arrival time, in km.
PLACE_KM = 3.0

METHOD = {
    "name": "Muskingum routing on the GEOGLOWS v2 network",
    "text": ("Q[t+1] = C1 I[t+1] + C2 I[t] + C3 Q[t] reach by reach, upstream to downstream, with GEOGLOWS's own "
             "k (seconds) and x per reach and its rapid_connect topology. Each reach takes the sub-reaches and "
             "sub-steps that keep every coefficient non-negative (2kx <= dt <= 2k(1 - x)), as river-route does; "
             "the hourly step is GEOGLOWS's own."),
    "citation": ("McCarthy (1938), the Muskingum method; Cunge (1969), J. Hydraul. Res. 7(2), 205-230; David et al. "
                 "(2011), J. Hydrometeorol. 12(5), 913-934, doi:10.1175/2011JHM1345.1 (RAPID); river-route, the "
                 "GEOGLOWS team's router (github.com/geoglows/river-route)."),
}

#: The check against GEOGLOWS's own hourly retrospective (docs/explorer.md, Route a flood): :func:`retro_check`
#: run on 2026-10-27 with ``lateral="specific"`` and the defaults; every number below is that function's output.
CHECK: dict[str, Any] = {
    "date": "2026-10-27",
    "reference": "GEOGLOWS v2 hourly retrospective simulation (retrospective/hourly.zarr), CC BY 4.0",
    "how": ("The retrospective flow leaving the upstream reach and every tributary draining at least 1 % of the "
            "area at the downstream reach, routed with this module; smaller tributaries and the path's own unit "
            "catchments get their area times the specific runoff of the flows read. Hourly, 10 days of spin-up "
            "not scored. 'No routing' adds the same inflows with no travel time."),
    "tolerance": "NSE at least 0.85, peak within 10 %, peak timing within 12 h",
    "cases": [
        {"reach": "Elbe, Dresden to Magdeburg (June 2013)", "upstream": 230494421, "downstream": 230538985,
         "window": ["2013-05-25", "2013-07-04"], "n_path": 89, "k_path_h": 106.7,
         "nse": 0.9818, "peak_error_pct": 2.78, "peak_timing_h": 0.0, "bias_pct": 2.48, "nse_no_routing": 0.4255},
        {"reach": "Rhine, Basel to Cologne (winter 2017-18)", "upstream": 230301988, "downstream": 230515098,
         "window": ["2017-12-20", "2018-02-09"], "n_path": 166, "k_path_h": 203.4,
         "nse": 0.9858, "peak_error_pct": 2.36, "peak_timing_h": -4.0, "bias_pct": 2.46, "nse_no_routing": 0.726},
        {"reach": "Danube, Passau to Vienna (June 2013)", "upstream": 220751824, "downstream": 220722952,
         "window": ["2013-05-25", "2013-07-04"], "n_path": 102, "k_path_h": 124.5,
         "nse": 0.9658, "peak_error_pct": 5.91, "peak_timing_h": -5.0, "bias_pct": 5.74, "nse_no_routing": -0.3143},
        {"reach": "Mississippi, St. Louis to Memphis (spring 2019)", "upstream": 760684821, "downstream": 760595232,
         "window": ["2019-04-15", "2019-07-14"], "n_path": 153, "k_path_h": 239.4,
         "nse": 0.8935, "peak_error_pct": -0.91, "peak_timing_h": 4.0, "bias_pct": -3.76, "nse_no_routing": -0.183},
    ],
    "summary": ("Checked against GEOGLOWS's own hourly retrospective on four rivers: NSE 0.89 to 0.99, the peak "
                "within 6 % and its timing within 5 h (no routing: NSE -0.31 to 0.73)."),
}

ATTRIBUTION = ("GEOGLOWS v2 routing configuration and retrospective simulation, CC BY 4.0. River network: TDX-Hydro "
               "(NGA), CC BY-SA 4.0, shown and not republished.")
LICENCE = {"routing": "CC BY 4.0 (GEOGLOWS v2)", "network": "CC BY-SA 4.0 (TDX-Hydro)"}
NOTE_LINEAR = ("Muskingum is linear: the wave is routed on its own, so these are the flows it adds to whatever the "
               "river already carries. Rivers that join below add their own water on top.")
NOTE_MODEL = ("Modelled: GEOGLOWS's travel times and weights, one per reach, the same at every flow. A real flood "
              "that leaves the channel travels slower and flattens more; a dam or a lake on the way holds it back.")

# ── the routing parameters, one VPU at a time ───────────────────────────────

_PARAMS: OrderedDict[int, tuple[Any, Any, Any]] = OrderedDict()


def _params(net: Any) -> tuple[Any, Any, Any]:
    """``(k_s, x, ds_index)`` for every reach of the unit, in the network's row order (-1: no reach below)."""
    import numpy as np
    import pandas as pd

    from aquascope import rivers

    if net.vpu in _PARAMS:
        _PARAMS.move_to_end(net.vpu)
        return _PARAMS[net.vpu]

    def column(name: str) -> Any:
        text = rivers._fetch_text(rivers.ROUTING_CONFIG.format(vpu=net.vpu, name=name))
        return pd.read_csv(io.StringIO(text), header=None).iloc[:, 0].to_numpy(dtype=float)

    k = column("k.csv")
    x = column("x.csv")
    if len(k) != len(net.ids) or len(x) != len(net.ids):
        raise RuntimeError(f"GEOGLOWS unit {net.vpu}: k.csv and x.csv do not match rapid_connect.csv row for row")
    ds_index = np.full(len(net.ids), -1, dtype=np.int64)
    pos = np.searchsorted(net._sorted, net.ds)
    pos = np.clip(pos, 0, len(net._sorted) - 1)
    hit = net._sorted[pos] == net.ds
    ds_index[hit] = net._order[pos[hit]]
    out = (k, x, ds_index)
    _PARAMS[net.vpu] = out
    while len(_PARAMS) > 3:
        _PARAMS.popitem(last=False)
    return out


# ── the Muskingum equation ──────────────────────────────────────────────────


def plan(k: Any, x: Any, dt: float = DT_S, max_steps: int = 2000) -> tuple[Any, Any]:
    """Sub-reaches ``n`` and sub-steps ``m`` per reach, the fewest (``n * m``) that keep every coefficient
    non-negative: ``2 x k/n <= dt/m <= 2 (1 - x) k/n``. With ``x`` at 0.5 or over no ratio fits and the reach is
    routed whole in ``max(1, round(k/dt))`` sub-reaches."""
    import numpy as np

    k = np.atleast_1d(np.asarray(k, dtype=float))
    x = np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), k.shape)
    n_out = np.ones(k.shape, dtype=np.int64)
    m_out = np.ones(k.shape, dtype=np.int64)
    cache: dict[tuple[float, float], tuple[int, int]] = {}
    for i, (kk, xx) in enumerate(zip(k, x)):
        key = (float(kk), float(xx))
        if key not in cache:
            cache[key] = _plan_one(float(kk), float(xx), float(dt), max_steps)
        n_out[i], m_out[i] = cache[key]
    return n_out, m_out


def _plan_one(k: float, x: float, dt: float, max_steps: int) -> tuple[int, int]:
    if not (k > 0 and math.isfinite(k)):
        return 1, 1
    lo, hi = 2.0 * x * k / dt, 2.0 * (1.0 - x) * k / dt   # need lo <= n/m <= hi
    if hi <= lo:
        return max(1, round(k / dt)), 1
    best: tuple[int, int] | None = None
    for m in range(1, max_steps + 1):
        n = max(1, math.ceil(lo * m - 1e-12))
        if n <= hi * m + 1e-12:
            if best is None or n * m < best[0] * best[1]:
                best = (n, m)
            if n == 1 or m > 4 * (best[0] * best[1]):
                break
    return best if best is not None else (1, max_steps)


def coefficients(k: float, x: float, dt: float) -> tuple[float, float, float]:
    """``(c1, c2, c3)`` of ``Q[t+1] = c1 I[t+1] + c2 I[t] + c3 Q[t]`` for one step ``dt`` (seconds)."""
    r = dt / k
    d = r + 2.0 * (1.0 - x)
    return (r - 2.0 * x) / d, (r + 2.0 * x) / d, (2.0 * (1.0 - x) - r) / d


def folded(k: float, x: float, dt: float, n: int, m: int) -> tuple[float, float, float, float]:
    """One sub-reach (``k / n``) over ``m`` sub-steps of ``dt / m``, folded into one step of ``dt``:
    ``Q[t+1] = C1 I[t+1] + C2 I[t] + C3 Q[t] + C4 L[t]``, the inflow interpolated linearly across the sub-steps
    and the lateral flow ``L`` (m3/s, steady over the step) added to the inflow at both ends of each sub-step,
    as RAPID and river-route add a catchment's runoff."""
    c1, c2, c3 = coefficients(k / n, x, dt / m)
    out = []
    for q0, i0, i1, lat in ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)):
        q = q0
        for j in range(1, m + 1):
            a, b = (j - 1) / m, j / m
            q = c3 * q + c1 * ((1 - b) * i0 + b * i1 + lat) + c2 * ((1 - a) * i0 + a * i1 + lat)
        out.append(q)
    c3f, c2f, c1f, c4f = out
    return c1f, c2f, c3f, c4f


def _route_reach(inflow: Any, lateral: Any, coef: tuple[float, float, float, float], n: int, q0: float) -> Any:
    """A reach's outflow at the step ends (``len(inflow)`` values) through ``n`` sub-reaches in series, each
    starting at ``q0``; the lateral flow enters the first."""
    import numpy as np
    from scipy.signal import lfilter

    c1, c2, c3, c4 = coef
    flow = np.asarray(inflow, dtype=float)
    for s in range(n):
        forcing = c1 * flow[1:] + c2 * flow[:-1]
        if s == 0 and lateral is not None:
            forcing = forcing + c4 * lateral
        out = np.empty_like(flow)
        out[0] = q0
        out[1:], _ = lfilter([1.0], [1.0, -c3], forcing, zi=[c3 * q0])
        flow = out
    return flow


def route_network(order: Any, ds_index: Any, k: Any, x: Any, n_steps: int, *, dt: float = DT_S,
                  inflow: dict[int, Any] | None = None, lateral: dict[int, Any] | None = None,
                  outflow: dict[int, Any] | None = None, q0: dict[int, float] | None = None,
                  through: dict[int, Any] | None = None) -> dict[int, Any]:
    """Route a set of reaches (row indices ``order``, upstream first) for ``n_steps`` steps of ``dt`` seconds.

    ``inflow[i]`` (``n_steps + 1`` values at the step ends, m3/s) enters reach ``i`` at its top, on top of what
    the reaches in the set above it send; ``lateral[i]`` (``n_steps`` values, one per step) is its own
    catchment's runoff; ``outflow[i]`` replaces the routed outflow of reach ``i`` (a reach whose flow is given,
    not routed). ``q0[i]`` is the starting outflow (0 by default). ``through[i]`` is a function the reach's
    outflow passes through before it goes on (a reservoir at the reach's downstream end, #662): it takes the series
    and returns the series that leaves. Returns ``{i: outflow series}`` for every reach in ``order``. A reach's
    water goes on to ``ds_index[i]`` when that reach is in the set."""
    import numpy as np

    inflow = inflow or {}
    lateral = lateral or {}
    outflow = outflow or {}
    q0 = q0 or {}
    through = through or {}
    order = [int(i) for i in order]
    members = set(order)
    nn, mm = plan(np.asarray(k, dtype=float)[order], np.asarray(x, dtype=float)[order], dt)
    acc: dict[int, Any] = {}
    out: dict[int, Any] = {}
    zeros = np.zeros(n_steps + 1)
    coef_cache: dict[tuple[float, float, int, int], tuple[float, float, float, float]] = {}
    for pos, i in enumerate(order):
        if i in outflow:
            q = np.asarray(outflow[i], dtype=float)
        else:
            top = acc.pop(i, None)
            if i in inflow:
                top = (zeros if top is None else top) + np.asarray(inflow[i], dtype=float)
            if top is None:
                top = zeros
            n, m = int(nn[pos]), int(mm[pos])
            kk, xx = float(k[i]), float(x[i])
            key = (kk, xx, n, m)
            if key not in coef_cache:
                coef_cache[key] = folded(kk, xx, dt, n, m)
            lat = lateral.get(i)
            q = _route_reach(top, None if lat is None else np.asarray(lat, dtype=float), coef_cache[key], n,
                             float(q0.get(i, 0.0)))
        if i in through:
            q = np.asarray(through[i](q), dtype=float)
        out[i] = q
        d = int(ds_index[i])
        if d >= 0 and d in members:
            acc[d] = q if d not in acc else acc[d] + q
    return out


# ── the inflow ──────────────────────────────────────────────────────────────


def hydrograph(peak_cms: float, duration_h: float, n_steps: int, *, shape: str = "triangle",
               dt: float = DT_S) -> Any:
    """The water entering, at ``n_steps + 1`` step ends (m3/s): ``block`` holds ``peak_cms`` for
    ``duration_h`` hours (a release); ``triangle`` rises to it at a third of the duration and falls to zero at
    the end (half the volume of the block)."""
    import numpy as np

    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, not {shape!r}")
    t = np.arange(n_steps + 1) * dt / 3600.0
    d = float(duration_h)
    if shape == "block":
        q = np.where((t > 0) & (t <= d + 1e-9), float(peak_cms), 0.0)
    else:
        rise = d / 3.0
        q = np.where(t <= rise, t / rise, np.where(t <= d, (d - t) / (d - rise), 0.0)) * float(peak_cms)
        q = np.clip(q, 0.0, None)
    return q


def _check_inputs(duration_h: Any, hours: Any, shape: str) -> tuple[float, int | None]:
    d = float(duration_h)
    if not (0 < d <= MAX_HOURS):
        raise ValueError(f"duration_h must be between 0 and {MAX_HOURS} hours")
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, not {shape!r}")
    h = None if hours in (None, "", 0) else int(round(float(hours)))
    if h is not None and not (1 <= h <= MAX_HOURS):
        raise ValueError(f"hours must be between 1 and {MAX_HOURS}")
    return d, h


def _rp_flow(river_id: int, return_period: float) -> tuple[float, str, dict[str, Any]]:
    """The reach's ``return_period``-year flow: :func:`aquascope.floodsim.design_flow`, the design flow every tool
    shares (Log-Pearson III on GEOGLOWS v2's annual maxima, as Flood it)."""
    from aquascope import floodsim

    d = floodsim.design_flow(river_id, return_period)
    q = float(d["flow_cms"])
    return q, f"the {d['return_period']:g}-year flow of this reach, {q:,.0f} m³/s ({d['words']})", d


# ── the answer per reach ────────────────────────────────────────────────────


def _sig(v: float) -> float:
    if not math.isfinite(v) or v == 0:
        return 0.0
    return float(f"{v:.3g}")


def _reach_rows(ids: list[int], series: list[Any], peak_in: float, dt: float, n_steps: int) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    hours = dt / 3600.0
    for rid, q in zip(ids, series):
        q = np.asarray(q, dtype=float)
        j = int(np.argmax(q))
        peak = float(q[j])
        if peak < MIN_SHARE * peak_in:
            rows.append({"river_id": rid, "peak_cms": _sig(peak), "peak_h": None, "front_h": None,
                         "attenuation_pct": None, "reached": False})
            continue
        front = int(np.argmax(q >= 0.5 * peak))
        still_rising = j >= n_steps
        rows.append({
            "river_id": rid, "peak_cms": _sig(peak), "peak_h": None if still_rising else round(j * hours, 1),
            "front_h": round(front * hours, 1),
            "attenuation_pct": round(100.0 * (1.0 - peak / peak_in), 1) if peak_in > 0 else None,
            "reached": True, "still_rising": still_rising,
        })
    return rows


def _frames(ids: list[int], series: list[Any], rows: list[dict[str, Any]], n_steps: int, dt: float,
            peak_ref: float) -> dict[str, Any]:
    """The animation: every ``stride`` steps, each reached reach's flow as a share of ``peak_ref`` (0 to 100,
    whole numbers) so the page can colour it without doing any arithmetic of its own."""
    import numpy as np

    keep = [j for j, r in enumerate(rows) if r.get("reached")]
    if len(keep) > MAX_FRAME_REACHES:
        keep = sorted(sorted(keep, key=lambda j: -rows[j]["peak_cms"])[:MAX_FRAME_REACHES])
    stride = max(1, math.ceil((n_steps + 1) / MAX_FRAMES))
    steps = list(range(0, n_steps + 1, stride))
    if steps[-1] != n_steps:
        steps.append(n_steps)
    if not keep or peak_ref <= 0:
        return {"t_h": [round(s * dt / 3600.0, 2) for s in steps], "ids": [], "share": [], "scale_cms": peak_ref}
    mat = np.vstack([np.asarray(series[j], dtype=float)[steps] for j in keep])
    share = np.clip(np.rint(100.0 * mat / peak_ref), 0, 100).astype(int)
    return {"t_h": [round(s * dt / 3600.0, 2) for s in steps], "ids": [ids[j] for j in keep],
            "share": share.T.tolist(), "scale_cms": _sig(peak_ref)}


def _along_km(trace: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Each row's km along the path from the start to the reach's downstream end, from the traced reach lengths."""
    km = 0.0
    lengths = {int(r["river_id"]): float(r.get("length_km") or 0.0) for r in trace.get("reaches") or []}
    for r in rows:
        km += lengths.get(int(r["river_id"]), 0.0)
        r["along_km"] = round(km, 2)


def places_on_path(result: dict[str, Any], places: list[dict[str, Any]], *, max_km: float = PLACE_KM
                   ) -> list[dict[str, Any]]:
    """Places (``[{"name", "lat", "lon", "kind"}]``, towns read off a map for example) within ``max_km`` of a
    :func:`route` result's path, in the order the water reaches them, each with the wave where it stands: the
    reach it sits on, the added peak, the hour of the peak, the front and the attenuation."""
    from aquascope import rivers

    coords = (result.get("geometry") or {}).get("coordinates") or []
    rows = [r for r in result.get("reaches") or [] if r.get("along_km") is not None]
    if len(coords) < 2 or not rows:
        return []
    index = rivers.PathIndex(coords, max_km)
    out = []
    for p in places or []:
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        loc = index.locate(lat, lon)
        if loc is None:
            continue
        row = next((r for r in rows if loc[1] <= r["along_km"] + 1e-6), rows[-1])
        out.append(_place_row({"kind": str(p.get("kind") or "town"), "name": str(p.get("name") or "")[:80],
                               "lat": lat, "lon": lon, "along_km": round(loc[1], 1),
                               "distance_km": round(loc[0], 2)}, row))
    return _dedupe(out)


def _place_row(p: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {**p, "river_id": row["river_id"], "peak_cms": row["peak_cms"], "peak_h": row["peak_h"],
            "front_h": row["front_h"], "attenuation_pct": row["attenuation_pct"], "reached": row["reached"]}


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In the order the water reaches them; one row per gauge (a station listed by two agencies, or twice by one
    at the same spot, keeps the first) and per town name."""
    seen: set[tuple[Any, ...]] = set()
    out = []
    for p in sorted(rows, key=lambda q: (q.get("along_km") or 0.0)):
        name = str(p.get("name") or "").strip().lower()
        key = (p["kind"], name) if p["kind"] != "gauge" else ("gauge", round(p["lat"], 3), round(p["lon"], 3))
        if key in seen or (p["kind"] == "gauge" and ("gauge-name", name) in seen):
            continue
        seen.add(key)
        if p["kind"] == "gauge":
            seen.add(("gauge-name", name))
        out.append(p)
    return out


def _gauges_on_path(trace: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Archive gauges the trace found, each with the wave at the reach it sits on."""
    out = []
    for g in trace.get("gauges") or []:
        if g.get("along_km") is None:
            continue
        row = next((r for r in rows if r.get("along_km") is not None and g["along_km"] <= r["along_km"] + 1e-6),
                   None)
        if row is None:
            continue
        out.append(_place_row({"kind": "gauge", "name": g.get("name") or g.get("station_id"),
                               "source": g.get("source"), "station_id": g.get("station_id"),
                               "lat": g.get("latitude"), "lon": g.get("longitude"), "along_km": g["along_km"],
                               "distance_km": g.get("distance_km")}, row))
    return out


def _auto_hours(duration_h: float, k_path_s: float) -> int:
    """Long enough for the peak to pass the path's last reach, held to 2 days at least and 30 at most."""
    return int(min(MAX_HOURS, max(48, math.ceil(duration_h + 1.2 * k_path_s / 3600.0 + 12))))


def route(river_id: int | str | None = None, *, lat: float | None = None, lon: float | None = None,
          peak_cms: float | None = None, return_period: float | None = None, duration_h: float = 24.0,
          shape: str = "triangle", hours: int | None = None, max_reaches: int = MAX_PATH,
          along: bool = True, places: list[dict[str, Any]] | None = None, frames: bool = True,
          stations: list[dict[str, Any]] | None = None, inflow_cms: list[float] | None = None,
          inflow_source: str | None = None, through: dict[int, Any] | None = None) -> dict[str, Any]:
    """A flood wave from a reach down its river to the sea: what arrives where, when, and how flat.

    Water enters at the downstream end of ``river_id`` (or the reach snapped at ``lat``/``lon``): a hydrograph
    with ``peak_cms`` (or the reach's ``return_period``-year flow, 2 to 500 years, from
    :func:`aquascope.floodsim.design_flow`) of
    ``shape`` (``triangle`` or ``block``, a release) over ``duration_h`` hours. It is routed reach by reach down
    the path with GEOGLOWS's own k and x for ``hours`` (by default long enough for the peak to reach the last
    reach followed, at most 30 days). ``max_reaches`` caps the path.

    Returns per reach (``reaches``, in the order the water passes them) the peak of the added flow, the hour it
    peaks, the wave's front (the hour the added flow first reaches half its own peak there) and the attenuation
    (how much lower the peak is than the peak that entered). With ``along`` the path is traced
    (:func:`aquascope.rivers.trace_downstream`) and ``places`` lists the Archive gauges on it and any ``places``
    given (``[{"name", "lat", "lon"}]``, towns for example) with the wave where they stand. ``frames`` is the
    animation for the map (each reach's flow as a share of the peak that entered, every few hours).

    ``inflow_cms`` replaces the made-up hydrograph with a real one (#659): the flow leaving the reach, one value an
    hour from the first (m3/s), routed for as many hours as it has (``hours`` at most) with every reach on the path
    starting at its first value, so a river already in flow does not fill from empty. ``inflow_source`` says where
    the series came from. The answer is then the flow that water alone brings down the path.

    ``through`` maps a reach on the path (its ``river_id``) to a function its outflow passes through, a reservoir
    at its downstream end (:func:`aquascope.sketch.reservoir.route_reservoir`, #662).
    """
    import numpy as np

    from aquascope import rivers

    series_in = None
    design: dict[str, Any] | None = None
    if inflow_cms is not None:
        series_in = np.asarray([float(v) for v in inflow_cms], dtype=float)
        if not (2 <= len(series_in) <= MAX_HOURS + 1) or not np.all(np.isfinite(series_in)) or series_in.min() < 0:
            raise ValueError(f"inflow_cms must be 2 to {MAX_HOURS + 1} hourly flows, none negative")
        duration_h, shape = float(len(series_in) - 1), "series"
        d, h = duration_h, (None if hours in (None, "", 0) else min(int(hours), len(series_in) - 1))
    else:
        d, h = _check_inputs(duration_h, hours, shape)
    snap = None
    if river_id is None:
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon to snap to the nearest reach")
        snap = rivers.snap_to_river(lat, lon)
        if not snap["snapped"]:
            return {"ok": False, "error": snap["message"], "snap": snap, "attribution": ATTRIBUTION}
        river_id = snap["river_id"]
    rid = rivers._river_id(river_id)
    if series_in is None and peak_cms is None and return_period is None:
        raise ValueError("give peak_cms (m3/s) or return_period (years)")
    if series_in is not None:
        peak = float(series_in.max())
        if peak <= 0:
            raise ValueError("inflow_cms has no flow in it")
        source = inflow_source or f"a hydrograph of {len(series_in) - 1} h, peaking at {peak:,.0f} m³/s"
    elif peak_cms is not None:
        peak = float(peak_cms)
        if not (0 < peak <= 1e6):
            raise ValueError("peak_cms must be above 0 and at most 1,000,000 m3/s")
        source = f"{peak:,.0f} m³/s set by hand"
    else:
        peak, source, design = _rp_flow(rid, float(return_period))  # type: ignore[arg-type]
    net = rivers._network_for(rid, lat, lon)
    k, x, ds_index = _params(net)
    i0 = net.index(rid)
    assert i0 is not None
    path = [i0]
    seen = {i0}
    while len(path) < int(max_reaches):
        nxt = int(ds_index[path[-1]])
        if nxt < 0 or nxt in seen:
            break
        seen.add(nxt)
        path.append(nxt)
    truncated = len(path) >= int(max_reaches) and int(ds_index[path[-1]]) >= 0
    k_path = float(np.sum(k[path[1:]])) if len(path) > 1 else 0.0
    if series_in is not None:
        n_steps = h if h is not None else len(series_in) - 1
        q_in = series_in[: n_steps + 1]
        start_q = {i: float(q_in[0]) for i in path}
    else:
        n_steps = h if h is not None else _auto_hours(d, k_path)
        q_in = hydrograph(peak, d, n_steps, shape=shape)
        start_q = None
    passes = {net.index(int(r)): fn for r, fn in (through or {}).items() if net.index(int(r)) is not None}
    routed = (route_network(path, ds_index, k, x, n_steps, outflow={i0: q_in}, q0=start_q, through=passes)
              if len(path) > 1 else {i0: passes[i0](q_in) if i0 in passes else q_in})
    ids = [int(net.ids[i]) for i in path]
    series = [routed[i] for i in path]
    rows = _reach_rows(ids, series, peak, DT_S, n_steps)
    rows[0].update({"front_h": rows[0]["front_h"], "attenuation_pct": 0.0, "start": True})
    for r, i in zip(rows, path):
        r["k_h"] = round(float(k[i]) / 3600.0, 2)
    last = max((j for j, r in enumerate(rows) if r["reached"] and r.get("peak_h") is not None), default=0)
    volume = float(np.sum((q_in[1:] + q_in[:-1]) / 2.0) * DT_S)
    res: dict[str, Any] = {
        "ok": True, "river_id": rid, "vpu": net.vpu, "snap": snap,
        "inflow": {"peak_cms": round(peak, 2), "duration_h": d, "shape": shape, "source": source,
                   "volume_m3": _sig(volume), "return_period": return_period, "design": design},
        "hours": n_steps, "dt_s": DT_S, "n_reaches": len(path), "truncated": truncated,
        "reaches": rows,
        "furthest": {"river_id": ids[last], "index": last, **{kk: rows[last][kk] for kk in
                                                             ("peak_cms", "peak_h", "front_h", "attenuation_pct")}},
        "method": METHOD, "check": {"summary": CHECK["summary"], "tolerance": CHECK["tolerance"]},
        "notes": [NOTE_LINEAR, NOTE_MODEL], "attribution": ATTRIBUTION, "licence": dict(LICENCE), "modelled": True,
    }
    if design is not None:
        res["licence"]["flows"] = f"{design['licence']} (GEOGLOWS v2 annual maxima)"
        res["attribution"] = f"{ATTRIBUTION} T-year flows: GEOGLOWS v2 annual maxima, {design['licence']}."
    if truncated:
        res["notes"].append(f"Followed {len(path):,} reaches; the river goes on.")
    if frames:
        res["frames"] = _frames(ids, series, rows, n_steps, DT_S, peak)
    res["places"] = []
    res["geometry"] = None
    if along:
        try:
            trace = rivers.trace_downstream(rid, lat=lat, lon=lon, max_reaches=last + 2, path_context=False,
                                            stations=stations, max_points=3000)
            res["geometry"] = trace.get("geometry")
            _along_km(trace, rows[: last + 2])
            res["places"] = _dedupe(_gauges_on_path(trace, rows) + places_on_path(res, places or []))
            res["furthest"]["along_km"] = rows[last].get("along_km")
        except Exception as exc:  # noqa: BLE001 - the wave stands without the places
            res["notes"].append(f"The gauges and towns on the way are unavailable: {exc}")
    res["message"] = _message(res)
    return res


def _hours_words(h: float | None) -> str:
    if h is None:
        return "after the hours routed"
    if h < 48:
        return f"{h:.0f} h"
    return f"{h / 24:.1f} days"


def _message(res: dict[str, Any]) -> str:
    inflow = res["inflow"]
    f = res["furthest"]
    reached = sum(1 for r in res["reaches"] if r["reached"])
    what = ("A hydrograph peaking at" if inflow["shape"] == "series" else f"A {inflow['shape']} of")
    head = (f"{what} {inflow['peak_cms']:,.0f} m³/s over {inflow['duration_h']:g} h from reach "
            f"{res['river_id']}, routed {res['hours']} h down {reached:,} reaches.")
    if f["index"] == 0:
        return head + " It does not get past the first reach in the hours routed."
    tail = (f" Its peak reaches reach {f['river_id']} after {_hours_words(f['peak_h'])} at {f['peak_cms']:,.0f} m³/s, "
            f"{f['attenuation_pct']:.0f} % lower than where it started.")
    towns = [p for p in res.get("places") or [] if p.get("reached") and p.get("peak_h") is not None]
    if towns:
        p = towns[-1]
        tail += f" At {p['name']}: peak after {_hours_words(p['peak_h'])}, {p['peak_cms']:,.0f} m³/s."
    return head + tail


# ── a drawn storm ───────────────────────────────────────────────────────────


def _inside(lon: Any, lat: Any, ring: list[list[float]]) -> Any:
    """Even-odd point in polygon over arrays of points."""
    import numpy as np

    inside = np.zeros(len(lon), dtype=bool)
    n = len(ring)
    for a in range(n):
        x1, y1 = ring[a]
        x2, y2 = ring[(a + 1) % n]
        cross = ((y1 > lat) != (y2 > lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (lon < xs)
    return inside


def _hydro(row: dict[str, Any], q: Any) -> dict[str, Any]:
    """A reach's hourly flow for a card's chart: its id, peak and the series (m3/s, three significant figures)."""
    return {"river_id": row["river_id"], "peak_cms": row["peak_cms"], "peak_h": row.get("peak_h"),
            "dt_h": DT_S / 3600.0, "cms": [_sig(float(v)) for v in q]}


def route_storm(polygon: list[list[float]], *, rain_mm: float, duration_h: float = 6.0,
                runoff_coefficient: float = 0.3, hours: int | None = None, vpu: int | None = None,
                lat: float | None = None, lon: float | None = None, frames: bool = True,
                through: dict[int, Any] | None = None) -> dict[str, Any]:
    """A drawn storm turned into runoff and routed down every river below it.

    ``polygon`` is a ring of ``[lon, lat]``. Every reach whose unit catchment's centre (GEOGLOWS's ERA5 weight
    table) falls inside gets ``runoff_coefficient * rain_mm`` of runoff over its own unit catchment, spread
    evenly over ``duration_h`` hours; the runoff coefficient is a plain share of the rain that runs off, said as
    such, with no soil or wetness behind it. All the reaches below are routed for ``hours``. The unit is the one
    the polygon's centre falls in (``vpu`` overrides it; ``lat``/``lon`` give the point to find it by).
    ``through`` maps a reach (its ``river_id``) to a function its outflow passes through (a reservoir, #662).
    Returns the reaches with the biggest peaks, the outlet reaches' waves and the animation frames, the hourly
    hydrographs of the reach with the biggest peak and of the main outlet (``hydrographs``), the box of the reaches
    the wave reaches (``bbox``) and, in ``storm``, the water that has left by the outlets by the end (``out_m3``,
    ``out_pct`` of the runoff: the mass balance).
    """
    import numpy as np

    from aquascope import rivers

    ring = [[float(p[0]), float(p[1])] for p in polygon or []]
    if len(ring) < 3:
        raise ValueError("a storm needs a polygon of at least three [lon, lat] points")
    rain = float(rain_mm)
    c = float(runoff_coefficient)
    if not (0 < rain <= 2000):
        raise ValueError("rain_mm must be above 0 and at most 2000 mm")
    if not (0 < c <= 1):
        raise ValueError("runoff_coefficient must be above 0 and at most 1")
    d, h = _check_inputs(duration_h, hours, "block")
    clon = sum(p[0] for p in ring) / len(ring) if lon is None else float(lon)
    clat = sum(p[1] for p in ring) / len(ring) if lat is None else float(lat)
    net = None
    if vpu is not None:
        net = rivers._network(int(vpu))
    else:
        for v, box in sorted(rivers._VPU_BBOX.items(),
                             key=lambda kv: math.hypot(max(kv[1][0] - clon, 0, clon - kv[1][2]),
                                                       max(kv[1][1] - clat, 0, clat - kv[1][3]))):
            w, s, e, n_ = box
            if not (w <= clon <= e and s <= clat <= n_):
                break
            cand = rivers._network(v)
            hit = _inside(cand.lon, cand.lat, ring)
            if hit.any():
                net = cand
                break
    if net is None:
        return {"ok": False, "error": "No GEOGLOWS reach has its catchment centre inside the storm.",
                "attribution": ATTRIBUTION}
    k, x, ds_index = _params(net)
    under = np.flatnonzero(_inside(net.lon, net.lat, ring) & np.isfinite(net.area_m2) & (net.area_m2 > 0))
    if not len(under):
        return {"ok": False, "error": "No GEOGLOWS reach has its catchment centre inside the storm.",
                "attribution": ATTRIBUTION}
    if len(under) > MAX_STORM_REACHES:
        return {"ok": False, "error": (f"The storm covers {len(under):,} reaches; draw it smaller (at most "
                                       f"{MAX_STORM_REACHES:,})."), "attribution": ATTRIBUTION}
    members: set[int] = set()
    for i in under:
        j = int(i)
        while j >= 0 and j not in members and len(members) < MAX_STORM_NETWORK:
            members.add(j)
            j = int(ds_index[j])
    order = sorted(members)   # the unit's row order is upstream first
    if any(int(ds_index[i]) >= 0 and int(ds_index[i]) < i for i in order):
        raise RuntimeError(f"GEOGLOWS unit {net.vpu}: rapid_connect.csv is not in upstream-first order")
    k_max = 0.0
    depth = {}
    for i in order:   # the longest travel time from a storm reach to the outlet, for the horizon
        depth[i] = 0.0
    for i in reversed(order):
        d_i = int(ds_index[i])
        depth[i] = float(k[i]) + (depth[d_i] if d_i >= 0 and d_i in depth else 0.0)
        k_max = max(k_max, depth[i])
    n_steps = h if h is not None else _auto_hours(d, k_max)
    t_end = np.arange(1, n_steps + 1) * DT_S / 3600.0
    on = ((t_end - DT_S / 3600.0) < d - 1e-9).astype(float)   # the steps that start inside the storm
    lateral = {}
    volume = 0.0
    for i in under:
        q = c * rain / 1000.0 * float(net.area_m2[i]) / (d * 3600.0)
        lateral[int(i)] = q * on
        volume += c * rain / 1000.0 * float(net.area_m2[i])
    passes = {net.index(int(r)): fn for r, fn in (through or {}).items() if net.index(int(r)) is not None}
    routed = route_network(order, ds_index, k, x, n_steps, lateral=lateral, through=passes)
    peaks = {i: float(np.max(routed[i])) for i in order}
    top = max(peaks.values()) if peaks else 0.0
    ids = [int(net.ids[i]) for i in order]
    rows = _reach_rows(ids, [routed[i] for i in order], top, DT_S, n_steps)
    for r, i in zip(rows, order):
        r["attenuation_pct"] = None
        r["in_storm"] = bool(i in lateral)
    outlets = [j for j, i in enumerate(order) if int(ds_index[i]) < 0 or int(ds_index[i]) not in members]
    best = max(range(len(rows)), key=lambda j: rows[j]["peak_cms"])
    # The water that has left by the end, for the mass balance the card states (#670): what the outlets passed.
    out_m3 = float(sum(np.sum((routed[order[j]][1:] + routed[order[j]][:-1]) / 2.0) * DT_S for j in outlets))
    main = max(outlets, key=lambda j: rows[j]["peak_cms"]) if outlets else best
    hydro = [{"role": "biggest", **_hydro(rows[best], routed[order[best]])}]
    if main != best:
        hydro.append({"role": "outlet", **_hydro(rows[main], routed[order[main]])})
    lit = [order[j] for j, r in enumerate(rows) if r.get("reached")] or list(under)
    res: dict[str, Any] = {
        "ok": True, "vpu": net.vpu,
        "storm": {"rain_mm": rain, "duration_h": d, "runoff_coefficient": c, "n_reaches": int(len(under)),
                  "area_km2": round(float(np.sum(net.area_m2[under])) / 1e6, 1), "runoff_m3": _sig(volume),
                  "out_m3": _sig(out_m3), "out_pct": round(100.0 * out_m3 / volume, 1) if volume > 0 else None,
                  "rule": (f"{c:g} of the rain runs off: {c * rain:g} mm over each unit catchment under the storm, "
                           f"spread evenly over {d:g} h. A plain coefficient, no soil, wetness or snow.")},
        "hours": n_steps, "dt_s": DT_S, "n_reaches": len(order),
        "biggest": rows[best], "outlets": [rows[j] for j in outlets], "hydrographs": hydro,
        "bbox": [round(float(np.min(net.lon[lit])), 4), round(float(np.min(net.lat[lit])), 4),
                 round(float(np.max(net.lon[lit])), 4), round(float(np.max(net.lat[lit])), 4)],
        "polygon": [[round(p[0], 5), round(p[1], 5)] for p in ring],
        "reaches": sorted(rows, key=lambda r: -r["peak_cms"])[:500],
        "method": METHOD, "check": {"summary": CHECK["summary"], "tolerance": CHECK["tolerance"]},
        "notes": [NOTE_MODEL, "Only the storm's runoff is routed: these are the flows it adds."],
        "attribution": ATTRIBUTION, "licence": LICENCE, "modelled": True,
    }
    if frames:
        res["frames"] = _frames(ids, [routed[i] for i in order], rows, n_steps, DT_S, top)
    b = res["biggest"]
    res["message"] = (f"{c * rain:g} mm of runoff on {len(under):,} reaches ({res['storm']['area_km2']:,.0f} km²) "
                      f"over {d:g} h, routed {n_steps} h down {len(order):,} reaches. The biggest added flow is "
                      f"{b['peak_cms']:,.0f} m³/s at reach {b['river_id']}"
                      + (f", after {_hours_words(b['peak_h'])}." if b.get("peak_h") is not None else "."))
    return res


# ── the check against GEOGLOWS's own retrospective ──────────────────────────

RETRO_HOURLY = "https://geoglows-v2.s3.us-west-2.amazonaws.com/retrospective/hourly.zarr"


class _Retro:
    """GEOGLOWS's hourly retrospective (``retrospective/hourly.zarr``, Q by time and river_id), a reach column
    chunk at a time. Each chunk holds 20 reaches and nearly the whole record (about 16 MB), so the check reads
    a few dozen; the decoder is numcodecs' Blosc (``pip install numcodecs``)."""

    def __init__(self, base: str = RETRO_HOURLY, fetch: Any = None):
        import numpy as np

        from aquascope.archive import warnings as fw

        self.base = base
        self.fetch = fetch or fw.Fetcher(timeout=300.0)
        self.meta = fw._meta(self.fetch, base)
        self.q = self.meta["Q/.zarray"]
        ids = self._read_ids()
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]
        units = (self.meta.get("time/.zattrs") or {}).get("units", "")
        if not units.startswith("seconds since "):
            raise RuntimeError(f"unexpected time units {units!r}")
        self.epoch = np.datetime64(units.replace("seconds since ", "").replace(" ", "T")[:19])
        self.seconds = fw._read_1d(self.fetch, base, "time", self.meta).astype(np.int64)

    def _read_ids(self) -> Any:
        """The 6.8 million river ids, in about 180 small chunks read 16 at a time."""
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np

        from aquascope.archive import warnings as fw

        arr = self.meta["river_id/.zarray"]
        n, c = arr["shape"][0], arr["chunks"][0]

        def one(i: int) -> Any:
            data = self.fetch(f"{self.base}/river_id/{i}")
            if data is None:
                raise RuntimeError(f"{self.base}/river_id/{i} is missing")
            return fw.decode_chunk(data, arr).reshape(-1)

        with ThreadPoolExecutor(16) as pool:
            parts = list(pool.map(one, range(-(-n // c))))
        return np.concatenate(parts)[:n].astype(np.int64)

    def times(self) -> Any:
        import numpy as np

        return self.epoch + self.seconds.astype("timedelta64[s]").astype(np.dtype("timedelta64[s]"))

    def columns(self, rids: list[int], t0: int, t1: int) -> dict[int, Any]:
        """Q of each reach in ``rids`` for time indices ``t0:t1``; the chunks are read six at a time and only
        the window of the reaches asked is kept."""
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np

        from aquascope.archive import warnings as fw

        tc, cc = self.q["chunks"]
        want: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for rid in dict.fromkeys(int(r) for r in rids):
            k = int(np.searchsorted(self.sorted, rid))
            if k >= len(self.sorted) or int(self.sorted[k]) != rid:
                raise KeyError(f"reach {rid} is not in the retrospective")
            col = int(self.order[k])
            for ti in range(t0 // tc, (t1 - 1) // tc + 1):
                want.setdefault((ti, col // cc), []).append((rid, col % cc))
        parts: dict[int, list[tuple[int, Any]]] = {}

        def one(key: tuple[int, int]) -> list[tuple[int, int, Any]]:
            ti, cj = key
            data = self.fetch(f"{self.base}/Q/{ti}.{cj}")
            if data is None:
                raise RuntimeError(f"chunk Q/{ti}.{cj} is missing")
            block = fw.decode_chunk(data, self.q)
            a = max(t0, ti * tc) - ti * tc
            b = min(t1, (ti + 1) * tc) - ti * tc
            return [(rid, ti, np.array(block[a:b, c], dtype=float)) for rid, c in want[key]]

        with ThreadPoolExecutor(6) as pool:
            for got in pool.map(one, list(want)):
                for rid, ti, v in got:
                    parts.setdefault(rid, []).append((ti, v))
        return {rid: np.concatenate([v for _, v in sorted(p, key=lambda z: z[0])]) for rid, p in parts.items()}


def _nse(sim: Any, obs: Any) -> float:
    import numpy as np

    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return float(1.0 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def retro_check(upstream: int, downstream: int, start: str, end: str, *, spin_up_h: int = 240,
                retro: Any = None, lateral: str = "specific", min_share: float = 0.01) -> dict[str, Any]:
    """Route GEOGLOWS's hourly retrospective from ``upstream`` to ``downstream`` and compare with its own
    retrospective there, for the hours from ``start`` to ``end`` (ISO dates).

    The flow leaving ``upstream`` and every tributary draining at least ``min_share`` of the area at
    ``downstream`` are read from the retrospective. The smaller tributaries and the path's own unit catchments
    get ``lateral`` runoff: ``specific`` (their area times the specific runoff of the upstream reach and the
    tributaries read, that hour: an estimate that does not look at the answer) or ``none``. The reaches start
    from the flows above them and the first ``spin_up_h`` hours are not scored. Returns the Nash-Sutcliffe
    efficiency (hourly), the bias, the error in the largest peak and in its timing, and the same for no routing
    at all (the same inflows added up with no travel time), the baseline routing has to beat."""
    import numpy as np

    from aquascope import rivers

    up, down = rivers._river_id(upstream), rivers._river_id(downstream)
    net = rivers._network_for(up)
    k, x, ds_index = _params(net)
    iu, idn = net.index(up), net.index(down)
    if iu is None or idn is None:
        raise ValueError("both reaches must be in the same GEOGLOWS unit")
    if lateral not in ("specific", "none"):
        raise ValueError("lateral is 'specific' or 'none'")
    path = []
    j = int(ds_index[iu])
    while j >= 0 and j != idn:
        path.append(j)
        j = int(ds_index[j])
    if j != idn:
        raise ValueError(f"reach {down} is not downstream of reach {up}")
    path.append(idn)
    on_path = set(path) | {iu}

    def drainage(i: int) -> float:
        order, _parent, _starts = rivers._upstream_tree(net, i)
        return float(np.nansum(net.area_m2[order]))

    a_down = drainage(idn)
    big: dict[int, list[int]] = {}
    small_area: dict[int, float] = {}
    area: dict[int, float] = {iu: drainage(iu)}
    for i in path:
        big[i] = []
        own = float(net.area_m2[i]) if math.isfinite(float(net.area_m2[i])) else 0.0
        small_area[i] = own
        for c in net.children(i):
            if c in on_path:
                continue
            area[c] = drainage(c)
            if area[c] >= min_share * a_down:
                big[i].append(c)
            else:
                small_area[i] += area[c]
    rt = retro or _Retro()
    times = rt.times()
    t0 = int(np.searchsorted(times, np.datetime64(start)))
    t1 = int(np.searchsorted(times, np.datetime64(end)))
    ts = max(0, t0 - int(spin_up_h))
    n = t1 - ts
    if n <= spin_up_h + 24:
        raise ValueError("the window is too short")
    step = float((times[ts + 1] - times[ts]) / np.timedelta64(1, "s"))
    if abs(step - DT_S) > 1:
        raise RuntimeError(f"the retrospective's step is {step} s, not {DT_S}")
    trib_idx = [c for i in path for c in big[i]]
    cols = rt.columns([int(net.ids[iu]), down, *[int(net.ids[c]) for c in trib_idx]], ts, t1)
    q_up = cols[int(net.ids[iu])]
    obs = cols[down]
    trib_q = {c: cols[int(net.ids[c])] for c in trib_idx}
    read_q = q_up + sum(trib_q.values()) if trib_q else q_up.copy()
    read_a = area[iu] + sum(area[c] for c in trib_q)
    spec = read_q / read_a   # m3/s per m2 at the step ends
    lat: dict[int, Any] = {}
    if lateral == "specific":
        for i in path:
            lat[i] = small_area[i] * (spec[1:] + spec[:-1]) / 2.0
    # each reach starts from the flow above it at the first hour, as if steady
    q0: dict[int, float] = {}
    run = float(q_up[0])
    for i in path:
        run += sum(float(trib_q[c][0]) for c in big[i]) + (small_area[i] * float(spec[0]) if lateral == "specific"
                                                             else 0.0)
        q0[i] = run
    order = sorted([iu, *trib_q, *path])
    routed = route_network(order, ds_index, k, x, n - 1, outflow={iu: q_up, **trib_q}, lateral=lat, q0=q0)
    sim = routed[idn][spin_up_h:]
    ref = obs[spin_up_h:]
    local = sum(small_area.values())
    flat = read_q[spin_up_h:] + (spec[spin_up_h:] * local if lateral == "specific" else 0.0)
    jo, js, jf = int(np.argmax(ref)), int(np.argmax(sim)), int(np.argmax(flat))
    hours = DT_S / 3600.0

    def metrics(s: Any, jj: int) -> dict[str, Any]:
        return {"nse": round(_nse(s, ref), 4),
                "bias_pct": round(100.0 * (float(np.sum(s)) / float(np.sum(ref)) - 1), 2),
                "peak_error_pct": round(100.0 * (float(s[jj]) / float(ref[jo]) - 1.0), 2),
                "peak_timing_h": round((jj - jo) * hours, 1)}

    return {
        "upstream": up, "downstream": down, "vpu": net.vpu, "start": str(times[t0])[:19],
        "end": str(times[t1 - 1])[:19], "hours_scored": int(len(ref)), "n_path": len(path),
        "n_tributaries_read": len(trib_q), "n_tributaries_small": sum(len([c for c in net.children(i)
                                                                           if c not in on_path]) for i in path)
        - len(trib_q),
        "k_path_h": round(float(np.sum(k[path])) / 3600.0, 1),
        "area_km2": round(a_down / 1e6, 0), "lateral_share_pct": round(100.0 * local / a_down, 2),
        "lateral": lateral, "min_share": min_share,
        "retro_peak_cms": round(float(ref[jo]), 1), "retro_peak_time": str(times[ts + spin_up_h + jo])[:16],
        "routed": metrics(sim, js), "no_routing": metrics(flat, jf),
    }
