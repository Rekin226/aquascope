"""Replay an event (#659): a past flood assembled from its data and told as a story on the map.

Give a flood (a Floods past news event, or a river reach or a point and a date window) and :func:`event_record`
assembles what the open data says about it, piece by piece. Each piece carries its source, its licence and its
coverage, and says plainly when it is not available:

* **The rain** (:func:`rain`): NASA's IMERG (V07, Late Run, daily) averaged over the upstream basin, through
  NASA/USAID SERVIR's ClimateSERV API: the basin total and the wettest day.
* **The gauges** (:func:`gauges`): the Archive's discharge gauges on the river near the event, their daily record in
  the window, against GEOGLOWS v2's retrospective simulation at the same reach: the observed and modelled peak, the
  timing and the error.
* **The forecast** (:func:`forecasts`): what GEOGLOWS's ensemble forecast, as issued 1, 3 and 7 days before the
  peak, said for the peak day, from the forecast archive on S3 (runs from 1 July 2024).
* **The wave** (:func:`wave`): the observed (or simulated) flow at the gauge routed down the river with Route it
  (:func:`aquascope.routing.route`): when the peak reaches each gauge and town downstream, and how big it is.
* **The flood** (:func:`flood`): a screening depth from Flood it (:mod:`aquascope.floodsim`) at the peak flow, and
  what is under it (buildings and roads from Overture Maps, people from GHS-POP).

:func:`story` turns the record into a **story**: an ordered list of stops, each with a camera (centre, zoom, pitch,
bearing), a date on the time bar, the layers that are on and a short note. Every note is a template filled by code
from the record; nothing here writes a number that did not come out of the data. :func:`report` writes the same as a
short Markdown report with the numbers, the sources and the licences. The story format is small and versioned
(:data:`STORY_SCHEMA`, :data:`STORY_VERSION`) so that other parts of AquaScope (Studio studies) can emit it too.

:func:`candidates` lists the floods Floods past has near a place in a month, so a model can pick one and the code
build the rest (Ask the map's ``replay`` action).

Experimental. Network reads only (no keys): ClimateSERV, the GEOGLOWS API and its S3 buckets, the Archive.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CLIMATESERV", "FORECAST_FIRST", "LEADS", "SCHEMA", "STORY_SCHEMA", "STORY_VERSION", "candidates",
    "event_record", "event_window", "flood", "forecasts", "gauges", "rain", "rain_poll", "replay", "report", "story",
    "wave",
]

IS_EMSCRIPTEN = sys.platform == "emscripten"

#: The story format: ``{"schema": STORY_SCHEMA, "version": STORY_VERSION, ...}`` (see :data:`SCHEMA`).
STORY_SCHEMA = "aquascope.story"
STORY_VERSION = 1

CLIMATESERV = "https://climateserv.servirglobal.net/api"
#: ClimateSERV's dataset number for NASA IMERG V07 Late Run, 1 day (its API page, read 2026-10-27).
IMERG_DATATYPE = 226
IMERG_FIRST = "2000-06-01"
#: GEOGLOWS v2's forecast archive: one Zarr per daily 00 UTC run, from this date (bucket listing, 2026-10-27).
FORECAST_BUCKET = "https://geoglows-v2-forecasts.s3.us-west-2.amazonaws.com"
FORECAST_FIRST = "2024-07-01"
#: The forecasts read: issued this many days before the peak.
LEADS = (1, 3, 7)
#: The window around a news event: this many days before its start and after its end, at most MAX_DAYS in all.
PRE_DAYS = 8
POST_DAYS = 6
MAX_DAYS = 45
#: The wave is routed from this many days before the peak to this many after (hourly, within Route it's 30 days).
WAVE_DAYS = (6, 14)
#: Gauges: catalogue stations within this distance of the event's reach, on its river (up or down) within
#: STEM_REACHES reaches, at most MAX_GAUGES of them.
GAUGE_KM = 30.0
STEM_REACHES = 80
MAX_GAUGES = 3
#: The wave's note and camera keep to the places this far down the river (km).
WAVE_NOTE_KM = 150.0
#: How far from a news event's centre its river is looked for.
EVENT_RADIUS_M = 12000.0
#: The main river of an event: the river with the largest upstream area within this distance of its centre (#664).
MAIN_RIVER_RADIUS_M = 25000.0
#: The upstream basin goes to ClimateSERV as cells of this size (degrees), coarser when it would be too many.
RAIN_CELL_DEG = (0.1, 0.25, 0.5, 1.0)
MAX_RAIN_PARTS = 120

LICENCES = {
    "rain": ("NASA GPM IMERG V07 Late Run, daily: NASA Earth science data, free and open with no restrictions on "
             "use; averaged over the basin by NASA/USAID SERVIR's ClimateSERV (its API page states no terms, read "
             "2026-10-27)"),
    "model": "GEOGLOWS v2 retrospective simulation, CC BY 4.0",
    "forecast": "GEOGLOWS v2 forecast archive (ECMWF ensemble), CC BY 4.0",
    "routing": "GEOGLOWS v2 routing configuration, CC BY 4.0; river network TDX-Hydro (NGA), CC BY-SA 4.0",
    "flood": "Flood it: Terrarium terrain (AWS Open Data), GEOGLOWS v2 rivers and flows (see aquascope.floodsim)",
    "exposure": "Overture Maps buildings and roads, ODbL 1.0; GHS-POP R2023A, CC BY 4.0",
    "floods_past": "Groundsource news events (Google), CC BY 4.0; Microsoft AI for Good Sentinel-1 floods, MIT",
}

#: The story's fields, as the docs give them.
SCHEMA: dict[str, Any] = {
    "schema": f"'{STORY_SCHEMA}'",
    "version": f"{STORY_VERSION} (an integer; a reader refuses a version it does not know)",
    "title": "a short title",
    "event": "{river_id, lat, lon, start, end, peak_date, name}: what the story is about",
    "stops": ("ordered list; each {id, kind, title, camera: {center: [lon, lat], zoom, pitch, bearing}, date "
              "(YYYY-MM-DD on the time bar, or null), layers: [layer ids], note, and optionally template + needs "
              "(a note the page fills from its own run, for example Flood it on the GPU), focus: {lon, lat}, "
              "line: [[lon, lat], ...], run: {...}}"),
    "sources": "list of {piece, source, licence, coverage}",
    "made_by": "what wrote it, for example 'aquascope.replay'",
}

FLOOD_LABEL = "a screening flood (no bathymetry, terrain errors of metres): not a design flood or a flood map"

# ── HTTP ────────────────────────────────────────────────────────────────────


def _get(url: str, *, timeout: float = 120.0) -> bytes:
    """The body of ``url`` (CPython: httpx; the browser: urllib through pyodide-http)."""
    if IS_EMSCRIPTEN:  # pragma: no cover - only under Pyodide
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - https data hosts
            return bytes(resp.read())
    import httpx

    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return bytes(r.content)


def _get_or_none(url: str) -> bytes | None:
    try:
        return _get(url)
    except Exception as exc:  # noqa: BLE001 - a missing chunk or run is an answer
        logger.info("GET %s failed: %s", url, exc)
        return None


def _json(url: str) -> Any:
    return json.loads(_get(url).decode("utf-8"))


#: Tests replace these.
FETCH: dict[str, Callable[..., Any]] = {"bytes": _get_or_none, "json": _json}

# ── small helpers ───────────────────────────────────────────────────────────


def _date(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _day(d: Any) -> str:
    """26 Nov 2024."""
    d = _date(d)
    return f"{d.day} {d.strftime('%b')} {d.year}"


def _q(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    v = float(v)
    return f"{v:,.0f} m³/s" if v >= 10 else f"{v:,.1f} m³/s"


def _num(v: Any, nd: int = 2) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _missing(piece: str, why: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "piece": piece, "note": why, "licence": LICENCES.get(piece, ""), **extra}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def event_window(start: Any, end: Any = None, *, pre: int = PRE_DAYS, post: int = POST_DAYS) -> tuple[date, date]:
    """The days a replay reads: ``pre`` days before the event's start to ``post`` after its end, at most
    :data:`MAX_DAYS` in all (the end is pulled in), never after yesterday."""
    s = _date(start)
    e = _date(end) if end else s
    if e < s:
        s, e = e, s
    a, b = s - timedelta(days=pre), e + timedelta(days=post)
    if (b - a).days + 1 > MAX_DAYS:
        b = a + timedelta(days=MAX_DAYS - 1)
    b = min(b, date.today() - timedelta(days=1))
    if b < a:
        raise ValueError("the event's window ends before it starts")
    return a, b


# ── the reach ───────────────────────────────────────────────────────────────


def _accumulated(net: Any, i0: int) -> dict[int, float]:
    """The upstream area (m2) of every reach above reach index ``i0``, ``i0`` included, by index."""
    import numpy as np

    from aquascope import rivers

    order, parent, starts = rivers._upstream_tree(net, i0, cap=400_000)
    acc = net.area_m2[order].astype(float)
    acc[~np.isfinite(acc)] = 0.0
    bounds = [*starts, len(order)]
    for k in range(len(starts) - 1, 0, -1):
        sl = slice(bounds[k], bounds[k + 1])
        np.add.at(acc, parent[sl], acc[sl])
    return {int(order[j]): float(acc[j]) for j in range(len(order))}


def main_river(lat: float, lon: float, seed_river_id: int, *, radius_m: float = MAIN_RIVER_RADIUS_M,
               near: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The main river near a point: of the reaches whose catchment centre is within ``radius_m``, the river with the
    largest upstream area (where it leaves the circle), and on its main stem (the bigger branch at each confluence)
    the reach nearest the point: by its line when it is among ``near`` (:func:`aquascope.rivers.reaches_near`'s
    answer, distances to the drawn lines), else by its catchment centre. ``seed_river_id`` is any reach near the
    point (it picks the network read). So a flood reported between a tributary and the main river is the main
    river's, not the nearer tributary's. Returns ``{"river_id", "lat", "lon", "distance_m", "upstream_area_km2"}``,
    or None when nothing is near."""
    import numpy as np

    from aquascope import rivers

    net = rivers._network_for(int(seed_river_id), lat, lon)
    kx = 111.32 * max(0.05, math.cos(math.radians(lat)))
    d_km = np.hypot((net.lon - lon) * kx, (net.lat - lat) * 110.574)
    inside = d_km <= radius_m / 1000.0
    idx = np.nonzero(inside)[0]
    if not idx.size:
        return None
    # The largest upstream area in the circle is where a river leaves it: a reach whose next one is outside.
    exits = []
    for i in idx:
        j = net.index(int(net.ds[i]))
        if j is None or not inside[j]:
            exits.append(int(i))
    best, best_area, areas = None, -1.0, {}
    for i in exits:
        acc = _accumulated(net, i)
        if acc[i] > best_area:
            best, best_area, areas = i, acc[i], acc
    if best is None:
        return None
    nearest, i, stem = best, best, [best]
    while True:
        kids = [k for k in net.children(i) if inside[k]]
        if not kids:
            break
        i = max(kids, key=lambda k: areas.get(int(k), 0.0))
        stem.append(i)
        if d_km[i] < d_km[nearest]:
            nearest = i
    on_stem = {int(net.ids[k]): k for k in stem}
    lines = sorted((r for r in near or [] if int(r["river_id"]) in on_stem), key=lambda r: float(r["distance_m"]))
    if lines:
        r = lines[0]
        k = on_stem[int(r["river_id"])]
        return {"river_id": int(r["river_id"]), "lat": r["lat"], "lon": r["lon"], "distance_m": float(r["distance_m"]),
                "upstream_area_km2": round(areas.get(int(k), 0.0) / 1e6, 1)}
    return {"river_id": int(net.ids[nearest]), "lat": _num(net.lat[nearest], 5), "lon": _num(net.lon[nearest], 5),
            "distance_m": round(float(d_km[nearest]) * 1000.0, 1),
            "upstream_area_km2": round(areas.get(int(nearest), 0.0) / 1e6, 1)}


def _reach_for(river_id: Any, lat: float | None, lon: float | None) -> dict[str, Any]:
    """The event's reach: ``river_id`` as given, or the main river near the point (:func:`main_river`: the largest
    upstream area within :data:`MAIN_RIVER_RADIUS_M`, its reach nearest the point); the main channel (highest
    Strahler order, the nearer on a tie) within :data:`EVENT_RADIUS_M` when the network cannot be read."""
    from aquascope import rivers

    if river_id is not None:
        rid = rivers._river_id(river_id)
        net = rivers._network_for(rid, lat, lon)
        i = net.index(rid)
        return {"river_id": rid, "lat": _num(net.lat[i], 5) if lat is None else float(lat),
                "lon": _num(net.lon[i], 5) if lon is None else float(lon), "how": "given"}
    if lat is None or lon is None:
        raise ValueError("give a river_id, or lat and lon")
    near = rivers.reaches_near(float(lat), float(lon), max_distance_m=EVENT_RADIUS_M, limit=60)
    main = rivers.main_channel(near)
    if main is None:
        raise LookupError(f"no river within {EVENT_RADIUS_M / 1000:g} km of {lat:.3f}, {lon:.3f}")
    try:
        big = main_river(float(lat), float(lon), int(main["river_id"]), near=near)
    except Exception as exc:  # noqa: BLE001 - the Strahler pick stands when the network cannot be read
        logger.info("main river near %s, %s unavailable: %s", lat, lon, exc)
        big = None
    if big is not None:
        return {"river_id": big["river_id"], "lat": big["lat"], "lon": big["lon"],
                "how": (f"the river with the largest upstream area within {MAIN_RIVER_RADIUS_M / 1000:g} km of the "
                        f"event ({big['upstream_area_km2']:,.0f} km² there), at its reach nearest the event's "
                        f"centre, {big['distance_m'] / 1000:.1f} km away")}
    return {"river_id": int(main["river_id"]), "lat": main["lat"], "lon": main["lon"],
            "how": (f"the largest river (Strahler order {main['strahler_order']}) within "
                    f"{EVENT_RADIUS_M / 1000:g} km of the event, {main['distance_m'] / 1000:.1f} km from its centre")}


def _stem(rid: int, lat: float | None = None, lon: float | None = None, n: int = STEM_REACHES) -> dict[int, int]:
    """The reaches of the river through ``rid``: its main stem up (the branch with the largest area at each
    confluence) and its path down, ``n`` reaches each way, as ``{river_id: steps}`` (negative upstream)."""
    import numpy as np

    from aquascope import rivers

    net = rivers._network_for(rid, lat, lon)
    i0 = net.index(rid)
    assert i0 is not None
    order, parent, starts = rivers._upstream_tree(net, i0, cap=400_000)
    acc = net.area_m2[order].astype(float)
    acc[~np.isfinite(acc)] = 0.0
    bounds = [*starts, len(order)]
    for k in range(len(starts) - 1, 0, -1):
        sl = slice(bounds[k], bounds[k + 1])
        np.add.at(acc, parent[sl], acc[sl])
    area = {int(order[j]): float(acc[j]) for j in range(len(order))}
    out = {rid: 0}
    i, step = i0, 0
    while step < n:
        kids = net.children(i)
        if not kids:
            break
        i = max(kids, key=lambda j: area.get(j, 0.0))
        step += 1
        out[int(net.ids[i])] = -step
    i, step = i0, 0
    while step < n:
        nxt = net.index(int(net.ds[i]))
        if nxt is None:
            break
        i = nxt
        step += 1
        out[int(net.ids[i])] = step
    return out


# ── the model at a reach ────────────────────────────────────────────────────


def _retro(rid: int, a: date, b: date) -> list[tuple[str, float]]:
    """GEOGLOWS's simulated daily flow at a reach from ``a`` to ``b`` (the API's retrospectivedaily)."""
    from aquascope import rivers

    url = f"{rivers.GEOGLOWS_API}/retrospectivedaily/{rid}"
    data = rivers._fetch_json(url, {"format": "json", "start_date": a.strftime("%Y%m%d"),
                                    "end_date": b.strftime("%Y%m%d")})
    vals = data.get(str(rid)) if isinstance(data, dict) else None
    times = data.get("datetime") if isinstance(data, dict) else None
    if not vals or not times:
        return []
    return [(str(t)[:10], float(v)) for t, v in zip(times, vals) if v is not None and float(v) >= 0]


def _curve(rid: int) -> dict[str, Any] | None:
    """The reach's flood flows: :func:`aquascope.floodsim.reach_curve`, the curve every tool's design flow reads."""
    from aquascope import floodsim

    try:
        c = floodsim.reach_curve(rid, fetch=FETCH["bytes"])
    except Exception as exc:  # noqa: BLE001 - the story stands without the return period
        logger.info("annual maxima for %s unavailable: %s", rid, exc)
        return None
    return None if c.get("error") else c


def _peak(rows: list[tuple[str, float]]) -> tuple[str, float] | None:
    if not rows:
        return None
    return max(rows, key=lambda r: r[1])


# ── the gauges ──────────────────────────────────────────────────────────────


def _catalog(stations: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if stations is not None:
        return stations
    from aquascope.archive.catalog import load_stations

    rows: list[dict[str, Any]] = load_stations()
    return rows


def _observed(source: str, station_id: str, a: date, b: date) -> tuple[list[tuple[str, float]], str]:
    """A gauge's daily discharge from ``a`` to ``b`` (the Archive's copy, else the agency's), and where it came
    from."""
    from aquascope.explore import fetch_series

    got = fetch_series(source, station_id, variable="discharge", years=max(1, date.today().year - a.year + 1))
    s = got.get("series")
    if s is None or len(s) == 0:
        return [], got.get("note") or "no record"
    s = s[(s.index >= str(a)) & (s.index <= str(b) + " 23:59")]
    rows = [(str(t)[:10], float(v)) for t, v in s.items() if v == v and float(v) >= 0]
    return rows, got.get("note") or ""


def gauges(river_id: int, a: date, b: date, *, lat: float | None = None, lon: float | None = None,
           stations: list[dict[str, Any]] | None = None, max_gauges: int = MAX_GAUGES,
           gauge_km: float = GAUGE_KM) -> dict[str, Any]:
    """The discharge gauges on the event's river near its reach, each against GEOGLOWS at its own reach.

    Catalogue stations with discharge within ``gauge_km`` of the reach are snapped to the river network
    (:func:`aquascope.rivers.snap_to_river`); those whose reach is on the event's river (its main stem up or its
    path down, :data:`STEM_REACHES` reaches each way) and that have a record in the window are kept, nearest along
    the river first. For each: the observed daily peak and its day, GEOGLOWS's simulated daily peak at the same
    reach in the window and its day, the peak error and the timing error."""
    from aquascope import rivers

    stem = _stem(int(river_id), lat, lon)
    net = rivers._network_for(int(river_id), lat, lon)
    i0 = net.index(int(river_id))
    clat = float(lat) if lat is not None else float(net.lat[i0])
    clon = float(lon) if lon is not None else float(net.lon[i0])
    near = []
    for r in _catalog(stations):
        if "discharge" not in (r.get("variables") or []):
            continue
        end = str(r.get("period_end") or "")[:10]
        if end and end < a.isoformat():
            continue
        try:
            glat, glon = float(r["latitude"]), float(r["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        d = _haversine_km(clat, clon, glat, glon)
        if d <= gauge_km:
            near.append((d, r))
    near.sort(key=lambda t: t[0])
    found: list[dict[str, Any]] = []
    tried = 0
    for _d, r in near[:16]:
        tried += 1
        extra = r.get("extra") if isinstance(r.get("extra"), dict) else {}
        area = _num(extra.get("catchment_area_km2"), 1)
        try:
            # A gauge's catchment area picks its river where a tributary joins beside it (the catalogue's area).
            snap = rivers.snap_to_river(float(r["latitude"]), float(r["longitude"]), max_distance_m=1000.0,
                                        area_km2=area if area and area > 0 else None)
        except Exception as exc:  # noqa: BLE001 - one gauge that cannot be placed does not stop the others
            logger.info("snap failed for %s: %s", r.get("station_id"), exc)
            continue
        if not snap.get("snapped") or int(snap["river_id"]) not in stem:
            continue
        found.append({"r": r, "river_id": int(snap["river_id"]), "steps": stem[int(snap["river_id"])]})
    out = []
    found.sort(key=lambda g: abs(g["steps"]))
    for g in found:
        if len(out) >= max_gauges:
            break
        r = g["r"]
        try:
            obs, note = _observed(str(r["source"]), str(r["station_id"]), a, b)
        except Exception as exc:  # noqa: BLE001 - a gauge whose record cannot be read is skipped, and said
            logger.info("record unavailable for %s: %s", r.get("station_id"), exc)
            continue
        if len(obs) < 3 or max(v for _, v in obs) <= 0:
            continue
        sim = _retro(g["river_id"], a, b)
        po, ps = _peak(obs), _peak(sim)
        row: dict[str, Any] = {
            "source": r["source"], "station_id": str(r["station_id"]),
            "name": _display_name(r.get("name") or str(r["station_id"])),
            "river": r.get("river") or None,
            "lat": float(r["latitude"]), "lon": float(r["longitude"]), "river_id": g["river_id"],
            "reaches_from_event": g["steps"], "observed": [[d, round(v, 3)] for d, v in obs],
            "modelled": [[d, round(v, 3)] for d, v in sim], "record_note": note,
            "licence": _licence_of(str(r["source"])),
        }
        if po:
            row["obs_peak"] = {"date": po[0], "cms": round(po[1], 2)}
        if ps:
            row["model_peak"] = {"date": ps[0], "cms": round(ps[1], 2)}
        if po and ps:
            row["peak_error_pct"] = round(100.0 * (ps[1] - po[1]) / po[1], 1) if po[1] > 0 else None
            row["timing_days"] = (_date(ps[0]) - _date(po[0])).days
        out.append(row)
    # Nearest along the river first; of two as near, the one with the bigger flood (the main river's).
    out.sort(key=lambda x: (abs(x["reaches_from_event"]), -float((x.get("obs_peak") or {}).get("cms") or 0.0)))
    if not out:
        return _missing("gauges", (f"No discharge gauge with a record from {_day(a)} to {_day(b)} on this river "
                                   f"within {gauge_km:g} km (checked {tried} of the catalogue's gauges nearby)."),
                        source="AquaScope Archive gauges and the agencies", coverage="", gauges=[])
    return {"available": True, "piece": "gauges", "gauges": out,
            "source": "AquaScope Archive daily discharge (the agency's record), and the GEOGLOWS v2 retrospective "
                      "at each gauge's reach",
            "licence": "; ".join(sorted({g["licence"] for g in out})) + "; " + LICENCES["model"],
            "coverage": f"daily means, {_day(a)} to {_day(b)}",
            "method": ("Each gauge is snapped to the GEOGLOWS reach it stands on; the peak error is the simulated "
                       "daily peak less the observed, over the observed; the timing is the day of the simulated "
                       "peak less the day of the observed, both inside the window.")}


def _display_name(name: str) -> str:
    """An agency's name as a reader writes it: USGS's are in capitals ("FRENCH BROAD RIVER AT ASHEVILLE, NC")."""
    if not name or name != name.upper() or not any(c.isalpha() for c in name):
        return name
    words = []
    for w in name.split(" "):
        bare = w.strip(",.()")
        words.append(w if len(bare) == 2 and bare.isalpha() and w.endswith(bare) and words and
                     words[-1].endswith(",") else w.capitalize())
    return " ".join(words).replace(" At ", " at ").replace(" Nr ", " near ").replace(" Near ", " near ")


def _licence_of(source: str) -> str:
    from aquascope.registry import SOURCES

    s = SOURCES.get(source)
    return f"{s.label}: {s.license}" if s and s.license else source


# ── the rain ────────────────────────────────────────────────────────────────


def basin_cells(river_id: int, *, lat: float | None = None, lon: float | None = None
                ) -> dict[str, Any]:
    """The upstream basin as cells of a regular grid: every cell holding the centre of a GEOGLOWS unit catchment
    upstream of the reach, at 0.1° (IMERG's own grid), or coarser when that is more than :data:`MAX_RAIN_PARTS`
    row runs. Returns the MultiPolygon (one rectangle per run of cells in a row), the cell size, the bbox and the
    upstream area summed from the routing tables."""
    import numpy as np

    from aquascope import rivers

    net = rivers._network_for(int(river_id), lat, lon)
    i = net.index(int(river_id))
    assert i is not None
    ups = rivers._upstream_of(net, i)
    lo, la = net.lon[ups], net.lat[ups]
    ok = np.isfinite(lo) & np.isfinite(la)
    lo, la = lo[ok], la[ok]
    area = float(np.nansum(net.area_m2[ups])) / 1e6
    if not len(lo):
        raise LookupError("the upstream reaches have no positions")
    for deg in RAIN_CELL_DEG:
        cx = np.floor(lo / deg).astype(int)
        cy = np.floor(la / deg).astype(int)
        cells = sorted(set(zip(cy.tolist(), cx.tolist())))
        runs: list[tuple[int, int, int]] = []
        for y, x in cells:
            if runs and runs[-1][0] == y and runs[-1][2] == x - 1:
                runs[-1] = (y, runs[-1][1], x)
            else:
                runs.append((y, x, x))
        if len(runs) <= MAX_RAIN_PARTS or deg == RAIN_CELL_DEG[-1]:
            break
    polys = [[[[round(x0 * deg, 4), round(y * deg, 4)], [round((x1 + 1) * deg, 4), round(y * deg, 4)],
               [round((x1 + 1) * deg, 4), round((y + 1) * deg, 4)], [round(x0 * deg, 4), round((y + 1) * deg, 4)],
               [round(x0 * deg, 4), round(y * deg, 4)]]] for y, x0, x1 in runs]
    bbox = [round(float(lo.min()), 4), round(float(la.min()), 4), round(float(lo.max()), 4),
            round(float(la.max()), 4)]
    # The basin's outline for the map: the cell edges that belong to one cell only.
    have = set(cells)
    edges = []
    for y, x in cells:
        for dy, dx, seg in ((-1, 0, ((x, y), (x + 1, y))), (1, 0, ((x, y + 1), (x + 1, y + 1))),
                            (0, -1, ((x, y), (x, y + 1))), (0, 1, ((x + 1, y), (x + 1, y + 1)))):
            if (y + dy, x + dx) not in have:
                edges.append([[round(a * deg, 4), round(b * deg, 4)] for a, b in seg])
    return {"geometry": {"type": "MultiPolygon", "coordinates": polys}, "cell_deg": deg, "cells": len(cells),
            "outline": {"type": "MultiLineString", "coordinates": edges},
            "bbox": bbox, "area_km2": round(area, 1), "n_reaches": len(ups)}


def _cs_date(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def rain_request(geometry: dict[str, Any], a: date, b: date) -> str:
    """Ask ClimateSERV for IMERG's daily basin mean from ``a`` to ``b``; returns the request id."""
    import urllib.parse

    q = urllib.parse.urlencode({"datatype": IMERG_DATATYPE, "begintime": _cs_date(a), "endtime": _cs_date(b),
                                "intervaltype": 0, "operationtype": 5,
                                "geometry": json.dumps(geometry, separators=(",", ":"))})
    got = FETCH["json"](f"{CLIMATESERV}/submitDataRequest/?{q}")
    if isinstance(got, list) and got and isinstance(got[0], str):
        return got[0]
    raise RuntimeError(f"ClimateSERV did not take the request: {str(got)[:200]}")


def rain_poll(piece: dict[str, Any]) -> dict[str, Any]:
    """One look at a pending rain request (``piece["request"]``): the finished piece, or the same piece with
    ``pending`` and the progress. The browser calls this every few seconds; :func:`rain` waits on it."""
    rid = piece.get("request")
    if not rid:
        return piece
    prog = FETCH["json"](f"{CLIMATESERV}/getDataRequestProgress/?id={rid}")
    p = float(prog[0]) if isinstance(prog, list) and prog else -1.0
    if p < 0:
        return {**_missing("rain", "ClimateSERV could not compute the basin's rain."), **_rain_meta(piece)}
    if p < 100:
        return {**piece, "pending": True, "progress": p}
    data = FETCH["json"](f"{CLIMATESERV}/getDataFromRequest/?id={rid}")
    rows = []
    for r in (data or {}).get("data", []) if isinstance(data, dict) else []:
        v = (r.get("value") or {}).get("avg")
        try:
            d = datetime.strptime(str(r.get("date")), "%m/%d/%Y").date()
        except ValueError:
            continue
        if v is not None and float(v) >= 0:
            rows.append((d.isoformat(), float(v)))
    rows.sort()
    if not rows:
        return {**_missing("rain", "ClimateSERV returned no IMERG values for these days."), **_rain_meta(piece)}
    total = sum(v for _, v in rows)
    pk = max(rows, key=lambda r: r[1])
    return {**_rain_meta(piece), "available": True, "pending": False,
            "daily_mm": [[d, round(v, 2)] for d, v in rows], "total_mm": round(total, 1),
            "peak": {"date": pk[0], "mm": round(pk[1], 1)}, "days": len(rows)}


def _rain_meta(piece: dict[str, Any]) -> dict[str, Any]:
    keep = ("piece", "source", "licence", "coverage", "method", "basin", "window", "request")
    return {k: piece[k] for k in keep if k in piece}


def rain(river_id: int, a: date, b: date, *, lat: float | None = None, lon: float | None = None,
         wait: bool = True, timeout_s: float = 180.0) -> dict[str, Any]:
    """IMERG's daily rain averaged over the reach's upstream basin from ``a`` to ``b``: the basin total and the
    wettest day. ``wait=False`` returns the pending request for :func:`rain_poll` (the browser polls)."""
    if a < _date(IMERG_FIRST):
        return _missing("rain", f"IMERG starts on {_day(IMERG_FIRST)}.", source="NASA GPM IMERG")
    try:
        basin = basin_cells(int(river_id), lat=lat, lon=lon)
    except Exception as exc:  # noqa: BLE001 - the rest of the story stands without the rain
        return _missing("rain", f"The upstream basin could not be outlined: {exc}", source="NASA GPM IMERG")
    piece: dict[str, Any] = {
        "piece": "rain", "source": "NASA GPM IMERG V07 Late Run, daily, through ClimateSERV (SERVIR)",
        "licence": LICENCES["rain"], "coverage": "June 2000 to near present, 0.1°, daily (UTC days)",
        "method": (f"The mean of the IMERG cells over the upstream basin: the {basin['cells']} cells of "
                   f"{basin['cell_deg']:g}° holding the centre of a GEOGLOWS unit catchment upstream of the reach "
                   f"({basin['n_reaches']:,} reaches, {basin['area_km2']:,.0f} km² summed from the routing tables)."),
        "basin": {k: basin[k] for k in ("bbox", "area_km2", "cell_deg", "cells", "geometry", "outline")},
        "window": [a.isoformat(), b.isoformat()],
    }
    try:
        piece["request"] = rain_request(basin["geometry"], a, b)
    except Exception as exc:  # noqa: BLE001
        return {**_missing("rain", f"ClimateSERV could not be reached: {exc}"), **_rain_meta(piece)}
    piece["pending"] = True
    if not wait:
        return piece
    t0 = time.monotonic()
    while True:
        try:
            piece = rain_poll(piece)
        except Exception as exc:  # noqa: BLE001
            return {**_missing("rain", f"ClimateSERV failed: {exc}"), **_rain_meta(piece)}
        if not piece.get("pending"):
            return piece
        if time.monotonic() - t0 > timeout_s:
            return {**_missing("rain", f"ClimateSERV did not finish in {timeout_s:.0f} s."), **_rain_meta(piece)}
        time.sleep(2.0)


# ── the forecast archive ────────────────────────────────────────────────────

_RIVID: dict[int, Any] = {}


def _decode(buf: bytes, meta: dict[str, Any]) -> Any:
    from aquascope.archive import warnings as fw

    return fw.decode_chunk(buf, meta)


def _forecast_meta(run: str) -> dict[str, Any] | None:
    raw = FETCH["bytes"](f"{FORECAST_BUCKET}/{run}.zarr/.zmetadata")
    if not raw:
        return None
    meta: dict[str, Any] = json.loads(raw)["metadata"]
    return meta


def _forecast_column(rid: int, base: str, meta: dict[str, Any]) -> int | None:
    """The reach's column in a forecast run. The ids are stored processing unit after processing unit, so their
    region (``id // 10,000,000``) never falls along the array: a binary search on the chunks' first ids finds the
    region's chunks (checked on the 2024-11-25 run: all 6,838,900 ids, the region never falls), and those are
    searched. The same order serves every run since 2024-07-01 (checked on three runs)."""
    import numpy as np

    arr = meta["rivid/.zarray"]
    n, c = int(arr["shape"][0]), int(arr["chunks"][0])
    nch = -(-n // c)

    def chunk(i: int) -> Any:
        if i not in _RIVID:
            raw = FETCH["bytes"](f"{base}/rivid/{i}")
            if raw is None:
                raise RuntimeError(f"{base}/rivid/{i} is missing")
            _RIVID[i] = _decode(raw, arr).reshape(-1)[: max(0, min(c, n - i * c))].astype("i8")
        return _RIVID[i]

    region = rid // 10_000_000
    lo, hi = 0, nch - 1
    while lo < hi:  # the last chunk whose first id's region is below this one's
        mid = (lo + hi + 1) // 2
        if int(chunk(mid)[0]) // 10_000_000 < region:
            lo = mid
        else:
            hi = mid - 1
    for i in range(lo, nch):
        ids = chunk(i)
        hit = np.nonzero(ids == rid)[0]
        if len(hit):
            return i * c + int(hit[0])
        if int(ids[-1]) // 10_000_000 > region:
            return None
    return None


def _forecast_run(rid: int, run: str) -> dict[str, Any] | None:
    """One run's ensemble at a reach: the 51 members' and the high-resolution run's daily means (UTC days)."""
    import numpy as np

    from aquascope.archive import warnings as fw

    meta = _forecast_meta(run)
    if meta is None:
        return None
    base = f"{FORECAST_BUCKET}/{run}.zarr"
    lay = fw.forecast_layout(meta)
    col = _forecast_column(rid, base, meta)
    if col is None:
        return {"error": f"reach {rid} is not in the forecast"}
    raw = FETCH["bytes"](f"{base}/Qout/0.0.{col // lay['chunk']}")
    if raw is None:
        return {"error": "the forecast chunk is missing"}
    block = _decode(raw, lay["zarray"])[:, :, col % lay["chunk"]].astype("f8")  # (members, steps)
    tmeta = meta["time/.zarray"]
    traw = FETCH["bytes"](f"{base}/time/0")
    if traw is None:
        return {"error": "the forecast's times are missing"}
    secs = _decode(traw, tmeta).reshape(-1)[: lay["steps"]].astype("i8")
    eraw = FETCH["bytes"](f"{base}/ensemble/0")
    members = (_decode(eraw, meta["ensemble/.zarray"]).reshape(-1)[: lay["members"]] if eraw
               else np.arange(1, lay["members"] + 1))
    daily = fw.daily_means(block[:, :, None], secs, fw.DAYS)[:, :, 0]  # (members, days)
    hres = members == fw.HIGH_RES_MEMBER
    start = _date(lay["start"])
    return {"start": start, "ens": daily[~hres], "hres": daily[hres][0] if hres.any() else None}


def forecasts(river_id: int, peak_date: Any, *, leads: tuple[int, ...] = LEADS,
              curve: dict[str, Any] | None = None, observed_cms: float | None = None) -> dict[str, Any]:
    """What GEOGLOWS's forecast said for ``peak_date`` when it was issued ``leads`` days before: for each lead,
    the 51 members' daily mean on the peak day (median, lowest and highest), the high-resolution run, the
    ensemble median's own 15-day peak and its day, and (with ``curve``) the share of members at or above the
    reach's 2-year flow. From the forecast archive (00 UTC runs from :data:`FORECAST_FIRST`)."""
    import numpy as np

    from aquascope import floodsim

    p = _date(peak_date)
    rid = int(river_id)
    out: list[dict[str, Any]] = []
    named = bool(curve) and all(k in curve for k in ("method", "first_year", "last_year"))  # type: ignore[operator]
    two_year = f" The 2-year flow: {floodsim.flow_words(curve)}." if named else ""  # type: ignore[arg-type]
    for lead in leads:
        issued = p - timedelta(days=int(lead))
        row: dict[str, Any] = {"lead_days": int(lead), "issued": issued.isoformat()}
        if issued < _date(FORECAST_FIRST):
            row["note"] = f"the archive starts on {_day(FORECAST_FIRST)}"
            out.append(row)
            continue
        run = issued.strftime("%Y%m%d") + "00"
        try:
            got = _forecast_run(rid, run)
        except Exception as exc:  # noqa: BLE001 - one run that cannot be read is said, the others go on
            got = {"error": str(exc)}
        if got is None:
            row["note"] = f"no run issued on {_day(issued)} in the archive"
        elif got.get("error"):
            row["note"] = got["error"]
        else:
            k = (p - got["start"]).days
            ens = got["ens"]
            if not 0 <= k < ens.shape[1] or not np.isfinite(ens[:, k]).any():
                row["note"] = "the peak day is outside this run's 15 days"
            else:
                day = ens[:, k][np.isfinite(ens[:, k])]
                med = np.nanmedian(ens, axis=0)
                j = int(np.nanargmax(np.where(np.isfinite(med), med, -np.inf)))
                row.update({
                    "run": run, "median_cms": round(float(np.median(day)), 2),
                    "low_cms": round(float(day.min()), 2), "high_cms": round(float(day.max()), 2),
                    "members": int(len(day)),
                    "hres_cms": _num(got["hres"][k]) if got["hres"] is not None else None,
                    "median_peak": {"date": (got["start"] + timedelta(days=j)).isoformat(),
                                    "cms": round(float(med[j]), 2)},
                })
                if curve:
                    q2 = float(curve["q"][0])
                    row["q2_cms"] = round(q2, 2)
                    row["share_above_q2"] = round(float((day >= q2).mean()), 3)
                if observed_cms:
                    row["median_vs_observed_pct"] = round(100.0 * (row["median_cms"] - observed_cms) / observed_cms, 1)
        out.append(row)
    ok = [r for r in out if "median_cms" in r]
    piece: dict[str, Any] = {
        "piece": "forecast", "river_id": rid, "peak_date": p.isoformat(), "runs": out,
        "source": f"GEOGLOWS v2 forecast archive, {FORECAST_BUCKET.replace('https://', 's3://').split('.')[0]}",
        "licence": LICENCES["forecast"],
        "coverage": (f"daily 00 UTC runs from {_day(FORECAST_FIRST)}, 15 days ahead, 51 members and a "
                     "high-resolution run"),
        "method": ("Each member's daily mean flow (UTC days) at the reach on the peak day; median, lowest and highest "
                   "of the 51 members." + two_year),
    }
    if not ok:
        reasons = "; ".join(f"{r['lead_days']} d: {r.get('note')}" for r in out)
        return {**piece, "available": False, "note": f"No archived forecast covers this peak ({reasons})."}
    return {**piece, "available": True}


# ── the wave ────────────────────────────────────────────────────────────────


def _hourly(rows: list[tuple[str, float]] | list[list[Any]]) -> list[float]:
    """Daily means as an hourly series: linear from one day's value to the next, one value an hour."""
    vals = [float(v) for _, v in rows]
    if len(vals) < 2:
        return vals
    out: list[float] = []
    for x, y in zip(vals, vals[1:]):
        out.extend(x + (y - x) * h / 24.0 for h in range(24))
    out.append(vals[-1])
    return out


def wave(river_id: int, rows: list[Any], *, source: str, peak_date: Any, lat: float | None = None,
         lon: float | None = None, places: list[dict[str, Any]] | None = None,
         stations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The flow at the reach routed down the river with Route it (:func:`aquascope.routing.route`): the daily
    values from :data:`WAVE_DAYS` before the peak to after it, read hour by hour (linear between the days), the
    first hour's flow already in every reach downstream. For each gauge and town on the path: when the peak
    passes, how many hours after it passed the reach, and the flow that water brings there."""
    from aquascope import routing

    p = _date(peak_date)
    a, b = p - timedelta(days=WAVE_DAYS[0]), p + timedelta(days=WAVE_DAYS[1])
    sel = [(d, v) for d, v in rows if a.isoformat() <= str(d)[:10] <= b.isoformat()]
    if len(sel) < 3:
        return _missing("wave", "Too few days of flow at the reach to route.", source="Route it")
    q = _hourly(sel)
    try:
        res = routing.route(int(river_id), lat=lat, lon=lon, inflow_cms=q, inflow_source=source,
                            places=places, stations=stations, max_reaches=1500)
    except Exception as exc:  # noqa: BLE001
        return _missing("wave", f"Route it could not route this river: {exc}", source="Route it")
    if not res.get("ok"):
        return _missing("wave", res.get("error") or "Route it failed.", source="Route it")
    t0 = _date(sel[0][0])
    start_peak = res["reaches"][0].get("peak_h")
    stops = []
    for pl in res.get("places") or []:
        if not pl.get("reached") or pl.get("peak_h") is None:
            continue
        lag = None if start_peak is None else round(float(pl["peak_h"]) - float(start_peak), 1)
        when = datetime.combine(t0, datetime.min.time()) + timedelta(hours=float(pl["peak_h"]))
        stops.append({"name": _display_name(str(pl.get("name") or "")), "kind": pl.get("kind"), "lat": pl.get("lat"),
                      "lon": pl.get("lon"),
                      "along_km": pl.get("along_km"), "peak_cms": pl.get("peak_cms"), "lag_h": lag,
                      "peak_at": when.strftime("%Y-%m-%dT%H:00")})
    return {
        "available": True, "piece": "wave", "river_id": int(river_id), "start": t0.isoformat(),
        "inflow_source": source, "inflow_peak_cms": res["inflow"]["peak_cms"], "hours": res["hours"],
        "places": stops, "route_places": res.get("places"), "inflow": res.get("inflow"),
        "furthest": res.get("furthest"), "n_reaches": res.get("n_reaches"),
        "geometry": res.get("geometry"), "frames": res.get("frames"), "reaches": res.get("reaches"),
        "source": "Route it: Muskingum routing on GEOGLOWS v2's own network (aquascope.routing)",
        "licence": LICENCES["routing"], "coverage": "every river GEOGLOWS v2 models",
        "method": (f"{source}, read hour by hour, routed with GEOGLOWS's own k and x reach by reach. The water from "
                   "this reach alone: the rivers that join below add their own on top."),
        "check": routing.CHECK["summary"],
    }


# ── the flood ───────────────────────────────────────────────────────────────


def flood(river_id: int, flow_cms: float, *, flow_source: str, lat: float | None = None,
          lon: float | None = None, coarse: bool = True, exposure: bool = True,
          terrain_fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """Flood it (:func:`aquascope.floodsim.flood_reach`) at ``flow_cms`` at the reach, to a near steady state, and
    what is under the largest depth: buildings and road length (Overture Maps) and people (GHS-POP), counted as
    the flood views count them, inside the wet area's box (at most 900 km², centred on the reach when larger).
    ``coarse`` runs the worker's grid (cells twice as large), so numpy finishes in a few minutes."""
    import numpy as np

    from aquascope import floodsim

    try:
        res = floodsim.flood_reach(int(river_id), lat=lat, lon=lon, flow_cms=float(flow_cms), coarse=coarse,
                                   frames=False, terrain_fetch=terrain_fetch)
    except Exception as exc:  # noqa: BLE001
        return _missing("flood", f"Flood it could not run here: {exc}", source="Flood it")
    s = res["summary"]
    g = res["grid"]
    piece: dict[str, Any] = {
        "available": True, "piece": "flood", "river_id": int(river_id), "flow_cms": round(float(flow_cms), 2),
        "flow_source": flow_source, "wet_km2": s["wet_km2"], "max_depth_m": s["max_depth_m"],
        "steady_at_h": s["steady_at_h"], "hours_run": s["hours_run"], "cell_m": s["cell_m"], "bounds": s["bounds"],
        "label": FLOOD_LABEL, "source": "Flood it (aquascope.floodsim): " + floodsim.SCHEME,
        "licence": LICENCES["flood"], "coverage": "any reach GEOGLOWS v2 draws; terrain from the Terrarium tiles",
    }
    if not exposure:
        return piece
    depth = floodsim.depth_over_terrain(g, res["run"]["h_max"])
    wet = depth > 0.05
    if not wet.any():
        piece["exposure"] = _missing("exposure", "Nothing is under water.")
        return piece
    rows, cols = np.nonzero(wet)
    lon0, lat0 = floodsim.cell_lonlat(g["window"], np.array([rows.min() - 1, rows.max() + 1]),
                                      np.array([cols.min() - 1, cols.max() + 1]))
    w, e = float(min(lon0)), float(max(lon0))
    s_, n_ = float(min(lat0)), float(max(lat0))
    clon = float(lon) if lon is not None else (w + e) / 2
    clat = float(lat) if lat is not None else (s_ + n_) / 2
    box = _clip_box([w, s_, e, n_], clon, clat)
    from aquascope import flood_exposure

    try:
        read = flood_exposure.read_sources(box)
        cmp = floodsim.compare_exposure(g, np.zeros_like(depth), depth, read)
    except Exception as exc:  # noqa: BLE001
        piece["exposure"] = _missing("exposure", f"The buildings, roads or people could not be read: {exc}")
        return piece
    piece["exposure"] = {
        "available": True, "piece": "exposure", "bbox": [round(v, 5) for v in box],
        "buildings": (cmp.get("buildings") or {}).get("after"), "roads_km": (cmp.get("roads_km") or {}).get("after"),
        "people": (cmp.get("people") or {}).get("after"), "threshold_m": cmp.get("threshold_m"),
        "errors": cmp.get("errors") or {}, "source": "Overture Maps buildings and roads; GHS-POP R2023A",
        "licence": LICENCES["exposure"], "coverage": "worldwide; buildings where Overture maps them",
        "method": "the flood views' own count (aquascope.flood_exposure): a building counts when the water reaches "
                  "its centre or a corner; people from the 100 m population grid under the water",
    }
    return piece


def _clip_box(box: list[float], lon: float, lat: float, max_km2: float = 900.0) -> list[float]:
    w, s, e, n = box
    kx = 111.32 * math.cos(math.radians((s + n) / 2))
    area = (e - w) * kx * (n - s) * 110.57
    if area <= max_km2 * 0.98:
        return box
    half = math.sqrt(max_km2 * 0.98) / 2
    dx, dy = half / kx, half / 110.57
    return [max(w, lon - dx), max(s, lat - dy), min(e, lon + dx), min(n, lat + dy)]


# ── Floods past: the events near a place ───────────────────────────────────


def candidates(lat: float, lon: float, *, month: Any = None, start: Any = None, end: Any = None,
               radius_km: float = 40.0, limit: int = 5) -> dict[str, Any]:
    """The floods Floods past has near a point: its news events within ``radius_km`` in ``month`` (or ``start`` to
    ``end``), grouped into runs of days (a gap of more than two days starts a new one), the runs with the most
    reports first. Each candidate: ``{rank, start, end, reports, lat, lon, area_km2}`` (the median position of
    its reports). A model picks one by ``rank``; the code builds the replay."""
    from statistics import median

    from aquascope.context import floods_past

    dlat = radius_km / 110.57
    dlon = radius_km / (111.32 * max(0.1, math.cos(math.radians(lat))))
    box = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
    res = floods_past.flood_events_month(month, start=start, end=end, bbox=box, limit=5000)
    if not res.get("ok", True) or not res.get("available", True):
        return {"ok": False, "candidates": [],
                "note": res.get("summary") or res.get("note") or "Floods past unavailable"}
    events = [e for e in (res.get("news") or {}).get("events") or [] if e.get("start")
              and _haversine_km(lat, lon, e["lat"], e["lon"]) <= radius_km]
    events.sort(key=lambda e: e["start"])
    groups: list[list[dict[str, Any]]] = []
    for e in events:
        last_end = max((_date(x.get("end") or x["start"]) for x in groups[-1]), default=None) if groups else None
        if last_end is not None and (_date(e["start"]) - last_end).days <= 2:
            groups[-1].append(e)
        else:
            groups.append([e])
    ranked = sorted(groups, key=lambda g: (-len(g), g[0]["start"]))[: max(1, int(limit))]
    out = []
    for k, g in enumerate(ranked, start=1):
        out.append({"rank": k, "start": min(e["start"] for e in g)[:10],
                    "end": max((e.get("end") or e["start"]) for e in g)[:10], "reports": len(g),
                    "lat": round(median(e["lat"] for e in g), 4), "lon": round(median(e["lon"] for e in g), 4),
                    "area_km2": round(sum(float(e.get("area_km2") or 0) for e in g), 1)})
    return {"ok": True, "candidates": out, "window": res.get("window"), "radius_km": radius_km,
            "source": "Floods past: Groundsource news events", "licence": LICENCES["floods_past"],
            "note": (f"{len(events):,} news reports within {radius_km:g} km, {len(groups)} runs of days."
                     if events else "No flood in the news within this distance in these months.")}


# ── the record ──────────────────────────────────────────────────────────────


def event_record(river_id: Any = None, *, lat: float | None = None, lon: float | None = None, start: Any = None,
                 end: Any = None, event: dict[str, Any] | None = None, stations: list[dict[str, Any]] | None = None,
                 places: list[dict[str, Any]] | None = None, with_flood: bool = True, rain_wait: bool = True,
                 coarse: bool = True, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Everything the data says about one flood: the reach, the window, and the pieces (rain, gauges, model,
    forecast, wave, flood), each with its source, licence and coverage, or ``available: False`` and why.

    Give ``event`` (a Floods past news event or candidate: ``start``, ``end``, ``lat``, ``lon``), or ``river_id``
    (or ``lat``/``lon``) and ``start``/``end``. ``stations`` is the gauge catalogue to search (the Archive's by
    default), ``places`` towns to place on the wave. ``with_flood`` runs Flood it at the peak (minutes in numpy; the
    Explorer runs it on the GPU instead and passes ``with_flood=False``). ``rain_wait=False`` leaves the rain as a
    pending request for :func:`rain_poll`."""
    say = progress or (lambda _m: None)
    ev = dict(event or {})
    if ev:
        lat = ev.get("lat", lat)
        lon = ev.get("lon", lon)
        start = ev.get("start", start)
        end = ev.get("end", end)
    if not start:
        raise ValueError("give the event's start date (and its end, or one day)")
    a, b = event_window(start, end)
    say("Finding the river…")
    reach = _reach_for(river_id, lat, lon)
    rid = int(reach["river_id"])
    rec: dict[str, Any] = {
        "made_by": "aquascope.replay", "made": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "event": {"start": str(_date(start)), "end": str(_date(end or start)), "lat": lat, "lon": lon,
                  "reports": ev.get("reports"), "from": "Floods past" if ev else "given"},
        "window": [a.isoformat(), b.isoformat()], "reach": reach, "pieces": {},
    }
    pc = rec["pieces"]
    say("Reading the gauges and GEOGLOWS at each…")
    pc["gauges"] = gauges(rid, a, b, lat=reach.get("lat"), lon=reach.get("lon"), stations=stations)
    primary = (pc["gauges"].get("gauges") or [None])[0]
    focus_rid = int(primary["river_id"]) if primary else rid
    model_rows = [(d, v) for d, v in (primary["modelled"] if primary else _retro(rid, a, b))]
    mp = _peak(model_rows)
    curve = _curve(focus_rid)
    pc["model"] = ({"available": True, "piece": "model", "river_id": focus_rid,
                   "daily": [[d, round(v, 3)] for d, v in model_rows],
                   "peak": {"date": mp[0], "cms": round(mp[1], 2)},
                   "return_period": _num(_rp(curve, mp[1]), 1) if curve else None,
                   "curve": curve, "source": "GEOGLOWS v2 retrospective simulation (daily)",
                   "licence": LICENCES["model"], "coverage": "1940 to a few weeks ago, every river GEOGLOWS models"}
                  if mp else _missing("model", "GEOGLOWS returned no simulated flow for these days."))
    if primary and primary.get("obs_peak"):
        peak_date, peak_q = primary["obs_peak"]["date"], float(primary["obs_peak"]["cms"])
        peak_from = f"the daily mean observed at {primary['name']}"
    elif mp:
        peak_date, peak_q = mp[0], float(mp[1])
        peak_from = "GEOGLOWS's simulated daily peak"
    else:
        peak_date, peak_q, peak_from = None, None, None
    rec["peak"] = {"date": peak_date, "cms": _num(peak_q), "from": peak_from, "river_id": focus_rid}
    if primary:
        rec["focus"] = {"lat": primary["lat"], "lon": primary["lon"], "name": primary["name"], "river_id": focus_rid}
    else:
        rec["focus"] = {"lat": reach.get("lat"), "lon": reach.get("lon"), "name": f"reach {rid}", "river_id": rid}
    say("Asking IMERG for the rain over the basin…")
    pc["rain"] = rain(focus_rid, a, b, lat=rec["focus"]["lat"], lon=rec["focus"]["lon"], wait=rain_wait)
    say("Reading what the forecast said before the peak…")
    pc["forecast"] = (forecasts(focus_rid, peak_date, curve=curve, observed_cms=peak_q if primary else None)
                     if peak_date else _missing("forecast", "No peak to look back from."))
    say("Routing the wave down the river…")
    if primary and len(primary.get("observed") or []) >= 3:
        rows, src = primary["observed"], f"the daily flow observed at {primary['name']}"
    else:
        rows, src = model_rows, "GEOGLOWS's simulated daily flow"
    pc["wave"] = (wave(focus_rid, rows, source=src, peak_date=peak_date, lat=rec["focus"]["lat"],
                      lon=rec["focus"]["lon"], places=places, stations=stations)
                 if peak_date else _missing("wave", "No peak to route."))
    if with_flood and peak_q:
        say("Flooding the valley at the peak flow (Flood it)…")
        pc["flood"] = flood(focus_rid, peak_q, flow_source=peak_from or "", lat=rec["focus"]["lat"],
                           lon=rec["focus"]["lon"], coarse=coarse)
    elif peak_q:
        pc["flood"] = {"available": False, "piece": "flood", "pending": True, "flow_cms": round(peak_q, 2),
                      "flow_source": peak_from, "river_id": focus_rid, "label": FLOOD_LABEL,
                      "note": "Flood it runs on the page, at the flow above.", "licence": LICENCES["flood"]}
    else:
        pc["flood"] = _missing("flood", "No peak flow to flood with.")
    rec["name"] = _event_name(rec)
    return rec


def _rp(curve: dict[str, Any] | None, q: float) -> float | None:
    from aquascope import floodsim

    if not curve:
        return None
    return floodsim.return_period_for(curve, float(q))


def _event_name(rec: dict[str, Any]) -> str:
    f = rec.get("focus") or {}
    when = rec.get("peak", {}).get("date") or rec["event"]["start"]
    d = _date(when)
    return f"{f.get('name') or 'the river'}, {d.strftime('%B')} {d.year}"


# ── the story ───────────────────────────────────────────────────────────────


def _zoom_for(bbox: list[float]) -> float:
    """The zoom that fits a box in about 800 by 480 pixels of a 512-pixel-tile map (MapLibre), the room the note
    leaves; latitude is stretched by the Mercator factor."""
    w, s, e, n = bbox
    stretch = 1.0 / max(0.2, math.cos(math.radians((s + n) / 2)))
    span = max(e - w, (n - s) * stretch * 800.0 / 480.0, 0.02)
    return round(max(3.0, min(13.0, math.log2(360.0 * 800.0 / 512.0 / span))), 2)


def _centre(bbox: list[float]) -> list[float]:
    return [round((bbox[0] + bbox[2]) / 2, 5), round((bbox[1] + bbox[3]) / 2, 5)]


def _signed(v: float, unit: str) -> str:
    return f"{'+' if v > 0 else ''}{v:,.0f} {unit}" if abs(v) >= 1 else f"{v:+.1f} {unit}"


def _days_words(n: int) -> str:
    if n == 0:
        return "on the same day"
    return f"{abs(n)} day{'s' if abs(n) != 1 else ''} {'later' if n > 0 else 'earlier'}"


def _hours_words(h: float) -> str:
    return f"{h:.0f} h" if h < 48 else f"{h / 24:.1f} days"


def story(rec: dict[str, Any], *, flood_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """The record as a story: ordered stops, each a camera, a date, the layers and a note written from the numbers
    in the record by the templates below. ``flood_result`` (``{"wet_km2", "max_depth_m", "buildings", "roads_km",
    "people", "engine"}``, the page's own Flood it run) fills the flood and exposure stops; without it, and without
    a flood in the record, those stops carry their ``template`` and what they ``need``."""
    pc = rec["pieces"]
    f = rec["focus"]
    peak = rec.get("peak") or {}
    stops: list[dict[str, Any]] = []
    rain_p, gauges_p, fc_p, wave_p, flood_p = (pc.get(k) or {} for k in ("rain", "gauges", "forecast", "wave", "flood"))
    basin_box = (rain_p.get("basin") or {}).get("bbox")
    here = [round(float(f["lon"]), 5), round(float(f["lat"]), 5)]
    wide = {"center": _centre(basin_box), "zoom": _zoom_for(basin_box), "pitch": 0, "bearing": 0} if basin_box else \
        {"center": here, "zoom": 8.0, "pitch": 0, "bearing": 0}

    def stop(sid: str, kind: str, title: str, camera: dict[str, Any], day: Any, layers: list[str], note: str,
             **extra: Any) -> None:
        stops.append({"id": sid, "kind": kind, "title": title, "camera": camera, "date": day, "layers": layers,
                      "note": note, **extra})

    ev = rec["event"]
    span = (f"{_day(ev['start'])}" if ev["start"] == ev["end"] else f"{_day(ev['start'])} to {_day(ev['end'])}")
    if ev.get("from") == "Floods past" and ev.get("reports"):
        note0 = f"{ev['reports']:,} reports of flooding in the news near {f['name']}, {span}."
    else:
        note0 = f"{rec['name']}: the days from {_day(rec['window'][0])} to {_day(rec['window'][1])}."
    stop("event", "event", "The flood", {**wide, "zoom": max(3.0, wide["zoom"] - 0.6)}, ev["start"],
         ["floods_past"], note0, focus={"lon": here[0], "lat": here[1]})
    if rain_p.get("available"):
        pk = rain_p["peak"]
        stop("rain", "rain", "The rain", {**wide, "pitch": 30, "bearing": -10}, pk["date"], ["precip", "basin"],
             f"IMERG: {rain_p['total_mm']:,.0f} mm fell on the {rain_p['basin']['area_km2']:,.0f} km² upstream from "
             f"{_day(rain_p['window'][0])} to {_day(rain_p['window'][1])}; the wettest day, {_day(pk['date'])}, "
             f"brought {pk['mm']:,.0f} mm.", basin=rain_p["basin"]["geometry"],
             outline=rain_p["basin"].get("outline"),
             series={"kind": "rain", "unit": "mm", "values": rain_p["daily_mm"]})
    elif rain_p.get("pending"):
        stop("rain", "rain", "The rain", {**wide, "pitch": 30, "bearing": -10}, peak.get("date"), ["precip", "basin"],
             "IMERG's rain over the basin is still being read.", needs="rain",
             basin=(rain_p.get("basin") or {}).get("geometry"))
    else:
        stop("rain", "rain", "The rain", {**wide, "pitch": 30, "bearing": -10}, peak.get("date"), ["precip"],
             f"Basin rain not available: {rain_p.get('note') or 'no reason given'}")
    near = {"center": here, "zoom": 11.2, "pitch": 50, "bearing": -20}
    if gauges_p.get("available"):
        g = gauges_p["gauges"][0]
        op, mp = g.get("obs_peak"), g.get("model_peak")
        note = f"{g['name']} peaked at {_q(op['cms'])} on {_day(op['date'])} (daily mean)."
        if mp:
            note += (f" GEOGLOWS simulated {_q(mp['cms'])} on {_day(mp['date'])}: "
                     f"{_signed(g['peak_error_pct'], '%')}, {_days_words(g['timing_days'])}.")
        rp = (pc.get("model") or {}).get("return_period")
        if rp and mp:
            note += f" GEOGLOWS's peak is a {rp:,.0f}-year flow on its own record since 1940."
        more = [x for x in gauges_p["gauges"][1:] if x.get("obs_peak")]
        stop("gauge", "gauge", "The gauge against the model", near, op["date"], ["rivers"], note,
             focus={"lon": g["lon"], "lat": g["lat"]},
             series={"kind": "flow", "unit": "m³/s", "observed": g["observed"], "modelled": g["modelled"],
                     "label": g["name"]},
             others=[{"name": x["name"], "obs_peak": x["obs_peak"], "model_peak": x.get("model_peak")} for x in more])
    elif (pc.get("model") or {}).get("available"):
        m = pc["model"]
        stop("gauge", "gauge", "The river in the model", near, m["peak"]["date"], ["rivers"],
             f"No gauge on this river near the flood. GEOGLOWS simulated a daily peak of {_q(m['peak']['cms'])} on "
             f"{_day(m['peak']['date'])}" + (f", a {m['return_period']:,.0f}-year flow on its own record." if
                                             m.get("return_period") else "."),
             series={"kind": "flow", "unit": "m³/s", "modelled": m["daily"], "label": f"reach {m['river_id']}"})
    for r in fc_p.get("runs") or []:
        cam = {**near, "zoom": near["zoom"] - 0.4 - 0.1 * r["lead_days"], "bearing": -20 + 6 * r["lead_days"]}
        title = f"The forecast, {r['lead_days']} day{'s' if r['lead_days'] != 1 else ''} before"
        if "median_cms" in r:
            note = (f"Issued {_day(r['issued'])}: for {_day(fc_p['peak_date'])} the forecast's median was "
                    f"{_q(r['median_cms'])} (members {_q(r['low_cms'])} to {_q(r['high_cms'])})")
            if r.get("median_vs_observed_pct") is not None:
                note += f", {_signed(r['median_vs_observed_pct'], '%')} against the gauge"
            if r.get("share_above_q2") is not None:
                note += (f"; {r['share_above_q2'] * 100:.0f} % of members passed the 2-year flow "
                         f"({_q(r['q2_cms'])})")
            note += "."
        else:
            note = f"Issued {_day(r['issued'])}: no archived forecast ({r.get('note')})."
        stop(f"forecast-{r['lead_days']}", "forecast", title, cam, r["issued"], ["rivers"], note,
             focus={"lon": here[0], "lat": here[1]}, run=r)
    if fc_p and not fc_p.get("runs"):
        stop("forecast", "forecast", "The forecast", near, peak.get("date"), ["rivers"],
             f"Forecast not available: {fc_p.get('note')}")
    if wave_p.get("available"):
        pts = [x for x in wave_p["places"] if x.get("lag_h") is not None and x["lag_h"] > 0
               and float(x.get("along_km") or 0.0) <= WAVE_NOTE_KM]
        far = pts[-1] if pts else None
        note = f"Routed from {f['name']} ({wave_p['inflow_source']}, peak {_q(wave_p['inflow_peak_cms'])})"
        if far:
            mid = pts[len(pts) // 2] if len(pts) > 2 else None
            note += (f": the peak reaches {far['name']} {_hours_words(far['lag_h'])} later, at "
                     f"{_q(far['peak_cms'])}")
            if mid and mid is not far:
                note += f"; {mid['name']} after {_hours_words(mid['lag_h'])}"
            note += ". The water from here alone; the rivers that join add their own."
        else:
            note += ". No gauge or town on the stretch it reached."
        box = _bbox_of([[float(f["lon"]), float(f["lat"])]] + [[float(x["lon"]), float(x["lat"])] for x in pts
                                                              if x.get("lon") is not None])
        cam = ({"center": _centre(box), "zoom": _zoom_for(box) - 0.2, "pitch": 35, "bearing": 0} if box else near)
        stop("wave", "wave", "The wave downstream", cam, peak.get("date"), ["route"], note)
    else:
        stop("wave", "wave", "The wave downstream", near, peak.get("date"), [],
             f"Route it not available: {wave_p.get('note')}")
    fl_cam = {"center": here, "zoom": 11.9, "pitch": 55, "bearing": 18}
    tmpl_flood = ("At {flow} ({flow_source}), Flood it puts {wet_km2} km² under water, {max_depth_m} m at the "
                  "deepest. A screening flood: no bathymetry, terrain errors of metres; not a design flood.")
    tmpl_expo = "Under that water: {buildings} buildings, {roads_km} km of road and about {people} people."
    fr = flood_result or (flood_p if flood_p.get("available") else None)
    if fr or flood_p.get("pending"):
        fill = _flood_fill(flood_p, fr) if fr else None
        stop("flood", "flood", "The flood at the peak", fl_cam, peak.get("date"), ["floodsim"],
             tmpl_flood.format(**fill) if fill else "Flood it is running at the peak flow…",
             template=tmpl_flood, needs=None if fill else "flood",
             fill=None if fill else {"flow": _q(flood_p.get("flow_cms")), "flow_source": flood_p.get("flow_source")},
             run={"river_id": flood_p.get("river_id"), "flow_cms": flood_p.get("flow_cms"), "lat": f["lat"],
                  "lon": f["lon"]})
        ex = (fr or {}).get("exposure")
        efill = _expo_fill(ex) if ex else None
        stop("exposure", "exposure", "What was under the water", {**fl_cam, "zoom": 12.7, "bearing": 40},
             peak.get("date"), ["floodsim", "exposure"],
             tmpl_expo.format(**efill) if efill else (
                 f"What is under the water could not be counted: {ex.get('note')}" if ex and not ex.get("available")
                 else "Counting the buildings, roads and people under the water…"),
             template=tmpl_expo, needs=None if (efill or (ex and not ex.get("available"))) else "exposure")
    else:
        stop("flood", "flood", "The flood at the peak", fl_cam, peak.get("date"), [],
             f"Flood it not available: {flood_p.get('note')}")
    stop("end", "end", "What the data says", {**wide, "pitch": 0, "bearing": 0}, peak.get("date"),
         ["floods_past", "rivers"], _summary(rec))
    sources = []
    for k in ("rain", "gauges", "model", "forecast", "wave", "flood"):
        p = pc.get(k) or {}
        sources.append({"piece": k, "available": bool(p.get("available")), "source": p.get("source", ""),
                        "licence": p.get("licence", LICENCES.get(k, "")), "coverage": p.get("coverage", ""),
                        **({"note": p["note"]} if not p.get("available") and p.get("note") else {})})
    if ev.get("from") == "Floods past":
        sources.append({"piece": "floods_past", "available": True, "source": "Floods past (AquaScope Archive)",
                        "licence": LICENCES["floods_past"], "coverage": "news from 2000; radar Oct 2014 to Sep 2024"})
    return {"schema": STORY_SCHEMA, "version": STORY_VERSION, "title": f"Replay: {rec['name']}",
            "event": {"river_id": rec["reach"]["river_id"], "lat": f["lat"], "lon": f["lon"], "start": ev["start"],
                      "end": ev["end"], "peak_date": peak.get("date"), "name": rec["name"],
                      "window": rec["window"]},
            "stops": stops, "sources": sources, "made_by": "aquascope.replay"}


def _bbox_of(coords: list[list[float]]) -> list[float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def _flood_fill(flood_p: dict[str, Any], fr: dict[str, Any]) -> dict[str, Any]:
    return {"flow": _q(fr.get("flow_cms", flood_p.get("flow_cms"))),
            "flow_source": fr.get("flow_source") or flood_p.get("flow_source") or "",
            "wet_km2": f"{float(fr['wet_km2']):,.1f}", "max_depth_m": f"{float(fr['max_depth_m']):,.1f}"}


def _expo_fill(ex: dict[str, Any]) -> dict[str, Any] | None:
    if not ex or not ex.get("available") or ex.get("buildings") is None:
        return None
    return {"buildings": f"{int(ex['buildings']):,}", "roads_km": f"{float(ex.get('roads_km') or 0):,.1f}",
            "people": f"{float(ex.get('people') or 0):,.0f}"}


def _summary(rec: dict[str, Any]) -> str:
    pc = rec["pieces"]
    bits = []
    g = (pc.get("gauges") or {}).get("gauges") or []
    if g and g[0].get("obs_peak"):
        bits.append(f"{g[0]['name']}: {_q(g[0]['obs_peak']['cms'])} on {_day(g[0]['obs_peak']['date'])}")
    m = pc.get("model") or {}
    if m.get("available"):
        bits.append(f"GEOGLOWS: {_q(m['peak']['cms'])}")
    r = pc.get("rain") or {}
    if r.get("available"):
        bits.append(f"IMERG: {r['total_mm']:,.0f} mm on the basin")
    runs = [x for x in (pc.get("forecast") or {}).get("runs") or [] if "median_cms" in x]
    if runs:
        x = max(runs, key=lambda z: z["lead_days"])
        bits.append(f"forecast {x['lead_days']} days ahead: {_q(x['median_cms'])}")
    missing = [k for k in ("rain", "gauges", "forecast", "wave", "flood")
               if not (pc.get(k) or {}).get("available") and not (pc.get(k) or {}).get("pending")]
    s = "; ".join(bits) + "." if bits else "The data for this flood is thin."
    if missing:
        s += " Not available: " + ", ".join(missing) + "."
    return s


# ── the report ──────────────────────────────────────────────────────────────


def report(rec: dict[str, Any], st: dict[str, Any] | None = None, *,
           flood_result: dict[str, Any] | None = None) -> str:
    """A short Markdown report of the replay: the event, each piece's numbers, the story's notes in order, and the
    sources with their licences and coverage."""
    st = st or story(rec, flood_result=flood_result)
    pc = rec["pieces"]
    lines = [f"# {st['title']}", "",
             f"River reach {rec['reach']['river_id']} ({rec['reach'].get('how')}); the days from "
             f"{_day(rec['window'][0])} to {_day(rec['window'][1])}. Made by aquascope.replay on {rec['made'][:10]}.",
             "", "## The story", ""]
    for i, s in enumerate(st["stops"], start=1):
        when = f" ({_day(s['date'])})" if s.get("date") else ""
        lines.append(f"{i}. **{s['title']}**{when}: {s['note']}")
    lines += ["", "## The numbers", ""]
    g = (pc.get("gauges") or {}).get("gauges") or []
    if g:
        lines += ["| Gauge | Observed peak | GEOGLOWS peak | Peak error | Timing |", "| --- | --- | --- | --- | --- |"]
        for x in g:
            op, mp = x.get("obs_peak") or {}, x.get("model_peak") or {}
            lines.append(f"| {x['name']} ({x['source']} {x['station_id']}) | {_q(op.get('cms'))}, {op.get('date')} | "
                         f"{_q(mp.get('cms'))}, {mp.get('date')} | "
                         f"{'' if x.get('peak_error_pct') is None else _signed(x['peak_error_pct'], '%')} | "
                         f"{'' if x.get('timing_days') is None else _signed(x['timing_days'], 'd')} |")
        lines.append("")
    runs = [x for x in (pc.get("forecast") or {}).get("runs") or []]
    if runs:
        lines += ["| Forecast issued | Lead | Median for the peak day | Members | High-res | Above the 2-year flow |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for x in runs:
            if "median_cms" in x:
                share = "" if x.get("share_above_q2") is None else f"{x['share_above_q2'] * 100:.0f} %"
                lines.append(f"| {x['issued']} | {x['lead_days']} d | {_q(x['median_cms'])} | "
                             f"{_q(x['low_cms'])} to {_q(x['high_cms'])} | {_q(x.get('hres_cms'))} | {share} |")
            else:
                lines.append(f"| {x['issued']} | {x['lead_days']} d | not available: {x.get('note')} | | | |")
        lines.append("")
    w = pc.get("wave") or {}
    if w.get("available") and w.get("places"):
        lines += ["| On the wave's path | km along | Peak passes | Hours after the start | Flow from upstream |",
                  "| --- | --- | --- | --- | --- |"]
        for x in w["places"][:12]:
            lines.append(f"| {x['name']} ({x['kind']}) | {x.get('along_km')} | {x.get('peak_at')} | "
                         f"{'' if x.get('lag_h') is None else x['lag_h']} | {_q(x.get('peak_cms'))} |")
        lines.append("")
    lines += ["## Sources, licences and coverage", ""]
    for s in st["sources"]:
        state = "" if s["available"] else f" Not available: {s.get('note', '')}"
        lines.append(f"- **{s['piece']}**: {s['source']}. Licence: {s['licence']}. Coverage: {s['coverage']}.{state}")
    lines += ["", "Every number above was computed by AquaScope from the data it names; none was written by a model.",
              "The flood is " + FLOOD_LABEL + "."]
    return "\n".join(lines) + "\n"


def replay(river_id: Any = None, **kwargs: Any) -> dict[str, Any]:
    """:func:`event_record`, :func:`story` and :func:`report` in one call: ``{"record", "story", "report"}``."""
    rec = event_record(river_id, **kwargs)
    st = story(rec)
    return {"record": rec, "story": st, "report": report(rec, st)}
