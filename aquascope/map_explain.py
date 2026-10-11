"""Explain what is on screen (#562): a fact list taken from the map by code, and short notes pinned to places.

Someone looking at the Explorer's globe asks "what am I seeing?" and gets three to five short notes pinned on
the map, each at the place it describes and each naming its source, rather than a report. This module is the
engine behind that, shared by the page, ``aquascope layers explain`` and the MCP tool ``explain_view``:

* :func:`view_facts` reads what is in a view (a bounding box, a date and the layers that are on) into a list
  of facts, by code: the share of the mapped land in each river status class, the largest area much above
  or much below normal, the reach forecast to pass the highest return-period flow, the gauges furthest from
  normal today, and the flood events in the news and seen by radar. Every fact carries its value, unit,
  place and source, and an id (``f1``, ``f2``...).
* :func:`template_notes` turns the facts into notes with fixed wording. It needs no model and no key, and it
  is what the page shows when no model is at hand.
* :func:`notes_prompt`, :func:`check_model_notes` and :func:`model_notes` are the model path. A model (the
  reader's own key, or the on-device model in the browser) only chooses and orders facts and writes the
  connecting words. It writes numbers as placeholders (``{f3}``), which code fills in from the facts, and a
  note whose words carry any number (in digits or in words) that is not in the facts it cites, or a sign or
  a digit set against a placeholder ("-{f3}", "{f3}0"), is refused (the claim lock, shared with
  :mod:`aquascope.map_scout` and :mod:`aquascope.mesh_review`).

:func:`explain_view` gathers the published layers itself (the world river status file, the Floods ahead
issue, the Floods past grid and today's gauge status) for the CLI and the MCP tool; the page hands
:func:`view_facts` what it has already loaded, so the two read the same files the same way.

Pure standard library and numpy (the status file), so it runs in the Explorer's light Pyodide worker.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

from aquascope.map_layers import STATUS_ATTRIBUTION, STATUS_CLASSES, STATUS_LICENCE

__all__ = [
    "LAYERS", "MAX_NOTES", "MIN_NOTES", "check_model_notes", "claim_lock", "explain", "explain_view", "model_notes",
    "name_notes", "notes_prompt", "place_name", "placeholder_touches", "status_sample", "template_notes",
    "view_facts",
]

#: The layers a view can be explained from, by the ids aquascope.map_commands.LAYERS uses (gauges are the
#: gauges coloured by today against normal).
LAYERS = ("status", "floods_ahead", "floods_past", "gauges")
MIN_NOTES, MAX_NOTES = 3, 5
#: How far from the view's edges (a share of it) a note's pin keeps, where its card would run under the box,
#: the legend or the time bar.
PIN_MARGIN = 0.06
EARTH_KM = 6371.0088

# Where each fact comes from: the short line a note shows on the map and the full credit its card shows.
CREDITS: dict[str, dict[str, str]] = {
    "status": {"short": "GEOGLOWS v2 river status, CC BY 4.0", "full": f"{STATUS_ATTRIBUTION}, {STATUS_LICENCE}; "
               "modelled, one class per HydroBASINS level-4 basin"},
    "floods_ahead": {"short": "GEOGLOWS v2 forecast, CC BY 4.0", "full": (
        "Floods ahead: GEOGLOWS v2 forecast (CC BY 4.0) and return periods (CC BY-NC-SA 4.0), classed daily by "
        "AquaScope; model output, not an official warning")},
    "news": {"short": "Groundsource news events, CC BY 4.0",
             "full": "Flood events from news: Groundsource (Mayo et al. 2026, Google), CC BY 4.0"},
    "radar": {"short": "Sentinel-1 via Microsoft AI for Good, MIT",
              "full": "Sentinel-1 flood detections: Microsoft AI for Good Lab (Misra et al. 2025), MIT"},
    "gauges": {"short": "gauge record", "full": "today against normal for the date, from the gauge's own record "
               "in the AquaScope Archive"},
}

#: The status classes as the Explorer paints them (explorer/src/now-core.js STATUS_CLASSES, the same five
#: colours as the gauges): the hex there and the words a note uses for it. tests/test_map_explain.py keeps
#: the hexes equal to the page's.
CLASS_COLOURS: dict[str, tuple[str, str]] = {
    "much_below": ("#a6611a", "dark brown"),
    "below": ("#dfc27d", "tan"),
    "normal": ("#7b8794", "grey"),
    "above": ("#80cdc1", "light teal"),
    "much_above": ("#018571", "dark teal"),
}
# Each class's bounds in the words a note uses, from map_layers.STATUS_CLASSES' percentiles.
_CLASS_BOUND = {c["id"]: (f"below the {c['below_pct']}th percentile" if "from_pct" not in c
                          else f"above the {c['from_pct']}th percentile" if "below_pct" not in c
                          else f"in the {c['from_pct']}th to {c['below_pct']}th percentile")
                for c in STATUS_CLASSES}
_CLASS_LABEL = {c["id"]: c["label"] for c in STATUS_CLASSES}
_CLASS_IDS = [c["id"] for c in STATUS_CLASSES]

# Floods ahead: one character per day of a reach's `daily`, the index into this list
# (aquascope.archive.warnings.DAILY_CODES), and the class colours of explorer/src/floods-ahead-core.js.
_DAILY_CODES = (0, 2, 5, 10, 25, 50, 100)
FLOOD_COLOURS = {2: "#f2c94c", 5: "#f2994a", 10: "#e2553f", 25: "#c2185b", 50: "#8e24aa", 100: "#4a148c"}
NEWS_COLOUR, RADAR_COLOUR = "#e8820c", "#8b5cf6"
FORECAST_DAYS = 15

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

PHOTON_REVERSE_URL = "https://photon.komoot.io/reverse"
PLACE_CREDIT = "Places: Photon by komoot, data © OpenStreetMap contributors (ODbL)"


# ── formatting: the one place a number becomes words ─────────────────────────


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _sig(x: float, digits: int = 2) -> float:
    """``x`` to ``digits`` significant figures (a flow of 4,312.6 m3/s reads as 4,300)."""
    if not x:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + digits - 1)


def _count(n: float) -> str:
    return f"{int(round(n)):,}"


def _flow(q: float) -> str:
    q = float(q)
    v = _sig(q, 3) if q >= 100 else _sig(q, 2)
    return f"{v:,.0f}" if v >= 10 else f"{v:,.1f}".rstrip("0").rstrip(".")


def _pct(share: float) -> str:
    p = 100 * float(share)
    return "under 1%" if 0 < p < 1 else f"{round(p)}%"


def _month_words(month: str) -> str:
    return f"{_MONTHS[int(month[5:7]) - 1]} {month[:4]}"


def _short_month(month: str) -> str:
    return f"{_MONTHS[int(month[5:7]) - 1][:3]} {month[:4]}"


def _span_words(months: list[str]) -> str:
    if not months:
        return ""
    if len(months) == 1:
        return _short_month(months[0])
    return f"{_short_month(months[0])} to {_short_month(months[-1])}"


def _day_words(iso: str, weekday: bool = False) -> str:
    try:
        d = date.fromisoformat(str(iso)[:10])
    except ValueError:
        return str(iso)
    out = f"{d.day} {_MONTHS[d.month - 1][:3]}"
    return f"{_DAYS[d.weekday()]} {out}" if weekday else out


def _fact(kind: str, layer: str, label: str, value: Any, unit: str, shown: str, *, place: dict[str, Any] | None = None,
          when: str = "", source: str = "", tone: str = "", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One fact. ``shown`` is the value as every note writes it (code made it, from ``value``)."""
    out = {"kind": kind, "layer": layer, "label": label, "value": value, "unit": unit, "shown": shown,
           "place": place, "when": when, "source": source or layer, "tone": tone}
    if extra:
        out.update(extra)
    return out


# ── the view ─────────────────────────────────────────────────────────────────


def _check_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox is [west, south, east, north] in degrees")
    w, s, e, n = (float(x) for x in bbox)
    if not all(math.isfinite(v) for v in (w, s, e, n)):
        raise ValueError("bbox holds a number that is not finite")
    s, n = max(-90.0, min(s, n)), min(90.0, max(s, n))
    if s >= n:
        raise ValueError("bbox has no height")
    if e - w >= 360 or e <= w - 360:
        return [-180.0, s, 180.0, n]
    # A map that has been panned round the world gives longitudes past 180; west > east crosses the antimeridian.
    width = (e - w) % 360 or 360.0
    w = ((w + 180) % 360) - 180
    e = w + width
    if e > 180:
        e -= 360
    return [round(w, 6), s, round(e, 6), n]


def _in_bbox(lat: float, lon: float, bbox: list[float]) -> bool:
    w, s, e, n = bbox
    if not s <= lat <= n:
        return False
    return w <= lon <= e if w <= e else (lon >= w or lon <= e)


def _check_disc(disc: Any) -> dict[str, float] | None:
    """The part of a globe that faces the reader, ``{"lat", "lon", "deg"}`` (its centre and how far from it, in
    degrees of arc, a point can still be read), or None for a flat map."""
    if not disc:
        return None
    lat, lon, deg = (float(disc[k]) for k in ("lat", "lon", "deg"))
    if not all(math.isfinite(v) for v in (lat, lon, deg)) or not 0 < deg <= 90:
        raise ValueError("disc is {lat, lon, deg} with deg between 0 and 90")
    return {"lat": lat, "lon": lon, "deg": deg}


def _arc(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Degrees of arc between two points."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _in_view(lat: float, lon: float, bbox: list[float], disc: dict[str, float] | None = None) -> bool:
    if not _in_bbox(lat, lon, bbox):
        return False
    return disc is None or _arc(lat, lon, disc["lat"], disc["lon"]) <= disc["deg"]


def _span(bbox: list[float]) -> tuple[float, float]:
    w, s, e, n = bbox
    return ((e - w) if w <= e else (e - w + 360)), n - s


# ── the world river status ──────────────────────────────────────────────────


def status_sample(data: bytes, bbox: list[float], max_cells: int = 360) -> dict[str, Any]:
    """The status file's classes inside ``bbox``, every cell or every k-th cell so that neither side passes
    ``max_cells``: ``{"classes", "lats", "lons", "deg"}`` (row-centre latitudes, column-centre longitudes,
    the sampled cell size in degrees). Only the file's tiles under the view are decompressed."""
    import numpy as np

    from aquascope.map_layers import status_classes, status_shape

    bbox = _check_bbox(bbox)
    height, width = status_shape(data)
    deg = 180.0 / height
    w, s, e, n = bbox
    r0 = max(0, int(math.ceil((90 - n) / deg - 0.5)))
    r1 = min(height - 1, int(math.floor((90 - s) / deg - 0.5)))
    c0 = int(math.ceil((w + 180) / deg - 0.5)) % width
    c1 = int(math.floor((e + 180) / deg - 0.5)) % width
    if w <= e and c1 < c0:
        c1 = c0                      # a view narrower than one cell: that cell, not the whole way round
    ncols = (c1 - c0) % width + 1 if not (w == -180 and e == 180) else width
    if r1 < r0:
        r1 = r0
    stride = max(1, math.ceil(max(r1 - r0 + 1, ncols) / max_cells))
    rows = np.arange(r0, r1 + 1, stride)
    cols = (c0 + np.arange(0, ncols, stride)) % width
    classes = status_classes(data, rows, cols)
    return {"classes": classes, "lats": 90 - (rows + 0.5) * deg, "lons": -180 + (cols + 0.5) * deg,
            "deg": deg * stride}


def _patches(mask: Any) -> list[list[tuple[int, int]]]:
    """The 4-connected groups of True cells, largest first."""
    h, w = mask.shape
    seen = [[False] * w for _ in range(h)]
    cells = mask.tolist()
    out = []
    for r in range(h):
        row = cells[r]
        for c in range(w):
            if not row[c] or seen[r][c]:
                continue
            seen[r][c] = True
            stack, group = [(r, c)], []
            while stack:
                y, x = stack.pop()
                group.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and cells[yy][xx] and not seen[yy][xx]:
                        seen[yy][xx] = True
                        stack.append((yy, xx))
            out.append(group)
    out.sort(key=len, reverse=True)
    return out


def _inside_cells(shape: tuple[int, int], rr: Any, cc: Any, a: Any, k: int = 3, ok: Any = None) -> list[int]:
    """Cells of an area well inside it, so a pin lands on the colour it describes rather than on an edge: the
    one with the most of the area around it (a square a twentieth of the view wide), the nearest to the
    area's middle among equals; then up to ``k - 1`` more as far from the others, a sixth of the view apart,
    for a note to move to when another note already sits on the first. ``ok`` (a boolean per cell of the area)
    keeps the spots away from the view's edges, where they can be."""
    import numpy as np

    h, w = shape
    mask = np.zeros((h + 1, w + 1), dtype=np.int64)
    mask[rr + 1, cc + 1] = 1
    integral = mask.cumsum(0).cumsum(1)
    r = max(1, max(h, w) // 40)
    y0, y1 = np.clip(rr - r, 0, h), np.clip(rr + r + 1, 0, h)
    x0, x1 = np.clip(cc - r, 0, w), np.clip(cc + r + 1, 0, w)
    around = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    size = float(a.sum())
    my, mx = float((rr * a).sum() / size), float((cc * a).sum() / size)
    near = (rr - my) ** 2 + (cc - mx) ** 2
    allowed = np.ones(len(rr), dtype=bool) if ok is None or not np.any(ok) else np.asarray(ok, dtype=bool)
    order = np.lexsort((near, -around, ~allowed))
    full = (around >= around[order[0]] * 0.8) & allowed
    chosen = [int(order[0])]
    apart = (max(h, w) / 6) ** 2
    for j in order[1:]:
        if len(chosen) == k:
            break
        j = int(j)
        if full[j] and all((rr[j] - rr[i]) ** 2 + (cc[j] - cc[i]) ** 2 >= apart for i in chosen):
            chosen.append(j)
    return chosen


def _status_facts(sample: dict[str, Any], month: str, disc: dict[str, float] | None = None,
                  pin_ok: Any = None) -> list[dict[str, Any]]:
    """The share of the mapped land in view in each class, and for each class but normal the largest connected
    area (its share and size, at spots well inside it). On a globe, only the side facing the reader."""
    import numpy as np

    classes, lats, lons, deg = sample["classes"], sample["lats"], sample["lons"], sample["deg"]
    if disc is not None:
        p1, l1 = np.radians(disc["lat"]), np.radians(disc["lon"])
        p2, l2 = np.radians(lats)[:, None], np.radians(lons)[None, :]
        cos_arc = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(l2 - l1)
        classes = np.where(cos_arc >= math.cos(math.radians(disc["deg"])), classes, 0)
    half = math.radians(deg) / 2
    # Each sampled cell's area in km2: a band of latitude times the cell's width.
    band = np.abs(np.sin(np.radians(lats) + half) - np.sin(np.radians(lats) - half))
    area = (EARTH_KM ** 2) * math.radians(deg) * band          # per row
    weights = np.repeat(area[:, None], classes.shape[1], axis=1)
    mapped = float(weights[classes > 0].sum())
    if mapped <= 0:
        return []
    when = _month_words(month)
    src = "status"
    facts = []
    share = {k: float(weights[classes == i + 1].sum()) / mapped for i, k in enumerate(_CLASS_IDS)}
    below, above = share["much_below"] + share["below"], share["above"] + share["much_above"]
    facts.append(_fact("status_share", "status", "Mapped land in view below or much below normal",
                       round(100 * below, 1), "%", _pct(below), when=when, source=src,
                       tone=CLASS_COLOURS["much_below"][0], extra={"side": "below"}))
    facts.append(_fact("status_share", "status", "Mapped land in view above or much above normal",
                       round(100 * above, 1), "%", _pct(above), when=when, source=src,
                       tone=CLASS_COLOURS["much_above"][0], extra={"side": "above"}))
    for k in ("much_above", "much_below", "above", "below"):
        idx = _CLASS_IDS.index(k) + 1
        groups = _patches(classes == idx)
        if not groups:
            continue
        g = groups[0]
        rr = np.array([p[0] for p in g])
        cc = np.array([p[1] for p in g])
        a = area[rr]
        size = float(a.sum())
        part = size / mapped
        if part < 0.02 or len(g) < 4:
            continue
        ok = [pin_ok(float(lats[y]), float(lons[x])) for y, x in zip(rr, cc)] if pin_ok else None
        spots = [{"lat": round(float(lats[rr[j]]), 4), "lon": round(float(lons[cc[j]]), 4)}
                 for j in _inside_cells(classes.shape, rr, cc, a, ok=ok)]
        # Spots under the legend or at the edge are no place for a pin; the area still counts.
        pinnable = [p for p in spots if not pin_ok or pin_ok(p["lat"], p["lon"])]
        place = (pinnable or spots)[0]
        spots = pinnable
        colour, words = CLASS_COLOURS[k]
        facts.append(_fact("status_class", "status", f"{_CLASS_LABEL[k].capitalize()} means",
                           _CLASS_BOUND[k], "", _CLASS_BOUND[k], when=when, source=src, tone=colour,
                           extra={"class": k}))
        facts.append(_fact("status_area", "status", f"Largest area {_CLASS_LABEL[k]} in view", round(100 * part, 1),
                           "% of the mapped land", _pct(part), place=place, when=when, source=src, tone=colour,
                           extra={"class": k, "colour_words": words, "cells": len(g), "spots": spots[1:],
                                  "pinnable": bool(spots)}))
        facts.append(_fact("status_size", "status", "Its size", _sig(size, 2), "km²", f"{_sig(size, 2):,.0f} km²",
                           place=place, when=when, source=src, tone=colour, extra={"class": k}))
    return facts


# ── Floods ahead ────────────────────────────────────────────────────────────


def _class_on(props: dict[str, Any], i: int) -> int:
    if i < 0:
        return int(props.get("rp") or 0)
    s = str(props.get("daily") or "")
    if i >= len(s) or not s[i].isdigit():
        return 0
    code = int(s[i])
    return _DAILY_CODES[code] if code < len(_DAILY_CODES) else 0


def _day_index(issue: str | None, day: str | None) -> int:
    try:
        i = (date.fromisoformat(str(day)[:10]) - date.fromisoformat(str(issue)[:10])).days
    except (TypeError, ValueError):
        return -1
    return i if 0 <= i < FORECAST_DAYS else -1


def _floods_ahead_facts(warnings: dict[str, Any], bbox: list[float], day: str | None,
                        disc: dict[str, float] | None = None, pin_ok: Any = None) -> list[dict[str, Any]]:
    issue = warnings.get("issue_date")
    i = _day_index(issue, day)
    reaches = []
    for f in warnings.get("features") or warnings.get("reaches") or []:
        p = dict(f.get("properties") or f)
        geom = f.get("geometry") or {}
        if geom.get("coordinates"):
            p.setdefault("lon", geom["coordinates"][0])
            p.setdefault("lat", geom["coordinates"][1])
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not _in_view(lat, lon, bbox, disc):
            continue
        rp = _class_on(p, i)
        if rp >= 2:
            reaches.append({**p, "lat": lat, "lon": lon, "cls": rp})
    if not reaches:
        return []
    when = f"on {_day_words(day, weekday=True)}" if i >= 0 else f"in the 15 days from {_day_words(issue)}"
    # The browser's file holds only the strongest reaches when the day is busy: then the count is a floor.
    cut = bool(warnings.get("truncated"))
    shown = f"at least {_count(len(reaches))}" if cut else _count(len(reaches))
    facts = [_fact("fa_count", "floods_ahead", "Reaches in view at their 2-year flow or more", len(reaches),
                   "reaches", shown, when=when, source="floods_ahead",
                   tone=FLOOD_COLOURS[2], extra={"issue_date": issue, "floor": cut})]

    def ratio(r: dict[str, Any]) -> float:
        peak = r.get("peak") if r.get("peak") is not None else r.get("peak_cms")
        q2 = r.get("q2")
        return float(peak) / float(q2) if peak and q2 else 0.0

    reaches.sort(key=lambda r: (-r["cls"], -ratio(r)))
    chosen: list[dict[str, Any]] = []
    for r in reaches:
        if len(chosen) == 3:
            break
        if pin_ok and not pin_ok(r["lat"], r["lon"]):
            continue
        if _spaced(chosen, r["lat"], r["lon"], bbox, 0.15):
            chosen.append(r)
    for rank, r in enumerate(chosen):
        place = {"lat": round(r["lat"], 4), "lon": round(r["lon"], 4)}
        tone = FLOOD_COLOURS.get(int(r["cls"]), FLOOD_COLOURS[2])
        extra = {"river_id": r.get("river_id"), "rank": rank, "issue_date": issue}
        facts.append(_fact("fa_class", "floods_ahead", "Forecast flow class", int(r["cls"]), "year flow",
                           f"{int(r['cls'])}-year flow", place=place, when=when, source="floods_ahead", tone=tone,
                           extra=extra))
        peak = r.get("peak") if r.get("peak") is not None else r.get("peak_cms")
        if peak:
            peak_day = r.get("day") or r.get("peak_date") or ""
            facts.append(_fact("fa_peak", "floods_ahead", "Forecast peak (ensemble mean)", _sig(float(peak), 3),
                               "m³/s", f"{_flow(peak)} m³/s", place=place,
                               when=f"on {_day_words(peak_day, weekday=True)}" if peak_day else "",
                               source="floods_ahead", tone=tone, extra=extra))
        x = ratio(r)
        if x:
            facts.append(_fact("fa_ratio", "floods_ahead", "Peak against the 2-year flow", round(x, 1), "times",
                               f"{x:.1f} times", place=place, source="floods_ahead", tone=tone, extra=extra))
    return facts


# ── Floods past ─────────────────────────────────────────────────────────────


def _floods_past_facts(floods: dict[str, Any], bbox: list[float],
                       disc: dict[str, float] | None = None, pin_ok: Any = None) -> list[dict[str, Any]]:
    from aquascope.context.floods_past import GRID_DEG, cell_centre

    deg = float(floods.get("deg") or GRID_DEG)
    months = [str(m) for m in floods.get("months") or []]
    span = _span_words(months)
    acc: dict[tuple[int, int], list[int]] = {}
    for cell in floods.get("cells") or []:
        row, col, news, radar = (int(x) for x in cell[:4])
        lat, lon = cell_centre(row, col, deg)
        if not _in_view(lat, lon, bbox, disc):
            continue
        a = acc.setdefault((row, col), [0, 0])
        a[0] += news
        a[1] += radar
    if not acc:
        return []
    facts = []
    for k, (kind, label, unit) in enumerate((("news", "Flood events in the news", "events"),
                                             ("radar", "Radar flood detections", "detections"))):
        total = sum(v[k] for v in acc.values())
        if not total:
            continue
        tone = NEWS_COLOUR if kind == "news" else RADAR_COLOUR
        facts.append(_fact(f"fp_{kind}_total", "floods_past", f"{label} in view", total, unit, _count(total),
                           when=span, source=kind, tone=tone))
        ranked = sorted(((v[k], -rc[0], -rc[1], rc) for rc, v in acc.items() if v[k]), reverse=True)
        chosen: list[dict[str, Any]] = []
        for count, _r, _c, (row, col) in ranked:
            if len(chosen) == (4 if kind == "news" else 2):
                break
            lat, lon = cell_centre(row, col, deg)
            if (pin_ok and not pin_ok(lat, lon)) or not _spaced(chosen, lat, lon, bbox, 0.12):
                continue
            chosen.append({"lat": lat, "lon": lon})
            facts.append(_fact(f"fp_{kind}_cell", "floods_past", f"{label} in this half-degree cell", count, unit,
                               _count(count), place={"lat": lat, "lon": lon}, when=span, source=kind, tone=tone,
                               extra={"cell": [row, col], "rank": len(chosen) - 1}))
    return facts


# ── gauges, today against normal ────────────────────────────────────────────


def _gauge_facts(gauges: list[dict[str, Any]], bbox: list[float],
                 disc: dict[str, float] | None = None, pin_ok: Any = None) -> list[dict[str, Any]]:
    rows = []
    for g in gauges or []:
        try:
            lat, lon, pct = float(g["lat"]), float(g["lon"]), float(g["pct"] if "pct" in g else g["percentile"])
        except (KeyError, TypeError, ValueError):
            continue
        cls = g.get("cls") or g.get("class")
        if cls not in _CLASS_IDS or not _in_view(lat, lon, bbox, disc):
            continue
        rows.append({**g, "lat": lat, "lon": lon, "pct": pct, "cls": cls})
    if not rows:
        return []
    facts = []
    many = {}
    for side, classes in (("above", ("much_above",)), ("below", ("much_below",))):
        n = many[side] = sum(r["cls"] in classes for r in rows)
        if n:
            facts.append(_fact("gauge_count", "gauges", f"Gauges in view {_CLASS_LABEL[classes[0]]} today", n,
                               "gauges", _count(n), source="gauges:", tone=CLASS_COLOURS[classes[0]][0],
                               extra={"side": side, "of": len(rows)}))
    # The side with more gauges far from normal chooses first; the other side's gauge keeps clear of it, so
    # two notes do not fight for one spot while the view holds many such gauges elsewhere.
    chosen: list[dict[str, Any]] = []
    for side in sorted(("above", "below"), key=lambda x: -many[x]):
        wanted = ("much_above", "above") if side == "above" else ("much_below", "below")
        pool = [r for r in rows if r["cls"] in wanted and (not pin_ok or pin_ok(r["lat"], r["lon"]))]
        if not pool:
            continue
        # Furthest from normal first; among equals, the longer record.
        pool.sort(key=lambda r: ((-r["pct"] if side == "above" else r["pct"]), -float(r.get("n_years") or 0)))
        g = next((r for r in pool if _spaced(chosen, r["lat"], r["lon"], bbox, 0.15)), pool[0])
        chosen.append(g)
        agency = str(g.get("agency") or g.get("source") or "")
        key = str(g.get("source") or "")
        place = {"lat": round(g["lat"], 4), "lon": round(g["lon"], 4),
                 "name": tidy_name(str(g.get("name") or g.get("station_id"))),
                 "station": f"{agency} {g.get('station_id')}".strip()}
        extra = {"side": side, "class": g["cls"], "station": f"{g.get('source')}/{g.get('station_id')}",
                 "agency": agency}
        tone = CLASS_COLOURS[g["cls"]][0]
        on_day = g.get("date") or g.get("value_date")
        when = f"on {_day_words(on_day)}" if on_day else ""
        pct = int(round(g["pct"]))
        facts.append(_fact("gauge_pct", "gauges", "Today's flow against the same date in other years", pct,
                           "percentile", f"{_ordinal(pct)} percentile", place=place, when=when,
                           source=f"gauges:{key}", tone=tone, extra=extra))
        if g.get("n_years"):
            facts.append(_fact("gauge_years", "gauges", "Years of record behind it", int(g["n_years"]), "years",
                               f"{int(g['n_years'])} years", place=place, source=f"gauges:{key}", tone=tone,
                               extra=extra))
        if g.get("value") is not None:
            facts.append(_fact("gauge_flow", "gauges", "Flow", _sig(float(g["value"]), 3), "m³/s",
                               f"{_flow(g['value'])} m³/s", place=place, when=when, source=f"gauges:{key}",
                               tone=tone, extra=extra))
    return facts


_SMALL_WORDS = frozenset("a an and at above below by de del des du for in la le near nr of on the to".split())


def tidy_name(name: str) -> str:
    """A gauge name in capitals ("VALATIE KILL NEAR NASSAU NY") as a reader writes it ("Valatie Kill near
    Nassau NY"): small words lower case, a closing state code and anything with a digit as they are. A name
    already in mixed case is left alone."""
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    letters = [c for c in name if c.isalpha()]
    if not letters or any(c.islower() for c in letters):
        return name
    words = name.split(" ")
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"[^A-Za-z]", "", w)
        last = i == len(words) - 1
        if any(c.isdigit() for c in w) or (last and len(core) == 2):
            out.append(w)
        elif i and core.lower() in _SMALL_WORDS:
            out.append(w.lower())
        else:
            out.append(re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), w))
    return " ".join(out)


# ── place names ─────────────────────────────────────────────────────────────


def _photon_reverse(lat: float, lon: float, radius_km: float = 50, timeout: float = 5.0) -> list[dict[str, Any]]:
    from aquascope import __version__

    # Photon looks only 1 km round the point unless told otherwise; a pin in the middle of a basin needs more.
    query = {"lat": round(lat, 4), "lon": round(lon, 4), "lang": "en", "limit": 1, "radius": radius_km}
    url = f"{PHOTON_REVERSE_URL}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": f"aquascope/{__version__} (explain the map)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - a fixed https host
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("features") or [])


def place_name(lat: float, lon: float, zoom: float = 5.0, fetch: Any = None, *, strict: bool = False) -> str:
    """A short name for a point from the gazetteer the search uses (Photon, OpenStreetMap data): the state
    and country far out, the town or county and state close in. Empty when there is nothing there (the sea)
    or the gazetteer cannot be reached (``strict`` raises then instead). Never a digit: a name with one (a
    postcode) is left out."""
    try:
        feats = (fetch or _photon_reverse)(lat, lon)
    except Exception:  # noqa: BLE001 - a name is a nicety; the pin stands without one
        if strict:
            raise
        return ""
    if not feats:
        return ""
    p = feats[0].get("properties") or {}

    def first(*keys: str) -> str:
        for k in keys:
            v = str(p.get(k) or "").strip()
            if v and not re.search(r"\d", v):
                return v
        return ""

    parts = [first("city", "county", "district"), first("state", "country")] if zoom >= 7 else \
        [first("state", "county"), first("country")]
    return ", ".join(dict.fromkeys(x for x in parts if x))


# ── the facts ───────────────────────────────────────────────────────────────


def view_facts(view: dict[str, Any], *, status: dict[str, Any] | None = None, warnings: dict[str, Any] | None = None,
               floods_past: dict[str, Any] | None = None, gauges: list[dict[str, Any]] | None = None
               ) -> dict[str, Any]:
    """The facts in one view of the map, extracted by code.

    ``view`` is ``{"bbox": [w, s, e, n], "date": "YYYY-MM-DD", "layers": [...], "zoom": z}``, and on a globe
    seen whole ``"disc": {"lat", "lon", "deg"}``, the side facing the reader. ``"avoid"`` lists boxes (the
    legend, the time bar) where no note may be pinned, though what is under them still counts. ``layers``
    says which of :data:`LAYERS` are on (a layer that is off gives no fact even when its data is passed).
    The data:

    * ``status``: ``{"month": "YYYY-MM", "data": <the month's GeoTIFF bytes>}`` (or ``"sample"`` from
      :func:`status_sample`);
    * ``warnings``: the Floods ahead issue, ``{"issue_date", "features": [GeoJSON points]}`` or
      ``{"issue_date", "reaches": [rows]}`` as :func:`aquascope.archive.warnings.flood_warnings` gives them,
      with ``"truncated": true`` when the reaches are only the strongest of the issue (the browser's file);
    * ``floods_past``: ``{"months": [...], "cells": [[row, col, news, radar], ...]}`` on the half-degree grid;
    * ``gauges``: rows of today's status, ``{source, station_id, name, agency, lat, lon, cls, pct, date,
      value, n_years}``.

    Returns ``{"facts", "skipped", "view", "credits"}``. Each fact has an ``id`` (``f1``...), ``kind``,
    ``layer``, ``label``, ``value``, ``unit``, ``shown`` (the value in the words every note uses), ``place``
    (``{lat, lon, name}`` or None for the whole view), ``when`` and ``source``. Places are named later, only
    for the notes that use them (:func:`name_notes`): a gazetteer call each.
    """
    bbox = _check_bbox(view.get("bbox"))
    disc = _check_disc(view.get("disc"))
    avoid = [_check_bbox(b) for b in view.get("avoid") or []]

    def pin_ok(lat: float, lon: float) -> bool:
        # Where a note can be pinned: off the view's edges, off the map's furniture, and on a globe off its rim.
        if disc is not None and _arc(lat, lon, disc["lat"], disc["lon"]) > disc["deg"] * 0.92:
            return False
        return _inner(lat, lon, bbox, PIN_MARGIN) and not any(_in_bbox(lat, lon, b) for b in avoid)

    on = set(view.get("layers") or LAYERS)
    zoom = float(view.get("zoom") or 4.0)
    facts: list[dict[str, Any]] = []
    skipped: list[str] = []
    if "status" in on:
        if status and status.get("month") and (status.get("data") or status.get("sample")):
            sample = status.get("sample") or status_sample(status["data"], bbox)
            got = _status_facts(sample, str(status["month"]), disc, pin_ok)
            facts += got
            if not got:
                skipped.append("World river status: no mapped basin in view")
        else:
            skipped.append("World river status: no map for this date")
    if "floods_ahead" in on:
        got = _floods_ahead_facts(warnings, bbox, view.get("date"), disc, pin_ok) if warnings else []
        facts += got
        if not got:
            skipped.append("Floods ahead: no reach in view at its 2-year flow")
    if "gauges" in on:
        got = _gauge_facts(gauges or [], bbox, disc, pin_ok)
        facts += got
        if not got:
            skipped.append("Gauges: none in view with today's status")
    if "floods_past" in on:
        got = _floods_past_facts(floods_past, bbox, disc, pin_ok) if floods_past else []
        facts += got
        if not got:
            skipped.append("Floods past: no event in view")
    for i, f in enumerate(facts, 1):
        f["id"] = f"f{i}"
    credits = sorted({_credit(f["source"])["full"] for f in facts})
    if len(credits) > 1 and CREDITS["gauges"]["full"] in credits:
        # "USGS (US-PD): today against normal ..." already says what the bare gauges credit says.
        named = [c for c in credits if c != CREDITS["gauges"]["full"] and c.endswith(CREDITS["gauges"]["full"])]
        if named:
            credits.remove(CREDITS["gauges"]["full"])
    return {"facts": facts, "skipped": skipped, "credits": credits,
            "view": {"bbox": bbox, "disc": disc, "avoid": avoid, "date": view.get("date"),
                     "layers": sorted(on & set(LAYERS)), "zoom": zoom}}


def _credit(source: str) -> dict[str, str]:
    if source.startswith("gauges:"):
        key = source.split(":", 1)[1]
        if not key:
            return dict(CREDITS["gauges"])
        from aquascope.registry import SOURCES

        spec = SOURCES.get(key)
        label, licence = (spec.label, spec.license) if spec else (key, "")
        terms = f", {licence}" if licence else ""
        return {"short": f"{label} gauge record{terms}",
                "full": f"{label}{f' ({licence})' if licence else ''}: {CREDITS['gauges']['full']}"}
    return CREDITS.get(source, {"short": source, "full": source})


# ── notes ───────────────────────────────────────────────────────────────────


def _by_id(facts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in facts}


def _anchor(facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((f for f in facts if f.get("place")), None)


def _spots(fact: dict[str, Any]) -> list[dict[str, Any]]:
    """Where a note on this fact may be pinned: its place, then (a large area) other spots inside it."""
    return [fact["place"], *(fact.get("spots") or [])]


def _note(facts: list[dict[str, Any]], title: str, text: str, place: dict[str, Any] | None = None) -> dict[str, Any]:
    """A note as the page pins it: at the first cited fact with a place (or ``place``, one of its spots),
    listing every cited fact and source."""
    anchor = _anchor(facts)
    p = place or anchor["place"]
    named = {f["source"] for f in facts if f["source"].startswith("gauges:") and f["source"] != "gauges:"}
    sources = []
    for f in facts:
        if f["source"] == "gauges:" and named:
            continue   # "USGS (US) gauge record" says it already
        c = _credit(f["source"])
        if c not in sources:
            sources.append(c)
    # A gauge's note is titled with its name; its card's second line says whose gauge it is.
    where = p.get("station") or p.get("name") or ""
    return {
        "title": title, "text": text, "lat": p["lat"], "lon": p["lon"], "place": where,
        "when": anchor.get("when") or "", "layer": anchor["layer"], "tone": anchor.get("tone") or "",
        "fact_ids": [f["id"] for f in facts],
        "facts": [{"label": f["label"], "value": f["shown"], **({"when": f["when"]} if f.get("when") else {})}
                  for f in facts],
        "source": "; ".join(c["short"] for c in sources), "credit": "; ".join(c["full"] for c in sources),
    }


def _first(facts: list[dict[str, Any]], kind: str, **match: Any) -> dict[str, Any] | None:
    return next((f for f in facts if f["kind"] == kind and all(f.get(k) == v for k, v in match.items())), None)


def _candidates(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every note the template can write: ``{prio, slot, facts, title, text}``, lower ``prio`` first. Notes
    that share a ``slot`` are alternatives (the next reach when the first one sits on another note's spot)."""
    out: list[dict[str, Any]] = []

    def add(prio: float, slot: str, cited: list[dict[str, Any] | None], title: str, text: str) -> None:
        out.append({"prio": prio, "slot": slot, "facts": [f for f in cited if f], "title": title, "text": text})

    # The world river status: what the colour means where it covers most, much above or below normal weighing
    # more, the side that covers more first.
    areas = [f for f in facts if f["kind"] == "status_area"]
    weight = {"much_above": 1.5, "much_below": 1.5, "above": 1.0, "below": 1.0}
    areas.sort(key=lambda f: -f["value"] * weight[f["class"]])
    sides = [("above" if a["class"] in ("above", "much_above") else "below") for a in areas]
    first_side = sides[0] if sides else "above"
    for a, side in zip(areas, sides):
        k = a["class"]
        bound = _first(facts, "status_class", **{"class": k})
        month = a["when"].split(" ")[0]
        text = (f"{a['colour_words'].capitalize()}: flow {bound['shown']} for {month}, "
                f"on {a['shown']} of the mapped land in view.")
        add(1 if side == first_side else 1.5, f"status_{side}",
            [a, bound, _first(facts, "status_size", **{"class": k}), _first(facts, "status_share", side=side)],
            _CLASS_LABEL[k].capitalize(), text)
    # Floods ahead: the reach forecast to pass the highest return-period flow (the next ones as alternatives).
    count = _first(facts, "fa_count")
    for c in [f for f in facts if f["kind"] == "fa_class"]:
        rank = c["rank"]
        peak, ratio = _first(facts, "fa_peak", rank=rank), _first(facts, "fa_ratio", rank=rank)
        bits = []
        if peak:
            bits.append(f"Forecast peak {peak['shown']} {peak['when']}".rstrip())
        if ratio:
            bits.append(f"{ratio['shown']} its 2-year flow")
        text = ", ".join(bits) + "." if bits else f"At its {c['shown']} {c['when']}."
        many = count and count["value"] > 1
        if many:
            shown = count["shown"]
            text += f" {shown[:1].upper()}{shown[1:]} reaches in view pass theirs."   # "At least 45 reaches"
        add(2 + 0.1 * rank, "floods_ahead", [c, peak, ratio, count if many else None],
            f"{c['shown'].capitalize()} ahead", text)
    # Gauges: the one furthest from normal on each side, the side with more gauges far from normal first; one
    # far out ranks before the forecast's reach.
    counts = {side: (_first(facts, "gauge_count", side=side) or {}).get("value", 0) for side in ("above", "below")}
    first_gauges = "below" if counts["below"] > counts["above"] else "above"
    for side in ("above", "below"):
        prio = 3.0 if side == first_gauges else 6.0
        pct = _first(facts, "gauge_pct", side=side)
        if not pct:
            continue
        years, flow = _first(facts, "gauge_years", side=side), _first(facts, "gauge_flow", side=side)
        count = _first(facts, "gauge_count", side=side)
        text = f"{_CLASS_LABEL[pct['class']].capitalize()} {pct['when']}: {pct['shown']}".rstrip()
        text += f" of {years['shown']}." if years else " for the date."
        many = count and count["value"] > 1
        if many:
            text += f" {count['shown']} gauges in view are {_CLASS_LABEL['much_' + side]}."
        name = pct["place"].get("name") or "A gauge"
        add(prio - (1.5 if abs(pct["value"] - 50) >= 45 else 0), f"gauge_{side}",
            [pct, flow, years, count if many else None], name[:48], text)
    # Floods past: where most were reported, and where radar saw the most water.
    for kind, prio, title in (("news", 4.0, "Flood reports"), ("radar", 7.0, "Water seen by radar")):
        total = _first(facts, f"fp_{kind}_total")
        for cell in [f for f in facts if f["kind"] == f"fp_{kind}_cell"]:
            what = "flood events in the news" if kind == "news" else "Sentinel-1 flood detections"
            text = f"{cell['shown']} {what} in this half-degree cell, {cell['when']}"
            more = total and total["value"] > cell["value"]
            text += f"; {total['shown']} in view." if more else "."
            add(prio + 0.1 * cell["rank"], f"floods_past_{kind}", [cell, total if more else None], title, text)
    out.sort(key=lambda c: c["prio"])
    return out


def _spaced(notes: list[dict[str, Any]], lat: float, lon: float, bbox: list[float], gap: float) -> bool:
    """Whether a pin at (lat, lon) keeps ``gap`` (a share of the view's size) from every pin already chosen."""
    wspan, hspan = _span(bbox)
    for n in notes:
        dx = abs(n["lon"] - lon)
        dx = min(dx, 360 - dx) / max(wspan, 1e-6)
        dy = abs(n["lat"] - lat) / max(hspan, 1e-6)
        if math.hypot(dx, dy) < gap:
            return False
    return True


def _inner(lat: float, lon: float, bbox: list[float], margin: float) -> bool:
    """Whether a point keeps ``margin`` (a share of the view) from the view's edges, where a note's card would
    run under the box, the legend or the time bar."""
    w, s, _e, n = bbox
    wspan, hspan = _span(bbox)
    x = ((lon - w) % 360) / max(wspan, 1e-6)
    y = (lat - s) / max(hspan, 1e-6)
    return margin <= x <= 1 - margin and margin <= y <= 1 - margin


def template_notes(facts: list[dict[str, Any]], bbox: list[float] | None = None, max_notes: int = MAX_NOTES,
                   gap: float = 0.14, margin: float = PIN_MARGIN) -> list[dict[str, Any]]:
    """Three to five notes from the facts with fixed wording (no model, no key): what the status colour means
    where it covers most, the reach forecast to pass the highest return-period flow, the gauges furthest
    from normal and where most floods were reported. Fewer than three when the view holds fewer things worth
    a note.

    Pins keep ``gap`` (a share of the view) apart, so two notes never sit on one spot, and ``margin`` from the
    view's edges. The most important
    notes are chosen first; among them, the ones tied to one point (a reach, a gauge, a cell) are placed
    before the ones that can move (a large status area has other spots inside it)."""
    bbox = _check_bbox(bbox) if bbox else [-180.0, -85.0, 180.0, 85.0]
    cands = _candidates(facts)
    slots: list[str] = []
    cands = [c for c in cands if _anchor(c["facts"]) and _anchor(c["facts"]).get("pinnable", True)]
    for c in cands:
        if c["slot"] not in slots:
            slots.append(c["slot"])
    placed: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def try_slot(slot: str, movable: bool | None) -> None:
        for c in (x for x in cands if x["slot"] == slot):
            spots = _spots(_anchor(c["facts"]))
            if movable is not None and (len(spots) > 1) != movable:
                return
            taken = [p for _c, p in placed.values()]
            for spot in spots:
                lat, lon = spot["lat"], spot["lon"]
                if _inner(lat, lon, bbox, margin) and _spaced(taken, lat, lon, bbox, gap):
                    placed[slot] = (c, spot)
                    return

    top = slots[:max_notes]
    for movable in (False, True):
        for slot in top:
            if slot not in placed:
                try_slot(slot, movable)
    for slot in slots:
        if len(placed) >= max_notes:
            break
        if slot not in placed:
            try_slot(slot, None)
    chosen = sorted(placed.values(), key=lambda cp: cp[0]["prio"])[:max_notes]
    return [{**_note(c["facts"], c["title"], c["text"], spot), "by": "rules"} for c, spot in chosen]


_NAMES: dict[tuple[float, float, bool], str] = {}


def name_notes(notes: list[dict[str, Any]], zoom: float = 5.0, fetch: Any = None) -> list[dict[str, Any]]:
    """Name each note's place with the gazetteer (one call per place, remembered), in place; a gauge keeps its
    own name. The first call that fails ends the naming: the notes stand without names rather than wait on a
    gazetteer that is down or busy. Returns the notes."""
    for n in notes:
        if n.get("place"):
            continue
        key = (round(n["lat"], 2), round(n["lon"], 2), zoom >= 7)
        if key not in _NAMES:
            try:
                _NAMES[key] = place_name(n["lat"], n["lon"], zoom, fetch, strict=True)
            except Exception:  # noqa: BLE001 - see above
                break
        n["place"] = _NAMES[key]
    return notes


def explain(view: dict[str, Any], *, status: dict[str, Any] | None = None, warnings: dict[str, Any] | None = None,
            floods_past: dict[str, Any] | None = None, gauges: list[dict[str, Any]] | None = None,
            name_places: bool = True, fetch: Any = None, context: str = "") -> dict[str, Any]:
    """:func:`view_facts`, then :func:`template_notes` with their places named, and the prompt a model would
    word them from (:func:`notes_prompt`): everything the page needs for one "what am I seeing?"."""
    res = view_facts(view, status=status, warnings=warnings, floods_past=floods_past, gauges=gauges)
    notes = template_notes(res["facts"], res["view"]["bbox"])
    if name_places:
        name_notes(notes, res["view"]["zoom"], fetch)
        if any(n.get("place") and n["layer"] != "gauges" for n in notes):
            res["credits"].append(PLACE_CREDIT)
    res["notes"] = notes
    res["prompt"] = notes_prompt(res["facts"], context) if res["facts"] else None
    return res


# ── the model path, claim-locked ────────────────────────────────────────────

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_PLACEHOLDER = re.compile(r"\{(f\d+)(?:\.(when|place))?\}")
# A number in words is a number too ("twelve gauges", "half the land"). "One" is left out: it is mostly a
# pronoun ("the one upstream").
_NUMBER_WORDS = re.compile(
    r"\b(?:zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen"
    r"|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|thousands?"
    r"|millions?|billions?|dozens?|half|halves|quarters?|twice|double[ds]?|triple[ds]?|thrice|tenfold"
    r"|hundredfold)\b", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    text = text or ""
    out = {m.replace(",", "") for m in _NUMBER.findall(text)}
    out |= {m.lower() for m in _NUMBER_WORDS.findall(text)}
    # Any other character that stands for a number: a fraction (½), a Roman numeral (Ⅻ), a superscript.
    out |= {c for c in text if not "0" <= c <= "9" and unicodedata.numeric(c, None) is not None}
    return out


# Any placeholder: Explain's and mesh review's {f3} and {f3.when}, Scout's {peak}.
_ANY_SLOT = r"\{\w+(?:\.\w+)?\}"
# A sign, a digit or a decimal point set against a placeholder changes the number code puts in: "-{f1}" turns
# 3.2 into -3.2, "{f1}0" turns 3 into 30, "{f1}{f2}" glues two values into one. A hyphen that joins a word
# ("mid-{month}") or two placeholders (a range, "{f1}-{f2}") and a spaced dash ("Floods ahead - {place}")
# are wording, not a sign.
_TOUCH = re.compile(
    r"(?:[+\u2212\u00b1]\s?"                    # a plus, a minus sign or a plus-minus, touching or one space off
    r"|(?<![^\W\d_]|\})[-\u2013\u2014]"            # a hyphen or a dash (en, em) not joining words or placeholders
    r"|(?:^|(?<=[(\[=]))\s*[-\u2013\u2014]\s+"      # "- {f1}" opening the words or a bracket
    r"|\b(?:minus|negative|plus)\s+"             # the sign in words
    r"|\d[.,]?|(?<!\w)\.)"                       # a digit or a decimal point before it ("1{f1}", ".{f1}")
    + _ANY_SLOT
    + r"|" + _ANY_SLOT
    + r"(?:\d|[.,]\d|e[-+]?\d|k\b|bn\b|" + _ANY_SLOT + ")",   # after: "{f1}0", "{f1}.5", "{f1}e3", "{f1}k", "{f1}{f2}"
    re.IGNORECASE)


def placeholder_touches(words: str) -> list[str]:
    """The places in ``words`` where a sign, a digit or a decimal point is set against a placeholder ("-{f1}",
    "minus {f1}", "{f1}0", "{f1}{f2}"): each would change the number code fills in. Empty means none. The one
    check Explain, Scout and mesh review all make."""
    return [m.group(0).strip() for m in _TOUCH.finditer(words or "")]


def claim_lock(words: str, cited: list[dict[str, Any]]) -> list[str]:
    """The numbers in a model's ``words`` (placeholders taken out) that are not in the facts it cites: the
    value as shown, its date, its place and its label. Digits, numbers in words ("twelve", "half") and other
    numeric characters ("½") all count, and so does a sign or a digit set against a placeholder
    (:func:`placeholder_touches`, listed as written, "-{f1}"). Empty means the words pass."""
    allowed: set[str] = set()
    for f in cited:
        # The label too: "Peak against the 2-year flow" is how code named the fact.
        for part in (f.get("shown"), f.get("when"), (f.get("place") or {}).get("name"), f.get("label")):
            allowed |= _numbers(str(part or ""))
    bare = _PLACEHOLDER.sub(" ", words or "")
    return sorted(n for n in _numbers(bare) if n not in allowed) + placeholder_touches(words)


def _part(fact: dict[str, Any], part: str | None) -> str:
    """What a placeholder puts in: the fact's value as shown, its date (``when``) or its place's name."""
    if part == "when":
        return str(fact.get("when") or "")
    if part == "place":
        return str((fact.get("place") or {}).get("name") or "")
    return str(fact["shown"])


def _fill(words: str, facts: dict[str, dict[str, Any]]) -> str:
    def one(m: re.Match[str]) -> str:
        return _part(facts[m.group(1)], m.group(2))

    out = re.sub(r"\s+", " ", _PLACEHOLDER.sub(one, words or "")).strip()
    # A value that opens a sentence ("at least 45") starts it with a capital.
    return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)


def notes_schema() -> dict[str, Any]:
    """The JSON Schema a model's notes are held to."""
    return {
        "type": "object", "required": ["notes"], "additionalProperties": False,
        "properties": {"notes": {"type": "array", "minItems": 1, "maxItems": MAX_NOTES, "items": {
            "type": "object", "required": ["facts", "title", "text"], "additionalProperties": False,
            "properties": {
                "facts": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}},
                "title": {"type": "string", "maxLength": 60},
                "text": {"type": "string", "maxLength": 240},
            }}}},
    }


def notes_prompt(facts: list[dict[str, Any]], context: str = "") -> dict[str, Any]:
    """The system prompt and reply schema for a model that writes the notes. The facts go in as a list with
    their values, so the model can tell which matter; it writes each value as the fact's placeholder, which
    code fills in."""
    lines = []
    for f in facts:
        name = (f.get("place") or {}).get("name")
        where = f"{name} (write {{{f['id']}.place}})" if name else \
            ("a point on the map" if f.get("place") else "the whole view")
        when = f" when {f['when']} (write {{{f['id']}.when}})" if f.get("when") else ""
        lines.append(f"{f['id']}: {f['label']} is {f['shown']} (write {{{f['id']}}}){when}; "
                     f"layer {f['layer']}; where: {where}")
    system = "\n".join([
        "You write short notes pinned on a world map of rivers, explaining what the reader sees.",
        "Here are facts taken from the map by code, each with the placeholder that stands for its value.",
        "",
        *lines,
        "",
        f"Choose the {MIN_NOTES} to {MAX_NOTES} facts most worth noting, group facts about the same place into "
        "one note, and order the notes by importance.",
        "For each note: 'facts' lists the fact ids it uses (the first one with a place decides where the pin "
        "goes), 'title' is at most six words, 'text' is one or two short plain sentences.",
        "Write every number, date and place name as its placeholder, exactly: {f3}, {f3.when}, {f3.place} "
        "(only the ones listed above; a place without one has no name yet, so do not name it). "
        "Never write a number yourself, in digits or in words (twelve, half, twice): a note with a number "
        "its facts do not hold is thrown away.",
        "Say only what the facts say. Do not guess causes, impacts or forecasts beyond them. "
        "Colours: much below normal is dark brown, below normal tan, above normal light teal, "
        "much above normal dark teal.",
        'Reply with ONE JSON object and nothing else: {"notes": [{"facts": ["f1"], "title": "...", "text": "..."}]}',
    ])
    if context:
        system += f"\n\nWhat the reader is looking at: {context}"
    return {"system": system, "schema": notes_schema()}


def check_model_notes(reply: Any, facts: list[dict[str, Any]], bbox: list[float] | None = None) -> dict[str, Any]:
    """A model's reply (text or parsed JSON) read into notes, each checked: known fact ids, at least one with a
    place, placeholders that exist, and the claim lock (no number that is not in its facts). Numbers are put in
    by code. When fewer than three notes pass, template notes on other facts top them up (marked
    ``"by": "rules"``). Returns ``{"notes", "refused"}``; ``refused`` says why each dropped note was dropped."""
    data = reply
    if isinstance(reply, str):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
        start = raw.find("{")
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:]) if start >= 0 else (None, 0)
        except ValueError:
            data = None
    items = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {"notes": [], "refused": ["the model did not reply with notes"]}
    by_id = _by_id(facts)
    bbox = _check_bbox(bbox) if bbox else [-180.0, -85.0, 180.0, 85.0]
    notes: list[dict[str, Any]] = []
    refused: list[str] = []
    for i, item in enumerate(items[:MAX_NOTES], 1):
        if not isinstance(item, dict):
            refused.append(f"note {i}: not an object")
            continue
        ids = [str(x) for x in item.get("facts") or [] if str(x) in by_id]
        title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()[:60]
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()[:240]
        cited = [by_id[x] for x in dict.fromkeys(ids)]
        if not cited or not title:
            refused.append(f"note {i}: no known fact or no title")
            continue
        if not any(f.get("place") for f in cited):
            refused.append(f"note {i}: none of its facts has a place")
            continue
        unknown = [m.group(1) for m in _PLACEHOLDER.finditer(title + " " + text) if m.group(1) not in by_id]
        if unknown:
            refused.append(f"note {i}: cites {', '.join(unknown)}, which is not a fact")
            continue
        # {f3.when} or {f3.place} of a fact with no date or no name would leave a hole in the words.
        empty = [m.group(0) for m in _PLACEHOLDER.finditer(title + " " + text)
                 if m.group(2) and not _part(by_id[m.group(1)], m.group(2))]
        if empty:
            refused.append(f"note {i}: {', '.join(empty)} has nothing to put in")
            continue
        # Placeholders the note uses count as cited too: their numbers are put in by code.
        used = [by_id[m.group(1)] for m in _PLACEHOLDER.finditer(title + " " + text)]
        for f in used:
            if f not in cited:
                cited.append(f)
        touch = placeholder_touches(title + " " + text)
        if touch:
            refused.append(f"note {i}: a sign or a digit against a placeholder ({', '.join(touch)})")
            continue
        stray = claim_lock(title + " " + text, cited)
        if stray:
            refused.append(f"note {i}: a number not in its facts ({', '.join(stray)})")
            continue
        spot = next((p for p in _spots(_anchor(cited)) if _spaced(notes, p["lat"], p["lon"], bbox, 0.06)), None)
        if spot is None:
            refused.append(f"note {i}: on the same spot as another note")
            continue
        notes.append(_note(cited, _fill(title, by_id), _fill(text, by_id), spot))
    # Fewer than three notes survived: the fixed wording tops them up, from facts the model did not use.
    if notes and len(notes) < MIN_NOTES:
        used = {i for n in notes for i in n["fact_ids"]}
        for t in template_notes(facts, bbox):
            if len(notes) >= MIN_NOTES:
                break
            if t["fact_ids"][0] in used or not _spaced(notes, t["lat"], t["lon"], bbox, 0.06):
                continue
            notes.append(t)
    return {"notes": notes, "refused": refused}


def model_notes(facts: list[dict[str, Any]], *, provider: str | None = None, model: str | None = None,
                api_key: str | None = None, base_url: str | None = None, context: str = "",
                bbox: list[float] | None = None, client: Any = None) -> dict[str, Any]:
    """Ask the reader's own model (their key, through the provider registry) to choose, order and word the
    notes, then check them (:func:`check_model_notes`). Returns its dict plus ``model``. No key of ours is
    ever used: with no key and no client it raises, and the caller keeps the template notes."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        model, provider = cfg["model"], cfg["provider"]
    prompt = notes_prompt(facts, context)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=900,
        messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": "Write the notes."}],
    )
    out = check_model_notes(resp.choices[0].message.content or "", facts, bbox)
    out["model"] = f"{model} via {provider}"
    return out


# ── the CLI and MCP face: gather the published layers, then explain ─────────


def _gather_status(month: str) -> dict[str, Any] | None:
    from aquascope.map_layers import river_status_month
    from aquascope.utils.http_client import CachedHTTPClient

    res = river_status_month(month)
    if not res.get("available"):
        return None
    client = CachedHTTPClient(timeout=60.0, retries=2, cache_ttl_seconds=6 * 3600)
    try:
        return {"month": res["month"], "data": client.get_bytes(res["url"])}
    finally:
        client.close()


def _gather_floods_past(day: str | None) -> dict[str, Any] | None:
    from aquascope.context import floods_past as fp

    index = fp.load_index()
    if not index:
        return None
    months_on_record = [m["month"] for m in index.get("months") or []]
    if not months_on_record:
        return None
    last = months_on_record[-1]
    end = fp.month_of(day) if day else last
    months = fp.resolve_window(last=min(end, last))
    cells = [c for m in months if m in set(months_on_record) for c in fp.load_month(m)]
    return {"months": months, "cells": cells, "deg": index.get("deg") or fp.GRID_DEG}


def _gather_gauges(bbox: list[float]) -> list[dict[str, Any]]:
    from aquascope.archive.catalog import load_stations
    from aquascope.watch import load_snapshot

    rows = load_snapshot()
    if not rows:
        return []
    where = {(str(s.get("source")), str(s.get("station_id"))): s for s in load_stations()}
    out = []
    for r in rows:
        s = where.get((str(r.get("source")), str(r.get("station_id"))))
        if not s or s.get("latitude") is None or s.get("longitude") is None:
            continue
        lat, lon = float(s["latitude"]), float(s["longitude"])
        if not _in_bbox(lat, lon, bbox):
            continue
        out.append({"source": r["source"], "station_id": r["station_id"], "name": s.get("name"),
                    "agency": s.get("agency") or r["source"], "lat": lat, "lon": lon, "cls": r.get("class"),
                    "pct": r.get("percentile"), "date": str(r.get("value_date") or "")[:10],
                    "value": r.get("value"), "n_years": r.get("n_years")})
    return out


def explain_view(bbox: list[float], day: str | None = None, layers: list[str] | None = None, *,
                 name_places: bool = True, llm: dict[str, Any] | None = None) -> dict[str, Any]:
    """Explain one view of the Explorer's map from the published layers: the facts, then three to five notes
    (fixed wording, or with ``llm`` = ``{provider, model, api_key, base_url}`` the reader's own model,
    claim-locked, falling back to the fixed wording). ``day`` is the map date (default: today); ``layers``
    are among :data:`LAYERS` (default: all four). Reads the month's status file, the Floods ahead issue,
    the Floods past grid and the gauges' status snapshot, as the page does."""
    bbox = _check_bbox(bbox)
    day = day or date.today().isoformat()
    on = [x for x in (layers or LAYERS) if x in LAYERS]
    errors: list[str] = []
    status = warnings = floods = None
    gauges: list[dict[str, Any]] = []
    if "status" in on:
        try:
            status = _gather_status(day[:7])
        except Exception as exc:  # noqa: BLE001 - one layer that cannot be read leaves the others
            errors.append(f"status: {type(exc).__name__}: {exc}")
    if "floods_ahead" in on:
        try:
            from aquascope.archive.warnings import flood_warnings

            fw = flood_warnings(bbox, limit=500)
            if fw.get("available"):
                warnings = {"issue_date": fw.get("issue_date"), "reaches": fw.get("reaches") or []}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"floods_ahead: {type(exc).__name__}: {exc}")
    if "floods_past" in on:
        try:
            floods = _gather_floods_past(day)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"floods_past: {type(exc).__name__}: {exc}")
    if "gauges" in on:
        try:
            gauges = _gather_gauges(bbox)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gauges: {type(exc).__name__}: {exc}")
    zoom = max(1.0, min(12.0, math.log2(360 / max(_span(bbox)[0], 1e-3)) + 1))
    res = explain({"bbox": bbox, "date": day, "layers": on, "zoom": zoom}, status=status, warnings=warnings,
                  floods_past=floods, gauges=gauges, name_places=name_places)
    res.pop("prompt", None)
    res["by"] = "rules"
    if llm and res["facts"]:
        try:
            got = model_notes(res["facts"], bbox=bbox, **llm)
            res["refused"] = got["refused"]
            if got["notes"]:
                res.update(notes=name_notes(got["notes"], zoom) if name_places else got["notes"], by=got["model"])
        except (RuntimeError, ValueError) as exc:
            errors.append(f"model: {exc}")
    if errors:
        res["errors"] = errors
    return res
