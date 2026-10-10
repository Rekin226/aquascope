"""Scout (#563): a scan of the published map layers that pins what is unusual on the globe.

The Explorer's Scout drops up to ten pins, each with its reason, its numbers and the data it came from. The
findings are made here by fixed rules, the same for the browser, the MCP tool, the CLI and the daily file:

* ``status``: the largest connected areas of the world river status map (GEOGLOWS v2 HydroSOS, one month)
  that are much above or much below normal, with, in the daily file, how this month's share compares with
  the same calendar month in every year since 1990.
* ``floods_ahead``: the reaches the daily Floods ahead issue expects to pass the highest return-period flows,
  grouped so that one river in flood is one pin, not fifty.
* ``floods_past``: the half-degree cells with the most flood events in the news (Groundsource) and radar
  detections (Microsoft Sentinel-1) in one month, grouped where they touch; in the daily file, ranked against
  the same calendar month since 2000.
* ``gauges_today``: the clusters of Archive gauges much above or much below normal for the date in the daily
  snapshot (``forecasts/status/latest.parquet``), with the most extreme gauge of each.
* ``models_disagree``: the gauges where even the best model scored in the evidence table
  (``skill/model_skill.parquet``) does no better than the gauge's own mean flow.

Every number is formatted here, by code, into the finding's ``slots``. The keyless titles and reasons are
templates over those slots. A model (the reader's own key, or the on-device model) may reorder the findings
and reword them, but its words are held to the slots: any digit, percent sign or number word outside a
``{slot}`` placeholder rejects that wording and the template stays (:func:`apply_wording`, the claim lock).

Two ways to run it:

* :func:`scan` reads the layers for a month and a view (a box, or a centre and radius for the globe) and is
  what the Explorer runs in its worker for the view on screen. The gauges and skill rows can be handed in
  (the page reads them with DuckDB); otherwise they are read with pyarrow.
* :func:`daily` writes ``scout/latest.json`` (and a dated copy) for the whole world, with the record checks
  that read the history; ``.github/workflows/flood-warnings.yml`` runs it after the day's Floods ahead issue
  and publishes ``scout/`` only. The Explorer's world view uses it when it is fresh.

Standard library and numpy (scipy's labelling when it is there), so it runs in the Explorer's light worker.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FOLDER = "scout"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
ARCHIVE_BASE = f"https://huggingface.co/datasets/{DEFAULT_REPO}/resolve/main/"
MAX_PINS = 10
#: Candidates per kind kept in the daily file (the browser picks from them for the half of the globe on screen).
PER_KIND = 25
#: The kinds in the order they choose their places, how many of each the first pass takes, and how much a kind's
#: score counts when the room left is filled, and the most of it the ten may hold (the past and the evidence
#: table weigh less than today).
KINDS: dict[str, dict[str, Any]] = {
    "floods_ahead": {"label": "Floods ahead", "cap": 3},
    "status": {"label": "World river status", "cap": 3},
    "gauges_today": {"label": "Gauges today", "cap": 2, "max": 3},
    "floods_past": {"label": "Floods past", "cap": 1, "weight": 0.6, "max": 2},
    "models_disagree": {"label": "Models and gauges", "cap": 1, "weight": 0.5, "max": 2},
}
SOURCES = {
    "status": "GEOGLOWS v2 HydroSOS monthly river status (modelled), CC BY 4.0",
    "floods_ahead": "GEOGLOWS v2 forecast, CC BY 4.0; return periods CC BY-NC-SA 4.0 (AquaScope Floods ahead)",
    "floods_past_news": "Groundsource flood events from news (Mayo et al. 2026, Google), CC BY 4.0",
    "floods_past_radar": "Microsoft AI for Good Sentinel-1 flood detections, MIT",
    "gauges_today": "Archive gauges (each agency's terms), today vs normal by AquaScope",
    "models_disagree": "AquaScope evidence table (skill/model_skill.parquet), each model's own licence",
    "places": "Places: Photon by komoot, data © OpenStreetMap contributors (ODbL)",
}
ABOUT = ("Scout (#563): what stands out on the map today, found by fixed rules over the published layers. "
         "Each finding carries its numbers, its reason and its source. Not a warning: follow your national "
         "hydrological or meteorological service.")
METHOD = {
    "status": "Connected areas (0.25 degree cells, 4-neighbour) where most of a cell is in the much above or much "
              "below normal class of the month's GEOGLOWS HydroSOS map, ranked by area. The record check sets "
              "this month's share of the area in that class against the same calendar month of every year "
              "since 1990.",
    "floods_ahead": "The Floods ahead reaches with the highest return-period class, then the largest peak "
                    "against that flow; reaches within {km:g} km of a stronger one are counted with it.",
    "floods_past": "Half-degree cells with at least {news} news events or {radar} radar detections in the month, "
                   "grouped where they touch, ranked by news events then radar detections. News coverage grows "
                   "over the years, so a recent record partly reflects more reporting.",
    "gauges_today": "Gauges much above or much below normal for the date (mid-rank percentile within 7 days "
                    "of the date in at least 10 other years), grouped within {km:g} km around the most "
                    "extreme one; larger groups first.",
    "models_disagree": "Gauges where the best model scored has KGE at or below {kge:g} (no better than the "
                       "gauge's mean flow) over at least {years} years, with the model's catchment within a "
                       "factor of two of the gauge's and its mean flow between {lo:g}% and +{hi:g}% of the "
                       "gauge's (beyond that it is a units or matching problem); the largest rivers first.",
}

STATUS_DEG = 0.25           # the status map is read in blocks of 5 x 5 of its 0.05 degree cells
STATUS_MIN_AREA_KM2 = 10000.0
AHEAD_GROUP_KM = 200.0
PAST_FLOOR = {"news": 2, "radar": 2000}   # the Explorer's standout floors (floods-past-core.js STANDOUT_FLOOR)
GAUGE_GROUP_KM = 150.0
SKILL_KGE_MAX = -0.41       # aquascope.evidence's grade D: no better than the mean flow
SKILL_MIN_DAYS = 3 * 365
SKILL_MIN_MEAN_CMS = 1.0
SKILL_PBIAS = (-80.0, 400.0)
EARTH_KM = 6371.0
MEMBERS = 51

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── formatting: every number a reader sees is made here ─────────────────────


def sig(x: float, digits: int = 3) -> float:
    """``x`` rounded to ``digits`` significant figures."""
    if not x or not math.isfinite(x):
        return 0.0 if not x else x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def fmt_num(x: float | None, digits: int = 3) -> str:
    """412345 -> '412,000', 18.864 -> '18.9', 0.1071 -> '0.107'."""
    if x is None or not math.isfinite(float(x)):
        return "unknown"
    v = sig(float(x), digits)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(v))))) if v else 0
    return f"{v:,.{decimals}f}"


def month_words(month: str) -> str:
    """'2026-09' -> 'September 2026'."""
    return f"{_MONTHS[int(month[5:7]) - 1]} {month[:4]}"


def day_words(day: str) -> str:
    """'2026-10-14' -> 'Wed 14 Oct'."""
    d = date.fromisoformat(str(day)[:10])
    return f"{_DAYS[d.weekday()]} {d.day} {_MON[d.month - 1]}"


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def short_name(name: str, limit: int = 40) -> str:
    """A gauge's name for a title: the part before a dash or bracket when it is long, cut at ``limit``."""
    name = re.sub(r"\s+", " ", str(name)).strip()
    if len(name) > limit:
        name = re.split(r" - | \[|\(", name)[0].strip() or name
    return name if len(name) <= limit else name[:limit - 1].rstrip() + "…"


def latlon_words(lat: float, lon: float) -> str:
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


# ── geometry ─────────────────────────────────────────────────────────────────


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def ortho_km(centre: list[float], lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """The distance between two points as the globe shows them from above ``centre`` ([lat, lon]), in km of the
    orthographic picture (the same as on the ground at the centre, shorter towards the edge)."""
    p0, l0 = math.radians(centre[0]), math.radians(centre[1])

    def xy(lat: float, lon: float) -> tuple[float, float]:
        p, dl = math.radians(lat), math.radians(lon) - l0
        return (EARTH_KM * math.cos(p) * math.sin(dl),
                EARTH_KM * (math.cos(p0) * math.sin(p) - math.sin(p0) * math.cos(p) * math.cos(dl)))

    (x1, y1), (x2, y2) = xy(lat1, lon1), xy(lat2, lon2)
    return math.hypot(x1 - x2, y1 - y2)


def check_view(view: dict[str, Any] | None) -> dict[str, Any] | None:
    """See :func:`_check_view`; a view may also carry ``min_km``, how far apart two pins must be to be told apart
    on screen (the page works it out from the zoom), which :func:`rank_findings` uses."""
    out = _check_view(view)
    if out is not None and view and view.get("min_km") is not None:
        km = float(view["min_km"])
        if not 0 < km <= 5000:
            raise ValueError("a view's min_km is between 0 and 5000")
        out["min_km"] = km
    return out


def _check_view(view: dict[str, Any] | None) -> dict[str, Any] | None:
    """A view is ``{"bbox": [w, s, e, n]}``, ``{"polygon": [[lon, lat], ...]}`` (the outline of what is on screen,
    which on a curved map is not a box; longitudes may run past 180 across the antimeridian) or
    ``{"center": [lat, lon], "radius_km": r}`` (the half of the globe on screen), or None for the world. Raises
    ValueError on a bad one."""
    if not view:
        return None
    if view.get("polygon") is not None:
        try:
            poly = [[float(x), float(y)] for x, y in view["polygon"]]
        except (TypeError, ValueError):
            raise ValueError("a view's polygon is a list of [lon, lat]") from None
        lons, lats = [p[0] for p in poly], [p[1] for p in poly]
        if len(poly) < 3 or not all(-90 <= y <= 90 for y in lats) or max(lons) - min(lons) > 360:
            raise ValueError("a view's polygon needs at least three [lon, lat] points")
        if max(lons) - min(lons) >= 359:
            return None
        return {"polygon": poly, "bbox": [_wrap(min(lons)), min(lats), _wrap(max(lons)), max(lats)]}
    if view.get("bbox") is not None:
        b = [float(x) for x in view["bbox"]]
        if len(b) != 4 or not (-90 <= b[1] < b[3] <= 90) or not all(-180 <= x <= 180 for x in (b[0], b[2])):
            raise ValueError("a view's bbox is [west, south, east, north] in degrees")
        if b[0] == -180 and b[2] == 180 and b[1] <= -60 and b[3] >= 75:
            return None
        return {"bbox": b}
    if view.get("center") is not None:
        lat, lon = (float(x) for x in view["center"])
        r = float(view.get("radius_km") or 0)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and r > 0):
            raise ValueError("a view's center is [lat, lon] with a positive radius_km")
        return {"center": [lat, lon], "radius_km": min(r, 20000.0)}
    raise ValueError("a view needs a bbox or a center and radius_km")


def _wrap(lon: float) -> float:
    x = ((lon + 180) % 360) - 180
    return 180.0 if x == -180 and lon > 0 else x


def in_view(lat: float, lon: float, view: dict[str, Any] | None) -> bool:
    if view is None:
        return True
    if "polygon" in view:
        return bool(view_mask(lat, lon, view))
    if "bbox" in view:
        w, s, e, n = view["bbox"]
        if not s <= lat <= n:
            return False
        return w <= lon <= e if w <= e else (lon >= w or lon <= e)
    c = view["center"]
    return km_between(c[0], c[1], lat, lon) <= view["radius_km"]


def view_size_km(view: dict[str, Any] | None) -> float:
    """About how wide the view is, for how far apart two pins must be."""
    if view is None:
        return 20000.0
    if "radius_km" in view:
        return 2 * view["radius_km"]
    w, s, e, n = view["bbox"]
    width = (e - w) % 360 or 360
    mid = math.radians((s + n) / 2)
    return math.hypot(width * 111.32 * max(0.2, math.cos(mid)), (n - s) * 111.32)


def view_mask(lat: Any, lon: Any, view: dict[str, Any] | None) -> Any:
    """``in_view`` over numpy arrays of cell centres (broadcast)."""
    import numpy as np

    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if view is None:
        return np.ones(np.broadcast(lat, lon).shape, dtype=bool)
    if "polygon" in view:
        poly = view["polygon"]
        lo, hi = min(p[0] for p in poly), max(p[0] for p in poly)
        lat, x = np.broadcast_arrays(lat, lon)
        x = np.where(x < lo, x + 360, x)
        x = np.where(x > hi, x - 360, x)
        inside = np.zeros(lat.shape, dtype=bool)
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):   # even-odd rule
            if y1 == y2:
                continue
            crosses = (y1 > lat) != (y2 > lat)
            inside ^= crosses & (x < (x2 - x1) * (lat - y1) / (y2 - y1) + x1)
        return inside
    if "bbox" in view:
        w, s, e, n = view["bbox"]
        ok_lat = (lat >= s) & (lat <= n)
        ok_lon = ((lon >= w) & (lon <= e)) if w <= e else ((lon >= w) | (lon <= e))
        return ok_lat & ok_lon
    c = view["center"]
    p1, p2 = math.radians(c[0]), np.radians(lat)
    dl = np.radians(lon - c[1])
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a))) <= view["radius_km"]


def label_components(mask: Any) -> tuple[Any, int]:
    """4-neighbour connected components of a boolean grid: (labels, n), labels 1..n and 0 off. scipy's
    labelling when it is there; otherwise a breadth-first walk over the cells that are on (same result)."""
    import numpy as np

    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask)
        return labels.astype(np.int32), int(n)
    except ImportError:
        pass
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    flat = mask.ravel()
    lab = labels.ravel()
    for start in np.flatnonzero(flat):
        if lab[start]:
            continue
        n += 1
        lab[start] = n
        queue = deque([int(start)])
        while queue:
            i = queue.popleft()
            r, c = divmod(i, w)
            for j, ok in ((i - w, r > 0), (i + w, r < h - 1), (i - 1, c > 0), (i + 1, c < w - 1)):
                if ok and flat[j] and not lab[j]:
                    lab[j] = n
                    queue.append(j)
    # Number the components in scan order of their first cell, as scipy does.
    return labels, n


# ── findings ──────────────────────────────────────────────────────────────────

TEMPLATES: dict[str, dict[str, str]] = {
    "status_much_above": {
        "title": "{place}: rivers much above normal",
        "reason": "Rivers across {area} were much above normal in {month} (modelled monthly flow over the 90th "
                  "percentile for the month).{record}",
    },
    "status_much_below": {
        "title": "{place}: rivers much below normal",
        "reason": "Rivers across {area} were much below normal in {month} (modelled monthly flow under the 10th "
                  "percentile for the month).{record}",
    },
    "floods_ahead": {
        "title": "{place}: {rp} flow forecast",
        "reason": "GEOGLOWS forecast of {issue}: {peak} on {day}, at or above the {rp} flow{rpflow}; the 2-year "
                  "flow is {q2}. {members} ensemble members reach it.{nearby}",
    },
    "floods_past": {
        "title": "{place}: floods in {month}",
        "reason": "{events} in the news started here in {month}{radar}.{record}",
    },
    "gauges_today_much_below": {
        "title": "{place}: gauges much below normal",
        "reason": "{count} much below normal for {date}. {gauge} is {extreme}.",
    },
    "gauges_today_much_above": {
        "title": "{place}: gauges much above normal",
        "reason": "{count} much above normal for {date}. {gauge} is {extreme}.",
    },
    "models_disagree": {
        "title": "{gauge_short}: models miss this river",
        "reason": "The best of {models} scored here, {model}, has a KGE of {kge} against the gauge over {period}, "
                  "no better than the gauge's own mean flow. Its mean flow is {bias}. Worth checking what the "
                  "gauge measures before trusting either.",
    },
}


def render(template: str, slots: dict[str, str]) -> str:
    return re.sub(r"\s+", " ", template.format(**slots)).strip()


def _finding(fid: str, kind: str, template: str, lat: float, lon: float, *, score: float,
             slots: dict[str, str], facts: list[dict[str, Any]], source: str, when: str,
             extent: list[float] | None = None) -> dict[str, Any]:
    slots = {"place": latlon_words(lat, lon), **slots}
    t = TEMPLATES[template]
    out = {"id": fid, "kind": kind, "template": template, "lat": round(lat, 4), "lon": round(lon, 4),
           "score": round(float(score), 4), "when": when, "slots": slots, "facts": facts, "source": source,
           "title": render(t["title"], slots), "reason": render(t["reason"], slots), "by": "rules"}
    if extent:
        out["extent"] = [round(x, 2) for x in extent]
    return out


def retitle(f: dict[str, Any]) -> dict[str, Any]:
    """Render a finding's template again after its slots changed (a place name filled in)."""
    t = TEMPLATES[f["template"]]
    if f.get("by", "rules") == "rules":
        f["title"], f["reason"] = render(t["title"], f["slots"]), render(t["reason"], f["slots"])
    return f


# ── world river status ───────────────────────────────────────────────────────


def status_blocks(classes: Any, factor: int = 5) -> dict[str, Any]:
    """Count the status map's cells per block of ``factor`` x ``factor``: much below (class 1), much above (5)
    and mapped (any class), on a grid of ``classes.shape / factor`` with each block's centre and area."""
    import numpy as np

    h, w = classes.shape
    hh, ww = h // factor, w // factor
    c = classes[:hh * factor, :ww * factor].reshape(hh, factor, ww, factor)
    deg = 180.0 / hh
    lat = 90 - (np.arange(hh) + 0.5) * deg
    lon = -180 + (np.arange(ww) + 0.5) * (360.0 / ww)
    cell_km2 = (deg * 111.32) * (360.0 / ww * 111.32) * np.cos(np.radians(lat))
    return {
        "low": (c == 1).sum(axis=(1, 3), dtype=np.int16), "high": (c == 5).sum(axis=(1, 3), dtype=np.int16),
        "mapped": (c > 0).sum(axis=(1, 3), dtype=np.int16), "n": factor * factor,
        "lat": lat, "lon": lon, "cell_km2": cell_km2, "deg": deg,
    }


def status_findings(classes: Any, month: str, *, view: dict[str, Any] | None = None, top: int = PER_KIND,
                    min_area_km2: float = STATUS_MIN_AREA_KM2, blocks: dict[str, Any] | None = None
                    ) -> list[dict[str, Any]]:
    """The largest connected areas much above and much below normal in one month's status classes (0 no data,
    1 much below to 5 much above, the whole world at any resolution that divides into 0.25 degree blocks)."""
    import numpy as np

    b = blocks or status_blocks(classes, max(1, round(STATUS_DEG / (180.0 / classes.shape[0]))))
    lat2 = b["lat"][:, None]
    lon2 = b["lon"][None, :]
    inside = view_mask(lat2, lon2, view)
    out: list[dict[str, Any]] = []
    for side, key in (("much_above", "high"), ("much_below", "low")):
        count = b[key]
        mapped = b["mapped"]
        on = (count * 2 > mapped) & (mapped * 4 >= b["n"]) & inside
        labels, n = label_components(on)
        if not n:
            continue
        area = (count / b["n"]) * b["cell_km2"][:, None]
        idx = labels.ravel()
        sums = np.bincount(idx, weights=area.ravel(), minlength=n + 1)
        wl = np.bincount(idx, weights=(area * lat2).ravel(), minlength=n + 1)
        x = np.cos(np.radians(lon2)) * area
        y = np.sin(np.radians(lon2)) * area
        wx = np.bincount(idx, weights=np.broadcast_to(x, area.shape).ravel(), minlength=n + 1)
        wy = np.bincount(idx, weights=np.broadcast_to(y, area.shape).ravel(), minlength=n + 1)
        order = [k for k in np.argsort(-sums[1:]) + 1 if sums[k] >= min_area_km2][:top]
        for rank, k in enumerate(order, 1):
            rows, cols = np.nonzero(labels == k)
            clat = wl[k] / sums[k]
            clon = math.degrees(math.atan2(wy[k], wx[k]))
            # The pin goes on the area's own cell nearest its centre, never in a hole of a curved area.
            d = (b["lat"][rows] - clat) ** 2 + ((b["lon"][cols] - clon) * math.cos(math.radians(clat))) ** 2
            j = int(np.argmin(d))
            plat, plon = float(b["lat"][rows[j]]), float(b["lon"][cols[j]])
            half = b["deg"] / 2
            extent = [float(b["lon"][cols].min()) - half, float(b["lat"][rows].min()) - half,
                      float(b["lon"][cols].max()) + half, float(b["lat"][rows].max()) + half]
            km2 = float(sums[k])
            word = "much above" if side == "much_above" else "much below"
            out.append(_finding(
                f"status:{side}:{month}:{rank}", "status", f"status_{side}", plat, plon,
                score=km2 / (km2 + 300000.0),
                slots={"area": f"{fmt_num(km2, 2)} km²", "month": month_words(month), "record": ""},
                facts=[{"label": f"Area {word} normal, {month_words(month)}", "value": sig(km2, 2), "unit": "km²"}],
                source=SOURCES["status"], when=month, extent=extent,
            ) | {"side": side})
    return out


def _region_of(lat: float, lon: float) -> dict[str, Any] | None:
    from aquascope.map_layers import STATUS_REGIONS

    for reg in STATUS_REGIONS:
        w, s, e, n = reg["bbox"]
        if s <= lat <= n and w <= lon <= e:
            return reg
    return None


def region_share(blocks: dict[str, Any], bbox: list[float], key: str) -> float | None:
    """The share of a box's mapped area in one class (``key`` "high" or "low"), area-weighted."""
    w, s, e, n = bbox
    rows = (blocks["lat"] >= s) & (blocks["lat"] <= n)
    cols = (blocks["lon"] >= w) & (blocks["lon"] <= e)
    wt = blocks["cell_km2"][rows][:, None]
    mapped = float((blocks["mapped"][rows][:, cols] * wt).sum())
    return float((blocks[key][rows][:, cols] * wt).sum()) / mapped if mapped else None


def status_record(findings: list[dict[str, Any]], month: str, blocks: dict[str, Any],
                  history: dict[str, dict[str, Any]]) -> None:
    """For each status finding inside one of the named regions (aquascope.map_layers.STATUS_REGIONS, fixed boxes
    chosen before any month was looked at, so the check is not drawn around this month's extreme), rank the
    region's share of mapped area in the finding's class this month against the same calendar month of earlier
    years (``history``: month -> status_blocks of that month). In place: two facts and the reason's tail."""
    years = sorted(m for m in history if m[5:7] == month[5:7] and m < month)
    if not years:
        return
    name = _MONTHS[int(month[5:7]) - 1]
    cache: dict[tuple[str, str], tuple[float | None, list[float]]] = {}
    for f in findings:
        if f["kind"] != "status":
            continue
        reg = _region_of(f["lat"], f["lon"])
        if reg is None:
            continue
        key = "high" if f["side"] == "much_above" else "low"
        if (reg["name"], key) not in cache:
            past = [x for x in (region_share(history[m], reg["bbox"], key) for m in years) if x is not None]
            cache[(reg["name"], key)] = (region_share(blocks, reg["bbox"], key), past)
        now, past = cache[(reg["name"], key)]
        if now is None or not past:
            continue
        rank = 1 + sum(1 for x in past if x > now + 1e-9)
        n = len(past) + 1
        word = "above" if f["side"] == "much_above" else "below"
        since = years[0][:4]
        if rank == 1:
            tail = (f" Across {reg['name']} as a whole, the share much {word} normal was the largest in {name} "
                    f"of {n} years since {since}.")
        elif rank <= 3:
            tail = (f" Across {reg['name']} as a whole, the share much {word} normal was the {ordinal(rank)} "
                    f"largest in {name} of {n} years since {since}.")
        else:
            tail = ""
        f["slots"]["record"] = tail
        f["facts"].append({"label": f"Share of {reg['name']} much {word} normal", "value": round(100 * now),
                           "unit": "%"})
        f["facts"].append({"label": f"Rank in {name}, {n} years since {since}", "value": rank})
        f["record_rank"] = rank
        f["record_region"] = reg["name"]
        if rank == 1:
            f["score"] = min(1.0, f["score"] + 0.15)
        retitle(f)


# ── floods ahead ─────────────────────────────────────────────────────────────


def _ahead_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ft in features or []:
        p = ft.get("properties") or ft
        g = ft.get("geometry") or {}
        lon, lat = (g.get("coordinates") or [p.get("lon"), p.get("lat")])[:2]
        if lat is None or lon is None or not p.get("rp"):
            continue
        rows.append({**p, "lat": float(lat), "lon": float(lon)})
    return rows


def ahead_score(r: dict[str, Any]) -> float:
    """How much a reach in Floods ahead stands out: half its return-period class (2 to 100 years, on a log
    scale), half the river's size (its 2-year flow, 1 to 10,000 m3/s on a log scale). A big river at its
    25-year flow outranks a creek at its 100-year flow, which is more often a model artefact."""
    rp = max(2, int(r.get("rp") or 2))
    q2 = max(1.0, float(r.get("q2") or 1.0))
    return round(0.5 * math.log(rp) / math.log(100) + 0.5 * min(1.0, math.log10(q2) / 4), 4)


def floods_ahead_findings(features: list[dict[str, Any]], manifest: dict[str, Any] | None = None, *,
                          view: dict[str, Any] | None = None, top: int = PER_KIND,
                          group_km: float = AHEAD_GROUP_KM) -> list[dict[str, Any]]:
    """The strongest Floods ahead reaches (the issue's GeoJSON features or rows), one per river group."""
    manifest = manifest or {}
    issue = manifest.get("issue_date") or ""
    rows = [r for r in _ahead_rows(features) if in_view(r["lat"], r["lon"], view)]

    rows.sort(key=lambda r: (-ahead_score(r), str(r.get("river_id"))))
    heads: list[dict[str, Any]] = []
    for r in rows:
        for h in heads:
            if km_between(h["lat"], h["lon"], r["lat"], r["lon"]) <= group_km:
                h["_nearby"] += 1
                break
        else:
            if len(heads) < top:
                heads.append({**r, "_nearby": 0})
            else:
                continue
    out = []
    for rank, h in enumerate(heads, 1):
        rp = int(h["rp"])
        peak = h.get("peak") or h.get("peak_cms")
        q = h.get(f"q{rp}")   # the issue's GeoJSON carries every return-period flow since #556
        q2 = h.get("q2")
        share = h.get("share")
        nearby = h["_nearby"]
        k = round(float(share) * MEMBERS) if share is not None else None
        members = "All 51" if k == MEMBERS else (f"{k} of {MEMBERS}" if k is not None else "Some of the")
        out.append(_finding(
            f"floods_ahead:{issue}:{h.get('river_id')}", "floods_ahead", "floods_ahead", h["lat"], h["lon"],
            score=ahead_score(h),
            slots={"rp": f"{rp}-year", "peak": f"{fmt_num(peak)} m³/s", "q2": f"{fmt_num(q2)} m³/s",
                   "rpflow": f" of {fmt_num(q)} m³/s" if q and rp != 2 else "",
                   "day": day_words(h["day"]) if h.get("day") else "a day in the next 15",
                   "issue": day_words(issue) if issue else "today", "members": members,
                   "nearby": (f" {nearby} more reach{'es' if nearby != 1 else ''} within {group_km:g} km "
                              f"{'do' if nearby != 1 else 'does'} too." if nearby else "")},
            facts=[{"label": "Forecast peak", "value": sig(float(peak), 3), "unit": "m³/s"},
                   *([{"label": f"{rp}-year flow", "value": sig(float(q), 3), "unit": "m³/s"}] if q else []),
                   {"label": "2-year flow", "value": sig(float(q2 or 0), 3), "unit": "m³/s"},
                   {"label": "Peak day", "value": h.get("day") or "unknown"},
                   {"label": f"Members of {MEMBERS} reaching the 2-year flow",
                    "value": k if k is not None else "unknown"},
                   {"label": "Reaches nearby also passing", "value": nearby}],
            source=SOURCES["floods_ahead"], when=issue,
        ) | {"river_id": h.get("river_id")})
    return out


# ── floods past ──────────────────────────────────────────────────────────────


def complete_month(index: dict[str, Any], before: str | None = None) -> str | None:
    """The newest month in the Floods past index that is not a stub (a month still being filled has a handful of
    events against tens of thousands): at least a quarter of the median of the twelve before it."""
    months = [m for m in (index or {}).get("months") or [] if not before or m["month"] <= before]
    for i in range(len(months) - 1, -1, -1):
        prev = sorted(m["news"] + m["radar"] for m in months[max(0, i - 12):i])
        if not prev:
            return months[i]["month"]
        if months[i]["news"] + months[i]["radar"] >= 0.25 * prev[len(prev) // 2]:
            return months[i]["month"]
    return None


def _cell_groups(cells: list[list[int]], floor: dict[str, int]) -> list[list[list[int]]]:
    """Cells over either floor, grouped where they touch (8 neighbours) on the half-degree grid."""
    keep = {(c[0], c[1]): c for c in cells if c[2] >= floor["news"] or c[3] >= floor["radar"]}
    seen: set[tuple[int, int]] = set()
    groups = []
    for key in sorted(keep):
        if key in seen:
            continue
        seen.add(key)
        group, queue = [], deque([key])
        while queue:
            r, c = queue.popleft()
            group.append(keep[(r, c)])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in keep and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        groups.append(group)
    return groups


def floods_past_findings(cells: list[list[int]], month: str, *, deg: float = 0.5,
                         view: dict[str, Any] | None = None, top: int = PER_KIND,
                         history: dict[str, list[list[int]]] | None = None,
                         floor: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The month's strongest groups of flood cells (``[row, col, news, radar]`` on the ``deg`` grid, row 0 at the
    south pole, as aquascope.context.floods_past writes it). ``history`` (month -> cells) ranks each group
    against the same calendar month of other years."""
    floor = floor or PAST_FLOOR

    def centre(r: int, c: int) -> tuple[float, float]:
        return -90 + (r + 0.5) * deg, -180 + (c + 0.5) * deg   # row 0 is at 90 S (floods_past.cell_of)

    shown = [c for c in cells or [] if in_view(*centre(c[0], c[1]), view)]
    groups = _cell_groups(shown, floor)
    groups.sort(key=lambda g: (-sum(c[2] for c in g), -sum(c[3] for c in g)))
    name = _MONTHS[int(month[5:7]) - 1]
    past = sorted(m for m in (history or {}) if m[5:7] == month[5:7] and m < month)
    out = []
    for g in groups[:top]:
        news, radar = sum(c[2] for c in g), sum(c[3] for c in g)
        best = max(g, key=lambda c: (c[2], c[3]))
        lat, lon = centre(best[0], best[1])
        lats = [centre(c[0], c[1])[0] for c in g]
        lons = [centre(c[0], c[1])[1] for c in g]
        record, facts_extra, rank = "", [], None
        if past:
            keys = {(c[0], c[1]) for c in g}
            counts = [sum(c[2] for c in history[m] if (c[0], c[1]) in keys) for m in past]
            rank = 1 + sum(1 for x in counts if x > news)
            n = len(past) + 1
            if news and rank == 1:
                record = f" The most news flood events in these cells in {name} of {n} years since {past[0][:4]}."
            elif news and rank <= 3:
                record = (f" The {ordinal(rank)} most in these cells in {name} of {n} years since "
                          f"{past[0][:4]}.")
            facts_extra = [{"label": f"Rank in {name}, {n} years since {past[0][:4]}", "value": rank}]
        events = f"{news:,} flood event{'s' if news != 1 else ''}" if news else "No flood events"
        out.append(_finding(
            f"floods_past:{month}:{best[0]}:{best[1]}", "floods_past", "floods_past", lat, lon,
            score=min(1.0, news / 200.0) * 0.8 + (0.2 if rank == 1 and news else 0.0),
            slots={"month": month_words(month), "events": events, "record": record,
                   "radar": (f", and Sentinel-1 radar saw {radar:,} flooded 20 m pixels" if radar else "")},
            facts=[{"label": "Flood events in the news", "value": news},
                   *([{"label": "Radar flood detections (20 m)", "value": radar}] if radar else []),
                   {"label": "Half-degree cells", "value": len(g)}, *facts_extra],
            source=SOURCES["floods_past_news"] + ("; " + SOURCES["floods_past_radar"] if radar else ""),
            when=month, extent=[min(lons) - deg / 2, min(lats) - deg / 2, max(lons) + deg / 2, max(lats) + deg / 2],
        ) | ({"record_rank": rank} if rank else {}))
    return out


# ── gauges today ─────────────────────────────────────────────────────────────


def _extreme_words(pct: float, n_years: int) -> str:
    if pct <= 0.0:
        return f"lower than every value within a week of this date in its other {n_years} years"
    if pct >= 100.0:
        return f"higher than every value within a week of this date in its other {n_years} years"
    return f"at the {ordinal(int(round(pct)))} percentile of {n_years} years for the date"


def gauges_today_findings(rows: list[dict[str, Any]], *, view: dict[str, Any] | None = None, top: int = PER_KIND,
                          group_km: float = GAUGE_GROUP_KM) -> list[dict[str, Any]]:
    """Groups of gauges much above or much below normal today (rows of the daily snapshot with lat, lon and
    name), each around its most extreme gauge."""
    out = []
    for side in ("much_below", "much_above"):
        sel = [r for r in rows or [] if r.get("class") == side and r.get("lat") is not None
               and r.get("percentile") is not None and in_view(float(r["lat"]), float(r["lon"]), view)]
        # Most extreme first, then the longest record.
        sel.sort(key=lambda r: ((r["percentile"] if side == "much_below" else 100 - r["percentile"]),
                                -(r.get("n_years") or 0), str(r.get("station_id"))))
        heads: list[dict[str, Any]] = []
        for r in sel:
            for h in heads:
                if km_between(h["lat"], h["lon"], r["lat"], r["lon"]) <= group_km:
                    h["_n"] += 1
                    break
            else:
                heads.append({**r, "_n": 1})
        heads.sort(key=lambda h: -h["_n"])
        for h in heads[:top]:
            n = h["_n"]
            name = str(h.get("name") or h.get("station_id"))
            word = side.replace("_", " ")
            when = str(h.get("value_date") or "")[:10]
            count = (f"{n} gauges within {group_km:g} km are" if n > 1 else "This gauge is")
            out.append(_finding(
                f"gauges_today:{side}:{h['source']}/{h['station_id']}", "gauges_today", f"gauges_today_{side}",
                float(h["lat"]), float(h["lon"]),
                score=min(1.0, n / 25.0) * 0.7 + (0.3 if h["percentile"] in (0.0, 100.0) else 0.1),
                slots={"count": count, "date": day_words(when) if when else "the date", "gauge": name,
                       "extreme": _extreme_words(float(h["percentile"]), int(h.get("n_years") or 0))},
                facts=[{"label": f"Gauges {word} normal nearby", "value": n},
                       {"label": f"{name}, percentile for the date", "value": float(h["percentile"])},
                       *([{"label": "Flow", "value": sig(float(h["value"]), 3), "unit": "m³/s"}]
                         if h.get("value") is not None else []),
                       {"label": "Years of record compared", "value": int(h.get("n_years") or 0)}],
                source=SOURCES["gauges_today"] + f" ({h['source']}/{h['station_id']})", when=when,
            ) | {"side": side, "gauge": f"{h['source']}/{h['station_id']}"})
    return out


# ── models and gauges ────────────────────────────────────────────────────────


def models_disagree_findings(rows: list[dict[str, Any]], *, view: dict[str, Any] | None = None,
                             top: int = PER_KIND, names: dict[str, str] | None = None,
                             min_km: float = 100.0) -> list[dict[str, Any]]:
    """Gauges where no scored model does better than the gauge's mean flow (rows of the evidence table)."""
    by_gauge: dict[str, list[dict[str, Any]]] = {}
    for r in rows or []:
        by_gauge.setdefault(f"{r['source']}/{r['station_id']}", []).append(r)
    cands = []
    for key, rs in by_gauge.items():
        best = next((r for r in rs if r.get("is_best")), None)
        if not best or best.get("kge") is None or best.get("lat") is None:
            continue
        kge = float(best["kge"])
        ratio = best.get("area_ratio")
        pbias = best.get("pbias")
        if (kge > SKILL_KGE_MAX or (best.get("n_days") or 0) < SKILL_MIN_DAYS
                or (best.get("mean_gauge") or 0) < SKILL_MIN_MEAN_CMS
                or (ratio is not None and not 0.5 <= float(ratio) <= 2.0)
                # A model at a hundredth or a tenfold of the gauge's volume is a units or matching problem,
                # not a model missing a river (seen in the table: gauges recorded about 1000 times too high).
                or pbias is None or not SKILL_PBIAS[0] <= float(pbias) <= SKILL_PBIAS[1]
                or not in_view(float(best["lat"]), float(best["lon"]), view)):
            continue
        n_models = int(best.get("n_models") or sum(1 for r in rs if r.get("kge") is not None))
        cands.append((-float(best["mean_gauge"]), key, best, n_models))
    cands.sort(key=lambda c: (c[0], c[1]))
    out: list[dict[str, Any]] = []
    for _, key, best, n_models in cands:
        kge = float(best["kge"])
        if len(out) >= top:
            break
        lat, lon = float(best["lat"]), float(best["lon"])
        if any(km_between(lat, lon, f["lat"], f["lon"]) < min_km for f in out):
            continue
        name = (names or {}).get(key) or str(best.get("name") or best["station_id"])
        pbias = best.get("pbias")
        bias = ("about the same as the gauge's" if pbias is None or abs(pbias) < 5 else
                f"{fmt_num(abs(pbias), 2)}% {'above' if pbias > 0 else 'below'} the gauge's")
        period = f"{str(best.get('start'))[:4]} to {str(best.get('end'))[:4]}"
        out.append(_finding(
            f"models_disagree:{key}", "models_disagree", "models_disagree", lat, lon,
            score=min(1.0, math.log10(max(1.0, float(best["mean_gauge"]))) / 3),
            slots={"gauge": name, "gauge_short": short_name(name),
                   "models": f"{n_models} model{'s' if n_models != 1 else ''}",
                   "model": str(best.get("label") or best.get("model")), "kge": f"{kge:.2f}", "period": period,
                   "bias": bias},
            facts=[{"label": f"Best KGE ({best.get('label') or best.get('model')})", "value": round(kge, 2)},
                   *([{"label": "Mean flow bias", "value": round(float(pbias)), "unit": "%"}]
                     if pbias is not None else []),
                   {"label": "Models scored", "value": n_models}, {"label": "Period", "value": period}],
            source=SOURCES["models_disagree"] + f" ({key})", when=str(best.get("computed_at") or "")[:10],
        ) | {"gauge": key})
    return out


# ── ranking ──────────────────────────────────────────────────────────────────


def rank_findings(findings: list[dict[str, Any]], *, max_pins: int = MAX_PINS, view: dict[str, Any] | None = None,
                  min_km: float | None = None, kinds: dict[str, dict[str, Any]] | None = None,
                  extra: int = 0) -> list[dict[str, Any]]:
    """Up to ``max_pins`` findings, then ``extra`` more for a model to choose among.

    Chosen kind by kind in :data:`KINDS` order, each kind's strongest first up to its cap, then any room left
    filled by weighted score; never two pins closer than ``min_km`` (by default a twelfth of the view, at most
    800 km, or the view's own ``min_km`` when that is more), so one place is one pin and the earlier kind keeps a
    contested place. Numbered by interleaving the kinds: the strongest of each kind first, then the second of
    each, and so on."""
    kinds = kinds or KINDS
    # A twelfth of the view (at most 800 km) keeps a briefing spread out; the page's min_km (about a pin's height
    # on screen) keeps two pins from hiding each other where the view is small on screen, as on a phone.
    sep = min_km if min_km is not None else max(5.0, min(800.0, view_size_km(view) / 12),
                                                float((view or {}).get("min_km") or 0))
    pools = {k: sorted([f for f in findings if f["kind"] == k and in_view(f["lat"], f["lon"], view)],
                       key=lambda f: (-f["score"], f["id"])) for k in kinds}
    picked: list[dict[str, Any]] = []

    centre = view.get("center") if view else None

    def apart(a: dict[str, Any], b: dict[str, Any]) -> float:
        # On the globe, how far apart they look: near the edge a long way is a short step on screen.
        if centre:
            return ortho_km(centre, a["lat"], a["lon"], b["lat"], b["lon"])
        return km_between(a["lat"], a["lon"], b["lat"], b["lon"])

    def fits(f: dict[str, Any]) -> bool:
        return all(apart(f, p) >= sep for p in picked)

    for k, meta in kinds.items():
        taken = 0
        while pools[k] and taken < meta["cap"] and len(picked) < max_pins:
            f = pools[k].pop(0)
            if fits(f):
                picked.append(f)
                taken += 1

    def weighted(f: dict[str, Any]) -> float:
        return f["score"] * float(kinds[f["kind"]].get("weight", 1.0))

    rest = sorted((f for k in kinds for f in pools[k]), key=lambda f: (-weighted(f), f["id"]))
    later = []
    for f in rest:
        if len(picked) >= max_pins and len(later) >= max(0, int(extra)):
            break
        if len(picked) < max_pins and sum(1 for p in picked if p["kind"] == f["kind"]) >= kinds[f["kind"]].get(
                "max", max_pins):
            continue
        if fits(f):
            if len(picked) < max_pins:
                picked.append(f)
            else:
                later.append(f)
                picked.append(f)   # an extra still keeps its place clear of the others
    picks = [f for f in picked if f not in later]
    order = list(kinds)
    seen: dict[str, int] = {}
    keyed = []
    for f in picks:
        seen[f["kind"]] = seen.get(f["kind"], 0) + 1
        keyed.append((seen[f["kind"]], order.index(f["kind"]), f))
    out = [f for *_, f in sorted(keyed, key=lambda x: (x[0], x[1]))] + later
    for i, f in enumerate(out, 1):
        f["rank"] = i
    return out


# ── places ───────────────────────────────────────────────────────────────────

PHOTON_REVERSE = "https://photon.komoot.io/reverse"


def place_url(lat: float, lon: float) -> str:
    """The gazetteer's reverse lookup for a point (Photon, CORS open, checked 2026-10-10): the nearest named
    feature within 25 km, in English."""
    import urllib.parse

    query = {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "limit": 1, "lang": "en", "radius": 25}
    return f"{PHOTON_REVERSE}?{urllib.parse.urlencode(query)}"


def _props(answer: Any) -> dict[str, Any] | None:
    """The first feature's properties of a Photon answer (a FeatureCollection), or the properties themselves."""
    if not isinstance(answer, dict):
        return None
    if "features" in answer:
        feats = answer.get("features") or []
        return feats[0].get("properties") if feats and isinstance(feats[0], dict) else None
    return answer


def _photon_reverse(lat: float, lon: float, timeout: float = 8.0) -> dict[str, Any] | None:
    import urllib.request

    req = urllib.request.Request(place_url(lat, lon),
                                 headers={"User-Agent": "aquascope-scout (+https://github.com/Rekin226/aquascope)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - a fixed https host
        return _props(json.loads(resp.read().decode("utf-8")))


def place_words(props: dict[str, Any] | None) -> str | None:
    """A Photon feature's properties as 'Meghalaya, India' (state and country), or the country alone."""
    if not props:
        return None
    country = props.get("country")
    region = props.get("state") or props.get("county") or (props.get("name") if props.get("type") in (
        "state", "county", "district") else None)
    if region and country and region != country:
        return f"{region}, {country}"
    return country or region


def region_words(lat: float, lon: float) -> str | None:
    """The named status region (aquascope.map_layers.STATUS_REGIONS) a point falls in, if any."""
    from aquascope.map_layers import STATUS_REGIONS

    for reg in STATUS_REGIONS:
        w, s, e, n = reg["bbox"]
        if s <= lat <= n and w <= lon <= e:
            name = reg["name"]
            return name[0].upper() + name[1:]
    return None


def apply_places(findings: list[dict[str, Any]], answers: list[Any]) -> int:
    """Fill each finding's ``place`` slot from its gazetteer answer (``answers`` lines up with ``findings``: a
    Photon FeatureCollection, its first feature's properties, or None), falling back to a named region, then to
    the coordinates already there. Returns how many the gazetteer named. The page fetches the answers itself, in
    parallel, from each finding's ``place_url``; this keeps the naming here."""
    named = 0
    for f, answer in zip(findings, list(answers) + [None] * (len(findings) - len(answers))):
        words = place_words(_props(answer))
        if words:
            named += 1
            f["placed_by"] = "photon"
        f["slots"]["place"] = words or region_words(f["lat"], f["lon"]) or f["slots"]["place"]
        f.pop("place_url", None)
        retitle(f)
    return named


def name_places(findings: list[dict[str, Any]], fetch: Any = None, budget_s: float = 20.0) -> int:
    """Look each finding's point up in the gazetteer (Photon reverse) and fill its ``place`` slot
    (:func:`apply_places`). In CPython the lookups run eight at a time; the browser's worker cannot, so the
    Explorer fetches them from the page instead (``places="page"`` in :func:`scan`)."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    fetch = fetch or _photon_reverse
    start = time.monotonic()

    def one(f: dict[str, Any]) -> Any:
        if time.monotonic() - start >= budget_s:
            return None
        try:
            return fetch(f["lat"], f["lon"])
        except Exception as exc:  # noqa: BLE001 - a name is a nicety; the coordinates stay
            logger.info("reverse geocode failed at %s,%s: %s", f["lat"], f["lon"], exc)
            return None

    if sys.platform == "emscripten" or len(findings) < 2:
        answers = [one(f) for f in findings]
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(one, findings))
    return apply_places(findings, answers)


def _mark_places(findings: list[dict[str, Any]]) -> None:
    """Give each finding not yet named by the gazetteer the URL the page fetches its name from, and meanwhile
    the name of the region it is in, if any."""
    for f in findings:
        if f.get("placed_by") != "photon":
            f["place_url"] = place_url(f["lat"], f["lon"])
            f["slots"]["place"] = region_words(f["lat"], f["lon"]) or f["slots"]["place"]
            retitle(f)


# ── the claim lock: a model may order and word, never write a number ────────

_NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion|half|twice|double|triple|thrice|quarter|dozen|percent|first|third|"
    r"record|highest|lowest|largest|smallest|most|unprecedented)\b", re.I)


def check_wording(text: str, slots: dict[str, str], limit: int) -> tuple[str | None, str | None]:
    """A model's title or reason, held to the slots: the filled text, or None and why it was refused. Any digit,
    percent sign, number word or superlative outside a ``{slot}`` refuses it (those claims are the code's)."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return None, "empty"
    names = re.findall(r"\{(\w+)\}", raw)
    unknown = [n for n in names if n not in slots]
    if unknown:
        return None, f"unknown slot {{{unknown[0]}}}"
    bare = re.sub(r"\{\w+\}", " ", raw)
    if re.search(r"[0-9%‰½¼¾]", bare) or _NUMBER_WORDS.search(bare) or "{" in bare or "}" in bare:
        return None, "a number or a ranking word outside the slots"
    filled = re.sub(r"\{(\w+)\}", lambda m: slots[m.group(1)], raw)
    filled = re.sub(r"\s+", " ", filled).strip()
    if len(filled) > limit:
        return None, "too long"
    return filled, None


def wording_prompt(findings: list[dict[str, Any]], max_pins: int = MAX_PINS, context: str = "") -> dict[str, Any]:
    """The system prompt, the user message and the reply schema for a model that orders and words the findings."""
    items = [{"id": f["id"], "kind": KINDS[f["kind"]]["label"], "where": f["slots"].get("place"),
              "rules_title": f["title"], "rules_reason": f["reason"], "slots": sorted(f["slots"])} for f in findings]
    system = "\n".join([
        "You brief a hydrologist on what stands out on a world map of rivers. Below are findings made by code from "
        "published data, each with an id, its rules wording and the names of its slots.",
        f"Pick at most {max_pins} findings, most worth a look first, and write for each a short title (at most "
        "70 characters) and a reason (one or two plain sentences, at most 260 characters).",
        "Never write a number, a percent sign, a number word or a word like record, largest, highest or most. "
        "Every number, date, place and ranking comes from the slots: write {slot_name} where it goes, for example "
        "{area}, {month}, {peak}, {place}. The code fills them in; wording that breaks this rule is thrown away.",
        "Do not add causes, forecasts or advice that the rules wording does not hold.",
        'Reply with ONE JSON object: {"order": [ids], "notes": [{"id", "title", "reason"}]}.',
    ])
    if context:
        system += f"\n\nWhat the reader is looking at: {context}"
    return {"system": system, "prompt": json.dumps({"findings": items}, ensure_ascii=False),
            "schema": {"type": "object", "required": ["order", "notes"], "properties": {
                "order": {"type": "array", "items": {"type": "string"}, "maxItems": max_pins},
                "notes": {"type": "array", "maxItems": max_pins, "items": {
                    "type": "object", "required": ["id", "title", "reason"],
                    "properties": {"id": {"type": "string"}, "title": {"type": "string"},
                                   "reason": {"type": "string"}}}}}}}


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


def apply_wording(findings: list[dict[str, Any]], reply: Any, *, max_pins: int = MAX_PINS,
                  by: str = "model") -> dict[str, Any]:
    """Hold a model's reply to the claim lock: ``{"findings", "refused", "ordered"}``. The model's order is kept
    for the ids it knows (the rules order fills the rest); each title and reason that passes
    :func:`check_wording` replaces the template, the others keep it and are listed in ``refused``."""
    data = _reply_json(reply)
    byid = {f["id"]: f for f in findings}
    refused: list[str] = []
    if not isinstance(data, dict):
        return {"findings": [dict(f) for f in findings[:max_pins]], "refused": ["the model did not reply with JSON"],
                "ordered": False}
    order = [i for i in dict.fromkeys(str(x) for x in data.get("order") or []) if i in byid][:max_pins]
    ordered = bool(order)
    if not order:
        order = [f["id"] for f in findings[:max_pins]]
    notes = {str(n.get("id")): n for n in data.get("notes") or [] if isinstance(n, dict)}
    out = []
    for rank, fid in enumerate(order, 1):
        f = {**byid[fid], "slots": dict(byid[fid]["slots"]), "rank": rank}
        n = notes.get(fid)
        if n:
            title, why_t = check_wording(n.get("title"), f["slots"], 80)
            reason, why_r = check_wording(n.get("reason"), f["slots"], 400)
            if title and reason:
                f.update(title=title, reason=reason, by=by)
            else:
                refused.append(f"{fid}: {why_t or why_r}")
        out.append(f)
    return {"findings": out, "refused": refused, "ordered": ordered}


def model_wording(findings: list[dict[str, Any]], *, provider: str | None = None, model: str | None = None,
                  api_key: str | None = None, base_url: str | None = None, context: str = "",
                  max_pins: int = MAX_PINS, client: Any = None) -> dict[str, Any]:
    """Ask the reader's own model (their key, the provider registry) to order and word the findings, and hold its
    reply to the claim lock. Nothing falls back to a key of ours."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        model, provider = cfg["model"], cfg["provider"]
    p = wording_prompt(findings, max_pins, context)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=1800,
        messages=[{"role": "system", "content": p["system"]}, {"role": "user", "content": p["prompt"]}],
    )
    res = apply_wording(findings, resp.choices[0].message.content or "", max_pins=max_pins, by="key")
    res["model"] = f"{model} via {provider}" if provider else str(model)
    return res


# ── reading the layers ───────────────────────────────────────────────────────


def _bytes(url: str, timeout: float = 90.0) -> bytes | None:
    """The body of ``url`` (httpx in CPython, urllib through pyodide-http in the browser), or None."""
    from aquascope.context._common import get_bytes

    try:
        return get_bytes(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a missing layer is said, not raised
        logger.info("could not read %s: %s", url, exc)
        return None


def _json(url: str) -> Any:
    raw = _bytes(url)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def _parquet_rows(url: str, columns: list[str] | None = None) -> list[dict[str, Any]] | None:
    import io

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    raw = _bytes(url)
    return pq.read_table(io.BytesIO(raw), columns=columns).to_pylist() if raw else None


def load_status(month: str | None) -> tuple[str | None, Any]:
    """(month, class grid) for a month of the world river status map (the newest when None)."""
    from aquascope.map_layers import river_status_month, status_classes

    res = river_status_month(month)
    if not res.get("available"):
        return None, None
    raw = _bytes(res["url"])
    return (res["month"], status_classes(raw)) if raw else (res["month"], None)


_STATIONS: dict[str, dict[str, dict[str, Any]]] = {}


def _stations(base: str) -> dict[str, dict[str, Any]]:
    """The catalogue's coordinates and names by "source/station_id" (needs pyarrow), read once."""
    if base not in _STATIONS:
        rows = _parquet_rows(f"{base}stations.parquet", ["source", "station_id", "name", "latitude", "longitude"])
        _STATIONS[base] = {f"{s['source']}/{s['station_id']}": s for s in rows or []}
    return _STATIONS[base]


def load_gauges_today(base: str = ARCHIVE_BASE) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """The daily snapshot's rows joined to the catalogue's coordinates and names (needs pyarrow)."""
    rows = _parquet_rows(f"{base}forecasts/status/latest.parquet")
    if rows is None:
        return None, None
    where = _stations(base)
    out = []
    for r in rows:
        s = where.get(f"{r['source']}/{r['station_id']}")
        if s and s.get("latitude") is not None:
            out.append({**r, "value_date": str(r.get("value_date"))[:10], "lat": float(s["latitude"]),
                        "lon": float(s["longitude"]), "name": s.get("name")})
    return out, _json(f"{base}forecasts/status/latest.json")


def load_skill(base: str = ARCHIVE_BASE) -> list[dict[str, Any]] | None:
    """The evidence table's rows with each gauge's name from the catalogue (needs pyarrow)."""
    rows = _parquet_rows(f"{base}skill/model_skill.parquet", [
        "source", "station_id", "lat", "lon", "model", "label", "is_best", "kge", "pbias", "n_days", "mean_gauge",
        "area_ratio", "start", "end", "computed_at"])
    if rows is None:
        return None
    where = _stations(base)
    for r in rows:
        s = where.get(f"{r['source']}/{r['station_id']}")
        r["name"] = s.get("name") if s else None
    return rows


# ── one scan ─────────────────────────────────────────────────────────────────


def scan(view: dict[str, Any] | None = None, month: str | None = None, *, max_pins: int = MAX_PINS,
         gauges: list[dict[str, Any]] | None = None, skill: list[dict[str, Any]] | None = None,
         warnings: dict[str, Any] | None = None, kinds: list[str] | None = None, places: bool | str = True,
         base: str = ARCHIVE_BASE, top: int = PER_KIND, extra: int = 0) -> dict[str, Any]:
    """Scout one view (None for the world) for one month of the status and Floods past layers (their newest by
    default): ``{"picks", "candidates", "findings", "inputs", "notes", "sources", ...}``. ``candidates`` are the
    picks and ``extra`` more after them, for a model to choose among. ``gauges`` and ``skill`` are the snapshot
    and evidence rows when the caller has read them (the browser); otherwise they are read here when pyarrow is
    installed. ``warnings`` is the Floods ahead GeoJSON with its manifest under ``manifest``. ``places`` names the
    candidates with the gazetteer here (True), not at all (False), or ``"page"``: each gets a ``place_url`` the
    caller fetches and hands to :func:`apply_places`."""
    t0 = time.monotonic()
    view = check_view(view)
    want = set(kinds or KINDS)
    timings: dict[str, float] = {}
    last = [t0]

    def _lap(name: str) -> None:
        now = time.monotonic()
        if name != "start":
            timings[name] = round(now - last[0], 2)
        last[0] = now

    findings: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    notes: list[str] = []

    _lap("start")
    if "status" in want:
        m, classes = load_status(month)
        if classes is not None:
            findings += status_findings(classes, m, view=view, top=top)
            inputs["status_month"] = m
        else:
            notes.append(f"the river status map for {month or 'the newest month'} could not be read")

    _lap("status")
    if "floods_ahead" in want:
        gj = warnings or _json(f"{base}forecasts/warnings/latest.geojson")
        manifest = (warnings or {}).get("manifest") or _json(f"{base}forecasts/warnings/manifest.json") or {}
        if gj and gj.get("features") is not None:
            findings += floods_ahead_findings(gj["features"], manifest, view=view, top=top)
            inputs["floods_ahead_issue"] = manifest.get("issue_date")
        else:
            notes.append("no Floods ahead issue could be read")

    _lap("floods_ahead")
    if "floods_past" in want:
        from aquascope.context.floods_past import load_index, load_month

        index = load_index()
        fm = complete_month(index or {}, (month or "")[:7] or None) if index else None
        if fm:
            findings += floods_past_findings(load_month(fm), fm, deg=float(index.get("deg") or 0.5), view=view,
                                             top=top)
            inputs["floods_past_month"] = fm
        else:
            notes.append("the Floods past index could not be read")

    _lap("floods_past")
    if "gauges_today" in want:
        meta = None
        if gauges is None:
            gauges, meta = load_gauges_today(base)
        if gauges is not None:
            findings += gauges_today_findings(gauges, view=view, top=top)
            inputs["gauges_date"] = (meta or {}).get("date") or max(
                (str(g.get("value_date"))[:10] for g in gauges), default=None)
        else:
            notes.append("today's gauge snapshot was not read here (it needs pyarrow, or the rows from the page)")

    _lap("gauges_today")
    if "models_disagree" in want:
        if skill is None:
            skill = load_skill(base)
        if skill is not None:
            findings += models_disagree_findings(skill, view=view, top=top)
            inputs["skill_computed"] = max((str(r.get("computed_at") or "")[:10] for r in skill), default=None)
        else:
            notes.append("the evidence table was not read here (it needs pyarrow, or the rows from the page)")

    _lap("models_disagree")
    cands = rank_findings(findings, max_pins=max_pins, view=view, extra=extra)
    _lap("rank")
    if places == "page":
        _mark_places(cands)
    elif places:
        name_places(cands)
    _lap("places")
    return {"view": view, "picks": [_public(f) for f in cands[:max_pins]],
            "candidates": [_public(f) for f in cands], "findings": [_public(f) for f in findings],
            "inputs": inputs, "notes": notes, "sources": _sources_used(findings), "about": ABOUT,
            "method": method_text(), "seconds": round(time.monotonic() - t0, 2), "timings": timings, "by": "rules"}


def _public(f: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in f.items() if not k.startswith("_")}


def _sources_used(findings: list[dict[str, Any]]) -> list[str]:
    seen = []
    for f in findings:
        for s in f["source"].split("; "):
            s = re.sub(r" \([^)]*/[^)]*\)$", "", s)
            if s not in seen:
                seen.append(s)
    return seen + [SOURCES["places"]]


def method_text() -> dict[str, str]:
    return {
        "status": METHOD["status"],
        "floods_ahead": METHOD["floods_ahead"].format(km=AHEAD_GROUP_KM),
        "floods_past": METHOD["floods_past"].format(**PAST_FLOOR),
        "gauges_today": METHOD["gauges_today"].format(km=GAUGE_GROUP_KM),
        "models_disagree": METHOD["models_disagree"].format(kge=SKILL_KGE_MAX, years=SKILL_MIN_DAYS // 365,
                                                            lo=SKILL_PBIAS[0], hi=SKILL_PBIAS[1]),
    }


# ── the daily file ───────────────────────────────────────────────────────────


def daily(out: str | Path, *, base: str = ARCHIVE_BASE, warnings_dir: str | Path | None = None,
          history: bool = True, places: bool = True, top: int = PER_KIND) -> dict[str, Any]:
    """Write ``<out>/scout/latest.json`` and ``<out>/scout/<date>.json``: the world's findings (``top`` per kind,
    the strongest named), today's picks, and the record checks against the status map's same calendar month
    since 1990 and Floods past's since 2000. ``warnings_dir`` reads a Floods ahead issue just written by
    ``aquascope.archive.warnings run`` instead of the published one."""
    from aquascope.context.floods_past import load_index, load_month
    from aquascope.map_layers import status_months

    t0 = time.monotonic()
    warnings = None
    if warnings_dir:
        root = Path(warnings_dir) / "forecasts" / "warnings"
        root = root if root.exists() else Path(warnings_dir)
        if (root / "latest.geojson").exists():
            warnings = json.loads((root / "latest.geojson").read_text())
            mpath = root / "manifest.json"
            warnings["manifest"] = json.loads(mpath.read_text()) if mpath.exists() else {}
            if warnings["manifest"].get("smoke"):
                warnings = None   # a smoke run is never the day's issue
    res = scan(None, None, max_pins=MAX_PINS, warnings=warnings, places=False, base=base, top=top)
    findings = res["findings"]
    # Record checks (the history is read here only: about 36 status files of 500 kB and a few hundred kB of
    # Floods past months).
    if history:
        sm = res["inputs"].get("status_month")
        if sm:
            m, classes = load_status(sm)
            blocks = status_blocks(classes, 5) if classes is not None else None
            if blocks is not None:
                live = status_findings(classes, m, top=top, blocks=blocks)
                past = {}
                for other in status_months():
                    if other[5:7] == m[5:7] and other < m:
                        _, cl = load_status(other)
                        if cl is not None:
                            past[other] = status_blocks(cl, 5)
                status_record(live, m, blocks, past)
                findings = [f for f in findings if f["kind"] != "status"] + [_public(f) for f in live]
        fm = res["inputs"].get("floods_past_month")
        index = load_index() if fm else None
        if fm and index:
            months = [x["month"] for x in index.get("months") or [] if x["month"][5:7] == fm[5:7] and x["month"] < fm]
            hist = {x: load_month(x) for x in months}
            live = floods_past_findings(load_month(fm), fm, deg=float(index.get("deg") or 0.5), history=hist,
                                        top=top)
            findings = [f for f in findings if f["kind"] != "floods_past"] + live
    picks = rank_findings([dict(f, slots=dict(f["slots"])) for f in findings], max_pins=MAX_PINS)
    if places:
        # Name what the world view shows and the strongest of each kind (a model words those).
        strongest = sorted(findings, key=lambda f: -f["score"])
        want = {f["id"] for f in picks}
        for k in KINDS:
            want |= {g["id"] for g in [f for f in strongest if f["kind"] == k][:4]}
        todo = [f for f in findings if f["id"] in want]
        name_places(todo, budget_s=90.0)
        byid = {f["id"]: f for f in todo}
        for p in picks:
            if p["id"] in byid:
                p["slots"] = dict(byid[p["id"]]["slots"])
                retitle(p)
    today = datetime.now(timezone.utc)
    doc = {
        "made": today.strftime("%Y-%m-%dT%H:%M:%SZ"), "date": today.date().isoformat(), "about": ABOUT,
        "method": method_text(), "inputs": res["inputs"], "notes": res["notes"],
        "picks": [p["id"] for p in picks], "findings": [_public(f) for f in findings],
        "counts": {k: sum(1 for f in findings if f["kind"] == k) for k in KINDS},
        "sources": _sources_used(findings), "seconds": round(time.monotonic() - t0, 1),
    }
    folder = Path(out) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), default=str)
    (folder / "latest.json").write_text(text, encoding="utf-8")
    (folder / f"{doc['date']}.json").write_text(text, encoding="utf-8")
    return {k: doc[k] for k in ("made", "inputs", "counts", "notes", "seconds")} | {"picks": len(picks)}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/scout`` and nothing else to the Archive dataset (needs HF_TOKEN with write access)."""
    src = Path(out) / FOLDER
    if not (src / "latest.json").exists():
        raise FileNotFoundError(f"{src / 'latest.json'} is missing; run the scout first")
    from aquascope.archive.publish import publish_folder

    doc = json.loads((src / "latest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(src, Path(tmp) / FOLDER)
        return str(publish_folder(Path(tmp), repo_id, token=token, allow_patterns=[f"{FOLDER}/*"],
                                  commit_message=f"scout: {doc['date']}"))


def from_published(doc: dict[str, Any], view: dict[str, Any] | None = None, *,
                   max_pins: int = MAX_PINS, extra: int = 0) -> dict[str, Any]:
    """The picks for a view from a published daily file: the file's own picks for the world, else its findings
    inside the view, ranked the same way; ``candidates`` adds ``extra`` more for a model to choose among."""
    view = check_view(view)
    findings = [dict(f, slots=dict(f["slots"])) for f in doc.get("findings") or []]
    cands = rank_findings(findings, max_pins=max_pins, view=view, extra=extra)
    if view is None and doc.get("picks"):
        byid = {f["id"]: f for f in findings}
        picks = [byid[i] for i in doc["picks"] if i in byid][:max_pins]
        for i, f in enumerate(picks, 1):
            f["rank"] = i
        cands = picks + [f for f in cands if f["id"] not in set(doc["picks"])][:max(0, int(extra))]
        for i, f in enumerate(cands, 1):
            f["rank"] = i
    else:
        picks = cands[:max_pins]
    return {"view": view, "picks": picks, "candidates": cands,
            "findings": [f for f in findings if in_view(f["lat"], f["lon"], view)],
            "inputs": doc.get("inputs") or {}, "notes": doc.get("notes") or [], "sources": doc.get("sources") or [],
            "about": doc.get("about") or ABOUT, "method": doc.get("method") or method_text(),
            "published": doc.get("made"), "by": "rules"}


def published(repo_id: str = DEFAULT_REPO) -> dict[str, Any] | None:
    """The published ``scout/latest.json``, or None before the daily workflow has written one."""
    doc = _json(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{FOLDER}/latest.json")
    return doc if isinstance(doc, dict) and doc.get("findings") is not None else None


def use_published(doc: dict[str, Any] | None, view: dict[str, Any] | None, month: str | None, *,
                  today: str | date | None = None, max_age_days: int = 2) -> bool:
    """Whether a view at a month can be answered from the daily file: it is fresh, its status month is the one
    asked for (or none was asked), and the view is wide (the world or half the globe), where a live scan is
    slowest and the record checks matter most."""
    if not doc or not doc.get("date"):
        return False
    t = date.fromisoformat(str(today)[:10]) if today else datetime.now(timezone.utc).date()
    try:
        age = (t - date.fromisoformat(str(doc["date"])[:10])).days
    except ValueError:
        return False
    if age < 0 or age > max_age_days:
        return False
    sm = (doc.get("inputs") or {}).get("status_month")
    if month and sm and str(month)[:7] != sm:
        return False
    v = check_view(view)
    return v is None or view_size_km(v) >= 8000


def scout_view(view: dict[str, Any] | None = None, month: str | None = None, *, doc: dict[str, Any] | None = None,
               today: str | date | None = None, **kw: Any) -> dict[str, Any]:
    """What the Explorer's Scout button runs: the daily file when it can answer (:func:`use_published`), a live
    scan of the view otherwise. ``doc`` is the daily file when the caller has read it."""
    if use_published(doc, view, month, today=today):
        res = from_published(doc, view, max_pins=kw.get("max_pins", MAX_PINS), extra=kw.get("extra", 0))
        missing = [p for p in res["candidates"] if p.get("placed_by") != "photon"]
        if missing and kw.get("places", True) == "page":
            _mark_places(missing)
        elif missing and kw.get("places", True):
            name_places(missing, budget_s=10.0)
        res["mode"] = "daily"
        return res
    res = scan(view, month, **kw)
    res["mode"] = "live"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.map_scout", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("daily")
    p.add_argument("--out", required=True)
    p.add_argument("--warnings", default=None, help="a folder written by aquascope.archive.warnings run")
    p.add_argument("--no-history", action="store_true")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "daily":
        info = daily(a.out, warnings_dir=a.warnings, history=not a.no_history)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
