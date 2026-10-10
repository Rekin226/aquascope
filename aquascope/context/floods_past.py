"""Floods past (#547): flood events in the news and floods seen by radar, month by month, worldwide.

The Archive's context mirror (#520) holds every Groundsource news event and the Microsoft Sentinel-1 monthly
detection counts on a 0.05 degree grid. Rolled up here to one row per half-degree cell and month, with the two
sources kept in separate columns, the whole history is a few MB and one month is one small file the Explorer
can animate:

    context/floods/monthly/grid.parquet             month, row, col, lat, lon, news, radar (every month)
    context/floods/monthly/index.json               the months, their totals per source, the cell size, licences
    context/floods/monthly/months/YYYY-MM.json.gz   {"month", "deg", "cells": [[row, col, news, radar], ...]}

* ``news`` counts the Groundsource events (Google, CC BY 4.0) that started in the month and whose affected area
  is centred in the cell. A news event is a report that a flood happened, not a measurement of it.
* ``radar`` sums the Microsoft AI for Good flood detections (Sentinel-1, 20 m pixels, after the dataset's own
  recommended false-positive filters; MIT) in the cell that month. Radar covers October 2014 to September 2024.

:func:`flood_events_month` reads it back for a month or a range of months, worldwide or inside a box. A small
box (a clicked cell) also lists its news events with their dates and its radar months, from the mirror's own
per-cell files, so the list and the counts agree.

The builder (:func:`grid_frame` and friends) needs pandas; the reader runs on the standard library, in the
Explorer's light worker as in CPython.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from collections.abc import Iterable
from typing import Any

from aquascope.context._common import (
    DEFAULT_REPO_ID,
    MIRROR_MISSING,
    cells_for_bbox,
    check_bbox,
    failed,
    get_bytes,
    layer_result,
    memo,
    mirror_url,
    num,
    read_cells,
)

__all__ = [
    "FOLDER",
    "GRID_DEG",
    "add_months",
    "cell_bbox",
    "cell_centre",
    "cell_of",
    "flood_events_month",
    "grid_frame",
    "index_payload",
    "merge_cells",
    "month_of",
    "month_payloads",
    "months_between",
    "resolve_window",
]

#: Half a degree: about 55 km at the equator. Fine enough to see a delta flood, coarse enough that the whole
#: history is a few MB (0.25 degrees doubles it; measured on the published mirror on 2026-10-10).
GRID_DEG = 0.5
FOLDER = "floods/monthly"
SOURCES = ("groundsource", "microsoft_floods")
RADAR_PERIOD = ("2014-10", "2024-09")
#: The most months one call reads (one small file each), and the default window.
MAX_MONTHS = 60
DEFAULT_WINDOW = 12
#: A box touching more mirror cells than this gets the grid's counts but no event list.
MAX_LIST_CELLS = 16

_MONTH = re.compile(r"^(\d{4})-(\d{2})(?:-\d{2})?$")
_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── months ───────────────────────────────────────────────────────────────────


def month_of(value: Any) -> str:
    """``"YYYY-MM"`` from a month, a day (``"2024-07-14"``) or a date; ValueError otherwise."""
    if hasattr(value, "year") and hasattr(value, "month"):
        return f"{int(value.year):04d}-{int(value.month):02d}"
    m = _MONTH.match(str(value or "").strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError(f"{value!r} is not a month (YYYY-MM)")
    return f"{m.group(1)}-{m.group(2)}"


def add_months(month: str, n: int) -> str:
    y, m = (int(p) for p in month_of(month).split("-"))
    total = y * 12 + (m - 1) + int(n)
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def months_between(start: str, end: str) -> list[str]:
    """Every month from ``start`` to ``end``, both included, in order (the ends may come either way round)."""
    a, b = sorted((month_of(start), month_of(end)))
    out = [a]
    while out[-1] < b:
        out.append(add_months(out[-1], 1))
    return out


def month_label(month: str) -> str:
    y, m = month_of(month).split("-")
    return f"{_NAMES[int(m) - 1]} {y}"


def span_label(months: list[str]) -> str:
    if not months:
        return "no months"
    if len(months) == 1:
        return month_label(months[0])
    return f"{month_label(months[0])} to {month_label(months[-1])}"


def resolve_window(month: Any = None, start: Any = None, end: Any = None, *, last: str | None = None,
                   size: int = DEFAULT_WINDOW) -> list[str]:
    """The months a call covers: one ``month``, ``start`` to ``end``, or else the ``size`` months ending at
    ``last`` (the last month on record). At most :data:`MAX_MONTHS`."""
    if month is not None and month != "":
        return [month_of(month)]
    if start or end:
        a = month_of(start or end)
        b = month_of(end or start)
        months = months_between(a, b)
    else:
        if not last:
            raise ValueError("give a month, or start and end")
        months = months_between(add_months(last, -(size - 1)), last)
    if len(months) > MAX_MONTHS:
        raise ValueError(f"{len(months)} months is too long a window (at most {MAX_MONTHS})")
    return months


# ── cells ────────────────────────────────────────────────────────────────────


def cell_of(lat: float, lon: float, deg: float = GRID_DEG) -> tuple[int, int]:
    """(row, col) of the grid cell holding a point; row 0 starts at 90 S, col 0 at 180 W."""
    rows, cols = round(180 / deg), round(360 / deg)
    row = min(rows - 1, max(0, int(math.floor((float(lat) + 90.0) / deg))))
    col = min(cols - 1, max(0, int(math.floor((float(lon) + 180.0) / deg))))
    return row, col


def cell_centre(row: int, col: int, deg: float = GRID_DEG) -> tuple[float, float]:
    return round(-90.0 + (row + 0.5) * deg, 6), round(-180.0 + (col + 0.5) * deg, 6)


def cell_bbox(row: int, col: int, deg: float = GRID_DEG) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a cell."""
    south, west = -90.0 + row * deg, -180.0 + col * deg
    return round(west, 6), round(south, 6), round(west + deg, 6), round(south + deg, 6)


def merge_cells(months: Iterable[Iterable[Iterable[int]]]) -> dict[tuple[int, int], list[int]]:
    """Several months' ``[row, col, news, radar]`` cells summed per cell (the Explorer draws the same sum)."""
    out: dict[tuple[int, int], list[int]] = {}
    for cells in months:
        for row, col, news, radar in cells:
            acc = out.setdefault((int(row), int(col)), [0, 0])
            acc[0] += int(news)
            acc[1] += int(radar)
    return out


def _inside(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    """Half-open, so a point on the line between two cells belongs to exactly one."""
    west, south, east, north = bbox
    return south <= lat < north and west <= lon < east


# ── building the grid (the mirror-context workflow) ──────────────────────────


def grid_frame(news: Any, radar: Any, deg: float = GRID_DEG) -> Any:
    """One row per (month, cell) with anything in it: ``month, row, col, lat, lon, news, radar``.

    ``news`` has Groundsource's ``start_date, lat, lon`` (one row per event); ``radar`` has the Microsoft
    mirror's ``lat, lon, year, month, n`` (detections per 0.05 degree cell and month). Either may be None.
    """
    import numpy as np
    import pandas as pd

    rows, cols = round(180 / deg), round(360 / deg)

    def cells(frame: Any) -> tuple[Any, Any]:
        r = np.clip(np.floor((frame["lat"].astype(float) + 90.0) / deg), 0, rows - 1).astype("int32")
        c = np.clip(np.floor((frame["lon"].astype(float) + 180.0) / deg), 0, cols - 1).astype("int32")
        return r, c

    parts = []
    if news is not None and len(news):
        f = news.dropna(subset=["lat", "lon", "start_date"])
        month = f["start_date"].astype(str).str.slice(0, 7)
        ok = month.str.match(r"^\d{4}-\d{2}$")
        f, month = f[ok], month[ok]
        r, c = cells(f)
        g = pd.DataFrame({"month": month.values, "row": r.values, "col": c.values})
        parts.append(g.groupby(["month", "row", "col"]).size().rename("news").to_frame().assign(radar=0))
    if radar is not None and len(radar):
        f = radar.dropna(subset=["lat", "lon", "year", "month", "n"])
        month = f["year"].astype(int).map("{:04d}".format) + "-" + f["month"].astype(int).map("{:02d}".format)
        r, c = cells(f)
        g = pd.DataFrame({"month": month.values, "row": r.values, "col": c.values,
                          "radar": f["n"].astype("int64").values})
        parts.append(g.groupby(["month", "row", "col"])["radar"].sum().to_frame().assign(news=0))
    if not parts:
        return pd.DataFrame(columns=["month", "row", "col", "lat", "lon", "news", "radar"])
    frame = pd.concat(parts).groupby(level=[0, 1, 2])[["news", "radar"]].sum().reset_index()
    frame = frame.astype({"row": "int32", "col": "int32", "news": "int64", "radar": "int64"})
    frame["lat"] = (-90.0 + (frame["row"] + 0.5) * deg).round(6)
    frame["lon"] = (-180.0 + (frame["col"] + 0.5) * deg).round(6)
    frame = frame[["month", "row", "col", "lat", "lon", "news", "radar"]]
    return frame.sort_values(["month", "row", "col"], kind="stable").reset_index(drop=True)


def month_payloads(frame: Any, deg: float = GRID_DEG) -> dict[str, dict[str, Any]]:
    """The browser's file for each month: ``{"month", "deg", "cells": [[row, col, news, radar], ...]}``."""
    out = {}
    for month, part in frame.groupby("month", sort=True):
        cells = part[["row", "col", "news", "radar"]].astype("int64").values.tolist()
        out[str(month)] = {"month": str(month), "deg": deg, "cells": cells}
    return out


def encode_month(payload: dict[str, Any]) -> bytes:
    """A month file as gzipped compact JSON (the Archive's CDN serves files as they are, uncompressed)."""
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), mtime=0)


def index_payload(frame: Any, *, deg: float = GRID_DEG, news_first: str | None = None,
                  news_last: str | None = None, built: str | None = None) -> dict[str, Any]:
    """``index.json``: every month with its totals and the largest cell per source (the Explorer scales its
    circles by the largest cell), the spans of the two sources, and their licences."""
    from aquascope.registry import CONTEXT_LAYERS

    months = []
    for month, part in frame.groupby("month", sort=True):
        months.append({"month": str(month), "cells": int(len(part)),
                       "news": int(part["news"].sum()), "radar": int(part["radar"].sum()),
                       "news_max": int(part["news"].max()), "radar_max": int(part["radar"].max())})
    with_news = [m["month"] for m in months if m["news"]]
    with_radar = [m["month"] for m in months if m["radar"]]
    sources = []
    for key, what in (("groundsource", "news"), ("microsoft_floods", "radar")):
        meta = CONTEXT_LAYERS[key]
        sources.append({"key": key, "column": what, "label": meta.label, "short": meta.short,
                        "licence": meta.license, "attribution": meta.attribution, "homepage": meta.homepage})
    return {
        "about": "Floods past (#547): flood events in the news (Groundsource) and flood detections by Sentinel-1 "
                 "radar (Microsoft AI for Good), per half-degree cell and month. news = events that started in "
                 "the month, centred in the cell; radar = filtered 20 m flood detections in the cell that month.",
        "deg": deg, "built": built,
        "news": {"first": news_first or (with_news[0] if with_news else None),
                 "last": news_last or (with_news[-1] if with_news else None)},
        "radar": {"first": with_radar[0] if with_radar else None, "last": with_radar[-1] if with_radar else None,
                  "period": list(RADAR_PERIOD)},
        "first": months[0]["month"] if months else None,
        "last": months[-1]["month"] if months else None,
        "rows": int(len(frame)),
        "months": months,
        "sources": sources,
    }


# ── reading it back ──────────────────────────────────────────────────────────


def _url(path: str, repo_id: str) -> str:
    return mirror_url(f"{FOLDER}/{path}", repo_id)


def load_index(repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any] | None:
    """The published ``index.json``, or None when the monthly grid is not published yet."""
    def build() -> dict[str, Any] | None:
        raw = get_bytes(_url("index.json", repo_id))
        return json.loads(raw.decode("utf-8")) if raw else None

    return memo(f"floods_past:index:{repo_id}", build)


def load_month(month: str, repo_id: str = DEFAULT_REPO_ID) -> list[list[int]]:
    """One month's ``[row, col, news, radar]`` cells ([] for a month with nothing on record)."""
    def build() -> list[list[int]]:
        raw = get_bytes(_url(f"months/{month}.json.gz", repo_id))
        if not raw:
            return []
        return json.loads(gzip.decompress(raw).decode("utf-8")).get("cells") or []

    return memo(f"floods_past:month:{repo_id}:{month}", build)


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _box_events(months: list[str], bbox: tuple[float, float, float, float], limit: int,
                repo_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """News events and radar months inside a small box, from the mirror's per-cell files."""
    keys = cells_for_bbox(*bbox)
    first, last = months[0], months[-1]
    news_rows = read_cells("groundsource", keys, repo_id=repo_id)
    news: dict[str, Any] = {}
    if news_rows is not None:
        events = []
        for r in news_rows:
            la, lo, start = num(r.get("lat")), num(r.get("lon")), r.get("start_date") or ""
            if la is None or lo is None or not (first <= start[:7] <= last) or not _inside(la, lo, bbox):
                continue
            events.append({"start": start or None, "end": r.get("end_date") or None, "lat": round(la, 4),
                           "lon": round(lo, 4), "area_km2": num(r.get("area_km2")), "source": "news"})
        events.sort(key=lambda e: (e["start"] or "", e["end"] or ""), reverse=True)
        news = {"events": events[:max(0, int(limit))], "events_listed": min(len(events), max(0, int(limit))),
                "events_found": len(events)}
    radar_rows = read_cells("microsoft_floods", keys, repo_id=repo_id)
    radar: dict[str, Any] = {}
    if radar_rows is not None:
        by_month: dict[str, int] = {}
        for r in radar_rows:
            la, lo, y, m, n = (num(r.get(k)) for k in ("lat", "lon", "year", "month", "n"))
            if None in (la, lo, y, m) or not _inside(la, lo, bbox):
                continue
            key = f"{int(y):04d}-{int(m):02d}"
            if first <= key <= last:
                by_month[key] = by_month.get(key, 0) + int(n or 0)
        radar = {"months_listed": dict(sorted(by_month.items()))}
    return news, radar


def flood_events_month(month: Any = None, *, start: Any = None, end: Any = None,
                       bbox: Iterable[float] | None = None, limit: int = 20, top: int = 5,
                       repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """Where floods were reported in the news and seen by radar in a month or a range of months.

    ``month`` ("2024-07"), or ``start`` and ``end`` (at most 60 months); with neither, the latest 12 months on
    record. ``bbox`` (west, south, east, north) narrows it to a box; without one the answer is worldwide.
    Counts come from the monthly half-degree grid: ``news`` counts Groundsource events that started in the
    window, centred in the box; ``radar`` sums the Sentinel-1 flood detections (2014-10 to 2024-09 only).
    A small box (a clicked cell) also lists up to ``limit`` news events with their dates, and the radar
    detections per month. ``hotspots`` names the ``top`` cells with the most news events and the most radar
    detections.
    """
    sources = list(SOURCES)
    box = check_bbox(*bbox) if bbox is not None else None
    # A window that was asked for is checked before anything is read.
    asked = resolve_window(month, start, end) if (month or start or end) else None
    try:
        index = load_index(repo_id)
        if index is None:
            months = asked or []
            return layer_result("floods_past", sources, ok=True, available=False, months=months,
                                bbox=list(box) if box else None, grid_deg=GRID_DEG,
                                note=MIRROR_MISSING.format(what="monthly flood grid"),
                                summary="The monthly flood grid is not published yet, so there is nothing to map.")
        deg = float(index.get("deg") or GRID_DEG)
        news_span = index.get("news") or {}
        last = (news_span.get("last") or index.get("last") or "")[:7] or None
        months = asked or resolve_window(last=last)
        present = {m["month"] for m in index.get("months") or []}
        merged = merge_cells(load_month(m, repo_id) for m in months if m in present)
        by_month = {m["month"]: m for m in index.get("months") or [] if m["month"] in months}
        if box is not None:
            kept = {k: v for k, v in merged.items() if _inside(*cell_centre(k[0], k[1], deg), box)}
        else:
            kept = merged
        n_news = sum(v[0] for v in kept.values())
        n_radar = sum(v[1] for v in kept.values())

        def hotspots(i: int) -> list[dict[str, Any]]:
            ranked = sorted(((v[i], k) for k, v in kept.items() if v[i] > 0), reverse=True)[:max(0, int(top))]
            out = []
            for value, (row, col) in ranked:
                lat, lon = cell_centre(row, col, deg)
                out.append({"lat": lat, "lon": lon, "news": kept[(row, col)][0], "radar": kept[(row, col)][1],
                            "bbox": list(cell_bbox(row, col, deg))})
            return out

        radar_months = [m for m in months if RADAR_PERIOD[0] <= m <= RADAR_PERIOD[1]]
        news_out: dict[str, Any] = {
            "available": True, "n_events": n_news,
            "by_month": ({m: by_month[m]["news"] for m in months if m in by_month} if box is None else None),
            "first_on_record": news_span.get("first"), "last_on_record": news_span.get("last"),
            "hotspots": hotspots(0),
        }
        radar_out: dict[str, Any] = {
            "available": True, "detections": n_radar, "period": list(RADAR_PERIOD),
            "covered": bool(radar_months),
            "by_month": ({m: by_month[m]["radar"] for m in months if m in by_month} if box is None else None),
            "hotspots": hotspots(1),
        }
        listed = False
        if box is not None and len(cells_for_bbox(*box)) <= MAX_LIST_CELLS:
            news_extra, radar_extra = _box_events(months, box, limit, repo_id)
            news_out.update(news_extra)
            radar_out.update(radar_extra)
            listed = bool(news_extra or radar_extra)
        if box is not None:
            news_out["by_month"] = None
            radar_out["by_month"] = radar_out.pop("months_listed", None)
    except Exception as exc:  # noqa: BLE001 - a network failure is an answer, not a crash
        return failed("floods_past", sources, exc)
    where = "in this box" if box is not None else "worldwide"
    when = span_label(months)
    news_first = str(news_span.get("first") or "")[:7]
    news_last = str(news_span.get("last") or "")[:7]
    if news_first and news_last and not any(news_first <= m <= news_last for m in months):
        # Nothing in the news here is not the same as no floods: the record does not reach these months.
        bits = [f"No news record {where} for {when}: news runs from {month_label(news_first)} to "
                f"{month_label(news_last)} only"]
    else:
        bits = [f"{_plural(n_news, 'flood event')} in the news {where}, {when}"]
    if not radar_months:
        bits.append(f"radar covers {month_label(RADAR_PERIOD[0])} to {month_label(RADAR_PERIOD[1])} only")
    elif n_radar:
        bits.append(f"Sentinel-1 radar made {_plural(n_radar, 'flood detection')} (20 m pixels)")
    else:
        bits.append("Sentinel-1 radar detected no flooding")
    summary = "; ".join(bits) + "."
    summary = summary[0].upper() + summary[1:]
    return layer_result("floods_past", sources, ok=True, available=True, months=months, window=when,
                        bbox=list(box) if box else None, grid_deg=deg, n_cells=len(kept), listed=listed,
                        news=news_out, radar=radar_out, summary=summary)
