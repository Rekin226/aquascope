"""Flood history and dams near a place, read from the Archive's context mirror (#520).

* Flood events from news: Google's Groundsource (CC BY 4.0), one row per event with its dates and the
  centre and box of the affected area.
* Floods seen by radar: the Microsoft AI for Good global Sentinel-1 flood dataset (MIT), condensed by the
  mirror to monthly detection counts on a 0.05 degree grid, plus the per-pixel "months with flooding" read
  live from the dataset's own GeoTIFF on Hugging Face.
* Dams: Global Dam Watch v1.0 barriers (CC BY 4.0).

The mirror is built by ``aquascope.archive.context_mirror`` (the mirror-context workflow). Until it is
published every function still answers, with a plain note saying so.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from aquascope.context._common import (
    DEFAULT_REPO_ID,
    MIRROR_MISSING,
    cells_for_bbox,
    check_bbox,
    check_point,
    failed,
    haversine_km,
    in_bbox,
    layer_result,
    num,
    radius_bbox,
    read_cells,
)
from aquascope.utils.cog import COGNotFound, open_cog

__all__ = ["dams", "dams_area", "flood_history", "flood_history_area", "radar_recurrence"]

MS_BASE = "https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset/resolve/main"
MS_PERIOD = ("2014-10", "2024-09")


# ── floods ───────────────────────────────────────────────────────────────────


def ms_tile(lat: float, lon: float) -> str:
    """The 3 x 3 degree tile of the Microsoft dataset holding a point, named by its south-west corner."""
    la = int(math.floor(lat / 3.0) * 3)
    lo = int(math.floor(lon / 3.0) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def radar_recurrence(lat: float, lon: float) -> dict[str, Any]:
    """Months with flooding seen by Sentinel-1 at a 20 m pixel (80 m buffer), October 2014 to September 2024.

    From the dataset's recurrence GeoTIFF: 0 none, 1 the exclusion mask (rough terrain, arid land or towns,
    where detections are unreliable), N >= 2 flooding in N - 1 distinct months.
    """
    tile = ms_tile(lat, lon)
    url = f"{MS_BASE}/{tile[:3]}/{tile}/{tile}-recurrence-80m-buffer.tif"
    try:
        raw = open_cog(url).value_at(lon, lat)
    except COGNotFound:
        # the dataset leaves the file out where nothing was detected after masking
        return {"months": 0, "status": "no detections in this tile", "tile": tile}
    if raw is None or raw == 0:
        return {"months": 0, "status": "no flooding detected", "tile": tile}
    if raw == 1:
        return {"months": None, "status": "excluded (terrain, arid land or built-up area)", "tile": tile}
    return {"months": int(raw) - 1, "status": "flooding detected", "tile": tile}


def _news_events(rows: list[dict[str, str]], keep) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        la, lo = num(r.get("lat")), num(r.get("lon"))
        if la is None or lo is None:
            continue
        d = keep(la, lo)
        if d is False:
            continue
        out.append({"start": r.get("start_date") or None, "end": r.get("end_date") or None,
                    "lat": round(la, 4), "lon": round(lo, 4), "area_km2": num(r.get("area_km2")),
                    **({"distance_km": round(d, 1)} if isinstance(d, float) else {})})
    out.sort(key=lambda e: e["start"] or "", reverse=True)
    return out


def _by_year(dates: list[str | None]) -> dict[str, int]:
    c = Counter(d[:4] for d in dates if d and len(d) >= 4)
    return dict(sorted(c.items()))


def _radar_months(rows: list[dict[str, str]], keep) -> dict[str, int]:
    months: Counter[str] = Counter()
    for r in rows:
        la, lo = num(r.get("lat")), num(r.get("lon"))
        y, m, n = num(r.get("year")), num(r.get("month")), num(r.get("n"))
        if la is None or lo is None or y is None or m is None or keep(la, lo) is False:
            continue
        months[f"{int(y):04d}-{int(m):02d}"] += int(n or 0)
    return dict(sorted(months.items()))


def _summary(news: dict[str, Any], radar: dict[str, Any], where: str) -> str:
    parts = []
    if news.get("available"):
        n = news["n_events"]
        if n:
            parts.append(f"{n} flood event{'s' if n != 1 else ''} in the news {where}, latest {news['latest']}")
        else:
            parts.append(f"no flood events in the news {where}")
    if radar.get("available") and radar.get("months_detected") is not None:
        k = radar["months_detected"]
        parts.append(f"radar saw flooding {where} in {k} month{'s' if k != 1 else ''} of Oct 2014 to Sep 2024"
                     if k else f"radar saw no flooding {where} from Oct 2014 to Sep 2024")
    pixel = radar.get("pixel") or {}
    if pixel.get("months"):
        k = pixel["months"]
        months = f"{k} month{'s' if k != 1 else ''}"
        parts.append(f"at this exact spot in {months}" if radar.get("available") else
                     f"radar saw flooding at this exact spot in {months} of Oct 2014 to Sep 2024")
    elif pixel.get("months") == 0 and not radar.get("available"):
        parts.append("Sentinel-1 radar saw no flooding at this exact spot from Oct 2014 to Sep 2024")
    if not news.get("available") and not radar.get("available"):
        # the news events and the radar months live in the Archive's context mirror
        parts.insert(0, "the flood-event mirror is not published yet")
    if not parts:
        return "Flood history is not available here yet."
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def flood_history(lat: float, lon: float, *, radius_km: float = 25.0, limit: int = 10,
                  repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """What flooded near a point before: news events (Groundsource) and radar detections (Sentinel-1, 2014-2024).

    Events count when the centre of their affected area lies within ``radius_km``. ``recent`` lists the
    ``limit`` latest news events; ``by_year`` counts them per year; ``radar.months`` counts flood
    detections per month within the radius, and ``radar.pixel`` the months flooded at the point itself.
    """
    lat, lon = check_point(lat, lon)
    radius_km = max(1.0, min(float(radius_km or 25.0), 200.0))
    sources = ["groundsource", "microsoft_floods"]
    try:
        bbox = radius_bbox(lat, lon, radius_km)
        keys = cells_for_bbox(*bbox)

        def keep(la: float, lo: float) -> float | bool:
            d = haversine_km(lat, lon, la, lo)
            return d if d <= radius_km else False

        news_rows = read_cells("groundsource", keys, repo_id=repo_id)
        if news_rows is None:
            news: dict[str, Any] = {"available": False, "note": MIRROR_MISSING.format(what="Groundsource")}
        else:
            events = _news_events(news_rows, keep)
            news = {"available": True, "n_events": len(events), "by_year": _by_year([e["start"] for e in events]),
                    "first": min((e["start"] for e in events if e["start"]), default=None),
                    "latest": max((e["start"] for e in events if e["start"]), default=None),
                    "recent": events[:max(0, int(limit))]}
        radar_rows = read_cells("microsoft_floods", keys, repo_id=repo_id)
        radar: dict[str, Any] = {"period": list(MS_PERIOD)}
        if radar_rows is None:
            radar.update(available=False, note=MIRROR_MISSING.format(what="Sentinel-1 flood"))
        else:
            months = _radar_months(radar_rows, keep)
            radar.update(available=True, months=months, months_detected=len(months),
                         years=sorted({m[:4] for m in months}))
        try:
            radar["pixel"] = radar_recurrence(lat, lon)
        except Exception as exc:  # noqa: BLE001 - the pixel read is a bonus, never the failure
            radar["pixel"] = {"months": None, "status": f"unreadable: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return failed("flood_history", sources, exc)
    where = f"within {radius_km:g} km"
    return layer_result("flood_history", sources, ok=True, lat=lat, lon=lon, radius_km=radius_km, news=news,
                        radar=radar, summary=_summary(news, radar, where))


def flood_history_area(west: float, south: float, east: float, north: float, *, limit: int = 10,
                       repo_id: str = DEFAULT_REPO_ID, max_points: int = 2000) -> dict[str, Any]:
    """News flood events and radar flood months inside a box; ``points`` holds up to ``max_points`` events
    as (lat, lon, start) for a map layer."""
    bbox = check_bbox(west, south, east, north)
    sources = ["groundsource", "microsoft_floods"]
    try:
        keys = cells_for_bbox(*bbox)
        if len(keys) > 64:
            raise ValueError("the box is too large for flood history (keep it under about 16 x 16 degrees)")

        def keep(la: float, lo: float) -> bool:
            return True if in_bbox(la, lo, bbox) else False

        news_rows = read_cells("groundsource", keys, repo_id=repo_id)
        if news_rows is None:
            news: dict[str, Any] = {"available": False, "note": MIRROR_MISSING.format(what="Groundsource")}
            points: list[list[Any]] = []
        else:
            events = _news_events(news_rows, keep)
            news = {"available": True, "n_events": len(events), "by_year": _by_year([e["start"] for e in events]),
                    "first": min((e["start"] for e in events if e["start"]), default=None),
                    "latest": max((e["start"] for e in events if e["start"]), default=None),
                    "recent": events[:max(0, int(limit))]}
            points = [[e["lat"], e["lon"], e["start"]] for e in events[:max(0, int(max_points))]]
        radar_rows = read_cells("microsoft_floods", keys, repo_id=repo_id)
        radar: dict[str, Any] = {"period": list(MS_PERIOD)}
        if radar_rows is None:
            radar.update(available=False, note=MIRROR_MISSING.format(what="Sentinel-1 flood"))
        else:
            months = _radar_months(radar_rows, keep)
            radar.update(available=True, months=months, months_detected=len(months),
                         years=sorted({m[:4] for m in months}))
    except Exception as exc:  # noqa: BLE001
        return failed("flood_history", sources, exc)
    return layer_result("flood_history", sources, ok=True, bbox=list(bbox), news=news, radar=radar, points=points,
                        points_truncated=bool(news.get("n_events", 0) > len(points)),
                        summary=_summary(news, radar, "in this box"))


# ── dams ─────────────────────────────────────────────────────────────────────


def _reservoir_link() -> dict[str, str]:
    """Global Water Watch, linked only: its reservoir series are not read or mirrored (licence not confirmed)."""
    from aquascope.registry import CONTEXT_LAYERS

    m = CONTEXT_LAYERS["global_water_watch"]
    return {"label": m.label, "url": m.homepage, "note": "linked only; its data licence is not confirmed"}


def _dam(r: dict[str, str]) -> dict[str, Any]:
    year = num(r.get("year"))
    lat, lon = num(r.get("lat")), num(r.get("lon"))
    if lat == 0 and lon == 0:  # a mirror built from GDW's LAT_DAM/LONG_DAM parked ~35,000 barriers at 0, 0
        lat = lon = None
    return {
        "name": r.get("name") or r.get("reservoir") or "unnamed", "reservoir": r.get("reservoir") or None,
        "river": r.get("river") or None, "country": r.get("country") or None,
        "year": int(year) if year and year > 0 else None, "height_m": num(r.get("height_m")),
        "capacity_mcm": num(r.get("capacity_mcm")), "main_use": r.get("main_use") or None,
        "dor_pc": num(r.get("dor_pc")), "lat": lat, "lon": lon,
        "gdw_id": r.get("gdw_id") or None,
    }


def _dam_line(d: dict[str, Any]) -> str:
    bits = [d["name"]]
    if d.get("river"):
        bits.append(f"on the {d['river']}")
    if d.get("distance_km") is not None:
        bits.append(f"{d['distance_km']:.0f} km away")
    extra = []
    if d.get("year"):
        extra.append(f"built {d['year']}")
    if d.get("capacity_mcm"):
        extra.append(f"{d['capacity_mcm']:,.0f} million m3")
    return " ".join(bits) + (f" ({', '.join(extra)})" if extra else "")


def dams(lat: float, lon: float, *, radius_km: float = 50.0, limit: int = 5,
         repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """Dams and barriers within ``radius_km`` of a point, nearest first (Global Dam Watch v1.0)."""
    lat, lon = check_point(lat, lon)
    radius_km = max(1.0, min(float(radius_km or 50.0), 300.0))
    try:
        rows = read_cells("dams", cells_for_bbox(*radius_bbox(lat, lon, radius_km)), repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        return failed("dams", ["dams"], exc)
    if rows is None:
        return layer_result("dams", ["dams"], ok=True, lat=lat, lon=lon, available=False, n_dams=None, nearest=[],
                            note=MIRROR_MISSING.format(what="Global Dam Watch"),
                            summary="Dams are not available yet (the mirror is not published).")
    found = []
    for r in rows:
        d = _dam(r)
        if d["lat"] is None or d["lon"] is None:
            continue
        dist = haversine_km(lat, lon, d["lat"], d["lon"])
        if dist <= radius_km:
            d["distance_km"] = round(dist, 1)
            found.append(d)
    found.sort(key=lambda d: d["distance_km"])
    total_cap = sum(d["capacity_mcm"] or 0 for d in found)
    if found:
        summary = (f"{len(found)} dam{'s' if len(found) != 1 else ''} within {radius_km:g} km; nearest: "
                   f"{_dam_line(found[0])}.")
    else:
        summary = f"No dams in Global Dam Watch within {radius_km:g} km."
    return layer_result("dams", ["dams"], ok=True, lat=lat, lon=lon, available=True, radius_km=radius_km,
                        n_dams=len(found), total_capacity_mcm=round(total_cap, 1) if found else 0.0,
                        nearest=found[:max(0, int(limit))], see_also=_reservoir_link(), summary=summary)


def dams_area(west: float, south: float, east: float, north: float, *, limit: int = 5,
              repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """Dams inside a box: how many, their combined storage, and the largest by capacity."""
    bbox = check_bbox(west, south, east, north)
    try:
        keys = cells_for_bbox(*bbox)
        if len(keys) > 64:
            raise ValueError("the box is too large for the dam list (keep it under about 16 x 16 degrees)")
        rows = read_cells("dams", keys, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        return failed("dams", ["dams"], exc)
    if rows is None:
        return layer_result("dams", ["dams"], ok=True, bbox=list(bbox), available=False, n_dams=None, largest=[],
                            note=MIRROR_MISSING.format(what="Global Dam Watch"),
                            summary="Dams are not available yet (the mirror is not published).")
    inside = [d for d in (_dam(r) for r in rows)
              if d["lat"] is not None and d["lon"] is not None and in_bbox(d["lat"], d["lon"], bbox)]
    inside.sort(key=lambda d: d["capacity_mcm"] or 0, reverse=True)
    total = sum(d["capacity_mcm"] or 0 for d in inside)
    summary = (f"{len(inside)} dam{'s' if len(inside) != 1 else ''} in this box, about {total:,.0f} million m3 of "
               f"storage; largest: {_dam_line(inside[0])}." if inside else "No dams in Global Dam Watch in this box.")
    return layer_result("dams", ["dams"], ok=True, bbox=list(bbox), available=True, n_dams=len(inside),
                        total_capacity_mcm=round(total, 1), largest=inside[:max(0, int(limit))],
                        see_also=_reservoir_link(), summary=summary)
