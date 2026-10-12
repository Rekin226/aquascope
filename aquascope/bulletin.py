"""The state of the rivers, once a month, in the style of WMO HydroSOS (#523).

:func:`status_bulletin` places every Archive gauge's monthly mean flow against the same calendar month in the
other years of its record:

* the month counts when it has at least :data:`MIN_MONTH_DAYS` days of data, and so does every year it is compared
  with; a gauge needs :data:`MIN_YEARS` such other years, or it is left out and counted;
* the percentile is the mid-rank one, and the class is one of the five that the USGS National Water Dashboard and
  WMO HydroSOS use (much below normal, below, normal, above, much above), the same thresholds as
  :func:`aquascope.nownext.flow_status`;
* the gauges are rolled up per country (the catalogue's ISO code) and per river basin (BasinATLAS ``main_bas``,
  the level-12 sub-basin at the river's outlet), the new monthly records and the furthest from normal are named,
  and the coverage says how many gauges were looked at, how many were classed and why the rest were not;
* a short summary paragraph is written from those numbers by rules, never by a model.

:func:`bulletin_document` turns a bulletin into the Studio's :class:`~aquascope.studio.document.Document`, so it
renders to the same print-ready HTML and Markdown as a study report, with one map of the gauges coloured by class.

The monthly workflow (``.github/workflows/bulletin.yml``) runs ``python -m aquascope.bulletin build`` and
``publish``, which write only ``bulletins/<YYYY-MM>/`` and ``bulletins/index.json`` in the Archive dataset. Without a
local copy, :func:`status_bulletin` reads the published bulletin for the month first and builds one from the
Archive's discharge bundles only when there is none.

Licences. Only sources the Archive mirrors under terms that allow a derived product without conditions beyond
attribution are used (public domain, CC BY, OGL, Licence Ouverte, dl-de/by and the like); a share-alike source is
left out and named. Basins are BasinATLAS (HydroATLAS v1.0, CC BY 4.0); the map's country outlines are Natural
Earth (public domain).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_MONTH_DAYS",
    "MIN_YEARS",
    "build_bulletin",
    "bulletin_document",
    "classify_station",
    "compute_bulletin",
    "latest_month",
    "licence_ok",
    "monthly_means",
    "parse_month",
    "read_published",
    "render_bulletin",
    "status_bulletin",
    "write_bulletin",
]

FOLDER = "bulletins"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
VARIABLE = "discharge"
#: Days of data a month needs before its mean is used, in the month reported and in every year it is compared with.
MIN_MONTH_DAYS = 25
#: Other years with a usable month a gauge needs before it is classed (as for today against normal).
MIN_YEARS = 10
#: Classed gauges a basin needs before it gets a row of its own.
MIN_BASIN_GAUGES = 3
#: Classed gauges a country needs before the summary calls it the driest or the wettest.
MIN_COUNTRY_GAUGES = 10
#: The furthest from normal are named among gauges whose usual flow for the month is at least this (m3/s), so a
#: trickle that doubled does not top the list.
MIN_USUAL_CMS = 1.0
#: How many gauges each notable list names.
NOTABLE_N = 5
#: The agency is asked for the newest days only when the month ended this recently (its fetch covers a year).
TOP_UP_WINDOW_DAYS = 330

#: The class colours: the same as the Explorer's (explorer/src/now-core.js), brown to teal, colour-blind safe.
CLASS_COLOURS = {"much_below": "#a6611a", "below": "#dfc27d", "normal": "#7b8794", "above": "#80cdc1",
                 "much_above": "#018571"}
NO_CLASS_COLOUR = "#d5dbe0"

#: The countries the catalogue covers today (ISO 3166-1 alpha-3); any other code is shown as the code.
COUNTRY_NAMES = {
    "AUS": "Australia", "BRA": "Brazil", "CHL": "Chile", "DEU": "Germany", "FRA": "France", "GBR": "United Kingdom",
    "GRC": "Greece", "IRL": "Ireland", "JPN": "Japan", "POL": "Poland", "TWN": "Taiwan", "USA": "United States",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

REASONS = {
    "no_days": "no day of data in the month",
    "few_days": f"fewer than {MIN_MONTH_DAYS} days of data in the month",
    "short_record": f"fewer than {MIN_YEARS} other years with {MIN_MONTH_DAYS} days in that month",
}

METHOD = {
    "name": "Monthly flow against the same month in other years (HydroSOS-style status)",
    "text": f"Each gauge's mean daily flow over the month (at least {MIN_MONTH_DAYS} days of data) ranked against "
    f"the same calendar month in every other year of its record that also has {MIN_MONTH_DAYS} days (mid-rank "
    f"percentile), then classed as the USGS National Water Dashboard and WMO HydroSOS do: much below normal (under "
    f"10), below (10 to 24), normal (25 to 75), above (76 to 90), much above (over 90). A gauge needs at least "
    f"{MIN_YEARS} such other years. A new record is a monthly mean above (or below) every other year's. Basins are "
    "BasinATLAS river basins (the level-12 sub-basin at the outlet). The furthest from normal are ranked by the "
    "month's mean as a share of the usual (median) one, among gauges classed above (or below) normal whose usual flow "
    f"is at least {MIN_USUAL_CMS:g} m³/s; new records at such gauges are named first.",
    "citation": "WMO (2022). Hydrological Status and Outlook System (HydroSOS) implementation plan; USGS National "
    "Water Dashboard, streamflow percentile classes.",
}


# ── small helpers ────────────────────────────────────────────────────────────


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def _pct(part: int, whole: int) -> int:
    return int(round(100.0 * part / whole)) if whole else 0


def parse_month(month: Any = None, *, today: Any = None) -> tuple[int, int]:
    """``"2026-09"`` (or a date, or ``"2026-09-14"``) as ``(2026, 9)``; None is the last full month (UTC)."""
    if month is None or month == "":
        t = today if isinstance(today, date) else (date.fromisoformat(str(today)[:10]) if today else _today())
        first = t.replace(day=1) - timedelta(days=1)
        return first.year, first.month
    if isinstance(month, (date, datetime)):
        return month.year, month.month
    text = str(month).strip()
    try:
        y, m = int(text[:4]), int(text[5:7])
        if len(text) < 7 or text[4] not in "-/" or not 1 <= m <= 12:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"not a month: {month!r} (use YYYY-MM)") from exc
    return y, m


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def month_label(year: int, month: int) -> str:
    return f"{MONTHS[month - 1]} {year}"


def _month_end(year: int, month: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def licence_ok(source: str) -> bool:
    """Whether a status derived from ``source`` may be published: the Archive mirrors it, and its licence asks for
    nothing beyond attribution (no share-alike, non-commercial or no-derivatives term)."""
    from aquascope.registry import SOURCES

    info = SOURCES.get(source)
    if info is None or not info.redistributable:
        return False
    words = str(info.license or "").upper().replace("_", "-").replace(" ", "-").split("-")
    return not ({"SA", "NC", "ND"} & set(words)) and "UNKNOWN" not in words


#: Country names that take "the" in a sentence.
_WITH_THE = {"GBR", "USA"}


def _country_phrase(c: dict[str, Any], start: bool = False) -> str:
    name = c.get("name") or country_name(c.get("country"))
    if str(c.get("country") or "").upper() in _WITH_THE:
        return ("The " if start else "the ") + name
    return name


def country_name(code: Any) -> str:
    c = str(code or "").strip().upper()
    return COUNTRY_NAMES.get(c, c or "unknown")


# ── the numbers ─────────────────────────────────────────────────────────────


def monthly_means(frame: Any, month: int) -> Any:
    """Per gauge and year, the mean and the day count of calendar ``month`` from a daily frame with ``source``,
    ``station_id``, ``date`` and ``value`` (m3/s). Returns ``source, station_id, year, mean, n_days``."""
    import pandas as pd

    cols = ["source", "station_id", "year", "mean", "n_days"]
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=cols)
    df = frame[["source", "station_id", "date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["date"].dt.month == int(month)].dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["day"] = df["date"].dt.normalize()
    daily = df.groupby(["source", "station_id", "day"], sort=False)["value"].mean().reset_index()
    daily["year"] = daily["day"].dt.year
    out = daily.groupby(["source", "station_id", "year"], sort=True)["value"].agg(["mean", "count"]).reset_index()
    return out.rename(columns={"count": "n_days"})[cols]


def classify_station(years: Any, means: Any, days: Any, year: int, *, min_days: int = MIN_MONTH_DAYS,
                     min_years: int = MIN_YEARS) -> dict[str, Any]:
    """One gauge's month against the same month in its other years.

    ``years``, ``means`` and ``days`` are parallel sequences (one entry per year that has any day in the month).
    Returns ``{"reason": ...}`` when the gauge cannot be classed (``no_days``, ``few_days``, ``short_record``), else
    ``value``, ``n_days``, ``percentile``, ``class``, ``label``, ``n_years``, ``median``, ``ratio`` (value over
    median), ``rank`` (1 = the highest of the ``n_years + 1``) and ``record`` (``"high"``, ``"low"`` or None, with
    the previous record and its year).
    """
    import numpy as np

    from aquascope.nownext import _class_of

    ys = [int(y) for y in years]
    vals = [float(v) for v in means]
    ns = [int(n) for n in days]
    if year not in ys:
        return {"reason": "no_days", "n_days": 0}
    i = ys.index(year)
    if ns[i] < min_days:
        return {"reason": "few_days", "n_days": ns[i]}
    value = vals[i]
    ref = [(y, v) for y, v, n in zip(ys, vals, ns) if y != year and n >= min_days and math.isfinite(v)]
    if len(ref) < min_years:
        return {"reason": "short_record", "n_days": ns[i], "n_years": len(ref)}
    arr = np.array([v for _, v in ref], dtype=float)
    below, equal = float((arr < value).sum()), float((arr == value).sum())
    pct = 100.0 * (below + 0.5 * equal) / len(arr)
    cls = _class_of(pct)
    median = float(np.median(arr))
    out: dict[str, Any] = {
        "value": _num(value), "n_days": ns[i], "percentile": round(pct, 1), "class": cls["id"], "label": cls["label"],
        "n_years": len(ref), "median": _num(median),
        "ratio": _num(value / median, 3) if median > 0 and value >= 0 else None,
        "rank": int((arr > value).sum()) + 1, "record": None,
        "first_year": min(ys), "last_year": max(ys),
    }
    hi_y, hi_v = max(ref, key=lambda r: r[1])
    lo_y, lo_v = min(ref, key=lambda r: r[1])
    if value > hi_v:
        out.update(record="high", previous=_num(hi_v), previous_year=hi_y)
    elif value < lo_v:
        out.update(record="low", previous=_num(lo_v), previous_year=lo_y)
    return out


def _group_rows(means: Any) -> dict[tuple[str, str], tuple[list[int], list[float], list[int]]]:
    groups: dict[tuple[str, str], tuple[list[int], list[float], list[int]]] = {}
    if means is None or len(means) == 0:
        return groups
    for (src, sid), g in means.groupby(["source", "station_id"], sort=True):
        groups[(str(src), str(sid))] = (g["year"].astype(int).tolist(), g["mean"].astype(float).tolist(),
                                        g["n_days"].astype(int).tolist())
    return groups


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    from aquascope.nownext import STATUS_CLASSES

    out = {c["id"]: 0 for c in STATUS_CLASSES}
    for r in rows:
        if r.get("class") in out:
            out[r["class"]] += 1
    return out


def _median_class(pcts: list[float]) -> tuple[float | None, dict[str, Any] | None]:
    from aquascope.nownext import _class_of

    if not pcts:
        return None, None
    import numpy as np

    med = float(np.median(np.array(pcts, dtype=float)))
    return round(med, 1), _class_of(med)


def _rollup(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        k = r.get(key)
        if k is None or k == "":
            continue
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in groups.items():
        med, cls = _median_class([r["percentile"] for r in rs])
        out.append({key: k, "n": len(rs), "counts": _counts(rs), "median_percentile": med,
                    "median_class": cls["id"] if cls else None, "median_label": cls["label"] if cls else None,
                    "records_high": sum(r.get("record") == "high" for r in rs),
                    "records_low": sum(r.get("record") == "low" for r in rs),
                    "sources": sorted({r["source"] for r in rs})})
    out.sort(key=lambda d: (-d["n"], str(d[key])))
    return out


def _gauge_name(r: dict[str, Any]) -> str:
    return str(r.get("name") or r.get("station_id") or "")


def _notable(rows: list[dict[str, Any]], n: int = NOTABLE_N) -> dict[str, Any]:
    # A tidal or canal gauge can report a negative monthly mean; it is classed and counted, but not named.
    def named(r: dict[str, Any]) -> bool:
        return (r.get("value") or 0.0) >= 0 and (r.get("previous") or 0.0) >= 0

    # A trickle's record (0 m3/s against 0.001 before) is a record, and counted, but it is named after the
    # records of gauges whose usual flow is at least MIN_USUAL_CMS.
    def small(r: dict[str, Any]) -> bool:
        return (r.get("median") or 0.0) < MIN_USUAL_CMS

    all_highs = [r for r in rows if r.get("record") == "high"]
    all_lows = [r for r in rows if r.get("record") == "low"]
    highs = sorted((r for r in all_highs if named(r)), key=lambda r: (small(r), -(r.get("ratio") or 0.0)))
    lows = sorted((r for r in all_lows if named(r)), key=lambda r: (small(r), r.get("ratio") or 0.0))
    # The furthest above (below) normal are among the gauges classed above (below) normal, so a skewed record
    # whose month is far above its median but still in the normal band is not listed as an anomaly.
    big = [r for r in rows if r.get("ratio") is not None and not small(r)]
    above = sorted((r for r in big if r["ratio"] > 1 and r.get("class") in ("above", "much_above")),
                   key=lambda r: -r["ratio"])[:n]
    below = sorted((r for r in big if r["ratio"] < 1 and r.get("class") in ("below", "much_below")),
                   key=lambda r: r["ratio"])[:n]
    keep = ("source", "station_id", "name", "river", "country", "value", "median", "ratio", "percentile", "class",
            "label", "n_years", "record", "previous", "previous_year")

    def slim(rs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: r.get(k) for k in keep} for r in rs]

    return {"n_record_high": len(all_highs), "n_record_low": len(all_lows), "record_high": slim(highs[: 2 * n]),
            "record_low": slim(lows[: 2 * n]), "furthest_above": slim(above), "furthest_below": slim(below)}


def summary_text(b: dict[str, Any]) -> str:
    """The bulletin's summary paragraph, written from its numbers by rules (no model)."""
    label = b["label"]
    mon = MONTHS[int(b["month"][5:7]) - 1]
    n = int(b["coverage"]["classed"])
    if n == 0:
        return (f"No gauge had {MIN_MONTH_DAYS} days of flow in {label} and {MIN_YEARS} other years of {mon} to "
                "compare with, so this bulletin classes none.")
    countries = [c for c in b["countries"] if c.get("country")]
    where = (f"in {_plural(len(countries), 'country', 'countries')}" if len(countries) > 1
             else f"in {_country_phrase(countries[0])}" if countries else "")
    c = b["counts"]
    low, mid, high = c["much_below"] + c["below"], c["normal"], c["above"] + c["much_above"]
    parts = [f"In {label}, {_plural(n, 'gauge')} {where} had at least {MIN_MONTH_DAYS} days of flow and "
             f"{MIN_YEARS} or more other years of {mon} to compare with."]
    parts.append(f"Flow was normal at {_pct(mid, n)} % of them, below or much below normal at {_pct(low, n)} % and "
                 f"above or much above normal at {_pct(high, n)} %.")
    big = [c for c in countries if c["n"] >= MIN_COUNTRY_GAUGES and c.get("median_percentile") is not None]
    if len(big) >= 2:
        dry = min(big, key=lambda c: c["median_percentile"])
        wet = max(big, key=lambda c: c["median_percentile"])
        parts.append(f"{_country_phrase(dry, start=True)} was the driest, its median gauge at the "
                     f"{_ordinal(int(round(dry['median_percentile'])))} percentile ({dry['median_label']}); "
                     f"{_country_phrase(wet)} the wettest, at the {_ordinal(int(round(wet['median_percentile'])))} "
                     f"({wet['median_label']}).")
    nt = b["notable"]
    hi, lo = nt["n_record_high"], nt["n_record_low"]
    if hi == 0 and lo == 0:
        parts.append(f"No gauge set a new {mon} high or low.")
    else:
        bits = []
        if lo:
            bits.append(f"{_plural(lo, 'gauge')} set a new {mon} low")
        if hi:
            bits.append(f"{_plural(hi, 'gauge') if not lo else hi} set a new {mon} high" if lo
                        else f"{_plural(hi, 'gauge')} set a new {mon} high")
        parts.append(" and ".join(bits) + ".")
    return " ".join(parts)


def compute_bulletin(means: Any, month: Any, *, catalog: list[dict[str, Any]] | None = None,
                     basins: dict[tuple[str, str], int] | None = None,
                     areas: dict[tuple[str, str], float] | None = None, considered: list[tuple[str, str]] | None = None,
                     left_out_sources: dict[str, int] | None = None, topped_up: dict[str, Any] | None = None,
                     made: str | None = None) -> dict[str, Any]:
    """Assemble the bulletin from monthly means (:func:`monthly_means`, any number of sources).

    ``catalog`` rows (``source, station_id, name, river, country, latitude, longitude``) name and place the gauges;
    ``basins`` maps ``(source, station_id)`` to a BasinATLAS ``main_bas`` and ``areas`` to its catchment area (km2,
    which names a basin after its largest gauge); ``considered`` lists every gauge looked at
    (default: those in ``means``) so one with no day in the month is counted too; ``left_out_sources`` counts the
    gauges of sources whose licence keeps them out; ``topped_up`` is what the agency top-up did.
    """
    from aquascope.nownext import STATUS_CLASSES

    y, m = parse_month(month)
    key = month_key(y, m)
    groups = _group_rows(means)
    looked = list(dict.fromkeys(considered or []))
    seen = set(looked)
    looked += [k for k in groups if k not in seen]
    cat = {(str(r.get("source")), str(r.get("station_id"))): r for r in (catalog or [])}
    basins = basins or {}
    rows: list[dict[str, Any]] = []
    excluded = {k: 0 for k in REASONS}
    by_source: dict[str, dict[str, int]] = {}
    for src, sid in looked:
        bs = by_source.setdefault(src, {"considered": 0, "classed": 0})
        bs["considered"] += 1
        ys, vs, ns = groups.get((src, sid), ([], [], []))
        st = classify_station(ys, vs, ns, y)
        if "reason" in st:
            excluded[st["reason"]] += 1
            continue
        bs["classed"] += 1
        c = cat.get((src, sid), {})
        row = {"source": src, "station_id": sid, "name": c.get("name") or None, "river": c.get("river") or None,
               "country": (str(c.get("country")).upper() if c.get("country") else None),
               "latitude": _num(c.get("latitude"), 5), "longitude": _num(c.get("longitude"), 5),
               "basin_id": int(basins[(src, sid)]) if basins.get((src, sid)) else None, **st}
        rows.append(row)
    rows.sort(key=lambda r: (r["source"], r["station_id"]))
    countries = _rollup(rows, "country")
    for c in countries:
        c["name"] = country_name(c["country"])
    basin_rows = [b for b in _rollup(rows, "basin_id") if b["n"] >= MIN_BASIN_GAUGES]
    by_basin: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("basin_id"):
            by_basin.setdefault(r["basin_id"], []).append(r)
    area = areas or {}
    for br in basin_rows:
        members = by_basin.get(br["basin_id"], [])
        # Named after its largest gauge: the largest usual flow for the month, then the largest catchment.
        top = max(members, key=lambda r: (r.get("median") or 0.0, area.get((r["source"], r["station_id"])) or 0.0))
        br["label"] = str(top.get("river") or _gauge_name(top))
        br["largest_gauge"] = f"{top['source']}/{top['station_id']}"
        cs = sorted({str(r["country"]) for r in members if r.get("country")})
        br["countries"] = [country_name(x) for x in cs]
    counts = _counts(rows)
    n = len(rows)
    top_info = dict(topped_up or {})
    left = dict(left_out_sources or {})
    b: dict[str, Any] = {
        "month": key, "label": month_label(y, m), "title": "State of the rivers", "variable": VARIABLE,
        "unit": "m3/s", "made": made or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_month_days": MIN_MONTH_DAYS, "min_years": MIN_YEARS, "method": METHOD,
        "classes": [{"id": c["id"], "label": c["label"], "low": c["low"], "high": c["high"],
                     "colour": CLASS_COLOURS[c["id"]]} for c in STATUS_CLASSES],
        "counts": counts, "shares": {k: _pct(v, n) for k, v in counts.items()},
        "coverage": {"considered": len(looked), "classed": n, "excluded": excluded,
                     "excluded_text": {k: REASONS[k] for k in REASONS}, "by_source": by_source,
                     "left_out_sources": left, "top_up": top_info},
        "countries": countries, "basins": basin_rows, "notable": _notable(rows), "gauges": rows,
        "credits": _credits(sorted(by_source), left),
    }
    b["summary"] = summary_text(b)
    b["headline"] = _headline(b)
    return b


def _headline(b: dict[str, Any]) -> str:
    n = b["coverage"]["classed"]
    if not n:
        return f"{b['label']}: no gauge could be classed."
    c = b["counts"]
    low, high = c["much_below"] + c["below"], c["above"] + c["much_above"]
    return (f"{b['label']}: {_pct(low, n)} % of {n:,} gauges below normal, {_pct(c['normal'], n)} % normal, "
            f"{_pct(high, n)} % above.")


def _credits(sources: list[str], left_out: dict[str, int]) -> list[dict[str, str]]:
    from aquascope.registry import SOURCES

    out = []
    for s in sources:
        info = SOURCES.get(s)
        if info is not None:
            out.append({"source": s, "label": info.label, "licence": info.license, "attribution": info.attribution})
    for s in sorted(left_out):
        info = SOURCES.get(s)
        if info is not None:
            out.append({"source": s, "label": info.label, "licence": info.license, "attribution": info.attribution,
                        "left_out": f"left out: a status derived from it would carry its {info.license} terms"})
    out.append({"source": "basinatlas", "label": "River basins", "licence": "CC BY 4.0",
                "attribution": "BasinATLAS, HydroATLAS v1.0 (Linke et al. 2019), CC BY 4.0"})
    return out


# ── reading the Archive ─────────────────────────────────────────────────────


def _bulletin_sources(sources: list[str] | None, archive: Path | None) -> tuple[list[str], list[str]]:
    """(sources used, sources left out for their licence), among those with a discharge bundle."""
    from aquascope.archive.observations import HARVESTABLE

    if archive is not None:
        have = sorted(p.stem for p in (archive / "obs" / VARIABLE).glob("*.parquet"))
    else:
        have = sorted(s for s, v in HARVESTABLE.items() if VARIABLE in v)
    if sources:
        want = {s.strip() for s in sources if s and s.strip()}
        unknown = sorted(want - set(have))
        if unknown:
            raise ValueError(f"no mirrored discharge for {', '.join(unknown)}; the Archive has {', '.join(have)}")
        have = [s for s in have if s in want]
    return [s for s in have if licence_ok(s)], [s for s in have if not licence_ok(s)]


def _bundle_file(source: str, archive: Path | None, repo_id: str = DEFAULT_REPO) -> Path | None:
    if archive is not None:
        p = archive / "obs" / VARIABLE / f"{source}.parquet"
        return p if p.exists() else None
    import httpx

    from aquascope.archive.bundles import bundle_url
    from aquascope.archive.catalog import _download, cache_dir

    local = cache_dir() / f"{repo_id.replace('/', '__')}__{VARIABLE}__{source}.parquet"
    try:
        return _download(bundle_url(source, VARIABLE, repo_id), local, False)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise


def read_month(path: Any, source: str, month: int) -> tuple[Any, Any]:
    """From one discharge bundle: the rows of calendar ``month`` in every year (``source, station_id, date,
    value``) and each station's last day of data (``source, station_id, last``). Reads with pyarrow so a bundle of
    tens of millions of rows never becomes a pandas frame whole."""
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["station_id", "date", "value"])
    last = t.group_by("station_id").aggregate([("date", "max")]).to_pandas()
    last = last.rename(columns={"date_max": "last"})
    last["last"] = pd.to_datetime(last["last"])
    last.insert(0, "source", source)
    sub = t.filter(pc.equal(pc.month(t["date"]), int(month))).to_pandas(date_as_object=False)
    sub.insert(0, "source", source)
    return sub, last


def _catalog_rows(archive: Path | None) -> list[dict[str, Any]]:
    from aquascope.archive.catalog import load_stations

    try:
        if archive is not None and (archive / "stations.parquet").exists():
            return load_stations(path=archive / "stations.parquet")
        return load_stations()
    except Exception as exc:  # noqa: BLE001 - names and places are a nicety; the classes still stand
        logger.warning("bulletin: no catalogue (%s); gauges go unnamed", exc)
        return []


def _basin_ids(archive: Path | None) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], float]]:
    """(source, station_id) -> BasinATLAS main_bas, and -> the gauge's catchment area (km2), or empty maps."""
    import pandas as pd

    try:
        from aquascope.archive.basins import load_topology
        from aquascope.archive.similar import load_station_catchments

        cpath: Path | None = None
        tpath: Path | None = None
        if archive is not None:
            cpath = archive / "basins" / "station_catchments.parquet"
            tpath = archive / "basins" / "lev12_topology.parquet"
            if not (cpath.exists() and tpath.exists()):
                return {}, {}
        catch = load_station_catchments(path=cpath)
        topo = load_topology(path=tpath)[["hybas_id", "main_bas"]]
    except Exception as exc:  # noqa: BLE001 - basins are a rollup; the gauges stand without them
        logger.warning("bulletin: no basin table (%s); no basin rollup", exc)
        return {}, {}
    catch = catch.dropna(subset=["hybas_id"])
    catch["hybas_id"] = pd.to_numeric(catch["hybas_id"], errors="coerce").astype("Int64")
    topo = topo.assign(hybas_id=pd.to_numeric(topo["hybas_id"], errors="coerce").astype("Int64"))
    joined = catch.merge(topo, on="hybas_id", how="left")
    basins: dict[tuple[str, str], int] = {}
    areas: dict[tuple[str, str], float] = {}
    for r in joined.itertuples(index=False):
        k = (str(r.source), str(r.station_id))
        mb = getattr(r, "main_bas", None)
        if mb is not None and not pd.isna(mb):
            basins[k] = int(mb)
        a = _num(getattr(r, "area_km2", None)) or _num(getattr(r, "up_area", None))
        if a:
            areas[k] = a
    return basins, areas


def _top_up(means: Any, last: Any, year: int, month: int, *, limit: int, workers: int, time_budget_s: float | None,
            today: date) -> tuple[Any, dict[str, Any]]:
    """Ask the agencies for the month's days the mirror does not have yet, for gauges that would be classed with
    them: a short month, enough other years, a record that stops before the month's end, a source the package
    reaches directly. At most ``limit`` gauges, within ``time_budget_s``."""
    import pandas as pd

    from aquascope.explore import DIRECT_FETCH_SOURCES

    info: dict[str, Any] = {"asked": 0, "added": 0, "failed": 0, "skipped_cap": 0, "skipped_time": 0}
    end = _month_end(year, month)
    if limit <= 0 or (today - end).days > TOP_UP_WINDOW_DAYS or today <= end:
        return means, info
    groups = _group_rows(means)
    lastd = {(str(r.source), str(r.station_id)): r.last.date() for r in last.itertuples(index=False)}
    cands = []
    for (src, sid), (ys, vs, ns) in groups.items():
        if src not in DIRECT_FETCH_SOURCES:
            continue
        have = ns[ys.index(year)] if year in ys else 0
        others = sum(1 for y, n in zip(ys, ns) if y != year and n >= MIN_MONTH_DAYS)
        ld = lastd.get((src, sid))
        # A gauge whose record stops more than a year before the month has most likely closed: asking for it
        # would only spend the time budget.
        if (have < MIN_MONTH_DAYS and others >= MIN_YEARS and ld is not None and ld < end
                and (end - ld).days <= 366):
            cands.append((src, sid, ld, have))
    cands.sort(key=lambda c: (-c[3], c[0], c[1]))
    if len(cands) > limit:
        info["skipped_cap"] = len(cands) - limit
        cands = cands[:limit]
    start = time.monotonic()

    def one(c: tuple[str, str, date, int]) -> tuple[tuple[str, str], Any, str]:
        src, sid, ld, _ = c
        if time_budget_s is not None and time.monotonic() - start > time_budget_s:
            return (src, sid), None, "time"
        from aquascope.explore import fetch_series
        from aquascope.nownext import _as_daily

        try:
            got = fetch_series(src, sid, years=1, prefer_archive=False, variable=VARIABLE)
        except Exception as exc:  # noqa: BLE001 - one agency failing must not sink the bulletin
            logger.info("bulletin top-up failed for %s/%s: %s", src, sid, exc)
            return (src, sid), None, "failed"
        s = _as_daily(got.get("series"))
        if s.empty or got.get("variable") not in (None, VARIABLE):
            return (src, sid), None, "failed"
        s = s[(s.index.date > ld) & (s.index.year == year) & (s.index.month == month)]
        return (src, sid), s, "ok"

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, cands))
    extra = []
    for (src, sid), s, how in results:
        if how == "time":
            info["skipped_time"] += 1
            continue
        info["asked"] += 1
        if how == "failed" or s is None:
            info["failed"] += 1
            continue
        if len(s):
            info["added"] += 1
            extra.append((src, sid, s))
    if not extra:
        return means, info
    means = means.copy()
    for src, sid, s in extra:
        mask = (means["source"] == src) & (means["station_id"] == sid) & (means["year"] == year)
        if mask.any():
            i = means.index[mask][0]
            n0, m0 = int(means.at[i, "n_days"]), float(means.at[i, "mean"])
            n1 = n0 + len(s)
            means.at[i, "mean"] = (m0 * n0 + float(s.sum())) / n1
            means.at[i, "n_days"] = n1
        else:
            means = pd.concat([means, pd.DataFrame([{"source": src, "station_id": sid, "year": year,
                                                     "mean": float(s.mean()), "n_days": len(s)}])],
                              ignore_index=True)
    return means, info


def build_bulletin(month: Any = None, sources: list[str] | None = None, *, archive: str | Path | None = None,
                   top_up: int = 0, workers: int = 4, time_budget_s: float | None = None,
                   max_items: int | None = None, today: Any = None) -> dict[str, Any]:
    """Build the month's bulletin from the Archive's discharge bundles: a local copy of the dataset (``archive``,
    with ``stations.parquet``, ``obs/discharge/*.parquet`` and ``basins/``), else the published files (downloaded
    once and cached a day, a few hundred MB). ``top_up`` asks the agencies for up to that many gauges' missing days
    (see :func:`_top_up`); ``max_items`` keeps only the first N gauges, for a smoke run."""
    import pandas as pd

    y, m = parse_month(month, today=today)
    t = today if isinstance(today, date) else (date.fromisoformat(str(today)[:10]) if today else _today())
    if _month_end(y, m) >= t:
        raise ValueError(f"{month_label(y, m)} is not over yet; a bulletin is for a finished month")
    root = Path(archive) if archive else None
    use, left = _bulletin_sources(sources, root)
    parts, lasts = [], []
    left_counts: dict[str, int] = {}
    for src in use + left:
        path = _bundle_file(src, root)
        if path is None:
            continue
        sub, last = read_month(path, src, m)
        if src in left:
            left_counts[src] = int(len(last))
            continue
        parts.append(sub)
        lasts.append(last)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["source", "station_id", "date", "value"])
    last = pd.concat(lasts, ignore_index=True) if lasts else pd.DataFrame(columns=["source", "station_id", "last"])
    if max_items:
        keep = last.sort_values(["source", "station_id"]).groupby("source").head(max(1, max_items // max(1, len(use))))
        keep = keep.head(max_items)
        last = keep
        frame = frame.merge(keep[["source", "station_id"]], on=["source", "station_id"])
    means = monthly_means(frame, m)
    means, top = _top_up(means, last, y, m, limit=int(top_up or 0), workers=workers, time_budget_s=time_budget_s,
                         today=t)
    basins, areas = _basin_ids(root)
    considered = [(str(r.source), str(r.station_id)) for r in last.itertuples(index=False)]
    b = compute_bulletin(means, month_key(y, m), catalog=_catalog_rows(root), basins=basins, areas=areas,
                         considered=considered, left_out_sources=left_counts, topped_up=top)
    b["origin"] = "built"
    b["smoke"] = bool(max_items)
    return b


# ── what is published ───────────────────────────────────────────────────────


def published_url(path: str, repo_id: str = DEFAULT_REPO) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


_CLIENT: Any = None


def _fetch_json(url: str) -> Any:
    """The Hub seam (tests replace it): a JSON file of the Archive, cached for an hour; None when there is none."""
    global _CLIENT
    if _CLIENT is None:
        from aquascope.archive.catalog import cache_dir
        from aquascope.utils.http_client import CachedHTTPClient

        _CLIENT = CachedHTTPClient(timeout=60.0, retries=1, cache_dir=cache_dir() / "bulletins",
                                   cache_ttl_seconds=3600)
    try:
        return _CLIENT.get_json(url)
    except Exception as exc:  # noqa: BLE001 - a missing bulletin is an answer, not a failure
        logger.info("no published file at %s (%s)", url, exc)
        return None


def latest_month(repo_id: str = DEFAULT_REPO) -> str | None:
    """The newest published bulletin's month (``YYYY-MM``), or None before the first one."""
    idx = _fetch_json(published_url(f"{FOLDER}/index.json", repo_id))
    return (idx or {}).get("latest") if isinstance(idx, dict) else None


def read_published(month: Any = None, repo_id: str = DEFAULT_REPO) -> dict[str, Any] | None:
    """The published bulletin for ``month`` (default: the latest), or None when there is none."""
    key = month_key(*parse_month(month)) if month else latest_month(repo_id)
    if not key:
        return None
    b = _fetch_json(published_url(f"{FOLDER}/{key}/bulletin.json", repo_id))
    if isinstance(b, dict) and b.get("month") == key:
        b["origin"] = "published"
        return b
    return None


def status_bulletin(month: Any = None, sources: list[str] | None = None, *, archive: str | Path | None = None,
                    rebuild: bool = False, top_up: int = 0, workers: int = 4, today: Any = None) -> dict[str, Any]:
    """The state of the rivers for ``month`` (``YYYY-MM``; default the latest published bulletin, else the last
    full month), HydroSOS style.

    Every Archive gauge with a mirrored discharge record covering the month: the monthly mean's percentile against
    the same month in its other years (25 days a month, 10 years, else left out and counted), the five classes, the
    roll-up per country and per BasinATLAS river basin, the notable gauges (new monthly highs and lows, the furthest
    from normal), the coverage, and a summary paragraph written by rules. The published bulletin is returned when
    there is one (``origin: "published"``) unless ``rebuild``, ``sources`` or ``archive`` is given; otherwise it is
    built from the discharge bundles (``origin: "built"``), with ``top_up`` gauges' missing days asked of the agency.
    """
    if not (rebuild or sources or archive):
        got = read_published(month)
        if got is not None:
            return got
    return build_bulletin(month, sources, archive=archive, top_up=top_up, workers=workers, today=today)


# ── the document ────────────────────────────────────────────────────────────


def _source_label(source: str) -> str:
    from aquascope.registry import SOURCES

    info = SOURCES.get(source)
    return info.label if info is not None else source


def _fmt_q(x: Any) -> str:
    from aquascope.studio.document.text import num

    return num(x) if x is not None else ""


def _share(x: Any) -> str:
    return f"{int(round(100 * float(x)))} %" if x is not None else ""


#: A map panel spans at most this many degrees of longitude and latitude; gauges further apart get their own panel.
PANEL_SPAN = (45.0, 32.0)
#: Panels drawn at most, and the gauges a cluster needs to get one (the rest are counted in the caption).
MAX_PANELS = 3
MIN_PANEL_GAUGES = 5


def map_regions(points: list[tuple[float, float]], *, max_panels: int = MAX_PANELS,
                min_gauges: int = MIN_PANEL_GAUGES) -> tuple[list[dict[str, Any]], int]:
    """Group gauge positions ``(lon, lat)`` into map panels: neighbouring 5-degree cells joined, then clusters of
    ``min_gauges`` or more joined again while the panel stays within :data:`PANEL_SPAN`. Returns the largest
    panels (``bbox`` as west, south, east, north and ``idx``, the positions in it), at most ``max_panels``, and how
    many gauges are in no panel."""
    cells: dict[tuple[int, int], list[int]] = {}
    for i, (lon, lat) in enumerate(points):
        cells.setdefault((math.floor(lon / 5.0), math.floor(lat / 5.0)), []).append(i)
    seen: set[tuple[int, int]] = set()
    comps: list[list[int]] = []
    for c in sorted(cells):
        if c in seen:
            continue
        stack, idx = [c], []
        seen.add(c)
        while stack:
            cx, cy = stack.pop()
            idx += cells[(cx, cy)]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(idx)

    def box(idx: list[int]) -> tuple[float, float, float, float]:
        xs, ys = [points[i][0] for i in idx], [points[i][1] for i in idx]
        return min(xs), min(ys), max(xs), max(ys)

    panels: list[list[int]] = []
    for comp in sorted(comps, key=len, reverse=True):
        if len(comp) < min_gauges:  # a stray gauge would stretch a panel across a continent; it is counted instead
            continue
        for p in panels:
            x0, y0, x1, y1 = box(p + comp)
            if x1 - x0 <= PANEL_SPAN[0] and y1 - y0 <= PANEL_SPAN[1]:
                p += comp
                break
        else:
            panels.append(list(comp))
    panels.sort(key=len, reverse=True)
    kept = [p for p in panels if len(p) >= min_gauges][:max_panels]
    left = len(points) - sum(len(p) for p in kept)
    return [{"bbox": box(p), "idx": sorted(p)} for p in kept], left


def map_png(b: dict[str, Any], *, dpi: int = 200, outlines: bool = True) -> tuple[bytes, int] | None:
    """The classed gauges coloured by class over Natural Earth country outlines, one panel per region
    (:func:`map_regions`), as PNG bytes and the number of gauges outside the panels; None without matplotlib or
    without a placed gauge."""
    pts = [r for r in b.get("gauges") or [] if r.get("latitude") is not None and r.get("longitude") is not None]
    panels, left = map_regions([(float(r["longitude"]), float(r["latitude"])) for r in pts])
    if not panels:
        return None
    try:
        from aquascope.viz import publication as pub

        pub.use_style()
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("bulletin map: matplotlib is not installed (pip install 'aquascope[viz]')")
        return None
    boxes = []
    for p in panels:
        x0, y0, x1, y1 = p["bbox"]
        pad = max(0.6, 0.06 * max(x1 - x0, y1 - y0))
        x0, x1, y0, y1 = x0 - pad, x1 + pad, max(-90.0, y0 - pad), min(90.0, y1 + pad)
        k = 1.0 / max(math.cos(math.radians((y0 + y1) / 2.0)), 0.2)
        boxes.append((x0, y0, x1, y1, k))
    widths = [(x1 - x0) / ((y1 - y0) * k) for x0, y0, x1, y1, k in boxes]
    height = max(2.2, min(3.6, pub.WIDTHS["full"] / max(sum(widths), 0.5)))
    fig, axes = plt.subplots(1, len(boxes), figsize=(pub.WIDTHS["full"], height + 0.45), layout="constrained",
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    countries = []
    if outlines:
        try:
            from aquascope.river_path import _countries

            countries = _countries()
        except Exception as exc:  # noqa: BLE001 - outlines are a backdrop; the gauges are the figure
            logger.info("bulletin map: no country outlines (%s)", exc)
    order = ["much_below", "below", "normal", "above", "much_above"]
    labels = {c["id"]: c["label"] for c in b["classes"]}
    totals = {k: sum(r.get("class") == k for r in pts) for k in order}
    for ax, (x0, y0, x1, y1, k), p in zip(axes[0], boxes, panels):
        for c in countries:
            bx0, by0, bx1, by1 = c.bbox
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            for ring in c.rings:
                ax.fill([q[0] for q in ring], [q[1] for q in ring], facecolor="#f2f2f0", edgecolor="#b9bfc4",
                        linewidth=0.3, zorder=1)
        sel = [pts[i] for i in p["idx"]]
        for cls in ["normal", "below", "above", "much_below", "much_above"]:  # the extremes drawn on top
            rs = [r for r in sel if r.get("class") == cls]
            if rs:
                ax.scatter([r["longitude"] for r in rs], [r["latitude"] for r in rs], s=7, c=CLASS_COLOURS[cls],
                           edgecolors="#33475a", linewidths=0.15, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(k)
        ax.tick_params(labelsize=6.5)
        ax.grid(False)
        names = sorted({country_name(r.get("country")) for r in sel if r.get("country")})
        ax.set_title(", ".join(names[:3]) + (" and more" if len(names) > 3 else ""), fontsize=7.5)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=CLASS_COLOURS[k],
                      markeredgecolor="#33475a", markeredgewidth=0.3) for k in order]
    fig.legend(handles, [f"{labels[k][:1].upper()}{labels[k][1:]} ({totals[k]:,})" for k in order],
               loc="outside lower center", ncol=5, fontsize=6.5, frameon=False, handletextpad=0.2,
               columnspacing=0.9)
    try:
        return pub.png_bytes(fig, dpi=dpi), left
    finally:
        plt.close(fig)


def bulletin_document(b: dict[str, Any], *, figure: bytes | None = None, outside: int = 0) -> Any:
    """The bulletin as a Studio :class:`~aquascope.studio.document.Document` (render with ``render_html`` or
    ``render_markdown``). ``figure`` is the map PNG (:func:`map_png`), ``outside`` the gauges it leaves out; without
    a figure the map is left out."""
    from aquascope.studio.document.model import Bullets, Callout, Document, Figure, Heading, KeyValue, Para, Table

    mon = MONTHS[int(b["month"][5:7]) - 1]
    cov = b["coverage"]
    n = int(cov["classed"])
    doc = Document(title=b.get("title") or "State of the rivers", subtitle=b["label"], kind="Monthly bulletin",
                   meta={"date": str(b.get("made", ""))[:10]}, status="")
    doc.add(KeyValue([("Month", b["label"]), ("Gauges classed", f"{n:,} of {int(cov['considered']):,} looked at"),
                      ("Made", str(b.get("made", ""))[:16].replace("T", " ") + " UTC"),
                      ("Method", "monthly mean against the same month in other years, HydroSOS classes")]))
    order = ["much_below", "below", "normal", "above", "much_above"]
    labels = {c["id"]: c["label"] for c in b["classes"]}
    if n:
        doc.add(Callout("In brief", _headline(b).split(": ", 1)[-1],
                        rows=[(labels[k][:1].upper() + labels[k][1:], f"{b['counts'][k]:,} gauges "
                               f"({b['shares'][k]} %)") for k in order]))
    doc.add(Para(b["summary"], style="lead"))
    if figure:
        far = (f" {_plural(outside, 'gauge')} far from the others {'is' if outside == 1 else 'are'} not drawn."
               if outside else "")
        doc.add(Figure("status_map", figure, f"The {n - outside:,} classed gauges coloured by where {b['label']}'s "
                       f"mean flow sits against {mon} in their other years. Country outlines: Natural Earth.{far}",
                       path="map.png"))
    cols = ["Much below", "Below", "Normal", "Above", "Much above"]
    if b["countries"]:
        doc.add(Heading("By country"))
        doc.add(Table("countries", ["Country", "Gauges", *cols, "Median percentile"],
                      [[c["name"], c["n"], *[c["counts"][k] for k in order],
                        f"{c['median_percentile']:.0f} ({c['median_label']})"] for c in b["countries"]],
                      "Classed gauges per country, by class, and the percentile of the median gauge.",
                      align=["l"] + ["r"] * 6 + ["l"],
                      notes=["A country is the catalogue's; the United Kingdom is England only (Environment "
                             "Agency)."] if any(c["country"] == "GBR" for c in b["countries"]) else []))
    if b["basins"]:
        doc.add(Heading("By river basin"))
        top = b["basins"][:15]
        doc.add(Table("basins", ["Basin (its largest gauge)", "Country", "Gauges", *cols, "Median percentile"],
                      [[bs["label"], ", ".join(bs.get("countries") or []), bs["n"], *[bs["counts"][k] for k in order],
                        f"{bs['median_percentile']:.0f}"] for bs in top],
                      f"The {len(top)} basins with the most classed gauges (BasinATLAS river basins with at least "
                      f"{MIN_BASIN_GAUGES}).", align=["l", "l"] + ["r"] * 7))
    nt = b["notable"]
    doc.add(Heading("Notable gauges"))
    items = []
    for kind, word in (("record_low", "low"), ("record_high", "high")):
        rs = nt[kind]
        total = nt[f"n_{kind}"]
        for r in rs[:NOTABLE_N]:
            items.append(f"**New {mon} {word}** at {_gauge_name(r)} ({country_name(r.get('country'))}): "
                         f"{_fmt_q(r['value'])} m³/s, past {_fmt_q(r.get('previous'))} m³/s in "
                         f"{r.get('previous_year')} ({r['n_years'] + 1} years of {mon}).")
        if total > NOTABLE_N:
            items.append(f"And {total - NOTABLE_N:,} more new {mon} {word}s.")
    if items:
        doc.add(Bullets(items))
    else:
        doc.add(Para(f"No gauge set a new {mon} high or low."))
    far = nt["furthest_above"] + nt["furthest_below"]
    if far:
        doc.add(Table("furthest", ["Gauge", "Country", f"{mon} mean (m³/s)", "Usual (m³/s)", "Of usual", "Class"],
                      [[_gauge_name(r), country_name(r.get("country")), _fmt_q(r["value"]), _fmt_q(r["median"]),
                        _share(r["ratio"]), r["label"] + (f", record {r['record']}" if r.get("record") else "")]
                       for r in far],
                      f"Furthest above and below normal: the month's mean as a share of the usual (median) {mon}, "
                      f"among gauges classed above or below normal whose usual flow is at least "
                      f"{MIN_USUAL_CMS:g} m³/s.",
                      align=["l", "l", "r", "r", "r", "l"]))
    doc.add(Heading("Coverage"))
    ex = cov["excluded"]
    left = cov.get("left_out_sources") or {}
    lines = [f"{int(cov['considered']):,} gauges with a mirrored discharge record were looked at and {n:,} classed."]
    for k, text in REASONS.items():
        if ex.get(k):
            lines.append(f"{_plural(int(ex[k]), 'gauge')} left out: {text}.")
    top = cov.get("top_up") or {}
    if top.get("asked"):
        lines.append(f"The agencies were asked for the missing days of {top['asked']:,} gauges and added days to "
                     f"{top['added']:,}.")
    for s, k in sorted(left.items()):
        lines.append(f"{_plural(int(k), 'gauge')} of {_source_label(s)} left out for the licence (see Data and "
                     "licences).")
    doc.add(Bullets(lines))
    rows = [[_source_label(s), f"{int(v['considered']):,}", f"{int(v['classed']):,}"]
            for s, v in sorted(cov["by_source"].items())]
    if rows:
        doc.add(Table("sources", ["Source", "Looked at", "Classed"], rows, "Gauges per source.",
                      align=["l", "r", "r"], compact=True))
    doc.add(Heading("Method"))
    doc.add(Para(b["method"]["text"]))
    doc.add(Para(b["method"]["citation"], style="small"))
    doc.add(Heading("Data and licences"))
    doc.add(Bullets([f"{c['label']}: {c['attribution']} ({c['licence']})"
                     + (f"; {c['left_out']}" if c.get("left_out") else "") for c in b.get("credits") or []]
                    + ["Country outlines: Natural Earth (public domain)."]))
    doc.add(Para("Made by AquaScope from the Archive (huggingface.co/datasets/Rekin226/aquascope-gauges). "
                 "Observed data as the agencies publish them, some of it provisional.", style="small"))
    return doc.finalize()


def render_bulletin(b: dict[str, Any], fmt: str = "html", *, figure: bytes | None = None, outside: int = 0) -> str:
    """The bulletin as print-ready HTML (the Studio's report styles) or Markdown."""
    from aquascope.studio.document.render_html import render_html
    from aquascope.studio.document.render_md import render_markdown
    from aquascope.studio.document.style import HouseStyle

    doc = bulletin_document(b, figure=figure, outside=outside)
    if fmt in ("md", "markdown"):
        return render_markdown(doc)
    return render_html(doc, HouseStyle(organisation="AquaScope", status=""))


STATUS_COLUMNS = ["source", "station_id", "month", "value", "n_days", "percentile", "class", "n_years", "median",
                  "ratio", "record", "country", "basin_id"]


def write_bulletin(b: dict[str, Any], out: str | Path, *, figure: bool = True) -> dict[str, Any]:
    """Write ``<out>/bulletins/<YYYY-MM>/``: ``bulletin.html``, ``bulletin.md``, ``bulletin.json``, ``map.png`` and
    ``status.parquet`` (one row per classed gauge, for the Explorer's map). Returns the paths written."""
    folder = Path(out) / FOLDER / b["month"]
    folder.mkdir(parents=True, exist_ok=True)
    drawn = map_png(b) if figure else None
    png, outside = drawn if drawn else (None, 0)
    written: dict[str, Any] = {}
    if png:
        (folder / "map.png").write_bytes(png)
        written["map"] = str(folder / "map.png")
    (folder / "bulletin.html").write_text(render_bulletin(b, "html", figure=png, outside=outside), encoding="utf-8")
    (folder / "bulletin.md").write_text(render_bulletin(b, "md", figure=png, outside=outside), encoding="utf-8")
    (folder / "bulletin.json").write_text(json.dumps(b, ensure_ascii=False, separators=(",", ":"), default=str),
                                         encoding="utf-8")
    written.update(html=str(folder / "bulletin.html"), markdown=str(folder / "bulletin.md"),
                   json=str(folder / "bulletin.json"))
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [{**{k: r.get(k) for k in STATUS_COLUMNS}, "month": b["month"]} for r in b["gauges"]]
        schema = pa.schema([("source", pa.string()), ("station_id", pa.string()), ("month", pa.string()),
                            ("value", pa.float64()), ("n_days", pa.int32()), ("percentile", pa.float64()),
                            ("class", pa.string()), ("n_years", pa.int32()), ("median", pa.float64()),
                            ("ratio", pa.float64()), ("record", pa.string()), ("country", pa.string()),
                            ("basin_id", pa.int64())])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / "status.parquet", compression="zstd")
        written["status"] = str(folder / "status.parquet")
    except ImportError:
        logger.info("bulletin: status.parquet needs pyarrow (pip install 'aquascope[archive]')")
    return written


def index_entry(b: dict[str, Any]) -> dict[str, Any]:
    return {"month": b["month"], "label": b["label"], "made": b.get("made"), "classed": b["coverage"]["classed"],
            "considered": b["coverage"]["considered"], "counts": b["counts"], "headline": b.get("headline"),
            "sources": sorted(b["coverage"]["by_source"])}


def update_index(previous: dict[str, Any] | None, b: dict[str, Any]) -> dict[str, Any]:
    """``bulletins/index.json`` with this month added (or replaced): every month newest first, and ``latest``."""
    months = [m for m in (previous or {}).get("months") or [] if isinstance(m, dict) and m.get("month") != b["month"]]
    months.append(index_entry(b))
    months.sort(key=lambda m: m["month"], reverse=True)
    return {"latest": months[0]["month"], "months": months, "folder": FOLDER,
            "about": "Monthly state-of-the-rivers bulletins (aquascope.bulletin, #523): bulletins/<YYYY-MM>/ holds "
                     "bulletin.html, bulletin.md, bulletin.json, map.png and status.parquet."}


def publish(out: str | Path, *, repo_id: str = DEFAULT_REPO, token: str | None = None) -> str:
    """Upload ``<out>/bulletins/<month>/`` and ``bulletins/index.json``, nothing else; refuses a smoke build."""
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    idx_path = src / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} is missing; run the build step first")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    month = (json.loads((src / "build.json").read_text(encoding="utf-8")) if (src / "build.json").exists()
             else {}).get("month") or idx["latest"]
    built = json.loads((src / month / "bulletin.json").read_text(encoding="utf-8"))
    if built.get("smoke"):
        raise RuntimeError("a smoke build is never published: its few gauges would stand for the month")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / FOLDER
        shutil.copytree(src / month, stage / month)
        shutil.copy2(idx_path, stage / "index.json")
        return publish_folder(Path(tmp), repo_id, token=token or os.environ.get("HF_TOKEN"),
                              commit_message=f"bulletin: {month} ({built['coverage']['classed']} gauges)",
                              allow_patterns=["*.parquet", "*.json", "*.html", "*.md", "*.png"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.bulletin", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build a month's bulletin from a local copy of the Archive")
    b.add_argument("--archive", required=True, help="stations.parquet, obs/discharge/*.parquet, basins/, bulletins/")
    b.add_argument("--out", required=True)
    b.add_argument("--month", default=None, help="YYYY-MM (default: the last full month)")
    b.add_argument("--top-up", type=int, default=0, help="Gauges whose missing days are asked of the agency")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--time-budget-min", type=float, default=None)
    b.add_argument("--smoke", action="store_true", help="A few gauges per source; never published")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "aquascope.collectors.base", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if a.cmd == "publish":
        print(publish(a.out, repo_id=a.repo))
        return 0
    started = time.monotonic()
    res = build_bulletin(a.month or None, archive=a.archive, top_up=a.top_up, workers=a.workers,
                         time_budget_s=a.time_budget_min * 60 if a.time_budget_min else None,
                         max_items=40 if a.smoke else None)
    written = write_bulletin(res, a.out)
    prev_path = Path(a.archive) / FOLDER / "index.json"
    prev = None
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except ValueError:
            prev = None
    root = Path(a.out) / FOLDER
    (root / "index.json").write_text(json.dumps(update_index(prev, res), indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    (root / "build.json").write_text(json.dumps({"month": res["month"], "smoke": res.get("smoke", False),
                                                 "seconds": round(time.monotonic() - started, 1)}), encoding="utf-8")
    print(json.dumps({"month": res["month"], "headline": res["headline"], "coverage": {
        k: v for k, v in res["coverage"].items() if k != "excluded_text"}, "files": written}, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
