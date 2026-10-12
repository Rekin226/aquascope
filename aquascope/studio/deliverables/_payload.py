"""Readers over the payloads the tools return, shared by the figure and the table makers.

The same thing is spelled several ways across the producers: a series is
``{"t", "v"}`` in :func:`aquascope.explore.analyze_series`, ``{"index",
"values"}`` in the workbench, ``points`` as ``[date, value]`` pairs from the
``get_timeseries`` tool; the annual maxima are ``annual_max`` with ``year``
and ``v`` at a station and a dated series from the workbench; the return
levels sit under ``ffa.fits`` at a station and at the top level of a
workbench ``return_periods`` result. Every reader here is defensive: it
returns ``None`` or an empty list when it finds nothing, and never raises on
a shape it does not know.
"""

from __future__ import annotations

import math
from typing import Any

#: Keys that are never a scalar worth tabulating (bulk data, prose, provenance).
BULK_KEYS = frozenset({
    "series", "observations", "points", "samples", "methods", "notes", "attribution", "license", "indices", "stations",
    "schedule", "per_season", "annual_max", "annual_maxima", "ffa", "fdc", "estimates", "similarity",
    "regression", "attributes", "sufficiency", "recon", "era5", "requested", "correlations", "parameters",
    "preview", "frame", "rows", "segments", "events", "propagation", "sgi", "eto", "features", "target",
    "skill", "features_used", "years_used", "empirical", "fit", "return_periods", "return_levels",
    "lower_bound", "upper_bound", "confidence_intervals", "params", "sub_indices", "components",
})


def num(x: Any) -> float | None:
    """A finite float, or None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def numbers(seq: Any) -> list[float | None]:
    """Every item of ``seq`` through :func:`num` (an empty list for anything that is not a sequence)."""
    if not isinstance(seq, (list, tuple)):
        return []
    return [num(x) for x in seq]


def is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def date_key(s: Any) -> str:
    """An ISO date string trimmed to the second, so ``numpy.datetime64`` accepts it whatever the producer wrote."""
    text = str(s)
    if "T" in text or " " in text:
        return text[:19].replace(" ", "T")
    return text[:10]


def year_of(s: Any) -> int | None:
    try:
        return int(str(s)[:4])
    except (TypeError, ValueError):
        return None


# ── the record, the unit, the period ─────────────────────────────────────


def variable_of(payload: dict[str, Any], default: str = "value") -> str:
    v = payload.get("variable") or payload.get("column") or default
    return str(v).replace("_", " ")


def unit_of(payload: dict[str, Any], default: str | None = None) -> str:
    u = payload.get("unit")
    if isinstance(u, str) and u:
        return u
    return default or ""


def _coord(lat: Any, lon: Any) -> str | None:
    la, lo = num(lat), num(lon)
    if la is None or lo is None:
        return None
    return f"{abs(la):.2f} {'N' if la >= 0 else 'S'}, {abs(lo):.2f} {'E' if lo >= 0 else 'W'}"


def site_point(payload: dict[str, Any] | None, site: dict[str, Any] | None = None) -> tuple[float, float] | None:
    """The site's (lat, lon) from the payload (``latitude``/``longitude`` or ``point``) or the ``site`` dict."""
    for d in (payload or {}, site or {}):
        if not isinstance(d, dict):
            continue
        p = d.get("point") if isinstance(d.get("point"), dict) else d
        lat = num(p.get("lat", p.get("latitude")))
        lon = num(p.get("lon", p.get("longitude")))
        if lat is not None and lon is not None:
            return lat, lon
    return None


def record_name(payload: dict[str, Any], site: dict[str, Any] | None = None) -> str:
    """How the caption names the record: ``uk_ea 3400TH``, ``the ERA5 cell at 51.42 N, 0.31 W``, ``the table``."""
    src, sid = payload.get("source"), payload.get("station_id")
    if src and sid:
        # The Analysts add the catalog name of the station the step ran on, so a caption says
        # "Kingston (uk_ea 3400TH)" rather than an id alone.
        name = payload.get("station_name")
        label = station_label(str(src), str(sid))
        if isinstance(name, str) and name.strip() and name.strip() != str(sid):
            return f"{name.strip()} ({label})"
        return label
    if payload.get("station") and isinstance(payload["station"], dict):
        st = payload["station"]
        if st.get("source") and st.get("station_id"):
            return f"{st['source']} {st['station_id']}"
    pt = site_point(payload, site)
    if pt is not None:
        return f"the site at {_coord(*pt)}"
    if payload.get("column"):
        return f"the table (column {payload['column']})"
    if payload.get("name"):
        return str(payload["name"])
    return "the record"


def station_label(source: str, station_id: str) -> str:
    """A station as a caption names it: ``USGS 01013500`` for ``("usgs", "USGS-01013500")`` (the agency prefix
    once, upper case), ``UK_EA 3400TH`` otherwise."""
    src, sid = str(source or ""), str(station_id or "")
    if sid.upper().startswith(src.upper() + "-"):
        sid = sid[len(src) + 1:]
    return f"{src.upper().replace('_', ' ')} {sid}".strip()


def period_of(payload: dict[str, Any], dates: list[str] | None = None) -> str:
    """``1986 to 2025`` from ``start``/``end``, else from the dates given."""
    y0 = year_of(payload.get("start")) if payload.get("start") else None
    y1 = year_of(payload.get("end")) if payload.get("end") else None
    if (y0 is None or y1 is None) and dates:
        y0, y1 = year_of(dates[0]), year_of(dates[-1])
    if y0 is None or y1 is None:
        return ""
    return f"{y0}" if y0 == y1 else f"{y0} to {y1}"


# ── series and maxima ─────────────────────────────────────────────────────


def series_of(payload: dict[str, Any], key: str = "series", *,
              preserve_time: bool = False) -> tuple[list[str], list[float | None]] | None:
    """The record as ``(dates, values)`` from ``{"t","v"}``, ``{"index","values"}`` or ``points`` pairs."""
    stamp = (lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)) if preserve_time else date_key
    s = payload.get(key)
    if isinstance(s, dict):
        t = s.get("t", s.get("index"))
        v = s.get("v", s.get("values"))
        if isinstance(t, list) and isinstance(v, list) and t and len(t) == len(v):
            return [stamp(x) for x in t], numbers(v)
    pts = payload.get("points")
    if isinstance(pts, list) and pts and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in pts):
        return [stamp(p[0]) for p in pts], [num(p[1]) for p in pts]
    return None


def annual_maxima_of(payload: dict[str, Any]) -> tuple[list[int], list[float]] | None:
    """``(years, values)`` from ``annual_max`` / ``annual_maxima`` in either spelling (year lists or dated)."""
    for key in ("annual_max", "annual_maxima"):
        am = payload.get(key)
        if not isinstance(am, dict):
            continue
        years = am.get("year")
        vals = am.get("v", am.get("values"))
        if years is None and isinstance(am.get("index"), list):
            years = [year_of(x) for x in am["index"]]
        if isinstance(years, list) and isinstance(vals, list) and years and len(years) == len(vals):
            pairs = [(int(y), num(v)) for y, v in zip(years, vals) if y is not None and num(v) is not None]
            if pairs:
                return [p[0] for p in pairs], [p[1] for p in pairs]  # type: ignore[misc]
    ffa = payload.get("ffa")
    if isinstance(ffa, dict) and isinstance(ffa.get("annual_max"), dict):
        return annual_maxima_of({"annual_max": ffa["annual_max"]})
    return None


def resolution_word(dates: list[str]) -> str:
    """``Daily``, ``Monthly``, ``Annual`` or ``""`` from the median spacing of the dates."""
    if len(dates) < 3:
        return ""
    try:
        import numpy as np

        d = np.array([date_key(x) for x in dates[:2000]], dtype="datetime64[s]")
        gaps = np.diff(d).astype("timedelta64[s]").astype(float) / 86400.0
        med = float(np.median(gaps)) if len(gaps) else 0.0
    except (ValueError, TypeError):
        return ""
    if 0.9 <= med <= 1.1:
        return "Daily"
    if 27 <= med <= 32:
        return "Monthly"
    if 360 <= med <= 370:
        return "Annual"
    if 6.5 <= med <= 7.5:
        return "Weekly"
    return ""


# ── frequency analysis ────────────────────────────────────────────────────


def return_levels_of(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return periods and levels in one shape, whichever tool wrote them.

    Returns ``{"T": [...], "gev": [...] | None, "lp3": [...] | None, "boot": [...] | None,
    "lower": [...] | None, "upper": [...] | None, "band": str | None, "distribution": str | None,
    "empirical": (T, value) | None}``; None when there is nothing.
    """
    out: dict[str, Any] = {"T": [], "gev": None, "lp3": None, "boot": None, "lower": None, "upper": None,
                           "band": None, "distribution": None, "empirical": None}
    ffa = payload.get("ffa") if isinstance(payload.get("ffa"), dict) else None
    src = ffa if ffa is not None else payload
    fits = src.get("fits") if isinstance(src.get("fits"), dict) else None
    if fits is not None and isinstance(src.get("return_periods"), list):
        out["T"] = [num(t) for t in src["return_periods"]]
        gev, lp3, boot = fits.get("gev_lmoments"), fits.get("lp3"), fits.get("gev_bootstrap")
        if isinstance(gev, dict) and isinstance(gev.get("q"), list):
            out["gev"] = numbers(gev["q"])
        if isinstance(lp3, dict) and isinstance(lp3.get("q"), list):
            out["lp3"] = numbers(lp3["q"])
        if isinstance(boot, dict) and isinstance(boot.get("q"), list):
            out["boot"] = numbers(boot["q"])
        for name, fit in (("GEV MLE/L-moments bootstrap", boot), ("Log-Pearson III", lp3)):
            ci = fit.get("ci") if isinstance(fit, dict) else None
            if isinstance(ci, list) and ci and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in ci):
                out["lower"] = [num(c[0]) for c in ci]
                out["upper"] = [num(c[1]) for c in ci]
                level = num(fit.get("ci_level"))
                out["band"] = name + (f" {level * 100:g} %" if level is not None else " (confidence level unrecorded)")
                break
    elif isinstance(src.get("return_periods"), list) and isinstance(src.get("return_levels"), list):
        # workbench return_periods: one distribution with bootstrap bounds and the empirical points
        out["T"] = numbers(src["return_periods"])
        dist = str(src.get("distribution") or "gev").lower()
        out["distribution"] = dist
        levels = numbers(src["return_levels"])
        if dist.startswith("lp3") or dist.startswith("pearson"):
            out["lp3"] = levels
        else:
            out["gev"] = levels
        if isinstance(src.get("lower_bound"), list) and isinstance(src.get("upper_bound"), list):
            out["lower"], out["upper"] = numbers(src["lower_bound"]), numbers(src["upper_bound"])
            cl = num(src.get("confidence_level"))
            out["band"] = f"{dist.upper()} bootstrap {cl * 100:.0f} %" if cl else f"{dist.upper()} bootstrap"
        emp = src.get("empirical")
        if isinstance(emp, dict) and isinstance(emp.get("return_period"), list):
            out["empirical"] = (numbers(emp["return_period"]), numbers(emp.get("value") or []))
    elif isinstance(src.get("return_periods"), dict):
        # workbench flood_frequency: {"T": level} with {"T": [lo, hi]}
        rp = src["return_periods"]
        keys = sorted(rp, key=lambda k: num(k) or 0.0)
        out["T"] = [num(k) for k in keys]
        out["gev"] = [num(rp[k]) for k in keys]
        out["distribution"] = str(src.get("distribution") or "gev")
        ci = src.get("confidence_intervals")
        if isinstance(ci, dict) and ci:
            out["lower"] = [num((ci.get(k) or [None, None])[0]) for k in keys]
            out["upper"] = [num((ci.get(k) or [None, None])[1]) for k in keys]
            out["band"] = "GEV bootstrap 90 %"
    if not out["T"] or all(t is None for t in out["T"]):
        return None
    if out["gev"] is None and out["lp3"] is None and out["boot"] is None:
        return None
    if out["empirical"] is None:
        am = annual_maxima_of(payload)
        if am:
            vals = sorted(am[1], reverse=True)
            n = len(vals)
            out["empirical"] = ([(n + 1) / r for r in range(1, n + 1)], vals)
    return out


# ── flow duration ─────────────────────────────────────────────────────────


def fdc_of(payload: dict[str, Any]) -> dict[str, Any] | None:
    """``{"exceedance": [...], "q": [...], "percentiles": {pct: value}}``; the curve lists may be empty."""
    out: dict[str, Any] = {"exceedance": [], "q": [], "percentiles": {}}
    fdc = payload.get("fdc") if isinstance(payload.get("fdc"), dict) else None
    src = fdc if fdc is not None else payload
    ex = src.get("exceedance")
    q = src.get("q", src.get("discharge"))
    if isinstance(ex, list) and isinstance(q, list) and ex and len(ex) == len(q):
        pairs = [(a, b) for a, b in zip(numbers(ex), numbers(q)) if a is not None and b is not None]
        out["exceedance"] = [p[0] for p in pairs]
        out["q"] = [p[1] for p in pairs]
    pct: dict[float, float] = {}
    p = src.get("percentiles")
    if isinstance(p, dict):
        for k, v in p.items():
            kk, vv = num(k), num(v)
            if kk is not None and vv is not None:
                pct[kk] = vv
    for k, v in src.items():
        if isinstance(k, str) and len(k) in (3, 4) and k[0] == "q" and k[1:].isdigit():
            kk, vv = float(int(k[1:])), num(v)
            if vv is not None:
                pct[kk] = vv
    sig = payload.get("signatures_m3s")
    if not pct and isinstance(sig, dict):
        for k, v in sig.items():
            if isinstance(k, str) and k[0] == "q" and k[1:].isdigit() and num(v) is not None:
                pct[float(int(k[1:]))] = num(v)  # type: ignore[assignment]
    out["percentiles"] = dict(sorted(pct.items()))
    if not out["q"] and not out["percentiles"]:
        return None
    return out


# ── drought indices ───────────────────────────────────────────────────────


def indices_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One panel per timescale: ``{"timescale", "dates", "spi", "spei" | None, "label"}``; the SGI as one panel."""
    panels: list[dict[str, Any]] = []
    rows = payload.get("indices")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("series"), dict):
                continue
            s = row["series"]
            dates = [date_key(x) for x in (s.get("index") or [])]
            if not dates:
                continue
            spi = numbers(s["spi"]) if isinstance(s.get("spi"), list) else None
            spei = numbers(s["spei"]) if isinstance(s.get("spei"), list) else None
            if spi is None and spei is None:
                continue
            panels.append({"timescale": row.get("timescale"), "dates": dates, "spi": spi, "spei": spei})
        return panels
    sgi = payload.get("sgi")
    if isinstance(sgi, dict) and isinstance(sgi.get("index"), list) and isinstance(sgi.get("values"), list):
        panels.append({"timescale": None, "dates": [date_key(x) for x in sgi["index"]], "spi": None, "spei": None,
                       "sgi": numbers(sgi["values"])})
    return panels


# ── stations and donors ───────────────────────────────────────────────────


def stations_of(payload: dict[str, Any], site: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rows with ``lat``, ``lon`` and a ``label`` from ``stations`` (a search, the reconnaissance, the donors),
    ``similarity.donors`` (a regionalisation) or the ``site`` dict."""
    cands: list[Any] = []
    for src in (payload, site or {}):
        if not isinstance(src, dict):
            continue
        if isinstance(src.get("stations"), list):
            cands = src["stations"]
            break
        sim = src.get("similarity")
        if isinstance(sim, dict) and isinstance(sim.get("donors"), list):
            cands = sim["donors"]
            break
        if isinstance(src.get("donors"), list):
            cands = src["donors"]
            break
    out: list[dict[str, Any]] = []
    for st in cands:
        if not isinstance(st, dict):
            continue
        lat = num(st.get("latitude", st.get("lat")))
        lon = num(st.get("longitude", st.get("lon")))
        if lat is None or lon is None:
            continue
        label = st.get("station_id") or st.get("id") or st.get("name") or ""
        out.append({**st, "lat": lat, "lon": lon, "label": str(label)})
    return out


# ── flat views for the key/value tables ───────────────────────────────────


def flatten(d: Any, *, prefix: str = "", depth: int = 2, skip: frozenset[str] = BULK_KEYS,
            max_list: int = 12) -> list[tuple[str, Any]]:
    """``(dotted key, scalar)`` pairs over a dict, nested dicts to ``depth``; short lists of scalars joined."""
    out: list[tuple[str, Any]] = []
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if str(k) in skip and not prefix:
            continue
        if is_scalar(v):
            out.append((key, v))
        elif isinstance(v, dict) and depth > 0:
            out.extend(flatten(v, prefix=f"{key}.", depth=depth - 1, skip=frozenset(), max_list=max_list))
        elif isinstance(v, (list, tuple)) and len(v) <= max_list and all(is_scalar(x) for x in v):
            out.append((key, "; ".join("" if x is None else str(x) for x in v)))
    return out


def records(rows: Any, columns: list[str] | None = None) -> tuple[list[str], list[list[Any]]] | None:
    """A list of dicts as ``(columns, rows)``; scalars only, lists joined, missing cells None."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return None
    if columns is None:
        seen: dict[str, None] = {}
        for r in rows:
            for k, v in r.items():
                if is_scalar(v) or (isinstance(v, (list, tuple)) and all(is_scalar(x) for x in v)):
                    seen.setdefault(str(k), None)
        columns = list(seen)
    if not columns:
        return None
    table: list[list[Any]] = []
    for r in rows:
        cells: list[Any] = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, (list, tuple)):
                v = "; ".join("" if x is None else str(x) for x in v)
            elif isinstance(v, dict):
                v = "; ".join(f"{a}={b}" for a, b in v.items() if is_scalar(b))
            cells.append(v)
        table.append(cells)
    return columns, table


def frame_records(block: Any) -> tuple[list[str], list[list[Any]]] | None:
    """A workbench ``jsonable(DataFrame)`` block (``columns`` + ``rows``) or a list of dicts as ``(columns, rows)``."""
    if isinstance(block, dict) and isinstance(block.get("columns"), list) and isinstance(block.get("rows"), list):
        cols = [str(c) for c in block["columns"]]
        rows = [list(r) for r in block["rows"] if isinstance(r, (list, tuple))]
        return (cols, rows) if cols and rows else None
    return records(block)
