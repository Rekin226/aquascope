"""Compare two to five gauges side by side: hydrographs, flow duration and flood frequency on one axis.

One engine, three faces: this is the function the Explorer's Compare view calls in the browser worker, and the
same one a script or the MCP server can call. It reuses :func:`aquascope.explore.fetch_series` for the records,
:func:`aquascope.hydrology.flow_duration.flow_duration_curve` for the duration curves and
:func:`aquascope.hydrology.flood_frequency.fit_gev_lmoments` for the flood curves, so a curve here is the same
curve the station page draws for that gauge.

Discharge is normalised by catchment area (to mm/d) only when every compared gauge has an area; otherwise all of
them are shown raw, and ``basis`` says which and why. Mixing normalised and raw lines on one axis would be
worse than either.

Everything here runs in Pyodide: pandas, numpy and the package only.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_PLACES = 2
MAX_PLACES = 5
#: Points per hydrograph line sent to the page; longer overlaps are shown as N-day means.
MAX_HYDRO_POINTS = 4000
#: Points per flow duration curve.
FDC_POINTS = 200
#: Return periods the flood curves are evaluated at (a smooth line on a log axis).
COMPARE_RETURN_PERIODS: list[float] = [1.25, 1.5, 2, 3, 5, 10, 20, 25, 50, 100]
#: Two records must overlap at least this long for the hydrographs to be cut to the shared window.
MIN_OVERLAP_DAYS = 365
#: m3/s over a catchment in km2 to mm/d.
MM_PER_DAY_PER_CMS_KM2 = 86.4


def _clean(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 5)


def _is_cms(unit: str) -> bool:
    u = str(unit or "").lower().replace(" ", "")
    return u in {"m3/s", "m³/s", "m^3/s", "cms", "cumecs"}


def _key(item: dict[str, Any], i: int) -> str:
    if item.get("key"):
        return str(item["key"])
    if item.get("source") and item.get("station_id"):
        return f"{item['source']}/{item['station_id']}"
    return f"place-{i + 1}"


def _area(value: Any) -> float | None:
    v = _clean(value)
    return v if v is not None and v > 0 else None


def _daily(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("D").mean().dropna()


def _binned(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Daily frame cut to at most MAX_HYDRO_POINTS rows by N-day means (NaN-aware). Returns (frame, N)."""
    n = len(frame)
    if n <= MAX_HYDRO_POINTS:
        return frame, 1
    size = int(math.ceil(n / MAX_HYDRO_POINTS))
    groups = np.arange(n) // size
    out = frame.groupby(groups).mean()
    out.index = frame.index[::size][: len(out)]
    return out, size


def _fdc(daily: pd.Series) -> dict[str, Any]:
    from aquascope.hydrology.flow_duration import flow_duration_curve

    fdc = flow_duration_curve(daily)
    step = max(1, len(fdc.exceedance) // FDC_POINTS)
    return {
        "exceedance": [_clean(x) for x in fdc.exceedance[::step]],
        "q": [_clean(x) for x in fdc.discharge[::step]],
        "q95": _clean(fdc.percentiles.get(95, float("nan"))),
        "q50": _clean(fdc.percentiles.get(50, float("nan"))),
        "q10": _clean(fdc.percentiles.get(10, float("nan"))),
    }


def _ffa(s: pd.Series) -> dict[str, Any]:
    from aquascope.explore import MIN_YEARS_FOR_FFA, _annual_max
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments

    am = _annual_max(s)
    n = int(len(am))
    if n < MIN_YEARS_FOR_FFA:
        return {"n_years": n, "error": f"needs {MIN_YEARS_FOR_FFA} complete years of daily flow, has {n}"}
    try:
        g = fit_gev_lmoments(am, return_periods=COMPARE_RETURN_PERIODS)
    except Exception as exc:  # noqa: BLE001
        return {"n_years": n, "error": str(exc)}
    # Weibull plotting positions of the observed maxima, so the fitted line sits beside its own points.
    ranked = np.sort(am.values)[::-1]
    t_emp = [(n + 1) / (i + 1) for i in range(n)]
    return {
        "n_years": n,
        "return_periods": list(COMPARE_RETURN_PERIODS),
        "q": [_clean(g.return_periods[rp]) for rp in COMPARE_RETURN_PERIODS],
        "q100": _clean(g.return_periods[100]),
        "empirical": {"T": [_clean(t) for t in t_emp], "q": [_clean(v) for v in ranked]},
    }


def compare_series(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Line up already-fetched records for comparison. Pure; JSON-safe output.

    Each item is ``{"key", "label", "series": pd.Series | None, "variable", "unit", "area_km2"?, "error"?}``.
    Returns ``places`` (one summary per item, in order), ``normalised`` and ``unit`` and ``basis`` (what the
    y axis is and why), ``hydrograph`` (``t`` plus one aligned value list per compared key), ``fdc`` and ``ffa``
    (per key, discharge only), ``notes`` and ``methods``.
    """
    from aquascope.explore import METHODS

    notes: list[str] = []
    places: list[dict[str, Any]] = []
    usable: list[tuple[dict[str, Any], pd.Series]] = []
    for i, item in enumerate(items):
        key = _key(item, i)
        entry: dict[str, Any] = {
            "key": key,
            "label": str(item.get("label") or key),
            "source": item.get("source"),
            "station_id": item.get("station_id"),
            "variable": item.get("variable") or None,
            "unit": item.get("unit") or None,
            "area_km2": _area(item.get("area_km2")),
            "compared": False,
        }
        s = item.get("series")
        daily = _daily(s) if isinstance(s, pd.Series) else pd.Series(dtype=float)
        if item.get("error"):
            entry["error"] = str(item["error"])
        elif daily.empty:
            entry["error"] = "no observations"
        else:
            entry.update({
                "n_days": int(len(daily)),
                "start": daily.index.min().date().isoformat(),
                "end": daily.index.max().date().isoformat(),
                "years": round((daily.index.max() - daily.index.min()).days / 365.25, 1),
            })
            usable.append((entry, daily))
        places.append(entry)

    out: dict[str, Any] = {
        "places": places, "n_places": len(places), "n_compared": 0, "variable": None, "unit": None,
        "normalised": False, "basis": "", "hydrograph": None, "fdc": {}, "ffa": {}, "notes": notes, "methods": [],
    }
    for p in places:
        if p.get("error"):
            notes.append(f"{p['label']}: left out ({p['error']}).")
    if not usable:
        out["basis"] = "Nothing to compare: no record came back."
        return out

    # One variable on one axis: discharge when any gauge has it, else the commonest.
    variables = Counter(e["variable"] for e, _ in usable)
    variable = "discharge" if "discharge" in variables else variables.most_common(1)[0][0]
    same_var = [(e, d) for e, d in usable if e["variable"] == variable]
    for e, _ in usable:
        if e["variable"] != variable:
            e["error"] = f"its record is {e['variable'] or 'another variable'}, not {variable}"
            notes.append(f"{e['label']}: left out, {e['error']}.")

    normalised = (variable == "discharge"
                  and all(_is_cms(e["unit"]) and e["area_km2"] for e, _ in same_var))
    if normalised:
        unit = "mm/d"
        compared = [(e, d * MM_PER_DAY_PER_CMS_KM2 / e["area_km2"]) for e, d in same_var]
        basis = ("Normalised by catchment area: daily mean flow divided by the upstream area, in mm/d, "
                 "so gauges of different size share one scale.")
    else:
        units = Counter(e["unit"] for e, _ in same_var)
        unit = units.most_common(1)[0][0]
        compared = []
        for e, d in same_var:
            if e["unit"] != unit:
                e["error"] = f"its unit is {e['unit']}, not {unit}"
                notes.append(f"{e['label']}: left out, {e['error']}.")
            else:
                compared.append((e, d))
        if variable == "discharge":
            missing = [e["label"] for e, _ in same_var if not e["area_km2"]]
            why = (f"no catchment area for {', '.join(missing)}" if missing
                   else "the units cannot be converted to mm/d")
            basis = f"Raw values in {unit}: not normalised, because {why}. Bigger catchments sit higher."
        else:
            basis = f"Raw values in {unit}."

    for e, _ in compared:
        e["compared"] = True
    out.update({"variable": variable, "unit": unit, "normalised": normalised, "basis": basis,
                "n_compared": len(compared)})
    if len(compared) < MIN_PLACES:
        notes.append("Fewer than two records are comparable, so there is nothing to overlay.")

    # Hydrographs on one daily index: the shared window when the records overlap by a year, else everything.
    starts = [d.index.min() for _, d in compared]
    ends = [d.index.max() for _, d in compared]
    lo, hi = max(starts), min(ends)
    overlap = (hi - lo).days >= MIN_OVERLAP_DAYS if len(compared) > 1 else True
    if not overlap:
        lo, hi = min(starts), max(ends)
        notes.append("The records do not share a full year, so the hydrographs span each record in full.")
    frame = pd.DataFrame({e["key"]: d for e, d in compared}).loc[lo:hi]
    frame = frame.reindex(pd.date_range(lo.normalize(), hi.normalize(), freq="D"))
    shown, bin_days = _binned(frame)
    out["hydrograph"] = {
        "t": [t.strftime("%Y-%m-%d") for t in shown.index],
        "series": {k: [_clean(v) for v in shown[k].values] for k in shown.columns},
        "window": {"start": lo.date().isoformat(), "end": hi.date().isoformat(), "overlap": bool(overlap)},
        "bin_days": bin_days,
    }
    if bin_days > 1:
        notes.append(f"Hydrographs are shown as {bin_days}-day means to keep the page quick.")

    if variable == "discharge":
        for e, d in compared:
            out["fdc"][e["key"]] = _fdc(d)
            e["q95"], e["q50"], e["q10"] = (out["fdc"][e["key"]][k] for k in ("q95", "q50", "q10"))
            f = _ffa(d)
            out["ffa"][e["key"]] = f
            e["q100"] = f.get("q100")
            if f.get("error"):
                notes.append(f"{e['label']}: no flood curve ({f['error']}).")
        out["methods"] = [METHODS["fdc"], METHODS["gev_lmoments"]]
        if out["fdc"]:
            notes.append("Flow duration and flood curves use each gauge's full record, not only the shared window.")
    else:
        notes.append("Flow duration and flood frequency are drawn for discharge only.")
    for e, d in compared:
        e["mean"] = _clean(d.mean())
    return out


def compare_stations(stations: list[dict[str, Any]], *, years: int | None = None) -> dict[str, Any]:
    """Fetch two to five gauges and compare them (see :func:`compare_series`).

    ``stations`` is a list of ``{"source", "station_id", "label"?, "area_km2"?, "period_start"?}``. The
    Explorer passes the catchment area it already holds (its catchment table is read on the page); without one
    the gauge is shown raw. ``years`` caps each record to the last N years.
    """
    from aquascope.explore import BrowserUnreachableError, fetch_series

    if not isinstance(stations, list) or not (MIN_PLACES <= len(stations) <= MAX_PLACES):
        n = len(stations) if isinstance(stations, list) else 0
        return {"error": f"Pick {MIN_PLACES} to {MAX_PLACES} places to compare; got {n}.", "places": []}
    items: list[dict[str, Any]] = []
    for st in stations:
        source, sid = str(st.get("source") or ""), str(st.get("station_id") or "")
        item: dict[str, Any] = {"source": source, "station_id": sid, "label": st.get("label") or f"{source}/{sid}",
                                "area_km2": st.get("area_km2"), "series": None}
        try:
            fetched = fetch_series(source, sid, years=years, period_start=st.get("period_start"))
            item.update({"series": fetched["series"], "variable": fetched["variable"], "unit": fetched["unit"]})
        except BrowserUnreachableError:
            item["error"] = "the agency cannot be reached from a browser"
        except Exception as exc:  # noqa: BLE001
            logger.info("compare: %s/%s failed: %s", source, sid, exc)
            item["error"] = f"could not fetch the record ({type(exc).__name__})"
        items.append(item)
    return compare_series(items)
