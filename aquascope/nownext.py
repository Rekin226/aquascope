"""Now and next: today's flow against normal, the 15-day forecast, and the forecast corrected to a gauge.

One engine for the Explorer's Now tab, the MCP tools and ``aquascope now``:

* :func:`flow_status` places one day's value in the record: its percentile against the same days of the
  year (7 days either side) in every other year, and the five classes the USGS National Water Dashboard
  and WMO HydroSOS use (much below normal, below, normal, above, much above). It needs at least
  :data:`MIN_STATUS_YEARS` years with values in that window, and says so when a record has fewer.
* :func:`forecast` reads the two global forecasts that answer a browser without a key: GEOGLOWS v2 (the
  51-member ECMWF ensemble statistics for a river reach, 15 days) and GloFAS through Open-Meteo's flood API
  (daily ensemble statistics for a 5 km cell). It adds the reach's return-period thresholds, fitted to the
  annual maxima of its simulated record since 1940 with the package's flood-frequency code.
* :func:`correct_to_gauge` maps the model onto a gauge's own record (flow-duration quantile mapping, one
  curve per calendar month, the MFDC-QM / SABER family) and applies it to the forecast. Its skill comes
  from :func:`hindcast_skill`: the mapping is fitted on the first part of the overlap and scored on the rest,
  raw against corrected (KGE with r, alpha and beta, percent bias, and the hit rate and false alarms above
  the gauge's 2-year flow).

Everything returns plain JSON. The forecasts are model output, labelled modelled wherever they appear.

Licences. GEOGLOWS v2 output is CC BY 4.0 (the GEOGloWS ECMWF Streamflow Service). Open-Meteo serves its
API data under CC BY 4.0 (GloFAS v4 from the Copernicus Emergency Management Service); its free API is
for non-commercial use.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FORECAST_DAYS",
    "MIN_STATUS_YEARS",
    "STATUS_CLASSES",
    "THRESHOLD_YEARS",
    "correct_to_gauge",
    "ensemble_daily",
    "flow_status",
    "forecast",
    "forecast_points",
    "hindcast_skill",
    "now",
    "plume",
    "station_status",
    "threshold_class",
    "top_up",
]

#: Days either side of the date that make up "the same time of year".
STATUS_WINDOW_DAYS = 7
#: Years with values in that window a percentile needs before it is said.
MIN_STATUS_YEARS = 10
#: The forecast horizon GEOGLOWS issues (and the GloFAS days asked for).
FORECAST_DAYS = 15
#: Years of paired model and gauge days the correction needs.
MIN_OVERLAP_YEARS = 3.0
#: The share of the overlap the mapping is fitted on when its skill is scored; the rest is scored.
SKILL_SPLIT = 0.6
#: A calendar month needs this many paired days before it gets its own flow-duration curve.
MIN_MONTH_DAYS = 60

#: The five classes, by the percentile rounded to a whole number (USGS National Water Dashboard, WMO HydroSOS).
STATUS_CLASSES: list[dict[str, Any]] = [
    {"id": "much_below", "label": "much below normal", "low": 0, "high": 9},
    {"id": "below", "label": "below normal", "low": 10, "high": 24},
    {"id": "normal", "label": "normal", "low": 25, "high": 75},
    {"id": "above", "label": "above normal", "low": 76, "high": 90},
    {"id": "much_above", "label": "much above normal", "low": 91, "high": 100},
]

GLOFAS_FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
GEOGLOWS_CREDIT = "GEOGLOWS v2 forecast (GEOGloWS ECMWF Streamflow Service), CC BY 4.0"
GLOFAS_CREDIT = "GloFAS v4 (Copernicus Emergency Management Service) via Open-Meteo, CC BY 4.0"

METHODS: dict[str, dict[str, str]] = {
    "status": {
        "name": "Today against normal (day-of-year percentile)",
        "text": "The day's value ranked against every value within 7 days either side of the same date in the other "
        "years of the record (mid-rank percentile), then classed as the USGS National Water Dashboard and WMO "
        "HydroSOS do: much below normal (under 10), below (10 to 24), normal (25 to 75), above (76 to 90), much "
        "above (over 90). Needs at least 10 years with values in that window.",
        "citation": "WMO (2022). Hydrological Status and Outlook System (HydroSOS) implementation plan; USGS "
        "National Water Dashboard, streamflow percentile classes.",
    },
    "forecast": {
        "name": "Global ensemble streamflow forecasts",
        "text": "GEOGLOWS v2: ECMWF's 51-member ensemble routed down the river network, 15 days; the 3-hourly "
        "ensemble statistics (mean, median, 25th to 75th percentile, minimum to maximum) and the high-resolution "
        "run are averaged to days. GloFAS v4 via Open-Meteo: daily ensemble statistics for the 5 km cell. "
        "Return-period thresholds: Log-Pearson III fitted to the reach's simulated annual maxima since 1940.",
        "citation": "Hales, R. C. et al. (2022). Advancing global hydrologic modeling with the GEOGloWS ECMWF "
        "streamflow service. J. Flood Risk Manag., doi:10.1111/jfr3.12859. Harrigan, S. et al. (2020). GloFAS-ERA5 "
        "operational global river discharge reanalysis 1979-present. Earth Syst. Sci. Data, 12, 2043-2060.",
    },
    "correction": {
        "name": "Flow-duration quantile mapping to the gauge (monthly)",
        "text": "For each calendar month, the model's flow-duration curve and the gauge's, both from the days they "
        "share, map a model value to the gauge flow with the same exceedance probability; beyond the model's "
        "range the ratio at the end of the curve is kept. A month with fewer than 60 shared days falls back to "
        "one curve for the whole year. Skill: the mapping is fitted on the first 60 % of the shared days and "
        "scored on the rest, raw and corrected: KGE (Gupta et al. 2009) with r, alpha and beta, percent bias, "
        "and the hit rate and false alarms on days above the gauge's 2-year flow.",
        "citation": "Hales, R. C. et al. (2023). Bias correcting discharge simulations from the GEOGloWS global "
        "hydrologic model. J. Hydrol., 626, 130279, doi:10.1016/j.jhydrol.2023.130279. Gupta, H. V. et al. "
        "(2009). J. Hydrol., 377, 80-91.",
    },
}

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_SUBJECT = {"discharge": "Flow", "water_level": "Water level", "groundwater_level": "Groundwater level"}


# ── small helpers ────────────────────────────────────────────────────────────


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _fmt_q(x: float | None) -> str:
    if x is None:
        return "?"
    ax = abs(x)
    if ax >= 100:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.3g}"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _day_month(d: date, year: bool = False) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]}" + (f" {d.year}" if year else "")


def _as_date(x: Any) -> date | None:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return date.fromisoformat(str(x)[:10])
    except ValueError as exc:
        raise ValueError(f"not a date: {x!r} (use YYYY-MM-DD)") from exc


def _as_daily(series: Any) -> Any:
    """A daily-mean pandas Series (naive dates, ascending, no gaps filled) from a Series, a ``{"t", "v"}`` or
    ``{"date", "value"}`` dict, or a list of ``(date, value)`` pairs."""
    import pandas as pd

    if series is None:
        return pd.Series(dtype=float)
    if isinstance(series, pd.Series):
        s = series
    elif isinstance(series, dict):
        t = next((series[k] for k in ("t", "date", "datetime") if series.get(k) is not None), [])
        v = next((series[k] for k in ("v", "value", "values") if series.get(k) is not None), [])
        s = pd.Series(list(v), index=pd.to_datetime(list(t)))
    else:
        pairs = list(series)
        s = pd.Series([p[1] for p in pairs], index=pd.to_datetime([p[0] for p in pairs]))
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s = pd.Series(pd.to_numeric(pd.Series(list(s.values)), errors="coerce").to_numpy(dtype=float), index=idx)
    s = s[~s.index.isna()].dropna().sort_index()
    if s.empty:
        return s
    return s.resample("D").mean().dropna()


def _same_day(year: int, month: int, day: int) -> date:
    """That month and day in ``year``; 29 February becomes the 28th in a common year."""
    try:
        return date(year, month, day)
    except ValueError:
        return date(year, month, 28)


def _class_of(percentile: float) -> dict[str, Any]:
    p = int(round(percentile))
    for c in STATUS_CLASSES:
        if c["low"] <= p <= c["high"]:
            return c
    return STATUS_CLASSES[0] if p < 0 else STATUS_CLASSES[-1]


# ── 1. today against normal ─────────────────────────────────────────────────


def flow_status(series: Any, date: Any = None, *, value: float | None = None, variable: str = "discharge",
                unit: str | None = None, subject: str | None = None, window_days: int = STATUS_WINDOW_DAYS,
                min_years: int = MIN_STATUS_YEARS, today: Any = None) -> dict[str, Any]:
    """Where one day's value sits in the record for that time of year.

    ``series`` is a daily record (a pandas Series, a ``{"t": [...], "v": [...]}`` dict or ``(date, value)``
    pairs). Without ``date`` the latest day is used; ``value`` compares a value from elsewhere (a forecast) with
    the record on ``date``. The reference is every value within ``window_days`` of the same month and day in each
    other year (a year counts when it has values on at least half of those days); the percentile is the mid-rank
    one, and the class follows the rounded percentile. Fewer than ``min_years`` reference years gives an
    ``error`` that says how many there were, not a class.

    Returns ``percentile``, ``class`` (``much_below``, ``below``, ``normal``, ``above``, ``much_above``),
    ``label``, ``n_years``, ``normal`` (the 10th, 25th, 50th, 75th and 90th percentiles of the reference),
    ``recent`` (the 30 days up to the date), ``age_days`` and ``sentence``, for example "Flow is above normal for
    8 October (82nd percentile of 46 years)."
    """
    import numpy as np

    if variable == "precipitation":
        return {"variable": variable, "error": "Today against normal is for flows and levels; a daily rainfall "
                "record is mostly dry days, so its percentile says little.", "class": None, "sentence": None}
    s = _as_daily(series)
    word = subject or _SUBJECT.get(variable, "The value")
    base: dict[str, Any] = {"variable": variable, "unit": unit, "window_days": int(window_days),
                            "min_years": int(min_years), "method": METHODS["status"], "class": None,
                            "label": None, "percentile": None, "sentence": None}
    if s.empty:
        return {**base, "error": "There is no record to compare with."}
    today_d = _as_date(today) or _today()
    explicit = date is not None
    target = _as_date(date) if explicit else s.index.max().date()
    if value is None:
        import pandas as pd

        stamp = pd.Timestamp(target)
        if stamp not in s.index:
            return {**base, "date": target.isoformat(), "error": f"The record has no value on {target.isoformat()}."}
        value = float(s.loc[stamp])
    value = float(value)
    years = range(s.index.min().year, s.index.max().year + 1)
    pieces, n_years = [], 0
    half = timedelta(days=int(window_days))
    need = int(window_days) + 1  # at least half of the 2 * window + 1 days
    for y in years:
        if y == target.year:
            continue
        centre = _same_day(y, target.month, target.day)
        chunk = s.loc[str(centre - half):str(centre + half)].to_numpy(dtype=float)
        if len(chunk) >= need:
            pieces.append(chunk)
            n_years += 1
    out = {**base, "date": target.isoformat(), "value": _num(value), "n_years": n_years,
           "first_year": int(s.index.min().year), "last_year": int(s.index.max().year),
           "age_days": (today_d - target).days}
    recent = s.loc[str(target - timedelta(days=30)):str(target)]
    out["recent"] = {"t": [d.strftime("%Y-%m-%d") for d in recent.index], "v": [_num(v) for v in recent.values]}
    if n_years < int(min_years):
        out["error"] = (f"Only {n_years} {'year has' if n_years == 1 else 'years have'} values within "
                        f"{window_days} days of {_day_month(target)}; today against normal needs at least "
                        f"{min_years}.")
        out["sentence"] = out["error"]
        return out
    ref = np.concatenate(pieces)
    below = float((ref < value).sum())
    equal = float((ref == value).sum())
    pct = 100.0 * (below + 0.5 * equal) / len(ref)
    cls = _class_of(pct)
    rank = int(round(pct))
    out.update({
        "percentile": round(pct, 1), "class": cls["id"], "label": cls["label"], "n_values": int(len(ref)),
        "normal": {f"p{q}": _num(np.percentile(ref, q)) for q in (10, 25, 50, 75, 90)},
    })
    of = f"({_ordinal(rank)} percentile of {n_years} years)"
    if out["age_days"] <= 2:
        out["sentence"] = f"{word} is {cls['label']} for {_day_month(target)} {of}."
    else:
        tail = "" if explicit else ", the latest day in the record"
        out["sentence"] = f"{word} was {cls['label']} on {_day_month(target, year=True)} {of}{tail}."
    return out


def top_up(series: Any, source: str, station_id: str, *, variable: str = "discharge", today: Any = None
           ) -> tuple[Any, str]:
    """The record with the agency's newest days added when the copy at hand is more than a day old.

    The Archive mirrors a gauge once a week, so "today" from the mirror alone can be a week stale. When the last
    value is older than yesterday and the source is one :func:`aquascope.explore.fetch_series` reaches directly,
    the agency is asked for the last year and only the days after the copy's end are added. Returns the series and
    a sentence saying what happened (empty when nothing was needed). Never raises: a failed top-up keeps the copy.
    """
    import pandas as pd

    s = _as_daily(series)
    if s.empty:
        return s, ""
    today_d = _as_date(today) or _today()
    last = s.index.max().date()
    age = (today_d - last).days
    if age <= 1:
        return s, ""
    from aquascope.explore import DIRECT_FETCH_SOURCES, fetch_series

    if source not in DIRECT_FETCH_SOURCES:
        return s, f"The latest value is from {last.isoformat()}; this source cannot be asked for newer days."
    try:
        got = fetch_series(source, station_id, years=1, prefer_archive=False, variable=variable)
    except Exception as exc:  # noqa: BLE001 - the copy at hand is still a record
        logger.info("top-up failed for %s/%s: %s", source, station_id, exc)
        return s, f"The latest value is from {last.isoformat()}; the agency could not be asked for newer days."
    fresh = _as_daily(got.get("series"))
    newer = fresh[fresh.index > s.index.max()] if len(fresh) else fresh
    if newer.empty:
        return s, f"The latest value is from {last.isoformat()}; the agency has nothing newer."
    out = pd.concat([s, newer]).sort_index()
    return out, (f"Added {len(newer)} recent {'day' if len(newer) == 1 else 'days'} from the agency, "
                 f"to {newer.index.max().date().isoformat()}.")


def station_status(source: str, station_id: str, *, series: Any = None, variable: str | None = None,
                   unit: str | None = None, date: Any = None, refresh: bool = True, today: Any = None
                   ) -> dict[str, Any]:
    """Today against normal at a gauge: its record (the Archive first, else the agency), topped up with the
    agency's newest days (``refresh``), through :func:`flow_status`. ``series`` passes a record already at hand
    (the Explorer's). Adds ``source``, ``station_id`` and ``top_up`` (what the top-up did)."""
    var, note = variable or "discharge", ""
    if unit is None and series is not None:
        from aquascope.archive.observations import ARCHIVE_UNITS

        unit = ARCHIVE_UNITS.get(var)
    if series is None:
        from aquascope.explore import fetch_series

        got = fetch_series(source, station_id, variable=variable)
        series, var, unit, note = got.get("series"), got.get("variable") or var, got.get("unit"), got.get("note", "")
        if series is None or len(series) == 0:
            return {"source": source, "station_id": station_id, "variable": var, "class": None, "sentence": None,
                    "error": "The source returned no observations for this station."}
    said = ""
    if refresh and date is None:
        series, said = top_up(series, source, station_id, variable=var, today=today)
    res = flow_status(series, date, variable=var, unit=unit, today=today)
    res.update({"source": source, "station_id": station_id, "top_up": said})
    if note:
        res["fetch_note"] = note
    return res


# ── 2. the forecast ─────────────────────────────────────────────────────────

_CLIENT: Any = None


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """The Open-Meteo seam (tests replace it). An hour's cache: the GloFAS forecast is issued once a day."""
    global _CLIENT
    if _CLIENT is None:
        from aquascope.utils.cache import cache_dir
        from aquascope.utils.http_client import CachedHTTPClient

        _CLIENT = CachedHTTPClient(timeout=60.0, retries=2, cache_dir=cache_dir() / "openmeteo-flood",
                                   cache_ttl_seconds=3600)
    return _CLIENT.get_json(url, params=params)


_GEOGLOWS_KEYS = (("flow_avg", "mean"), ("flow_med", "median"), ("flow_25p", "p25"), ("flow_75p", "p75"),
                  ("flow_min", "min"), ("flow_max", "max"), ("high_res", "high_res"))
_GLOFAS_KEYS = (("river_discharge_mean", "mean"), ("river_discharge_median", "median"),
                ("river_discharge_p25", "p25"), ("river_discharge_p75", "p75"), ("river_discharge_min", "min"),
                ("river_discharge_max", "max"))
STAT_KEYS = ("mean", "median", "p25", "p75", "min", "max", "high_res")


def _utc_naive(ts: Any) -> datetime | None:
    """An ISO timestamp as a naive UTC datetime (a naive one is taken to be UTC already), None if unreadable."""
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo is not None else d


def _daily_geoglows(stats: dict[str, Any], days: int) -> dict[str, Any]:
    """GEOGLOWS's hourly-then-3-hourly statistics as daily means, the first ``days`` days.

    Plain Python, no pandas: the Explorer reads the forecast in a light worker that does not load it."""
    times = [_utc_naive(t) for t in list(stats.get("datetime") or [])]
    cols = [(key, name) for key, name in _GEOGLOWS_KEYS
            if isinstance(stats.get(key), list) and len(stats[key]) == len(times)]
    if not cols:
        return {"date": []}
    by_day: dict[date, dict[str, list[float]]] = {}
    for i, t in enumerate(times):
        if t is None:
            continue
        slot = by_day.setdefault(t.date(), {})
        for key, name in cols:
            v = _num(stats[key][i], 12)
            if v is not None:
                slot.setdefault(name, []).append(v)
    kept = sorted(d for d, slot in by_day.items() if slot)[: int(days)]
    out: dict[str, Any] = {"date": [d.strftime("%Y-%m-%d") for d in kept]}
    valid = [t for t in times if t is not None]
    if valid:
        # The run's start: GEOGLOWS's newest run is often the previous day's 00 UTC one.
        out["initialized"] = min(valid).strftime("%Y-%m-%dT%H:%MZ")
    for _key, name in cols:
        means = []
        for d in kept:
            vals = by_day[d].get(name)
            means.append(_num(sum(vals) / len(vals)) if vals else None)
        out[name] = means
    return out


def _glofas(lat: float, lon: float, days: int) -> dict[str, Any]:
    params = {"latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
              "daily": ",".join(k for k, _ in _GLOFAS_KEYS), "forecast_days": int(days)}
    raw = _fetch_json(GLOFAS_FLOOD_API, params)
    daily = (raw or {}).get("daily") or {}
    if not daily.get("time"):
        return {"error": "GloFAS returned no forecast for this cell."}
    out: dict[str, Any] = {"date": [str(t)[:10] for t in daily["time"]]}
    for key, name in _GLOFAS_KEYS:
        vals = daily.get(key)
        if isinstance(vals, list):
            out[name] = [_num(v) for v in vals]
    out.update({"lat": _num((raw or {}).get("latitude"), 5), "lon": _num((raw or {}).get("longitude"), 5),
                "modelled": True, "source": "GloFAS v4 forecast via Open-Meteo", "attribution": GLOFAS_CREDIT,
                "licence": "CC BY 4.0", "url": GLOFAS_FLOOD_API})
    return out


def _thresholds(series: Any, return_periods: list[float] | None = None) -> dict[str, Any]:
    """Return-period flows from a daily record's annual maxima: Log-Pearson III, GEV by L-moments if it fails."""
    from aquascope.explore import MIN_YEARS_FOR_FFA, _annual_max, _return_periods
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments, fit_lp3

    s = _as_daily(series)
    rps = _return_periods(return_periods)
    if s.empty:
        return {"error": "no record to fit"}
    am = _annual_max(s)
    if len(am) < MIN_YEARS_FOR_FFA:
        return {"error": f"Return periods need at least {MIN_YEARS_FOR_FFA} complete years; this record has "
                f"{len(am)}.", "n_years": int(len(am))}
    try:
        fit = fit_lp3(am, return_periods=rps, ci_level=0.90)
        method = "Log-Pearson III (Bulletin 17C style)"
    except Exception as exc:  # noqa: BLE001 - fall back to the other fit the package reports
        logger.info("LP3 failed, using GEV: %s", exc)
        fit = fit_gev_lmoments(am, return_periods=rps)
        method = "GEV by L-moments"
    return {"return_periods": rps, "q": [_num(fit.return_periods[rp]) for rp in rps], "method": method,
            "n_years": int(len(am)), "first_year": int(am.index.min().year), "last_year": int(am.index.max().year)}


#: A reach whose simulated mean flow is within this factor of the gauge's is taken to be the gauge's river (the
#: same factor :func:`aquascope.explore.snap_glofas_cell` uses for a GloFAS cell).
REACH_MATCH_FACTOR = 2.0


def _reach_check(reach_series: Any, obs: Any) -> dict[str, Any]:
    """The reach's simulated mean flow against the gauge's over the days they share.

    The snap takes the nearest reach, and beside a confluence that can be a small stream next to the gauge's
    river. Quantile mapping still stretches it onto the gauge, but its days follow the wrong river, so a ratio
    outside :data:`REACH_MATCH_FACTOR` is said, not hidden."""
    frame = _paired(reach_series, obs)
    if frame.empty or float(frame["obs"].mean()) <= 0:
        return {"matches": None, "note": None}
    reach_mean, gauge_mean = float(frame["sim"].mean()), float(frame["obs"].mean())
    ratio = reach_mean / gauge_mean
    ok = (1.0 / REACH_MATCH_FACTOR) <= ratio <= REACH_MATCH_FACTOR
    out = {"reach_mean": _num(reach_mean), "gauge_mean": _num(gauge_mean), "ratio": _num(ratio, 3),
           "factor": REACH_MATCH_FACTOR, "matches": ok, "note": None}
    if not ok:
        out["note"] = (f"The reach's simulated mean flow ({_fmt_q(reach_mean)} m³/s) is far from the gauge's "
                       f"({_fmt_q(gauge_mean)} m³/s), so the gauge may be on another river than this reach; "
                       "read the corrected forecast with care.")
    return out


def _peak_sentence(series: dict[str, Any], thresholds: dict[str, Any] | None, *, what: str, unit: str = "m³/s"
                   ) -> str:
    vals = series.get("mean") or []
    dates = series.get("date") or []
    pairs = [(v, d) for v, d in zip(vals, dates) if v is not None]
    if not pairs:
        return ""
    peak, when = max(pairs)
    head = f"{what} peaks at {_fmt_q(peak)} {unit} on {_day_month(date.fromisoformat(when))}"
    q = (thresholds or {}).get("q") or []
    rps = (thresholds or {}).get("return_periods") or []
    known = [(t, x) for t, x in zip(rps, q) if x is not None]
    if not known:
        return head + "."
    passed = [(t, x) for t, x in known if peak >= x]
    if passed:
        t, x = passed[-1]
        return f"{head}, above the {t:g}-year flow ({_fmt_q(x)} {unit})."
    t, x = known[0]
    return f"{head}, under the {t:g}-year flow ({_fmt_q(x)} {unit})."


def _reach_history(river_id: int) -> tuple[dict[str, Any], Any]:
    """The reach's simulated record (GEOGLOWS retrospective) and its daily series."""
    from aquascope import rivers

    store: dict[str, Any] = {}
    rec = rivers.reach_record(river_id, store=store)
    return rec, store.get("series")


def forecast(lat: float | None = None, lon: float | None = None, *, river_id: int | str | None = None,
             days: int = FORECAST_DAYS, return_periods: list[float] | None = None, obs: Any = None,
             glofas: bool = True, glofas_at: tuple[float, float] | None = None,
             match_mean_flow: float | None = None, snap: bool = True, by: str = "month",
             max_distance_m: float = 1000.0, store: dict[str, Any] | None = None, history: bool = True,
             prefer: str = "main", known_geoglows: dict[str, Any] | None = None) -> dict[str, Any]:
    """The next ``days`` days of flow at a river reach or a point, from two global models, with thresholds.

    Give a GEOGLOWS ``river_id``, or ``lat``/``lon`` to snap to the nearest reach (``snap=False`` skips the
    snap: a place with no river gets GloFAS only). Returns:

    * ``geoglows``: daily ``date``, ``mean``, ``median``, ``p25``, ``p75``, ``min``, ``max`` and ``high_res``
      (m3/s), with ``initialized`` (the start of the ensemble run, UTC) and ``generated`` (when the API answered);
    * ``glofas``: daily ``date`` and the ensemble ``mean``, ``median``, ``p25``, ``p75``, ``min``, ``max`` for the
      cell at ``glofas_at``, else the cell nearest the reach (or the point). With ``match_mean_flow`` (a gauge's
      mean flow) the cell is picked from the 5 x 5 around it by flow magnitude
      (:func:`aquascope.explore.snap_glofas_cell`), and the answer says so;
    * ``thresholds``: the reach's 2- to 100-year flows from its simulated annual maxima since 1940, and
      ``status``: where the reach's flow on the first forecast day sits against its simulated record;
    * with ``obs`` (a gauge's daily discharge): ``correction`` from :func:`correct_to_gauge` (the corrected
      forecast and its hindcast skill), ``gauge_thresholds`` from the gauge's own annual maxima, and
      ``reach_check``: the reach's simulated mean flow against the gauge's over the days they share. Outside a
      factor of 2 the gauge may sit on another river than the reach it snapped to, and ``reach_check["note"]``
      says so;
    * ``sentence``: the peak against the thresholds, and ``notes``, ``methods``, ``attribution``.

    ``store`` (a dict) receives the reach's daily simulated series under ``"reach_series"``.

    Speed. ``history=False`` skips the reach's simulated record (86 years of daily flow, the slow read): no
    ``thresholds``, ``status`` or correction, only the forecasts, so a caller can show the next days at once
    and ask for the rest after; ``"history"`` in the answer says which it was. ``known_geoglows`` (the
    ``geoglows`` part of an earlier answer for the same reach) is used rather than read again. ``prefer`` goes to
    the snap (:func:`aquascope.rivers.snap_to_river`): ``"main"`` for a place, ``"nearest"`` for a gauge.
    """
    from aquascope import rivers

    days = max(1, min(int(days), 30))
    out: dict[str, Any] = {"days": days, "modelled": True, "river_id": None, "lat": _num(lat, 6),
                           "lon": _num(lon, 6), "history": bool(history), "notes": [],
                           "methods": [METHODS["forecast"]], "attribution": [GEOGLOWS_CREDIT, GLOFAS_CREDIT]}
    rid = None
    if river_id not in (None, ""):
        rid = rivers._river_id(river_id)
    elif lat is not None and lon is not None and snap:
        sn = rivers.snap_to_river(lat, lon, max_distance_m=max_distance_m, prefer=prefer)
        out["snap"] = sn
        if sn.get("snapped"):
            rid = int(sn["river_id"])
            lat, lon = sn["snap_lat"], sn["snap_lon"]
        else:
            out["notes"].append(f"{sn.get('message')} GEOGLOWS forecasts rivers, so only GloFAS is shown here.")
    elif lat is None or lon is None:
        raise ValueError("give a river_id, or lat and lon")
    elif glofas:
        out["notes"].append("No river reach was given here, so only GloFAS is shown.")
    out["river_id"] = rid

    reach_series = None
    if rid is not None:
        if isinstance(known_geoglows, dict) and known_geoglows.get("date"):
            out["geoglows"] = dict(known_geoglows)
        else:
            try:
                stats = rivers.forecast_stats(rid)
            except Exception as exc:  # noqa: BLE001 - the other model can still answer
                stats = {"error": f"GEOGLOWS did not answer ({exc})."}
            if stats.get("error"):
                out["geoglows"] = {"error": stats["error"]}
            else:
                daily = _daily_geoglows(stats, days)
                daily.update({"modelled": True, "source": stats.get("source"), "attribution": GEOGLOWS_CREDIT,
                              "licence": "CC BY 4.0", "url": stats.get("url"), "generated": stats.get("generated"),
                              "unit": "m3/s"})
                out["geoglows"] = daily
    if rid is not None and history:
        try:
            rec, reach_series = _reach_history(rid)
        except Exception as exc:  # noqa: BLE001 - a forecast without thresholds is still a forecast
            rec, reach_series = {"error": f"GEOGLOWS did not answer for the simulated record ({exc})."}, None
        if reach_series is None or len(reach_series) == 0:
            out["thresholds"] = {"error": rec.get("error") or "No simulated record for this reach."}
        else:
            out["thresholds"] = {**_thresholds(reach_series, return_periods),
                                 "source": "GEOGLOWS v2 retrospective simulation (modelled)"}
            g = out.get("geoglows") or {}
            if g.get("date") and g.get("mean") and g["mean"][0] is not None:
                st = flow_status(reach_series, g["date"][0], value=g["mean"][0], subject="Simulated flow")
                st.pop("recent", None)
                out["status"] = st
            if store is not None:
                store["reach_series"] = reach_series

    if glofas:
        try:
            gl_lat, gl_lon, cell_note = lat, lon, ""
            if glofas_at is not None:
                gl_lat, gl_lon = glofas_at
            else:
                ref = match_mean_flow
                if ref is None and reach_series is not None and len(reach_series):
                    ref = float(reach_series.mean())
                if ref is not None and ref > 0 and lat is not None:
                    from aquascope.explore import snap_glofas_cell

                    cell = snap_glofas_cell(lat, lon, float(ref))
                    if cell.get("lat") is not None and cell.get("flow_magnitude_matches"):
                        gl_lat, gl_lon = cell["lat"], cell["lon"]
                        cell_note = (f"GloFAS cell picked by mean flow, {cell.get('offset_km', 0):.1f} km away "
                                     "(not checked to drain the same catchment).")
            if gl_lat is None or gl_lon is None:
                raise ValueError("GloFAS is read at a position; give lat and lon with the river_id")
            gl = _glofas(float(gl_lat), float(gl_lon), days)
            if cell_note and not gl.get("error"):
                gl["cell_note"] = cell_note
            out["glofas"] = gl
        except ValueError as exc:
            out["glofas"] = {"error": f"{exc}."}
        except Exception as exc:  # noqa: BLE001 - GEOGLOWS may still have answered
            logger.info("GloFAS forecast failed: %s", exc)
            out["glofas"] = {"error": f"GloFAS did not answer ({exc})."}

    if obs is not None and reach_series is not None and (out.get("geoglows") or {}).get("date"):
        out["reach_check"] = _reach_check(reach_series, obs)
        corr = correct_to_gauge(reach_series, obs, out["geoglows"], by=by)
        out["correction"] = corr
        out["gauge_thresholds"] = {**_thresholds(obs, return_periods), "source": "the gauge's observed record"}
        if not corr.get("error"):
            out["methods"].append(METHODS["correction"])

    corrected = (out.get("correction") or {}).get("forecast")
    if corrected:
        out["sentence"] = _peak_sentence(corrected, out.get("gauge_thresholds"),
                                         what="Corrected to the gauge, the GEOGLOWS ensemble mean")
    elif (out.get("geoglows") or {}).get("mean"):
        out["sentence"] = _peak_sentence(out["geoglows"], out.get("thresholds"), what="The GEOGLOWS ensemble mean")
    elif (out.get("glofas") or {}).get("mean"):
        out["sentence"] = _peak_sentence(out["glofas"], None, what="The GloFAS ensemble mean")
    else:
        out["sentence"] = "No forecast answered for this place."
    out["notes"].append("Both forecasts are model output, not measurements. GEOGLOWS's thresholds come from its own "
                        "simulated record, so they compare like with like; a gauge's own record outranks both.")
    return out


# ── 3. corrected to the gauge ───────────────────────────────────────────────


def _paired(model: Any, obs: Any) -> Any:
    import pandas as pd

    m, o = _as_daily(model), _as_daily(obs)
    frame = pd.DataFrame({"sim": m, "obs": o}).dropna()
    return frame[(frame["sim"] >= 0) & (frame["obs"] >= 0)]


def _fit_mapping(frame: Any, by: str) -> dict[str, Any]:
    """Sorted model and gauge values per calendar month (or one pair for the year)."""
    import numpy as np

    use = by
    if by == "month":
        counts = frame.groupby(frame.index.month).size()
        if len(counts) < 12 or int(counts.min()) < MIN_MONTH_DAYS:
            use = "year"
    curves: dict[int, tuple[Any, Any]] = {}
    if use == "month":
        for m, part in frame.groupby(frame.index.month):
            curves[int(m)] = (np.sort(part["sim"].to_numpy(dtype=float)), np.sort(part["obs"].to_numpy(dtype=float)))
    else:
        curves[0] = (np.sort(frame["sim"].to_numpy(dtype=float)), np.sort(frame["obs"].to_numpy(dtype=float)))
    return {"by": use, "curves": curves}


def _map_values(mapping: dict[str, Any], values: Any, months: Any) -> Any:
    """Map model values to gauge flows with the same exceedance probability (per month, or for the year)."""
    import numpy as np

    vals = np.asarray(values, dtype=float)
    keys = np.asarray(months, dtype=int) if mapping["by"] == "month" else np.zeros(vals.shape, dtype=int)
    out = np.full(vals.shape, np.nan)
    for key, (sim, obs) in mapping["curves"].items():
        sel = (keys == key) & np.isfinite(vals)
        if not sel.any() or not len(sim):
            continue
        x = vals[sel]
        n, m = len(sim), len(obs)
        # The exceedance probability of x on the model's curve (mid-rank, so ties sit in the middle), then the
        # gauge's flow at that probability.
        p = (np.searchsorted(sim, x, "left") + np.searchsorted(sim, x, "right")) / (2.0 * n)
        y = np.interp(p, (np.arange(m) + 0.5) / m, obs)
        # Beyond the model's range, keep the ratio at the end of the curve.
        top = sim[-1] > 0 and obs[-1] / sim[-1]
        low = sim[0] > 0 and obs[0] / sim[0]
        y = np.where(x > sim[-1], x * top if top else obs[-1], y)
        y = np.where(x < sim[0], x * low if low else obs[0], y)
        out[sel] = y
    return out


def _scores(obs: Any, sim: Any, threshold: float | None) -> dict[str, Any]:
    import numpy as np

    o, s = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    ok = np.isfinite(o) & np.isfinite(s)
    o, s = o[ok], s[ok]
    res: dict[str, Any] = {"n_days": int(len(o))}
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return {**res, "kge": None, "r": None, "alpha": None, "beta": None, "pbias": None}
    r = float(np.corrcoef(o, s)[0, 1]) if s.std() > 0 else 0.0
    alpha = float(s.std() / o.std())
    beta = float(s.mean() / o.mean())
    kge = 1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    res.update({"kge": _num(kge, 3), "r": _num(r, 3), "alpha": _num(alpha, 3), "beta": _num(beta, 3),
                "pbias": _num(100.0 * float((s - o).sum()) / float(o.sum()), 1)})
    if threshold is not None and math.isfinite(threshold):
        ev, fc = o > threshold, s > threshold
        hits, misses = int((ev & fc).sum()), int((ev & ~fc).sum())
        fa, cn = int((~ev & fc).sum()), int((~ev & ~fc).sum())
        res.update({"days_above": hits + misses, "hits": hits, "misses": misses, "false_alarms": fa,
                    "hit_rate": _num(hits / (hits + misses), 3) if hits + misses else None,
                    "false_alarm_ratio": _num(fa / (hits + fa), 3) if hits + fa else None,
                    "false_alarm_rate": _num(fa / (fa + cn), 4) if fa + cn else None})
    return res


def _skill_detail(raw: dict[str, Any], cor: dict[str, Any]) -> str:
    """Bias and the days above the 2-year flow, raw against corrected, in one short sentence."""
    bits = []
    if cor.get("pbias") is not None and raw.get("pbias") is not None:
        bits.append(f"Bias {cor['pbias']:+.0f} % (raw {raw['pbias']:+.0f} %).")
    n = cor.get("days_above")
    if n:
        bits.append(f"Days above the 2-year flow caught: {cor.get('hits', 0)} of {n} (raw {raw.get('hits', 0)}), "
                    f"with {cor.get('false_alarms', 0)} false alarms (raw {raw.get('false_alarms', 0)}).")
    elif n == 0:
        bits.append("The gauge did not pass its 2-year flow in the scored years.")
    return " ".join(bits)


def _span(index: Any) -> dict[str, Any]:
    return {"start": index.min().strftime("%Y-%m-%d"), "end": index.max().strftime("%Y-%m-%d")}


def hindcast_skill(model_hist: Any, obs: Any, *, by: str = "month", split: float = SKILL_SPLIT,
                   min_overlap_years: float = MIN_OVERLAP_YEARS, threshold: float | None = None) -> dict[str, Any]:
    """How well the correction does where it was not fitted.

    Over the days the model and the gauge share, the mapping is fitted on the first ``split`` of them (in time)
    and both the raw model and the corrected one are scored against the gauge on the rest: KGE with its r, alpha
    and beta, percent bias, and, on days above ``threshold`` (default the gauge's 2-year flow from its whole
    record), the hit rate, the false alarm ratio (false alarms over all days the model said "above") and the false
    alarm rate (false alarms over all days the gauge stayed below). This scores the simulation (the model driven by
    observed weather), not the forecast at each lead time: that skill builds up in the archive of issued forecasts.
    """
    frame = _paired(model_hist, obs)
    if frame.empty:
        return {"error": "The model and the gauge share no days.", "n_overlap_days": 0}
    years = len(frame) / 365.25
    if years < float(min_overlap_years):
        return {"error": f"The model and the gauge share {years:.1f} years of days; the correction needs "
                f"{min_overlap_years:g}.", "n_overlap_days": int(len(frame)), "overlap": _span(frame.index)}
    k = int(len(frame) * float(split))
    fit, score = frame.iloc[:k], frame.iloc[k:]
    mapping = _fit_mapping(fit, by)
    corrected = _map_values(mapping, score["sim"].to_numpy(), score.index.month)
    thr_info: dict[str, Any] = {"T": 2}
    if threshold is None:
        t2 = _thresholds(obs, [2])
        threshold = (t2.get("q") or [None])[0]
        thr_info.update({"method": t2.get("method"), "n_years": t2.get("n_years"), "error": t2.get("error")})
    thr_info["q"] = _num(threshold)
    raw_s = _scores(score["obs"].to_numpy(), score["sim"].to_numpy(), threshold)
    cor_s = _scores(score["obs"].to_numpy(), corrected, threshold)
    fit_span, score_span = _span(fit.index), _span(score.index)
    out = {"by": mapping["by"], "split": float(split), "n_overlap_days": int(len(frame)), "overlap": _span(frame.index),
           "fit_period": fit_span, "score_period": score_span, "threshold": thr_info, "raw": raw_s,
           "corrected": cor_s}
    kr, kc = raw_s.get("kge"), cor_s.get("kge")
    if kr is not None and kc is not None:
        y0, y1 = score_span["start"][:4], score_span["end"][:4]
        out["skill_line"] = (f"Corrected forecast: KGE {kc:.2f} on the {y0}-{y1} hindcast, raw {kr:.2f}.")
        out["skill_detail"] = _skill_detail(raw_s, cor_s)
        if kc < kr:
            out["note"] = "The correction scored worse than the raw model here; read the raw forecast."
    return out


def correct_to_gauge(model_hist: Any, obs: Any, model_fcst: Any = None, *, by: str = "month",
                     split: float = SKILL_SPLIT, min_overlap_years: float = MIN_OVERLAP_YEARS) -> dict[str, Any]:
    """Map a model onto a gauge's record and apply it to a forecast.

    ``model_hist`` is the model's daily simulation at the gauge's reach, ``obs`` the gauge's daily discharge,
    ``model_fcst`` the forecast to correct: a dict with ``date`` and any of ``mean``, ``median``, ``p25``,
    ``p75``, ``min``, ``max`` and ``high_res`` (what :func:`forecast` returns under ``geoglows``), or a Series.
    The mapping is flow-duration quantile mapping on the days the two share, one curve per calendar month
    (``by="month"``; a month with fewer than 60 shared days makes it one curve for the year, and ``by`` in the
    answer says which was used). The forecast is corrected with the mapping fitted on the whole overlap; the skill
    (:func:`hindcast_skill`) comes from fitting on the first part and scoring on the rest.
    """
    import numpy as np
    import pandas as pd

    skill = hindcast_skill(model_hist, obs, by=by, split=split, min_overlap_years=min_overlap_years)
    out: dict[str, Any] = {"method": METHODS["correction"], "skill": skill}
    if skill.get("error"):
        return {**out, "error": skill["error"]}
    frame = _paired(model_hist, obs)
    mapping = _fit_mapping(frame, by)
    out.update({"by": mapping["by"], "overlap": _span(frame.index), "n_overlap_days": int(len(frame)),
                "skill_line": skill.get("skill_line"), "skill_detail": skill.get("skill_detail")})
    if mapping["by"] != by:
        out["note"] = "Some calendar months have too few shared days, so one flow-duration curve covers the year."
    if model_fcst is None:
        return out
    if isinstance(model_fcst, pd.Series):
        s = _as_daily(model_fcst)
        fixed = _map_values(mapping, s.to_numpy(), s.index.month)
        out["forecast"] = {"date": [d.strftime("%Y-%m-%d") for d in s.index], "mean": [_num(v) for v in fixed]}
        return out
    dates = list(model_fcst.get("date") or [])
    months = np.array([int(str(d)[5:7]) for d in dates], dtype=int)
    fixed_fc: dict[str, Any] = {"date": dates}
    for key in STAT_KEYS:
        vals = model_fcst.get(key)
        if isinstance(vals, list) and len(vals) == len(dates):
            arr = np.array([np.nan if v is None else float(v) for v in vals])
            fixed_fc[key] = [_num(v) for v in _map_values(mapping, arr, months)]
    out["forecast"] = fixed_fc
    return out


# ── 4. one call: now and next ───────────────────────────────────────────────


def _station_row(source: str, station_id: str) -> dict[str, Any] | None:
    from aquascope.archive.catalog import load_stations

    for r in load_stations():
        if str(r.get("source")) == source and str(r.get("station_id")) == station_id:
            return r
    return None


def now(lat: float | None = None, lon: float | None = None, *, station: str | None = None,
        river_id: int | str | None = None, days: int = FORECAST_DAYS, date: Any = None, refresh: bool = True,
        with_forecast: bool = True, correct: bool = True, history: bool = True) -> dict[str, Any]:
    """Now and next in one call: a gauge's status today and the forecast for its reach, corrected to it.

    ``station`` is ``"source/station_id"``: its record gives the status (:func:`station_status`) and, for a
    discharge record, the correction (:func:`correct_to_gauge`); its position is snapped to a river reach for the
    forecast. Without a station, ``lat``/``lon`` or ``river_id`` give the forecast for that reach or point.
    ``history=False`` is the quick forecast (:func:`forecast`): no thresholds, reach status or correction.
    """
    out: dict[str, Any] = {"station": None, "status": None, "forecast": None}
    obs, var = None, None
    if station:
        if "/" not in str(station):
            raise ValueError("give the station as source/station_id, for example usgs/01646500")
        source, sid = str(station).split("/", 1)
        from aquascope.explore import fetch_series

        got = fetch_series(source, sid)
        obs, var = got.get("series"), got.get("variable")
        out["station"] = {"source": source, "station_id": sid, "variable": var, "unit": got.get("unit"),
                          "fetch_note": got.get("note")}
        if obs is None or len(obs) == 0:
            out["status"] = {"error": "The source returned no observations for this station.", "class": None}
        else:
            out["status"] = station_status(source, sid, series=obs, variable=var, unit=got.get("unit"), date=date,
                                           refresh=refresh)
            out["status"].pop("recent", None)
        if lat is None or lon is None:
            row = _station_row(source, sid)
            if row:
                lat, lon = row.get("latitude"), row.get("longitude")
                out["station"]["name"] = row.get("name")
    if with_forecast and (river_id not in (None, "") or (lat is not None and lon is not None)):
        use_obs = obs if (correct and var == "discharge" and obs is not None and len(obs)) else None
        mean = float(_as_daily(use_obs).mean()) if use_obs is not None else None
        # A gauge sits on its own river: its position takes the nearest line, not the main channel near it.
        out["forecast"] = forecast(lat, lon, river_id=river_id, days=days, obs=use_obs, match_mean_flow=mean,
                                   prefer="nearest" if station else "main", history=history)
    elif with_forecast and station:
        out["forecast"] = {"error": "The catalog has no position for this station, so it cannot be snapped to a "
                           "river; give --river-id."}
    bits = [(out.get("status") or {}).get("sentence"), (out.get("forecast") or {}).get("sentence")]
    out["sentence"] = " ".join(b for b in bits if b)
    return out


# ── 5. the FEWS view: threshold classes, the ensemble plume, forecast points (#556) ──
#
# Plain Python, no pandas: the Explorer reads these in a light worker that does not load it.

#: The return periods (years) whose flows class a forecast: the Floods ahead layer's
#: (:data:`aquascope.archive.warnings.RETURN_PERIODS`), so a reach reads the same on the map and in the card.
THRESHOLD_YEARS = (2, 5, 10, 25, 50, 100)
#: One character per day in ``daily``: the index into this tuple, as the Floods ahead file writes it.
DAILY_CODES = (0, 2, 5, 10, 25, 50, 100)
#: Days of a gauge's record drawn before the forecast starts, so the plume starts from what the river did.
PLUME_OBS_DAYS = 21
#: GEOGLOWS's high-resolution run rides along as member 52; it is not one of the 51 ensemble members.
HIGH_RES_KEY = "ensemble_52"

METHODS["plume"] = {
    "name": "The ensemble plume and its threshold classes (the FEWS view)",
    "text": "Each of the 51 GEOGLOWS ensemble members (the high-resolution run left out) is averaged over each UTC "
    "day from the run's start. For each day the plume is the members' median, their middle half (25th to 75th "
    "percentile) and their full range (lowest to highest), with the ensemble mean. A day's class is the largest "
    "return period (2, 5, 10, 25, 50 or 100 years) whose flow the ensemble mean reaches, the rule the Floods ahead "
    "layer uses; 'members' counts the members whose own flow that day reaches each return-period flow. At an "
    "Archive gauge the daily forecast job's statistics are used, corrected to the gauge's record, with the gauge's "
    "own return-period flows.",
    "citation": "Werner, M. et al. (2013). The Delft-FEWS flow forecasting system. Environ. Model. Softw., 40, 65-77. "
    "Alfieri, L. et al. (2013). GloFAS: global ensemble streamflow forecasting and flood early warning. HESS, 17, "
    "1161-1175.",
}


def threshold_map(thresholds: Any) -> dict[int, float]:
    """Return-period flows as ``{years: flow}`` from a dict (``{2: q}``, ``{"q2": q}`` or ``{"2": q}``), a
    ``{"return_periods": [...], "q": [...]}`` answer (:func:`forecast`'s ``thresholds``), or a list in
    :data:`THRESHOLD_YEARS` order. Missing, non-finite and non-positive flows are left out."""
    if not thresholds:
        return {}
    pairs: list[tuple[Any, Any]]
    if isinstance(thresholds, dict) and "q" in thresholds and isinstance(thresholds.get("q"), list):
        pairs = list(zip(thresholds.get("return_periods") or THRESHOLD_YEARS, thresholds["q"]))
    elif isinstance(thresholds, dict):
        pairs = [(str(k).lstrip("q"), v) for k, v in thresholds.items()]
    else:
        pairs = list(zip(THRESHOLD_YEARS, thresholds))
    out: dict[int, float] = {}
    for k, v in pairs:
        try:
            t = int(float(k))
        except (TypeError, ValueError):
            continue
        q = _num(v, 6)
        if t in THRESHOLD_YEARS and q is not None and q > 0:
            out[t] = q
    return dict(sorted(out.items()))


def threshold_class(value: Any, thresholds: Any) -> int:
    """The largest return period (years) whose flow ``value`` reaches, or 0: the Floods ahead rule
    (:func:`aquascope.archive.warnings.classify`). Without a 2-year flow nothing is classed."""
    q = threshold_map(thresholds)
    v = _num(value, 12)
    if v is None or 2 not in q:
        return 0
    best = 0
    for t, flow in q.items():
        if v >= flow:
            best = t
    return best


def daily_code(classes: list[int]) -> str:
    """One character per day: each day's class as its index in :data:`DAILY_CODES` (the Floods ahead ``daily``)."""
    index = {c: i for i, c in enumerate(DAILY_CODES)}
    return "".join(str(index.get(int(c or 0), 0)) for c in classes)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """numpy's default (linear) quantile of an already sorted, non-empty list."""
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def ensemble_daily(times: list[Any], members: dict[str, list[Any]], *, days: int = FORECAST_DAYS,
                   thresholds: Any = None) -> dict[str, Any]:
    """The ensemble's daily plume from its members.

    ``times`` are the forecast's time steps (ISO, UTC) and ``members`` each member's flows at them (``None`` for a
    gap), as :func:`aquascope.rivers.forecast_ensemble` returns them. Each member is averaged over each UTC day from
    the first step's day (the first ``days`` days); then, day by day, over the members: ``mean``, ``median``,
    ``p25``, ``p75``, ``min`` and ``max``. :data:`HIGH_RES_KEY` is kept apart as ``high_res``. With
    ``thresholds``, ``members_at`` counts per return period and day the members whose daily flow reaches its flow,
    and ``share`` is, per return period, the fraction of members whose own 15-day peak reaches it (the Floods ahead
    ``share`` for the 2-year flow).
    """
    stamps = [_utc_naive(t) for t in times]
    valid = [t for t in stamps if t is not None]
    if not valid:
        return {"date": [], "n_members": 0}
    first = min(valid).date()
    keep = [first + timedelta(days=i) for i in range(int(days))]
    slot = {d: i for i, d in enumerate(keep)}

    def daily_of(vals: list[Any]) -> list[float | None]:
        sums = [0.0] * len(keep)
        counts = [0] * len(keep)
        for t, v in zip(stamps, vals):
            x = _num(v, 12)
            if t is None or x is None:
                continue
            i = slot.get(t.date())
            if i is not None:
                sums[i] += x
                counts[i] += 1
        return [sums[i] / counts[i] if counts[i] else None for i in range(len(keep))]

    per_member = {k: daily_of(v) for k, v in sorted(members.items()) if k != HIGH_RES_KEY}
    stats: dict[str, list[float | None]] = {k: [] for k in ("mean", "median", "p25", "p75", "min", "max")}
    for i in range(len(keep)):
        vals = sorted(v[i] for v in per_member.values() if v[i] is not None)
        if not vals:
            for k in stats:
                stats[k].append(None)
            continue
        stats["mean"].append(_num(sum(vals) / len(vals)))
        stats["median"].append(_num(_quantile(vals, 0.5)))
        stats["p25"].append(_num(_quantile(vals, 0.25)))
        stats["p75"].append(_num(_quantile(vals, 0.75)))
        stats["min"].append(_num(vals[0]))
        stats["max"].append(_num(vals[-1]))
    # A trailing day no member reaches (the run's last steps fall short of it) is not a forecast day.
    n = len(keep)
    while n and stats["mean"][n - 1] is None:
        n -= 1
    out: dict[str, Any] = {"date": [d.isoformat() for d in keep[:n]], "n_members": len(per_member),
                           **{k: v[:n] for k, v in stats.items()},
                           "initialized": min(valid).strftime("%Y-%m-%dT%H:%MZ")}
    if HIGH_RES_KEY in members:
        out["high_res"] = [_num(v) for v in daily_of(members[HIGH_RES_KEY])[:n]]
    q = threshold_map(thresholds)
    if q and per_member:
        out["members_at"] = {str(t): [sum(1 for v in per_member.values() if v[i] is not None and v[i] >= flow)
                                      for i in range(n)] for t, flow in q.items()}
        peaks = [max((x for x in v[:n] if x is not None), default=None) for v in per_member.values()]
        counted = [p for p in peaks if p is not None]
        out["share"] = {str(t): _num(sum(1 for p in counted if p >= flow) / len(counted), 4) if counted else None
                        for t, flow in q.items()}
    return out


def _obs_window(obs: Any, first: date, before: int, after: int) -> dict[str, list[Any]] | None:
    """A gauge's daily values from ``before`` days ahead of ``first`` to ``after`` days past it, as ``{t, v}``."""
    if obs is None:
        return None
    if isinstance(obs, dict):
        t = next((obs[k] for k in ("t", "date") if obs.get(k) is not None), [])
        v = next((obs[k] for k in ("v", "value") if obs.get(k) is not None), [])
        pairs = list(zip(t, v))
    else:
        pairs = list(obs)
    lo, hi = first - timedelta(days=int(before)), first + timedelta(days=int(after))
    kept: dict[str, float] = {}
    for when, val in pairs:
        try:
            d = date.fromisoformat(str(when)[:10])
        except ValueError:
            continue
        x = _num(val)
        if x is not None and lo <= d < hi:
            kept[d.isoformat()] = x
    if not kept:
        return None
    days_ = sorted(kept)
    return {"t": days_, "v": [kept[d] for d in days_]}


def _classes(part: dict[str, Any], q: dict[int, float]) -> dict[str, Any]:
    """The day-by-day class of the ensemble mean, the peak and its class, and the first day at the 2-year flow."""
    dates, mean = part.get("date") or [], part.get("mean") or []
    daily = [threshold_class(v, q) for v in mean]
    pairs = [(v, d) for v, d in zip(mean, dates) if v is not None]
    out: dict[str, Any] = {"class_daily": daily, "daily": daily_code(daily), "rp": max(daily, default=0),
                           "peak": None, "peak_date": None, "first_date": None}
    if pairs:
        peak, when = max(pairs)
        out.update(peak=_num(peak), peak_date=when)
    first = next((d for d, c in zip(dates, daily) if c), None)
    out["first_date"] = first
    return out


def _members_line(share: dict[str, Any] | None, n: int, days: int = FORECAST_DAYS) -> str:
    """'In the 15 days, 49 of the 51 members reach the 2-year flow and 15 the 5-year flow.'"""
    if not share or not n:
        return ""
    bits = []
    for t in THRESHOLD_YEARS:
        s = share.get(str(t))
        if s is None:
            continue
        k = int(round(float(s) * n))
        if not k and bits:
            break
        bits.append((t, k))
    if not bits:
        return ""
    lead = f"In the {days} days,"
    t0, k0 = bits[0]
    if not k0:
        return f"{lead} none of the {n} members reaches the {t0}-year flow."
    words = [f"all {n} members reach the {t0}-year flow" if k0 == n else f"{k0} of the {n} members reach the "
             f"{t0}-year flow"]
    words += [f"{'all' if k == n else k} the {t}-year flow" for t, k in bits[1:] if k]
    body = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return f"{lead} {body}."


def _issue_thresholds(river_id: int) -> dict[int, float]:
    """The reach's return-period flows from the published Floods ahead issue, when it is in it (pyarrow needed)."""
    try:
        from aquascope.archive import warnings as fw

        res = fw.flood_warnings(limit=10**7)
    except Exception as exc:  # noqa: BLE001 - no issue, no pyarrow or no network: the caller falls back
        logger.info("no Floods ahead thresholds: %s", exc)
        return {}
    for r in res.get("reaches") or []:
        if int(r.get("river_id") or -1) == int(river_id):
            return threshold_map({f"q{t}": r.get(f"q{t}") for t in THRESHOLD_YEARS})
    return {}


def plume(river_id: int | str | None = None, *, lat: float | None = None, lon: float | None = None,
          thresholds: Any = None, obs: Any = None, days: int = FORECAST_DAYS, obs_days: int = PLUME_OBS_DAYS,
          history: bool = False, look_up: bool = True, run: str | None = None,
          ensemble: dict[str, Any] | None = None) -> dict[str, Any]:
    """The FEWS view of a river reach: the 15-day ensemble plume, its threshold classes and the members past them.

    Give a GEOGLOWS ``river_id``, or ``lat``/``lon`` to snap to the main river there. The 51 members of the newest
    run, or of ``run`` (``YYYY-MM-DD``, the Floods ahead issue the map shows; the newest when that one does not
    answer, said in ``notes``), are read (:func:`aquascope.rivers.forecast_ensemble`; ``ensemble`` passes an answer
    already at hand) and reduced day by day (:func:`ensemble_daily`): ``median``, ``p25``/``p75`` (the middle
    half), ``min``/``max`` (the full range), ``mean`` and ``high_res``. Should the members not answer,
    GEOGLOWS's own statistics are used and the member counts are left out (``from`` says which).

    ``thresholds`` are the return-period flows to class against (any form :func:`threshold_map` reads); the
    Explorer passes the Floods ahead layer's. Without them the published Floods ahead issue is asked for this
    reach (``look_up``; it needs pyarrow), and with ``history`` a Log-Pearson III fit to the reach's simulated
    record is the last resort.
    ``thresholds["source"]`` says where they came from.

    Returns the plume, ``class_daily`` (each day's class from the ensemble mean, the Floods ahead rule), ``rp``,
    ``peak``, ``peak_date``, ``first_date`` (the first day at the 2-year flow), ``members_at`` and ``share``,
    ``observed`` (``obs``, a gauge's daily record, from ``obs_days`` before the run's start), ``issued`` (the run's
    start day), ``sentence``, ``members_line``, ``method``, ``attribution`` and ``licence``. Model output.
    """
    from aquascope import rivers

    days = max(1, min(int(days), 30))
    if river_id in (None, ""):
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon")
        sn = rivers.snap_to_river(lat, lon, prefer="main")
        if not sn.get("snapped"):
            return {"river_id": None, "modelled": True, "snap": sn, "error": sn.get("message")}
        river_id = sn["river_id"]
    rid = rivers._river_id(river_id)
    out: dict[str, Any] = {"river_id": rid, "modelled": True, "days": days, "method": METHODS["plume"],
                           "attribution": GEOGLOWS_CREDIT, "licence": "CC BY 4.0", "unit": "m3/s", "notes": []}
    ens = ensemble
    if ens is None:
        try:
            ens = rivers.forecast_ensemble(rid, run) if run else rivers.forecast_ensemble(rid)
        except Exception as exc:  # noqa: BLE001 - the newest run or the statistics below can still answer
            ens = {"error": f"GEOGLOWS did not answer for the members ({exc})."}
        if run and ens.get("error"):
            try:
                ens = rivers.forecast_ensemble(rid)
                out["notes"].append(f"The run of {run} did not answer, so this is the newest run.")
            except Exception as exc:  # noqa: BLE001 - the statistics below can still answer
                ens = {"error": f"GEOGLOWS did not answer for the members ({exc})."}
    q = threshold_map(thresholds)
    source = (thresholds or {}).get("source") if isinstance(thresholds, dict) else None
    if not q and look_up:
        q = _issue_thresholds(rid)
        source = "the Floods ahead issue (GEOGLOWS return periods, Gumbel on simulated annual maxima)" if q else None
    if not q and history:
        try:
            _rec, series = _reach_history(rid)
            fit = _thresholds(series) if series is not None and len(series) else {}
            q = threshold_map(fit)
            source = f"{fit.get('method')} fitted to the reach's simulated annual maxima" if q else None
        except Exception as exc:  # noqa: BLE001 - a plume without thresholds is still a plume
            logger.info("no simulated record for thresholds: %s", exc)
    if not ens.get("error") and ens.get("members"):
        part = ensemble_daily(ens["datetime"], ens["members"], days=days, thresholds=q)
        part["from"] = "members"
    else:
        try:
            stats = rivers.forecast_stats(rid)
        except Exception as exc:  # noqa: BLE001 - said in the answer
            stats = {"error": f"GEOGLOWS did not answer ({exc})."}
        if stats.get("error"):
            return {**out, "error": ens.get("error") or stats["error"]}
        part = _daily_geoglows(stats, days)
        part.update({"from": "statistics", "n_members": 51})
        out["notes"].append("The members did not answer, so the plume is GEOGLOWS's own 3-hourly statistics "
                            "averaged to days, and the member counts are left out.")
        ens = stats
    if not part.get("date"):
        return {**out, "error": "GEOGLOWS returned no forecast days for this reach."}
    out.update(part)
    out.update({"generated": ens.get("generated"), "url": ens.get("url"),
                "issued": str(part.get("initialized") or part["date"][0])[:10]})
    out["thresholds"] = {"return_periods": list(q), "q": list(q.values()), "source": source} if q else None
    if q and isinstance(thresholds, dict) and thresholds.get("licence"):
        out["thresholds"]["licence"] = thresholds["licence"]
    elif q and source and source.startswith("the Floods ahead issue"):
        out["thresholds"]["licence"] = "CC BY-NC-SA 4.0 (GEOGLOWS v2 return periods)"
    out.update(_classes(part, q))
    if obs is not None:
        out["observed"] = _obs_window(obs, date.fromisoformat(part["date"][0]), obs_days, len(part["date"]))
    peak_q = {"return_periods": list(q), "q": list(q.values())} if q else None
    out["sentence"] = _peak_sentence(part, peak_q, what="The ensemble mean")
    out["members_line"] = _members_line(out.get("share"), int(part.get("n_members") or 0), len(part["date"]))
    out["notes"].append("Model output, not an official warning: no forecaster, no local knowledge.")
    return out


def forecast_points(rows: list[dict[str, Any]], *, model: str = "geoglows") -> dict[str, Any]:
    """The Archive's forecast gauges in the FEWS view, from the rows of one issued file (#517).

    ``rows`` are ``forecasts/issued/<date>.parquet`` rows (one per gauge, model and valid day; any extra columns are
    ignored). For each gauge with ``model`` rows: the plume by valid day (``median``, ``p25``, ``p75``, ``min``,
    ``max``, ``mean``), corrected to the gauge where the job corrected every day (``corrected``), else raw; the
    gauge's own return-period flows (``gauge_q2`` to ``gauge_q100``); each day's class from the mean against them,
    the 15-day class ``rp``, the peak and the first day at the 2-year flow, and a ``note`` where the correction
    scored worse than the raw model on the hindcast. ``counts`` tallies the gauges by class
    (0 is below the 2-year flow, or no flows to class against: ``classed`` says which).
    """
    by_gauge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows or []:
        if str(r.get("model") or "") != model or r.get("source") is None or r.get("station_id") is None:
            continue
        by_gauge.setdefault((str(r["source"]), str(r["station_id"])), []).append(r)
    points = []
    for (source, sid), group in sorted(by_gauge.items()):
        group.sort(key=lambda r: str(r.get("valid_date") or ""))
        corrected = all(_num(r.get("mean_c")) is not None for r in group)
        suffix = "_c" if corrected else ""
        part: dict[str, Any] = {"date": [str(r.get("valid_date"))[:10] for r in group]}
        for k in ("mean", "median", "p25", "p75", "min", "max"):
            part[k] = [_num(r.get(f"{k}{suffix}")) for r in group]
        q = threshold_map({f"q{t}": group[0].get(f"gauge_q{t}") for t in THRESHOLD_YEARS})
        if 2 not in q:   # without the 2-year flow nothing is classed, so no lines either
            q = {}
        head = group[0]
        point = {"key": f"{source}/{sid}", "source": source, "station_id": sid,
                 "river_id": int(head["river_id"]) if head.get("river_id") is not None else None,
                 "issue_date": str(head.get("issue_date") or "")[:10] or None,
                 "issued": str(head.get("init_date") or head.get("issue_date") or "")[:10] or None,
                 "corrected": corrected, "classed": bool(q), **part,
                 "thresholds": {"return_periods": list(q), "q": list(q.values()),
                                "source": "the gauge's own annual maxima"} if q else None,
                 "kge_raw": _num(head.get("kge_raw"), 3), "kge_corrected": _num(head.get("kge_corrected"), 3),
                 "reach_mean_ratio": _num(head.get("reach_mean_ratio"), 3)}
        kr, kc = point["kge_raw"], point["kge_corrected"]
        if corrected and kr is not None and kc is not None and kc < kr:
            # The rule correct_to_gauge's skill gives: the card says so rather than show the plume as trusted.
            point["note"] = "The correction scored worse than the raw model on the hindcast, so read it with care."
        point.update(_classes(part, q))
        points.append(point)
    counts = {str(t): sum(1 for p in points if p["rp"] == t) for t in DAILY_CODES}
    issue = next((p["issue_date"] for p in points if p.get("issue_date")), None)
    return {"issue_date": issue, "model": model, "n": len(points), "counts": counts, "points": points,
            "method": METHODS["plume"], "attribution": GEOGLOWS_CREDIT, "licence": "CC BY 4.0"}
