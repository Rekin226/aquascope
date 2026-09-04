"""Site-level tools the playbooks call: a point or a station in, a JSON verdict out.

The playbooks (:mod:`aquascope.playbooks`) can only pass what a plan knows: a
site, a station id, an intake value, or a number an earlier step computed.
The functions here take exactly that and do the fetching and the joining a
workbench analysis over a DataFrame cannot: SPI and SPEI at a rain gauge or
for the ERA5 cell (:func:`drought_indices`), the SPI-to-SGI propagation lag at
a well (:func:`drought_propagation`), the low-flow context of a gauge
(:func:`low_flow_context`), run-of-river supply reliability against a demand,
gauged or regionalised (:func:`supply_reliability`), and a crop's seasonal
water demand from reanalysis ET0 (:func:`crop_water_demand`). Each is an
Analyst tool, an MCP tool and a study step, and returns plain JSON with the
methods it used; a failure is ``{"error": ...}`` so the runner's gates and the
model both see it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from aquascope.explore import METHODS as EXPLORE_METHODS
from aquascope.explore import fetch_series
from aquascope.registry import SOURCES, build_collector
from aquascope.workbench import SPEI_METHOD, SPI_METHOD, THORNTHWAITE_METHOD, jsonable, standardized_indices

logger = logging.getLogger(__name__)

__all__ = ["crop_water_demand", "drought_indices", "drought_propagation", "low_flow_context", "supply_reliability"]

#: ERA5 as served by Open-Meteo starts here.
ERA5_START = date(1940, 1, 1)
#: ERA5 lags real time by a few days.
ERA5_LAG_DAYS = 7
#: A month of daily reanalysis counts when this many days are present.
_MIN_DAYS_PER_MONTH = 25
_ATTRIBUTION = ("Open-Meteo.com (CC BY 4.0); ERA5: Copernicus Climate Change Service (C3S), "
                "Hersbach et al. (2020).")
ML_DAY_TO_M3S = 1000.0 / 86400.0


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _iso(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, pd.Timestamp):
        return d.date().isoformat()
    if isinstance(d, (date, datetime)):
        return d.isoformat()
    return str(d)


def _years(index: pd.DatetimeIndex) -> float:
    return round((index.max() - index.min()).days / 365.25, 1) if len(index) > 1 else 0.0


# ── ERA5 for a point ────────────────────────────────────────────────────────


def era5_daily(
    lat: float,
    lon: float,
    *,
    years: float | None = None,
    start: date | None = None,
    end: date | None = None,
    variables: tuple[str, ...] = ("precipitation_sum", "temperature_2m_mean", "et0_fao_evapotranspiration"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily ERA5 variables for the cell containing the point (Open-Meteo archive, keyless).

    ``years`` back from a week ago, or an explicit ``start`` and ``end``; the
    window is clipped to ERA5's start (1940). Returns the daily table and a
    small metadata dict (window, elevation, source).
    """
    end = end or (_today() - timedelta(days=ERA5_LAG_DAYS))
    if start is None:
        start = end - timedelta(days=int(float(years or 10) * 365.25))
    start = max(start, ERA5_START)
    if start >= end:
        raise ValueError(f"empty ERA5 window {start} to {end}")
    weather = build_collector("openmeteo", mode="weather")
    raw = weather.fetch_raw(latitude=float(lat), longitude=float(lon), start_date=start.isoformat(),
                            end_date=end.isoformat(), daily=list(variables))
    daily = raw.get("daily", {}) or {}
    idx = pd.to_datetime(daily.get("time", []))
    frame = pd.DataFrame({v: pd.Series(daily.get(v, []), index=idx, dtype="float64") for v in variables})
    meta = {"source": "ERA5 via Open-Meteo", "start": start.isoformat(), "end": end.isoformat(),
            "elevation_m": jsonable(raw.get("elevation")), "n_days": int(len(frame))}
    return frame, meta


def _monthly_total(s: pd.Series, *, min_fraction: float = 0.8) -> pd.Series:
    """Monthly totals of a gauge's precipitation, months with too few days dropped.

    Sub-daily increments are summed to days first; a record already coarser
    than daily (monthly totals from some agencies) is summed as it comes.
    """
    s = s.dropna().sort_index().astype(float)
    if len(s) < 2:
        return s.iloc[:0]
    spacing = pd.Series(s.index).diff().median()
    if spacing is not None and spacing > pd.Timedelta(days=2):
        return s.resample("MS").sum(min_count=1).dropna()
    daily = s.resample("D").sum(min_count=1)
    counts = daily.resample("MS").count()
    total = daily.resample("MS").sum(min_count=1)
    enough = counts >= min_fraction * pd.Index(total.index).days_in_month
    return total[enough].dropna()


def _temperature_trend(t_monthly: pd.Series) -> dict[str, Any] | None:
    """Sen's slope and Mann-Kendall p-value on annual mean temperature (complete years only)."""
    counts = t_monthly.resample("YS").count()
    annual = t_monthly.resample("YS").mean()[counts >= 12].dropna()
    if len(annual) < 10:
        return None
    try:
        from aquascope.analysis.trends import mann_kendall, sens_slope

        mk = mann_kendall(annual.to_numpy())
        slope = sens_slope(annual.to_numpy())
    except Exception as exc:  # noqa: BLE001
        logger.info("temperature trend skipped: %s", exc)
        return None
    return {
        "mean_c": jsonable(float(t_monthly.mean())),
        "trend_c_per_decade": jsonable(float(slope.slope) * 10.0),
        "p_value": jsonable(float(mk.p_value)),
        "trend": str(mk.trend),
        "n_years": int(len(annual)),
        "on": "annual mean 2 m temperature (ERA5)",
    }


# ── drought ─────────────────────────────────────────────────────────────────


def drought_indices(
    lat: float,
    lon: float,
    *,
    years: int = 40,
    timescales: list[int] | tuple[int, ...] = (1, 3, 12),
    source: str | None = None,
    station_id: str | None = None,
    pet: str = "thornthwaite",
    threshold: float = -1.0,
) -> dict[str, Any]:
    """SPI and SPEI at a rain gauge or for the ERA5 cell, at several timescales, with their divergence.

    With ``source`` and ``station_id`` the gauge's precipitation record (its
    whole span) is the P of both indices and ERA5 supplies the PET; without a
    gauge, ERA5 precipitation for the cell over the last ``years`` is used.
    ``pet`` is ``thornthwaite`` (from ERA5 temperature, the PET SPEI was
    introduced with), ``fao56`` (ERA5 FAO-56 ET0 as Open-Meteo serves it) or
    ``none`` (SPI only). The result carries the block of
    :func:`aquascope.workbench.standardized_indices` plus the record used,
    the ERA5 temperature trend (the warming SPEI responds to) and the methods.
    """
    out: dict[str, Any] = {
        "latitude": round(float(lat), 5), "longitude": round(float(lon), 5),
        "pet_method": pet, "methods": [], "notes": [],
    }
    scales = [int(s) for s in (timescales or (1, 3, 12))]
    station_p: pd.Series | None = None
    if source and station_id:
        if source not in SOURCES:
            return {"error": f"unknown source {source!r}", **out}
        try:
            fetched = fetch_series(source, station_id, variable="precipitation")
        except Exception as exc:  # noqa: BLE001 - the tool answers, the caller decides
            return {"error": f"precipitation fetch failed for {source} {station_id}: {exc}", **out}
        s = fetched["series"]
        if s is None or s.empty:
            out["notes"].append(f"{source} {station_id} returned no precipitation; the ERA5 cell is used instead.")
        else:
            station_p = _monthly_total(s)
            out["station"] = {
                "source": source, "station_id": station_id, "variable": fetched["variable"], "unit": fetched["unit"],
                "start": _iso(s.index.min()), "end": _iso(s.index.max()), "years": _years(s.index),
                "n_months": int(len(station_p)), "fetch_note": fetched["note"],
            }
            if len(station_p) < 24:
                out["notes"].append(f"Only {len(station_p)} complete months at {source} {station_id}; the ERA5 cell "
                                    "is used instead.")
                station_p = None
    end = _today() - timedelta(days=ERA5_LAG_DAYS)
    start = end - timedelta(days=int(years * 365.25))
    if station_p is not None:
        start = min(start, station_p.index[0].date())
    try:
        era5, meta = era5_daily(lat, lon, start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        if station_p is None:
            return {"error": f"ERA5 climate unavailable: {exc}", **out}
        era5, meta = pd.DataFrame(), {"source": "ERA5 via Open-Meteo", "error": str(exc)}
        out["notes"].append(f"ERA5 climate unavailable ({exc}); SPI only, from the gauge.")
    p_era = era5["precipitation_sum"].resample("MS").sum(min_count=_MIN_DAYS_PER_MONTH).dropna() if len(era5) else None
    t_era = era5["temperature_2m_mean"].resample("MS").mean().dropna() if len(era5) else None
    et0_era = (era5["et0_fao_evapotranspiration"].resample("MS").sum(min_count=_MIN_DAYS_PER_MONTH).dropna()
               if len(era5) else None)
    precip = station_p if station_p is not None else p_era
    if precip is None or len(precip) < 24:
        return {"error": "fewer than two years of monthly precipitation for the indices", **out}
    out["precipitation_source"] = (f"{source} {station_id} (gauge)" if station_p is not None
                                   else "ERA5 cell via Open-Meteo")
    pet_series: pd.Series | None = None
    if pet == "none" or t_era is None:
        if pet != "none":
            out["notes"].append("No temperature series: SPI only.")
        out["pet_method"] = "none"
    elif pet == "fao56" and et0_era is not None and et0_era.notna().sum() >= 24:
        pet_series = et0_era
    else:
        if pet == "fao56":
            out["notes"].append("ERA5 FAO-56 ET0 not served for this window; Thornthwaite PET from temperature.")
        from aquascope.climate.indices import thornthwaite_pet

        pet_series = thornthwaite_pet(t_era, float(lat))
        out["pet_method"] = "thornthwaite"
    try:
        block = standardized_indices(precip, pet_series, timescales=scales, threshold=threshold)
    except ValueError as exc:
        return {"error": str(exc), **out}
    out.update(block)
    if t_era is not None:
        trend = _temperature_trend(t_era)
        if trend:
            out["temperature"] = trend
    out["era5"] = meta
    out["methods"] = [SPI_METHOD] + ([SPEI_METHOD] if pet_series is not None else [])
    if out["pet_method"] == "thornthwaite":
        out["methods"].append(THORNTHWAITE_METHOD)
    if len(era5):
        out["methods"].append(EXPLORE_METHODS["era5"])
    if pet_series is not None and out["pet_method"] == "thornthwaite":
        out["notes"].append("PET is Thornthwaite from ERA5 temperature: a temperature-only approximation, the "
                            "formulation SPEI was introduced with; FAO-56 ET0 is the better PET where full weather "
                            "exists.")
    if station_p is None:
        out["notes"].append("The indices describe the ERA5 cell (about 9 km), not a rain gauge.")
    out["attribution"] = _ATTRIBUTION
    return out


def drought_propagation(
    source: str,
    station_id: str,
    lat: float,
    lon: float,
    *,
    years: int | None = None,
    timescales: list[int] | tuple[int, ...] = (1, 3, 6, 12, 24),
    max_lag: int = 24,
    threshold: float = -1.0,
) -> dict[str, Any]:
    """The Standardised Groundwater Index at a well and the SPI accumulation period and lag that best explain it.

    Monthly mean levels give the SGI (Bloomfield and Marchant 2013); ERA5
    precipitation for the cell over the same span, extended back by the longest
    accumulation, gives SPI at each timescale; the cross-correlation over lags
    0 to ``max_lag`` months picks the best pair. The result says how long a
    rainfall deficit takes to reach the water table here.
    """
    from aquascope.climate.indices import standardized_precipitation_index
    from aquascope.groundwater.drought import drought_events, propagation_lag, standardised_groundwater_index

    out: dict[str, Any] = {"source": source, "station_id": station_id, "latitude": round(float(lat), 5),
                           "longitude": round(float(lon), 5), "methods": [], "notes": []}
    if source not in SOURCES:
        return {"error": f"unknown source {source!r}", **out}
    try:
        fetched = fetch_series(source, station_id, years=years, variable="groundwater_level")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"groundwater fetch failed for {source} {station_id}: {exc}", **out}
    s = fetched["series"]
    if s is None or s.empty:
        return {"error": "The source returned no groundwater levels for this station.", **out}
    monthly = s.dropna().resample("MS").mean().dropna()
    out.update({"variable": fetched["variable"], "unit": fetched["unit"], "start": _iso(s.index.min()),
                "end": _iso(s.index.max()), "years": _years(s.index), "n_months": int(len(monthly)),
                "fetch_note": fetched["note"]})
    try:
        sgi = standardised_groundwater_index(monthly)
    except ValueError as exc:
        return {"error": f"SGI: {exc}", **out}
    valid = sgi.dropna()
    if valid.empty:
        return {"error": "the level record is too short for an SGI (five values per calendar month needed)", **out}
    events = drought_events(sgi, threshold=threshold)
    out["sgi"] = {
        "current": jsonable(valid.iloc[-1]), "date": _iso(valid.index[-1]), "worst": jsonable(valid.min()),
        "worst_date": _iso(valid.idxmin()), "n": int(len(valid)), "threshold": threshold,
        "events": len(events), "in_drought": bool(valid.iloc[-1] <= threshold),
        "last_event": ({"start": _iso(events[-1].start), "end": _iso(events[-1].end), "duration": events[-1].duration,
                        "severity": jsonable(events[-1].severity), "peak": jsonable(events[-1].peak)}
                       if events else None),
    }
    scales = sorted({int(x) for x in (timescales or (1, 3, 6, 12, 24))})
    era5_end = min(monthly.index[-1].date(), _today() - timedelta(days=ERA5_LAG_DAYS))
    era5_start = (monthly.index[0] - pd.DateOffset(months=max(scales) + 1)).date()
    try:
        era5, meta = era5_daily(lat, lon, start=era5_start, end=era5_end, variables=("precipitation_sum",))
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"ERA5 precipitation unavailable ({exc}); the SGI stands without the propagation lag.")
        out["propagation"] = None
        out["methods"] = [_SGI_METHOD]
        return out
    p = era5["precipitation_sum"].resample("MS").sum(min_count=_MIN_DAYS_PER_MONTH).dropna()
    by: dict[str, Any] = {}
    best_key: str | None = None
    best_spi: pd.Series | None = None
    for n in scales:
        try:
            spi_n = standardized_precipitation_index(p, scale=n)
            res = propagation_lag(spi_n, sgi, max_lag=int(max_lag))
        except ValueError as exc:
            by[str(n)] = {"error": str(exc)}
            continue
        by[str(n)] = {"lag_months": res.lag_months, "correlation": jsonable(round(res.correlation, 3)), "n": res.n}
        if best_key is None or res.correlation > by[best_key]["correlation"]:
            best_key, best_spi = str(n), spi_n
    best = ({"timescale": int(best_key), **by[best_key]} if best_key else None)
    out["propagation"] = {
        "best": best, "by_timescale": by, "max_lag_months": int(max_lag),
        "precipitation": "ERA5 cell via Open-Meteo", "era5": meta,
    }
    frame = pd.concat({"sgi": sgi, **({"spi": best_spi} if best_spi is not None else {})}, axis=1).dropna(how="all")
    step = max(1, len(frame) // 1200)
    frame = frame.iloc[::step]
    out["series"] = {"index": [_iso(t) for t in frame.index], "step": step,
                     **{c: [jsonable(v) for v in frame[c].to_numpy()] for c in frame.columns}}
    out["methods"] = [_SGI_METHOD, SPI_METHOD, EXPLORE_METHODS["sgi_propagation"], EXPLORE_METHODS["era5"]]
    out["notes"].append("SPI is computed on ERA5 precipitation for the cell, not on a rain gauge.")
    out["attribution"] = _ATTRIBUTION
    return out


_SGI_METHOD = {
    "name": "Standardised Groundwater Index",
    "text": "Monthly mean levels transformed to a standard normal variate calendar month by calendar month; a "
            "drought runs while the index stays at or below the threshold.",
    "citation": "Bloomfield, J. P., & Marchant, B. P. (2013). Analysis of groundwater drought building on the "
                "standardised precipitation index approach. Hydrol. Earth Syst. Sci. 17, 4769-4787.",
}


# ── low flows and supply ────────────────────────────────────────────────────


def _flow_context(q: pd.Series, *, years: float) -> dict[str, Any]:
    """Flow-duration percentiles, baseflow index, 7Q10 where the record allows, and where the recent flow sits."""
    from aquascope.hydrology.baseflow import lyne_hollick
    from aquascope.hydrology.flow_duration import flow_duration_curve, low_flow_stat

    daily = q.dropna().resample("D").mean().dropna()
    fdc = flow_duration_curve(daily, percentiles=[5, 10, 25, 50, 75, 90, 95])
    out: dict[str, Any] = {
        "fdc": {f"q{int(k):02d}": jsonable(v) for k, v in fdc.percentiles.items()},
        "n_days": int(len(daily)),
    }
    try:
        out["bfi"] = jsonable(lyne_hollick(daily).bfi)
    except Exception as exc:  # noqa: BLE001
        out["bfi"] = None
        logger.info("baseflow index skipped: %s", exc)
    out["low_flow"] = None
    if years >= 10:
        try:
            out["low_flow"] = {"7q10": jsonable(low_flow_stat(daily, n_day=7, return_period=10)),
                               "text": "minimum 7-day mean flow with a 10-year return period (Weibull)"}
        except Exception as exc:  # noqa: BLE001
            logger.info("7Q10 skipped: %s", exc)
    recent: dict[str, Any] = {"end": _iso(daily.index[-1])}
    for days in (30, 90):
        window = daily.iloc[-days:]
        if len(window) >= days * 0.8:
            mean = float(window.mean())
            recent[f"last_{days}d_mean"] = jsonable(mean)
            recent[f"last_{days}d_exceedance_pct"] = jsonable(100.0 * float((daily > mean).mean()))
    out["recent"] = recent
    return out


def low_flow_context(source: str, station_id: str, *, years: int | None = None) -> dict[str, Any]:
    """Q95, Q50, Q10, the baseflow index, 7Q10 and where the last month's flow sits in the record, at a gauge.

    The discharge context of a drought question: how low is low here, and is
    the river low now. ``years`` caps the record (the whole record by default).
    """
    out: dict[str, Any] = {"source": source, "station_id": station_id, "methods": [], "notes": []}
    if source not in SOURCES:
        return {"error": f"unknown source {source!r}", **out}
    try:
        fetched = fetch_series(source, station_id, years=years, variable="discharge")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"discharge fetch failed for {source} {station_id}: {exc}", **out}
    s = fetched["series"]
    if s is None or s.empty:
        return {"error": "The source returned no discharge for this station.", **out}
    s = s.dropna()
    out.update({"variable": fetched["variable"], "unit": fetched["unit"], "start": _iso(s.index.min()),
                "end": _iso(s.index.max()), "years": _years(s.index), "fetch_note": fetched["note"],
                "stats": {"mean": jsonable(float(s.mean())), "min": jsonable(float(s.min())),
                          "max": jsonable(float(s.max()))}})
    out.update(_flow_context(s, years=out["years"]))
    out["methods"] = [EXPLORE_METHODS["fdc"], _BFI_METHOD] + ([_LOW_FLOW_METHOD] if out.get("low_flow") else [])
    meta = SOURCES[source]
    out["license"], out["attribution"] = meta.license, meta.attribution
    return out


_BFI_METHOD = {
    "name": "Baseflow index (Lyne-Hollick filter)",
    "text": "Recursive digital filter (alpha 0.925, three passes); the index is the baseflow volume over the total.",
    "citation": "Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling. Inst. Eng. "
                "Aust. Natl. Conf. Publ. 79/10, 89-93.",
}
_LOW_FLOW_METHOD = {
    "name": "Low-flow frequency (7Q10)",
    "text": "Annual minima of the 7-day mean flow ranked by the Weibull plotting position; the 10-year value read "
            "from the curve.",
    "citation": "Smakhtin, V. U. (2001). Low flow hydrology: a review. J. Hydrol. 240, 147-186.",
}


def _demand(demand_m3s: Any, demand_ml_day: Any) -> tuple[float | None, str | None]:
    if demand_m3s is not None:
        return float(demand_m3s), "m3/s"
    if demand_ml_day is not None:
        return float(demand_ml_day) * ML_DAY_TO_M3S, "ML/day"
    return None, None


def _verdict(reliability: float) -> str:
    if reliability >= 0.99:
        return "reliable"
    if reliability >= 0.95:
        return "mostly reliable"
    if reliability >= 0.80:
        return "seasonal shortfalls"
    return "unreliable"


def supply_reliability(
    *,
    demand_m3s: float | None = None,
    demand_ml_day: float | None = None,
    source: str | None = None,
    station_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    share: float = 0.1,
    reserve: str | float = "q95",
    months: list[int] | tuple[int, ...] | None = None,
    years: int | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Can a river supply a demand, as a run-of-river screening: gauged from the record, ungauged from donors.

    The demand is ``demand_m3s`` or ``demand_ml_day`` (megalitres a day,
    converted). On any day the abstraction may take at most ``share`` of the
    flow and must leave ``reserve`` in the river (``q95`` keeps the flow
    exceeded 95 % of the time, a number is m3/s, ``none`` keeps nothing):
    reliability is the fraction of days (and of years without a shortfall,
    and of the volume) on which what may be taken meets the demand, over the
    whole year or over ``months`` (an irrigation season). With ``source`` and
    ``station_id`` it is read off the gauge's daily record; with ``lat`` and
    ``lon`` only, the flow-duration points transferred from ``k`` donor
    catchments (Q95, median, Q05 with their bands) give the exceedance of the
    flow the demand needs, as a band. It is a screening rule in the tradition
    of flow-duration-curve environmental-flow practice (Smakhtin and Eriyagama
    2008; Acreman and Dunbar 2004), not a storage-yield analysis.
    """
    demand, given_as = _demand(demand_m3s, demand_ml_day)
    out: dict[str, Any] = {"methods": [], "notes": []}
    if demand is None or not np.isfinite(demand) or demand <= 0:
        return {"error": "give the demand as demand_m3s or demand_ml_day (positive)", **out}
    share = float(share) if share is not None else 1.0
    if not 0 < share <= 1:
        return {"error": f"share must be between 0 and 1, got {share}", **out}
    month_list = sorted({int(m) for m in months}) if months else None
    if month_list and any(m < 1 or m > 12 for m in month_list):
        return {"error": f"months must be 1..12, got {month_list}", **out}
    out.update({"demand_m3s": round(demand, 4), "demand_given_as": given_as, "share": share,
                "months": month_list, "unit": "m3/s"})
    if source and station_id:
        return _supply_gauged(out, source, station_id, demand, share, reserve, month_list, years)
    if lat is not None and lon is not None:
        return _supply_regional(out, float(lat), float(lon), demand, share, reserve, k)
    return {"error": "give source and station_id for a gauge, or lat and lon for an ungauged point", **out}


def _reserve_value(reserve: str | float, q95: float | None) -> tuple[float, str]:
    if isinstance(reserve, str):
        key = reserve.strip().lower()
        if key in ("none", "0", ""):
            return 0.0, "no reserve"
        if key == "q95":
            return float(q95 or 0.0), "Q95 kept in the river"
        try:
            return float(key), f"{float(key):g} m3/s kept in the river"
        except ValueError:
            raise ValueError(f"reserve must be q95, none or a flow in m3/s, got {reserve!r}") from None
    return float(reserve), f"{float(reserve):g} m3/s kept in the river"


def _supply_gauged(out: dict[str, Any], source: str, station_id: str, demand: float, share: float,
                   reserve: str | float, months: list[int] | None, years: int | None) -> dict[str, Any]:
    if source not in SOURCES:
        return {"error": f"unknown source {source!r}", **out}
    try:
        fetched = fetch_series(source, station_id, years=years, variable="discharge")
    except Exception as exc:  # noqa: BLE001
        return {"error": f"discharge fetch failed for {source} {station_id}: {exc}", **out}
    s = fetched["series"]
    if s is None or s.empty:
        return {"error": "The source returned no discharge for this station.", **out}
    s = s.dropna()
    daily = s.resample("D").mean().dropna()
    out.update({"mode": "gauged", "source": source, "station_id": station_id, "variable": fetched["variable"],
                "unit": fetched["unit"], "start": _iso(s.index.min()), "end": _iso(s.index.max()),
                "years": _years(s.index), "fetch_note": fetched["note"]})
    ctx = _flow_context(daily, years=out["years"])
    out.update(ctx)
    try:
        reserve_m3s, reserve_rule = _reserve_value(reserve, ctx["fdc"].get("q95"))
    except ValueError as exc:
        return {"error": str(exc), **out}
    sel = daily[daily.index.month.isin(months)] if months else daily
    if sel.empty:
        return {"error": "no flow days in the requested months", **out}
    room = (sel - reserve_m3s).clip(lower=0.0)
    available = np.minimum(room, share * sel)
    ok = available >= demand
    ok_reserve_only = room >= demand
    per_year_short = (~ok).groupby(ok.index.year).sum()
    per_year_days = ok.groupby(ok.index.year).count()
    full_years = per_year_days[per_year_days >= 0.8 * per_year_days.median()].index
    short_full = per_year_short.reindex(full_years)
    worst_year = int(short_full.idxmax()) if len(short_full) else None
    required = max(demand / share, reserve_m3s + demand)
    reliability = float(ok.mean())
    out["reserve_m3s"] = jsonable(reserve_m3s)
    out["reserve_rule"] = reserve_rule
    out["required_flow_m3s"] = jsonable(required)
    out["reliability"] = {
        "daily": jsonable(reliability),
        "daily_reserve_only": jsonable(float(ok_reserve_only.mean())),
        "annual": jsonable(float((short_full == 0).mean())) if len(short_full) else None,
        "volumetric": jsonable(float(np.minimum(available, demand).sum() / (demand * len(sel)))),
        "days_short_per_year": jsonable(float(short_full.mean())) if len(short_full) else None,
        "worst_year": ({"year": worst_year, "days_short": int(short_full.loc[worst_year])}
                       if worst_year is not None else None),
        "n_days": int(len(sel)), "n_years": int(len(short_full)),
    }
    out["verdict"] = _verdict(reliability)
    out["text"] = (f"On {reliability:.0%} of days" + (f" in months {months}" if months else "") +
                   f" the river can give {demand:g} m3/s while keeping {reserve_m3s:g} m3/s in the channel and "
                   f"taking no more than {share:.0%} of the flow (the river must carry {required:g} m3/s).")
    out["methods"] = [EXPLORE_METHODS["fdc"], EXPLORE_METHODS["supply_reliability"], _BFI_METHOD]
    if out.get("low_flow"):
        out["methods"].append(_LOW_FLOW_METHOD)
    out["notes"].append("A run-of-river screening rule: no storage, no return flows, no licence conditions; the "
                        "reserve and the share are stated assumptions.")
    meta = SOURCES[source]
    out["license"], out["attribution"] = meta.license, meta.attribution
    return out


def _exceedance_of(required: float, curve: list[tuple[float, float]]) -> float:
    """The fraction of time a flow of ``required`` is exceeded, read off (exceedance, flow) points in log space."""
    pts = sorted((float(e), float(q)) for e, q in curve if q is not None and q > 0)
    if not pts:
        return float("nan")
    if required <= pts[-1][1]:
        return pts[-1][0]
    if required >= pts[0][1]:
        return pts[0][0]
    logs = np.log([q for _, q in pts])
    exc = np.array([e for e, _ in pts])
    return float(np.interp(np.log(required), logs[::-1], exc[::-1]))


def _supply_regional(out: dict[str, Any], lat: float, lon: float, demand: float, share: float,
                     reserve: str | float, k: int) -> dict[str, Any]:
    from aquascope.mcp_server import describe_catchment, regionalize_signatures

    out.update({"mode": "regional", "latitude": round(lat, 5), "longitude": round(lon, 5)})
    desc = describe_catchment(lat, lon)
    if desc.get("error"):
        return {"error": f"catchment area needed to turn mm/d into m3/s: {desc['error']}", **out}
    attrs = desc.get("attributes") or {}
    area = attrs.get("upstream_area_km2") or attrs.get("area_km2")
    if not isinstance(area, (int, float)) or area <= 0:
        return {"error": "BasinATLAS gave no upstream area for this point", **out}
    reg = regionalize_signatures(lat, lon, k=int(k))
    if reg.get("error"):
        return {"error": f"regionalisation failed: {reg['error']}", **out}
    est = reg.get("estimates") or {}
    factor = float(area) / 86.4  # mm/d over km2 to m3/s

    def to_m3s(name: str) -> dict[str, Any] | None:
        e = est.get(name)
        if not isinstance(e, dict) or e.get("value") is None:
            return None
        row = {kk: jsonable(float(e[kk]) * factor) for kk in ("value", "low", "high") if e.get(kk) is not None}
        row["n_donors"] = e.get("n_donors")
        skill = ((reg.get("skill") or {}).get("by_signature") or {}).get(name) or {}
        if isinstance(skill, dict) and skill.get("nse") is not None:
            row["loo_nse"] = jsonable(skill["nse"])
        return row

    sig = {name: to_m3s(name) for name in ("q95_mm", "q_median_mm", "q05_mm", "q_mean_mm")}
    if sig["q95_mm"] is None or sig["q_median_mm"] is None or sig["q05_mm"] is None:
        return {"error": "the donors gave no Q95, median and Q05 to build a flow-duration curve from", **out}
    out["area_km2"] = jsonable(float(area))
    out["signatures_m3s"] = {"q95": sig["q95_mm"], "q50": sig["q_median_mm"], "q05": sig["q05_mm"],
                             "q_mean": sig["q_mean_mm"]}
    out["n_donors"] = reg.get("n_donors_available")
    out["regionalisation_method"] = reg.get("method")
    band: dict[str, Any] = {}
    for label in ("value", "low", "high"):
        q95, q50, q05 = (sig[n].get(label) for n in ("q95_mm", "q_median_mm", "q05_mm"))
        if None in (q95, q50, q05):
            continue
        try:
            reserve_m3s, reserve_rule = _reserve_value(reserve, q95)
        except ValueError as exc:
            return {"error": str(exc), **out}
        required = max(demand / share, reserve_m3s + demand)
        band[label] = {"required_flow_m3s": jsonable(required), "reserve_m3s": jsonable(reserve_m3s),
                       "reliability": jsonable(_exceedance_of(required, [(0.95, q95), (0.5, q50), (0.05, q05)]))}
    central = band.get("value") or {}
    out["reserve_rule"] = reserve_rule
    out["reserve_m3s"] = central.get("reserve_m3s")
    out["required_flow_m3s"] = central.get("required_flow_m3s")
    rel = central.get("reliability")
    out["reliability"] = {
        "daily": rel,
        "low": (band.get("low") or {}).get("reliability"),
        "high": (band.get("high") or {}).get("reliability"),
        "basis": "exceedance of the required flow read off the transferred Q95, median and Q05 (log-linear); "
                 "at most 0.95 and at least 0.05 can be read from three points",
    }
    out["verdict"] = _verdict(float(rel)) if rel is not None else "unknown"
    out["text"] = (f"From {out['n_donors']} donor catchments the flow exceeded 95 % of the time is about "
                   f"{sig['q95_mm']['value']:g} m3/s (band {sig['q95_mm'].get('low')} to {sig['q95_mm'].get('high')});"
                   f" the {demand:g} m3/s demand needs the river to carry {central.get('required_flow_m3s')} m3/s, "
                   f"exceeded about {rel:.0%} of the time." if rel is not None else "no reliability could be read")
    out["methods"] = [EXPLORE_METHODS["supply_reliability"]]
    for m in reg.get("methods") or []:
        if isinstance(m, dict):
            out["methods"].append(m)
    out["notes"].append("Transferred signatures: quote the band and the leave-one-out skill with the number; "
                        "three flow-duration points give the reliability to within the band, not beyond it.")
    out["notes"].append("A run-of-river screening rule: no storage, no return flows, no licence conditions.")
    return out


# ── irrigation ──────────────────────────────────────────────────────────────


_CROP_ALIASES = {"corn": "maize", "rice": "rice_paddy", "paddy": "rice_paddy", "winterwheat": "wheat_winter",
                 "wheat": "wheat_winter", "soy": "soybean", "soybeans": "soybean", "peanut": "groundnut",
                 "peanuts": "groundnut", "vine": "grape", "grapes": "grape", "tomatoes": "tomato", "potatoes": "potato",
                 "olives": "olive", "onions": "onion"}


def _crop_key(crop: Any, table: dict[str, Any]) -> str | None:
    """The FAO-56 table key for a crop name written any which way ("Sugar Cane", "winter wheat", "corn")."""
    import re

    flat = re.sub(r"[^a-z]", "", str(crop or "").lower())
    if not flat:
        return None
    by_flat = {re.sub(r"[^a-z]", "", k): k for k in table}
    if flat in by_flat:
        return by_flat[flat]
    alias = _CROP_ALIASES.get(flat)
    return alias if alias in table else None


def crop_water_demand(
    lat: float,
    lon: float,
    *,
    crop: str,
    area_ha: float,
    planting_month: int,
    efficiency: float = 0.7,
    years: int = 10,
    method: str = "single",
) -> dict[str, Any]:
    """A crop's seasonal water demand at a point: FAO-56 single Kc on ERA5 reference ET0, in mm, m3 and m3/s.

    For every year of the ERA5 window that holds a whole season from the first
    of ``planting_month``, the crop water requirement and irrigation schedule
    (``aquascope.agri``) are run on the day's ET0 and rainfall; the seasonal
    totals are averaged and their range kept. Demand is the gross irrigation
    depth over ``area_ha`` as a volume, as a mean rate over the season and as
    the peak-month rate, the number a supply check should use. Supply is not
    checked here.
    """
    from aquascope.agri.crop_water import DEFAULT_STAGE_LENGTHS, KC_TABLE, get_kc, irrigation_schedule

    key = _crop_key(crop, KC_TABLE)
    out: dict[str, Any] = {"latitude": round(float(lat), 5), "longitude": round(float(lon), 5), "methods": [],
                           "notes": []}
    if key is None:
        return {"error": f"unknown crop {crop!r}; the FAO-56 table has {', '.join(sorted(KC_TABLE))}", **out}
    try:
        month = int(planting_month)
        area = float(area_ha)
        eff = float(efficiency)
    except (TypeError, ValueError):
        return {"error": "planting_month (1..12), area_ha and efficiency must be numbers", **out}
    if not 1 <= month <= 12 or area <= 0 or not 0 < eff <= 1:
        return {"error": "planting_month must be 1..12, area_ha positive, efficiency between 0 and 1", **out}
    lengths = DEFAULT_STAGE_LENGTHS[key]
    season_days = int(sum(lengths.values()))
    try:
        era5, meta = era5_daily(lat, lon, years=int(years) + 1,
                                variables=("et0_fao_evapotranspiration", "precipitation_sum"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ERA5 climate unavailable: {exc}", **out}
    eto = era5["et0_fao_evapotranspiration"].dropna()
    rain = era5["precipitation_sum"].fillna(0.0)
    if eto.empty:
        return {"error": "ERA5 served no reference evapotranspiration for this point", **out}
    seasons: list[dict[str, Any]] = []
    months_in_season: list[int] = []
    for year in range(eto.index[0].year, eto.index[-1].year + 1):
        planting = date(year, month, 1)
        last = planting + timedelta(days=season_days - 1)
        if pd.Timestamp(planting) < eto.index[0] or pd.Timestamp(last) > eto.index[-1]:
            continue
        sched = irrigation_schedule(eto, rain, key, planting, efficiency=eff, method=method)
        dates = pd.to_datetime(sched["date"])
        monthly = sched.groupby(dates.dt.to_period("M"))["gross_irrigation"].sum()
        peak = monthly.idxmax()
        if not months_in_season:
            months_in_season = sorted({int(m) for m in dates.dt.month})
        seasons.append({
            "year": year, "etc_mm": round(float(sched["etc"].sum()), 1),
            "effective_rain_mm": round(float(sched["effective_rain"].sum()), 1),
            "net_irrigation_mm": round(float(sched["net_irrigation"].sum()), 1),
            "gross_irrigation_mm": round(float(sched["gross_irrigation"].sum()), 1),
            "eto_mean_mm_per_day": round(float(sched["eto"].mean()), 2),
            "peak_month": f"{peak.year}-{peak.month:02d}", "peak_month_mm": round(float(monthly.max()), 1),
        })
    if not seasons:
        return {"error": f"the ERA5 window {meta['start']} to {meta['end']} holds no complete {season_days}-day "
                         f"season from the first of month {month}", **out}
    table = pd.DataFrame(seasons)
    area_m2 = area * 1e4
    gross_mm = float(table["gross_irrigation_mm"].mean())
    net_mm = float(table["net_irrigation_mm"].mean())
    peak_mm = float(table["peak_month_mm"].mean())
    gross_m3 = gross_mm / 1000.0 * area_m2
    out.update({
        "crop": key, "kc": get_kc(key), "stage_lengths_days": dict(lengths), "season_days": season_days,
        "planting_month": month, "area_ha": area, "efficiency": eff, "method": method,
        "years_used": [int(y) for y in table["year"]],
        "season": {"months": months_in_season, "start": f"{month:02d}-01",
                   "end": (date(2001, month, 1) + timedelta(days=season_days - 1)).strftime("%m-%d")},
        "eto": {"mean_mm_per_day": round(float(table["eto_mean_mm_per_day"].mean()), 2),
                "source": "FAO-56 ET0 from ERA5 via Open-Meteo", "era5": meta},
        "demand": {
            "etc_mm": round(float(table["etc_mm"].mean()), 1),
            "effective_rain_mm": round(float(table["effective_rain_mm"].mean()), 1),
            "net_irrigation_mm": round(net_mm, 1),
            "gross_irrigation_mm": round(gross_mm, 1),
            "gross_irrigation_mm_range": [round(float(table["gross_irrigation_mm"].min()), 1),
                                          round(float(table["gross_irrigation_mm"].max()), 1)],
            "net_m3": round(net_mm / 1000.0 * area_m2, 1),
            "gross_m3": round(gross_m3, 1),
            "mean_m3s": round(gross_m3 / (season_days * 86400.0), 5),
            "peak_month_mm": round(peak_mm, 1),
            "peak_month_m3s": round(peak_mm / 1000.0 * area_m2 / (30.44 * 86400.0), 5),
            "unit_depth": "mm over the season", "unit_rate": "m3/s",
        },
        "per_season": seasons,
        "supply_checked": False,
    })
    lo, hi = out["demand"]["gross_irrigation_mm_range"]
    out["text"] = (f"{key.replace('_', ' ')} on {area:g} ha planted on {month:02d}-01: gross irrigation "
                   f"{gross_mm:,.0f} mm over {season_days} days (range {lo:,.0f} to {hi:,.0f} mm across "
                   f"{len(seasons)} seasons), {gross_m3:,.0f} m3, a mean {out['demand']['mean_m3s']:.4g} m3/s and "
                   f"{out['demand']['peak_month_m3s']:.4g} m3/s in the peak month.")
    out["methods"] = [EXPLORE_METHODS["crop_water"], EXPLORE_METHODS["era5"]]
    out["notes"] += [
        "Kc and stage lengths are FAO-56 (1998) Table 12 values pending the 2025 revision (#310).",
        "Reanalysis-forced ET0 carries bias against station-based ET0 (Agric. Water Manage. 2024, "
        "doi:10.1016/j.agwat.2024.108732); the demand is a planning estimate, not a measurement.",
        "Supply was not checked here; a supply_reliability step does that against a gauge.",
    ]
    out["attribution"] = _ATTRIBUTION
    return out
