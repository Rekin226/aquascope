"""Advanced study methods: change, models, projections and regions, as study steps.

The single-gauge study answers "what is the 100-year flow here, assuming the
past is a fair sample of the future". The functions in this module answer the
questions that come after it, each as one JSON-returning step a plan can name:

- :func:`change_points`: is the record stationary? Pettitt's test for a step
  change, PELT for the segments and Mann-Kendall with Sen's slope for a trend,
  on the annual maxima or the annual means. The verdict gates every method
  that assumes stationarity (#376).
- :func:`nonstationary_flood`: a GEV whose location moves with time, next to
  the stationary fit, with the likelihood-ratio test and AIC that say whether
  the extra parameter earns its place, and the T-year level at the start of
  the record, today and at a horizon.
- :func:`pot_flood`: peaks over a threshold (declustered) fitted to a
  Generalised Pareto, which uses every large flood rather than one per year.
- :func:`catchment_model`: GR4J calibrated on the gauge's flow and ERA5
  rainfall and FAO-56 reference evaporation (split-sample, with an
  uncertainty band scored on the validation years), and "what if" runs with
  the rainfall and evaporation changed.
- :func:`climate_projection`: CMIP6 HighResMIP change factors at the point
  (seven models, Open-Meteo, bias-corrected onto ERA5-Land): rainfall,
  evaporation, temperature and the wettest day, and, given a calibrated GR4J,
  the change in mean flow, low flow and the flood quantile model by model.
- :func:`regional_flood` and :func:`compare_gauges`: the gauges around the
  site studied together (Hosking and Wallis index flood, field significance of
  the trends) or side by side.

Everything runs in CPython and in the Explorer's Pyodide worker with numpy,
scipy and pandas: the station records come through
:func:`aquascope.explore.fetch_series` (the Archive first), the forcing and
the projections from Open-Meteo (keyless, CORS-enabled). No model sits
between the numbers and the result; each payload carries ``methods`` with the
citations and ``notes`` with what the numbers can and cannot support.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CMIP6_MODELS",
    "catchment_model",
    "change_points",
    "climate_projection",
    "compare_gauges",
    "nonstationary_flood",
    "pot_flood",
    "regional_flood",
]

#: The seven CMIP6 HighResMIP models the Open-Meteo Climate API serves (1950-2050, daily, 10 km downscaled).
CMIP6_MODELS = ("CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR",
                "MPI_ESM1_2_XR", "NICAM16_8S")
#: The last day the Climate API serves.
CMIP6_END = date(2050, 1, 1)
#: Return periods reported by default.
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
#: Reference evaporation change assumed per degree of warming when a scenario gives a temperature change only.
#: A rough mid value of the FAO-56 sensitivity; a scenario can set ``dpet_pct`` itself.
PET_PCT_PER_C = 3.0
#: GR4J's honest ceiling as a lumped model (km2, #273).
LUMPED_MAX_AREA_KM2 = 10_000.0
#: Years of record GR4J is calibrated and validated on by default (bounded so a browser run stays short).
MODEL_YEARS = 20
#: The share of precipitation falling below 0 degrees C above which the catchment model adds a snow store.
SNOW_FRACTION = 0.10
#: The ensemble needs at least this many models before its spread is quoted.
MIN_MODELS = 3

_REFS = {
    "pettitt": {
        "name": "Pettitt change-point test",
        "text": "Non-parametric test for one shift in the median of a series (rank-based Mann-Whitney statistic).",
        "citation": "Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. "
        "J. R. Stat. Soc. C, 28(2), 126-135.",
    },
    "pelt": {
        "name": "PELT segmentation",
        "text": "Optimal partition of the series into segments of different mean and variance, BIC-type penalty.",
        "citation": "Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a "
        "linear computational cost. JASA, 107(500), 1590-1598.",
    },
    "mann_kendall": {
        "name": "Mann-Kendall test with Sen's slope",
        "text": "Rank-based monotonic trend test; the slope is the median of pairwise slopes.",
        "citation": "Mann (1945); Kendall (1975); Sen, P. K. (1968). JASA, 63(324), 1379-1389.",
    },
    "ns_gev": {
        "name": "Nonstationary GEV (time-varying location)",
        "text": "GEV fitted by maximum likelihood with location mu(t) = mu0 + mu1 (t - mean year); a likelihood-"
        "ratio test against the stationary GEV says whether the trend term is supported.",
        "citation": "Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer, ch. 6.",
    },
    "gpd_pot": {
        "name": "Peaks over threshold with a Generalised Pareto",
        "text": "Independent peaks above a threshold (declustered by a minimum separation), exceedances fitted to a "
        "GPD, return levels from the Poisson rate of peaks per year.",
        "citation": "Lang, M., Ouarda, T. B. M. J., & Bobée, B. (1999). Towards operational guidelines for over-"
        "threshold modeling. J. Hydrol., 225, 103-117.",
    },
    "gr4j": {
        "name": "GR4J rainfall-runoff model",
        "text": "Four-parameter daily lumped conceptual model, calibrated by differential evolution on KGE.",
        "citation": "Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for "
        "streamflow simulation. J. Hydrol., 279, 275-289.",
    },
    "degree_day": {
        "name": "Degree-day snow store",
        "text": "Precipitation split into rain and snow around a threshold temperature; the pack melts in proportion "
        "to degrees above it. Two parameters, calibrated with GR4J's four.",
        "citation": "Hock, R. (2003). Temperature index melt modelling in mountain areas. J. Hydrol., 282, 104-115.",
    },
    "oudin": {
        "name": "Oudin temperature-based potential evaporation",
        "text": "PE = Re / (lambda rho) (T + 5) / 100 from extraterrestrial radiation and mean air temperature.",
        "citation": "Oudin, L. et al. (2005). Which potential evapotranspiration input for a lumped rainfall-runoff "
        "model? Part 2. J. Hydrol., 303, 290-306.",
    },
    "kge": {
        "name": "Kling-Gupta efficiency",
        "text": "Skill score combining correlation, variability ratio and bias ratio (1 is perfect).",
        "citation": "Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). J. Hydrol., 377, 80-91.",
    },
    "split_sample": {
        "name": "Split-sample test",
        "text": "Calibrate on one period, report skill on another the model never saw.",
        "citation": "Klemeš, V. (1986). Operational testing of hydrological simulation models. "
        "Hydrol. Sci. J., 31(1), 13-24.",
    },
    "era5": {
        "name": "ERA5 forcing via Open-Meteo",
        "text": "Daily precipitation and FAO-56 reference evapotranspiration for the ERA5 cell of the point.",
        "citation": "Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., 146, "
        "1999-2049.",
    },
    "cmip6_highresmip": {
        "name": "CMIP6 HighResMIP via Open-Meteo",
        "text": "Seven high-resolution global models (about 20-50 km), downscaled to 10 km and linearly bias-"
        "corrected onto ERA5-Land by Open-Meteo; the future runs follow the highest-emission forcing CMIP6 "
        "HighResMIP offers (close to SSP5-8.5).",
        "citation": "Haarsma, R. J. et al. (2016). High Resolution Model Intercomparison Project (HighResMIP "
        "v1.0) for CMIP6. Geosci. Model Dev., 9, 4185-4208.",
    },
    "delta_change": {
        "name": "Change factors (delta method)",
        "text": "The change each model shows between its own baseline and future runs is applied to the observed "
        "statistic, so model bias in the absolute values cancels; the spread across models is reported.",
        "citation": "Wasko, C. et al. (2024). A systematic review of climate change science relevant to "
        "Australian design flood estimation. HESS, 28, 1251-1285.",
    },
    "index_flood": {
        "name": "Index-flood regional frequency analysis",
        "text": "At-site maxima scaled by their mean, a pooled GEV growth curve from record-weighted L-moment "
        "ratios, discordancy and heterogeneity measures.",
        "citation": "Hosking, J. R. M., & Wallis, J. R. (1997). Regional Frequency Analysis. Cambridge UP.",
    },
}


def _m(*keys: str) -> list[dict[str, str]]:
    return [dict(_REFS[k]) for k in keys]


def _clean(x: Any) -> Any:
    """A JSON-safe value: NaN and inf become None, numpy scalars become Python numbers."""
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _rps(return_periods: Any) -> list[float]:
    if not return_periods:
        return list(RETURN_PERIODS)
    out = sorted({float(t) for t in return_periods if isinstance(t, (int, float)) and float(t) > 1})
    return out or list(RETURN_PERIODS)


def _p(p: float | None) -> str:
    """A p-value in words-friendly precision: three decimals near the usual thresholds, two elsewhere."""
    if p is None or not math.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.2 else f"{p:.2f}"


def _t_key(t: float) -> str:
    return str(int(t)) if float(t).is_integer() else f"{t:g}"


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or not math.isfinite(new) or not math.isfinite(old) or old == 0:
        return None
    return round(100.0 * (new - old) / abs(old), 2)


# ── records ─────────────────────────────────────────────────────────────────


#: The last few records fetched, so a plan whose steps all read the same gauge fetches it once.
_RECORDS: dict[tuple[Any, ...], dict[str, Any]] = {}
_RECORDS_MAX = 6


#: The pseudo-source of a river reach's simulated flow: ``station_id`` is a GEOGLOWS v2 ``river_id`` (#565).
GEOGLOWS_SOURCE = "geoglows"


def _geoglows_series(river_id: str, years: int | None) -> dict[str, Any]:
    """A GEOGLOWS v2 reach's simulated daily flow (m3/s, modelled) in the shape :func:`_fetch` returns."""
    from aquascope import rivers

    rid = rivers._river_id(river_id)
    params: dict[str, Any] = {"format": "json"}
    if years:
        start = datetime.now(timezone.utc).date() - timedelta(days=int(float(years) * 365.25))
        params["start_date"] = start.strftime("%Y%m%d")
    data = rivers._fetch_json(f"{rivers.GEOGLOWS_API}/retrospectivedaily/{rid}", params)
    values = data.get(str(rid)) if isinstance(data, dict) else None
    times = data.get("datetime") if isinstance(data, dict) else None
    if not values or not times or len(values) != len(times):
        return {"series": None, "variable": "discharge", "unit": "m3/s",
                "note": f"GEOGLOWS returned no simulated flow for river reach {rid}"}
    idx = pd.to_datetime(list(times), utc=True).tz_convert(None)
    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float), index=idx)
    return {"series": s.where(s >= 0).dropna(), "variable": "discharge", "unit": "m3/s",
            "note": f"GEOGLOWS v2 simulated flow for river reach {rid} (modelled, CC BY 4.0), not a gauge"}


def _fetch(source: str, station_id: str, years: int | None, variable: str | None) -> dict[str, Any]:
    from aquascope.explore import fetch_series

    key = (source, station_id, years, variable)
    if key not in _RECORDS:
        while len(_RECORDS) >= _RECORDS_MAX:
            _RECORDS.pop(next(iter(_RECORDS)))
        if source == GEOGLOWS_SOURCE:
            _RECORDS[key] = _geoglows_series(station_id, years)
        else:
            _RECORDS[key] = fetch_series(source, station_id, years=years, variable=variable)
    return _RECORDS[key]


def _station(source: str, station_id: str, *, years: int | None = None,
             variable: str | None = "discharge") -> tuple[pd.Series, dict[str, Any]]:
    """The station's daily-mean series (the Archive first) and a small description of it."""
    fetched = _fetch(source, station_id, years, variable)
    s = fetched.get("series")
    if s is None or len(s.dropna()) == 0:
        raise ValueError(f"no {variable or 'record'} data for {source}/{station_id}: {fetched.get('note') or ''}")
    s = s.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.DatetimeIndex(s.index)
    daily = s.resample("D").mean()
    meta = {"source": source, "station_id": station_id, "variable": fetched.get("variable") or variable,
            "unit": fetched.get("unit") or "", "note": fetched.get("note") or "",
            "start": daily.dropna().index.min().date().isoformat(),
            "end": daily.dropna().index.max().date().isoformat(),
            "years": round((daily.dropna().index.max() - daily.dropna().index.min()).days / 365.25, 1)}
    return daily, meta


def _annual_max(daily: pd.Series, *, min_days: int = 292) -> pd.Series:
    """Calendar-year maxima of daily means, keeping years with about 80 % of their days (the Explorer's rule)."""
    counts = daily.resample("YS").count()
    am = daily.resample("YS").max()
    am = am[counts >= min_days].dropna()
    am.index = am.index.year
    return am.astype(float)


def _annual_mean(daily: pd.Series, *, min_days: int = 292) -> pd.Series:
    counts = daily.resample("YS").count()
    mean = daily.resample("YS").mean()
    mean = mean[counts >= min_days].dropna()
    mean.index = mean.index.year
    return mean.astype(float)


# ── change points (#376) ────────────────────────────────────────────────────


def _pettitt(x: np.ndarray) -> tuple[int, float, float]:
    """Pettitt's statistic: the index of the last value before the change, K, and the approximate p-value.

    The p-value is returned whether or not it is significant (the library's
    :func:`aquascope.analysis.changepoint.pettitt_test` returns only a
    significant change), so a report can say how far from a change the record is.
    """
    n = len(x)
    sgn = np.sign(x[:, None] - x[None, :])
    # U_t = sum_{i<=t} sum_{j>t} sign(x_i - x_j)
    u = np.array([sgn[: t + 1, t + 1:].sum() for t in range(n - 1)], dtype=float)
    t = int(np.argmax(np.abs(u)))
    k = float(np.abs(u[t]))
    p = float(min(1.0, 2.0 * math.exp(-6.0 * k * k / (n ** 3 + n ** 2))))
    return t, k, p


def change_points(
    source: str,
    station_id: str,
    *,
    variable: str = "discharge",
    series: str = "annual_max",
    years: int | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Is the record stationary? A step change (Pettitt), segments (PELT) and a trend (Mann-Kendall, Sen).

    ``series`` is ``annual_max`` (what a flood frequency fit assumes has no
    trend or shift) or ``annual_mean`` (the water balance). ``stationary`` is
    False when Pettitt or Mann-Kendall is significant at ``alpha``; that is the
    flag a stationary method's gate reads.
    """
    from aquascope.analysis.changepoint import pelt
    from aquascope.analysis.trends import mann_kendall, sens_slope

    if series not in ("annual_max", "annual_mean"):
        return {"error": f"series must be annual_max or annual_mean, not {series!r}"}
    try:
        daily, meta = _station(source, station_id, years=years, variable=variable)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    annual = _annual_max(daily) if series == "annual_max" else _annual_mean(daily)
    n = len(annual)
    out: dict[str, Any] = {**meta, "tested": series, "n": n, "alpha": alpha,
                           "annual": {"year": [int(y) for y in annual.index], "value": [_clean(v) for v in annual]},
                           "notes": [], "methods": _m("pettitt", "pelt", "mann_kendall")}
    if n < 10:
        out["error"] = f"{n} complete years; a change-point test needs at least 10"
        return out
    x = annual.to_numpy(dtype=float)
    t, k, p = _pettitt(x)
    before, after = x[: t + 1], x[t + 1:]
    change_year = int(annual.index[t + 1])
    pet = {"change_year": change_year, "last_year_before": int(annual.index[t]), "k": _clean(k), "p_value": _clean(p),
           "significant": bool(p < alpha), "mean_before": _clean(before.mean()), "mean_after": _clean(after.mean()),
           "change_pct": _pct(float(after.mean()), float(before.mean()))}
    out["pettitt"] = pet
    mk = mann_kendall(x, alpha=alpha)
    ss = sens_slope(x)
    mean = float(np.mean(x))
    out["mann_kendall"] = {"trend": mk.trend, "p_value": _clean(mk.p_value), "tau": _clean(mk.tau),
                           "significant": bool(mk.p_value < alpha), "sen_slope_per_year": _clean(ss.slope),
                           "sen_slope_pct_per_decade": _clean(100.0 * 10.0 * ss.slope / mean) if mean else None}
    try:
        seg = pelt(pd.Series(x, index=pd.to_datetime([f"{y}-01-01" for y in annual.index])),
                   min_segment_length=max(5, n // 6), cost="mean")
        segments = []
        for s0 in seg.segments or []:
            i0, i1 = int(s0.get("start", 0)), int(s0.get("end", n))
            i1 = min(max(i1, i0 + 1), n)
            segments.append({"start_year": int(annual.index[i0]), "end_year": int(annual.index[i1 - 1]),
                             "mean": _clean(float(x[i0:i1].mean())), "n": i1 - i0})
        out["pelt"] = {"n_changepoints": int(seg.n_changepoints), "segments": segments}
    except Exception as exc:  # noqa: BLE001 - segmentation is supporting evidence, not the verdict
        out["pelt"] = {"n_changepoints": None, "segments": [], "note": f"segmentation failed: {exc}"}
    shift, trend = pet["significant"], out["mann_kendall"]["significant"]
    out["stationary"] = not (shift or trend)
    what = "annual maxima" if series == "annual_max" else "annual means"
    if out["stationary"]:
        out["verdict"] = (f"No significant step change (Pettitt p = {_p(p)}) or trend (Mann-Kendall p = "
                          f"{_p(mk.p_value)}) in {n} years of {what}: a stationary analysis is not contradicted "
                          "by the record.")
    else:
        bits = []
        if shift:
            bits.append(f"a step change around {change_year} (Pettitt p = {_p(p)}; mean "
                        f"{pet['change_pct']:+.0f}% after)" if pet["change_pct"] is not None else
                        f"a step change around {change_year} (Pettitt p = {_p(p)})")
        if trend:
            bits.append(f"a {mk.trend} trend (Mann-Kendall p = {_p(mk.p_value)}, Sen's slope "
                        f"{out['mann_kendall']['sen_slope_pct_per_decade']:+.1f}% per decade)")
        out["verdict"] = (f"The {what} show " + " and ".join(bits) + ". A stationary fit to the whole record "
                          "mixes regimes; quote it with that caveat, and look at the nonstationary fit or the "
                          "recent segment.")
    if n < 30:
        out["notes"].append(f"{n} years is a short record for detecting change: a non-significant result is weak "
                            "evidence of stationarity.")
    out["notes"].append("A detected change says the record is not one sample; it does not say why (a dam, a "
                        "rating change, land use and climate all leave the same mark). Check the station's history.")
    return _clean(out)


# ── nonstationary GEV ───────────────────────────────────────────────────────


def nonstationary_flood(
    source: str,
    station_id: str,
    *,
    return_periods: list[float] | None = None,
    horizon_year: int | None = None,
    years: int | None = None,
    n_boot: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    """The T-year flood when the annual maxima have a trend: a GEV with a time-varying location.

    Reports the stationary GEV (maximum likelihood) and the nonstationary one
    side by side, the likelihood-ratio test and both AICs, and the T-year
    level at the first year, the last year and ``horizon_year`` (default: the
    last year plus 25, flagged as an extrapolation of the fitted trend). The
    ``preferred`` model is the nonstationary one only when the trend term is
    significant and lowers the AIC. ``n_boot`` parametric bootstrap refits
    give a 90 % interval on the last-year level (0 skips it).

    Design guidance for a changing climate is not settled (Wasko et al. 2024)
    and nonstationary estimates are fragile to their parameters: the result is
    a sensitivity next to the stationary answer, not a replacement for it.
    """
    from scipy.stats import genextreme

    from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

    rps = _rps(return_periods)
    try:
        daily, meta = _station(source, station_id, years=years)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    am = _annual_max(daily)
    n = len(am)
    out: dict[str, Any] = {**meta, "n_years": n, "return_periods": rps, "notes": [],
                           "annual_maxima": {"year": [int(y) for y in am.index], "value": [_clean(v) for v in am]},
                           "methods": _m("ns_gev")}
    if n < 20:
        out["error"] = f"{n} complete annual maxima; a nonstationary GEV needs at least 20 (30 or more is better)"
        return out
    data = am.to_numpy(dtype=float)
    yrs = np.asarray(am.index, dtype=float)
    ns = fit_nonstationary_gev(data, yrs, return_periods=rps)
    c0, loc0, sc0 = ns.stationary_params
    lr, p, aic_st = ns.lr_statistic, ns.p_value, ns.aic_stationary
    first, last = int(yrs.min()), int(yrs.max())
    horizon = int(horizon_year) if horizon_year else last + 25
    ybar = float(yrs.mean())

    def level(t: float, year: float, mu0: float, mu1: float, scale: float, shape: float) -> float:
        return float(genextreme.ppf(1.0 - 1.0 / t, shape, loc=mu0 + mu1 * (year - ybar), scale=scale))

    rows = []
    for t in rps:
        st = float(genextreme.ppf(1.0 - 1.0 / t, c0, loc=loc0, scale=sc0))
        a = level(t, first, ns.loc_intercept, ns.loc_trend, ns.scale, ns.shape)
        b = level(t, last, ns.loc_intercept, ns.loc_trend, ns.scale, ns.shape)
        h = level(t, horizon, ns.loc_intercept, ns.loc_trend, ns.scale, ns.shape)
        rows.append({"T": t, "stationary": _clean(st), "first_year": _clean(a), "last_year": _clean(b),
                     "horizon": _clean(h), "change_first_to_last_pct": _pct(b, a)})
    preferred = "nonstationary" if (p < 0.05 and ns.aic < aic_st) else "stationary"
    out.update({
        "stationary": {"shape": _clean(c0), "loc": _clean(loc0), "scale": _clean(sc0), "aic": _clean(aic_st),
                       "q_by_T": {_t_key(r["T"]): r["stationary"] for r in rows}},
        "nonstationary": {"mu0": _clean(ns.loc_intercept), "mu1_per_year": _clean(ns.loc_trend),
                          "mu1_pct_per_decade": _clean(100.0 * 10.0 * ns.loc_trend / ns.loc_intercept)
                          if ns.loc_intercept else None,
                          "scale": _clean(ns.scale), "shape": _clean(ns.shape), "aic": _clean(ns.aic),
                          "q_by_T_last_year": {_t_key(r["T"]): r["last_year"] for r in rows},
                          "q_by_T_horizon": {_t_key(r["T"]): r["horizon"] for r in rows}},
        "likelihood_ratio": {"statistic": _clean(lr), "p_value": _clean(p), "significant": bool(p < 0.05)},
        "first_year": first, "last_year": last, "horizon_year": horizon, "mean_year": _clean(ybar),
        "preferred": preferred, "table": rows,
    })
    # the level through time for the figure (the design T and the 10-year)
    grid = list(range(first, max(horizon, last) + 1))
    out["through_time"] = {"year": grid, **{
        _t_key(t): [_clean(level(t, y, ns.loc_intercept, ns.loc_trend, ns.scale, ns.shape)) for y in grid]
        for t in (10.0, max(rps))}}
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        t_hi = max(rps)
        draws = []
        mu_t = ns.loc_intercept + ns.loc_trend * (yrs - ybar)
        for _ in range(int(n_boot)):
            sim = genextreme.rvs(ns.shape, loc=mu_t, scale=ns.scale, random_state=rng)
            try:
                b = fit_nonstationary_gev(sim, yrs, return_periods=[t_hi])
                draws.append(level(t_hi, last, b.loc_intercept, b.loc_trend, b.scale, b.shape))
            except Exception:  # noqa: BLE001 - a failed refit is dropped, the count is reported
                continue
        draws = [d for d in draws if math.isfinite(d)]
        if len(draws) >= 20:
            lo, hi = np.percentile(draws, [5, 95])
            out["nonstationary"]["ci_last_year"] = {"T": t_hi, "lower": _clean(lo), "upper": _clean(hi),
                                                    "ci": [_clean(lo), _clean(hi)],
                                                    "n_boot": len(draws), "level": 0.9}
    if horizon > last:
        out["notes"].append(f"The {horizon} level extends the fitted trend {horizon - last} years beyond the record; "
                            "a trend is not a forecast, and the interval there is wider than the one quoted for "
                            f"{last}.")
    out["notes"].append("A sensitivity next to the stationary estimate, not a replacement: design-flood guidance "
                        "under a changing climate is not settled and nonstationary fits are fragile to the record's "
                        "ends (Wasko et al. 2024).")
    if preferred == "stationary":
        near = ", borderline" if p < 0.1 else ""
        out["verdict"] = (f"The trend term is not supported (likelihood-ratio p = {_p(p)}{near}); the stationary "
                          f"{_t_key(max(rps))}-year flow of {rows[-1]['stationary']:.4g} stands.")
    else:
        r = rows[-1]
        out["verdict"] = (f"The trend term is supported (likelihood-ratio p = {_p(p)}, AIC {ns.aic:.1f} vs "
                          f"{aic_st:.1f}): the {_t_key(r['T'])}-year level moved from {r['first_year']:.4g} in "
                          f"{first} to {r['last_year']:.4g} in {last} (stationary: {r['stationary']:.4g}).")
    return _clean(out)


# ── peaks over threshold ────────────────────────────────────────────────────


def _decluster(daily: pd.Series, threshold: float, min_sep: int) -> pd.Series:
    """Independent peaks above ``threshold``: exceedances closer than ``min_sep`` days are one event (its maximum)."""
    exc = daily[daily > threshold].dropna()
    if exc.empty:
        return exc
    peaks: list[tuple[pd.Timestamp, float]] = []
    cur_t, cur_v, last_t = exc.index[0], float(exc.iloc[0]), exc.index[0]
    for t, v in exc.iloc[1:].items():
        if (t - last_t).days > min_sep:
            peaks.append((cur_t, cur_v))
            cur_t, cur_v = t, float(v)
        elif float(v) > cur_v:
            cur_t, cur_v = t, float(v)
        last_t = t
    peaks.append((cur_t, cur_v))
    return pd.Series([v for _, v in peaks], index=pd.DatetimeIndex([t for t, _ in peaks]))


def pot_flood(
    source: str,
    station_id: str,
    *,
    threshold: float | None = None,
    events_per_year: float = 2.0,
    min_separation_days: int = 7,
    return_periods: list[float] | None = None,
    years: int | None = None,
) -> dict[str, Any]:
    """Flood frequency from every independent peak over a threshold (a Generalised Pareto), next to the annual-max GEV.

    Without ``threshold``, the one that yields about ``events_per_year``
    declustered peaks is used (2 by default: enough peaks, still extreme).
    Peaks closer than ``min_separation_days`` count as one event.
    """
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments, fit_gpd

    rps = _rps(return_periods)
    try:
        daily, meta = _station(source, station_id, years=years)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    d = daily.dropna()
    n_years = len(d) / 365.25
    out: dict[str, Any] = {**meta, "n_years": _clean(n_years), "return_periods": rps, "notes": [],
                           "methods": _m("gpd_pot")}
    if n_years < 10:
        out["error"] = f"{n_years:.1f} years of daily flow; peaks over threshold needs at least 10"
        return out
    sep = max(1, int(min_separation_days))
    if threshold is None:
        best: tuple[float, float, float] | None = None  # (distance to target, -threshold, threshold)
        for q in np.arange(0.995, 0.8999, -0.0025):
            u = float(d.quantile(q))
            rate = len(_decluster(d, u, sep)) / n_years
            cand = (abs(rate - events_per_year), -u, u)
            if best is None or cand < best:
                best = cand
        threshold = best[2] if best else float(d.quantile(0.95))
        out["threshold_rule"] = (f"the threshold between the 90th and 99.5th percentile of daily flow whose peaks come "
                                 f"closest to {events_per_year:g} a year")
    else:
        out["threshold_rule"] = "given"
    peaks = _decluster(d, float(threshold), sep)
    out.update({"threshold": _clean(threshold), "min_separation_days": sep, "n_peaks": len(peaks),
                "peaks_per_year": _clean(len(peaks) / n_years),
                "peaks": {"date": [t.date().isoformat() for t in peaks.index], "value": [_clean(v) for v in peaks]}})
    if len(peaks) < 10:
        out["error"] = f"{len(peaks)} independent peaks over {threshold:.4g}; a GPD needs at least 10"
        return out
    gpd = fit_gpd(peaks.to_numpy(dtype=float), float(threshold), return_periods=rps, total_observations=n_years)
    shape, _u, scale = gpd.params
    am = _annual_max(daily)
    gev = fit_gev_lmoments(am.to_numpy(dtype=float), return_periods=rps) if len(am) >= 10 else None
    rows = []
    for t in rps:
        g = gev.return_periods.get(t) if gev else None
        q = gpd.return_periods.get(t)
        rows.append({"T": t, "pot_gpd": _clean(q), "annual_max_gev": _clean(g), "difference_pct": _pct(q, g)})
    out.update({"gpd": {"shape": _clean(shape), "scale": _clean(scale),
                        "q_by_T": {_t_key(r["T"]): r["pot_gpd"] for r in rows}},
                "annual_max_gev": {"n_years": len(am), "q_by_T": {_t_key(r["T"]): r["annual_max_gev"]
                                                                  for r in rows}} if gev else None,
                "table": rows})
    if gev is not None:
        out["methods"].append({"name": "GEV fitted by L-moments", "text": "Annual-max comparison fit.",
                               "citation": "Hosking, J. R. M. (1990). J. R. Stat. Soc. B, 52(1), 105-124."})
    if shape is not None and shape > 0.3:
        out["notes"].append(f"GPD shape {shape:.2f} is a heavy tail: long return levels grow fast and are uncertain.")
    out["notes"].append("Return levels are daily-mean flows (not instantaneous peaks). Independence is by a "
                        f"{sep}-day separation; a slow, large river may need a longer one.")
    big = rows[-1]
    if big["difference_pct"] is not None:
        out["verdict"] = (f"{len(peaks)} independent peaks ({len(peaks) / n_years:.1f} a year) over "
                          f"{threshold:.4g}: the {_t_key(big['T'])}-year flow is {big['pot_gpd']:.4g} from peaks "
                          f"over threshold and {big['annual_max_gev']:.4g} from annual maxima "
                          f"({big['difference_pct']:+.0f}%).")
    return _clean(out)


# ── the catchment model (GR4J) and what-if runs ─────────────────────────────


def _forcing(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    from aquascope.problems import era5_daily

    frame, _meta = era5_daily(lat, lon, start=start, end=end,
                              variables=("precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_mean"))
    frame = frame.rename(columns={"precipitation_sum": "p", "et0_fao_evapotranspiration": "pet",
                                  "temperature_2m_mean": "t"})
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None) if getattr(frame.index, "tz", None) \
        else pd.DatetimeIndex(frame.index)
    return frame


def _flow_stats(q: pd.Series, flood_rp: float | None = None) -> dict[str, float | None]:
    q = q.dropna()
    if q.empty:
        return {"mean": None, "q95": None, "q05": None, "amax_median": None, "flood": None}
    am = q.groupby(q.index.year).max()
    out = {"mean": float(q.mean()), "q95": float(q.quantile(0.05)), "q05": float(q.quantile(0.95)),
           "amax_median": float(am.median()) if len(am) else None, "flood": None}
    if flood_rp is not None:
        try:
            out["flood"] = _gev_q(am.to_numpy(dtype=float), float(flood_rp))
        except (ValueError, FloatingPointError, RuntimeError):
            out["flood"] = None
    return out


#: A parameter set is behavioural when its validation KGE is within this of the calibrated set's.
BEHAVIOURAL_KGE = 0.05
#: The spread of the perturbed parameter sets, as a share of each parameter's bound width.
MEMBER_SPREAD = 0.05
_CHANGE_KEYS = ("mean_flow_change_pct", "q95_change_pct", "q05_change_pct", "flood_change_pct")


def _kge_on(o: pd.Series, s: pd.Series) -> float | None:
    from aquascope.analysis import metrics

    v = metrics.kge(o.to_numpy(dtype=float), s.to_numpy(dtype=float))
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _behavioural_ranges(members: int, best: dict[str, float], use_snow: bool, kge_valid: float | None,
                        runs: list[dict[str, Any]], plan: list[tuple[Any, ...]], rp: float | None, *,
                        simulate: Any, score: Any, valid_span: tuple[int, int], stats: Any,
                        scenario_stats: Any, seed: int = 42) -> dict[str, Any]:
    """The range of each scenario's changes over the parameter sets that do as well on the validation years.

    Adds ``range`` (``{key: [low, high]}``) to each run in place and returns what was done: how many sets were
    tried and kept, and the rule.
    """
    from aquascope.models.rainfall_runoff import GR4J_PARAM_BOUNDS, SNOW_PARAM_BOUNDS

    bounds = {**GR4J_PARAM_BOUNDS, **(SNOW_PARAM_BOUNDS if use_snow else {})}
    rng = np.random.default_rng(seed)
    floor = (kge_valid - BEHAVIOURAL_KGE) if kge_valid is not None else None
    kept: list[dict[str, float]] = [dict(best)]
    for _ in range(max(0, members - 1)):
        p = {k: float(np.clip(best[k] + rng.normal(0.0, MEMBER_SPREAD * (hi - lo)), lo, hi))
             for k, (lo, hi) in bounds.items() if k in best}
        k = score(*valid_span, simulate(p))
        if floor is not None and k is not None and k >= floor:
            kept.append(p)
    changes: list[list[dict[str, float | None]]] = [[] for _ in runs]
    for p in kept:
        b = stats(simulate(p))
        for i, (_sc, dp, dt, de, demand) in enumerate(plan):
            st = scenario_stats(dp, dt, de, demand, p)
            changes[i].append({"mean_flow_change_pct": _pct(st["mean"], b["mean"]),
                               "q95_change_pct": _pct(st["q95"], b["q95"]),
                               "q05_change_pct": _pct(st["q05"], b["q05"]),
                               "flood_change_pct": _pct(st["flood"], b["flood"]) if rp is not None else None})
    for r, ch in zip(runs, changes):
        span: dict[str, list[float] | None] = {}
        for key in _CHANGE_KEYS:
            if key == "flood_change_pct" and rp is None:
                continue
            v = np.array([c[key] for c in ch if c[key] is not None], dtype=float)
            if not len(v):
                span[key] = None
            elif len(v) >= 10:
                span[key] = [_clean(float(np.percentile(v, 5))), _clean(float(np.percentile(v, 95)))]
            else:
                span[key] = [_clean(float(v.min())), _clean(float(v.max()))]
        r["range"] = span
    return {"tried": int(members), "kept": len(kept), "spread": MEMBER_SPREAD, "kge_tolerance": BEHAVIOURAL_KGE,
            "rule": (f"{members} parameter sets around the calibrated one ({MEMBER_SPREAD:.0%} of each bound's "
                     f"width), kept when the validation KGE is within {BEHAVIOURAL_KGE} of the calibrated set's; "
                     "each range is the 5th to 95th percentile of the change across the kept sets (the extremes "
                     "when fewer than ten)."),
            "note": None if len(kept) >= 5 else
            f"Only {len(kept)} parameter set{'s' if len(kept) != 1 else ''} did as well on the validation years: "
            "the range is narrow because few were kept, not because the change is certain."}


def _given_params(params: Any) -> dict[str, float] | None:
    """Parameters handed back from an earlier answer, checked: X1-X4 (and DDF and T0 together), finite."""
    if not isinstance(params, dict):
        return None
    keys = ["X1", "X2", "X3", "X4"] + (["DDF", "T0"] if "DDF" in params or "T0" in params else [])
    try:
        out = {k: float(params[k]) for k in keys}
    except (KeyError, TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out.values()) else None


def _default_scenarios() -> list[dict[str, Any]]:
    return [{"label": "10% less rain", "dp_pct": -10.0}, {"label": "10% more rain", "dp_pct": 10.0},
            {"label": "2 °C warmer", "dt_c": 2.0}, {"label": "10% less rain and 2 °C warmer", "dp_pct": -10.0,
                                                    "dt_c": 2.0}]


def catchment_model(
    source: str,
    station_id: str,
    lat: float,
    lon: float,
    *,
    area_km2: float | None = None,
    years: int = MODEL_YEARS,
    scenarios: list[dict[str, Any]] | None = None,
    objective: str = "kge",
    maxiter: int = 25,
    warmup_days: int = 365,
    pet_pct_per_c: float = PET_PCT_PER_C,
    snow: str | bool = "auto",
    flood_rp: float | None = None,
    members: int = 0,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """GR4J calibrated on the gauge, validated on years it never saw, then run under "what if" changes.

    The flow (m3/s) is turned into depth over ``area_km2`` (mm/day) and the
    forcing is ERA5 precipitation and FAO-56 reference evapotranspiration for
    the point. The last ``years`` of overlap are split in two: the first half
    calibrates (after a ``warmup_days`` spin-up), the second half validates.
    A 90 % band from the validation residuals (heteroscedastic, in the square
    root of flow) is scored by how often the observations fall inside it.

    ``scenarios`` are ``{"label", "dp_pct", "dpet_pct", "dt_c"}``: rainfall
    and evaporation scaled by the percentages (a temperature change alone
    becomes ``pet_pct_per_c`` percent of evaporation per degree, a stated
    assumption). Each is simulated over the whole period with the calibrated
    parameters and reported as the change in mean flow, low flow (Q95, the
    flow exceeded 95 % of days), high flow (Q05) and the median annual maximum.
    A scenario's ``demand_cms`` takes a constant flow (m3/s) out of the
    simulated river, never below zero. ``source="geoglows"`` with a GEOGLOWS
    v2 ``river_id`` as ``station_id`` calibrates on that reach's simulated
    flow instead of a gauge (#565), and says so.

    ``flood_rp`` (a return period in years) adds the change in that flood: a
    GEV by L-moments on the simulated annual maxima, before and after. With
    ``members`` (for example 24) every change also gets a ``range``: that many
    parameter sets near the calibrated one (Gaussian, :data:`MEMBER_SPREAD` of
    each bound's width), kept when their validation KGE is within
    :data:`BEHAVIOURAL_KGE` of the calibrated set's on the years it never saw,
    each run with and without the scenario; the range is the spread of the
    change across the kept sets (5th to 95th percentile from ten, else the
    extremes). A fixed seed makes it the same every run. ``params`` (the
    ``params`` of an earlier answer for the same record) skips the
    calibration and repeats only the runs, so a second scenario is quick.

    ``snow`` puts a degree-day snow store in front of GR4J (two more
    calibrated parameters, :func:`aquascope.models.rainfall_runoff.degree_day_snow`).
    ``"auto"`` (the default) turns it on when at least
    :data:`SNOW_FRACTION` of the precipitation falls on days below 0 degrees C;
    a warming scenario then warms the snow store too, not only the evaporation.
    """
    from aquascope.analysis import metrics
    from aquascope.models.rainfall_runoff import (
        GR4J,
        calibrate,
        calibrate_snow,
        degree_day_snow,
        residual_quantile_bands,
    )

    out: dict[str, Any] = {"source": source, "station_id": station_id, "latitude": _clean(float(lat)),
                           "longitude": _clean(float(lon)), "notes": [],
                           "methods": _m("gr4j", "kge", "split_sample", "era5")}
    try:
        area = float(area_km2) if area_km2 is not None else None
    except (TypeError, ValueError):
        area = None
    if not area or area <= 0:
        out["error"] = ("the catchment area is needed to turn the gauge's flow into a depth for GR4J; pass area_km2 "
                        "(the describe_catchment step's sub_basin.up_area)")
        return out
    out["area_km2"] = _clean(area)
    try:
        # A reach's simulated record is read for the modelled years only (with the spin-up), not since 1940.
        daily, meta = _station(source, station_id, years=int(years) + 3 if source == GEOGLOWS_SOURCE else None)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out
    out.update({k: meta[k] for k in ("unit", "note")})
    if source == GEOGLOWS_SOURCE:
        out["calibrated_on"] = "modelled"
        out["notes"].append("Calibrated on GEOGLOWS v2 simulated flow for the reach, not on a gauge: GR4J here "
                            "emulates that model, and its skill says how well, not how true either is.")
    era5_start = date(1941, 1, 1)
    end = min(daily.dropna().index.max().date(), datetime.now(timezone.utc).date() - timedelta(days=10))
    start = max(daily.dropna().index.min().date(), era5_start, end - timedelta(days=int(years * 365.25)))
    fstart = start - timedelta(days=int(warmup_days))
    if (end - start).days < 5 * 365:
        out["error"] = f"{(end - start).days / 365.25:.1f} years of flow overlap ERA5; GR4J needs at least 5"
        return out
    try:
        forcing = _forcing(lat, lon, fstart, end)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"ERA5 forcing unavailable: {exc}"
        return out
    forcing = forcing.dropna(subset=["p", "pet"])
    q_mm = (daily * 86.4 / area).reindex(forcing.index)
    n = len(forcing)
    wu = int((pd.Timestamp(start) - forcing.index[0]).days) if n else 0
    wu = max(0, min(wu, n // 4))
    mid = wu + (n - wu) // 2
    cal_p, cal_e, cal_q = forcing["p"].iloc[:mid], forcing["pet"].iloc[:mid], q_mm.iloc[:mid]
    if cal_q.iloc[wu:].notna().sum() < 3 * 365:
        out["error"] = "fewer than three years of observed flow in the calibration half"
        return out
    cold = forcing["t"] < 0.0
    snow_share = float(forcing.loc[cold, "p"].sum() / forcing["p"].sum()) if forcing["p"].sum() > 0 else 0.0
    use_snow = bool(snow) if not isinstance(snow, str) else snow_share >= SNOW_FRACTION
    given = _given_params(params)
    if given is not None:
        use_snow = "DDF" in given   # the parameters say which model they belong to
    out["snow"] = {"used": use_snow, "share_below_freezing": _clean(snow_share), "rule": str(snow)}
    if use_snow:
        out["methods"] += _m("degree_day")
    if given is not None:
        # A previous calibration of this same record and forcing (the fit is seeded, so it is the same answer
        # again): only the runs are repeated. The skill below is recomputed from these parameters, not trusted.
        cal = SimpleNamespace(params=given)
        out["calibration_reused"] = True
    elif use_snow:
        cal = calibrate_snow(cal_p, cal_e, forcing["t"].iloc[:mid], cal_q, objective=objective, warmup_days=wu,
                             maxiter=int(maxiter))
    else:
        cal = calibrate(cal_p, cal_e, cal_q, objective=objective, warmup_days=wu, maxiter=int(maxiter))
    model = GR4J(**{k.lower(): cal.params[k] for k in ("X1", "X2", "X3", "X4")})

    def run(p_scale: float = 1.0, e_scale: float = 1.0, dt: float = 0.0, params: dict[str, float] | None = None
            ) -> pd.Series:
        m = model if params is None else GR4J(**{k.lower(): params[k] for k in ("X1", "X2", "X3", "X4")})
        p_in = forcing["p"] * p_scale
        if use_snow:
            pp = params if params is not None else cal.params
            p_in = degree_day_snow(p_in, forcing["t"] + dt, ddf=float(pp["DDF"]), t0=float(pp["T0"]))
        return m.simulate(p_in, forcing["pet"] * e_scale, warmup_days=wu).streamflow

    sim = run()
    obs = q_mm

    def skill(a: int, b: int) -> dict[str, Any]:
        o, s = obs.iloc[a:b].to_numpy(dtype=float), sim.iloc[a:b].to_numpy(dtype=float)
        return {"start": forcing.index[a].date().isoformat(), "end": forcing.index[b - 1].date().isoformat(),
                "n_days": int(np.isfinite(o).sum()), "kge": _clean(metrics.kge(o, s)),
                "nse": _clean(metrics.nse(o, s)), "log_nse": _clean(metrics.log_nse(o, s)),
                "pbias_pct": _clean(metrics.pbias(o, s))}

    calib, valid = skill(wu, mid), skill(mid, n)
    out.update({"params": {k: _clean(v) for k, v in cal.params.items()}, "objective": objective,
                "calibration": calib, "validation": valid, "record": meta})
    # the band, from validation residuals in sqrt space, scored on the same validation years
    try:
        bands = residual_quantile_bands(sim.iloc[mid:], obs.iloc[mid:], quantiles=(0.05, 0.5, 0.95),
                                        heteroscedastic=True)
        lo, hi = bands[0.05], bands[0.95]
        o = obs.iloc[mid:]
        inside = ((o >= lo) & (o <= hi))[o.notna()]
        out["band"] = {"level": 0.9, "coverage": _clean(float(inside.mean())) if len(inside) else None,
                       "note": "from the validation residuals; coverage near 0.9 means the band is honest"}
    except Exception as exc:  # noqa: BLE001
        out["band"] = {"level": 0.9, "coverage": None, "note": f"band not computed: {exc}"}
    # water balance (mm/year) over the evaluated period
    per = slice(wu, n)
    yrs = (n - wu) / 365.25
    wb = {"precipitation": float(forcing["p"].iloc[per].sum() / yrs),
          "pet": float(forcing["pet"].iloc[per].sum() / yrs),
          "observed_runoff": float(obs.iloc[per].mean() * 365.25) if obs.iloc[per].notna().any() else None,
          "simulated_runoff": float(sim.iloc[per].mean() * 365.25)}
    wb["runoff_ratio_observed"] = (wb["observed_runoff"] / wb["precipitation"]) if wb["observed_runoff"] and \
        wb["precipitation"] else None
    out["water_balance_mm_per_year"] = _clean(wb)
    # the reference-evaporation climatology GR4J was calibrated on (mm/day by day of year), for climate_projection
    clim = forcing["pet"].iloc[wu:].groupby(forcing.index[wu:].dayofyear).mean()
    out["pet_climatology"] = [_clean(float(clim.get(d, clim.mean()))) for d in range(1, 367)]
    # monthly means for the figure (observed and simulated, m3/s)
    to_cms = area / 86.4
    mo = pd.DataFrame({"observed": obs.iloc[wu:] * to_cms, "simulated": sim.iloc[wu:] * to_cms}).resample("MS").mean()
    out["monthly"] = {"month": [t.date().isoformat() for t in mo.index],
                      "observed": [_clean(v) for v in mo["observed"]],
                      "simulated": [_clean(v) for v in mo["simulated"]]}
    # what-if runs
    rp = float(flood_rp) if flood_rp is not None and float(flood_rp) > 1 else None
    base = _flow_stats(sim.iloc[wu:] * to_cms, rp)
    plan = []
    for sc in (scenarios if scenarios is not None else _default_scenarios()):
        if not isinstance(sc, dict):
            continue
        dp = float(sc.get("dp_pct") or 0.0)
        dt = float(sc.get("dt_c") or 0.0)
        de = float(sc["dpet_pct"]) if sc.get("dpet_pct") is not None else dt * float(pet_pct_per_c)
        demand = max(0.0, float(sc.get("demand_cms") or 0.0))
        plan.append((sc, dp, dt, de, demand))

    def scenario_stats(dp: float, dt: float, de: float, demand: float, params: dict[str, float] | None = None
                       ) -> dict[str, float | None]:
        s2 = run(1 + dp / 100.0, 1 + de / 100.0, dt, params).iloc[wu:] * to_cms
        if demand:
            s2 = (s2 - demand).clip(lower=0.0)
        return _flow_stats(s2, rp)

    runs = []
    for sc, dp, dt, de, demand in plan:
        st = scenario_stats(dp, dt, de, demand)
        label = str(sc.get("label") or (f"P {dp:+g}%, PET {de:+g}%" + (f", {demand:g} m3/s taken" if demand else "")))
        r = {"label": label, "dp_pct": dp, "dpet_pct": de,
             "dt_c": dt or None, "mean_flow_change_pct": _pct(st["mean"], base["mean"]),
             "q95_change_pct": _pct(st["q95"], base["q95"]), "q05_change_pct": _pct(st["q05"], base["q05"]),
             "amax_median_change_pct": _pct(st["amax_median"], base["amax_median"]),
             "mean_flow": _clean(st["mean"]), "q95": _clean(st["q95"])}
        if demand:
            r["demand_cms"] = demand
        if rp is not None:
            r["flood"] = _clean(st["flood"])
            r["flood_change_pct"] = _pct(st["flood"], base["flood"])
        runs.append(r)
    out["baseline_simulated"] = _clean({k: v for k, v in base.items() if k != "flood" or rp is not None})
    if rp is not None:
        out["flood_rp"] = rp
    out["scenarios"] = runs
    if members and int(members) > 0 and runs:
        out["ensemble"] = _behavioural_ranges(
            int(members), cal.params, use_snow, valid.get("kge"), runs, plan, rp,
            simulate=lambda params: run(params=params),
            score=lambda a, b, s_: _kge_on(obs.iloc[a:b], s_.iloc[a:b]), valid_span=(mid, n),
            stats=lambda s_: _flow_stats(s_.iloc[wu:] * to_cms, rp), scenario_stats=scenario_stats)
    if any(r["dt_c"] and r["dpet_pct"] == r["dt_c"] * pet_pct_per_c for r in runs):
        out["notes"].append(f"A warming scenario raises reference evaporation by an assumed {pet_pct_per_c:g}% per "
                            "degree; pass dpet_pct to set it.")
    if area > LUMPED_MAX_AREA_KM2:
        out["notes"].append(f"{area:,.0f} km2 is above GR4J's honest ceiling as a lumped model "
                            f"({LUMPED_MAX_AREA_KM2:,.0f} km2, #273): a poor fit is the structure, not the parameters.")
    kv = valid.get("kge")
    grade = "good" if kv is not None and kv >= 0.75 else "fair" if kv is not None and kv >= 0.5 else "poor"
    out["skill_grade"] = grade
    out["notes"].append("The scenario changes apply to the whole daily series (the same weather, scaled); they show "
                        "the catchment's sensitivity, not a projection. Use climate_projection for model futures.")
    how = {"good": "well", "fair": "fairly well", "poor": "poorly"}[grade]
    out["verdict"] = (f"GR4J{' with a snow store' if use_snow else ''} reproduces the flow {how} on years it was not "
                      "calibrated on (validation KGE "
                      f"{kv if kv is not None else float('nan'):.2f}, NSE {valid.get('nse') or float('nan'):.2f}).")
    if runs:
        worst = min(runs, key=lambda r: r["mean_flow_change_pct"] if r["mean_flow_change_pct"] is not None else 0)
        if worst["mean_flow_change_pct"] is not None:
            out["verdict"] += (f" Under '{worst['label']}' the mean flow changes by "
                               f"{worst['mean_flow_change_pct']:+.0f}% "
                               f"and the low flow (Q95) by {worst['q95_change_pct'] or 0:+.0f}%.")
    return _clean(out)


# ── CMIP6 projections ───────────────────────────────────────────────────────


#: The projections fetched in this process, by request (and the refusals, so a fallback does not repeat one).
_CMIP6_CACHE: dict[tuple[Any, ...], Any] = {}


def _cmip6(lat: float, lon: float, start: date, end: date, models: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Daily precipitation and mean temperature per model, bias-corrected onto ERA5-Land by Open-Meteo.

    The Climate API weighs a request by models, years and variables: by its
    own documentation seven models over a century of one variable counts as
    about 1,850 calls, so the two variables over 66 years here count as about
    2,400, against a free allowance of 10,000 calls a day per address (about
    four projections a day). Two variables only, each request made once per
    process (a refusal is remembered too), one retry after a per-minute refusal.
    """
    key = (round(float(lat), 3), round(float(lon), 3), start.isoformat(), end.isoformat(), tuple(models))
    if key in _CMIP6_CACHE:
        hit = _CMIP6_CACHE[key]
        if isinstance(hit, Exception):
            raise hit
        return hit
    try:
        out = _cmip6_fetch(lat, lon, start, end, models)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        if "429" in text or "limit" in text.lower():
            exc = RuntimeError("Open-Meteo refused the request (HTTP 429): the free allowance for climate projections "
                               "from this address is used up for now (a projection counts as about 2,400 of its "
                               "10,000 free calls a day). Try again later.")
        _CMIP6_CACHE[key] = exc
        raise exc from None
    _CMIP6_CACHE[key] = out
    return out


def _cmip6_fetch(lat: float, lon: float, start: date, end: date, models: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    import sys
    import time

    from aquascope.registry import build_collector

    col = build_collector("openmeteo", mode="climate")
    kwargs = {"latitude": float(lat), "longitude": float(lon), "start_date": start.isoformat(),
              "end_date": end.isoformat(), "models": ",".join(models),
              "daily": ["precipitation_sum", "temperature_2m_mean"]}
    try:
        raw = col.fetch_raw(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if "minute" not in str(exc).lower():
            raise
        if sys.platform == "emscripten":  # the browser worker does not wait a minute in silence
            raise RuntimeError("Open-Meteo's per-minute limit for climate projections was reached; run the step "
                               "again in a minute") from exc
        time.sleep(61)
        raw = col.fetch_raw(**kwargs)
    if raw.get("error"):
        raise RuntimeError(str(raw.get("reason") or raw))
    daily = raw.get("daily") or {}
    idx = pd.to_datetime(daily.get("time") or [])
    out: dict[str, pd.DataFrame] = {}
    for m in models:
        p = daily.get(f"precipitation_sum_{m}")
        t = daily.get(f"temperature_2m_mean_{m}")
        if p is None or t is None:
            continue
        frame = pd.DataFrame({"p": pd.Series(p, index=idx, dtype="float64"),
                              "t": pd.Series(t, index=idx, dtype="float64")})
        frame["t"] = frame["t"].interpolate(limit=5, limit_direction="both")
        frame["pet_oudin"] = oudin_pet(frame["t"], float(lat))
        if frame["p"].notna().sum() > 365:
            out[m] = frame
    return out


def oudin_pet(temperature: pd.Series, lat: float) -> pd.Series:
    """Oudin's temperature-based potential evaporation (mm/day): ``Re / (lambda rho) * (T + 5) / 100``, zero below -5 C.

    ``Re`` is the extraterrestrial radiation for the latitude and day of the
    year (FAO-56 eq. 21). Oudin et al. (2005) built it for lumped rainfall-runoff
    models such as GR4J, which is why the projection chain uses it for the
    change in evaporation.

    Oudin, L. et al. (2005). Which potential evapotranspiration input for a lumped rainfall-runoff model? Part 2.
    J. Hydrol., 303, 290-306.
    """
    doy = pd.DatetimeIndex(temperature.index).dayofyear.to_numpy(dtype=float)
    phi = math.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * math.pi * doy / 365)
    delta = 0.409 * np.sin(2 * math.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-math.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = 24 * 60 / math.pi * 0.0820 * dr * (ws * math.sin(phi) * np.sin(delta)
                                             + math.cos(phi) * np.cos(delta) * np.sin(ws))
    t = temperature.to_numpy(dtype=float)
    pet = np.where(t + 5 > 0, ra / 2.45 * (t + 5) / 100.0, 0.0)
    return pd.Series(pet, index=temperature.index, name="pet_oudin")


def _window(frame: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    return frame[(frame.index.year >= y0) & (frame.index.year <= y1)]


def _gev_q(am: np.ndarray, t: float) -> float | None:
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments

    am = am[np.isfinite(am)]
    if len(am) < 10:
        return None
    return float(fit_gev_lmoments(am, return_periods=[t]).return_periods.get(t))


def _ensemble(values: list[float | None]) -> dict[str, Any]:
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return {"n": 0, "median": None, "min": None, "max": None, "n_up": 0, "n_down": 0}
    return {"n": len(v), "median": _clean(float(np.median(v))), "min": _clean(min(v)), "max": _clean(max(v)),
            "n_up": sum(1 for x in v if x > 0), "n_down": sum(1 for x in v if x < 0)}


def climate_projection(
    lat: float,
    lon: float,
    *,
    params: dict[str, float] | None = None,
    area_km2: float | None = None,
    baseline: list[int] | tuple[int, int] = (1985, 2014),
    future: list[int] | tuple[int, int] = (2020, 2049),
    return_period: float = 20.0,
    models: list[str] | None = None,
    pet_climatology: list[float] | None = None,
) -> dict[str, Any]:
    """How the climate at the point, and the flow when a calibrated GR4J is given, changes in seven CMIP6 models.

    Each model is compared with itself (``future`` against ``baseline``
    years), so its bias cancels (change factors). Reported per model and as an
    ensemble (median, range, how many models agree on the sign): annual
    rainfall, potential evaporation (Oudin, from each model's temperature),
    mean temperature and the ``return_period`` wettest day. With ``params``
    (GR4J X1-X4, and DDF, T0 when the model has a snow store, from a
    ``catchment_model`` step) and ``area_km2``, each model's rainfall and
    temperature also drive the calibrated model: the change in mean flow, low
    flow (Q95) and the ``return_period`` daily flood.

    The evaporation GR4J sees is ``pet_climatology`` (the ERA5 reference
    evaporation by day of year it was calibrated on; the ``catchment_model``
    step returns it, else it is fetched) times each day's Oudin evaporation
    over the model's own baseline Oudin climatology: the magnitude the model
    was calibrated with, the change and the day-to-day variation the climate
    model gives.
    """
    b0, b1 = int(baseline[0]), int(baseline[1])
    f0, f1 = int(future[0]), int(future[1])
    if not (1951 <= b0 < b1 < f0 <= f1 <= 2049):
        return {"error": "baseline and future must be year ranges inside 1951-2049, baseline first"}
    use = tuple(m for m in (models or CMIP6_MODELS) if m in CMIP6_MODELS) or CMIP6_MODELS
    out: dict[str, Any] = {"latitude": _clean(float(lat)), "longitude": _clean(float(lon)),
                           "baseline": [b0, b1], "future": [f0, f1], "return_period": return_period, "notes": [],
                           "methods": _m("cmip6_highresmip", "delta_change", "oudin")}
    try:
        frames = _cmip6(lat, lon, date(b0 - 1, 1, 1), date(f1, 12, 31), use)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"CMIP6 projections unavailable: {exc}"
        return out
    out["n_models"] = len(frames)
    if len(frames) < MIN_MODELS:
        out["error"] = f"{len(frames)} CMIP6 models answered; an ensemble needs at least {MIN_MODELS}"
        return out
    gr4j = None
    snow_p: dict[str, float] | None = None
    if params and area_km2:
        from aquascope.models.rainfall_runoff import GR4J

        try:
            gr4j = GR4J(**{str(k).lower(): float(v) for k, v in params.items() if str(k).upper() in
                           ("X1", "X2", "X3", "X4")})
            snow_p = ({"ddf": float(params["DDF"]), "t0": float(params["T0"])}
                      if params.get("DDF") is not None and params.get("T0") is not None else None)
            out["methods"] += _m("gr4j")
        except Exception as exc:  # noqa: BLE001
            out["notes"].append(f"GR4J parameters not usable ({exc}); climate only.")
            gr4j = None
    clim: pd.Series | None = None
    if gr4j is not None:
        if pet_climatology and len(pet_climatology) >= 365:
            clim = pd.Series([float(v) if v is not None else np.nan for v in pet_climatology][:366],
                             index=range(1, len(pet_climatology[:366]) + 1))
        else:
            from aquascope.problems import era5_daily

            try:
                era, _meta = era5_daily(lat, lon, start=date(b0, 1, 1), end=date(b1, 12, 31),
                                        variables=("et0_fao_evapotranspiration",))
                clim = era["et0_fao_evapotranspiration"].groupby(era.index.dayofyear).mean()
            except Exception as exc:  # noqa: BLE001
                out["notes"].append(f"No evaporation climatology for GR4J ({exc}); climate only.")
                gr4j = None
        if clim is not None:
            clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    rows = []
    for m, fr in frames.items():
        b, f = _window(fr, b0, b1), _window(fr, f0, f1)
        row: dict[str, Any] = {"model": m}
        pb, pf = b["p"].groupby(b.index.year).sum().mean(), f["p"].groupby(f.index.year).sum().mean()
        eb = b["pet_oudin"].groupby(b.index.year).sum().mean()
        ef = f["pet_oudin"].groupby(f.index.year).sum().mean()
        row["precip_change_pct"] = _pct(float(pf), float(pb))
        row["pet_change_pct"] = _pct(float(ef), float(eb))
        row["temp_change_c"] = _clean(float(f["t"].mean() - b["t"].mean())) if b["t"].notna().any() else None
        qb = _gev_q(b["p"].groupby(b.index.year).max().to_numpy(dtype=float), return_period)
        qf = _gev_q(f["p"].groupby(f.index.year).max().to_numpy(dtype=float), return_period)
        row["wettest_day_change_pct"] = _pct(qf, qb)
        if gr4j is not None and clim is not None:
            ff = fr.dropna(subset=["p"])
            doy = ff.index.dayofyear
            base_clim = b["pet_oudin"].groupby(b.index.dayofyear).mean().reindex(range(1, 367)).interpolate(
                limit_direction="both")
            ratio = ff["pet_oudin"].to_numpy() / np.maximum(base_clim.reindex(doy).to_numpy(), 0.05)
            pet_in = pd.Series(clim.reindex(doy).to_numpy() * np.clip(ratio, 0.0, 5.0), index=ff.index)
            p_in = ff["p"]
            if snow_p is not None:
                from aquascope.models.rainfall_runoff import degree_day_snow

                p_in = degree_day_snow(p_in, ff["t"], **snow_p)
            sim = gr4j.simulate(p_in, pet_in, warmup_days=365).streamflow * float(area_km2) / 86.4
            sb, sf = _window(sim.to_frame("q"), b0, b1)["q"], _window(sim.to_frame("q"), f0, f1)["q"]
            row["mean_flow_change_pct"] = _pct(float(sf.mean()), float(sb.mean()))
            row["q95_change_pct"] = _pct(float(sf.quantile(0.05)), float(sb.quantile(0.05)))
            fb = _gev_q(sb.groupby(sb.index.year).max().to_numpy(dtype=float), return_period)
            fq = _gev_q(sf.groupby(sf.index.year).max().to_numpy(dtype=float), return_period)
            row["flood_change_pct"] = _pct(fq, fb)
        rows.append(_clean(row))
    out["models"] = rows
    keys = ["precip_change_pct", "pet_change_pct", "temp_change_c", "wettest_day_change_pct"]
    if gr4j is not None:
        keys += ["mean_flow_change_pct", "q95_change_pct", "flood_change_pct"]
    out["ensemble"] = {k: _ensemble([r.get(k) for r in rows]) for k in keys}
    out["notes"] += [
        "One emission pathway only: HighResMIP future runs follow the highest CMIP6 forcing (close to SSP5-8.5); "
        "a lower-emission future would change less. The horizon ends in 2050.",
        "Change factors (each model against its own baseline) cancel much of the model bias; the absolute values "
        "are not a forecast of the weather in any year.",
        "Report the spread across models, never the median alone (Wasko et al. 2024).",
        "Evaporation change is Oudin's temperature-based formula on each model's temperature (Oudin et al. 2005); "
        "a change in radiation, humidity or wind is not in it.",
    ]
    if gr4j is not None:
        out["notes"].append("The flow changes hold the calibrated GR4J parameters fixed and use daily-mean flow; land "
                            "use, abstraction and regulation are held as they were.")
    e = out["ensemble"]

    def say(key: str, what: str, unit: str = "%") -> str | None:
        s = e.get(key) or {}
        if not s.get("n"):
            return None
        sign = "more" if (s["median"] or 0) > 0 else "less"
        agree = max(s["n_up"], s["n_down"])
        return (f"{what} {s['median']:+.1f}{unit} (models {s['min']:+.1f} to {s['max']:+.1f}; {agree} of {s['n']} "
                f"agree on {sign})")

    bits = [x for x in (say("precip_change_pct", "annual rainfall"), say("temp_change_c", "temperature", " °C"),
                        say("wettest_day_change_pct", f"the {int(return_period)}-year wettest day"),
                        say("mean_flow_change_pct", "mean flow"), say("flood_change_pct",
                                                                      f"the {int(return_period)}-year flood"))
            if x]
    out["verdict"] = f"From {b0}-{b1} to {f0}-{f1}: " + "; ".join(bits) + "."
    out["attribution"] = "Open-Meteo.com (CC BY 4.0); CMIP6 HighResMIP model output (CC BY 4.0)."
    return _clean(out)


# ── more than one gauge ─────────────────────────────────────────────────────


def _bbox(lat: float, lon: float, radius_km: float) -> list[float]:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def regional_flood(
    lat: float,
    lon: float,
    *,
    radius_km: float = 75.0,
    source: str | None = None,
    station_id: str | None = None,
    max_sites: int = 25,
    return_period: float = 100.0,
) -> dict[str, Any]:
    """The gauges around the site studied together: an index-flood regional frequency analysis and field significance.

    Wraps :func:`aquascope.area_study.study_area` over a box of ``radius_km``
    around the point. When the study's own gauge (``source`` + ``station_id``)
    is in the pool, its at-site and pooled T-year flows are set side by side:
    pooling borrows strength from neighbours, which is what a short record needs.
    """
    from aquascope.area_study import study_area

    box = _bbox(float(lat), float(lon), float(radius_km))
    try:
        res = study_area(bbox=box, question="Regional flood study around the site", max_sites=int(max_sites),
                         max_live=min(10, int(max_sites)), n_sim=200)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"regional study failed: {exc}"}
    if res.get("error"):
        return res
    rf = res.get("regional_frequency") or {}
    out: dict[str, Any] = {"latitude": _clean(float(lat)), "longitude": _clean(float(lon)), "radius_km": radius_km,
                           "bbox": _clean(box), "summary": res.get("summary"), "headline": res.get("headline"),
                           "field_significance": res.get("field_significance"), "regional_frequency": rf,
                           "sites": res.get("sites"), "geojson": res.get("geojson"),
                           "notes": list(res.get("notes") or []),
                           "methods": list(res.get("methods") or []) or _m("index_flood"),
                           "n_studied": (res.get("summary") or {}).get("n_studied")}
    out["n_pooled"] = rf.get("n_sites") or out["n_studied"] or 0
    if source and station_id:
        key = f"{source}/{station_id}"

        def bare(sid: Any) -> str:  # "USGS-01013500" and "01013500" name the same gauge
            text = str(sid or "").lower()
            return text.split("-", 1)[1] if text.startswith(f"{source.lower()}-") else text

        row = next((s for s in (res.get("sites") or []) if s.get("source") == source
                    and bare(s.get("station_id")) == bare(station_id)), None)
        if row is not None:
            out["target"] = {"key": key, "at_site_q100": row.get("q100"), "pooled_q100": row.get("regional_q100"),
                             "years": row.get("record_years"), "n_amax": row.get("n_amax")}
        else:
            out["notes"].append(f"The study's gauge {key} is not in the regional pool (outside the box, or no usable "
                                "record there).")
    het = rf.get("heterogeneity") or {}
    h = het.get("H") if isinstance(het, dict) else None
    out["heterogeneity_H"] = _clean(h)
    if isinstance(h, (int, float)):
        out["verdict"] = (f"{out['n_pooled']} gauges pooled; heterogeneity H = {h:.2f} ("
                          + ("acceptably homogeneous" if h < 1 else "possibly heterogeneous" if h < 2 else
                             "definitely heterogeneous: the pooled curve is a weak guide") + ").")
    return _clean(out)


def compare_gauges(stations: list[dict[str, Any]] | None = None, *, lat: float | None = None,
                   lon: float | None = None, k: int = 3, years: int | None = None) -> dict[str, Any]:
    """Two to five gauges side by side: named, or the ``k`` nearest discharge gauges to the point."""
    from aquascope.compare import compare_stations

    if not stations:
        if lat is None or lon is None:
            return {"error": "give stations, or lat and lon to compare the nearest gauges"}
        from aquascope.mcp_server import find_stations

        near = find_stations(near=[float(lat), float(lon)], variable="discharge", limit=max(2, min(int(k), 5)))
        stations = [{"source": r.get("source"), "station_id": r.get("station_id"), "label": r.get("name")}
                    for r in (near.get("stations") or near.get("results") or [])]
    res = compare_stations(list(stations)[:5], years=years)
    return _clean(res)
