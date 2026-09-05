"""``(source, station) -> answer``: the entry point behind the Explorer and the MCP server.

:func:`analyze_station` fetches one station's observed record through
aquascope's own collectors and computes the Phase-0 analytics: hydrograph and
annual maxima, flood frequency (GEV by L-moments, Log-Pearson III with
confidence limits, an on-demand bootstrap GEV band), the flow-duration curve,
and a Mann-Kendall trend, plus the method citations. It runs unchanged in
CPython (tests, CLI, MCP) and inside Pyodide in the browser (the Explorer's
worker imports it from the wheel), so what any surface shows is exactly what
``pip install aquascope`` computes.

Everything returned is plain JSON: lists, dicts, numbers, strings, ``None``.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from aquascope.registry import SOURCES, build_collector

logger = logging.getLogger(__name__)

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
MIN_YEARS_FOR_FFA = 10

#: How far back a full-record request reaches when the catalog has no start
#: date for the station (#270). Generous on purpose: the agencies filter
#: server-side, so asking for years that do not exist costs nothing, while a
#: 40-year cap silently cut the Thames at Kingston (catalogued from 1883) to
#: 39 annual maxima.
FULL_RECORD_YEARS = 150

#: CWA CODIS answers one calendar year per request and each takes several
#: seconds at the source, so that fetch is capped rather than asked in full.
CWA_MAX_YEARS = 10

METHODS: dict[str, dict[str, str]] = {
    "gev_lmoments": {
        "name": "GEV fitted by L-moments",
        "text": "Annual maxima (calendar years) fitted to a Generalized Extreme Value distribution with "
        "L-moment estimators (Hosking 1990); return levels from the fitted quantile function.",
        "citation": "Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear "
        "combinations of order statistics. J. R. Stat. Soc. B, 52(1), 105-124.",
    },
    "lp3": {
        "name": "Log-Pearson III (Bulletin 17C style)",
        "text": "Log-transformed annual maxima fitted to a Pearson type III distribution; station skew, "
        "frequency factors and analytical confidence limits after Bulletin 17C.",
        "citation": "England, J. F. Jr. et al. (2018). Guidelines for determining flood flow frequency, "
        "Bulletin 17C. USGS Techniques and Methods 4-B5.",
    },
    "gev_bootstrap": {
        "name": "GEV (MLE, L-moment seeded) with bootstrap CI",
        "text": "Maximum-likelihood GEV seeded from L-moments with the shape bounded to |k| <= 0.5; "
        "90 % confidence bands from 1,000 bootstrap resamples of the annual maxima.",
        "citation": "Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.",
    },
    "fdc": {
        "name": "Flow-duration curve",
        "text": "Empirical exceedance probabilities of daily flow (Weibull plotting positions); "
        "Q95 and Q10 read from the curve.",
        "citation": "Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves I: new interpretation and "
        "confidence intervals. J. Water Resour. Plann. Manage., 120(4), 485-504.",
    },
    "era5": {
        "name": "ERA5 reanalysis via Open-Meteo",
        "text": "Daily precipitation, 2 m temperature and FAO-56 reference evapotranspiration for the grid cell "
        "(about 9 km) containing the point, from ECMWF's ERA5 reanalysis served by Open-Meteo.",
        "citation": "Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., 146, "
        "1999-2049; Open-Meteo.com (CC BY 4.0).",
    },
    "glofas": {
        "name": "GloFAS modelled discharge via Open-Meteo",
        "text": "Daily river discharge simulated by the Global Flood Awareness System (LISFLOOD, ~5 km grid) "
        "for the cell containing the point. Modelled, not observed: use it for context, not for design values.",
        "citation": "Harrigan, S. et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis "
        "1979-present. Earth Syst. Sci. Data, 12, 2043-2060.",
    },
    "fao56": {
        "name": "FAO-56 reference evapotranspiration and aridity index",
        "text": "ET0 after Allen et al. (1998); aridity index = mean annual precipitation / mean annual ET0 "
        "(UNEP classes: hyper-arid < 0.05, arid < 0.2, semi-arid < 0.5, dry sub-humid < 0.65, humid otherwise).",
        "citation": "Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration. "
        "FAO Irrigation and Drainage Paper 56.",
    },
    "spi": {
        "name": "Standardized Precipitation Index",
        "text": "Monthly precipitation accumulated over 1, 3 and 12 months, a gamma distribution fitted per calendar "
        "month (a point mass at zero), the probability mapped to a standard-normal score; at or below -1 is drought.",
        "citation": "McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and "
        "duration to time scales. Proc. 8th Conf. on Applied Climatology, 179-184; WMO (2012). Standardized "
        "Precipitation Index User Guide, WMO-No. 1090.",
    },
    "spei": {
        "name": "Standardized Precipitation-Evapotranspiration Index",
        "text": "The climatic water balance (precipitation minus PET) accumulated over the timescale, a log-logistic "
        "(generalized logistic) distribution fitted per calendar month by L-moments, the probability mapped to a "
        "standard-normal score; sees the evaporative-demand drought under warming that SPI misses.",
        "citation": "Vicente-Serrano, S. M., Begueria, S., & Lopez-Moreno, J. I. (2010). A multiscalar drought index "
        "sensitive to global warming: the Standardized Precipitation Evapotranspiration Index. J. Climate 23, "
        "1696-1718. doi:10.1175/2009JCLI2909.1",
    },
    "thornthwaite": {
        "name": "Thornthwaite potential evapotranspiration",
        "text": "Monthly PET from mean air temperature and the annual heat index, corrected for day length and the "
        "days in the month; a temperature-only approximation, the PET SPEI was introduced with.",
        "citation": "Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. "
        "Geographical Review 38, 55-94.",
    },
    "sgi_propagation": {
        "name": "SPI to SGI drought propagation",
        "text": "Cross-correlation between SPI at several accumulation periods, lagged 0 to 24 months, and the "
        "Standardised Groundwater Index; the accumulation period and lag that maximise it say how long a rainfall "
        "deficit takes to reach the water table.",
        "citation": "Bloomfield, J. P., & Marchant, B. P. (2013). Analysis of groundwater drought building on the "
        "standardised precipitation index approach. Hydrol. Earth Syst. Sci. 17, 4769-4787.",
    },
    "supply_reliability": {
        "name": "Run-of-river supply reliability (flow-duration screening)",
        "text": "The fraction of days (and of years without a shortfall) on which the flow, less an environmental "
        "reserve kept in the river (Q95 by default) and capped at an abstraction share of the flow, meets the "
        "demand; a screening rule in the flow-duration-curve tradition of environmental-flow practice, not a "
        "storage-yield analysis.",
        "citation": "Vogel, R. M., & Fennessey, N. M. (1994). Flow-duration curves I. J. Water Resour. Plann. "
        "Manage. 120, 485-504; Smakhtin, V., & Eriyagama, N. (2008). Developing a software package for global "
        "desktop assessment of environmental flows. Environ. Model. Softw. 23, 1396-1406; Acreman, M., & Dunbar, "
        "M. J. (2004). Defining environmental river flow requirements: a review. Hydrol. Earth Syst. Sci. 8, 861-876.",
    },
    "crop_water": {
        "name": "FAO-56 crop water requirement from reanalysis ET0",
        "text": "Reference ET0 (FAO-56 Penman-Monteith from ERA5 via Open-Meteo) times the single crop coefficient "
        "over the FAO-56 stage lengths, effective rainfall subtracted, the net depth divided by the irrigation "
        "efficiency; the season repeated over the years of the window and averaged, the range kept.",
        "citation": "Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration. FAO "
        "Irrigation and Drainage Paper 56; FAO (2025). Crop evapotranspiration, revised edition, "
        "doi:10.4060/cd6621en; reanalysis-forced ET0 bias: Agric. Water Manage. (2024), "
        "doi:10.1016/j.agwat.2024.108732.",
    },
    "trend": {
        "name": "Mann-Kendall trend on annual means",
        "text": "Non-parametric Mann-Kendall test with Sen's slope on the annual mean series.",
        "citation": "Mann, H. B. (1945). Nonparametric tests against trend. Econometrica, 13, 245-259; "
        "Sen, P. K. (1968). J. Am. Stat. Assoc., 63, 1379-1389.",
    },
    "who_screen": {
        "name": "WHO drinking-water guideline screen",
        "text": "Share of samples outside the WHO guideline range per recognised parameter; over 10 % is an alert, "
        "any exceedance a warning.",
        "citation": "World Health Organization (2022). Guidelines for drinking-water quality, 4th edition, "
        "incorporating the first and second addenda.",
    },
    "ccme_wqi": {
        "name": "CCME Water Quality Index 1.0",
        "text": "Scope (F1), frequency (F2) and amplitude (F3) of guideline exceedances over the sampled parameters, "
        "combined as 100 - sqrt(F1^2 + F2^2 + F3^2) / 1.732; Excellent 95-100, Good 80-94, Fair 65-79, "
        "Marginal 45-64, Poor 0-44. Guidelines: WHO 2022 (drinking), FAO 29 (irrigation) or CCME (aquatic life).",
        "citation": "CCME (2001). CCME Water Quality Index 1.0, User's Manual. Canadian Council of Ministers of the "
        "Environment, Winnipeg.",
    },
    "nsf_wqi": {
        "name": "NSF Water Quality Index",
        "text": "Nine parameters rated on their sub-index curves (digitised approximations of the published ones) and "
        "combined with the published weights; weights renormalised when parameters are missing.",
        "citation": "Brown, R. M., McClelland, N. I., Deininger, R. A. and Tozer, R. G. (1970). A water quality "
        "index: do we dare? Water and Sewage Works 117, 339-343.",
    },
    "iwqi": {
        "name": "Irrigation water quality (FAO 29)",
        "text": "SAR, sodium percentage and residual sodium carbonate (meq/L) with the USSL, Wilcox and Eaton classes, "
        "and the FAO 29 degree of restriction on use (none, slight to moderate, severe) per component.",
        "citation": "Ayers, R. S. and Westcot, D. W. (1985). Water quality for agriculture. FAO Irrigation and "
        "Drainage Paper 29, Rev. 1. FAO, Rome.",
    },
}


def _iso(d: Any) -> str | None:
    if d is None:
        return None
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


def _clean(x: Any) -> Any:
    """Make numbers JSON-safe (NaN/inf -> None)."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)
    return x


# ── fetching ────────────────────────────────────────────────────────────────


def _records_to_series(records: list, prefer: str | None = None) -> tuple[pd.Series | None, str, str]:
    """Turn a list of aquascope readings into (series, variable, unit).

    Handles StreamflowReading, WaterLevelReading, ClimateReading and
    WaterQualitySample (discharge / gage height / rainfall parameters).
    """
    if not records:
        return None, "", ""
    rows: list[tuple[datetime, float]] = []
    variable, unit = "", ""
    for r in records:
        name = type(r).__name__
        if name == "StreamflowReading":
            variable, unit = "discharge", getattr(r, "unit", "m3/s")
            rows.append((r.reading_datetime, float(r.discharge_cms)))
        elif name == "WaterLevelReading":
            variable, unit = "water_level", getattr(r, "unit", "m")
            rows.append((r.reading_datetime, float(r.water_level)))
        elif name == "ClimateReading":
            if r.parameter != "rainfall_mm":
                continue
            variable, unit = "precipitation", getattr(r, "unit", "mm")
            rows.append((r.sample_datetime, float(r.value)))
        elif name == "WaterQualitySample":
            p = (r.parameter or "").lower()
            if "discharge" in p or p == "q":
                var = "discharge"
            elif "gage" in p or "level" in p or p == "h":
                var = "water_level"
            elif "rain" in p or "precip" in p:
                var = "precipitation"
            else:
                continue
            if prefer and var != prefer:
                continue
            variable, unit = var, getattr(r, "unit", "")
            rows.append((r.sample_datetime, float(r.value)))
    if not rows:
        return None, "", ""
    s = pd.Series({t: v for t, v in rows}).sort_index()
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    # SI in, SI out: USGS gage height arrives in feet.
    if variable == "water_level" and str(unit).lower() in ("ft", "feet", "foot"):
        s = s * 0.3048
        unit = "m"
    return s, variable, unit


# Which agency parameter serves which archive variable.
_USGS_CODES = {"discharge": "00060", "water_level": "00065"}


def _parse_date(value: Any) -> date | None:
    """An ISO date (a date, a datetime or their string) as a date; ``None`` when it is not one."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def request_window(
    source: str, station_id: str, *, years: int | None = None, period_start: Any = None,
) -> dict[str, Any]:
    """The window a fetch asks for, and the words that say so (#270).

    ``years`` is an explicit cap: the last N years. Without it the whole record
    is requested, from the catalog's first date for the station when that is
    known (``period_start``, passed by a caller that holds the catalog row, else
    looked up in the catalog at hand, never downloaded), and otherwise from
    :data:`FULL_RECORD_YEARS` back. Returns ``start`` and ``end`` (dates),
    ``years`` (the cap, or ``None``), ``catalog_start`` (ISO string or ``None``)
    and ``asked``: the clause the fetch note carries, so the reader sees what
    was actually requested rather than what happened to come back.
    """
    end = datetime.now(timezone.utc).date()
    listed = _parse_date(period_start)
    if listed is None:
        from aquascope.archive.catalog import catalog_period

        listed = _parse_date(catalog_period(source, station_id)[0])
    if years is not None and years > 0:
        start = end - timedelta(days=int(years * 365.25))
        asked = f"last {int(years)} years requested (from {start.isoformat()})"
    elif listed is not None and listed < end:
        start = listed
        asked = f"full record requested (from {start.isoformat()}, the catalog's first date for this station)"
    else:
        start = end - timedelta(days=int(FULL_RECORD_YEARS * 365.25))
        asked = (
            f"full record requested (back to {start.isoformat()}; the catalog has no start date for this station)"
        )
    return {
        "start": start,
        "end": end,
        "years": int(years) if years else None,
        "catalog_start": listed.isoformat() if listed else None,
        "asked": asked,
    }


def _record_note(s: pd.Series | None, window: dict[str, Any]) -> str:
    """One sentence when the served record starts well after the catalog's first date.

    The Thames at Kingston is catalogued from 1883 and served from 1986: a
    reader who sees "full record requested" next to 39 annual maxima is told
    which of the two the agency actually answered with.
    """
    listed = _parse_date(window.get("catalog_start"))
    if s is None or s.empty or listed is None:
        return ""
    first = s.index.min().date()
    if (first - listed).days <= 366:
        return ""
    if window["start"] > listed:
        return (
            f" The catalog lists this station from {listed.isoformat()}; "
            f"only the last {window['years']} years were requested."
        )
    return f" The catalog lists this station from {listed.isoformat()}; the served record starts {first.isoformat()}."


def _fetched(s: pd.Series | None, var: str, unit: str, note: str, window: dict[str, Any]) -> dict[str, Any]:
    return {
        "series": s,
        "variable": var,
        "unit": unit,
        "note": note,
        "requested": {
            "start": window["start"].isoformat(),
            "end": window["end"].isoformat(),
            "years": window["years"],
            "catalog_start": window["catalog_start"],
        },
    }


def fetch_series(
    source: str,
    station_id: str,
    *,
    years: int | None = None,
    prefer_archive: bool = True,
    variable: str | None = None,
    period_start: Any = None,
) -> dict[str, Any]:
    """Fetch the observed record for one station.

    The Archive is tried first (a harvested daily file, one HTTPS GET, no
    agency load) when ``prefer_archive`` is set and the source is one the
    harvest mirrors; otherwise, or when the archive has no file for the
    station, the record comes straight from the agency through aquascope's
    collector. ``variable`` asks for one variable (``discharge``,
    ``water_level``, ``precipitation``, ``groundwater_level``); by default the
    source's variables are tried in its preferred order (discharge first).

    By default the full record is requested (#270): from the catalog's first
    date for the station when it is known, else :data:`FULL_RECORD_YEARS` back.
    ``years`` caps that to the last N years, and ``period_start`` is the
    catalog's first date when the caller already holds the station's row (the
    Explorer does), saving the lookup. See :func:`request_window`.

    Returns ``{"series": pd.Series | None, "variable": str, "unit": str,
    "note": str, "requested": dict}``; ``note`` says where the data came from,
    what was requested and any record-length limit of the source, and
    ``requested`` is ``{"start", "end", "years", "catalog_start"}`` as ISO
    strings, so a page can show the request beside the record it got.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}")
    window = request_window(source, station_id, years=years, period_start=period_start)
    start, end, asked = window["start"], window["end"], window["asked"]
    note = ""

    if prefer_archive:
        from aquascope.archive.observations import ARCHIVE_UNITS, fetch_archived_series, harvestable_variables

        candidates = (variable,) if variable else harvestable_variables(source)
        for var in candidates:
            if var not in harvestable_variables(source):
                continue
            archived = fetch_archived_series(source, station_id, var)
            if archived is None or archived.empty:
                continue
            if window["years"]:
                archived = archived[archived.index >= pd.Timestamp(start)]
                if archived.empty:
                    continue  # nothing in the capped window: let the agency say the same
            note = (
                f"From the AquaScope archive (daily {var.replace('_', ' ')} harvested from "
                f"{SOURCES[source].agency}; {archived.index.min().date()} to {archived.index.max().date()}); "
                f"{asked}."
            ) + _record_note(archived, window)
            return _fetched(archived, var, ARCHIVE_UNITS.get(var, ""), note, window)

    if source == "usgs":
        # Pass the catalog id as-is ("USGS-01646500" or another agency's "CA574-09527500");
        # the collector maps it onto NWIS (number + agencyCd) or the OGC monitoring_location_id.
        c = build_collector("usgs")
        span = (end - start).days
        s, var, unit = None, "", ""
        for want in (variable,) if variable else ("discharge", "water_level"):
            code = _USGS_CODES.get(want or "")
            if code is None:
                continue
            recs = c.collect(station_id=station_id, days=span, collection="daily", parameter=code, max_items=None)
            s, var, unit = _records_to_series(recs)
            if s is not None:
                break
        note = f"USGS daily values (NWIS); {asked}."
    elif source == "uk_ea":
        c = build_collector("uk_ea")
        measure, measure_var = _uk_ea_pick_measure(c, station_id, variable=variable)
        s = None
        var = unit = ""
        if measure:
            recs = c.collect(measure=measure, min_date=start.isoformat(), max_date=end.isoformat(), max_items=None)
            s, var, unit = _records_to_series(recs)
            if s is not None and measure_var == "groundwater_level":
                var = "groundwater_level"  # WaterLevelReading, but the measure is a borehole / tubewell
        note = f"Environment Agency Hydrology API, measure {measure or 'n/a'}; {asked}."
    elif source == "hubeau_hydrometrie":
        c = build_collector("hubeau_hydrometrie")
        s, var, unit = None, "", ""
        if variable in (None, "discharge"):
            # Long record first: obs_elab QmnJ is the daily mean discharge (multi-decade where the
            # station computes it); fall back to the real-time feed (last 30 days) for H-only stations.
            recs = c.collect(
                code_station=station_id, elaborated="QmnJ", date_debut_obs=start.isoformat(),
                date_fin_obs=end.isoformat(), size=20_000, max_items=None,
            )
            s, var, unit = _records_to_series(recs)
            note = f"Hub'Eau elaborated daily mean discharge (obs_elab QmnJ); {asked}."
        if s is None and variable in (None, "discharge"):
            recs = c.collect(code_station=station_id, grandeur_hydro="Q", days=30)
            s, var, unit = _records_to_series(recs)
            note = "Hub'Eau real-time observations (last 30 days); this station has no elaborated daily discharge."
        if s is None and variable in (None, "water_level"):
            recs = c.collect(code_station=station_id, grandeur_hydro="H", days=30)
            s, var, unit = _records_to_series(recs)
            note = "Hub'Eau real-time water level (last 30 days)."
    elif source == "pegelonline":
        c = build_collector("pegelonline")
        recs = c.collect(station_id=station_id, timeseries=("Q", "W"), days=31)
        s, var, unit = _records_to_series(recs, prefer=variable or "discharge")
        if s is None and not variable:
            s, var, unit = _records_to_series(recs)
        note = "PEGELONLINE serves the last 31 days only."
    elif source == "ireland_opw":
        c = build_collector("ireland_opw")
        recs = c.collect(stations=[{"properties": {"ref": station_id}}])
        s, var, unit = _records_to_series(recs)
        note = "waterlevel.ie month file (15-minute levels, last month)."
    elif source == "taiwan_cwa":
        # CODIS answers one calendar year per request and each takes several
        # seconds at the source, so the full record is never asked for here:
        # CWA_MAX_YEARS keeps the click-to-chart wait tolerable, and the note
        # says so rather than claiming the whole record was requested.
        cwa_years = min(int(years), CWA_MAX_YEARS) if years else CWA_MAX_YEARS
        cwa_start = end - timedelta(days=int(cwa_years * 365.25))
        window.update(start=cwa_start, years=cwa_years)
        c = build_collector("taiwan_cwa")
        recs = c.collect(station_ids=[station_id], start=cwa_start.isoformat(), end=end.isoformat())
        s, var, unit = _records_to_series(recs)
        note = (
            f"CWA CODIS daily rainfall, last {cwa_years} years requested, not the full record "
            f"(one request per year at the source, a few seconds each; capped at {CWA_MAX_YEARS} years)."
        )
    else:
        raise ValueError(f"{source} has no Explorer fetch path yet")

    if variable and s is not None and var != variable:
        s, var, unit = None, "", ""  # the station has no record of the variable asked for
    return _fetched(s, var, unit, note + _record_note(s, window), window)


# EA stations publish several measures per property (daily min / mean / max,
# 15-minute instantaneous, and for boreholes the manual "dipped" readings).
# One series at a time: per variable, the daily statistic that best stands
# for the day comes first; the 15-minute series is the last resort.
_UK_EA_VARIABLE_ORDER = ("discharge", "water_level", "precipitation", "groundwater_level")
_UK_EA_STAT_ORDER = (
    (86400, "mean"), (86400, "total"), (86400, "maximum"), (0, "instantaneous"), (900, "instantaneous"),
)


def _uk_ea_measure_variable(m: dict) -> str | None:
    """Archive variable served by an EA measure, from its parameter, unit and notation."""
    parameter = str(m.get("parameter") or "")
    unit = str(m.get("unitName") or m.get("unit") or "")
    mid = str(m.get("@id") or "")
    if parameter == "flow":
        return "discharge"
    if parameter == "rainfall":
        return "precipitation"
    if parameter == "groundwaterLevel" or "-gw-" in mid or unit.startswith("mAOD"):
        return "groundwater_level"
    if parameter == "level":
        return "water_level"
    return None


def _uk_ea_pick_measure(collector, station_id: str, *, variable: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(measure notation, variable)`` to fetch for a station, or ``(None, None)``.

    ``variable`` restricts the choice to measures serving that archive variable;
    otherwise variables are tried in ``_UK_EA_VARIABLE_ORDER`` (flow first).
    """
    try:
        data = collector.client.get_json(f"id/stations/{station_id}.json")
    except Exception as exc:  # noqa: BLE001
        logger.info("uk_ea station lookup failed for %s: %s", station_id, exc)
        return None, None
    items = data.get("items") or []
    station = items[0] if isinstance(items, list) and items else items
    measures = station.get("measures") or []
    if isinstance(measures, dict):
        measures = [measures]

    def stat(m: dict) -> str:
        v = m.get("valueStatistic")
        v = v.get("@id", "") if isinstance(v, dict) else str(v or "")
        return v.rsplit("/", 1)[-1]

    def notation(m: dict) -> str | None:
        mid = m.get("@id")
        return str(mid).rsplit("/", 1)[-1] if mid else None  # the collector wants the notation, not the URL

    wanted = (variable,) if variable else _UK_EA_VARIABLE_ORDER
    for want in wanted:
        mine = [m for m in measures if _uk_ea_measure_variable(m) == want and m.get("@id")]
        for period, statistic in _UK_EA_STAT_ORDER:
            for m in mine:
                if int(m.get("period") or 0) == period and stat(m) == statistic:
                    return notation(m), want
        if mine:
            daily = [m for m in mine if int(m.get("period") or 0) == 86400]
            return notation((daily or mine)[0]), want
    if variable or not measures:
        return None, None
    return notation(measures[0]), _uk_ea_measure_variable(measures[0])


# ── analytics ───────────────────────────────────────────────────────────────


def _annual_max(s: pd.Series) -> pd.Series:
    daily = s.resample("D").mean()
    counts = daily.resample("YS").count()
    am = daily.resample("YS").max()
    # keep years with at least ~80 % coverage so a partial first/last year does not fake a low maximum
    return am[counts >= 292].dropna()


def analyze_series(s: pd.Series, variable: str, unit: str) -> dict[str, Any]:
    """Compute Phase-0 analytics for a series. Pure function, JSON-safe output."""
    from aquascope.hydrology.flood_frequency import fit_gev_lmoments, fit_lp3
    from aquascope.hydrology.flow_duration import flow_duration_curve

    s = s.dropna()
    out: dict[str, Any] = {
        "variable": variable,
        "unit": unit,
        "n": int(len(s)),
        "start": _iso(s.index.min()) if len(s) else None,
        "end": _iso(s.index.max()) if len(s) else None,
        "years": round((s.index.max() - s.index.min()).days / 365.25, 1) if len(s) > 1 else 0.0,
        "stats": {
            "mean": _clean(float(s.mean())) if len(s) else None,
            "median": _clean(float(s.median())) if len(s) else None,
            "min": _clean(float(s.min())) if len(s) else None,
            "max": _clean(float(s.max())) if len(s) else None,
        },
        "methods": [],
        "notes": [],
    }
    if not len(s):
        return out

    # hydrograph: daily means, capped at ~25k points for the browser
    daily = s.resample("D").mean().dropna()
    if len(daily) > 25_000:
        daily = daily.iloc[:: int(math.ceil(len(daily) / 25_000))]
    out["series"] = {"t": [d.strftime("%Y-%m-%d") for d in daily.index], "v": [_clean(float(v)) for v in daily.values]}

    am = _annual_max(s)
    out["annual_max"] = {"year": [int(y) for y in am.index.year], "v": [_clean(float(v)) for v in am.values]}

    if variable == "discharge":
        fdc = flow_duration_curve(daily)
        step = max(1, len(fdc.exceedance) // 200)
        out["fdc"] = {
            "exceedance": [_clean(float(x)) for x in fdc.exceedance[::step]],
            "q": [_clean(float(x)) for x in fdc.discharge[::step]],
            "q95": _clean(float(fdc.percentiles.get(95, float("nan")))),
            "q50": _clean(float(fdc.percentiles.get(50, float("nan")))),
            "q10": _clean(float(fdc.percentiles.get(10, float("nan")))),
        }
        out["methods"].append(METHODS["fdc"])

        if len(am) >= MIN_YEARS_FOR_FFA:
            ffa: dict[str, Any] = {"n_years": int(len(am)), "return_periods": RETURN_PERIODS, "fits": {}}
            try:
                g = fit_gev_lmoments(am, return_periods=RETURN_PERIODS)
                ffa["fits"]["gev_lmoments"] = {
                    "q": [_clean(float(g.return_periods[rp])) for rp in RETURN_PERIODS],
                    "params": [_clean(float(p)) for p in g.params],
                }
                out["methods"].append(METHODS["gev_lmoments"])
            except Exception as exc:  # noqa: BLE001
                ffa["fits"]["gev_lmoments"] = {"error": str(exc)}
            try:
                lp3 = fit_lp3(am, return_periods=RETURN_PERIODS, ci_level=0.90)
                ffa["fits"]["lp3"] = {
                    "q": [_clean(float(lp3.return_periods[rp])) for rp in RETURN_PERIODS],
                    "ci": [[_clean(float(a)), _clean(float(b))] for a, b in
                           (lp3.confidence_intervals.get(rp, (float("nan"), float("nan"))) for rp in RETURN_PERIODS)],
                    "params": [_clean(float(p)) for p in lp3.params],
                }
                out["methods"].append(METHODS["lp3"])
            except Exception as exc:  # noqa: BLE001
                ffa["fits"]["lp3"] = {"error": str(exc)}
            out["ffa"] = ffa
        else:
            out["notes"].append(
                f"Flood frequency needs at least {MIN_YEARS_FOR_FFA} complete years of daily flow; this record has "
                f"{len(am)}."
            )

    # Trend on annual means of the well-covered years. "Well covered" is relative
    # to the record's own sampling (daily gauges vs monthly borehole dips), so a
    # 40-year manual groundwater record still gets a trend.
    counts = s.resample("YS").count()
    typical = float(counts[counts > 0].median()) if (counts > 0).any() else 0.0
    covered = counts[counts >= 0.8 * typical].index if typical else counts.index[:0]
    if len(covered) >= 8:
        try:
            from aquascope.analysis.trends import mann_kendall, sens_slope

            annual_mean = s.resample("YS").mean().reindex(covered).dropna()
            mk = mann_kendall(annual_mean.values)
            slope = sens_slope(annual_mean.values)
            out["trend"] = {
                "on": "annual mean",
                "p_value": _clean(float(mk.p_value)),
                "tau": _clean(float(mk.tau)),
                "trend": str(mk.trend),
                "sens_slope_per_year": _clean(float(slope.slope)),
                "n_years": int(mk.n_samples),
            }
            out["methods"].append(METHODS["trend"])
        except Exception as exc:  # noqa: BLE001
            logger.info("trend skipped: %s", exc)
    return out


def flood_ci(s: pd.Series) -> dict[str, Any]:
    """The slow part: bootstrap GEV confidence bands (called on demand)."""
    from aquascope.hydrology.flood_frequency import fit_gev

    am = _annual_max(s.dropna())
    r = fit_gev(am, return_periods=RETURN_PERIODS, ci_level=0.90)
    return {
        "q": [_clean(float(r.return_periods[rp])) for rp in RETURN_PERIODS],
        "ci": [[_clean(float(a)), _clean(float(b))] for a, b in
               (r.confidence_intervals.get(rp, (float("nan"), float("nan"))) for rp in RETURN_PERIODS)],
        "params": [_clean(float(p)) for p in r.params],
        "n_bootstrap": r.n_bootstrap,
        "n_bootstrap_discarded": r.n_bootstrap_discarded,
        "method": METHODS["gev_bootstrap"],
    }


def analyze_station(
    source: str,
    station_id: str,
    *,
    years: int | None = None,
    store: dict[str, Any] | None = None,
    variable: str | None = None,
    period_start: Any = None,
) -> dict[str, Any]:
    """Fetch + analyse one station. The entry point the browser worker calls.

    Pass ``store`` (any dict) to keep the fetched pandas Series under
    ``store["series"]`` for follow-up calls such as :func:`flood_ci` and
    :func:`to_csv` without a second fetch. ``variable`` picks one of the
    station's variables (default: the source's preferred one, discharge first).
    By default the full record is requested (#270); ``years`` caps it to the
    last N years, and ``period_start`` is the catalog's first date when the
    caller already holds the station's row. ``fetch_note`` in the result says
    what was requested and what came back; ``requested`` carries the window.
    """
    meta = SOURCES[source]
    fetched = fetch_series(source, station_id, years=years, variable=variable, period_start=period_start)
    if store is not None:
        store["series"] = fetched["series"]
        store["source"], store["station_id"] = source, station_id
    result: dict[str, Any] = {
        "source": source,
        "station_id": station_id,
        "agency": meta.agency,
        "license": meta.license,
        "attribution": meta.attribution,
        "fetch_note": fetched["note"],
        "requested": fetched.get("requested"),
    }
    s = fetched["series"]
    if s is None or s.empty:
        result.update({"n": 0, "error": "The source returned no observations for this station."})
        return result
    result.update(analyze_series(s, fetched["variable"], fetched["unit"]))
    return result


# ── water-quality samples ───────────────────────────────────────────────────
# The archive carries no water-quality variables until Phase 3 (#188), so the
# samples come straight from the agency: USGS daily water-quality values and
# the Water Quality Portal's discrete samples. A screening, not a bulk
# download: the window and the parameter list are capped by default.

#: The USGS daily-value parameter codes the catalog's ``water_quality`` flag stands for.
USGS_WQ_CODES: dict[str, str] = {"temperature": "00010", "conductivity": "00095", "dissolved_oxygen": "00300",
                                 "ph": "00400"}
_USGS_WQ_NAMES = {"temperature": "temperature", "temp": "temperature", "water temperature": "temperature",
                  "conductivity": "conductivity", "specific conductance": "conductivity", "ec": "conductivity",
                  "dissolved oxygen": "dissolved_oxygen", "dissolved_oxygen": "dissolved_oxygen",
                  "do": "dissolved_oxygen", "ph": "ph"}
#: WQP characteristic names asked for by default, per use: a screening list, not the portal's whole catalogue.
WQP_CHARACTERISTICS: dict[str, tuple[str, ...]] = {
    "drinking": ("pH", "Dissolved oxygen (DO)", "Temperature, water", "Specific conductance", "Turbidity",
                 "Nitrate", "Escherichia coli", "Arsenic", "Lead", "Fluoride"),
    "irrigation": ("Specific conductance", "Sodium", "Calcium", "Magnesium", "Potassium", "Bicarbonate",
                   "Chloride", "Boron", "pH", "Nitrate", "Total dissolved solids"),
    "aquatic life": ("pH", "Dissolved oxygen (DO)", "Temperature, water", "Nitrate", "Chloride", "Arsenic",
                     "Lead", "Copper", "Zinc", "Cadmium"),
}
WQ_DEFAULT_YEARS = 5
WQ_MAX_SAMPLES = 20_000


def water_quality_samples(
    source: str,
    station_id: str,
    *,
    years: int | None = None,
    parameters: list[str] | None = None,
    use: str | None = None,
    max_samples: int = WQ_MAX_SAMPLES,
) -> dict[str, Any]:
    """Sampled water-quality parameters at one station, as tidy rows with counts, units, period and licence.

    ``years`` caps the window (default :data:`WQ_DEFAULT_YEARS`, the last
    five years; ``0`` asks for the full record from the catalog's first date).
    ``parameters`` names what to fetch (USGS: temperature, conductivity,
    dissolved oxygen, pH, or their codes; WQP: characteristic names); without
    it the USGS four are fetched, or the WQP screening list for ``use``
    (``drinking`` by default, ``irrigation``, ``aquatic life``). The ``samples``
    rows (``datetime``, ``parameter``, ``value``, ``unit``) feed the workbench's
    ``wqi``, ``iwqi`` and ``who_screen`` directly, or a study step through
    ``from_step``.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}")
    meta = SOURCES[source]
    if "water_quality" not in meta.variables:
        raise ValueError(f"{source} carries no water-quality samples")
    cap = WQ_DEFAULT_YEARS if years is None else int(years)
    window = request_window(source, station_id, years=cap if cap > 0 else None)
    start, end, asked = window["start"], window["end"], window["asked"]
    use_key = str(use or "drinking").strip().lower().replace("_", " ")
    if use_key not in WQP_CHARACTERISTICS:
        use_key = "drinking"

    if source == "usgs":
        codes: list[str] = []
        for p in parameters or list(USGS_WQ_CODES):
            key = str(p).strip().lower()
            if key in USGS_WQ_CODES.values():
                codes.append(key)
            elif _USGS_WQ_NAMES.get(key) in USGS_WQ_CODES:
                codes.append(USGS_WQ_CODES[_USGS_WQ_NAMES[key]])
        if not codes:
            raise ValueError(f"USGS daily water-quality values cover {sorted(USGS_WQ_CODES)} only; got {parameters}")
        c = build_collector("usgs")
        recs = c.collect(station_id=station_id, days=(end - start).days, collection="daily",
                         parameter=",".join(sorted(set(codes))), statCd="00003", max_items=None)
        note = (f"USGS daily mean values (NWIS, statistic 00003) for parameter codes "
                f"{', '.join(sorted(set(codes)))}; {asked}.")
    elif source == "wqp":
        names = list(parameters) if parameters else list(WQP_CHARACTERISTICS[use_key])
        c = build_collector("wqp")
        recs = c.collect(site_id=station_id, characteristic_name=names, start_date=start.strftime("%m-%d-%Y"),
                         end_date=end.strftime("%m-%d-%Y"), max_results=int(max_samples))
        note = (f"Water Quality Portal (WQX 3.0) discrete samples for {len(names)} characteristics; {asked}. "
                "The portal is slow on large windows, so the window and the characteristic list are capped.")
    else:
        raise ValueError(
            f"{meta.label} has no water-quality fetch path yet; samples are served through aquascope for USGS and "
            "the Water Quality Portal (United States). The archive carries no water-quality variables until Phase 3 "
            "(#188)."
        )

    samples = [r for r in recs if type(r).__name__ == "WaterQualitySample"]
    rows = [
        {"datetime": r.sample_datetime.isoformat(), "parameter": r.parameter, "value": float(r.value),
         "unit": r.unit or ""}
        for r in samples
    ]
    rows.sort(key=lambda r: (r["parameter"], r["datetime"]))
    rows = rows[: int(max_samples)]
    result: dict[str, Any] = {
        "source": source,
        "station_id": station_id,
        "agency": meta.agency,
        "license": meta.license,
        "attribution": meta.attribution,
        "fetch_note": note,
        "requested": {"start": start.isoformat(), "end": end.isoformat(), "years": window["years"],
                      "catalog_start": window["catalog_start"]},
    }
    if not rows:
        result.update({"n_samples": 0, "n_parameters": 0, "samples": [], "sample_counts": {},
                       "error": "The source returned no water-quality samples for this station in the window."})
        return result
    frame = pd.DataFrame(rows)
    frame["t"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    per: dict[str, dict[str, Any]] = {}
    for name, g in frame.groupby("parameter", sort=True):
        units = g["unit"].astype(str).replace("", pd.NA).dropna()
        unit = str(units.mode().iloc[0]) if not units.empty else ""
        vals = g["value"].astype(float)
        per[str(name)] = {
            "n": int(len(g)), "unit": unit,
            "start": _iso(g["t"].min()), "end": _iso(g["t"].max()),
            "min": _clean(float(vals.min())), "median": _clean(float(vals.median())), "max": _clean(float(vals.max())),
        }
    units_all = [p["unit"] for p in per.values() if p["unit"]]
    t0, t1 = frame["t"].min(), frame["t"].max()
    result.update({
        "n_samples": int(len(rows)),
        "n_parameters": len(per),
        "parameters": per,
        "sample_counts": {k: v["n"] for k, v in per.items()},
        "units": {k: v["unit"] for k, v in per.items()},
        "unit": max(set(units_all), key=units_all.count) if units_all else "",
        "start": _iso(t0), "end": _iso(t1),
        "years": round((t1 - t0).days / 365.25, 2) if pd.notna(t0) and pd.notna(t1) else None,
        "samples": rows,
        "methods": [],
    })
    return result


def _aridity_class(index: float | None) -> str | None:
    if index is None:
        return None
    if index < 0.05:
        return "hyper-arid"
    if index < 0.2:
        return "arid"
    if index < 0.5:
        return "semi-arid"
    if index < 0.65:
        return "dry sub-humid"
    return "humid"


def anywhere(lat: float, lon: float, *, years: int = 10) -> dict[str, Any]:
    """The "hydrology of anywhere" card: ERA5 climate + FAO-56 ET0 + GloFAS modelled discharge for a point.

    Uses Open-Meteo (keyless, CORS-enabled) through the OpenMeteo collector,
    so it works from the browser worker too. Returns JSON only.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=7)  # ERA5 lags a few days
    start = end - timedelta(days=int(years * 365.25))
    out: dict[str, Any] = {
        "latitude": round(float(lat), 5),
        "longitude": round(float(lon), 5),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "years": years,
        "methods": [],
        "notes": [],
    }

    weather = build_collector("openmeteo", mode="weather")
    try:
        raw = weather.fetch_raw(
            latitude=lat, longitude=lon, start_date=start.isoformat(), end_date=end.isoformat(),
            daily=["precipitation_sum", "temperature_2m_mean", "et0_fao_evapotranspiration"],
        )
        daily = raw.get("daily", {})
        idx = pd.to_datetime(daily.get("time", []))
        p = pd.Series(daily.get("precipitation_sum", []), index=idx, dtype="float64")
        t = pd.Series(daily.get("temperature_2m_mean", []), index=idx, dtype="float64")
        et0 = pd.Series(daily.get("et0_fao_evapotranspiration", []), index=idx, dtype="float64")
        annual_p = p.resample("YS").sum(min_count=300)
        annual_et0 = et0.resample("YS").sum(min_count=300)
        monthly_p = p.groupby(p.index.month).mean() * 30.44
        monthly_et0 = et0.groupby(et0.index.month).mean() * 30.44
        mean_p = float(annual_p.dropna().mean()) if annual_p.notna().any() else None
        mean_et0 = float(annual_et0.dropna().mean()) if annual_et0.notna().any() else None
        aridity = (mean_p / mean_et0) if (mean_p is not None and mean_et0) else None
        out["climate"] = {
            "source": "ERA5 via Open-Meteo",
            "precipitation_mm_per_year": _clean(mean_p),
            "et0_mm_per_year": _clean(mean_et0),
            "temperature_mean_c": _clean(float(t.mean())) if len(t) else None,
            "aridity_index": _clean(aridity),
            "aridity_class": _aridity_class(aridity),
            "monthly_precipitation_mm": [_clean(float(monthly_p.get(m, float("nan")))) for m in range(1, 13)],
            "monthly_et0_mm": [_clean(float(monthly_et0.get(m, float("nan")))) for m in range(1, 13)],
            "annual_precipitation": {
                "year": [int(y) for y in annual_p.dropna().index.year],
                "mm": [_clean(float(v)) for v in annual_p.dropna().values],
            },
            "wettest_day_mm": _clean(float(p.max())) if len(p) else None,
        }
        out["methods"].extend([METHODS["era5"], METHODS["fao56"]])
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"ERA5 climate unavailable: {exc}")

    flood = build_collector("openmeteo", mode="flood")
    try:
        # GloFAS goes back to 1984 and is cheap: ask for at least 20 years so
        # the (indicative) flood frequency has enough complete years.
        flood_start = end - timedelta(days=int(max(years, 20) * 365.25))
        raw = flood.fetch_raw(latitude=lat, longitude=lon, start_date=flood_start.isoformat(),
                              end_date=end.isoformat(), daily=["river_discharge"])
        daily = raw.get("daily", {})
        idx = pd.to_datetime(daily.get("time", []))
        q = pd.Series(daily.get("river_discharge", []), index=idx, dtype="float64").dropna()
        if len(q):
            summary = analyze_series(q, "discharge", "m3/s")
            summary.pop("series", None)
            summary.pop("methods", None)
            summary["source"] = "GloFAS v4 (modelled) via Open-Meteo"
            summary["modelled"] = True
            out["glofas"] = summary
            out["notes"].append(
                "GloFAS discharge is a model output for the grid cell (about 5 km), not a gauge reading; "
                "return levels from it are indicative only."
            )
            out["methods"].append(METHODS["glofas"])
            if "ffa" in summary:
                out["methods"].extend([METHODS["gev_lmoments"], METHODS["lp3"]])
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"GloFAS discharge unavailable: {exc}")

    out["attribution"] = (
        "Open-Meteo.com (CC BY 4.0); ERA5 and GloFAS: Copernicus Climate Change and Emergency Management Services."
    )
    return out


def to_csv(result: dict[str, Any]) -> str:
    """CSV of the daily series in a result dict (for the download button)."""
    series = result.get("series") or {"t": [], "v": []}
    unit = result.get("unit", "")
    lines = [f"date,{result.get('variable', 'value')}_{unit}".replace("/", "_per_")]
    lines += [f"{t},{'' if v is None else v}" for t, v in zip(series["t"], series["v"])]
    return "\n".join(lines) + "\n"


# ── reconnaissance ──────────────────────────────────────────────────────────
# What exists at a place, and what that supports, before any analysis runs.
# The catalog gives the record spans (no agency call), BasinATLAS the
# catchment, the similarity search the donors; aquascope.methods turns them
# into the sufficiency table. Same function behind `aquascope assess`, the
# MCP tool, the Analyst tool and the Explorer card.

#: Variables a method in the registry can consume.
RECORD_VARIABLES = ("discharge", "water_level", "precipitation", "groundwater_level", "water_quality")
#: Sources whose live feed is a short window, whatever the catalog span says.
SERVED_WINDOW = {"pegelonline": "the last 31 days", "ireland_opw": "the last month"}
_NEAR_CANDIDATES = 400
_MAX_STATIONS_LISTED = 25
_DONOR_K = 10
_STALE_AFTER_YEARS = 5


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _span_years(start: Any, end: Any, today: date) -> float | None:
    """Record length in years from the catalog span; an open end runs to today."""
    s = _parse_date(start)
    if s is None:
        return None
    e = _parse_date(end) or today
    return max(0.0, round((e - s).days / 365.25, 1))


def _station_entry(row: dict[str, Any], lat: float, lon: float, today: date) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "station_id": row.get("station_id"),
        "name": row.get("name"),
        "distance_km": round(_haversine_km(lat, lon, float(row["latitude"]), float(row["longitude"])), 1),
        "variables": [v for v in (row.get("variables") or []) if v],
        "period_start": row.get("period_start"),
        "period_end": row.get("period_end"),
        "years": _span_years(row.get("period_start"), row.get("period_end"), today),
        "url": row.get("url"),
    }


def _label(st: dict[str, Any]) -> str:
    name = st.get("name") or st.get("station_id")
    return f"{name} ({st['source']}/{st['station_id']})"


def _catchment_subset(desc: dict[str, Any]) -> dict[str, Any]:
    """The few catchment facts the sufficiency table and a card need, from describe_catchment."""
    sb = desc.get("sub_basin") or {}
    attrs = desc.get("attributes") or {}

    def value(key: str) -> Any:
        entry = attrs.get(key)
        return entry.get("value") if isinstance(entry, dict) else entry

    up_area = attrs.get("upstream_area_km2")
    if not isinstance(up_area, (int, float)):
        up_area = sb.get("up_area")
    area = attrs.get("area_km2") if isinstance(attrs.get("area_km2"), (int, float)) else up_area
    return {
        "hybas_id": sb.get("hybas_id"),
        "area_km2": _clean(float(area)) if isinstance(area, (int, float)) else None,
        "upstream_area_km2": _clean(float(up_area)) if isinstance(up_area, (int, float)) else None,
        "n_sub_basins": (desc.get("upstream") or {}).get("n_sub_basins"),
        "elevation_m": value("elevation_m"),
        "precipitation_mm_yr": value("precipitation_mm_yr"),
        "aridity": value("aridity_index"),
        "dams": value("degree_of_regulation_pct"),
        "source": "BasinATLAS (HydroATLAS v1.0)",
    }


def assess_site(
    lat: float,
    lon: float,
    *,
    radius_km: float = 50.0,
    problem: str | None = None,
    return_period: float | None = None,
    area_km2: float | None = None,
    donors: int | None = None,
) -> dict[str, Any]:
    """What can be answered at a place: the gauges in reach, the catchment, and what the record supports.

    Reads the published station catalog only (true catalog spans, no agency
    call), asks BasinATLAS for the catchment and the similarity search for
    donors, builds a :class:`aquascope.methods.SiteContext` and returns the
    sufficiency table for every method (or those for one ``problem``), each
    row carrying the station it would use. ``area_km2`` and ``donors`` let a
    caller that already knows them (the Explorer page holds both) skip those
    lookups. Everything returned is plain JSON.

    Returns ``{"point", "stations", "catchment", "context", "sufficiency", "notes"}``.
    """
    from aquascope.archive.catalog import load_stations, search_stations
    from aquascope.methods import METHODS, SiteContext, sufficiency_table

    lat, lon = float(lat), float(lon)
    radius_km = float(radius_km)
    known_problems = sorted({p for m in METHODS.values() for p in m.problems})
    if problem is not None and problem not in known_problems:
        raise ValueError(f"unknown problem {problem!r}; one of {known_problems}")
    notes: list[str] = []
    today = datetime.now(timezone.utc).date()

    # ── inventory: the nearest catalog stations, true spans, honest distances
    rows = load_stations()
    nearby = [
        _station_entry(r, lat, lon, today)
        for r in search_stations(rows, near=(lat, lon), limit=_NEAR_CANDIDATES)
        if r.get("latitude") is not None and r.get("longitude") is not None
    ]
    nearby.sort(key=lambda s: s["distance_km"])
    within = [s for s in nearby if s["distance_km"] <= radius_km]

    years_by: dict[str, float] = {}
    resolution_by: dict[str, str] = {}
    station_by: dict[str, dict[str, Any]] = {}
    unspanned: dict[str, dict[str, Any]] = {}
    for st in within:
        for var in st["variables"]:
            if var not in RECORD_VARIABLES or var in station_by:
                continue
            if st["years"] is None:
                unspanned.setdefault(var, st)
                continue
            years_by[var] = st["years"]
            resolution_by[var] = "daily"
            station_by[var] = st
    # Only the variables a method in the table consumes deserve a "nearest gauge is too far" note.
    wanted = {m.variable for m in METHODS.values() if m.variable and (problem is None or problem in m.problems)}
    for var in RECORD_VARIABLES:
        if var in station_by or var not in wanted:
            continue
        if var in unspanned:
            st = unspanned[var]
            notes.append(f"{_label(st)} measures {var.replace('_', ' ')} but the catalog has no record span for it; "
                         "not counted.")
            continue
        farther = next((s for s in nearby if var in s["variables"] and s["years"] is not None), None)
        if within and farther is not None and farther["distance_km"] > radius_km:
            notes.append(
                f"Nearest {var.replace('_', ' ')} gauge is {_label(farther)} at {farther['distance_km']:,.0f} km, "
                f"beyond the {radius_km:g} km radius; not counted."
            )
    if not within:
        if nearby:
            notes.append(f"No catalog gauge within {radius_km:g} km; the nearest is {_label(nearby[0])} at "
                         f"{nearby[0]['distance_km']:,.0f} km.")
        else:
            notes.append("No catalog gauge near this point.")
    if years_by:
        notes.append("Record resolution is not in the catalog; daily is assumed for every variable.")
    used: dict[tuple[str, str], list[str]] = {}
    for var, st in station_by.items():
        used.setdefault((st["source"], st["station_id"]), []).append(var)
    for key, vars_ in used.items():
        st = station_by[vars_[0]]
        what = " and ".join(v.replace("_", " ") for v in vars_)
        window = SERVED_WINDOW.get(st["source"])
        if window:
            notes.append(f"{_label(st)} lists {st['years']:g} years of {what} but the source serves only {window}; "
                         "a computed answer will not see the full span.")
        if st["years"] < 2:
            notes.append(f"The catalog span for {_label(st)} is only {st['years']:g} yr; suspiciously short, the "
                         "agency may hold more.")
        end = _parse_date(st["period_end"])
        if end is not None and (today - end).days > _STALE_AFTER_YEARS * 365:
            notes.append(f"The {what} record at {_label(st)} ends in {end.year}.")

    # ── catchment (BasinATLAS), unless the caller already knows the area
    catchment: dict[str, Any]
    if area_km2 is not None:
        catchment = {"area_km2": _clean(float(area_km2)), "upstream_area_km2": _clean(float(area_km2)),
                     "source": "caller"}
        notes.append("Catchment area supplied by the caller; BasinATLAS was not consulted.")
    else:
        from aquascope.mcp_server import describe_catchment

        desc = describe_catchment(lat, lon)
        if desc.get("error"):
            catchment = {"error": str(desc["error"])}
            notes.append(f"Catchment not described: {desc['error']}")
        else:
            catchment = _catchment_subset(desc)
    ctx_area = catchment.get("upstream_area_km2") or catchment.get("area_km2")

    # ── donors for the regionalisation path
    ctx_donors: int | None
    if donors is not None:
        ctx_donors = int(donors)
        notes.append("Donor count supplied by the caller.")
    else:
        from aquascope.mcp_server import similar_basins

        sim = similar_basins(lat=lat, lon=lon, k=_DONOR_K)
        if sim.get("error"):
            ctx_donors = None
            notes.append(f"Donor search not available: {sim['error']}")
        else:
            ctx_donors = len(sim.get("stations") or [])
            pool = sim.get("n_candidates")
            if isinstance(pool, int):
                notes.append(f"{ctx_donors} donor gauges from a pool of {pool:,} gauged catchments.")

    # ── point products: the ERA5 / GloFAS path applies to any point on land
    available = {"glofas", "temperature", "forcing"}
    notes.append("ERA5 temperature and forcing and GloFAS discharge are assumed reachable for any point on land "
                 "(Open-Meteo); not checked here.")
    notes.append("CMIP6 change factors need model output you supply (aquascope.climate works on downloaded data); "
                 "not counted.")

    ctx = SiteContext(
        years_by_variable=years_by,
        resolution_by_variable=resolution_by,
        area_km2=float(ctx_area) if isinstance(ctx_area, (int, float)) else None,
        return_period=float(return_period) if return_period is not None else None,
        donors=ctx_donors,
        available=available,
    )
    if ctx.ungauged:
        notes.append(f"No gauge with a usable record within {radius_km:g} km: at-site methods are not defensible; "
                     "what remains is the regionalisation path (similar_basins, regionalize_signatures) and the "
                     "GloFAS cross-check.")

    # ── the table, each row with the station it would use
    longest = max(years_by, key=years_by.get) if years_by else None
    table = sufficiency_table(ctx, problem=problem)
    for row in table:
        pre = METHODS[row["method"]]
        st = None
        if pre.variable is not None:
            st = station_by.get(pre.variable)
        elif pre.min_years is not None and not pre.ungauged and longest is not None:
            st = station_by[longest]
        row["station"] = {"source": st["source"], "station_id": st["station_id"]} if st else None

    # The nearest few, plus the station each variable's span came from: at a dense site the well that gives
    # ``years_by_variable.groundwater_level`` can be the 30th nearest, and a playbook reading the context would
    # otherwise select a branch whose station is not in the list.
    listed = within[:_MAX_STATIONS_LISTED]
    keys = {(s["source"], s["station_id"]) for s in listed}
    for st in station_by.values():
        if (st["source"], st["station_id"]) not in keys:
            listed.append(st)
            keys.add((st["source"], st["station_id"]))

    return {
        "point": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "stations": listed,
        "catchment": catchment,
        "context": {
            "years_by_variable": years_by,
            "resolution_by_variable": resolution_by,
            "area_km2": ctx.area_km2,
            "return_period": ctx.return_period,
            "donors": ctx.donors,
            "available": sorted(available),
            "ungauged": ctx.ungauged,
        },
        "sufficiency": table,
        "notes": notes,
    }
