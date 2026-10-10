"""aquascope-mcp: the world's public gauges and aquascope's analyses as MCP tools (#113).

Run it with ``aquascope mcp`` (stdio). Every MCP-speaking assistant (Claude
Desktop, Claude Code, Cursor, ...) can then find stations anywhere on Earth
from the published catalog, pull the observed record through aquascope's
collectors, and get flood frequency, flow duration and trend with citations,
without writing Python.

Design rules:

* Tools read the Archive first (fast, no agency call) and only touch an
  agency API when a series is asked for.
* Responses are bounded: catalog searches are capped, series are resampled
  and truncated, analyses drop the raw arrays. An LLM context is not a
  data lake.
* Everything is plain JSON and comes from the same functions the CLI and the
  Explorer use (``aquascope.registry``, ``aquascope.archive.catalog``,
  ``aquascope.explore``).

Requires the ``mcp`` extra (``pip install "aquascope[mcp]"``). Works with the
official Python SDK 1.x (``FastMCP``) and 2.x (``MCPServer``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aquascope import __version__
from aquascope.registry import SOURCES, redistributable_sources, station_sources
from aquascope.schemas.station import VARIABLES

logger = logging.getLogger(__name__)

SERVER_NAME = "aquascope"
INSTRUCTIONS = (
    "AquaScope gives you the world's public water gauges (USGS, UK EA, Hub'Eau, PEGELONLINE, Ireland OPW, "
    "Greece Hydroscope, Taiwan CWA and more) behind one schema. Start with find_stations (no agency call), "
    "then get_timeseries or "
    "analyze_station for a specific station. For a place or a station, assess_site(lat, lon) first says which "
    "methods the record there supports; do not run one it marks not_defensible. place_context(lat, lon) gives "
    "the flood history, surface water, flood hazard, dams, soil, evapotranspiration and nearest rain gauge of a "
    "place. Flood frequency needs at least "
    "10 complete years of daily flow. For a river with no gauge, snap_to_river then reach_record gives 86 years "
    "of simulated flow, labelled modelled; model_skill says how well each global model reproduces a gauge (graded "
    "A to D) and model_to_lean_on which one to trust near a site. Always show the licence/attribution returned "
    "with the data."
)

MAX_STATIONS = 200
MAX_POINTS = 2_000


def _server():
    """Return an MCP server instance from whichever SDK generation is installed."""
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer

        return MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, version=__version__)
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP

            return FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The MCP server needs the 'mcp' package. Install it with: pip install 'aquascope[mcp]'"
            ) from exc


# ── tool implementations (plain functions, testable without an MCP client) ──


def list_sources() -> dict[str, Any]:
    """Every data source aquascope knows: agency, country, variables, licence, whether it has a station catalog."""
    out = []
    for key in sorted(SOURCES):
        m = SOURCES[key]
        out.append(
            {
                "key": key,
                "label": m.label,
                "agency": m.agency,
                "country": m.country,
                "region": m.region,
                "variables": list(m.variables),
                "station_catalog": m.supports_station_lookup,
                "supports_bbox": m.supports_bbox,
                "requires_api_key": m.requires_api_key,
                "license": m.license,
                "redistributable": m.redistributable,
                "homepage": m.homepage,
            }
        )
    return {
        "n_sources": len(out),
        "with_station_catalog": station_sources(),
        "redistributable": redistributable_sources(),
        "variables": list(VARIABLES),
        "sources": out,
    }


def find_stations(
    query: str | None = None,
    bbox: list[float] | None = None,
    near: list[float] | None = None,
    variable: str | None = None,
    sources: list[str] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Search the published station catalog (no agency call).

    query: words from the station name, id or river, accent-insensitive ("Kingston Thames" finds the Thames at
    Kingston). bbox: [west, south, east, north] in degrees.
    near: [lat, lon]; results are ordered nearest-first. variable: one of the registry vocabulary
    (discharge, water_level, precipitation, groundwater_level, ...). Returns at most ``limit`` (<= 200)
    sites with representative ids you can pass to get_timeseries / analyze_station.
    Each entry includes record_count; multi-record sites include all records and their individual ids.
    The representative satisfies the search filters; other records may measure different variables.
    """
    from aquascope.archive.catalog import load_stations, search_stations

    if variable and variable not in VARIABLES:
        return {"error": f"unknown variable {variable!r}; allowed: {list(VARIABLES)}"}
    limit = max(1, min(int(limit or 25), MAX_STATIONS))
    rows = load_stations()
    hits = search_stations(
        rows,
        bbox=tuple(bbox) if bbox else None,
        near=tuple(near) if near else None,
        variable=variable,
        sources=sources,
        query=query,
        limit=limit,
        group_sites=True,
    )
    fields = ("source", "station_id", "site_id", "name", "latitude", "longitude", "variables",
              "period_start", "period_end", "river", "country", "agency", "license", "url")
    slim = []
    for row in hits:
        entry = {k: row.get(k) for k in fields}
        entry["record_count"] = row["record_count"]
        if row["record_count"] > 1:
            entry["records"] = [{k: record.get(k) for k in fields} for record in row["records"]]
            entry["site_note"] = f"{row['record_count']} records at this site"
        slim.append(entry)
    return {"n_catalog": len(rows), "n_returned": len(slim), "limit": limit, "stations": slim}


def get_timeseries(
    source: str,
    station_id: str,
    years: int = 10,
    resample: str = "D",
    max_points: int = 400,
    variable: str | None = None,
) -> dict[str, Any]:
    """Observed record for one station (archive first, then the agency), resampled and bounded.

    resample: 'D' daily, 'W' weekly, 'M' monthly means, 'Y' annual means. Values beyond ``max_points``
    are thinned evenly (never more than 2,000). variable: discharge (default), water_level,
    precipitation or groundwater_level, for stations that have several. Returns unit, variable, stats
    and the points as [date, value] pairs, plus licence and attribution.
    """
    import pandas as pd

    from aquascope.explore import fetch_series

    if source not in SOURCES:
        return {"error": f"unknown source {source!r}"}
    if variable and variable not in VARIABLES:
        return {"error": f"unknown variable {variable!r}; allowed: {list(VARIABLES)}"}
    meta = SOURCES[source]
    fetched = fetch_series(source, station_id, years=int(years), variable=variable)
    s = fetched["series"]
    if s is None or s.empty:
        return {"source": source, "station_id": station_id, "n": 0, "error": "no observations returned",
                "note": fetched["note"]}
    rule = {"D": "D", "W": "W", "M": "MS", "Y": "YS"}.get(str(resample).upper(), "D")
    r = s.resample(rule).mean().dropna()
    max_points = max(10, min(int(max_points or 400), MAX_POINTS))
    step = max(1, -(-len(r) // max_points))
    thinned = r.iloc[::step]
    return {
        "source": source,
        "station_id": station_id,
        "variable": fetched["variable"],
        "unit": fetched["unit"],
        "start": s.index.min().date().isoformat(),
        "end": s.index.max().date().isoformat(),
        "n_observations": int(len(s)),
        "resample": rule,
        "n_points": int(len(thinned)),
        "thinning_step": step,
        "stats": {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max())},
        "points": [[d.date().isoformat(), None if pd.isna(v) else round(float(v), 4)] for d, v in thinned.items()],
        "note": fetched["note"],
        "license": meta.license,
        "attribution": meta.attribution,
    }


def water_quality_samples(
    source: str,
    station_id: str,
    years: int | None = None,
    parameters: list[str] | None = None,
    use: str | None = None,
) -> dict[str, Any]:
    """Sampled water-quality parameters at one station: USGS daily water-quality values (temperature,
    conductivity, dissolved oxygen, pH) or Water Quality Portal discrete samples, as tidy rows (datetime,
    parameter, value, unit) with per-parameter counts, units and period, plus licence and attribution. A
    screening, not a bulk download: the last 5 years and a short parameter list by default (the WQP is slow on
    large windows); years=0 asks for the full record. use (drinking, irrigation, aquatic life) picks the WQP
    parameter list. Feed the rows to analyse_table(csv, "wqi" | "iwqi" | "who_screen").
    """
    from aquascope.explore import water_quality_samples as _samples

    if source not in SOURCES:
        return {"error": f"unknown source {source!r}"}
    try:
        return _samples(source, station_id, years=years, parameters=parameters, use=use)
    except ValueError as exc:
        return {"error": str(exc)}


def analyze_station(
    source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False, variable: str | None = None,
    return_periods: list[float] | None = None, exclude_years: list[int] | None = None,
) -> dict[str, Any]:
    """Fetch and analyse one station: record summary, annual maxima, flood frequency (GEV L-moments and
    Log-Pearson III with 90 % CI; optional bootstrap GEV band), flow-duration percentiles, Mann-Kendall
    trend, and the method citations. Raw daily arrays are omitted; use get_timeseries for those.
    exclude_years drops those years' annual maxima from the flood fit and its tests.
    variable picks one of the station's variables (discharge by default; water_level, precipitation,
    groundwater_level where the station has them). By default the full record is requested, from the
    catalog's first date for the station; years caps it to the last N years. fetch_note in the result says
    what was requested and what the agency actually served.
    """
    from aquascope.explore import analyze_station as _analyze
    from aquascope.explore import flood_ci

    if source not in SOURCES:
        return {"error": f"unknown source {source!r}"}
    if variable and variable not in VARIABLES:
        return {"error": f"unknown variable {variable!r}; allowed: {list(VARIABLES)}"}
    store: dict[str, Any] = {}
    res = _analyze(source, station_id, years=int(years) if years else None, store=store, variable=variable,
                   return_periods=return_periods, exclude_years=exclude_years)
    res.pop("series", None)
    if "fdc" in res:
        res["fdc"] = {k: res["fdc"][k] for k in ("q95", "q50", "q10")}
    if bootstrap_ci and res.get("ffa") and store.get("series") is not None:
        try:
            ci = flood_ci(store["series"], return_periods=return_periods, exclude_years=exclude_years)
            res["ffa"]["fits"]["gev_bootstrap"] = {
                k: ci[k] for k in ("q", "ci", "params", "n_bootstrap", "n_bootstrap_discarded",
                                     "estimator", "interval_method", "ci_level") if k in ci
            }
            res.setdefault("methods", []).append(ci["method"])
        except Exception as exc:  # noqa: BLE001
            res.setdefault("notes", []).append(f"bootstrap CI failed: {exc}")
    return res


def flood_frequency(
    source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False,
    return_periods: list[float] | None = None, exclude_years: list[int] | None = None,
) -> dict[str, Any]:
    """Return levels for T = 2, 5, 10, 25, 50, 100 years at a station (subset of analyze_station); pass
    return_periods to add others (a 200-year design). years caps the record to the last N years; by default the
    full record is requested. exclude_years drops those years' annual maxima from the fit.
    """
    res = analyze_station(source, station_id, years=years, bootstrap_ci=bootstrap_ci, return_periods=return_periods,
                          exclude_years=exclude_years)
    return _flood_result(res)


def _flood_result(res: dict[str, Any]) -> dict[str, Any]:
    """The compact flood payload, also used by Studio before retaining its input separately."""
    if "error" in res:
        return res
    keep = {k: res.get(k) for k in ("source", "station_id", "agency", "license", "attribution", "unit",
                                    "start", "end", "years", "n", "stats", "ffa", "notes", "methods",
                                    "fetch_note", "requested", "variable", "data_snapshot", "archive_revision",
                                    "software_revision", "eligibility", "annual_max",
                                    "annual_max_excluded")}
    # The annual maxima are the fitted sample (complete years only): the frequency curve plots them at their
    # plotting positions, without which the fit cannot be checked by eye.
    for key in ("annual_max", "annual_max_excluded"):
        if keep.get(key) is None:
            keep.pop(key, None)
    if not keep.get("ffa"):
        keep["error"] = "flood frequency not available (see notes)"
    return keep


def assess_site(
    lat: float,
    lon: float,
    radius_km: float = 50.0,
    problem: str | None = None,
    return_period: float | None = None,
) -> dict[str, Any]:
    """What can be answered at a place, before any analysis. Call this first for a question about a place or a
    station. Returns the gauges within radius_km from the catalog (true record spans, no agency call), the
    BasinATLAS catchment, the site context (years per variable, area, donors) and a sufficiency table: for every
    method, defensible | marginal | not_defensible here, the reason (record length, resolution, catchment size
    for a lumped model, return period against record length, donors), the tool that runs it and the station it
    would use. Respect it: do not run a method marked not_defensible, say why, and offer what is defensible.
    problem narrows the table: flood_risk, ungauged_flow, drought, groundwater_decline, supply_reliability,
    climate_change, irrigation, water_quality. return_period is the T the question asks for, if any.
    """
    from aquascope.explore import assess_site as _assess

    try:
        return _assess(float(lat), float(lon), radius_km=float(radius_km), problem=problem or None,
                       return_period=float(return_period) if return_period is not None else None)
    except ValueError as exc:
        return {"error": str(exc)}


def study_area(
    bbox: list[float] | None = None,
    stations: list[str] | None = None,
    question: str | None = None,
    max_sites: int = 60,
    max_live: int = 25,
) -> dict[str, Any]:
    """Study the gauges of an area together (flood focus). bbox: [west, south, east, north]; or stations: a list of
    "source/station_id". Reads the AquaScope Archive first and fetches at most max_live gauges live from an agency
    (keyless rate limits); the rest are listed as skipped. Returns a per-site table (record span, mean, Q100 from a
    GEV by L-moments, Q100 per km2 where the area is known, Mann-Kendall trend on annual maxima), field
    significance of the trends (counts up/down/none, Benjamini-Hochberg FDR, Walker test, with the independence
    caveat), an index-flood regional growth curve (L-moments, discordancy, heterogeneity H), GeoJSON points and a
    one-line headline. Report the headline and the caveats; do not claim a regional trend the field test rejects.
    """
    from aquascope import area_study as _area

    if not bbox and not stations:
        return {"error": "give bbox [west, south, east, north] or stations ['source/station_id', ...]"}
    try:
        return _area.study_area(stations or None, bbox=tuple(bbox) if bbox and not stations else None,
                                question=question, max_sites=max(1, min(int(max_sites), _area.MAX_SITES)),
                                max_live=max(0, min(int(max_live), _area.MAX_LIVE_FETCHES)))
    except ValueError as exc:
        return {"error": str(exc)}


def snap_to_river(lat: float, lon: float, max_distance_m: float = 1000.0, prefer: str = "main",
                  area_km2: float | None = None) -> dict[str, Any]:
    """Snap a point to its GEOGLOWS v2 river reach (about 6.8 million worldwide): its river_id, Strahler order and
    how far it is. Among the reaches within max_distance_m the main channel wins (the highest stream order, the
    nearer on a tie), and the answer names a smaller stream that was nearer; prefer="nearest" takes the nearest
    line. Give area_km2 (a gauge's catchment area) to take the reach whose upstream area matches it. Beyond
    max_distance_m it says there is no stream within that distance and names the nearest reach found, so a
    hillside is not taken for a river. Call it before reach_record or trace_downstream when you only have a
    place."""
    from aquascope import rivers

    try:
        return rivers.snap_to_river(lat, lon, max_distance_m=max_distance_m, prefer=prefer, area_km2=area_km2)
    except ValueError as exc:
        return {"error": str(exc)}


def reach_record(river_id: int | None = None, lat: float | None = None, lon: float | None = None,
                 years: int | None = None, return_periods: list[float] | None = None) -> dict[str, Any]:
    """The simulated daily discharge of a river reach since 1940 (GEOGLOWS v2, MODELLED, not measured), analysed
    the way a gauge is: annual maxima, return periods (GEV L-moments and Log-Pearson III with 90 % CI),
    flow-duration percentiles, the monthly regime and a Mann-Kendall trend. Give the river_id from snap_to_river,
    or lat/lon to snap here (a point with no stream within 1 km gets an error, not a record). A gauge on the
    same river outranks it; say it is modelled whenever you quote it. The daily arrays are left out here."""
    from aquascope import rivers

    return rivers.reach_summary(river_id, lat=lat, lon=lon, years=years, return_periods=return_periods)


def model_skill(source: str | None = None, station_id: str | None = None, lat: float | None = None,
                lon: float | None = None, area_km2: float | None = None, models: list[str] | None = None,
                years: int = 30) -> dict[str, Any]:
    """How well each global model reproduces a gauge (the evidence ladder): GEOGLOWS v2 and GloFAS scored live,
    NWM v3 (US) and Google GRRR from the published monthly table. Per model: KGE with r, alpha and beta, NSE,
    percent bias, the 2-, 10- and 100-year flows of gauge and model (each from its own GEV fit) with the error in
    %, and a grade A to D (A: KGE >= 0.75, B >= 0.5, C > -0.41, else D; one letter lower when the 100-year flow is
    off by more than 50 %). Give the station's source and station_id (lat/lon/area_km2 are then optional).
    `sentence` says which model fits best and where they disagree; quote it, and say models are modelled."""
    from aquascope import evidence

    res = evidence.model_skill(source, station_id, lat=lat, lon=lon, area_km2=area_km2, models=models,
                               years=years or None)
    res.pop("series", None)
    return res


def model_to_lean_on(lat: float, lon: float, radius_km: float = 150.0) -> dict[str, Any]:
    """Which global model (GEOGLOWS, NWM, Google GRRR) tracked the gauges near a site best, from the published
    monthly skill table: the median KGE of each over the nearest graded gauges within radius_km. For an ungauged
    site, it says which model's numbers to lean on and why."""
    from aquascope import evidence

    return evidence.lean_on(lat, lon, radius_km=radius_km)


def upstream_area(river_id: int, lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    """The area draining to a GEOGLOWS v2 river reach (km2) and how many reaches lie upstream, summed from the
    model's unit catchments (within about 7 % of four agency-published gauge areas in our checks). lat/lon,
    where the reach roughly is (snap_to_river gives them), only pick which processing unit is read first."""
    from aquascope import rivers

    return rivers.upstream_area(river_id, lat=lat, lon=lon)


def trace_downstream(river_id: int | None = None, lat: float | None = None, lon: float | None = None,
                     gauge_km: float = 2.0, dam_km: float = 2.0) -> dict[str, Any]:
    """Follow a river reach (or the reach a point snaps to) down to its outlet: how many reaches, how many km,
    where it ends, the catalog gauges within gauge_km of the path in the order the water reaches them, the
    Global Dam Watch dams within dam_km of it (name, capacity in million m3, purpose, degree of regulation where
    GDW gives it, km along the path), the countries it crosses (Natural Earth) and the dams upstream of the
    first reach. The path geometry is thinned to 400 points (TDX-Hydro, CC BY-SA 4.0: for display)."""
    from aquascope import rivers

    res = rivers.trace_downstream(river_id, lat=lat, lon=lon, gauge_km=gauge_km, dam_km=dam_km, max_points=400)
    reaches = res.get("reaches") or []
    if len(reaches) > 40:
        res["reaches"] = reaches[:20] + reaches[-20:]
        res["reaches_note"] = f"{len(reaches)} reaches; the first and last 20 are listed."
    dams = res.get("dams") or []
    if len(dams) > 30:
        res["dams"] = sorted(dams, key=lambda d: d.get("capacity_mcm") or 0.0, reverse=True)[:30]
        res["dams"].sort(key=lambda d: d.get("along_km") or 0.0)
        res["dams_note"] = f"{len(dams)} dams on the path; the 30 with the most storage are listed."
    return res


def upstream_dams(river_id: int | None = None, lat: float | None = None, lon: float | None = None,
                  with_flow: bool = True) -> dict[str, Any]:
    """Is a river regulated upstream of a reach (or of the reach a point snaps to)? The Global Dam Watch dams that
    drain to it (largest storage first), their total storage in million m3, and with_flow the degree of
    regulation: that storage as a % of a year's mean flow at the reach (GEOGLOWS v2, modelled; one more 10 s
    request). Approximate: each dam is matched to its nearest river reach. Very large basins are not searched."""
    from aquascope import rivers

    return rivers.upstream_dams(river_id, lat=lat, lon=lon, with_flow=with_flow)
def flow_status(source: str, station_id: str, date: str | None = None) -> dict[str, Any]:
    """Today against normal at a gauge: where its latest flow (or level) sits against the same days of the year
    (7 either side) in every other year of its record, as a percentile and one of the five classes the USGS
    dashboard and WMO HydroSOS use (much below normal, below, normal, above, much above). The Archive copy is
    topped up with the agency's newest days first. date (YYYY-MM-DD) asks about another day. Needs 10 years in
    that window; quote the sentence it returns, which says the date and how many years it rests on."""
    from aquascope import nownext

    try:
        res = nownext.station_status(source, station_id, date=date)
    except ValueError as exc:
        return {"error": str(exc)}
    res.pop("recent", None)
    return res


def status_bulletin(month: str | None = None, sources: list[str] | None = None,
                    country: str | None = None) -> dict[str, Any]:
    """The month's state of the rivers in the WMO HydroSOS style: every Archive gauge's monthly mean flow placed
    against the same month in its other years (25 days a month, 10 years, else left out and counted) in the five
    classes (much below normal to much above normal), rolled up per country and per BasinATLAS river basin, with
    the new monthly records, the gauges furthest from normal, the coverage and a summary paragraph written by rules.
    month is YYYY-MM (default the latest published bulletin, else the last full month); the published bulletin is
    read when there is one, else it is built from the Archive's discharge records (slow the first time). country
    (ISO3, e.g. GBR) lists that country's classed gauges. Quote the summary; it says how many gauges it rests on."""
    from aquascope import bulletin

    try:
        res = bulletin.status_bulletin(month, sources)
    except ValueError as exc:
        return {"error": str(exc)}
    gauges = res.pop("gauges", None) or []
    res["basins"] = (res.get("basins") or [])[:15]
    if country:
        code = country.strip().upper()
        res["gauges"] = [g for g in gauges if (g.get("country") or "").upper() == code][:200]
    return res


def flood_warnings(bbox: list[float] | None = None, min_rp: int = 2, limit: int = 50) -> dict[str, Any]:
    """Floods ahead: the river reaches the GEOGLOWS v2 global forecast expects to reach their 2-year flow in the next
    15 days, from the daily published issue (Strahler order 5 and up, ensemble-mean daily peak against the reach's
    own 2- to 100-year flows). bbox is [west, south, east, north] in degrees; min_rp keeps only reaches at or above
    that return period (2, 5, 10, 25, 50 or 100). Returns the issue date, counts by class, the reaches (highest class
    first, at most limit) with peak flow, peak day, class and the share of members that agree, and the method. MODEL
    OUTPUT, not an official warning: always say so when you quote it."""
    from aquascope.archive import warnings

    try:
        return warnings.flood_warnings(bbox, min_rp=min_rp, limit=limit)
    except ValueError as exc:
        return {"error": str(exc)}


def flow_forecast(lat: float | None = None, lon: float | None = None, river_id: int | None = None,
                  station: str | None = None, days: int = 15, quick: bool = False) -> dict[str, Any]:
    """The next 15 days of river flow from two global models, MODELLED: GEOGLOWS v2 (ECMWF 51-member ensemble
    statistics for the river reach: mean, median, 25-75 and min-max bands, high-res run) and GloFAS v4 via
    Open-Meteo (daily ensemble statistics for the 5 km cell), with the reach's 2- to 100-year flows from its
    simulated record since 1940. Give a point (snapped to its reach), a river_id, or station "source/station_id":
    a gauge also gets today's status and the forecast corrected to its own record, with the correction's
    hindcast skill. quick=True returns only the two forecasts, without the thresholds and the correction (it
    skips the 86-year simulated record, the slow read). Say it is a model forecast whenever you quote it."""
    from aquascope import nownext

    try:
        return nownext.now(lat, lon, station=station, river_id=river_id, days=days, history=not quick)
    except ValueError as exc:
        return {"error": str(exc)}


def correct_to_gauge(source: str, station_id: str, river_id: int | None = None, days: int = 15) -> dict[str, Any]:
    """The GEOGLOWS forecast at a gauge's river reach corrected to the gauge's own record (flow-duration quantile
    mapping per calendar month, the MFDC-QM / SABER family), and how much to trust it: KGE (with r, alpha, beta),
    percent bias, and the hit rate and false alarms above the gauge's 2-year flow, raw against corrected, scored
    on the later part of the overlap after fitting on the earlier part. Quote the skill_line with the forecast."""
    from aquascope import nownext

    try:
        res = nownext.now(station=f"{source}/{station_id}", river_id=river_id, days=days)
    except ValueError as exc:
        return {"error": str(exc)}
    fc = res.get("forecast") or {}
    corr = fc.get("correction") or {"error": fc.get("error") or "No forecast to correct at this gauge."}
    return {"station": res.get("station"), "river_id": fc.get("river_id"), "raw": fc.get("geoglows"),
            "correction": corr, "gauge_thresholds": fc.get("gauge_thresholds"), "sentence": fc.get("sentence"),
            "notes": fc.get("notes"), "attribution": fc.get("attribution")}


def watch_digest(items: list[Any], since: str | None = None, thresholds: dict[str, Any] | None = None,
                 forecast: str = "auto") -> dict[str, Any]:
    """What changed at watched places since a date (#521). items: gauges "source/station_id", river reaches
    "river:<id>", areas "area:west,south,east,north" (or dicts with kind and those fields). since: YYYY-MM-DD
    (default a week ago). thresholds: per item id, a value in the record's unit (300) or a return period ("10y");
    without one, forecasts are checked against the 2-year flow. Per item: new days of data and the latest value,
    today's status class against the one on the since date, the forecast peak in the next 15 days against the
    threshold (modelled; corrected to the gauge where possible), and flood events in the news nearby since then;
    one line each and a summary. forecast: auto, archive, live or off."""
    from aquascope import watch

    specs = []
    for it in items or []:
        spec = dict(it) if isinstance(it, dict) else it
        try:
            key = watch.parse_item(spec)["id"]
        except ValueError:
            specs.append(spec)
            continue
        if thresholds and key in thresholds:
            spec = {**(spec if isinstance(spec, dict) else {"id": key}), "threshold": thresholds[key]}
        specs.append(spec)
    try:
        return watch.watch_digest(specs, since, forecast=forecast)
    except ValueError as exc:
        return {"error": str(exc)}


def describe_methods() -> dict[str, Any]:
    """What each analysis computes and the reference to cite."""
    from aquascope.explore import METHODS, MIN_YEARS_FOR_FFA, RETURN_PERIODS

    return {"return_periods": RETURN_PERIODS, "min_years_for_ffa": MIN_YEARS_FOR_FFA, "methods": METHODS}


def dated_layers(live: bool = False) -> dict[str, Any]:
    """The map layers that change with the date (NASA GIBS satellite imagery, IMERG rain, SMAP soil
    moisture, MODIS snow and land temperature, GRACE water storage): their cadence, first and last day,
    licence and tile template. live=True reads the exact intervals, gaps included, from GIBS."""
    from aquascope.map_time import dated_layers as _dated

    return _dated(live=bool(live))


def layer_frames(layer: str, start: str, end: str, step: str = "day", max_frames: int = 60) -> dict[str, Any]:
    """The frames of a time-lapse of one dated map layer: each date from start to end (YYYY-MM-DD) at a
    step of day, week or month, with its XYZ tile URL, skipping dates the layer cannot show (at most 60)."""
    from aquascope.map_time import layer_frames as _frames

    return _frames(layer, start, end, step=step, max_frames=max_frames)


def archive_health() -> dict[str, Any]:
    """Status of the last catalog harvest per source (health.json from the Archive)."""
    import httpx

    from aquascope.archive.catalog import catalog_url

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(catalog_url(filename="health.json"))
        resp.raise_for_status()
        return resp.json()


def describe_catchment(lat: float, lon: float, upstream: bool = True) -> dict[str, Any]:
    """The catchment of a point from BasinATLAS (HydroATLAS v1.0, CC BY 4.0) in the Archive: which
    level-12 sub-basin the point sits in, how many sub-basins drain to it, and area-weighted attributes
    (elevation, slope, precipitation, PET, aridity, temperature, snow, runoff, natural discharge, land
    cover, soils, groundwater table, population, regulation by dams). upstream=False describes only the
    local sub-basin. Works anywhere on land; needs the basins files to be published.
    """
    from aquascope.archive.basins import describe_catchment as _describe

    try:
        return _describe(float(lat), float(lon), upstream=bool(upstream))
    except ImportError as exc:
        return {"error": f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"catchment lookup failed: {type(exc).__name__}: {exc}"}


def place_context(lat: float, lon: float, layers: list[str] | None = None) -> dict[str, Any]:
    """What a hydrologist asks first about a point, from open global datasets, each with its licence: flood
    history (flood events in the news from Google Groundsource, and Sentinel-1 radar flood detections
    2014-2024), surface water since 1984 (JRC Global Surface Water: how often this 30 m pixel was water, and
    the change), modelled flood depth at the 10 to 500-year floods (JRC CEMS-GloFAS hazard maps), dams nearby
    (Global Dam Watch), soil texture and plant-available water (SoilGrids), actual evapotranspiration (FAO
    WaPOR v3) and the nearest real rain gauge with a summary of its record (NOAA GHCN-Daily). layers picks
    some of: flood_history, surface_water, flood_hazard, dams, rain_gauge, actual_et, soil (default all).
    Every layer has a one-line summary; quote the attribution with the numbers.
    """
    from aquascope import context

    try:
        return context.place_context(float(lat), float(lon), layers=layers)
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"place context failed: {type(exc).__name__}: {exc}"}


def area_context(west: float, south: float, east: float, north: float,
                 layers: list[str] | None = None) -> dict[str, Any]:
    """The place-context layers over a box (west, south, east, north in degrees): flood events from the news
    and radar flood months inside it, dams and their combined storage, rain gauges, and surface water, flood
    depth, soil and actual evapotranspiration sampled on a small grid. Keep the box under about 16 x 16
    degrees. Same layer names as place_context.
    """
    from aquascope import context

    try:
        return context.area_context(float(west), float(south), float(east), float(north), layers=layers)
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"area context failed: {type(exc).__name__}: {exc}"}


def similar_basins(
    lat: float | None = None,
    lon: float | None = None,
    source: str | None = None,
    station_id: str | None = None,
    k: int = 10,
    method: str = "combined",
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """The gauged basins in the Archive whose catchments most resemble a point's (or a station's) catchment:
    donor selection for prediction in ungauged basins. Give lat/lon for a point, or source + station_id for a
    station (itself excluded). method: 'similarity' (standardised BasinATLAS attribute space: area, relief,
    climate, land cover, soils, human pressure), 'proximity' (distance on the ground) or 'combined'. Returns
    up to k stations with ids you can pass to analyze_station, the per-feature deltas, and the citation.
    """
    from aquascope.archive.similar import similar_for_point, similar_for_station

    k = max(1, min(int(k or 10), 50))
    try:
        if source and station_id:
            return similar_for_station(source, station_id, k=k, method=method, sources=sources)
        if lat is None or lon is None:
            return {"error": "give lat and lon, or source and station_id"}
        return similar_for_point(float(lat), float(lon), k=k, method=method, sources=sources)
    except ImportError as exc:
        return {"error": f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"similar basins lookup failed: {type(exc).__name__}: {exc}"}


def regionalize_signatures(lat: float, lon: float, k: int = 10, method: str = "similarity") -> dict[str, Any]:
    """Estimated flow regime of an UNGAUGED point, transferred from the gauged donors in the Archive: mean, median,
    Q95 (low) and Q05 (high) daily flow in mm/d, mean annual maximum, runoff ratio, baseflow index, FDC slope,
    high/low-flow frequency, zero-flow fraction, seasonality and flashiness, each with an uncertainty band and
    the donors used. method: 'similarity' (weighted mean over the k most similar catchments), 'regression'
    (ridge on catchment attributes over all donors) or 'both'. Comes with the leave-one-out skill (NSE, median
    error) of each estimate so you can say how much to trust it. Prediction in ungauged basins (PUB).
    """
    from aquascope.archive.regionalize import regionalize_point

    k = max(1, min(int(k or 10), 50))
    try:
        return regionalize_point(float(lat), float(lon), k=k, method=method)
    except ImportError as exc:
        return {"error": f"{exc}"}
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"regionalisation failed: {type(exc).__name__}: {exc}"}


def drought_indices(
    lat: float,
    lon: float,
    years: int = 40,
    timescales: list[int] | None = None,
    source: str | None = None,
    station_id: str | None = None,
    pet: str = "thornthwaite",
    threshold: float = -1.0,
) -> dict[str, Any]:
    """Drought status at a place: SPI and SPEI at several timescales (default 1, 3 and 12 months) with the
    divergence between them. Give source + station_id for a rain gauge (its whole record is the P of both
    indices, ERA5 supplies the PET); without one, ERA5 precipitation for the cell over the last `years`. pet:
    thornthwaite (from ERA5 temperature, the PET SPEI was introduced with), fao56 (ERA5 FAO-56 ET0) or none
    (SPI only). Returns current values and classes, the worst month, drought events, the ERA5 temperature
    trend, the thinned series and the citations. SPEI is preferable under warming; a record shorter than 30
    years is marginal (20 is the floor). threshold is the index value at or below which a month counts as
    drought (-1 by default, McKee et al. 1993).
    """
    from aquascope.problems import drought_indices as _run

    try:
        return _run(float(lat), float(lon), years=int(years), timescales=timescales or (1, 3, 12),
                    source=source or None, station_id=station_id or None, pet=pet or "thornthwaite",
                    threshold=float(threshold) if threshold is not None else -1.0)
    except Exception as exc:  # noqa: BLE001 - the model gets to see it
        return {"error": f"drought_indices failed: {type(exc).__name__}: {exc}"}


def drought_propagation(
    source: str,
    station_id: str,
    lat: float,
    lon: float,
    years: int | None = None,
    max_lag: int = 24,
) -> dict[str, Any]:
    """Groundwater drought at a well and how rainfall deficits reach it: the Standardised Groundwater Index
    (current, worst, events) and the SPI accumulation period (1 to 24 months, on ERA5 precipitation for the
    cell) and lag (0 to max_lag months) whose cross-correlation with the SGI is highest (Bloomfield and
    Marchant 2013). Ten years of monthly levels is the registry's floor for the SGI.
    """
    from aquascope.problems import drought_propagation as _run

    try:
        return _run(source, station_id, float(lat), float(lon), years=int(years) if years else None,
                    max_lag=int(max_lag))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"drought_propagation failed: {type(exc).__name__}: {exc}"}


def low_flow_context(source: str, station_id: str, years: int | None = None) -> dict[str, Any]:
    """How low is low at a gauge, and is the river low now: Q95, Q50, Q10 (and Q05, Q25, Q75, Q90), the baseflow
    index (Lyne-Hollick), the 7Q10 low-flow statistic when the record has ten years, and the last 30 and 90
    days' mean flow with the share of the record that exceeds it.
    """
    from aquascope.problems import low_flow_context as _run

    try:
        return _run(source, station_id, years=int(years) if years else None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"low_flow_context failed: {type(exc).__name__}: {exc}"}


def supply_reliability(
    demand_m3s: float | None = None,
    demand_ml_day: float | None = None,
    source: str | None = None,
    station_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    share: float = 0.1,
    reserve: str = "q95",
    months: list[int] | None = None,
) -> dict[str, Any]:
    """Can a river supply a demand, as a run-of-river screening. demand in m3/s or ML/day. On any day the
    abstraction may take at most `share` of the flow and must leave `reserve` in the river (q95 by default, a
    number in m3/s, or none). Gauged (source + station_id): the fraction of days, of years without a shortfall
    and of the volume the record would have supplied, over the year or over `months`; also Q95/Q50/Q10, the
    baseflow index and 7Q10. Ungauged (lat + lon): the reliability read off Q95, median and Q05 transferred
    from donor catchments, as a band with the leave-one-out skill. A screening rule (flow-duration-curve
    environmental-flow practice), not a storage-yield analysis.
    """
    from aquascope.problems import supply_reliability as _run

    try:
        return _run(demand_m3s=demand_m3s, demand_ml_day=demand_ml_day, source=source or None,
                    station_id=station_id or None, lat=lat, lon=lon, share=float(share), reserve=reserve,
                    months=months or None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"supply_reliability failed: {type(exc).__name__}: {exc}"}


def crop_water_demand(
    lat: float,
    lon: float,
    crop: str,
    area_ha: float,
    planting_month: int,
    efficiency: float = 0.7,
    years: int = 10,
) -> dict[str, Any]:
    """A crop's seasonal irrigation demand at a point: FAO-56 single Kc (Table 12 crops: maize, wheat_winter,
    rice_paddy, ...) on ERA5 FAO-56 reference ET0, effective rainfall subtracted, divided by the irrigation
    efficiency; the season from the first of planting_month is run for every year of the ERA5 window and
    averaged (range kept). Returns the depth in mm, the volume in m3 over area_ha, the mean and peak-month
    rates in m3/s, and the season's months for a supply check. Supply is not checked here.
    """
    from aquascope.problems import crop_water_demand as _run

    try:
        return _run(float(lat), float(lon), crop=crop, area_ha=float(area_ha), planting_month=int(planting_month),
                    efficiency=float(efficiency), years=int(years))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"crop_water_demand failed: {type(exc).__name__}: {exc}"}


def analyse_table(
    csv: str,
    analysis: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one of the workbench analyses on a table of *your own* data (CSV text).

    The analyses are the ones the dashboard pages offer, and they are the same
    code the Explorer runs in the browser: eda, quality, preprocess, insights,
    who_screen, wqi (CCME WQI 1.0 against WHO drinking-water, FAO 29 irrigation or
    CCME aquatic-life guidelines, plus the NSF WQI; params use, variant,
    guidelines), iwqi (FAO 29 irrigation suitability), flow_duration, baseflow,
    recession, flood_frequency, signatures, return_periods, sgi_drought,
    recharge, aquifer_drawdown.

    Pass the data as CSV text (a header row and one row per observation) and the
    parameters of the analysis as a dict, for example
    ``{"method": "eckhardt", "alpha": 0.98}``.
    """
    from io import StringIO

    import pandas as pd

    from aquascope import workbench

    if analysis not in workbench.TOOLS:
        return {"error": f"Unknown analysis {analysis!r}", "available": sorted(workbench.TOOLS)}
    spec = workbench.TOOLS[analysis]
    kwargs = dict(params or {})
    if spec["needs"] == "none":
        return workbench.run(analysis, **kwargs)
    if not csv or not csv.strip():
        return {"error": f"{analysis} needs a table; pass the data as CSV text."}
    df = pd.read_csv(StringIO(csv))
    result = workbench.run(analysis, df, **kwargs)
    result.pop("frame", None)          # the cleaned frame is not JSON
    return result


def list_analyses() -> dict[str, Any]:
    """Every workbench analysis with what it needs and what it is for."""
    from aquascope import workbench

    return {
        "analyses": [
            {"name": name, "needs": spec["needs"], "summary": spec["summary"]}
            for name, spec in workbench.TOOLS.items()
        ],
        "note": "Run one with analyse_table(csv, analysis, params). These are the dashboard's analyses, "
                "and the same code the Explorer runs in the browser.",
    }


# ── Solve: playbooks, a plan to review, a study to run (#307, #308) ─────────


def list_playbooks() -> dict[str, Any]:
    """The problem playbooks: for each class of problem (flood risk, ungauged flow, groundwater decline, drought
    status, supply reliability, irrigation feasibility, water quality), the method chain aquascope follows for
    the data that exists at a site, as data. Each has intake fields, branches over the reconnaissance, gates per
    step, the sentences it prints when it declines, caveats and citations.
    Use solve_plan to get the study a playbook fills for a problem at a point, and solve_run to execute it.
    """
    from aquascope import playbooks as pbk

    rows = pbk.list_playbooks()
    return {"n": len(rows), "playbooks": rows,
            "note": "describe_playbook(id) shows the whole tree; solve_plan(problem, lat, lon) fills it."}


def describe_playbook(playbook: str) -> dict[str, Any]:
    """One playbook in full: intake fields, branches with their conditions and steps (tool, arguments with
    placeholders, gates, fallback), decline rules, caveats and citations."""
    from aquascope import playbooks as pbk

    try:
        return pbk.describe(playbook)
    except pbk.PlaybookError as exc:
        return {"error": str(exc), "known": [p["id"] for p in pbk.list_playbooks()]}


def solve_plan(
    problem: str, lat: float, lon: float, playbook: str | None = None, intake: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Plan (do not run) a problem at a point: reconnaissance of the site (assess_site), the playbook the
    keyword rules pick (or the one named), the branch its tree selects for the data that exists, and the
    study-v2 it fills: steps with arguments, rationale and gates. Zero model calls. Review the study, edit it
    if you like, then pass it to solve_run. `declined` with a reason means the playbook refuses this ask
    (record too short for the return period, cause attribution without pumping data, out of scope).
    intake: the playbook's intake fields, for example {"return_period": 100}.
    """
    from aquascope.ai_engine.team import _recon_summary, solve

    res = solve(problem, lat=float(lat), lon=float(lon), playbook=playbook, intake=intake, execute=False)
    plan = res.study.plan or {}
    return {
        "declined": res.declined,
        "reason": res.declined_reason,
        "playbook": plan.get("playbook"),
        "branch": plan.get("branch"),
        "rationale": plan.get("rationale"),
        "n_steps": len(res.study.steps),
        "study": res.study.to_dict(),
        "study_yaml": res.study_yaml,
        "recon": _recon_summary(res.recon),
        "timeline": res.timeline,
        "note": "Review the study (edit arguments or drop steps), then solve_run(study) executes it with its gates.",
    }


def solve_run(study: dict[str, Any] | str) -> dict[str, Any]:
    """Execute a study (the dict from solve_plan, or study YAML text) with no model in the loop: every step
    runs in order, its gates are evaluated, a failed gate runs the step's fallback once or stops the study
    with the reason. Returns the gate outcomes, the report and the study with its results written in, which
    `aquascope run` reproduces.
    """
    from aquascope.study import Study, loads, run_study

    try:
        st = loads(study) if isinstance(study, str) else Study.from_dict(dict(study))
    except (ValueError, TypeError) as exc:
        return {"error": f"could not read the study: {exc}"}
    if not st.steps:
        return {"error": "the study has no steps"}
    # The team's execute-and-report tail, keyless: the same gates, Reviewer
    # list and template prose the Explorer's Solve surface shows.
    from aquascope.ai_engine.team import run_reviewed

    result = run_reviewed(st)
    run = result.run if result.run is not None else run_study(st)
    return {
        "ok": run.ok,
        "stopped_at": run.stopped_at,
        "stop_reason": run.stop_reason,
        "gates": run.gates,
        "answer": result.answer,
        "not_established": result.not_established,
        "caveats": result.caveats,
        "report": result.to_markdown(),
        "manifest": run.manifest(),
        "study": st.to_dict(),
        "study_yaml": st.to_yaml(),
    }


# ── inline views (MCP Apps) ─────────────────────────────────────────────────
# A client that supports the MCP Apps extension (SEP-1865, in the 2026-07 spec)
# can render HTML a server returns, inline in the conversation. A hydrograph is
# worth more than a page of JSON, so analyze_station can hand one back. Clients
# without the extension are unaffected: they get the JSON they always got.

_WIDGET_CSS = (
    "body{margin:0;font:13px/1.5 system-ui,sans-serif;color:#1f2933}"
    ".k{display:flex;gap:.6rem;flex-wrap:wrap;margin:.4rem 0}"
    ".k div{border:1px solid #e3e8ee;border-radius:8px;padding:.3rem .5rem}"
    ".k b{display:block;font-size:1rem}"
    "svg{width:100%;height:120px}"
    ".m{color:#6b7785;font-size:11px;line-height:1.45}"
)


# ── Studio over MCP: a crew of roles does a complete study at a place; the workspace dict in and out ──


def _studio(workspace: dict[str, Any] | None, *, lat: float | None = None, lon: float | None = None,
            intake: dict[str, Any] | None = None, provider: str | None = None, model: str | None = None,
            api_key: str | None = None, base_url: str | None = None, max_usd: float | None = None,
            tables: dict[str, str] | None = None) -> Any:
    from aquascope.studio import Studio

    return Studio(lat, lon, provider=provider, model=model, api_key=api_key, base_url=base_url,
                  workspace=workspace, intake=intake, max_usd=max_usd,
                  data={str(k): str(v) for k, v in (tables or {}).items()} if workspace is None and tables else None)


def _with_tables(studio: Any, tables: dict[str, str] | None) -> Any:
    """Tables handed to a stateless studio_* call (name to CSV text) go in through Studio.add_table, which plans
    again or re-runs as the status requires; the last reply is returned, or None when no table was given."""
    reply = None
    for name, csv in (tables or {}).items():
        reply = studio.add_table(str(name), str(csv))
    return reply


def _studio_reply(studio: Any, reply: Any) -> dict[str, Any]:
    from aquascope.study_map import workspace_features

    return {"reply": reply.to_dict(), "status": studio.workspace.status, "summary": studio.workspace.summary(),
            "workspace": studio.to_dict(), "map": workspace_features(studio.workspace)}


def studio_start(
    problem: str, lat: float, lon: float, intake: dict[str, Any] | None = None, provider: str | None = None,
    model: str | None = None, api_key: str | None = None, base_url: str | None = None,
    max_usd: float | None = None, tables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Open a study at a point with the Studio crew (Consultant, Scout, Methodologist, Analysts, Critic, Author).
    The Consultant takes the brief from the problem text; the reply is either `questions` (answer them with
    studio_say, or say "just go") or, when nothing needs asking, the `plan` to review (approve it with
    studio_approve) or `declined` with the reason. Keyless unless a provider/model is given. Keep the returned
    `workspace` dict and pass it to the next studio_* call: the tools are stateless.
    intake: playbook intake fields, for example {"return_period": 100}.
    """
    try:
        studio = _studio(None, lat=float(lat), lon=float(lon), intake=intake, provider=provider, model=model,
                         api_key=api_key, base_url=base_url, max_usd=max_usd, tables=tables)
        return _studio_reply(studio, studio.say(problem))
    except Exception as exc:  # noqa: BLE001 - an MCP tool answers with an error, not a traceback
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_say(
    workspace: dict[str, Any], text: str, proposed: dict[str, Any] | None = None, provider: str | None = None,
    model: str | None = None, api_key: str | None = None, base_url: str | None = None,
    max_usd: float | None = None, tables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Continue the conversation with the Studio crew: answer the Consultant's questions (in order, or "just go"
    for the defaults), change the brief at review, or ask a follow-up after the report. Returns the next reply
    (questions, plan, report, answer or declined) and the updated workspace to pass on.
    proposed: a brief a model of your own wrote from the text, {"brief": {decision, quantities, period, horizon,
    constraints, kind, playbook, intake, assumptions, questions}, "source": "device"}; it is merged with the
    coercion a model reply gets (studio_context with role "consultant" gives the prompt and the context).
    tables: your own tables as {name: CSV text}; while the crew waits for data (reply kind `data_request`) or at
    review they are inventoried and the plan is written again, after the report they run as a follow-up; say
    "continue without" at a data_request to go on at the lower grade it names.
    """
    try:
        studio = _studio(workspace, provider=provider, model=model, api_key=api_key, base_url=base_url,
                         max_usd=max_usd)
        added = _with_tables(studio, tables)
        if added is not None and not (text or "").strip():
            return _studio_reply(studio, added)
        return _studio_reply(studio, studio.say(text, proposed=proposed))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_approve(
    workspace: dict[str, Any], edits: dict[str, Any] | None = None, plan: dict[str, Any] | None = None,
    provider: str | None = None, model: str | None = None, api_key: str | None = None,
    base_url: str | None = None, max_usd: float | None = None,
) -> dict[str, Any]:
    """Approve the plan in the workspace (optionally with edits: {"s3": {"arguments": {"k": 8}}} or a
    replacement {"steps": [...]}, revalidated) and run the crew to the report: the Analysts with their gates,
    the Critic, the Author. The reply is `report` (answer, key numbers, sections, what is not established)
    with the artifacts listed; the workspace carries their bytes. max_usd: a spend ceiling for the model
    calls (priced models only); past it the roles run keyless and the footer says so.
    plan: a plan a model of your own wrote in the Methodologist's reply shape ({objective, decision, methodology,
    steps: [{id, tool, arguments, rationale, method, expects, fallback, depends_on, outputs}], assumptions,
    alternatives, limitations_expected, citations, source}); it goes through the validator like a model plan
    (repair, pruning, the playbook tree when nothing valid remains) and the reply's payload says `plan_used`
    (proposed or tree) and `plan_errors`.
    """
    try:
        studio = _studio(workspace, provider=provider, model=model, api_key=api_key, base_url=base_url,
                         max_usd=max_usd)
        return _studio_reply(studio, studio.approve(edits=edits, plan=plan))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_narrate(workspace: dict[str, Any], sections: dict[str, str] | list[dict[str, Any]],
                   source: str = "device") -> dict[str, Any]:
    """After the report: replace the prose of the named sections with text a model of your own wrote
    (sections: {section id: text} or [{id, text}]; ids as in the report's sections plus "answer" and
    "recommendations"). Every sentence passes the Critic's check first: one whose numbers are in no tool
    result is dropped and counted (`dropped` in the payload). The report says who wrote which section
    (`written_by`), the deliverables are rebuilt, the reply is the report.
    """
    try:
        studio = _studio(workspace)
        return _studio_reply(studio, studio.narrate(sections, source=source))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_context(workspace: dict[str, Any], role: str, text: str | None = None) -> dict[str, Any]:
    """The prompt and the compact context one role of the crew would send a model at this point, so any model
    can run it and hand the reply back: role "consultant" (text: the client's message; reply -> studio_say
    proposed), "methodologist" (text: a change request after the report, else the plan; reply -> studio_approve
    plan), "author" (reply's sections -> studio_narrate). The context carries the system prompt under
    `system`. Nothing runs and the workspace is not changed.
    """
    try:
        studio = _studio(workspace)
        if role == "consultant":
            context = studio.consultant_context(text or "")
        elif role == "methodologist":
            context = studio.methodologist_context(text)
        elif role == "author":
            context = studio.author_context()
        else:
            return {"error": f"unknown role {role!r}; one of consultant, methodologist, author"}
        return {"role": role, "context": context, "status": studio.workspace.status}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_follow_up(
    workspace: dict[str, Any], text: str, provider: str | None = None, model: str | None = None,
    api_key: str | None = None, base_url: str | None = None, max_usd: float | None = None,
) -> dict[str, Any]:
    """A follow-up after the report: a question is answered from the workspace (reply `answer`); a change
    (another return period, another statistic) is planned, run and re-authored (reply `report`).
    """
    try:
        studio = _studio(workspace, provider=provider, model=model, api_key=api_key, base_url=base_url,
                         max_usd=max_usd)
        return _studio_reply(studio, studio.follow_up(text))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def studio_export(workspace: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """Write the study's bundle (report, study.yaml, workspace.json, figures and documents when made) into
    out_dir; returns the paths by file name.
    """
    try:
        studio = _studio(workspace)
        return {"paths": studio.export(out_dir), "out_dir": str(out_dir)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


MAX_EXPORT_CHARS = 60_000


def engineering_export(
    source: str, station_id: str, tool: str, years: int | None = None, variable: str | None = None,
    regional_skew: float | None = None, regional_skew_mse: float | None = None, out_dir: str | None = None,
    max_chars: int = MAX_EXPORT_CHARS,
) -> dict[str, Any]:
    """Inputs for an engineering tool from a gauge's record: tool is one of hec-hms, hec-ras, hec-ssp, dss,
    swmm, modflow6, fews, raven (or "all"). Returns each file's text (a long file is cut at max_chars and marked
    truncated) and the notes to read before using it; pass out_dir to also write the files there. hec-ssp adds
    the Bulletin 17C settings and AquaScope's own result to compare; regional_skew weights its skew. DSS is a
    real .dss where HEC's hecdss loads, else the CSV hecdss reads.
    """
    from aquascope.io import engineering as eng

    if source not in SOURCES:
        return {"error": f"unknown source {source!r}"}
    if tool != "all" and tool not in eng.TOOLS:
        return {"error": f"unknown tool {tool!r}; choose from {', '.join(eng.TOOLS)} or all"}
    try:
        res = eng.export_station(source, station_id, tool, years=int(years) if years else None, variable=variable,
                                 regional_skew=regional_skew, regional_skew_mse=regional_skew_mse)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if "error" in res:
        return res
    if out_dir:
        res["written"] = eng.write_files(eng.files_from(res), out_dir)
    for f in res["files"]:
        f.pop("base64", None)  # a binary .dss stays on disk (out_dir), never in the reply
        text = f.get("text")
        if isinstance(text, str) and len(text) > max_chars:
            f["text"] = text[:max_chars]
            f["truncated"] = True
    meta = SOURCES[source]
    res.update({"license": meta.license, "attribution": meta.attribution})
    return res


def _sparkline(values: list[float], width: int = 560, height: int = 120) -> str:
    """A dependency-free hydrograph: the shape of a record, in an inline SVG."""
    clean = [v for v in values if isinstance(v, (int, float))]
    if len(clean) < 2:
        return ""
    lo, hi = min(clean), max(clean)
    span = (hi - lo) or 1.0
    step = max(1, len(clean) // width)
    pts = clean[::step]
    dx = width / max(len(pts) - 1, 1)
    coords = " ".join(
        f"{i * dx:.1f},{height - (v - lo) / span * (height - 8) - 4:.1f}" for i, v in enumerate(pts)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        f'aria-label="hydrograph"><polyline fill="none" stroke="#1565c0" stroke-width="1.2" '
        f'points="{coords}"/></svg>'
    )


def station_view(source: str, station_id: str, years: int | None = None) -> dict[str, Any]:
    """analyze_station, plus a small HTML view of it for clients that render one.

    The ``_meta`` block is what an MCP Apps client looks for; everything else is
    the ordinary tool result, so a client that ignores views loses nothing.
    """
    import html as _html

    result = analyze_station(source, station_id, years=years)
    if result.get("error"):
        return result
    stats = result.get("stats") or {}
    series = (result.get("series") or {}).get("v") or []
    ffa = ((result.get("ffa") or {}).get("fits") or {}).get("gev_lmoments") or {}
    rp = (result.get("ffa") or {}).get("return_periods") or []
    q100 = ""
    if ffa.get("q") and 100 in rp:
        q100 = f"<div>100-yr flood<b>{ffa['q'][rp.index(100)]:.4g} {_html.escape(result.get('unit') or '')}</b></div>"
    body = (
        f"<h3 style='margin:.2rem 0'>{_html.escape(str(result.get('name') or station_id))}</h3>"
        f"<div class='m'>{_html.escape(source)} / {_html.escape(station_id)} · "
        f"{_html.escape(str(result.get('start')))} to {_html.escape(str(result.get('end')))}</div>"
        f"<div class='k'>"
        f"<div>mean<b>{(stats.get('mean') or 0):.4g} {_html.escape(result.get('unit') or '')}</b></div>"
        f"<div>max<b>{(stats.get('max') or 0):.4g}</b></div>"
        f"<div>years<b>{result.get('years')}</b></div>{q100}</div>"
        f"{_sparkline(series)}"
        f"<div class='m'>Data: {_html.escape(str(result.get('attribution') or ''))} "
        f"({_html.escape(str(result.get('license') or ''))}). Computed with aquascope.</div>"
    )
    result["_meta"] = {
        "openai/outputTemplate": "text/html+skybridge",
        "mcp/view": {
            "mimeType": "text/html",
            "html": f"<style>{_WIDGET_CSS}</style><main>{body}</main>",
        },
    }
    return result



# ── server wiring ──────────────────────────────────────────────────────────


def build_server():
    """Create the MCP server with all tools and resources registered."""
    server = _server()
    server.tool()(list_sources)
    server.tool()(find_stations)
    server.tool()(get_timeseries)
    server.tool()(water_quality_samples)
    server.tool()(analyze_station)
    server.tool()(flood_frequency)
    server.tool()(describe_methods)
    server.tool()(assess_site)
    server.tool()(study_area)
    server.tool()(describe_catchment)
    server.tool()(snap_to_river)
    server.tool()(reach_record)
    server.tool()(upstream_area)
    server.tool()(trace_downstream)
    server.tool()(upstream_dams)
    server.tool()(model_skill)
    server.tool()(model_to_lean_on)
    server.tool()(flow_status)
    server.tool()(flow_forecast)
    server.tool()(status_bulletin)
    server.tool()(flood_warnings)
    server.tool()(correct_to_gauge)
    server.tool()(watch_digest)
    server.tool()(place_context)
    server.tool()(area_context)
    server.tool()(similar_basins)
    server.tool()(regionalize_signatures)
    from aquascope.archive.signatures import filter_gauges  # the map's signature filter (signatures.parquet)

    server.tool()(filter_gauges)
    server.tool()(drought_indices)
    server.tool()(drought_propagation)
    server.tool()(low_flow_context)
    server.tool()(supply_reliability)
    server.tool()(crop_water_demand)
    # the advanced study steps: change, nonstationary floods, catchment model, projections, regions
    from aquascope import advanced

    for fn in (advanced.change_points, advanced.nonstationary_flood, advanced.pot_flood, advanced.catchment_model,
               advanced.climate_projection, advanced.regional_flood, advanced.compare_gauges):
        server.tool()(fn)
    server.tool()(archive_health)
    server.tool()(dated_layers)
    server.tool()(layer_frames)
    server.tool()(list_analyses)
    server.tool()(analyse_table)
    server.tool()(station_view)
    server.tool()(engineering_export)
    server.tool()(list_playbooks)
    server.tool()(describe_playbook)
    server.tool()(solve_plan)
    server.tool()(solve_run)
    server.tool()(studio_start)
    server.tool()(studio_say)
    server.tool()(studio_approve)
    server.tool()(studio_follow_up)
    server.tool()(studio_narrate)
    server.tool()(studio_context)
    server.tool()(studio_export)

    @server.resource("aquascope://sources")
    def sources_resource() -> str:
        """The source registry as JSON."""
        return json.dumps(list_sources(), ensure_ascii=False)

    @server.resource("aquascope://methods")
    def methods_resource() -> str:
        """Analysis methods and citations as JSON."""
        return json.dumps(describe_methods(), ensure_ascii=False)

    return server


def main(transport: str = "stdio") -> None:
    """Entry point for ``aquascope mcp``."""
    logging.basicConfig(level=logging.WARNING)
    build_server().run(transport)


if __name__ == "__main__":  # pragma: no cover
    main()
