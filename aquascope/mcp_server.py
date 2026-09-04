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
    "Taiwan CWA and more) behind one schema. Start with find_stations (no agency call), then get_timeseries or "
    "analyze_station for a specific station. For a place or a station, assess_site(lat, lon) first says which "
    "methods the record there supports; do not run one it marks not_defensible. Flood frequency needs at least "
    "10 complete years of daily flow. Always show the licence/attribution returned with the data."
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
    stations with ids you can pass to get_timeseries / analyze_station.
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
    )
    slim = [
        {k: r.get(k) for k in ("source", "station_id", "name", "latitude", "longitude", "variables",
                               "period_start", "period_end", "river", "country", "agency", "license", "url")}
        for r in hits
    ]
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
    source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False, variable: str | None = None
) -> dict[str, Any]:
    """Fetch and analyse one station: record summary, annual maxima, flood frequency (GEV L-moments and
    Log-Pearson III with 90 % CI; optional bootstrap GEV band), flow-duration percentiles, Mann-Kendall
    trend, and the method citations. Raw daily arrays are omitted; use get_timeseries for those.
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
    res = _analyze(source, station_id, years=int(years) if years else None, store=store, variable=variable)
    res.pop("series", None)
    if "fdc" in res:
        res["fdc"] = {k: res["fdc"][k] for k in ("q95", "q50", "q10")}
    if bootstrap_ci and res.get("ffa") and store.get("series") is not None:
        try:
            ci = flood_ci(store["series"])
            res["ffa"]["fits"]["gev_bootstrap"] = {
                k: ci[k] for k in ("q", "ci", "params", "n_bootstrap", "n_bootstrap_discarded") if k in ci
            }
            res.setdefault("methods", []).append(ci["method"])
        except Exception as exc:  # noqa: BLE001
            res.setdefault("notes", []).append(f"bootstrap CI failed: {exc}")
    return res


def flood_frequency(
    source: str, station_id: str, years: int | None = None, bootstrap_ci: bool = False
) -> dict[str, Any]:
    """Return levels for T = 2, 5, 10, 25, 50, 100 years at a station (subset of analyze_station).
    years caps the record to the last N years; by default the full record is requested.
    """
    res = analyze_station(source, station_id, years=years, bootstrap_ci=bootstrap_ci)
    if "error" in res:
        return res
    keep = {k: res.get(k) for k in ("source", "station_id", "agency", "license", "attribution", "unit",
                                    "start", "end", "years", "n", "ffa", "notes", "methods",
                                    "fetch_note", "requested")}
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


def describe_methods() -> dict[str, Any]:
    """What each analysis computes and the reference to cite."""
    from aquascope.explore import METHODS, MIN_YEARS_FOR_FFA, RETURN_PERIODS

    return {"return_periods": RETURN_PERIODS, "min_years_for_ffa": MIN_YEARS_FOR_FFA, "methods": METHODS}


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
) -> dict[str, Any]:
    """Drought status at a place: SPI and SPEI at several timescales (default 1, 3 and 12 months) with the
    divergence between them. Give source + station_id for a rain gauge (its whole record is the P of both
    indices, ERA5 supplies the PET); without one, ERA5 precipitation for the cell over the last `years`. pet:
    thornthwaite (from ERA5 temperature, the PET SPEI was introduced with), fao56 (ERA5 FAO-56 ET0) or none
    (SPI only). Returns current values and classes, the worst month, drought events, the ERA5 temperature
    trend, the thinned series and the citations. SPEI is preferable under warming; a record shorter than 30
    years is marginal (20 is the floor).
    """
    from aquascope.problems import drought_indices as _run

    try:
        return _run(float(lat), float(lon), years=int(years), timescales=timescales or (1, 3, 12),
                    source=source or None, station_id=station_id or None, pet=pet or "thornthwaite")
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
    server.tool()(describe_catchment)
    server.tool()(similar_basins)
    server.tool()(regionalize_signatures)
    server.tool()(drought_indices)
    server.tool()(drought_propagation)
    server.tool()(low_flow_context)
    server.tool()(supply_reliability)
    server.tool()(crop_water_demand)
    server.tool()(archive_health)
    server.tool()(list_analyses)
    server.tool()(analyse_table)
    server.tool()(station_view)
    server.tool()(list_playbooks)
    server.tool()(describe_playbook)
    server.tool()(solve_plan)
    server.tool()(solve_run)

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
