"""What each method needs from the data before its answer is worth quoting.

This is the sufficiency registry the reconnaissance step (``assess_site``),
the playbooks and the plan-first Analyst all read. It is data, not code: one
entry per method with the variable it needs, the record length below which
it is marginal or not defensible, and the other conditions practice attaches
to it (a catchment-size ceiling for a lumped model, a return-period cap
relative to record length, donors for a transfer). The thresholds are the
ones applied hydrology uses and the citations say where they come from; a
gate in a study (``aquascope run``) quotes the same numbers.

Nothing here fetches data. Callers describe what exists at a site as a
:class:`SiteContext` and ask :func:`assess_method` (one method) or
:func:`sufficiency_table` (all of them) what that record supports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "METHODS",
    "MethodPrecondition",
    "SiteContext",
    "assess_method",
    "describe_preconditions",
    "method_ids",
    "sufficiency_table",
]

DEFENSIBLE = "defensible"
MARGINAL = "marginal"
NOT_DEFENSIBLE = "not_defensible"


@dataclass(frozen=True)
class MethodPrecondition:
    """One method and what it needs. ``tool`` is the engine function that runs it."""

    id: str
    label: str
    #: The variable the method consumes (``discharge``, ``precipitation``,
    #: ``groundwater_level``, ``water_level``), or None when it needs no record.
    variable: str | None
    #: Record length (years) at or above which the method is defensible, and the
    #: floor below which it is not; between the two it is marginal.
    min_years: float | None = None
    marginal_years: float | None = None
    #: ``daily`` or ``monthly``: the coarsest resolution the method accepts.
    resolution: str | None = None
    #: For methods that exist because there is no gauge.
    ungauged: bool = False
    #: A lumped model's honest ceiling (km²), from #273.
    max_area_km2: float | None = None
    #: The return period may not exceed this many times the record length.
    max_return_period_factor: float | None = None
    #: Donor gauges a transfer needs.
    min_donors: int | None = None
    #: Other inputs the site context must report as available.
    needs: tuple[str, ...] = ()
    tool: str | None = None
    problems: tuple[str, ...] = ()
    citation: str | None = None
    note: str | None = None


@dataclass
class SiteContext:
    """What exists at a site, as the reconnaissance step reports it."""

    #: Variables with a usable record, mapped to their record length in years.
    years_by_variable: dict[str, float] = field(default_factory=dict)
    #: Coarsest resolution available per variable (``daily`` / ``monthly``).
    resolution_by_variable: dict[str, str] = field(default_factory=dict)
    #: Upstream catchment area at the site, when known.
    area_km2: float | None = None
    #: The return period the question asks for, when it asks for one.
    return_period: float | None = None
    #: Donor gauges a similarity search can offer.
    donors: int | None = None
    #: Other inputs present: ``temperature``, ``forcing``, ``gcms>=3``, ``glofas``, ...
    available: set[str] = field(default_factory=set)

    @property
    def ungauged(self) -> bool:
        return not self.years_by_variable


_RES_RANK = {"daily": 2, "monthly": 1}


METHODS: dict[str, MethodPrecondition] = {
    m.id: m
    for m in [
        MethodPrecondition(
            "at_site_flood_frequency",
            "At-site flood frequency (GEV, LP3 / Bulletin 17C)",
            "discharge",
            min_years=20,
            marginal_years=10,
            resolution="daily",
            max_return_period_factor=3,
            tool="analyze_station",
            problems=("flood_risk", "climate_change"),
            citation="England et al. (2019) Bulletin 17C; Hosking (1990) L-moments",
            note="Rare quantiles move with the distribution and the estimator; quote the spread across fits.",
        ),
        MethodPrecondition(
            "flow_duration",
            "Flow duration curve and Q95",
            "discharge",
            min_years=5,
            marginal_years=2,
            resolution="daily",
            tool="analyze_station",
            problems=("supply_reliability", "ungauged_flow", "drought", "irrigation"),
            citation="Vogel & Fennessey (1994)",
        ),
        MethodPrecondition(
            "supply_reliability",
            "Run-of-river supply reliability against a demand (flow-duration screening)",
            "discharge",
            min_years=10,
            marginal_years=5,
            resolution="daily",
            tool="supply_reliability",
            problems=("supply_reliability", "irrigation"),
            citation="Vogel & Fennessey (1994); Smakhtin & Eriyagama (2008) FDC-shift environmental flows",
            note="A screening rule: the fraction of days the flow exceeds the demand plus an environmental reserve "
            "(Q95 by default) under an abstraction share. Storage and yield are a separate analysis.",
        ),
        MethodPrecondition(
            "low_flow_frequency",
            "Low-flow frequency (annual minima)",
            "discharge",
            min_years=10,
            marginal_years=5,
            resolution="daily",
            tool="analyze_station",
            problems=("supply_reliability", "drought"),
        ),
        MethodPrecondition(
            "baseflow_separation",
            "Baseflow separation and baseflow index",
            "discharge",
            min_years=3,
            marginal_years=1,
            resolution="daily",
            tool="baseflow",
            problems=("supply_reliability", "drought", "groundwater_decline"),
            citation="Lyne & Hollick (1979); Eckhardt (2005)",
            note="Filter choice moves the index; run two filters and report the spread.",
        ),
        MethodPrecondition(
            "trend_mann_kendall",
            "Mann-Kendall trend with Sen's slope",
            None,
            min_years=20,
            marginal_years=10,
            resolution="monthly",
            tool="analyze_station",
            problems=("flood_risk", "drought", "groundwater_decline", "climate_change"),
            citation="Mann (1945); Kendall (1975); Sen (1968)",
            note="Says whether something is changing, never why.",
        ),
        MethodPrecondition(
            "spi",
            "Standardized Precipitation Index",
            "precipitation",
            min_years=30,
            marginal_years=20,
            resolution="monthly",
            tool="drought_indices",
            problems=("drought",),
            citation="McKee et al. (1993); WMO (2012)",
        ),
        MethodPrecondition(
            "spei",
            "Standardized Precipitation-Evapotranspiration Index",
            "precipitation",
            min_years=30,
            marginal_years=20,
            resolution="monthly",
            needs=("temperature",),
            tool="drought_indices",
            problems=("drought",),
            citation="Vicente-Serrano et al. (2010)",
            note="Preferred over SPI where warming matters; needs a temperature or PET series.",
        ),
        MethodPrecondition(
            "spei_reanalysis",
            "SPI and SPEI from ERA5 reanalysis for the cell (no rain gauge)",
            None,
            needs=("forcing",),
            tool="drought_indices",
            problems=("drought",),
            citation="Vicente-Serrano et al. (2010); Hersbach et al. (2020) ERA5",
            note="A 9 km cell's climate since 1940, not a gauge; the indices describe the area, not a point record.",
        ),
        MethodPrecondition(
            "sgi",
            "Standardized Groundwater Index",
            "groundwater_level",
            min_years=10,
            marginal_years=5,
            resolution="monthly",
            tool="sgi_drought",
            problems=("drought", "groundwater_decline"),
            citation="Bloomfield & Marchant (2013)",
            note="With a rainfall index beside it, the SPI-to-SGI lag says how long a deficit takes to reach the well.",
        ),
        MethodPrecondition(
            "groundwater_trend",
            "Groundwater-level trend (m/yr) with confidence interval",
            "groundwater_level",
            min_years=10,
            marginal_years=5,
            resolution="monthly",
            tool="analyze_station",
            problems=("groundwater_decline",),
            citation="Jasechko et al. (2024)",
            note="Remove the seasonal cycle first; compare the recent decade with the full record.",
        ),
        MethodPrecondition(
            "recharge_wtf",
            "Recharge by water-table fluctuation",
            "groundwater_level",
            min_years=3,
            marginal_years=1,
            resolution="daily",
            tool="recharge",
            problems=("groundwater_decline", "supply_reliability"),
            citation="Healy & Cook (2002)",
            note="Needs a specific yield; a defaulted value is a stated assumption. Triangulate with baseflow.",
        ),
        MethodPrecondition(
            "gr4j_calibration",
            "GR4J rainfall-runoff calibration",
            "discharge",
            min_years=5,
            marginal_years=3,
            resolution="daily",
            max_area_km2=10_000,
            needs=("forcing",),
            tool="gr4j",
            problems=("climate_change", "supply_reliability", "ungauged_flow"),
            citation="Perrin et al. (2003)",
            note="A lumped model; above the ceiling the structure is wrong, not the parameters (#273). "
            "Weak on low flows next to GR5J/GR6J.",
        ),
        MethodPrecondition(
            "similar_basins",
            "Donor gauges by catchment similarity",
            None,
            ungauged=True,
            tool="similar_basins",
            problems=("ungauged_flow", "flood_risk", "supply_reliability"),
            citation="Oudin et al. (2008)",
        ),
        MethodPrecondition(
            "regionalize_signatures",
            "Flow signatures transferred from donors",
            None,
            ungauged=True,
            min_donors=3,
            tool="regionalize_signatures",
            problems=("ungauged_flow", "flood_risk", "supply_reliability"),
            note="Quote the band and the leave-one-out skill with every number.",
        ),
        MethodPrecondition(
            "glofas_cross_check",
            "GloFAS modelled discharge as an independent check",
            None,
            needs=("glofas",),
            tool="anywhere",
            problems=("flood_risk", "ungauged_flow"),
            citation="Harrigan et al. (2020)",
        ),
        MethodPrecondition(
            "climate_projection",
            "CMIP6 ensemble change factors on the baseline statistic",
            None,
            needs=("gcms>=3",),
            tool="climate",
            problems=("climate_change",),
            citation="Wasko et al. (2024) HESS",
            note="Report the ensemble spread, never the mean alone; design guidance under change is immature.",
        ),
        MethodPrecondition(
            "fao56_et0",
            "FAO-56 Penman-Monteith reference evapotranspiration",
            None,
            needs=("temperature",),
            tool="reference_et",
            problems=("irrigation", "drought"),
            citation="Allen et al. (1998); FAO (2025) revised edition",
        ),
        MethodPrecondition(
            "crop_water_requirement",
            "Crop water requirement (single or dual Kc)",
            None,
            needs=("temperature",),
            tool="crop_water_demand",
            problems=("irrigation",),
            citation="Allen et al. (1998); FAO (2025) revised edition, doi:10.4060/cd6621en",
            note="Kc from the FAO-56 (1998) tables pending the 2025 revision (#310); reanalysis-forced ET0 carries "
            "bias (Agric. Water Manage. 2024, doi:10.1016/j.agwat.2024.108732).",
        ),
        MethodPrecondition(
            "water_quality_index",
            "Water quality index against guideline values (CCME WQI 1.0, NSF WQI)",
            "water_quality",
            min_years=0,
            tool="wqi",
            problems=("water_quality",),
            citation="CCME (2001) WQI 1.0; Brown et al. (1970) NSF WQI; WHO (2022) drinking-water guidelines; "
            "Ayers & Westcot (1985) FAO 29",
            note="Only over sampled parameters; the archive carries none until Phase 3. CCME asks for at least "
            "four parameters sampled four times each.",
        ),
        MethodPrecondition(
            "iwqi",
            "Irrigation water quality (SAR, sodium percentage, RSC, FAO 29 restriction)",
            "water_quality",
            min_years=0,
            tool="iwqi",
            problems=("water_quality", "irrigation"),
            citation="Ayers & Westcot (1985) FAO 29; Richards (1954); Wilcox (1955); Eaton (1950)",
            note="Needs the major ions (Na, Ca, Mg, HCO3) and conductivity; what was not sampled is not judged.",
        ),
    ]
}


def method_ids(problem: str | None = None) -> list[str]:
    """Every method id, or those that apply to one problem kind."""
    return [m.id for m in METHODS.values() if problem is None or problem in m.problems]


def assess_method(method: MethodPrecondition | str, ctx: SiteContext) -> dict[str, Any]:
    """Whether ``method`` is defensible at a site, with the reason in plain words."""
    pre = METHODS[method] if isinstance(method, str) else method
    reasons: list[str] = []
    status = DEFENSIBLE

    if pre.ungauged and not ctx.ungauged and pre.min_donors is None:
        reasons.append("meant for an ungauged point; a gauge is available here")
        status = MARGINAL

    if pre.variable is not None and pre.variable != "water_quality":
        years = ctx.years_by_variable.get(pre.variable)
        if years is None:
            return {
                "method": pre.id,
                "status": NOT_DEFENSIBLE,
                "reason": f"no {pre.variable.replace('_', ' ')} record at this site",
            }
        if pre.min_years is not None and years < pre.min_years:
            floor = pre.marginal_years if pre.marginal_years is not None else pre.min_years
            if years < floor:
                return {
                    "method": pre.id,
                    "status": NOT_DEFENSIBLE,
                    "reason": f"{years:g} years of {pre.variable.replace('_', ' ')}, below the {floor:g}-year floor",
                }
            status = MARGINAL
            reasons.append(f"{years:g} years of record, {pre.min_years:g} wanted")
        res = ctx.resolution_by_variable.get(pre.variable)
        if pre.resolution and res and _RES_RANK.get(res, 0) < _RES_RANK.get(pre.resolution, 0):
            return {
                "method": pre.id,
                "status": NOT_DEFENSIBLE,
                "reason": f"needs {pre.resolution} data, the record is {res}",
            }
        if pre.max_return_period_factor and ctx.return_period and pre.min_years is not None:
            cap = pre.max_return_period_factor * years
            if ctx.return_period > cap:
                status = MARGINAL
                reasons.append(
                    f"T = {ctx.return_period:g} years is beyond about {cap:.0f} years "
                    f"({pre.max_return_period_factor:g} times the record)"
                )
    elif pre.variable is None and pre.min_years is not None and not pre.ungauged:
        # A method over any record (trend): judge it on the longest record present.
        if not ctx.years_by_variable:
            return {"method": pre.id, "status": NOT_DEFENSIBLE, "reason": "no record at this site"}
        years = max(ctx.years_by_variable.values())
        if years < pre.min_years:
            floor = pre.marginal_years if pre.marginal_years is not None else pre.min_years
            if years < floor:
                return {
                    "method": pre.id,
                    "status": NOT_DEFENSIBLE,
                    "reason": f"{years:g} years of record, below the {floor:g}-year floor",
                }
            status = MARGINAL
            reasons.append(f"{years:g} years of record, {pre.min_years:g} wanted")

    if pre.max_area_km2 and ctx.area_km2 and ctx.area_km2 > pre.max_area_km2:
        return {
            "method": pre.id,
            "status": NOT_DEFENSIBLE,
            "reason": f"catchment of {ctx.area_km2:,.0f} km² is above the {pre.max_area_km2:,.0f} km² "
            "ceiling for a lumped model",
        }
    if pre.min_donors and (ctx.donors or 0) < pre.min_donors:
        return {
            "method": pre.id,
            "status": NOT_DEFENSIBLE,
            "reason": f"needs at least {pre.min_donors} donor gauges, {ctx.donors or 0} found",
        }
    for need in pre.needs:
        if need not in ctx.available:
            return {
                "method": pre.id,
                "status": NOT_DEFENSIBLE,
                "reason": f"needs {need.replace('>=', ' of at least ')}, not available here",
            }
    if (
        pre.variable == "water_quality"
        and "water_quality" not in ctx.years_by_variable
        and "water_quality" not in ctx.available
    ):
        return {"method": pre.id, "status": NOT_DEFENSIBLE, "reason": "no water-quality samples here"}

    return {"method": pre.id, "status": status, "reason": "; ".join(reasons) if reasons else "the record supports it"}


def sufficiency_table(ctx: SiteContext, *, problem: str | None = None) -> list[dict[str, Any]]:
    """Every method (or those for one problem) assessed against a site, defensible first."""
    rows = [
        dict(assess_method(m, ctx), label=m.label, tool=m.tool)
        for m in METHODS.values()
        if problem is None or problem in m.problems
    ]
    order = {DEFENSIBLE: 0, MARGINAL: 1, NOT_DEFENSIBLE: 2}
    return sorted(rows, key=lambda r: (order[r["status"]], r["method"]))


def describe_preconditions() -> list[dict[str, Any]]:
    """The registry as plain dicts, for docs, MCP and the page."""
    return [asdict(m) for m in METHODS.values()]
