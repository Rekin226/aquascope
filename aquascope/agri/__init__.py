"""Agricultural water management module.

Implements FAO-56 Penman-Monteith reference evapotranspiration,
crop water requirements, and soil water balance modeling.
"""

from __future__ import annotations

from aquascope.agri.benchmark import (
    AgricultureBenchmarkResult,
    benchmark_aquastat,
    list_benchmark_metrics,
)
from aquascope.agri.crop_water import crop_water_requirement, get_kc, irrigation_schedule
from aquascope.agri.eto import hargreaves, penman_monteith_daily
from aquascope.agri.planner import (
    IrrigationPlan,
    default_season_end_date,
    fetch_openmeteo_plan_inputs,
    plan_irrigation,
)
from aquascope.agri.productivity import (
    WaPORProductivityResult,
    estimate_wapor_productivity,
    list_productivity_metrics,
)
from aquascope.agri.water_balance import SoilWaterBalance

__all__ = [
    "AgricultureBenchmarkResult",
    "benchmark_aquastat",
    "penman_monteith_daily",
    "hargreaves",
    "get_kc",
    "crop_water_requirement",
    "irrigation_schedule",
    "IrrigationPlan",
    "SoilWaterBalance",
    "default_season_end_date",
    "fetch_openmeteo_plan_inputs",
    "list_benchmark_metrics",
    "list_productivity_metrics",
    "plan_irrigation",
    "WaPORProductivityResult",
    "estimate_wapor_productivity",
]
