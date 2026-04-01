"""Groundwater analysis module.

Provides tools for GRACE satellite groundwater estimation, well data analysis,
recharge estimation, and aquifer hydraulics.
"""

from __future__ import annotations

from aquascope.groundwater.aquifer import (
    AquiferParams,
    SafeYieldResult,
    cooper_jacob,
    estimate_transmissivity,
    safe_yield,
    theis_drawdown,
    theis_recovery,
)
from aquascope.groundwater.grace import (
    DepletionResult,
    GRACEProcessor,
    GWSAnomaly,
    GWSResult,
    TrendResult,
)
from aquascope.groundwater.recharge import (
    RechargeResult,
    baseflow_recharge,
    chloride_mass_balance,
    soil_water_balance_recharge,
    water_table_fluctuation,
)
from aquascope.groundwater.wells import (
    HydrographResult,
    RecessionResult,
    SeasonalResult,
    WellTrendResult,
    recession_analysis,
    seasonal_decomposition,
    trend_detection,
    well_hydrograph,
)

__all__ = [
    # grace
    "GRACEProcessor",
    "GWSResult",
    "TrendResult",
    "GWSAnomaly",
    "DepletionResult",
    # wells
    "HydrographResult",
    "WellTrendResult",
    "SeasonalResult",
    "RecessionResult",
    "well_hydrograph",
    "trend_detection",
    "seasonal_decomposition",
    "recession_analysis",
    # recharge
    "RechargeResult",
    "water_table_fluctuation",
    "chloride_mass_balance",
    "baseflow_recharge",
    "soil_water_balance_recharge",
    # aquifer
    "AquiferParams",
    "SafeYieldResult",
    "theis_drawdown",
    "cooper_jacob",
    "theis_recovery",
    "estimate_transmissivity",
    "safe_yield",
]
