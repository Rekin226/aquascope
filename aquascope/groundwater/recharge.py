"""Groundwater recharge estimation methods.

Implements four common approaches:

- **Water Table Fluctuation (WTF)** — R = Sy × Δh
- **Chloride Mass Balance (CMB)** — R = P × (Cl_p / Cl_gw)
- **Baseflow separation** — uses baseflow as a recharge proxy
- **Soil Water Balance** — R = P − ET − Q − ΔS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RechargeResult:
    """Result of a recharge estimation.

    Attributes
    ----------
    method:
        Name of the estimation method used.
    value_mm_per_year:
        Estimated recharge rate (mm/year).
    uncertainty:
        Estimated uncertainty (mm/year), or None if not quantified.
    metadata:
        Additional method-specific information.
    """

    method: str
    value_mm_per_year: float
    uncertainty: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def water_table_fluctuation(
    levels: pd.Series,
    specific_yield: float,
) -> RechargeResult:
    """Estimate recharge using the Water Table Fluctuation method.

    ``R = Sy × Δh`` where Δh is the total rise in water level per year.

    Parameters
    ----------
    levels:
        Water-level series (hydraulic head, m) with DatetimeIndex.
    specific_yield:
        Specific yield (dimensionless, 0–1).

    Returns
    -------
    RechargeResult
        Recharge estimate in mm/year.

    Raises
    ------
    ValueError
        If specific yield is out of range or series is too short.
    """
    if not 0 < specific_yield < 1:
        raise ValueError(f"Specific yield must be in (0, 1), got {specific_yield}.")
    if len(levels) < 2:
        raise ValueError("Need at least 2 data points.")

    y = levels.dropna().values.astype(float)
    idx = levels.dropna().index

    # Sum positive rises (each rise event contributes to recharge)
    diffs = np.diff(y)
    total_rise_m = float(np.sum(diffs[diffs > 0]))

    # Duration in years
    period_days = (idx[-1] - idx[0]).total_seconds() / 86400.0
    period_years = period_days / 365.25 if period_days > 0 else 1.0

    recharge_m_per_year = specific_yield * total_rise_m / period_years
    recharge_mm = recharge_m_per_year * 1000.0

    logger.info("WTF recharge: %.1f mm/year (Sy=%.3f, Δh_total=%.2f m)", recharge_mm, specific_yield, total_rise_m)
    return RechargeResult(
        method="water_table_fluctuation",
        value_mm_per_year=recharge_mm,
        uncertainty=None,
        metadata={"specific_yield": specific_yield, "total_rise_m": total_rise_m, "period_years": period_years},
    )


def chloride_mass_balance(
    precip_cl: float,
    gw_cl: float,
    precip_mm: float,
) -> RechargeResult:
    """Estimate recharge using the Chloride Mass Balance method.

    ``R = P × (Cl_p / Cl_gw)``

    Parameters
    ----------
    precip_cl:
        Chloride concentration in precipitation (mg/L).
    gw_cl:
        Chloride concentration in groundwater (mg/L).
    precip_mm:
        Annual precipitation (mm/year).

    Returns
    -------
    RechargeResult
        Recharge estimate in mm/year.

    Raises
    ------
    ValueError
        If chloride concentrations are non-positive.
    """
    if precip_cl <= 0 or gw_cl <= 0:
        raise ValueError("Chloride concentrations must be positive.")
    if precip_mm <= 0:
        raise ValueError("Precipitation must be positive.")

    recharge_mm = precip_mm * (precip_cl / gw_cl)

    logger.info(
        "CMB recharge: %.1f mm/year (Cl_p=%.2f, Cl_gw=%.2f, P=%.0f mm)",
        recharge_mm, precip_cl, gw_cl, precip_mm,
    )
    return RechargeResult(
        method="chloride_mass_balance",
        value_mm_per_year=recharge_mm,
        uncertainty=None,
        metadata={"precip_cl_mg_l": precip_cl, "gw_cl_mg_l": gw_cl, "precip_mm": precip_mm},
    )


def baseflow_recharge(
    discharge: pd.Series,
    area_km2: float,
) -> RechargeResult:
    """Estimate recharge using baseflow separation as a proxy.

    Applies the Lyne–Hollick filter to separate baseflow, then
    converts total baseflow volume to a depth over the contributing area.

    Parameters
    ----------
    discharge:
        Stream discharge series (m³/s) with DatetimeIndex.
    area_km2:
        Contributing catchment area in km².

    Returns
    -------
    RechargeResult
        Recharge estimate in mm/year.

    Raises
    ------
    ValueError
        If discharge is empty or area is non-positive.
    """
    if len(discharge) == 0:
        raise ValueError("Discharge series is empty.")
    if area_km2 <= 0:
        raise ValueError("Area must be positive.")

    from aquascope.hydrology.baseflow import lyne_hollick

    result = lyne_hollick(discharge)
    bf = result.df["baseflow"]

    # Total baseflow volume (m³) — assume daily time step
    idx = bf.index
    if len(idx) > 1:
        dt_seconds = np.median(np.diff(idx).astype("timedelta64[s]").astype(float))
    else:
        dt_seconds = 86400.0

    total_volume_m3 = float(bf.sum() * dt_seconds)

    # Duration in years
    period_days = (idx[-1] - idx[0]).total_seconds() / 86400.0
    period_years = period_days / 365.25 if period_days > 0 else 1.0

    volume_per_year = total_volume_m3 / period_years
    area_m2 = area_km2 * 1e6
    recharge_mm = (volume_per_year / area_m2) * 1000.0

    logger.info("Baseflow recharge: %.1f mm/year (BFI=%.3f)", recharge_mm, result.bfi)
    return RechargeResult(
        method="baseflow_recharge",
        value_mm_per_year=recharge_mm,
        uncertainty=None,
        metadata={"bfi": result.bfi, "area_km2": area_km2, "period_years": period_years},
    )


def soil_water_balance_recharge(
    precip: pd.Series,
    et: pd.Series,
    runoff: pd.Series,
    delta_s: pd.Series | None = None,
) -> RechargeResult:
    """Estimate recharge from the soil water balance.

    ``R = P − ET − Q − ΔS``

    Parameters
    ----------
    precip:
        Precipitation series (mm) with DatetimeIndex.
    et:
        Evapotranspiration series (mm).
    runoff:
        Runoff series (mm).
    delta_s:
        Change in soil storage (mm). Defaults to zero if not provided.

    Returns
    -------
    RechargeResult
        Recharge estimate in mm/year.

    Raises
    ------
    ValueError
        If input series have mismatched lengths.
    """
    if not (len(precip) == len(et) == len(runoff)):
        raise ValueError("All input series must have the same length.")

    p = precip.values.astype(float)
    e = et.values.astype(float)
    q = runoff.values.astype(float)
    ds = delta_s.values.astype(float) if delta_s is not None else np.zeros(len(p))

    recharge = p - e - q - ds
    total_recharge_mm = float(np.sum(recharge))

    idx = precip.index
    period_days = (idx[-1] - idx[0]).total_seconds() / 86400.0
    period_years = period_days / 365.25 if period_days > 0 else 1.0

    recharge_mm_per_year = total_recharge_mm / period_years

    logger.info(
        "SWB recharge: %.1f mm/year (total=%.1f mm over %.1f years)",
        recharge_mm_per_year, total_recharge_mm, period_years,
    )
    return RechargeResult(
        method="soil_water_balance",
        value_mm_per_year=recharge_mm_per_year,
        uncertainty=None,
        metadata={
            "total_recharge_mm": total_recharge_mm,
            "period_years": period_years,
            "total_precip_mm": float(np.sum(p)),
            "total_et_mm": float(np.sum(e)),
            "total_runoff_mm": float(np.sum(q)),
        },
    )
