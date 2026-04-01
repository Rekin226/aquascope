"""Agricultural water data schemas.

References
----------
- Allen, R. G., et al. (1998). FAO Irrigation and Drainage Paper 56.
  ISBN 92-5-104219-5
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class ETReference(BaseModel):
    """Reference evapotranspiration (ET₀) record."""

    date: datetime.date
    eto_mm: float = Field(..., description="Reference ET in mm/day")
    method: str = "penman_monteith"
    t_min: float | None = None
    t_max: float | None = None
    humidity_mean: float | None = None
    wind_speed: float | None = None
    solar_radiation: float | None = None
    station_id: str | None = None


class CropWaterRequirement(BaseModel):
    """Crop water requirement for a growth stage."""

    crop: str
    stage: str  # "initial", "development", "mid", "late"
    kc: float = Field(..., description="Crop coefficient")
    etc_mm: float = Field(..., description="Crop ET in mm/day")
    duration_days: int | None = None


class IrrigationDemand(BaseModel):
    """Irrigation water demand record."""

    date: datetime.date
    crop: str
    etc_mm: float
    effective_rain_mm: float = 0.0
    net_irrigation_mm: float = 0.0
    gross_irrigation_mm: float = 0.0
    efficiency: float = 0.7


class SoilWaterStatus(BaseModel):
    """Daily soil water balance status."""

    date: datetime.date
    soil_moisture_mm: float
    depletion_mm: float
    deep_percolation_mm: float = 0.0
    runoff_mm: float = 0.0
    irrigation_trigger: bool = False


class AquastatRecord(BaseModel):
    """FAO AQUASTAT country-level water data."""

    country: str
    country_code: str
    year: int
    variable: str
    value: float
    unit: str
    source: str = "AQUASTAT"


class WaPORObservation(BaseModel):
    """FAO WaPOR cube observation for ET or productivity workflows."""

    cube_code: str
    value: float
    source: str = "WAPOR"
    cube_label: str | None = None
    date: datetime.date | None = None
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    bbox: tuple[float, float, float, float] | None = None
    unit: str | None = None
    statistic: str | None = None
    aoi_id: str | None = None
    level: str | None = None
