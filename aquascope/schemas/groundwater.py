"""Pydantic schemas for groundwater data.

Provides data models for groundwater level observations, aquifer
properties, and GRACE satellite observations.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from aquascope.schemas.water_data import DataSource, GeoLocation


class GroundwaterLevel(BaseModel):
    """A single groundwater level measurement."""

    source: DataSource
    station_id: str
    station_name: str | None = None
    location: GeoLocation | None = None
    measurement_datetime: datetime
    water_level_m: float = Field(..., description="Depth to water below ground surface (m)")
    unit: str = "m"
    aquifer_name: str | None = None
    well_depth_m: float | None = None


class AquiferProperties(BaseModel):
    """Hydraulic properties of an aquifer."""

    aquifer_id: str
    name: str
    transmissivity: float | None = Field(default=None, description="m²/day")
    storativity: float | None = None
    specific_yield: float | None = None
    hydraulic_conductivity: float | None = Field(default=None, description="m/day")
    aquifer_type: str | None = Field(default=None, description="confined, unconfined, or semi-confined")


class GRACEObservation(BaseModel):
    """GRACE satellite Total Water Storage observation."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: datetime
    tws_anomaly_mm: float
    uncertainty_mm: float | None = None
