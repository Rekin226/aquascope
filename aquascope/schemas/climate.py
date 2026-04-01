"""Pydantic schemas for climate projection data."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ClimateProjection(BaseModel):
    """Metadata for a CMIP6 climate-model projection dataset."""

    model_name: str = Field(..., description="GCM name, e.g. 'ACCESS-CM2'")
    scenario: str = Field(..., description="SSP scenario identifier, e.g. 'ssp245'")
    variable: str = Field(..., description="Climate variable, e.g. 'tas', 'pr', 'tasmax', 'tasmin'")
    start_year: int
    end_year: int
    spatial_resolution: str | None = None
    temporal_resolution: str = "monthly"
    unit: str | None = None
    metadata: dict | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_name": "ACCESS-CM2",
                    "scenario": "ssp245",
                    "variable": "tas",
                    "start_year": 2015,
                    "end_year": 2100,
                    "spatial_resolution": "1.875x1.25",
                    "temporal_resolution": "monthly",
                    "unit": "K",
                }
            ]
        }
    }


class DownscaledData(BaseModel):
    """Record describing a downscaled dataset."""

    method: str = Field(..., description="Downscaling method, e.g. 'quantile_mapping'")
    variable: str
    station_id: str | None = None
    start_date: datetime
    end_date: datetime
    n_points: int
    metrics: dict | None = None


class ClimateIndex(BaseModel):
    """A computed climate index value."""

    index_name: str = Field(..., description="Name of the climate index, e.g. 'PDSI', 'PCI'")
    value: float
    unit: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    classification: str | None = None
