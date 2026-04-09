"""
Pydantic schemas for unified water data representation.

All collectors normalise their output into these schemas so that
downstream analytics and AI operate on a single data model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Supported data providers."""

    TAIWAN_MOENV = "taiwan_moenv"
    TAIWAN_WRA = "taiwan_wra"
    TAIWAN_CIVIL_IOT = "taiwan_civil_iot"
    USGS = "usgs"
    SDG6 = "sdg6"
    GEMSTAT = "gemstat"
    WQP = "wqp"
    OPENMETEO = "openmeteo"
    COPERNICUS = "copernicus"
    AQUASTAT = "aquastat"
    WAPOR = "wapor"
    USGS_GW = "usgs_groundwater"
    GRACE = "grace"
    EU_WFD = "eu_wfd"
    JAPAN_MLIT = "japan_mlit"
    KOREA_WAMIS = "korea_wamis"
    TAIWAN_WRA_FHY = "taiwan_wra_fhy"
    TAIWAN_WRA_IOT = "taiwan_wra_iot"
    TAIWAN_DATAGOV = "taiwan_datagov"


class GeoLocation(BaseModel):
    """Geographic coordinates for a monitoring station or sample point."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    datum: str = Field(default="WGS84")


class WaterQualitySample(BaseModel):
    """A single water-quality measurement."""

    source: DataSource
    station_id: str
    station_name: str | None = None
    location: GeoLocation | None = None
    sample_datetime: datetime
    parameter: str = Field(..., description="e.g. pH, DO, BOD5, NH3-N, COD, SS")
    value: float
    unit: str
    basin: str | None = None
    river: str | None = None
    county: str | None = None
    remark: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "taiwan_moenv",
                    "station_id": "01001",
                    "station_name": "Tamsui River - Guandu Bridge",
                    "location": {"latitude": 25.115, "longitude": 121.459, "datum": "WGS84"},
                    "sample_datetime": "2025-12-15T10:00:00",
                    "parameter": "DO",
                    "value": 5.2,
                    "unit": "mg/L",
                    "basin": "Tamsui River",
                    "county": "Taipei",
                }
            ]
        }
    }


class WaterLevelReading(BaseModel):
    """A single water-level observation."""

    source: DataSource
    station_id: str
    station_name: str | None = None
    location: GeoLocation | None = None
    reading_datetime: datetime
    water_level: float
    unit: str = "m"
    remark: str | None = None


class ReservoirStatus(BaseModel):
    """Daily reservoir status record."""

    source: DataSource
    reservoir_name: str
    date: datetime
    effective_capacity_m3: float | None = None
    current_storage_m3: float | None = None
    storage_percentage: float | None = None
    inflow_cms: float | None = None
    outflow_cms: float | None = None
    water_level: float | None = None
    remark: str | None = None


class SDG6Indicator(BaseModel):
    """UN SDG 6 indicator value for a country."""

    indicator_code: str = Field(..., description="e.g. 6.1.1, 6.3.1, 6.4.2")
    indicator_name: str | None = None
    country_code: str
    country_name: str | None = None
    year: int
    value: float | None = None
    unit: str | None = None
    series_code: str | None = None
