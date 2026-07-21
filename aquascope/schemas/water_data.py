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
    INDIA_WRIS = "india_wris"
    HUBEAU = "france_hubeau"
    GRDC = "grdc"
    CAMELS_CL = "camels_cl"

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


class StreamflowReading(BaseModel):
    """A single river discharge (streamflow) observation."""

    source: DataSource
    station_id: str
    station_name: str | None = None
    location: GeoLocation | None = None
    reading_datetime: datetime
    discharge_cms: float = Field(..., description="Discharge in cubic meters per second")
    source_type: str = Field(..., description="'in_situ' (gauge) or 'satellite' (remote sensing estimate)")
    uncertainty_cms: float | None = Field(None, description="Estimated uncertainty, satellite products only")
    catchment_area_km2: float | None = Field(
        None, description="Upstream drainage area in km2, if known — enables mm/day normalization"
    )
    unit: str = "m3/s"
    remark: str | None = None

    @property
    def runoff_mm_day(self) -> float | None:
        """Area-normalized daily streamflow in mm/day.

        ``None`` when ``catchment_area_km2`` is not set — this is a derived
        convenience, not a persisted field, so callers must handle absence
        rather than getting a misleading zero.
        """
        if self.catchment_area_km2 is None or self.catchment_area_km2 <= 0:
            return None
        return discharge_cms_to_runoff_mm_day(self.discharge_cms, self.catchment_area_km2)


def discharge_cms_to_runoff_mm_day(discharge_cms: float, catchment_area_km2: float) -> float:
    """Convert discharge (m3/s) to area-normalized daily runoff (mm/day).

    runoff [mm/day] = discharge [m3/s] * 86400 [s/day] / area [km2] / 1000 [km2->m2 in millions]

    Equivalently: (discharge_cms * 86400) / (catchment_area_km2 * 1_000_000) * 1000
    """
    if catchment_area_km2 <= 0:
        raise ValueError(f"catchment_area_km2 must be positive, got {catchment_area_km2}")
    seconds_per_day = 86_400
    km2_to_m2 = 1_000_000
    volume_m3_per_day = discharge_cms * seconds_per_day
    area_m2 = catchment_area_km2 * km2_to_m2
    depth_m_per_day = volume_m3_per_day / area_m2
    return depth_m_per_day * 1000  # m -> mm
