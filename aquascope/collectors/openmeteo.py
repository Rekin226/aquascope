"""
Open-Meteo collector — global weather and hydrological data (free, no API key).

Wraps the Open-Meteo API (https://open-meteo.com/) to fetch:
- Historical weather observations (ERA5 reanalysis)
- Weather forecasts (up to 16 days)
- Historical hydrology (river discharge from GloFAS / ERA5-Land)

All data is free, unrestricted, and requires **no API key**.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterQualitySample

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1"
_FORECAST_URL = "https://api.open-meteo.com/v1"
_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"


class OpenMeteoCollector(BaseCollector):
    """Fetch weather, climate reanalysis, and river-discharge data from Open-Meteo.

    Parameters
    ----------
    mode : str
        ``"weather"`` (default), ``"forecast"``, or ``"flood"`` (GloFAS discharge).

    Example
    -------
    >>> collector = OpenMeteoCollector(mode="weather")
    >>> records = collector.collect(
    ...     latitude=25.03, longitude=121.57,
    ...     start_date="2023-01-01", end_date="2023-12-31",
    ...     daily=["temperature_2m_mean", "precipitation_sum"],
    ... )
    """

    name = "openmeteo"

    def __init__(self, mode: str = "weather", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def fetch_raw(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str | None = None,
        end_date: str | None = None,
        daily: list[str] | None = None,
        hourly: list[str] | None = None,
        forecast_days: int = 7,
    ) -> dict:
        """Call the Open-Meteo API and return the raw JSON response.

        Parameters
        ----------
        latitude, longitude : float
            Site coordinates.
        start_date, end_date : str | None
            ISO-8601 date strings (required for archive mode).
        daily : list[str] | None
            Daily variables to request (e.g. ``["precipitation_sum"]``).
        hourly : list[str] | None
            Hourly variables (e.g. ``["river_discharge"]``).
        forecast_days : int
            Number of forecast days (only for ``mode="forecast"``).
        """
        params: dict = {"latitude": latitude, "longitude": longitude, "timezone": "auto"}

        if self.mode == "weather":
            url = f"{_ARCHIVE_URL}/era5"
            if not (start_date and end_date):
                raise ValueError("start_date and end_date are required for weather mode")
            params["start_date"] = start_date
            params["end_date"] = end_date
            params["daily"] = ",".join(daily or ["precipitation_sum", "temperature_2m_mean"])
        elif self.mode == "forecast":
            url = f"{_FORECAST_URL}/forecast"
            params["forecast_days"] = forecast_days
            params["daily"] = ",".join(daily or ["precipitation_sum", "temperature_2m_max", "temperature_2m_min"])
        elif self.mode == "flood":
            url = _FLOOD_URL
            if start_date and end_date:
                params["start_date"] = start_date
                params["end_date"] = end_date
            params["daily"] = ",".join(daily or ["river_discharge"])
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        return self.client.get_json(url, params=params)

    def normalise(self, raw: dict) -> Sequence[BaseModel]:
        """Convert Open-Meteo JSON into unified ``WaterQualitySample`` records.

        Weather / climate variables are stored as WaterQualitySample with
        ``parameter`` set to the variable name (e.g. ``precipitation_sum``).
        """
        lat = raw.get("latitude", 0)
        lon = raw.get("longitude", 0)
        location = GeoLocation(latitude=lat, longitude=lon)

        records: list[WaterQualitySample] = []

        daily_data = raw.get("daily", {})
        time_col = daily_data.get("time", [])
        daily_units = raw.get("daily_units", {})

        for key, values in daily_data.items():
            if key == "time":
                continue
            unit = daily_units.get(key, "")
            for ts, val in zip(time_col, values):
                if val is None:
                    continue
                records.append(WaterQualitySample(
                    source=DataSource.OPENMETEO,
                    station_id=f"openmeteo_{lat}_{lon}",
                    station_name=f"Open-Meteo ({lat}, {lon})",
                    location=location,
                    sample_datetime=datetime.fromisoformat(ts),
                    parameter=key,
                    value=float(val),
                    unit=unit,
                ))

        hourly_data = raw.get("hourly", {})
        hourly_time = hourly_data.get("time", [])
        hourly_units = raw.get("hourly_units", {})

        for key, values in hourly_data.items():
            if key == "time":
                continue
            unit = hourly_units.get(key, "")
            for ts, val in zip(hourly_time, values):
                if val is None:
                    continue
                records.append(WaterQualitySample(
                    source=DataSource.OPENMETEO,
                    station_id=f"openmeteo_{lat}_{lon}",
                    station_name=f"Open-Meteo ({lat}, {lon})",
                    location=location,
                    sample_datetime=datetime.fromisoformat(ts),
                    parameter=key,
                    value=float(val),
                    unit=unit,
                ))

        logger.info("Normalised %d records from Open-Meteo", len(records))
        return records
