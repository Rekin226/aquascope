"""
Collector for WRA 防災資訊網 (Disaster Prevention Information) API.

API docs : https://fhy.wra.gov.tw/WraApi
Endpoints:
  Water    : /v1/Water/Station, /v1/Water/RealTimeInfo
  Rainfall : /v1/Rainfall/Station, /v1/Rainfall/RealTimeInfo
  Flow     : /v1/Flow/Station, /v1/Flow/RealTimeInfo

No authentication required for most endpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

FHY_BASE = "https://fhy.wra.gov.tw/WraApi"

_REALTIME_PATHS: dict[str, str] = {
    "water": "v1/Water/RealTimeInfo",
    "rainfall": "v1/Rainfall/RealTimeInfo",
    "flow": "v1/Flow/RealTimeInfo",
}

_PARAM_NAMES: dict[str, str] = {
    "water": "WaterLevel",
    "rainfall": "Rainfall",
    "flow": "Discharge",
}

_PARAM_UNITS: dict[str, str] = {
    "water": "m",
    "rainfall": "mm",
    "flow": "cms",
}


class TaiwanWRAFhyCollector(BaseCollector):
    """
    Collect real-time hydrological data from the WRA 防災資訊網 (Fhy) API.

    Parameters
    ----------
    data_type : str
        One of ``"water"`` (water level), ``"rainfall"``, or ``"flow"``
        (river discharge).  Defaults to ``"water"``.
    """

    name = "taiwan_wra_fhy"

    def __init__(
        self,
        data_type: str = "water",
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=FHY_BASE,
                rate_limiter=RateLimiter(max_calls=12, period_seconds=60),
                cache_ttl_seconds=600,
            )
        )
        if data_type not in _REALTIME_PATHS:
            raise ValueError(
                f"data_type must be one of {list(_REALTIME_PATHS)}, got {data_type!r}"
            )
        self.data_type = data_type

    def fetch_raw(self, **kwargs) -> list[dict]:
        """Fetch real-time data for the configured data type."""
        path = _REALTIME_PATHS[self.data_type]
        data = self.client.get_json(path)
        if isinstance(data, list):
            return data
        return data.get("Data", data.get("data", data.get("records", [])))

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        param_name = _PARAM_NAMES[self.data_type]
        unit = _PARAM_UNITS[self.data_type]
        samples: list[WaterQualitySample] = []

        for rec in raw:
            try:
                value_str = (
                    rec.get(param_name)
                    or rec.get(param_name.lower())
                    or rec.get("Value")
                    or rec.get("value")
                )
                if value_str is None or str(value_str).strip() in ("", "-", "--", "ND"):
                    continue

                loc = None
                lat = rec.get("Latitude") or rec.get("lat")
                lon = rec.get("Longitude") or rec.get("lon")
                if lat and lon:
                    loc = GeoLocation(latitude=float(lat), longitude=float(lon))

                time_str = (
                    rec.get("RecordTime")
                    or rec.get("ObservationTime")
                    or rec.get("DateTime")
                    or rec.get("time")
                    or ""
                )
                sample_dt = datetime.fromisoformat(time_str) if time_str else datetime.utcnow()

                samples.append(
                    WaterQualitySample(
                        source=DataSource.TAIWAN_WRA_FHY,
                        station_id=str(
                            rec.get("StationIdentifier")
                            or rec.get("StationNo")
                            or rec.get("ID")
                            or "unknown"
                        ),
                        station_name=rec.get("StationName") or rec.get("stationName"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=param_name,
                        value=float(value_str),
                        unit=unit,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping FHY record: %s", exc)

        return samples
