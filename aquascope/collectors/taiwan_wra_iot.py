"""
Collector for WRA 水利署水文開放資料 IoT API.

API docs : https://iot.wra.gov.tw  (Swagger: https://iot.wra.gov.tw/swagger)
No authentication required (免驗證).

Provides real-time access to:
  - Groundwater level data (地下水位)
  - Rainfall accumulation data (累積雨量)
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

IOT_BASE = "https://iot.wra.gov.tw"

# Candidate paths per data type, tried in order.
# The correct path may be versioned (/api/v1/...) or under /opendata/.
# Check https://iot.wra.gov.tw/swagger/index.html for the authoritative list.
_DATA_TYPE_PATHS: dict[str, list[str]] = {
    "groundwater": [
        "api/Groundwater/RealTimeInfo",
        "api/v1/Groundwater/RealTimeInfo",
        "opendata/Groundwater/RealTimeInfo",
    ],
    "rainfall": [
        "api/Rainfall/Accumulation",
        "api/v1/Rainfall/Accumulation",
        "opendata/Rainfall/Accumulation",
    ],
}

# Hint the server to return JSON rather than an HTML error page.
_JSON_HEADERS = {"Accept": "application/json"}

_PARAM_NAMES: dict[str, str] = {
    "groundwater": "GroundwaterLevel",
    "rainfall": "RainfallAccumulation",
}

_PARAM_UNITS: dict[str, str] = {
    "groundwater": "m",
    "rainfall": "mm",
}


class TaiwanWRAIoTCollector(BaseCollector):
    """
    Collect real-time hydrological data from the WRA IoT open-data API.

    Parameters
    ----------
    data_type : str
        One of ``"groundwater"`` (地下水位) or ``"rainfall"`` (累積雨量).
        Defaults to ``"groundwater"``.
    """

    name = "taiwan_wra_iot"

    def __init__(
        self,
        data_type: str = "groundwater",
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=IOT_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=600,
            )
        )
        if data_type not in _DATA_TYPE_PATHS:
            raise ValueError(
                f"data_type must be one of {sorted(_DATA_TYPE_PATHS)}, got {data_type!r}"
            )
        self.data_type = data_type

    def fetch_raw(self, **kwargs) -> list[dict]:
        """Fetch real-time data for the configured data type.

        Tries each candidate path in order with an ``Accept: application/json``
        header.  ``CachedHTTPClient.get_json`` strips any BOM / leading
        whitespace and checks Content-Type before parsing, so non-JSON bodies
        surface as ``ValueError`` rather than an opaque ``JSONDecodeError``.

        On failure the first 200 chars of the response body are logged to help
        diagnose whether the server returned HTML, XML, or malformed JSON.
        """
        import json as _json  # local import — only needed for the error branch

        candidates = _DATA_TYPE_PATHS[self.data_type]
        last_error: Exception | None = None

        for path in candidates:
            try:
                data = self.client.get_json(path, headers=_JSON_HEADERS)
            except ValueError as exc:
                # get_json raises ValueError for HTML/XML bodies or
                # JSONDecodeError.  The message already contains a preview of
                # the response body (first 200–500 chars).
                logger.warning(
                    "[%s] Path %r returned non-JSON or malformed body: %s",
                    self.name,
                    path,
                    exc,
                )
                last_error = exc
                continue
            except _json.JSONDecodeError as exc:
                # Fallback for callers that might bypass _parse_response_json.
                logger.warning(
                    "[%s] JSONDecodeError on path %r (char %d): %s",
                    self.name,
                    path,
                    exc.pos,
                    exc,
                )
                last_error = exc
                continue

            logger.debug("[%s] Fetched data from path %r", self.name, path)
            if isinstance(data, list):
                return data
            return data.get("Data", data.get("data", data.get("records", [])))

        raise RuntimeError(
            f"[{self.name}] All candidate paths failed for data_type={self.data_type!r}. "
            f"Last error: {last_error}. "
            f"Check https://iot.wra.gov.tw/swagger/index.html for the correct "
            f"endpoint and update _DATA_TYPE_PATHS in taiwan_wra_iot.py."
        ) from last_error

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
                lat = rec.get("Latitude") or rec.get("lat") or rec.get("TWD97Lat")
                lon = rec.get("Longitude") or rec.get("lon") or rec.get("TWD97Lon")
                if lat and lon:
                    loc = GeoLocation(latitude=float(lat), longitude=float(lon))

                time_str = (
                    rec.get("RecordTime")
                    or rec.get("ObservationTime")
                    or rec.get("DateTime")
                    or ""
                )
                sample_dt = datetime.fromisoformat(time_str) if time_str else datetime.utcnow()

                samples.append(
                    WaterQualitySample(
                        source=DataSource.TAIWAN_WRA_IOT,
                        station_id=str(
                            rec.get("StationNo")
                            or rec.get("StationIdentifier")
                            or rec.get("ID")
                            or "unknown"
                        ),
                        station_name=rec.get("StationName") or rec.get("stationName"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=param_name,
                        value=float(value_str),
                        unit=unit,
                        county=rec.get("County") or rec.get("county"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping WRA IoT record: %s", exc)

        return samples
