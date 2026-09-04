"""
Collector for Taiwan 政府資料開放平台 (data.gov.tw).

Portal  : https://data.gov.tw
API     : https://opendata.wra.gov.tw/api/v2  (datasets re-hosted by WRA)
License : Open Government Data License v1.0 (free, attribution required)

Key datasets (data.gov.tw id → WRA API UUID):
  25768  → 73c4c3de-4045-4765-abeb-89f9f9cd5ff0  Real-time river water level (即時水位)
  161082 → 58a7aa39-287a-4b96-985d-47ffbc7abbd4  Real-time groundwater level (地下水位即時資料)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterLevelReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

DATAGOV_BASE = "https://opendata.wra.gov.tw/api/v2"

DATASET_WATER_LEVEL = "73c4c3de-4045-4765-abeb-89f9f9cd5ff0"
DATASET_GROUNDWATER = "58a7aa39-287a-4b96-985d-47ffbc7abbd4"
LEGACY_DATASET_IDS = {
    "25768": DATASET_WATER_LEVEL,
    "161082": DATASET_GROUNDWATER,
}


class TaiwanDataGovCollector(BaseCollector):
    """
    Collect real-time water level data from Taiwan's open government data
    platform (data.gov.tw).

    Parameters
    ----------
    dataset_id : str
        WRA Open Data UUID.  Use :data:`DATASET_WATER_LEVEL` (default) for
        real-time river water level or :data:`DATASET_GROUNDWATER` for
        real-time groundwater level.
    """

    name = "taiwan_datagov"

    def __init__(
        self,
        dataset_id: str = DATASET_WATER_LEVEL,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=DATAGOV_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=600,
                relax_strict_tls=True,
            )
        )
        self.dataset_id = LEGACY_DATASET_IDS.get(dataset_id, dataset_id)

    def fetch_raw(self, limit: int | None = None, offset: int = 0, **kwargs) -> list[dict]:
        """
        Fetch records from the WRA Open Data API for the configured dataset.

        The API returns the full dataset in one response; ``offset`` and an
        optional ``limit`` are applied client-side. By default all records are
        returned.
        """
        data = self.client.get_json(self.dataset_id, params={"format": "json"})
        records = data if isinstance(data, list) else data.get("result", data.get("records", []))
        if offset:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    def normalise(self, raw: list[dict]) -> Sequence[WaterLevelReading]:
        readings: list[WaterLevelReading] = []
        for rec in raw:
            try:
                level_str = (
                    rec.get("WaterLevel")
                    or rec.get("waterLevel")
                    or rec.get("waterlevel")
                    or rec.get("water_level")
                    or rec.get("GWLevel")
                    or rec.get("gwLevel")
                )
                if level_str is None or str(level_str).strip() in ("", "-", "--", "ND"):
                    continue

                loc = None
                lat = rec.get("Latitude") or rec.get("lat") or rec.get("TWD97Lat")
                lon = rec.get("Longitude") or rec.get("lon") or rec.get("TWD97Lon")
                if lat and lon:
                    try:
                        loc = GeoLocation(latitude=float(lat), longitude=float(lon))
                    except (ValueError, TypeError):
                        pass

                time_str = (
                    rec.get("RecordTime")
                    or rec.get("recordtime")
                    or rec.get("ObservationTime")
                    or rec.get("DateTime")
                    or rec.get("datetime")
                    or rec.get("time")
                    or ""
                )
                reading_dt = datetime.fromisoformat(time_str) if time_str else datetime.utcnow()

                readings.append(
                    WaterLevelReading(
                        source=DataSource.TAIWAN_DATAGOV,
                        station_id=str(
                            rec.get("StationIdentifier")
                            or rec.get("stationid")
                            or rec.get("StationNo")
                            or rec.get("wellidentifier")
                            or rec.get("SiteId")
                            or "unknown"
                        ),
                        station_name=(
                            rec.get("StationName")
                            or rec.get("name")
                            or rec.get("observatoryidentifier")
                            or rec.get("SiteName")
                        ),
                        location=loc,
                        reading_datetime=reading_dt,
                        water_level=float(level_str),
                        unit="m",
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping data.gov.tw record: %s", exc)

        return readings
