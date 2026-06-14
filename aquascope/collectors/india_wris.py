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

BASE_URL = "https://indiawris.gov.in"


class IndiaWRISCollector(BaseCollector):
    """Collector for India WRIS river water level data."""

    name = "india_wris"

    def __init__(
        self,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=BASE_URL,
                rate_limiter=RateLimiter(
                    max_calls=25,
                    period_seconds=60,
                ),
            )
        )

    def fetch_raw(
        self,
        state_name: str,
        district_name: str,
        agency_name: str,
        startdate: str,
        enddate: str,
        page: int = 0,
        size: int = 100,
        **kwargs,
    ) -> list[dict]:

        data = self.client.post_json(
            "/Dataset/River Water Level",
            params={
                "stateName": state_name,
                "districtName": district_name,
                "agencyName": agency_name,
                "startdate": startdate,
                "enddate": enddate,
                "download": False,
                "page": page,
                "size": size,
            },
        )

        return data.get("data", [])

    def normalise(
        self,
        raw: list[dict],
    ) -> Sequence[WaterLevelReading]:

        readings = []

        for row in raw:
            try:
                readings.append(
                    WaterLevelReading(
                        source=DataSource.INDIA_WRIS,
                        station_id=row["stationCode"],
                        station_name=row["stationName"],
                        location=GeoLocation(
                            latitude=row["latitude"],
                            longitude=row["longitude"],
                        ),
                        reading_datetime=datetime.fromisoformat(
                            row["dataTime"]
                        ),
                        water_level=float(row["dataValue"]),
                        unit=row.get("unit", "m"),
                        remark=row.get("description"),
                    )
                )

            except (KeyError, ValueError, TypeError) as exc:
                logger.debug(
                    "Skipping India WRIS record: %s",
                    exc,
                )

        return readings
