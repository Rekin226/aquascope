"""
Collector for Canada - HYDAT / Environment and Climate Change Canada.

Reads daily mean discharge from the MSC GeoMet OGC API, which serves the
HYDAT archive. No API key is required.

API docs: https://eccc-msc.github.io/open-data/msc-geomet/ogc_api_en/
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from aquascope.collectors.base import BaseCollector, CollectorError
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    Quality,
    StreamflowReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.weather.gc.ca"
DAILY_MEAN_PATH = "/collections/hydrometric-daily-mean/items"
PAGE_SIZE = 1000

# HYDAT flags unusual daily values with a text symbol (DISCHARGE_SYMBOL_EN).
# Only "Ice Conditions" has been checked against real data so far; extend this
# table after looking at more records. Unmapped symbols become Quality.UNKNOWN.
_QUALITY_BY_SYMBOL = {
    "Ice Conditions": Quality.ESTIMATED,
}


class CanadaHydatCollector(BaseCollector):
    """Collect daily mean discharge from Canada's HYDAT archive via GeoMet."""

    name = "canada_hydat"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=BASE_URL,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def fetch_raw(
        self,
        station_id: str | None = None,
        datetime_start: str | None = None,
        datetime_end: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        max_items: int | None = None,
        **kwargs,
    ) -> list[dict]:
        """Fetch GeoJSON features of daily mean values, following pagination.

        Args:
            station_id: HYDAT station number, e.g. ``"05BB001"``.
            datetime_start: First day, ``YYYY-MM-DD``. Omit for an open start.
            datetime_end: Last day, ``YYYY-MM-DD``. Omit for an open end.
            bbox: ``(west, south, east, north)`` in WGS84 degrees.
            max_items: Stop after this many features.

        Raises:
            CollectorError: When a request fails.
        """
        params: dict[str, str | int] = {"f": "json", "limit": PAGE_SIZE}
        offset = 0

        station = (station_id or "").strip()
        if station:
            params["STATION_NUMBER"] = station
        if datetime_start or datetime_end:
            params["datetime"] = f"{datetime_start or '..'}/{datetime_end or '..'}"
        if bbox is not None:
            params["bbox"] = ",".join(str(v) for v in bbox)

        features: list[dict] = []
        while True:
            params["offset"] = offset
            try:
                data = self.client.get_json(DAILY_MEAN_PATH, params=params)
            except Exception as exc:
                url = f"{BASE_URL}{DAILY_MEAN_PATH}"
                raise CollectorError(
                    f"Canada HYDAT fetch failed for {url} (offset {params['offset']}): {exc}",
                    source=self.name,
                    url=url,
                    cause=exc,
                ) from exc

            page = data.get("features", [])
            features.extend(page)

            if len(page) < PAGE_SIZE:
                break  # a short page is the last page
            if max_items is not None and len(features) >= max_items:
                break
            offset += PAGE_SIZE

        if max_items is not None:
            features = features[:max_items]
        return features

    def normalise(self, raw: list[dict]) -> Sequence[StreamflowReading]:
        """Transform GeoJSON features into StreamflowReading objects."""
        readings: list[StreamflowReading] = []
        invalid = 0

        for row in raw:
            try:
                properties = row["properties"]

                # Expected, not an error: some stations only measure water level.
                discharge = properties.get("DISCHARGE")
                if discharge is None:
                    logger.debug("Skipping row: no discharge")
                    continue

                # The API stores float32, so 4.98 arrives as 4.980000019073486.
                # Six significant figures removes that noise.
                discharge_cms = float(f"{float(discharge):.6g}")

                # GeoJSON order is [longitude, latitude]. Location is optional in
                # the schema, so a missing geometry must not discard a good value.
                coordinates = (row.get("geometry") or {}).get("coordinates")
                location = None
                if coordinates:
                    location = GeoLocation(latitude=coordinates[1], longitude=coordinates[0])

                # No symbol means HYDAT did not flag the value. For an archive that
                # is the published, reviewed record, so it is treated as approved.
                symbol = properties.get("DISCHARGE_SYMBOL_EN")
                if symbol is None:
                    quality = Quality.APPROVED
                else:
                    quality = _QUALITY_BY_SYMBOL.get(symbol, Quality.UNKNOWN)

                readings.append(
                    StreamflowReading(
                        source=DataSource.CANADA_HYDAT,
                        station_id=properties["STATION_NUMBER"],
                        station_name=properties.get("STATION_NAME"),
                        location=location,
                        reading_datetime=properties["DATE"],
                        discharge_cms=discharge_cms,
                        source_type="in_situ",
                        quality=quality,
                        quality_raw=symbol,
                    )
                )
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                invalid += 1
                logger.debug("Skipping row: %s", exc)

        if invalid:
            logger.warning("[%s] Skipped %d malformed record(s) out of %d.", self.name, invalid, len(raw))
        return readings
