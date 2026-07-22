"""Collector for Germany's PEGELONLINE hydrometry service."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
SUPPORTED_TIMESERIES = {"W", "Q"}
MAX_HISTORY_DAYS = 31


class PegelonlineCollector(BaseCollector):
    """Collect recent water-level and discharge readings from PEGELONLINE.

    PEGELONLINE exposes raw measurements for only the most recent 31 days.
    Stations are addressed by UUID because station names and numbers are not
    guaranteed to be unique or stable.
    """

    name = "pegelonline"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=BASE_URL,
                cache_ttl_seconds=300,
                rate_limiter=RateLimiter(max_calls=30, period_seconds=60),
            )
        )

    @staticmethod
    def _timeseries_codes(timeseries: str | Sequence[str] | None) -> list[str]:
        if timeseries is None:
            return ["W", "Q"]
        values = timeseries.split(",") if isinstance(timeseries, str) else timeseries
        codes = list(dict.fromkeys(value.strip().upper() for value in values if value.strip()))
        unsupported = set(codes) - SUPPORTED_TIMESERIES
        if unsupported:
            raise ValueError(
                "PEGELONLINE timeseries must contain only 'W' (water level) "
                f"or 'Q' (discharge); got {sorted(unsupported)}."
            )
        if not codes:
            raise ValueError("At least one PEGELONLINE timeseries is required.")
        return codes

    @staticmethod
    def _measurement_params(*, days: int, start: str | None, end: str | None) -> dict[str, str]:
        if start is None:
            if not 1 <= days <= MAX_HISTORY_DAYS:
                raise ValueError(f"PEGELONLINE days must be between 1 and {MAX_HISTORY_DAYS}; got {days}.")
            params = {"start": f"P{days}D"}
        else:
            params = {"start": start}
        if end is not None:
            params["end"] = end
        return params

    def fetch_raw(
        self,
        station_id: str,
        timeseries: str | Sequence[str] | None = None,
        days: int = 30,
        start: str | None = None,
        end: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch station metadata and recent W/Q measurements.

        Args:
            station_id: PEGELONLINE station UUID.
            timeseries: ``W`` (water level), ``Q`` (discharge), or both.
            days: Relative history window when ``start`` is omitted (1-31).
            start: Optional API timestamp or ISO-8601 duration.
            end: Optional API timestamp.
        """
        if not station_id.strip():
            raise ValueError("PEGELONLINE station_id must not be empty.")

        codes = self._timeseries_codes(timeseries)
        params = self._measurement_params(days=days, start=start, end=end)
        station = self.client.get_json(
            f"/stations/{station_id.strip()}.json",
            params={"includeTimeseries": "true"},
        )
        station_uuid = station["uuid"]
        metadata = {
            item.get("shortname"): item
            for item in station.get("timeseries", [])
            if item.get("shortname") in SUPPORTED_TIMESERIES
        }

        series = []
        for code in codes:
            if code not in metadata:
                logger.info(
                    "PEGELONLINE station %s does not publish timeseries %s; skipping.",
                    station_uuid,
                    code,
                )
                continue
            measurements = self.client.get_json(
                f"/stations/{station_uuid}/{code}/measurements.json",
                params=params,
            )
            series.append(
                {
                    "shortname": code,
                    "unit": metadata[code].get("unit"),
                    "measurements": measurements,
                }
            )

        return {"station": station, "series": series}

    def normalise(self, raw: dict[str, Any]) -> Sequence[WaterLevelReading | StreamflowReading]:
        """Convert PEGELONLINE W/Q measurements to AquaScope records."""
        station = raw.get("station", {})
        location = self._location(station)
        station_id = station.get("uuid")
        station_name = station.get("longname") or station.get("shortname")
        river = (station.get("water") or {}).get("longname")
        remark = f"River: {river}" if river else None
        readings: list[WaterLevelReading | StreamflowReading] = []
        total = 0
        skipped = 0

        for series in raw.get("series", []):
            code = series.get("shortname")
            unit = self._normalise_unit(series.get("unit"), code)
            for row in series.get("measurements", []):
                total += 1
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                    value = float(row["value"])
                    if code == "W":
                        reading: WaterLevelReading | StreamflowReading = WaterLevelReading(
                            source=DataSource.PEGELONLINE,
                            station_id=station_id,
                            station_name=station_name,
                            location=location,
                            reading_datetime=timestamp,
                            water_level=value,
                            unit=unit,
                            remark=remark,
                        )
                    elif code == "Q":
                        reading = StreamflowReading(
                            source=DataSource.PEGELONLINE,
                            station_id=station_id,
                            station_name=station_name,
                            location=location,
                            reading_datetime=timestamp,
                            discharge_cms=value,
                            source_type="in_situ",
                            unit=unit,
                            remark=remark,
                        )
                    else:
                        raise ValueError(f"Unsupported timeseries {code!r}")
                    readings.append(reading)
                except (KeyError, TypeError, ValueError) as exc:
                    skipped += 1
                    logger.debug("Skipping PEGELONLINE measurement: %s", exc)

        if skipped:
            logger.warning("PEGELONLINE normalisation skipped %d/%d measurements.", skipped, total)
        return readings

    @staticmethod
    def _location(station: dict[str, Any]) -> GeoLocation | None:
        latitude = station.get("latitude")
        longitude = station.get("longitude")
        if latitude is None or longitude is None:
            return None
        try:
            return GeoLocation(latitude=float(latitude), longitude=float(longitude))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_unit(unit: str | None, code: str | None) -> str:
        if unit:
            return unit.replace("m³/s", "m3/s")
        return "cm" if code == "W" else "m3/s"
