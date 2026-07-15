"""
Collector for the UK Environment Agency hydrology data service.

Uses the public Hydrology API:
    https://environment.data.gov.uk/hydrology
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterLevelReading, WaterQualitySample
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

UKEA_BASE = "https://environment.data.gov.uk/hydrology"

_PARAMETER_LABELS: dict[str, str] = {
    "flow": "Flow",
    "level": "Level",
    "rainfall": "Rainfall",
}

_PARAMETER_UNITS: dict[str, str] = {
    "flow": "m3/s",
    "rainfall": "mm",
    "waterlevel": "m",
    "groundwaterlevel": "mAOD",
}


class UKEACollector(BaseCollector):
    """Collect readings from the UK Environment Agency hydrology API."""

    name = "uk_ea"

    def __init__(
        self,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=UKEA_BASE,
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def fetch_raw(
        self,
        observed_property: str | None = None,
        measure: str | None = None,
        station: str | None = None,
        station_wiski_id: str | None = None,
        bbox: str | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        days: int | None = None,
        limit: int = 10_000,
        max_items: int | None = 2_000,
        **kwargs,
    ) -> list[dict]:
        """Fetch readings from the UK EA Hydrology API."""
        if not observed_property or observed_property not in _PARAMETER_UNITS.keys() and not measure:
            raise ValueError(
                "One of the following observedProperty values must be passed: "
                f"{', '.join(_PARAMETER_UNITS.keys())}."
            )

        if bbox:
            bounding_box_limits = UKEACollector._parse_bbox(bbox)
            if not bounding_box_limits:
                raise ValueError(
                    'Invalid bbox string. Must be a string of 4 comma-separated floats '
                    'in the form "min-lon,min-lat,max-lon,max-lat)". '
                    'For example, "2.0,51.1,3.3,52.7"',
                )

        # If min_date and max_date not provided, set date range to the last `days` days (default 30)
        if not min_date and not max_date:
            end = date.today()
            start = end - timedelta(days=days if days else 30)
            min_date = start.isoformat()
            max_date = end.isoformat()

        # If only min_date is provided, set max_date to `days` days (default 30) after min_date (or, if that exceeds today, set max_date to today)
        if min_date and not max_date:
            end = date.fromisoformat(min_date) + timedelta(days=days if days else 30)
            if end > date.today():
                end = date.today()
            max_date = end.isoformat()

        # If only max_date is provided, set min_date to `days` days (default 30) before max_date
        if not min_date and max_date:
            start = date.fromisoformat(max_date) - timedelta(days=days if days else 30)
            min_date = start.isoformat()

        if min_date and max_date and days:
            logger.warning(
                "Both min_date/max_date and days were provided. Ignoring days and using min_date/max_date range."
            )

        params: dict[str, Any] = {
            "_limit": limit,
        }
        if observed_property:
            params["observedProperty"] = observed_property
        if measure:
            params["measure"] = measure
        if station:
            params["station"] = station
        if station_wiski_id:
            params["station.wiskiID"] = station_wiski_id
        if bounding_box_limits:
            min_lon, min_lat, max_lon, max_lat = bounding_box_limits
            params["mineq-lon"] = min_lon
            params["mineq-lat"] = min_lat
            params["maxeq-lon"] = max_lon
            params["maxeq-lat"] = max_lat
        if min_date:
            params["mineq-date"] = min_date
        if max_date:
            params["maxeq-date"] = max_date
        params.update(kwargs)

        station_meta = None
        if station or station_wiski_id or measure:
            station_id = station or UKEACollector._extract_station_suid_from_measure_id(measure)
            if station_id:
                station_meta = self._fetch_station_metadata(
                    station=station_id if station else None,
                    station_wiski_id=station_wiski_id,
                )

        all_items: list[dict] = []
        all_items.append(params) # Metadata for use in normalise()
        offset = 0

        while True:
            params["_offset"] = offset
            try:
                data = self.client.get_json("data/readings.json", params=params)
            except Exception as exc:
                logger.error("UK EA fetch failed: %s", exc)
                return []

            page_items = data.get("items", [])
            if not page_items:
                break

            for item in page_items:
                if station_meta:
                    item["_station"] = station_meta

            all_items.extend(page_items)

            if max_items is not None and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                logger.debug("UKEA max_items=%d reached — stopping pagination.", max_items)
                break

            offset += limit

        return all_items

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample] | Sequence[WaterLevelReading]:
        if not raw:
            return []

        raw_request_metadata = raw[0]
        if raw_request_metadata.get("observedProperty") in ["flow", "rainfall"]:
            return self._normalise_water_quality_samples(raw[1:])
        elif raw_request_metadata.get("observedProperty") in ["level", "groundwaterlevel"]:
            return self._normalise_water_level_readings(raw[1:])

    def _normalise_water_quality_samples(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for item in raw:
            try:
                measure = item.get("measure", {}) or {}
                measure_id = measure.get("@id", "") or ""
                measure_name = measure_id.rsplit("/", 1)[-1] if measure_id else ""
                station_suid = UKEACollector._extract_station_suid_from_measure_name(measure_name)

                raw_parameter_key = measure.get("parameter")
                parameter_key = raw_parameter_key.lower() if raw_parameter_key else ""
                parameter = _PARAMETER_LABELS.get(parameter_key, parameter_key.replace("-", " ").title())

                if not parameter:
                    parameter = "unknown"

                value = item.get("value")
                if value is None or value == "":
                    continue

                datetime_str = item.get("dateTime") or item.get("date")
                if not datetime_str:
                    continue
                sample_datetime = datetime.fromisoformat(datetime_str)

                station_meta = item.get("_station") or {}
                location = None
                station_name = None
                river = None
                if station_meta:
                    station_name = station_meta.get("label")
                    river = station_meta.get("riverName")
                    lat = station_meta.get("lat")
                    lon = station_meta.get("long")
                    if lat is not None and lon is not None:
                        try:
                            location = GeoLocation(latitude=float(lat), longitude=float(lon))
                        except (ValueError, TypeError):
                            location = None

                unit = _PARAMETER_UNITS.get(parameter_key, "")
                data_quality_explanation = item.get("dataQualityMessage") or item.get("explanation") or "No Explanation Available"
                remark = (f"Data Quality: {item.get('category', 'N/A')};{data_quality_explanation}, "
                          f"Qualifier: {item.get('qualifier', 'N/A')}")

                samples.append(
                    WaterQualitySample(
                        source=DataSource.UK_EA,
                        station_id=station_meta.get("stationSuid", station_suid or "unknown"),
                        station_name=station_name,
                        location=location,
                        sample_datetime=sample_datetime,
                        parameter=parameter,
                        value=float(value),
                        unit=unit,
                        river=river,
                        remark=remark,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping UK EA item: %s", exc)
        return samples
    
    
    def _normalise_water_level_readings(self, raw: list[dict]) -> Sequence[WaterLevelReading]:
        ...

    def _fetch_station_metadata(
        self,
        station: str | None = None,
        station_wiski_id: str | None = None,
    ) -> dict | None:
        if not station and not station_wiski_id:
            return None

        params: dict[str, Any] = {}
        if station:
            params["stationSuid"] = station
        if station_wiski_id:
            params["wiskiID"] = station_wiski_id

        try:
            data = self.client.get_json("id/stations.json", params=params)
        except Exception as exc:
            logger.warning("Failed to fetch UK EA station metadata: %s", exc)
            return None

        items = data.get("items", []) or []
        if len(items) > 1:
            logger.warning(
                "Multiple UK EA stations found for station=%s, station_wiski_id=%s. First result used.",
                station,
                station_wiski_id,
            )
        return items[0] if items else None

    @staticmethod
    def _extract_station_suid_from_measure_id(
        measure: str | None
    ) -> str | None:
        if not measure:
            return None

        # Since the SUID is based on GUID style identifiers, the SUID is always the first 36 characters of the measure ID.
        return measure[:36]

    @staticmethod
    def _parse_bbox(value: str) -> tuple[float, float, float, float] | None:
        """Convert a bbox string or sequence into a 4-float tuple."""
        if not isinstance(value, str):
            return None

        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            return None

        try:
            min_lon, min_lat, max_lon, max_lat = (float(part) for part in parts)
        except (TypeError, ValueError):
            return None

        return min_lon, min_lat, max_lon, max_lat