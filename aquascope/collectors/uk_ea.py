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
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterQualitySample
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

UKEA_BASE = "https://environment.data.gov.uk/hydrology"

_PARAMETER_LABELS: dict[str, str] = {
    "flow": "Flow",
    "waterlevel": "Level",
    "rainfall": "Rainfall",
    "groundwaterlevel": "Groundwater Level",
    "dissolved-oxygen": "Dissolved Oxygen",
    "fdom": "Fluorescent Dissolved Organic Matter",
    "bga": "Blue-Green Algae",
    "turbidity": "Turbidity",
    "chlorophyll": "Chlorophyll",
    "conductivity": "Conductivity",
    "temperature": "Temperature",
    "ammonium": "Ammonium",
    "nitrate": "Nitrate",
    "ph": "PH",
}

_PARAMETER_UNITS: dict[str, str] = {
    "flow": "m3/s",
    "waterlevel": "m",
    "rainfall": "mm",
    "groundwaterlevel": "mAOD",
    "dissolved-oxygen": "%",
    "fdom": "RFU",
    "bga": "RFU",
    "turbidity": "NTU",
    "chlorophyll": "µg/L",
    "conductivity": "µS/cm",
    "temperature": "oC",
    "ammonium": "mg/L",
    "nitrate": "mg/L",
    "ph": "",
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
        measure: str | None = None,
        station: str | None = None,
        station_wiski_id: str | None = None,
        observed_property: str | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        days: int | None = None,
        limit: int = 1000,
        max_items: int | None = 100_000,
        **kwargs,
    ) -> list[dict]:
        """Fetch readings from the UK EA Hydrology API."""
        if not any([measure, station, station_wiski_id, observed_property]):
            raise ValueError(
                "At least one of measure, station, station_wiski_id or observed_property must be provided."
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
        if measure:
            params["measure"] = measure
        if station:
            params["station"] = station
        if station_wiski_id:
            params["station.wiskiID"] = station_wiski_id
        if observed_property:
            params["observedProperty"] = observed_property
        if min_date:
            params["min-date"] = min_date
        if max_date:
            params["max-date"] = max_date
        params.update(kwargs)

        station_meta = None
        if station or station_wiski_id or measure:
            station_id = station or self._extract_station_guid_from_measure_id(measure)
            if station_id:
                station_meta = self._fetch_station_metadata(
                    station=station_id if station else None,
                    station_wiski_id=station_wiski_id,
                )

        all_items: list[dict] = []
        offset = 0
        while True:
            params["_offset"] = offset
            try:
                data = self.client.get_json("data/readings.json", params=params)
            except Exception as exc:
                logger.error("UK EA fetch failed: %s", exc)
                return []

            page_items = data.get("items", []) or []
            if not page_items:
                break

            if station_meta is not None:
                for item in page_items:
                    item["_station"] = station_meta

            all_items.extend(page_items)

            if max_items is not None and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                break

            if len(page_items) < limit:
                break

            offset += limit

        return all_items

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for item in raw:
            try:
                measure = item.get("measure", {}) or {}
                measure_id = measure.get("@id", "") or ""
                measure_name = measure_id.rsplit("/", 1)[-1] if measure_id else ""
                station_guid = self._extract_station_id_from_measure_name(measure_name)

                raw_parameter_key = measure.get("parameter") or self._parameter_from_measure_name(measure_name)
                parameter_key = raw_parameter_key.lower() if raw_parameter_key else ""
                parameter = _PARAMETER_LABELS.get(parameter_key, parameter_key.replace("-", " ").title())
                if parameter.lower() == "ph":
                    parameter = "PH"
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
                basin = None
                river = None
                if station_meta:
                    station_name = station_meta.get("label")
                    basin = station_meta.get("riverName")
                    river = station_meta.get("riverName")
                    lat = station_meta.get("lat")
                    lon = station_meta.get("long")
                    if lat is not None and lon is not None:
                        try:
                            location = GeoLocation(latitude=float(lat), longitude=float(lon))
                        except (ValueError, TypeError):
                            location = None

                unit = _PARAMETER_UNITS.get(parameter_key, "")
                remark = item.get("quality") or item.get("qualifier")

                samples.append(
                    WaterQualitySample(
                        source=DataSource.UK_EA,
                        station_id=station_meta.get("stationGuid", station_guid or "unknown"),
                        station_name=station_name,
                        location=location,
                        sample_datetime=sample_datetime,
                        parameter=parameter,
                        value=float(value),
                        unit=unit,
                        basin=basin,
                        river=river,
                        remark=remark,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping UK EA item: %s", exc)
        return samples

    def _fetch_station_metadata(
        self,
        station: str | None = None,
        station_wiski_id: str | None = None,
    ) -> dict | None:
        if not station and not station_wiski_id:
            return None

        params: dict[str, Any] = {}
        if station:
            params["stationGuid"] = station
        if station_wiski_id:
            params["wiskiID"] = station_wiski_id

        try:
            data = self.client.get_json("id/stations.json", params=params)
        except Exception as exc:
            logger.warning("Failed to fetch UK EA station metadata: %s", exc)
            return None

        items = data.get("items", []) or []
        return items[0] if items else None

    @staticmethod
    def _extract_station_guid_from_measure_id(measure: str | None) -> str | None:
        if not measure:
            return None
        name = measure.rsplit("/", 1)[-1]
        if len(name) >= 36 and name[36] == "-":
            return name[:36]
        return None

    @staticmethod
    def _extract_station_id_from_measure_name(measure_name: str) -> str | None:
        if len(measure_name) >= 36 and measure_name[36] == "-":
            return measure_name[:36]
        return None

    @staticmethod
    def _parameter_from_measure_name(measure_name: str) -> str:
        if not measure_name or len(measure_name) <= 36:
            return ""
        remainder = measure_name[36:].lstrip("-")
        parts = remainder.split("-")
        return parts[0] if parts else ""
