"""
Collector for Taiwan Civil IoT Data Service Platform — Water Resources.

Uses OGC SensorThings API v1.1:
    https://sta.ci.taiwan.gov.tw/STA_WaterResource_v2/v1.1/

Provides real-time sensor data for:
- River water level stations (WRA)
- Flow sensors
- Rainfall sensors
- COD / SS / sewage discharge sensors (CAPMI)
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

CIVIL_IOT_BASE = "https://sta.ci.taiwan.gov.tw/STA_WaterResource_v2/v1.1"


class TaiwanCivilIoTCollector(BaseCollector):
    """
    Collect real-time water resource data from Taiwan's Civil IoT
    SensorThings API.

    The ``entity`` parameter is consumed by :meth:`fetch_raw`, not by
    ``__init__``; see that method's docstring for valid values.
    """

    name = "taiwan_civil_iot"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=CIVIL_IOT_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=600,
                verify=False,
            )
        )

    def fetch_raw(
        self,
        entity: str = "Datastreams",
        top: int = 100,
        expand: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch SensorThings entities.

        Parameters
        ----------
        entity : str
            ``"Things"`` | ``"Datastreams"`` | ``"Observations"``
        top : int
            Max items per page.
        expand : str, optional
            OData $expand clause. If not given, one is built from
            ``start_date`` / ``end_date``.
        start_date : str, optional
            ISO date string ``"YYYY-MM-DD"`` — filters ``phenomenonTime ge``.
        end_date : str, optional
            ISO date string ``"YYYY-MM-DD"`` — filters ``phenomenonTime le``.
        """
        if expand is None:
            time_parts = []
            if start_date:
                time_parts.append(f"phenomenonTime ge {start_date}T00:00:00Z")
            if end_date:
                time_parts.append(f"phenomenonTime le {end_date}T23:59:59Z")
            filter_clause = (";$filter=" + " and ".join(time_parts)) if time_parts else ""
            expand = f"Thing,Observations($top=5;$orderby=phenomenonTime desc{filter_clause})"
        all_items: list[dict] = []
        params = {"$top": top, "$expand": expand}

        url = entity
        while True:
            data = self.client.get_json(url, params=params)
            items = data.get("value", [])
            all_items.extend(items)

            next_link = data.get("@iot.nextLink")
            if not next_link or len(items) == 0:
                break
            url = next_link
            params = {}

            if len(all_items) >= top * 5:  # safety limit
                break

        return all_items

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """
        Normalise SensorThings Datastreams with their latest Observation.
        """
        samples: list[WaterQualitySample] = []
        for ds in raw:
            try:
                thing = ds.get("Thing", {})
                obs_list = ds.get("Observations", [])
                if not obs_list:
                    continue
                obs = obs_list[0]

                result = obs.get("result")
                if result is None:
                    continue

                # Extract location from Thing
                loc = None
                thing_loc = thing.get("Locations", [{}])
                if thing_loc:
                    coords = thing_loc[0].get("location", {}).get("coordinates", [])
                    if len(coords) >= 2:
                        loc = GeoLocation(latitude=float(coords[1]), longitude=float(coords[0]))

                phen_time = obs.get("phenomenonTime", "")
                if "T" in phen_time:
                    sample_dt = datetime.fromisoformat(phen_time.replace("Z", "+00:00"))
                else:
                    sample_dt = datetime.fromisoformat(phen_time)

                # Derive parameter name from Datastream description
                ds_name = ds.get("name", ds.get("description", "unknown"))
                unit = ds.get("unitOfMeasurement", {}).get("symbol", "")

                samples.append(
                    WaterQualitySample(
                        source=DataSource.TAIWAN_CIVIL_IOT,
                        station_id=str(thing.get("@iot.id", thing.get("name", "unknown"))),
                        station_name=thing.get("name"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=ds_name,
                        value=float(result),
                        unit=unit,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping Civil IoT record: %s", exc)

        return samples
