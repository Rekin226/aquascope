"""
Collector for the US Water Quality Portal (WQP).

The WQP integrates data from USGS, EPA, and 400+ agencies with
430M+ records.

API docs : https://www.waterqualitydata.us/webservices_documentation/
Endpoint : https://www.waterqualitydata.us/data/Result/search
"""

from __future__ import annotations

import csv
import io
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

WQP_BASE = "https://www.waterqualitydata.us/data"


class WQPCollector(BaseCollector):
    """
    Collect discrete water quality data from the US Water Quality Portal.

    Supports filtering by state, county, characteristic (parameter),
    date range, and bounding box.
    """

    name = "wqp"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=WQP_BASE,
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def fetch_raw(
        self,
        state_code: str | None = None,
        characteristic_name: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        bbox: str | None = None,
        max_results: int = 1000,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch water quality results from WQP.

        Parameters
        ----------
        state_code : str | None
            e.g. ``"US:06"`` for California.
        characteristic_name : str | None
            e.g. ``"Dissolved oxygen (DO)"``, ``"pH"``
        start_date : str | None
            ``"MM-DD-YYYY"`` format.
        end_date : str | None
            ``"MM-DD-YYYY"`` format.
        bbox : str | None
            Bounding box: ``"west,south,east,north"`` in decimal degrees.
        max_results : int
            Limit number of results (WQP default returns CSV).
        """
        params = {
            "mimeType": "csv",
            "sorted": "no",
            "zip": "no",
        }
        if state_code:
            params["statecode"] = state_code
        if characteristic_name:
            params["characteristicName"] = characteristic_name
        if start_date:
            params["startDateLo"] = start_date
        if end_date:
            params["startDateHi"] = end_date
        if bbox:
            params["bBox"] = bbox

        # WQP returns CSV, so we need to handle it differently
        import httpx

        url = f"{WQP_BASE}/Result/search"
        try:
            resp = httpx.get(url, params=params, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            logger.error("WQP fetch failed: %s", exc)
            return []

        reader = csv.DictReader(io.StringIO(text))
        records = []
        for i, row in enumerate(reader):
            if i >= max_results:
                break
            records.append(dict(row))

        return records

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                val_str = row.get("ResultMeasureValue", "")
                if not val_str or val_str.strip() in ("", "-"):
                    continue

                loc = None
                lat = row.get("LatitudeMeasure")
                lon = row.get("LongitudeMeasure")
                if lat and lon:
                    try:
                        loc = GeoLocation(latitude=float(lat), longitude=float(lon))
                    except (ValueError, TypeError):
                        pass

                date_str = row.get("ActivityStartDate", "")
                time_str = row.get("ActivityStartTime/Time", "00:00:00")
                try:
                    sample_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        sample_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue

                samples.append(
                    WaterQualitySample(
                        source=DataSource.WQP,
                        station_id=row.get("MonitoringLocationIdentifier", "unknown"),
                        station_name=row.get("MonitoringLocationName"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=row.get("CharacteristicName", "unknown"),
                        value=float(val_str),
                        unit=row.get("ResultMeasure/MeasureUnitCode", ""),
                        county=row.get("CountyCode"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping WQP row: %s", exc)

        return samples
