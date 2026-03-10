"""
Collector for USGS (United States Geological Survey) water data.

Uses the new OGC-compliant API:
    https://api.waterdata.usgs.gov/ogcapi/v0/

Collections
-----------
- ``daily``       — daily-value statistics (mean, min, max)
- ``sta``         — continuous (instantaneous) sensor readings
- ``discrete``    — discrete field measurements
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

USGS_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

# Common USGS parameter codes relevant to water quality
PARAM_LABELS: dict[str, str] = {
    "00010": "Temperature",
    "00060": "Discharge",
    "00065": "Gage height",
    "00095": "Conductivity",
    "00300": "DO",
    "00400": "pH",
    "00410": "Alkalinity",
    "00600": "TN",
    "00665": "TP",
    "00680": "TOC",
    "00940": "Chloride",
    "00945": "Sulfate",
    "71846": "NH3-N",
    "80154": "SS",
}


class USGSCollector(BaseCollector):
    """
    Collect daily-value water data from USGS via OGC API.

    Parameters
    ----------
    api_key : str
        Optional USGS API key for higher rate limits (get one at
        https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys).
    """

    name = "usgs"

    def __init__(
        self,
        api_key: str = "DEMO_KEY",
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=USGS_BASE,
                rate_limiter=RateLimiter(max_calls=25, period_seconds=60),
            )
        )
        self.api_key = api_key

    def fetch_raw(
        self,
        collection: str = "daily",
        datetime_range: str = "P30D",
        limit: int = 500,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch features from a USGS OGC collection.

        Parameters
        ----------
        collection : str
            ``"daily"`` | ``"sta"`` | ``"discrete"``
        datetime_range : str
            ISO 8601 duration or interval, e.g. ``"P7D"`` for last 7 days.
        limit : int
            Max features per page.
        """
        all_features: list[dict] = []
        params: dict[str, Any] = {
            "f": "json",
            "limit": limit,
            "datetime": datetime_range,
            "api_key": self.api_key,
        }

        url = f"collections/{collection}/items"
        while True:
            data = self.client.get_json(url, params=params)
            features = data.get("features", [])
            all_features.extend(features)

            # follow pagination
            next_link = next(
                (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            if not next_link or len(features) == 0:
                break
            # next_link is absolute; switch to direct fetch
            url = next_link
            params = {}

        return all_features

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for feat in raw:
            try:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [None, None]) if geom else [None, None]

                param_code = props.get("parameter_code", "")
                param_label = PARAM_LABELS.get(param_code, param_code)

                val = props.get("value")
                if val is None:
                    continue

                loc = None
                if coords[0] is not None:
                    loc = GeoLocation(latitude=coords[1], longitude=coords[0])

                samples.append(
                    WaterQualitySample(
                        source=DataSource.USGS,
                        station_id=props.get("monitoring_location_id", "unknown"),
                        location=loc,
                        sample_datetime=datetime.fromisoformat(props["time"]),
                        parameter=param_label,
                        value=float(val),
                        unit=props.get("unit_of_measure", ""),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping USGS feature: %s", exc)
        return samples
