"""
Collector for NOAA NWPS data.

API docs: https://api.water.noaa.gov/nwps/v1/docs/
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, StreamflowReading
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

NWPS_BASE = "https://api.water.noaa.gov/nwps/v1"

class TallyLogger:
    def __init__(self):
        self.logger = logger
        self.counter = Counter()

    def add(self, key):
        self.counter[key] += 1

    def flush(self):
        for key, total in self.counter.items():
            self.logger.warning("[Repetitive Warning] '%s' occurred %d times.", key, total)
        self.counter.clear()

class NOAANWPSCollector(BaseCollector):
    name = "noaa_nwps"

    def __init__(
        self,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=NWPS_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
            )
        )

    def _build_observed_url(self, lid: str) -> str:
        """
        Build a NOAA NWPS stageflow endpoint URL for the supplied lid.
        Parameters
        ----------
        lid: str
            Identifier associated with a gauge station.
        """
        if (len(lid) != 5):
            raise ValueError("NOAA-NWPS: LID must be exactly 5 characters long.")

        return f"{NWPS_BASE}/gauges/{lid}/stageflow/observed"

    def _parse_bbox(self, value: Any) -> tuple[float, float, float, float] | None:
        """
        Convert a bbox string or sequence into a 4-float tuple.
        Returns None if the input value is invalid or cannot be converted.
        """
        if value in (None, ""):
            return None

        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, Sequence):
            parts = list(value)
        else:
            return None

        if len(parts) != 4:
            return None

        try:
            west, south, east, north = (float(part) for part in parts)
        except (TypeError, ValueError):
            return None

        return west, south, east, north

    def _build_discovery_url(
        self,
        bbox: tuple[float, float, float, float],
        srid: str = "EPSG_4326",
        catfim: bool = False,
    ) -> str:
        """
        Build the NOAA NWPS /gauges endpoint URL for searching using bbox.
        Spatial reference system ID for input bbox is "EPSG_4326".
        Parameters
        ----------
        bbox: tuple[float, float, float, float]  |  Required
            Bounding box for filtering gauges in the format (xmin, ymin, xmax, ymax).
            Correspond to west, south, east, and north coordinates respectively.
        catfim: bool      |  Optional
            Filter gauges based on CatFIM configuration. Default is False.
        """
        params: list[str] = []

        if bbox is not None and len(bbox) == 4:
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

            if bbox_xmin > bbox_xmax:
                raise ValueError("NOAA-NWPS: bbox.xmin cannot be greater than bbox.xmax.")
            if bbox_ymin > bbox_ymax:
                raise ValueError("NOAA-NWPS: bbox.ymin cannot be greater than bbox.ymax.")

            # NOAA NWPS usage here is constrained to western/northern hemispheres.
            if bbox_xmin > 0 or bbox_xmax > 0:
                raise ValueError("NOAA-NWPS: bbox x values (xmin/xmax) must be non-positive.")
            if bbox_ymin < 0 or bbox_ymax < 0:
                raise ValueError("NOAA-NWPS: bbox y values (ymin/ymax) must be non-negative.")

            params.append(f"bbox.xmin={bbox_xmin}")
            params.append(f"bbox.ymin={bbox_ymin}")
            params.append(f"bbox.xmax={bbox_xmax}")
            params.append(f"bbox.ymax={bbox_ymax}")
        else:
            raise ValueError("NOAA-NWPS: bbox search parameters require 4 values (xmin, ymin, xmax, ymax)")

        if catfim:
            params.append("catfim=true")

        if srid:
            params.append(f"srid={srid}")

        url = f"{NWPS_BASE}/gauges"

        if params:
            url = f"{url}?{'&'.join(params)}"
        return url

    def _extract_lid(self, discovery_raw: Any) -> list[str]:
        """
        Extract the LID from the discovery raw data.
        Parameters
        ----------
        discovery_raw: Any  |  Required
            Raw discovery data from the NOAA_NWPS /gauges endpoint.
        """
        if not isinstance(discovery_raw, dict):
            logger.warning("NOAA-NWPS: Discovery raw data is not a dictionary.")
            return []

        gauges = discovery_raw.get("gauges", [])
        if not isinstance(gauges, list):
            logger.warning("NOAA-NWPS: 'gauges' key is not a list in discovery raw data.")
            return []

        results: list[str] = []
        for gauge in gauges:
            if not isinstance(gauge, dict):
                continue
            lid = gauge.get("lid")
            if isinstance(lid, str) and lid:
                results.append(lid)
        return results

    def _build_gauge_url(self, lid: str) -> str:
        """
        Build the NOAA NWPS /gauges/{lid} endpoint URL to get the station_id, geolocation, and station_name.
        """
        if not isinstance(lid, str) or len(lid) != 5:
            raise ValueError("NOAA-NWPS: LID must be exactly 5 characters long.")
        return f"{NWPS_BASE}/gauges/{lid}"

    def _build_combined_dictionary(self, lid: str) -> dict[str, Any]:
        """
        Pull LID data from /gauges/{lid} and /gauges/{lid}/observed into a combined dictionary.
        This dictionary will as complete as possible and be used to build a StreamflowReading object.

        Contents
        --------
        source
        station_id
        location
        data [ This will be used to populate measurements, dates and times, and flow rates]
        """

        data_dictionary: dict[str, Any] = {}

        gauge_url = self._build_gauge_url(lid)
        observed_url = self._build_observed_url(lid)

        gauge_data = self.client.get_json(gauge_url)
        observed_data = self.client.get_json(observed_url)

        latitude = gauge_data["latitude"]
        longitude = gauge_data["longitude"]

        if latitude is None or longitude is None:
            latitude = 0.00
            longitude = 0.00
            logger.warning("NOAA-NWPS: Missing latitude or longitude in gauge data, defaulting to 0.00.")
        location = (
            GeoLocation(latitude=float(latitude), longitude=float(longitude))
        )

        data_dictionary["source"] = DataSource.NOAA_NWPS
        data_dictionary["station_id"] = gauge_data["lid"]
        data_dictionary["station_name"] = gauge_data["name"]
        data_dictionary["location"] = location
        data_dictionary["data_container"] = observed_data.get("data", [])
        data_dictionary["source_type"] = "in_situ"
        data_dictionary["unit"] = observed_data.get("secondaryUnits")

        return data_dictionary

    def _to_discharge_cms(self, value: Any, unit: str | None) -> float | None:
        """
        Convert NWPS flow values to cubic meters per second.
        """
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if unit is None:
            return None

        if unit == "kcfs":
            return numeric * 28.316846592
        if unit == "cfs":
            return numeric * 0.028316846592
        if unit == "m3/s" or unit == "cms":
            return numeric
        else:
            logger.warning("NOAA-NWPS: Unknown NWPS unit, skipping entry.")
            return None

    @staticmethod
    def _parse_nwps_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            return None

        text = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).replace(tzinfo=None)
        except ValueError:
            return None

    def fetch_raw(
        self,
        *,
        lid: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        srid: str = "EPSG_4326",
        catfim: bool = False,
        **_: Any,
    ) -> Any:
        """
        Fetch NWPS payload for one lid or enumerate a list of lids discovered by bbox.
        The first 5 LIDs will be iterated through _build_combined_data_dictionary
        """
        if lid and bbox:
            raise ValueError("NOAA-NWPS: Provide only one of 'lid' or 'bbox'.")
        if not lid and not bbox:
            raise ValueError("NOAA-NWPS: One of 'lid' or 'bbox' is required.")

        if lid:
            return self._build_combined_dictionary(lid)
        if bbox:
            discovery_url = self._build_discovery_url(bbox=bbox, srid=srid, catfim=catfim)
            discovery_raw = self.client.get_json(discovery_url)
            lids_list = self._extract_lid(discovery_raw)

            if not lids_list:
                logger.warning("NOAA-NWPS: No stations found in discovery for bbox %r", bbox)
                return []

            lids_list = lids_list[:5]  # Limit to 5 LIDs, 2 calls per LID @ 10 requests per minute
            for lid in lids_list:
                logger.info(f"Fetching data for LID: {lid}")

            return [self._build_combined_dictionary(lid) for lid in lids_list]

    def _normalise_single(self, data_dictionary: dict[str, Any]) -> list[StreamflowReading]:
        """Convert one station dictionary into StreamflowReading records."""
        source = data_dictionary.get("source", DataSource.NOAA_NWPS)
        station_id = data_dictionary.get("station_id", "")
        station_name = data_dictionary.get("station_name")
        location = data_dictionary.get("location")
        source_type = data_dictionary.get("source_type", "in_situ")
        unit = data_dictionary.get("unit")

        records: list[StreamflowReading] = []

        data_container = data_dictionary.get("data_container", [])

        tally = TallyLogger()

        if not isinstance(data_container, list):
            return records

        if unit is None or unit == "":
            tally.add("Unknown unit of measurement.")

        for entry in data_container:
            if not isinstance(entry, dict):
                continue

            if entry.get("secondary") in (-999, None):
                tally.add("Entry in data_container missing secondary measurement, skipping entry.")
                continue

            reading_datetime = self._parse_nwps_datetime(entry.get("validTime"))
            if reading_datetime is None:
                continue

            discharge_cms = self._to_discharge_cms(entry.get("secondary"), unit)
            if discharge_cms is None:
                continue

            records.append(
                StreamflowReading(
                    source=source,
                    station_id=station_id,
                    station_name=station_name,
                    location=location,
                    reading_datetime=reading_datetime,
                    discharge_cms=discharge_cms,
                    source_type=source_type,
                    unit="m3/s",
                )
            )

        tally.flush()
        return records

    def normalise(self, data_dictionary: dict[str, Any]) -> Sequence[StreamflowReading]:
        """
        Convert NWPS observed entries into StreamflowReading records.
        """
        if isinstance(data_dictionary, list):
            records: list[StreamflowReading] = []
            for station in data_dictionary:
                if isinstance(station, dict):
                    records.extend(self._normalise_single(station))
            return records

        if not isinstance(data_dictionary, dict):
            return []

        return self._normalise_single(data_dictionary)
