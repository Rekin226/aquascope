"""
Collector for the UK Environment Agency.

API docs: https://environment.data.gov.uk/hydrology/doc/reference
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
import datetime

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

UKEA_BASE = "https://environment.data.gov.uk/hydrology/"

PARAM_LABELS: dict[str, str] = {
    "waterFlow": "flow-m-86400-m3s-qualified",
    "waterLevel": "level-i-900-m-qualified",
    "rainfall": "rainfall-t-86400-mm-qualified",
    "groundwaterLevel": "gw-logged-i-subdaily-mAOD-qualified"
}

class UKEnvironmentAgencyCollector(BaseCollector):
    """Collect water data from the UK Environment Agency's Hydrology API.

    Parameters
    ----------
    api_key : str | None
        UK Environment Agency offers open data without authentication.
        Kept for interface parity with other collectors.
    """

    name = "uk_environment_agency"

    def __init__(self, api_key: str | None = None, client: CachedHTTPClient | None = None):
        super().__init__(
            client or CachedHTTPClient(
                base_url=UKEA_BASE,
                rate_limiter=RateLimiter(max_calls=25, period_seconds=60),
            )
        )
        self.api_key = api_key

    def fetch_raw(
            self,
            suid_station_code: str | None = None,
            water_quality_station_code: str | None = None,
            wiski_identifier: str | None = None,
            measure: str | None = None,
            min_date: str | None = None,
            max_date: str | None = None,
            limit: int | None = None,
            offset: int | None = None,
            observed_property: str | None = None,
            **kwargs
        ) -> list[dict]:
        """Fetch raw data from the API.
        
        Parameters
        ----------
        suid_station_code : str
            A GUID-style identifier known as an SUID (Station Unique IDentifier)
            Used as the primary identifier for monitoring stations
            (e.g. "052d0819-2a32-47df-9b99-c243c9c8235b").
        # water_quality_station_code : str
        #     An alternative, shorter form of station code used as the primary
        #     identifier for water quality stations.
        #     (e.g. "").
        wiski_identifier : str
            An string identifier used to disambiguate stations with the same SUID.
            Ambiguity arises when multiple distinct sampling points are
            co-located at the same physical station. Passing a suid_station_code
            without a wiski_identifier will return all co-located sampling points.
            By passing both a suid_station_code and a wiski_identifier as a composite
            key, you can retrieve data for a specific sampling point.
            (e.g. "037048U").
        measure: str
            The type of measurement to retrieve from the API for a station
            (e.g. "waterFlow", "waterLevel", "rainfall", "groundwaterLevel")
        limit : int
            API calls have a soft limit of 100_000 rows, with a hard limit of 2_000_000 rows
        offset: int
            The offset denoting the first row to return 
        observed_property : str
            The property to observe.
            (waterFlow | waterLevel | rainfall | groundWaterLevel)
        """
        if limit is not None and limit > 2_000_000:
            logger.warning("Limit exceeds hard cap of 2,000,000 rows. Setting limit to 2,000,000...")
        params = {
            "_format": "json",
        }
        if suid_station_code:
            url = f"id/stations/{suid_station_code}"
        else:
            url = "id/stations/"
        data = self.client.get_json(url, params=params)
        return data.get("records", [])

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """Transform raw API records into WaterQualitySample objects."""
        samples = []
        for row in raw:
            try:
                samples.append(
                    WaterQualitySample(
                        source=DataSource.UK_ENVIRONMENT_AGENCY,
                        station_id=row["station_id"],
                        station_name=row.get("name"),
                        sample_datetime=datetime.fromisoformat(
                            row["datetime"].replace("Z", "+00:00")
                        ).replace(tzinfo=None),
                        parameter=row["parameter"],
                        value=float(row["value"]),
                        unit=row.get("unit", ""),
                    )
                )
                samples.append(
                    WaterLevelReading(
                        source=DataSource.UK_ENVIRONMENT_AGENCY,
                        station_id=row["station_id"],
                        station_name=row.get("name"),
                        sample_datetime=datetime.fromisoformat(
                            row["datetime"].replace("Z", "+00:00")
                        ).replace(tzinfo=None),
                        parameter=row["parameter"],
                        value=float(row),
                        unit=row.get("unit", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping row: %s", exc)
        return samples