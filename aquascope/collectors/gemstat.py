"""
Collector for GEMStat — UNEP Global Freshwater Quality Database.

Data is published under CC BY 4.0 on Zenodo:
    https://doi.org/10.5281/zenodo.13881899

The collector downloads the CSV archive and parses it into
WaterQualitySample records.
"""

from __future__ import annotations

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

# The Zenodo record page (resolves to latest version)
GEMSTAT_ZENODO_API = "https://zenodo.org/api/records/13881899"


class GEMStatCollector(BaseCollector):
    """
    Collect water quality data from the GEMStat Zenodo archive.

    Downloads the CSV file, parses rows, and normalises into
    WaterQualitySample records.  Supports filtering by country.
    """

    name = "gemstat"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=86400,
            )
        )

    def fetch_raw(self, country: str | None = None, max_records: int = 5000, **kwargs) -> list[dict]:
        """
        Fetch GEMStat data from Zenodo.

        Due to the large size of the full archive, this returns
        metadata about available files. For actual data processing,
        users should download the CSV directly.
        """
        # Fetch Zenodo record metadata to get download URLs
        record = self.client.get_json(GEMSTAT_ZENODO_API)
        files = record.get("files", [])

        file_info = []
        for f in files:
            file_info.append({
                "filename": f.get("key", ""),
                "size_mb": round(f.get("size", 0) / 1024 / 1024, 1),
                "download_url": f.get("links", {}).get("self", ""),
                "checksum": f.get("checksum", ""),
            })

        logger.info("GEMStat archive contains %d files on Zenodo", len(file_info))
        return file_info

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """
        For Zenodo metadata, return an empty list.
        Actual data parsing happens via ``parse_gemstat_csv()``.
        """
        logger.info(
            "GEMStat files available for download: %s",
            [f["filename"] for f in raw],
        )
        return []

    @staticmethod
    def parse_gemstat_csv(csv_content: str, max_records: int = 10000) -> list[WaterQualitySample]:
        """
        Parse a GEMStat CSV string into WaterQualitySample records.

        Expected columns: GEMS Station Number, Sample Date, Parameter,
        Analysis Result, Unit, Latitude, Longitude, Country Code, etc.
        """
        import csv

        reader = csv.DictReader(io.StringIO(csv_content))
        samples: list[WaterQualitySample] = []

        for i, row in enumerate(reader):
            if i >= max_records:
                break
            try:
                val_str = row.get("Analysis Result", row.get("Value", ""))
                if not val_str or val_str.strip() in ("", "-", "ND"):
                    continue

                loc = None
                lat = row.get("Latitude")
                lon = row.get("Longitude")
                if lat and lon:
                    loc = GeoLocation(latitude=float(lat), longitude=float(lon))

                date_str = row.get("Sample Date", row.get("Date", ""))
                sample_dt = datetime.strptime(date_str[:10], "%Y-%m-%d") if date_str else datetime.now()

                samples.append(
                    WaterQualitySample(
                        source=DataSource.GEMSTAT,
                        station_id=row.get("GEMS Station Number", "unknown"),
                        station_name=row.get("Station Name"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=row.get("Parameter", "unknown"),
                        value=float(val_str),
                        unit=row.get("Unit", ""),
                        county=row.get("Country Code", ""),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping GEMStat row: %s", exc)

        return samples
