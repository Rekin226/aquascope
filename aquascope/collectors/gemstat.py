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

    # Core parameters included by default — covers the most common water quality metrics
    DEFAULT_PARAMETERS = [
        "pH",
        "Temperature",
        "Dissolved_Gas",
        "Oxygen_Demand",
        "Other_Nitrogen",
        "Phosphorus",
        "Optical",
        "Electrical_Conductance",
    ]

    def fetch_raw(
        self,
        country: str | None = None,
        max_records: int = 5_000,
        parameters: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Download the GEMStat Zenodo archive (once), then join station metadata
        with observation rows and return filtered results.

        The ZIP (~200 MB) is cached to ``data/cache/`` on first call;
        subsequent calls load from the local file and are fast.

        Parameters
        ----------
        country : str, optional
            Full or partial country name (e.g. ``"Germany"``, ``"Canada"``).
            Case-insensitive substring match against the station metadata.
            GEMStat covers ~42 countries — Taiwan is not included.
        max_records : int
            Hard cap on returned rows across all parameters (default 5 000).
        parameters : list[str], optional
            Parameter CSV names without ``.csv`` (e.g. ``["pH", "Temperature"]``).
            Defaults to :attr:`DEFAULT_PARAMETERS`.
        start_date : str, optional
            ISO date ``"YYYY-MM-DD"`` — only include rows on or after this date.
        end_date : str, optional
            ISO date ``"YYYY-MM-DD"`` — only include rows on or before this date.
        """
        import csv
        import hashlib
        import io
        import zipfile
        from pathlib import Path

        import httpx

        # 1. Zenodo metadata → download URL
        record = self.client.get_json(GEMSTAT_ZENODO_API)
        files = record.get("files", [])
        if not files:
            logger.warning("GEMStat: no files found in Zenodo record")
            return []

        zip_entry = next((f for f in files if f["key"].endswith(".zip")), files[0])
        download_url = zip_entry["links"]["self"]
        checksum = zip_entry.get("checksum", download_url)
        size_mb = round(zip_entry.get("size", 0) / 1_048_576)

        # 2. Local ZIP cache (~200 MB, download once)
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(checksum.encode()).hexdigest()
        zip_path = cache_dir / f"gemstat_{cache_key}.zip"

        if not zip_path.exists():
            logger.info("GEMStat: downloading archive (%d MB) — one-time download…", size_mb)
            with httpx.stream("GET", download_url, follow_redirects=True, timeout=600) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
            logger.info("GEMStat: archive cached → %s", zip_path)
        else:
            logger.debug("GEMStat: using cached archive %s", zip_path)

        country_lower = country.lower().strip() if country else None
        date_start = start_date[:10] if start_date else None
        date_end = end_date[:10] if end_date else None

        with zipfile.ZipFile(zip_path) as zf:
            available = {n[:-4] for n in zf.namelist() if n.endswith(".csv")}

            # 3. Load station metadata: GEMS Station Number → {country, lat, lon, name}
            station_meta: dict[str, dict] = {}
            with zf.open("GEMStat_station_metadata.csv") as fh:
                for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
                    station_meta[row["GEMS Station Number"]] = {
                        "country": row.get("Country Name", ""),
                        "lat": row.get("Latitude", ""),
                        "lon": row.get("Longitude", ""),
                        "name": row.get("Station Identifier", row["GEMS Station Number"]),
                    }

            # 4. Determine valid station IDs for the requested country
            valid_ids: set[str] | None = None
            if country_lower:
                valid_ids = {sid for sid, m in station_meta.items() if country_lower in m["country"].lower()}
                if not valid_ids:
                    known = sorted({m["country"] for m in station_meta.values() if m["country"]})
                    logger.warning("GEMStat: no stations for country=%r. Available: %s", country, known)
                    return []
                logger.info("GEMStat: %d stations match country=%r", len(valid_ids), country)

            # 5. Read parameter CSVs and join with station metadata
            params_to_read = parameters or self.DEFAULT_PARAMETERS
            rows: list[dict] = []

            for param_name in params_to_read:
                if param_name not in available:
                    logger.debug("GEMStat: parameter file %s.csv not found", param_name)
                    continue
                with zf.open(f"{param_name}.csv") as fh:
                    reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
                    for row in reader:
                        sid = row.get("GEMS Station Number", "")
                        if valid_ids is not None and sid not in valid_ids:
                            continue
                        if date_start or date_end:
                            row_date = str(row.get("Sample Date", ""))[:10]
                            if date_start and row_date < date_start:
                                continue
                            if date_end and row_date > date_end:
                                continue
                        meta = station_meta.get(sid, {})
                        row["_country"] = meta.get("country", "")
                        row["_lat"] = meta.get("lat", "")
                        row["_lon"] = meta.get("lon", "")
                        row["_station_name"] = meta.get("name", sid)
                        rows.append(dict(row))
                        if len(rows) >= max_records:
                            logger.info("GEMStat: max_records=%d reached", max_records)
                            return rows

        logger.info("GEMStat: loaded %d rows (country=%s)", len(rows), country or "all")
        return rows

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        if not raw:
            return []

        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                val_str = row.get("Value", row.get("Analysis Result", ""))
                if not val_str or str(val_str).strip() in ("", "-", "ND", "---"):
                    continue

                loc = None
                lat, lon = row.get("_lat"), row.get("_lon")
                if lat and lon:
                    loc = GeoLocation(latitude=float(lat), longitude=float(lon))

                date_str = row.get("Sample Date", row.get("Date", ""))
                if not date_str:
                    continue
                sample_dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")

                samples.append(
                    WaterQualitySample(
                        source=DataSource.GEMSTAT,
                        station_id=row.get("GEMS Station Number", "unknown"),
                        station_name=row.get("_station_name"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=row.get("Parameter Code", row.get("Parameter", "unknown")),
                        value=float(val_str),
                        unit=row.get("Unit", ""),
                        county=row.get("_country", ""),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping GEMStat row: %s", exc)

        return samples

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
                if not date_str:
                    logger.debug("Skipping GEMStat CSV row without date")
                    continue
                sample_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")

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
