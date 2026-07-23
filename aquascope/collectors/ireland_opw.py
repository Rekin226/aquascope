"""
Collector for Ireland — OPW (Office of Public Works) hydrometric network.

Real-time river/lake water level from waterlevel.ie, at 15-minute
resolution from hundreds of stations nationwide:
    https://waterlevel.ie/page/api/

Two upstream resources are used:

- Station list: a single GeoJSON file with coordinates, station name/ref,
  and (per Rekin226's confirmation reviewing this collector) a
  ``csv_file`` property already carrying the correct per-station CSV path.
  https://waterlevel.ie/geojson/
- Per-station series: CSV, not JSON — the URL pattern is
  ``https://waterlevel.ie/data/month/<last-5-digits-of-ref>_0001.csv``
  (``0001`` = water level in metres; other suffixes are OD/temperature/
  battery voltage and are not used here). We prefer the station's own
  ``csv_file`` property when present, falling back to constructing the
  URL from the station ref otherwise.

No API key required (open data, unauthenticated). Only station refs in
1-41000 are suitable for republication per OPW's terms; refs above that
range are filtered out.

CSV column names are resolved defensively (a datetime-like and a
value-like column, picked from a candidate list) rather than hardcoded,
since waterlevel.ie serves at least two differently-shaped CSV exports
(a raw per-reading series and a daily-aggregate "summary" series) and the
exact header on the live ``/data/month/`` endpoint should be confirmed
against a real response before relying on a single fixed name.
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterLevelReading
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

OPW_BASE = "https://waterlevel.ie"
STATIONS_URL = f"{OPW_BASE}/geojson/"

MIN_VALID_REF = 1
MAX_VALID_REF = 41_000

DATETIME_COLUMN_CANDIDATES = ("Datetime", "datetime", "date", "Date")
VALUE_COLUMN_CANDIDATES = ("Value", "value")


class IrelandOPWCollector(BaseCollector):
    """
    Collect real-time river/lake water level from Ireland's OPW (waterlevel.ie).

    Parameters
    ----------
    client : CachedHTTPClient, optional
        Injected for testing; a default client is created otherwise.
    """

    name = "ireland_opw"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=30, period_seconds=60),
                cache_ttl_seconds=900,  # station readings update every 15 min
            )
        )

    def fetch_stations(self) -> list[dict[str, Any]]:
        """Fetch the GeoJSON list of active OPW stations."""
        data = self.client.get_json(STATIONS_URL)
        return data.get("features", [])

    def fetch_raw(
        self,
        stations: list[dict[str, Any]] | None = None,
        max_stations: int | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Fetch raw water-level rows for every valid OPW station.

        Parameters
        ----------
        stations : list of dict, optional
            Pre-fetched GeoJSON features, to avoid a second station-list
            call (mainly for tests). Fetched automatically if omitted.
        max_stations : int, optional
            Cap on the number of stations to fetch series for. ``None``
            means all valid stations — useful to limit during testing or
            exploratory use, since there are hundreds of stations.
        """
        if stations is None:
            stations = self.fetch_stations()

        rows: list[dict[str, Any]] = []
        n_fetched = 0
        for feature in stations:
            if max_stations is not None and n_fetched >= max_stations:
                break

            props = feature.get("properties", {})
            ref = props.get("ref") or props.get("station_ref") or props.get("id")
            if ref is None:
                continue

            ref_str = str(ref).lstrip("0") or "0"
            try:
                ref_int = int(ref_str)
            except ValueError:
                continue
            if not (MIN_VALID_REF <= ref_int <= MAX_VALID_REF):
                continue

            csv_url = props.get("csv_file")
            if not csv_url:
                last5 = str(ref).zfill(5)[-5:]
                csv_url = f"{OPW_BASE}/data/month/{last5}_0001.csv"
            elif not csv_url.startswith(("http://", "https://")):
                csv_url = f"{OPW_BASE}/{csv_url.lstrip('/')}"

            try:
                text = self.client.get_text(csv_url)
            except Exception as exc:  # noqa: BLE001 - one bad station shouldn't abort the run
                logger.debug("Ireland OPW: failed to fetch %s: %s", csv_url, exc)
                continue

            for row in self._parse_csv(text):
                row["station_ref"] = str(ref)
                row["station_name"] = props.get("name") or props.get("station_name")
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates")
                if coords and len(coords) >= 2:
                    row["longitude"], row["latitude"] = coords[0], coords[1]
                rows.append(row)

            n_fetched += 1

        return rows

    @staticmethod
    def _parse_csv(text: str) -> list[dict[str, Any]]:
        """Parse a station CSV into {"datetime": ..., "value": ...} rows.

        Column names are resolved defensively (see module docstring) so a
        header shape mismatch fails loudly with the real headers rather
        than silently producing zero rows.
        """
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return []

        dt_col = next((c for c in DATETIME_COLUMN_CANDIDATES if c in reader.fieldnames), None)
        val_col = next((c for c in VALUE_COLUMN_CANDIDATES if c in reader.fieldnames), None)
        if dt_col is None or val_col is None:
            logger.warning(
                "Ireland OPW: could not resolve datetime/value columns in CSV header %s",
                reader.fieldnames,
            )
            return []

        parsed: list[dict[str, Any]] = []
        for row in reader:
            raw_val = (row.get(val_col) or "").strip()
            raw_dt = (row.get(dt_col) or "").strip()
            if not raw_val or not raw_dt:
                continue
            parsed.append({"datetime": raw_dt, "value": raw_val})
        return parsed

    def normalise(self, raw: list[dict[str, Any]]) -> Sequence[WaterLevelReading]:
        readings: list[WaterLevelReading] = []
        skipped = 0
        for row in raw:
            try:
                lat, lon = row.get("latitude"), row.get("longitude")
                loc = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None

                readings.append(
                    WaterLevelReading(
                        source=DataSource.IRELAND_OPW,
                        station_id=row["station_ref"],
                        station_name=row.get("station_name"),
                        location=loc,
                        reading_datetime=self._parse_datetime(row["datetime"]),
                        water_level=float(row["value"]),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                skipped += 1
                logger.debug("Skipping Ireland OPW row: %s", exc)

        if skipped:
            logger.warning(
                "Ireland OPW normalise(): skipped %d/%d row(s) (missing/invalid fields)",
                skipped,
                len(raw),
            )
        return readings

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        """Parse either a bare date (YYYY-MM-DD) or a full ISO timestamp.

        waterlevel.ie timestamps are UTC. A trailing "Z" is normalised
        explicitly since datetime.fromisoformat() only accepts a bare "Z"
        on Python 3.11+, and CI runs Python 3.10. Stored tz-naive to match
        the convention used by other collectors in this codebase.
        """
        value = value.strip()
        if len(value) == 10:  # "YYYY-MM-DD", no time component
            return datetime.strptime(value, "%Y-%m-%d")
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)