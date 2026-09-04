"""
Collector for the US Water Quality Portal (WQP).

The WQP integrates data from USGS, EPA, and 400+ agencies with
430M+ records. We use the WQX 3.0 API.

API docs : https://www.waterqualitydata.us/webservices_documentation/
Endpoint : https://www.waterqualitydata.us/wqx3/Result/search
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
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

# The base path for the new WQX 3.0 API
WQP_BASE = "https://www.waterqualitydata.us/wqx3"
# We use the "narrow" profile for all queries since it captures every field
# required by ``normalise()``
WQP_PROFILE = "narrow"


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
                # A payload-appropriate timeout is applied in order to
                # avoid timeouts on larger queries.
                timeout=600.0,
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def fetch_raw(
        self,
        state_code: str | None = None,
        characteristic_name: str | Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        bbox: str | None = None,
        max_results: int = 1000,
        site_id: str | Sequence[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch water quality results from WQP.

        Parameters
        ----------
        state_code : str | None
            e.g. ``"US:06"`` for California.
        characteristic_name : str | Sequence[str] | None
            e.g. ``"Dissolved oxygen (DO)"``, ``"pH"``; a list asks for several at once.
        site_id : str | Sequence[str] | None
            WQP monitoring location id(s), e.g. ``"USGS-01646500"``; a list asks for several.
        start_date : str | None
            ``"MM-DD-YYYY"`` format.
        end_date : str | None
            ``"MM-DD-YYYY"`` format.
        bbox : str | None
            Bounding box: ``"west,south,east,north"`` in decimal degrees.
        max_results : int
            Limit number of results (WQP default returns CSV).
        """
        params: dict[str, Any] = {
            "mimeType": "csv",
            "sorted": "no",
            "zip": "no",
            "dataProfile": WQP_PROFILE,
        }
        if state_code:
            params["statecode"] = state_code
        if characteristic_name:
            params["characteristicName"] = (list(characteristic_name) if isinstance(characteristic_name, (list, tuple))
                                            else characteristic_name)
        if site_id:
            params["siteid"] = list(site_id) if isinstance(site_id, (list, tuple)) else site_id
        if start_date:
            params["startDateLo"] = start_date
        if end_date:
            params["startDateHi"] = end_date
        if bbox:
            params["bBox"] = bbox

        # WQP returns CSV with no server-side row cap, so a state query can be
        # hundreds of MB even when only `max_results` rows are wanted. Stream the
        # CSV and stop reading once we have enough rows rather than buffering the
        # whole body (and writing it to the disk cache).
        if self._supports_streaming():
            line_source = self._stream_csv_lines(params)
        else:
            # Streaming transport unavailable (e.g. browser/WASM): fall back to a
            # buffered read. Warn that streaming is off and that the fetch may be
            # slow and time out. Don't cache the (potentially huge) body.
            logger.warning(
                "WQP streaming is not available; falling back to a buffered request. "
                "This may transfer a very large body and the fetch may time out."
            )
            line_source = iter(
                self.client.get_text("/Result/search", params=params, use_cache=False).splitlines()
            )

        records: list[dict] = []
        try:
            reader = csv.DictReader(line_source)
            for row in reader:
                if len(records) >= max_results:
                    break
                records.append(dict(row))
        except Exception as exc:
            logger.error("WQP fetch failed: %s", exc)
            raise

        logger.info("WQP fetch returned %d raw rows (max_results=%d).", len(records), max_results)
        return records

    def _supports_streaming(self) -> bool:
        """True when the shared client's transport can stream a response body."""
        return callable(getattr(self.client._client, "stream", None))

    def _stream_csv_lines(self, params: dict[str, Any]) -> Iterator[str]:
        """Yield the WQP CSV body one line at a time, without caching.

        Streams directly off the underlying httpx transport so the caller can
        stop early once ``max_results`` rows are seen. The disk cache is
        deliberately not touched: a generated multi-hundred-MB payload is not
        worth caching, and a partial stream must never be treated as a complete
        response. Errors propagate (HTTP status, transport error, timeout) so a
        dead or slow endpoint is distinguishable from a genuine empty answer.
        """
        if self.client.rate_limiter:
            self.client.rate_limiter.wait_if_needed()
        url = f"{self.client.base_url}/Result/search"
        with self.client._client.stream(
            "GET", url, params=params, headers={"Accept": "text/csv"}
        ) as resp:
            resp.raise_for_status()
            yield from resp.iter_lines()

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                val_str = row.get("Result_Measure", "")
                if not val_str or val_str.strip() in ("", "-"):
                    logger.debug(
                        "Skipping WQP row: missing or empty Result_Measure for station %s",
                        row.get("Location_Identifier", "unknown"),
                    )
                    continue

                loc = None
                lat = row.get("Location_Latitude")
                lon = row.get("Location_Longitude")
                if lat and lon:
                    try:
                        loc = GeoLocation(latitude=float(lat), longitude=float(lon))
                    except (ValueError, TypeError):
                        pass

                date_str = row.get("Activity_StartDate", "")
                time_str = row.get("Activity_StartTime", "00:00:00")
                try:
                    sample_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        sample_dt = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        logger.debug(
                            "Skipping WQP row: unparseable date '%s' for station %s",
                            date_str,
                            row.get("Location_Identifier", "unknown"),
                        )
                        continue

                samples.append(
                    WaterQualitySample(
                        source=DataSource.WQP,
                        station_id=row.get("Location_Identifier", "unknown"),
                        station_name=row.get("Location_Name"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=row.get("Result_Characteristic", "unknown"),
                        value=float(val_str),
                        unit=row.get("Result_MeasureUnit", ""),
                        county=row.get("Location_CountyCode"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug(
                    "Skipping WQP row due to error: %s (station: %s)",
                    exc,
                    row.get("Location_Identifier", "unknown"),
                )

        return samples
