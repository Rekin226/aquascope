"""
Collector for UN SDG 6 (Clean Water and Sanitation) indicator data.

API docs : https://sdg6data.org/en/api
Indicators :
  6.1.1  — Safely managed drinking water services (%)
  6.2.1  — Safely managed sanitation services (%)
  6.3.1  — Safely treated domestic wastewater (%)
  6.3.2  — Water bodies with good ambient water quality (%)
  6.4.1  — Water-use efficiency change (%)
  6.4.2  — Water stress level (%)
  6.5.1  — Integrated water resources management (0-100)
  6.5.2  — Transboundary cooperation (%)
  6.6.1  — Water-related ecosystems change (%)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import SDG6Indicator
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

SDG6_BASE = "https://sdg6data.org/api"

ALL_SDG6_INDICATORS = [
    "6.1.1", "6.2.1", "6.3.1", "6.3.2",
    "6.4.1", "6.4.2", "6.5.1", "6.5.2", "6.6.1",
]


class SDG6Collector(BaseCollector):
    """
    Collect SDG 6 indicator data per country/year from the UN-Water portal.
    """

    name = "sdg6"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=SDG6_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=86400,  # data rarely changes
            )
        )

    def fetch_raw(
        self,
        indicator_codes: list[str] | None = None,
        country_codes: str | None = None,
        year_range: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch raw indicator data for the given codes.

        Parameters
        ----------
        indicator_codes : list[str] | None
            e.g. ["6.3.1", "6.4.2"]. Defaults to all.
        country_codes : str | None
            Comma-separated ISO3 codes, e.g. "TWN,USA,BFA".
        year_range : str | None
            e.g. "2015:2023".
        """
        codes = indicator_codes or ALL_SDG6_INDICATORS
        all_records: list[dict] = []

        for code in codes:
            params: dict[str, Any] = {"_format": "json", "per_page": 500}
            if country_codes:
                params["country"] = country_codes
            if year_range:
                params["date"] = year_range

            page = 1
            while True:
                params["page"] = page
                data = self.client.get_json(f"indicator/{code}", params=params)
                items = data if isinstance(data, list) else data.get("data", [])
                if not items:
                    break
                all_records.extend(items)
                if len(items) < params["per_page"]:
                    break
                page += 1

        return all_records

    def normalise(self, raw: list[dict]) -> Sequence[SDG6Indicator]:
        records: list[SDG6Indicator] = []
        for rec in raw:
            try:
                val = rec.get("Value")
                records.append(
                    SDG6Indicator(
                        indicator_code=rec.get("Indicator", ""),
                        indicator_name=rec.get("SeriesDescription"),
                        country_code=rec.get("GeoAreaCode", ""),
                        country_name=rec.get("GeoAreaName"),
                        year=int(rec.get("TimePeriod", 0)),
                        value=float(val) if val not in (None, "", "NA") else None,
                        unit=rec.get("Units"),
                        series_code=rec.get("SeriesCode"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping SDG6 record: %s", exc)
        return records
