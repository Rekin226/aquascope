"""FAO AQUASTAT data collector.

Fetches country-level water resources data from FAO's AQUASTAT database
via the FAOSTAT API.

API endpoint: https://www.fao.org/faostat/api/v1/
Data domain: https://www.fao.org/aquastat/en/databases/

References
----------
FAO. (2023). AQUASTAT - FAO's Global Information System on Water and Agriculture.
    https://www.fao.org/aquastat/
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.agriculture import AquastatRecord
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

AQUASTAT_BASE = "https://www.fao.org/faostat/api/v1"

# Key AQUASTAT variable IDs
AQUASTAT_VARIABLES: dict[int, str] = {
    4263: "Total water withdrawal",
    4253: "Agricultural water withdrawal",
    4254: "Industrial water withdrawal",
    4255: "Municipal water withdrawal",
    4192: "Total renewable water resources",
    4312: "Total area equipped for irrigation",
}


class AquastatCollector(BaseCollector):
    """Collect country-level water data from FAO AQUASTAT.

    Parameters
    ----------
    client : CachedHTTPClient | None
        HTTP client instance. A default is created if *None*.

    References
    ----------
    FAO. (2023). AQUASTAT. https://www.fao.org/aquastat/
    """

    name = "aquastat"

    def __init__(self, client: CachedHTTPClient | None = None) -> None:
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=AQUASTAT_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=86400,
            )
        )

    def fetch_raw(
        self,
        country_code: str = "all",
        variable_ids: list[int] | None = None,
        start_year: int = 2000,
        end_year: int = 2023,
        **kwargs: Any,
    ) -> list[dict]:
        """Fetch raw AQUASTAT data from the FAOSTAT API.

        Parameters
        ----------
        country_code : str
            ISO3 country code, or ``'all'`` for global data.
        variable_ids : list[int] | None
            AQUASTAT variable IDs. Defaults to all key variables.
        start_year : int
            Start year (default 2000).
        end_year : int
            End year (default 2023).

        Returns
        -------
        list[dict]
            Raw API response records.
        """
        var_ids = variable_ids or list(AQUASTAT_VARIABLES.keys())
        params: dict[str, Any] = {
            "area": country_code,
            "element": ",".join(str(v) for v in var_ids),
            "year": ",".join(str(y) for y in range(start_year, end_year + 1)),
            "output_type": "objects",
        }

        data = self.client.get_json("en/data/AQUASTAT", params=params)
        if isinstance(data, dict):
            return data.get("data", [])
        return data if isinstance(data, list) else []

    def normalise(self, raw: list[dict]) -> Sequence[AquastatRecord]:
        """Convert raw FAOSTAT response into ``AquastatRecord`` objects.

        Parameters
        ----------
        raw : list[dict]
            Records from ``fetch_raw``.

        Returns
        -------
        Sequence[AquastatRecord]
            Normalised AQUASTAT records.
        """
        records: list[AquastatRecord] = []
        for rec in raw:
            try:
                value = rec.get("Value")
                if value is None or value == "":
                    continue
                records.append(
                    AquastatRecord(
                        country=rec.get("Area", ""),
                        country_code=rec.get("Area Code", ""),
                        year=int(rec.get("Year", 0)),
                        variable=rec.get("Element", ""),
                        value=float(value),
                        unit=rec.get("Unit", ""),
                        source="AQUASTAT",
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping AQUASTAT record: %s", exc)
        return records
