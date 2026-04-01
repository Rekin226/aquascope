"""FAO WaPOR data collector.

Fetches satellite-based evapotranspiration and water productivity data
from FAO's WaPOR (Water Productivity Open-access portal) v3 API.

API: https://io.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3

References
----------
FAO. (2024). WaPOR v3 - The FAO portal to monitor Water Productivity
    through Open access of Remotely sensed derived data.
    https://www.fao.org/in-action/remote-sensing-for-water-productivity/
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.agriculture import WaPORObservation
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

WAPOR_BASE = "https://io.apps.fao.org/gismgr/api/v2"

# WaPOR v3 variable codes
WAPOR_VARIABLES: dict[str, str] = {
    "AETI": "Actual EvapoTranspiration and Interception",
    "NPP": "Net Primary Production",
    "RET": "Reference EvapoTranspiration",
}


def _parse_date(value: Any) -> dt.date | None:
    """Convert a WaPOR date-like value to ``datetime.date``."""
    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Convert a bbox string or sequence into a 4-float tuple."""
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


def _extract_numeric_value(record: dict[str, Any]) -> float | None:
    """Return the first usable numeric value from a WaPOR API record."""
    for key in ("value", "mean", "summaryValue", "rasterValue", "eto"):
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


class WaPORCollector(BaseCollector):
    """Collect satellite-based ET data from FAO WaPOR v3.

    Parameters
    ----------
    client : CachedHTTPClient | None
        HTTP client instance. A default is created if *None*.

    References
    ----------
    FAO. (2024). WaPOR v3. https://www.fao.org/in-action/remote-sensing-for-water-productivity/
    """

    name = "wapor"

    def __init__(self, client: CachedHTTPClient | None = None) -> None:
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=WAPOR_BASE,
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=86400,
            )
        )

    def fetch_raw(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        start_date: str = "2020-01-01",
        end_date: str = "2020-12-31",
        variable: str = "RET",
        **kwargs: Any,
    ) -> list[dict]:
        """Fetch raw WaPOR raster catalogue/summary data.

        Parameters
        ----------
        bbox : tuple[float, float, float, float] | None
            Bounding box as ``(west, south, east, north)`` in decimal degrees.
        start_date : str
            ISO date string for start of period.
        end_date : str
            ISO date string for end of period.
        variable : str
            WaPOR variable code (``'AETI'``, ``'NPP'``, ``'RET'``).

        Returns
        -------
        list[dict]
            Raw API response records.
        """
        if variable not in WAPOR_VARIABLES:
            raise ValueError(f"Unknown WaPOR variable '{variable}'. Available: {sorted(WAPOR_VARIABLES)}")

        params: dict[str, Any] = {
            "startDate": start_date,
            "endDate": end_date,
        }
        if bbox is not None:
            params["bbox"] = ",".join(str(c) for c in bbox)

        path = f"catalog/workspaces/WAPOR-3/cubes/{variable}"
        data = self.client.get_json(path, params=params)
        if isinstance(data, dict):
            raw_records = data.get("response", data.get("data", [data]))
        else:
            raw_records = data if isinstance(data, list) else []

        records = raw_records if isinstance(raw_records, list) else [raw_records]
        normalised_records: list[dict] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            enriched = dict(record)
            enriched.setdefault("_cube_code", variable)
            enriched.setdefault("_cube_label", WAPOR_VARIABLES.get(variable))
            if bbox is not None:
                enriched.setdefault("_bbox", bbox)
            enriched.setdefault("_requested_start_date", start_date)
            enriched.setdefault("_requested_end_date", end_date)
            normalised_records.append(enriched)
        return normalised_records

    def normalise(self, raw: list[dict]) -> Sequence[WaPORObservation]:
        """Convert raw WaPOR response into ``WaPORObservation`` records.

        Parameters
        ----------
        raw : list[dict]
            Records from ``fetch_raw``.

        Returns
        -------
        Sequence[WaPORObservation]
            Normalised WaPOR observations.
        """
        records: list[WaPORObservation] = []
        for rec in raw:
            try:
                value = _extract_numeric_value(rec)
                if value is None:
                    continue

                cube_code = str(rec.get("cube_code") or rec.get("_cube_code") or rec.get("variable") or "").strip()
                if not cube_code:
                    continue

                records.append(
                    WaPORObservation(
                        cube_code=cube_code,
                        cube_label=rec.get("cube_label") or rec.get("_cube_label") or WAPOR_VARIABLES.get(cube_code),
                        date=_parse_date(rec.get("date") or rec.get("time") or rec.get("time_start") or rec.get("startDate")),
                        start_date=_parse_date(rec.get("start_date") or rec.get("startDate") or rec.get("time_start") or rec.get("_requested_start_date")),
                        end_date=_parse_date(rec.get("end_date") or rec.get("endDate") or rec.get("time_end") or rec.get("_requested_end_date")),
                        bbox=_parse_bbox(rec.get("bbox") or rec.get("_bbox")),
                        value=value,
                        unit=rec.get("unit") or rec.get("units") or rec.get("measureUnit"),
                        statistic=rec.get("statistic") or rec.get("aggregation") or rec.get("measure"),
                        aoi_id=rec.get("aoi_id") or rec.get("aoiId"),
                        level=rec.get("level") or rec.get("rasterLevel") or rec.get("cube_level"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping WaPOR record: %s", exc)
        return records
