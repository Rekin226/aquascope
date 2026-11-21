"""FAO WaPOR data collector.

Fetches satellite-based evapotranspiration and water productivity data
from FAO's WaPOR (Water Productivity Open-access portal) v3 API.

API: https://data.apps.fao.org/gismgr/shared/docs/api-reference/index.html

References
----------
FAO. (2024). WaPOR v3 - The FAO portal to monitor Water Productivity
    through Open access of Remotely sensed derived data.
    https://www.fao.org/in-action/remote-sensing-for-water-productivity/
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from aquascope.collectors.base import BaseCollector, CollectorError
from aquascope.schemas.agriculture import WaPORObservation
from aquascope.utils.cog import COG, COGError, Fetch, open_cog
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

WAPOR_BASE = "https://data.apps.fao.org/gismgr/api/v2"
_MAPSETS = "catalog/workspaces/WAPOR-3/mapsets"
_BUCKET = "https://storage.googleapis.com/fao-gismgr-wapor-3-data/DATA/WAPOR-3/MAPSET"
_MAX_PERIODS = 100
_MAX_WINDOW_CELLS = 250_000
_MAX_RANGE_BYTES = 4 << 20
_COLLECTION_BYTE_BUDGET = 32 << 20
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")

# WaPOR v3 variable codes
WAPOR_VARIABLES: dict[str, str] = {
    "AETI": "Actual EvapoTranspiration and Interception",
    "NPP": "Net Primary Production",
    "RET": "Reference EvapoTranspiration",
}


def _range_fetcher(client: httpx.Client) -> Fetch:
    """Read verified ranges only, even if a proxy replies with a full raster."""
    remaining = _COLLECTION_BYTE_BUDGET

    def fetch(url: str, start: int, end: int) -> bytes:
        nonlocal remaining
        if not 0 < end - start + 1 <= min(_MAX_RANGE_BYTES, remaining):
            raise COGError("WaPOR raster read exceeds the byte limit; use a smaller area or date range")
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            returned = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if response.status_code != 206 or returned is None:
                raise COGError("WaPOR raster server did not honour the byte-range request")
            first, last, total = map(int, returned.groups())
            if first != start or last != min(end, total - 1) or first > last:
                raise COGError("WaPOR raster server returned a different byte range")
            size = last - first + 1
            result = bytearray()
            for chunk in response.iter_raw(chunk_size=65536):
                if len(result) + len(chunk) > size:
                    raise COGError("WaPOR raster response exceeded its declared byte range")
                result.extend(chunk)
            if len(result) != size:
                raise COGError("WaPOR raster response ended before its declared byte range")
        remaining -= len(result)
        return bytes(result)

    return fetch


def _bbox_mean(cog: COG, bbox: tuple[float, float, float, float]) -> float | None:
    """Mean of valid cells weighted by their intersected spherical areas."""
    transform = cog.transform
    if cog.epsg != 4326 or transform is None:
        raise COGError("WaPOR raster must be georeferenced in WGS84")
    west, south, east, north = bbox
    x0, dx, x_skew, y0, y_skew, dy = transform
    if dx <= 0 or dy >= 0 or x_skew != 0 or y_skew != 0:
        raise COGError("WaPOR raster must have a north-up grid")
    c0 = max(0, math.floor((west - x0) / dx))
    c1 = min(cog.width, math.ceil((east - x0) / dx))
    r0 = max(0, math.floor((north - y0) / dy))
    r1 = min(cog.height, math.ceil((south - y0) / dy))
    if c1 <= c0 or r1 <= r0:
        return None
    if (c1 - c0) * (r1 - r0) > _MAX_WINDOW_CELLS:
        raise ValueError(f"WaPOR bounding box exceeds {_MAX_WINDOW_CELLS:,} raster cells; use a smaller area")
    values = cog.read_window(c0, r0, c1 - 1, r1 - 1)
    left = x0 + dx * np.arange(c0, c1)
    top = y0 + dy * np.arange(r0, r1)
    lon_width = np.minimum(left + dx, east) - np.maximum(left, west)
    lat_area = np.sin(np.radians(np.minimum(top, north))) - np.sin(np.radians(np.maximum(top + dy, south)))
    weights = np.multiply.outer(np.maximum(0, lat_area), np.maximum(0, lon_width))
    valid = np.isfinite(values) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else None


def _raster_record(
    item: dict[str, Any],
    mapset: str,
    bbox: tuple[float, float, float, float],
    fetch: Fetch,
) -> dict[str, Any] | None:
    code = item["code"]
    if not re.fullmatch(rf"WAPOR-3\.{re.escape(mapset)}\.\d{{4}}-\d{{2}}-D[1-3]", code):
        raise ValueError(f"unexpected WaPOR raster identifier {code!r}")
    url = f"{_BUCKET}/{mapset}/{code}.tif"
    if item["downloadUrl"] != url:
        raise ValueError("WaPOR returned an unexpected raster download location")
    period = next(d["member"] for d in item["dimensions"] if d["type"] == "TIME")
    start = dt.date.fromisoformat(period["startDate"])
    end = dt.date.fromisoformat(period["endDate"])
    if end < start:
        raise ValueError("WaPOR raster has an invalid observation period")
    cog = open_cog(url, fetch=fetch)
    if (cog.scale, cog.offset, cog.nodata) != (item["scale"], item["offset"], item["noDataValue"]):
        raise COGError("WaPOR raster scale, offset or nodata differs from the catalog")
    value = _bbox_mean(cog, bbox)
    if value is None:
        return None
    return {
        "date": start.isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "bbox": bbox,
        "value": value,
        "unit": item["measureUnit"],
        "statistic": "area_weighted_mean",
        "level": "L1",
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
        start_date: str | None = None,
        end_date: str | None = None,
        variable: str = "RET",
        **kwargs: Any,
    ) -> list[dict]:
        """Fetch one area-weighted observation per intersecting WaPOR dekad.

        Parameters
        ----------
        bbox : tuple[float, float, float, float] | None
            Required bounding box as ``(west, south, east, north)`` in WGS84
            decimal degrees. Only valid pixels contribute to the mean.
        start_date : str
            Inclusive ISO date. With neither date set, use the 2020 calendar year.
        end_date : str
            Inclusive ISO date. If only a start date is given, end today.
        variable : str
            WaPOR variable code (``'AETI'``, ``'NPP'``, ``'RET'``).

        Returns
        -------
        list[dict]
            Area-weighted means in the original raster units, with the complete
            source period even when the query intersects only part of it.
        """
        if variable not in WAPOR_VARIABLES:
            raise ValueError(f"Unknown WaPOR variable '{variable}'. Available: {sorted(WAPOR_VARIABLES)}")
        if bbox is None:
            raise ValueError("WaPOR requires a bounding box (west, south, east, north)")
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("WaPOR requires ordered WGS84 bounding box coordinates")
        start = dt.date.fromisoformat(start_date or "2020-01-01")
        if end_date is not None:
            end = dt.date.fromisoformat(end_date)
        else:
            end = dt.date.today() if start_date else dt.date(2020, 12, 31)
        if end < start:
            raise ValueError("WaPOR end_date must be on or after start_date")
        # Match the periods across AETI, RET and NPP for productivity joins.
        mapset = f"L1-{variable}-D"
        path = f"{_MAPSETS}/{mapset}/rasters"
        catalog_url = f"{WAPOR_BASE}/{path}"
        try:
            response = self.client.get_json(
                path, params={"filter": f"time:OVERLAPS:{start}:{end}", "sort": "code", "size": _MAX_PERIODS}
            )["response"]
            items = response["items"]
            if not isinstance(items, list) or not isinstance(response["hasNext"], bool):
                raise ValueError("WaPOR catalog returned an invalid raster list")
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise CollectorError(
                "WaPOR raster catalog request failed", source=self.name, url=catalog_url, cause=exc
            ) from exc
        if response["hasNext"] or len(items) > _MAX_PERIODS:
            raise ValueError(f"WaPOR returns at most {_MAX_PERIODS} dekads per request; split the date range")

        records = []
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            fetch = _range_fetcher(client)
            for item in items:
                try:
                    record = _raster_record(item, mapset, bbox, fetch)
                except (httpx.HTTPError, COGError, ValueError, KeyError, TypeError, StopIteration) as exc:
                    raise CollectorError(
                        "WaPOR raster could not be read", source=self.name, url=catalog_url, cause=exc
                    ) from exc
                if record is not None:
                    record["_cube_code"] = variable
                    records.append(record)
        return records

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
