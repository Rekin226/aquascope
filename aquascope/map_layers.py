"""Whole-world map layers (#544): the world river status, month by month since 1990.

GEOGLOWS v2 publishes a global map of river status for every month since
January 1990: ``hydrosos/cogs/YYYY-MM.tif`` in its public bucket, one RGB
cloud-optimised GeoTIFF per month (EPSG:4326, 0.05 degree, 7200 x 3600, deflate,
about 500 KB, CORS ``*``). Each HydroBASINS level-4 basin is painted in one of
the five WMO HydroSOS classes: the month's mean flow at the basin's outlets
against the 10th, 25th, 75th and 90th percentiles of the same calendar month in
the GEOGLOWS retrospective simulation (``hydrosos/thresholds.parquet``).

How the classes and colours are made was read from GEOGLOWS's own pipeline,
``retrospective-update/monthly_products.py`` in
https://github.com/geoglows/rfs-v2-retrospective-update (checked 2026-10-10),
and confirmed against the files: 2026-09 and 1990-01 hold exactly these five
colours and nodata (0, 0, 0). The red band alone tells the classes apart, which
is what the Explorer decodes (``explorer/src/status-core.js``; a test keeps the
two tables equal).

:func:`river_status_month` is the one function the faces share: the CLI's
``aquascope layers status [YYYY-MM]``, the MCP tool of the same name, and the
Explorer's layer, which lists the same bucket in the browser to find the newest
month.

Pure standard library apart from the HTTP client, so it imports on a bare
install and in the Explorer's Pyodide worker.
"""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any

GEOGLOWS_BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com"
STATUS_PREFIX = "hydrosos/cogs/"
STATUS_FIRST = "1990-01"
#: When the bucket was last read by hand, and what it held then. Used only when it cannot be listed.
CHECKED = "2026-10-10"
CHECKED_LATEST = "2026-09"
CHECKED_MISSING = ("2026-03",)

STATUS_LICENCE = "CC BY 4.0"
STATUS_ATTRIBUTION = "River status: GEOGLOWS v2 HydroSOS monthly map (GEOGloWS ECMWF Streamflow Service)"
STATUS_METHOD_URL = "https://github.com/geoglows/rfs-v2-retrospective-update"
STATUS_THRESHOLDS_URL = f"{GEOGLOWS_BUCKET}/hydrosos/thresholds.parquet"

#: The five classes as GEOGLOWS draws them: ``rgb`` is the colour in the file (class 1 to 5 of
#: monthly_products.py), the percentiles are the bounds of the month's mean flow against the same
#: calendar month in the retrospective simulation. The ids are aquascope.nownext.STATUS_CLASSES'.
STATUS_CLASSES: list[dict[str, Any]] = [
    {"id": "much_below", "label": "much below normal", "hydrosos": "notably low",
     "below_pct": 10, "range": "below the 10th percentile", "rgb": [205, 35, 63], "hex": "#cd233f"},
    {"id": "below", "label": "below normal", "hydrosos": "below normal",
     "from_pct": 10, "below_pct": 25, "range": "10th to 25th percentile", "rgb": [255, 168, 133], "hex": "#ffa885"},
    {"id": "normal", "label": "normal", "hydrosos": "normal range",
     "from_pct": 25, "below_pct": 75, "range": "25th to 75th percentile", "rgb": [231, 226, 188], "hex": "#e7e2bc"},
    {"id": "above", "label": "above normal", "hydrosos": "above normal",
     "from_pct": 75, "below_pct": 90, "range": "75th to 90th percentile", "rgb": [142, 206, 238], "hex": "#8eceee"},
    {"id": "much_above", "label": "much above normal", "hydrosos": "notably high",
     "from_pct": 90, "range": "90th percentile and above", "rgb": [44, 125, 205], "hex": "#2c7dcd"},
]
NODATA_RGB = [0, 0, 0]

GRID = {
    "crs": "EPSG:4326", "resolution_deg": 0.05, "width": 7200, "height": 3600,
    "bounds": [-180.0, -90.0, 180.0, 90.0], "bands": "RGB, uint8", "nodata": 0,
    "units": "HydroBASINS level-4 basins, painted whole",
}

HOW = ("Each HydroBASINS level-4 basin takes the class of its outlets' summed monthly mean flow in the GEOGLOWS v2 "
       "retrospective simulation, against the 10th, 25th, 75th and 90th percentiles of the same calendar month "
       "(hydrosos/thresholds.parquet). Modelled, not measured, and one colour per basin.")

_MONTH = re.compile(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$")
_KEY = re.compile(r"<Key>" + re.escape(STATUS_PREFIX) + r"(\d{4}-\d{2})\.tif</Key>")
_TOKEN = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")
_LIST_TTL = 6 * 3600
_listed: dict[str, Any] = {"at": 0.0, "months": None}


def status_url(month: str) -> str:
    """The COG of one month (``YYYY-MM``)."""
    return f"{GEOGLOWS_BUCKET}/{STATUS_PREFIX}{month}.tif"


def parse_month(value: str | date | None) -> str | None:
    """``YYYY-MM`` from a month, a day or a date; None for None. Raises ValueError on anything else."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return f"{value.year:04d}-{value.month:02d}"
    m = _MONTH.match(str(value).strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ValueError(f"not a month: {value!r} (use YYYY-MM)")
    return f"{m.group(1)}-{m.group(2)}"


def parse_listing(xml: str) -> tuple[list[str], str | None]:
    """The months in one page of an S3 ListObjectsV2 answer, and the token for the next page."""
    token = _TOKEN.search(xml)
    return sorted(set(_KEY.findall(xml))), (token.group(1) if token else None)


def _all_months(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def missing_months(months: list[str]) -> list[str]:
    """The months between the first and the last that have no map."""
    if not months:
        return []
    have = set(months)
    return [m for m in _all_months(months[0], months[-1]) if m not in have]


def status_months(refresh: bool = False) -> list[str]:
    """Every month the bucket holds a status map for, oldest first. Listed live, cached for six hours."""
    now = time.time()
    if not refresh and _listed["months"] is not None and now - _listed["at"] < _LIST_TTL:
        return list(_listed["months"])
    from aquascope.utils.http_client import CachedHTTPClient

    client = CachedHTTPClient(timeout=30.0, retries=2, cache_ttl_seconds=_LIST_TTL)
    months: list[str] = []
    token: str | None = None
    try:
        for _ in range(20):   # 440 keys fit in one page today; follow the token if that changes
            params = {"list-type": "2", "prefix": STATUS_PREFIX}
            if token:
                params["continuation-token"] = token
            page, token = parse_listing(client.get_text(f"{GEOGLOWS_BUCKET}/", params=params, use_cache=not refresh))
            months.extend(page)
            if not token:
                break
    finally:
        client.close()
    months = sorted(set(months))
    _listed.update(at=now, months=months)
    return list(months)


def legend() -> list[dict[str, Any]]:
    """The five classes, driest first, with the colour each has in the file."""
    return [dict(c) for c in STATUS_CLASSES]


def river_status_month(month: str | None = None, live: bool = True) -> dict[str, Any]:
    """The world river status map for one month: where it is, what its colours mean, and which months exist.

    ``month`` is ``YYYY-MM`` (a ``YYYY-MM-DD`` day is read as its month); left out, it is the newest month
    the bucket holds. ``live=False`` skips listing the bucket and uses what it held on ``CHECKED``.
    """
    try:
        wanted = parse_month(month)
    except ValueError as exc:
        return {"error": str(exc)}
    out: dict[str, Any] = {
        "layer": "river_status", "label": "World river status",
        "attribution": STATUS_ATTRIBUTION, "licence": STATUS_LICENCE,
        "method": HOW, "method_source": STATUS_METHOD_URL, "thresholds": STATUS_THRESHOLDS_URL,
        "grid": dict(GRID), "legend": legend(), "nodata_rgb": list(NODATA_RGB),
    }
    months: list[str] | None = None
    if live:
        try:
            months = status_months()
        except Exception as exc:  # noqa: BLE001 - the recorded range below still stands
            out["live_error"] = f"{type(exc).__name__}: {exc}"
    if months:
        first, latest, missing = months[0], months[-1], missing_months(months)
        out["listed"] = True
    else:
        first, latest, missing = STATUS_FIRST, CHECKED_LATEST, list(CHECKED_MISSING)
        out["listed"] = False
        out["checked"] = CHECKED
    out["valid_range"] = {"first": first, "latest": latest, "months": len(_all_months(first, latest)) - len(missing)}
    out["missing"] = missing
    target = wanted or latest
    out["month"] = target
    if target < first or target > latest:
        out["available"] = False
        out["error"] = f"no status map for {target}: they run from {first} to {latest}"
        return out
    if target in missing:
        out["available"] = False
        out["error"] = f"no status map for {target} (a gap in the GEOGLOWS series)"
        return out
    out["available"] = True
    out["url"] = status_url(target)
    return out
