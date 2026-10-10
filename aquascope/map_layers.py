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

:func:`river_status_summary` reads one month's file (with the standard library
and numpy) and says in one line where the rivers are low or high: the share of
each of a few named regions in each class, and the Explorer's caption over the
globe, which status-core.js makes the same way from the file it has already
decoded (a test keeps the two equal).

Pure standard library apart from the HTTP client (and numpy, imported only by
the summary), so it imports on a bare install and in the Explorer's Pyodide worker.
"""

from __future__ import annotations

import re
import struct
import time
import zlib
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


# ── the one-line summary (#543 design pass) ─────────────────────────────────

#: A few regions a reader knows by name, as rough boxes [west, south, east, north] in degrees. They overlap a
#: little and are not basins: they only let one line say where the map is mostly low or high. The Explorer's
#: status-core.js holds the same list (tests/test_map_layers.py keeps them equal).
STATUS_REGIONS: list[dict[str, Any]] = [
    {"name": "the Amazon", "bbox": [-80, -15, -44, 5]},
    {"name": "the La Plata basin", "bbox": [-66, -35, -43, -15]},
    {"name": "Mexico and Central America", "bbox": [-118, 7, -77, 32]},
    {"name": "the western US", "bbox": [-125, 31, -102, 49]},
    {"name": "the eastern US", "bbox": [-102, 25, -67, 49]},
    {"name": "Canada", "bbox": [-141, 49, -52, 70]},
    {"name": "Europe", "bbox": [-10, 36, 40, 71]},
    {"name": "the Sahel", "bbox": [-17, 11, 38, 18]},
    {"name": "the Congo basin", "bbox": [12, -13, 32, 8]},
    {"name": "East Africa", "bbox": [29, -12, 52, 11]},
    {"name": "southern Africa", "bbox": [10, -35, 41, -13]},
    {"name": "the Middle East", "bbox": [34, 12, 63, 42]},
    {"name": "Central Asia", "bbox": [50, 36, 90, 55]},
    {"name": "Siberia", "bbox": [60, 50, 180, 75]},
    {"name": "South Asia", "bbox": [66, 6, 92, 36]},
    {"name": "Southeast Asia", "bbox": [92, -10, 141, 22]},
    {"name": "China", "bbox": [98, 22, 123, 45]},
    {"name": "Australia", "bbox": [112, -44, 154, -10]},
]
#: A region is "mostly" below (or above) normal when this share of its mapped area is in those classes.
HEADLINE_SHARE = 0.5
#: Regions with less of their box mapped than this (desert, sea) are left out of the summary.
MIN_COVER = 0.2
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                "October", "November", "December"]
_RED_TO_CLASS = {c["rgb"][0]: i + 1 for i, c in enumerate(STATUS_CLASSES)}


def _tiff_red(data: bytes) -> Any:
    """The red band of a status file as a (height, width) uint8 array: a tiled, deflated, little-endian
    GeoTIFF with one plane per band, which is how GEOGLOWS writes them (checked on 2026-09 and 1990-01)."""
    import numpy as np

    if data[:4] != b"II*\x00":
        raise ValueError("not a little-endian TIFF")
    (ifd,) = struct.unpack_from("<I", data, 4)
    (n,) = struct.unpack_from("<H", data, ifd)
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 12: 8, 16: 8}
    tags: dict[int, list[int]] = {}
    for i in range(n):
        tag, typ, count, value = struct.unpack_from("<HHII", data, ifd + 2 + 12 * i)
        if typ not in (3, 4, 16):
            continue
        fmt = {3: "H", 4: "I", 16: "Q"}[typ]
        if count * sizes[typ] <= 4:
            tags[tag] = list(struct.unpack_from(f"<{count}{fmt}", data, ifd + 10 + 12 * i))
        else:
            tags[tag] = list(struct.unpack_from(f"<{count}{fmt}", data, value))
    width, height = tags[256][0], tags[257][0]
    if tags.get(259, [1])[0] not in (8, 32946) or tags.get(284, [1])[0] != 2 or tags.get(317, [1])[0] != 1:
        raise ValueError("unexpected TIFF layout (want deflate, planar, no predictor)")
    tw, th = tags[322][0], tags[323][0]
    across, down = -(-width // tw), -(-height // th)
    out = np.zeros((down * th, across * tw), dtype=np.uint8)
    offsets, counts = tags[324], tags[325]
    for t in range(across * down):        # the red plane is the first across * down tiles
        tile = np.frombuffer(zlib.decompress(data[offsets[t]:offsets[t] + counts[t]]), dtype=np.uint8)
        r, c = divmod(t, across)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile.reshape(th, tw)
    return out[:height, :width]


def status_classes(data: bytes) -> Any:
    """A status file as classes, 0 for no data and 1 (much below normal) to 5 (much above)."""
    import numpy as np

    lut = np.zeros(256, dtype=np.uint8)
    for red, k in _RED_TO_CLASS.items():
        lut[red] = k
    return lut[_tiff_red(data)]


def region_shares(classes: Any) -> list[dict[str, Any]]:
    """For each named region: the share of its mapped area below normal (much below and below), above
    normal (above and much above), and how much of its box is mapped at all. Area-weighted by latitude.
    ``classes`` covers the world, 90 N to 90 S and 180 W to 180 E (0.05 degree in the files)."""
    import numpy as np

    height, width = classes.shape
    deg = 180 / height
    lats = 90 - (np.arange(height) + 0.5) * deg
    weights = np.cos(np.radians(lats))
    out = []
    for reg in STATUS_REGIONS:
        w, s, e, n = reg["bbox"]
        r0, r1 = int(np.ceil((90 - n) / deg - 0.5)), int(np.floor((90 - s) / deg - 0.5)) + 1
        c0, c1 = int(np.ceil((w + 180) / deg - 0.5)), int(np.floor((e + 180) / deg - 0.5)) + 1
        r0, r1, c0, c1 = max(r0, 0), min(r1, height), max(c0, 0), min(c1, width)
        sub, wt = classes[r0:r1, c0:c1], weights[r0:r1]
        per = [float(((sub == k).sum(axis=1) * wt).sum()) for k in range(6)]
        mapped, box = sum(per[1:]), float(wt.sum() * (c1 - c0))
        out.append({
            "name": reg["name"],
            "below": round((per[1] + per[2]) / mapped, 3) if mapped else 0.0,
            "above": round((per[4] + per[5]) / mapped, 3) if mapped else 0.0,
            "cover": round(mapped / box, 3) if box else 0.0,
        })
    return out


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def status_headline(month: str, regions: list[dict[str, Any]]) -> str:
    """One line for a month from :func:`region_shares`: the regions mostly below normal and mostly above,
    the larger side first (two at most) and the other side's strongest, or that no region is either."""
    label = f"{_MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"
    sides: dict[str, list[tuple[float, int, str]]] = {"below": [], "above": []}
    for i, r in enumerate(regions):
        if r["cover"] < MIN_COVER:
            continue
        for side, other in (("below", "above"), ("above", "below")):
            if r[side] >= HEADLINE_SHARE and r[side] > r[other]:
                sides[side].append((-r[side], i, r["name"]))
    if not sides["below"] and not sides["above"]:
        return f"River status, {label}: no large region mostly above or below normal"
    first = "below" if len(sides["below"]) >= len(sides["above"]) else "above"
    second = "above" if first == "below" else "below"
    parts = [f"much of {_names([n for *_, n in sorted(sides[first])[:2]])} {first} normal"]
    if sides[second]:
        parts.append(f"much of {sorted(sides[second])[0][2]} {second}")
    return f"River status, {label}: " + ", ".join(parts)


def river_status_summary(month: str | None = None, live: bool = True) -> dict[str, Any]:
    """Where the rivers are low or high in one month of the world river status map, in one line, with the
    share of each named region below and above normal. Reads the month's file (about 500 kB)."""
    res = river_status_month(month, live=live)
    if not res.get("available"):
        return res
    from aquascope.utils.http_client import CachedHTTPClient

    client = CachedHTTPClient(timeout=60.0, retries=2, cache_ttl_seconds=_LIST_TTL)
    try:
        data = client.get_bytes(res["url"])
    except Exception as exc:  # noqa: BLE001 - say why rather than fail the whole answer
        return {**res, "error": f"could not read {res['url']}: {type(exc).__name__}: {exc}"}
    finally:
        client.close()
    regions = region_shares(status_classes(data))
    return {**res, "regions": regions, "headline": status_headline(res["month"], regions),
            "regions_note": ("Rough named boxes, not basins; shares of the mapped area, weighted by latitude. "
                             f"'Much of' means at least {int(HEADLINE_SHARE * 100)}%.")}
