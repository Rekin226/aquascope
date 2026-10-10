"""Rivers as objects: snap a point to a river reach, its simulated record, its upstream area, its way to the sea.

The unit is a GEOGLOWS v2 river reach (about 6.8 million of them, keyed by a
9-digit ``river_id``, the TDX-Hydro ``LINKNO``). The functions below are one engine for
the Explorer, the MCP server and the CLI:

* :func:`snap_to_river` reads the global stream network (``streams.pmtiles``,
  2.4 GB, read in place by byte ranges) around a point and returns the reach it
  stands for (the main channel among the reaches within the tolerance, or the
  one whose upstream area matches a gauge's catchment) and how far it is, or
  says there is no stream within the tolerance.
* :func:`reach_record` fetches the reach's simulated daily discharge since 1940
  from the GEOGLOWS REST API and runs the same analysis a gauge gets
  (:func:`aquascope.explore.analyze_series`): annual maxima, return periods with
  confidence intervals, the flow-duration curve, the monthly regime. It is
  labelled modelled everywhere.
* :func:`upstream_area` adds up the unit catchments upstream of a reach.
* :func:`trace_downstream` follows the network to the outlet and returns the
  path geometry, its length and the Archive gauges along it, with the dams on
  the path, the countries it crosses and the dams upstream
  (:mod:`aquascope.river_path`).
* :func:`upstream_dams` says whether the river is regulated upstream of a
  reach: the Global Dam Watch dams that drain to it and their storage as a
  share of a year's flow.
* :func:`upstream_ids` and :func:`downstream_ids` list the reaches that drain
  to a reach and the reaches from it to the sea, which the Explorer lights up
  on the map.

Everything is keyless and readable from a browser page (both hosts answer CORS
for any origin), and every reader here is plain Python (no pyarrow, no
compiled tile library), so the Explorer's Pyodide worker runs this module
unchanged.

Licences. The GEOGLOWS v2 model output is CC BY 4.0. The river network
geometry is TDX-Hydro (NGA), CC BY-SA 4.0: it may be shown, but a vector
product derived from it carries the share-alike terms, so nothing here writes
geometry to the Archive. The small VPU index below (which processing unit a
``river_id`` prefix can belong to, and each unit's bounding box) was read from
GEOGLOWS's ``tables/package-metadata-table.parquet`` on 2026-10-08 (6,838,900 reaches); it only orders the
downloads and is checked against the unit's own table every time.
"""

from __future__ import annotations

import gzip
import io
import logging
import math
import struct
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ATTRIBUTION",
    "downstream_ids",
    "forecast_stats",
    "main_channel",
    "match_by_area",
    "reach_record",
    "reach_summary",
    "reaches_near",
    "snap_to_river",
    "trace_downstream",
    "upstream_area",
    "upstream_dams",
    "upstream_ids",
]

GEOGLOWS_API = "https://geoglows.ecmwf.int/api/v2"
BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com"
STREAMS_PMTILES = f"{BUCKET}/hydrography-global/streams.pmtiles"
ROUTING_CONFIG = BUCKET + "/routing-configs/vpu={vpu}/{name}"

#: Zoom at which every reach is in the tile archive (tippecanoe kept all reaches from zoom 8, and 12 is the
#: deepest level, where nothing is dropped as too dense).
SNAP_ZOOM = 12
DEFAULT_MAX_DISTANCE_M = 1000.0
#: How many Strahler orders bigger a river beyond the snap tolerance must be than the reach chosen before the snap
#: names it (``larger``): a different size of river, not a neighbouring stream of the same size.
LARGER_ORDER_GAP = 2
RETRO_START = "1940-01-01"

ATTRIBUTION = ("GEOGLOWS v2, the GEOGloWS ECMWF Streamflow Service: simulated discharge, CC BY 4.0. River network: "
               "TDX-Hydro (NGA), CC BY-SA 4.0, shown and not republished.")
LICENCE = {"discharge": "CC BY 4.0", "network": "CC BY-SA 4.0 (TDX-Hydro)",
           "url": f"{BUCKET}/licenses.md"}
MODELLED_NOTE = ("Simulated by the GEOGLOWS v2 hydrologic model (ERA5 runoff routed down the TDX-Hydro network), "
                 "not measured. A gauge on the same river outranks it; use it where there is none.")

# ── the VPU index ───────────────────────────────────────────────────────────
# GEOGLOWS splits the world into 125 vector processing units (VPUs), each a set of whole basins, so a reach's
# path to the outlet never leaves its unit. ``river_id // 10_000_000`` names the TDX-Hydro region, which maps
# to one to ten units; the bounding boxes (lon/lat of the reach outlets in each unit) order the candidates.

_VPU_BY_PREFIX: dict[int, tuple[int, ...]] = {
    11: (101, 102, 103), 12: (104, 105, 106), 13: (107, 108, 109), 14: (110, 111, 112),
    15: (113, 114, 115, 116, 117, 118, 119, 120, 121), 16: (122,), 17: (123,), 18: (124, 125, 126), 21: (201, 202),
    22: (203, 204, 205, 206), 23: (207, 208, 209), 24: (210, 211), 25: (212,), 28: (213, 214, 215),
    29: (216, 217, 218, 219, 220, 221), 31: (301, 302), 32: (303,), 34: (304,), 36: (305,), 41: (401, 402),
    42: (403, 404, 405, 406), 43: (407, 408), 44: (409, 410, 411, 412), 45: (413,), 46: (414, 415),
    47: (416, 417, 418, 419), 48: (420, 421, 422), 49: (423,), 51: (501,), 52: (502,), 53: (503,),
    54: (504, 505, 506, 507, 508, 509, 510, 511, 512, 513), 57: (514,), 61: (601, 602, 603), 62: (604, 605, 606),
    63: (607, 608), 64: (609, 610), 65: (611, 612), 66: (613,), 67: (614,), 71: (701, 702, 703), 72: (704, 705),
    73: (706, 707), 74: (708, 709, 710, 711), 75: (712, 713), 76: (714,), 77: (715, 716, 717), 78: (718,),
    81: (801, 802, 803), 82: (804,),
}
_VPU_BBOX: dict[int, tuple[float, float, float, float]] = {
    101: (30.57, -18.2, 40.83, -5.5), 102: (31.73, 2.09, 54.39, 30.11), 103: (32.66, -7.72, 46.04, 9.54),
    104: (17, -34.77, 32.92, -25.46), 105: (11.74, -32.62, 30.24, -12.36), 106: (18.46, -26.47, 36.79, -9.05),
    107: (11.75, -18.02, 19.04, -6.1), 108: (6.61, -6.05, 14.89, 7.37), 109: (11.8, -13.39, 33.96, 9.17),
    110: (-6.97, 4.33, 6.73, 14.74), 111: (-11.55, 4.28, 15.8, 23.92), 112: (-17.24, 4.36, -5.48, 18.41),
    113: (10.09, 20.7, 23.88, 31.26), 114: (-11.8, 24.46, -1.92, 35.68), 115: (15.89, 23.46, 32.26, 32.81),
    116: (-5.5, 27.24, 7.82, 37), 117: (-17.05, 16.51, -7.71, 28.18), 118: (-10.7, 15.09, 1.62, 25.03),
    119: (4.45, 23.44, 15.82, 37.28), 120: (20.11, 14.3, 32, 30.05), 121: (-5.71, 20.18, 5.89, 29.91),
    122: (23.49, -3.96, 39.73, 31.59), 123: (43.24, -25.59, 50.47, -12.26), 124: (7.09, 14.87, 16.64, 25.12),
    125: (13.41, 12.34, 24.8, 21.24), 126: (7.4, 5.42, 24.38, 16.53), 201: (-5.63, 36.03, 19.23, 48.11),
    202: (18.97, 29.06, 38, 42.85), 203: (22.6, 45.25, 37.66, 55.82), 204: (29.21, 36.94, 43.86, 45.43),
    205: (8.21, 41.27, 29.72, 50.18), 206: (32.71, 44.45, 46.2, 53.99), 207: (-10.31, 50.09, 1.74, 62.31),
    208: (-9.42, 36.16, 3.78, 46.23), 209: (-4.77, 44.61, 16.74, 53.84), 210: (17.05, 49.05, 37.95, 66.05),
    211: (8.11, 49.44, 30.6, 69.18), 212: (4.95, 58.02, 41.28, 80.33), 213: (41.27, 33.72, 66.69, 44.59),
    214: (32.06, 42.5, 60.37, 61.91), 215: (50.06, 41.65, 60.62, 54.66), 216: (41.93, 12.6, 59.75, 26.21),
    217: (36.87, 26.18, 51.92, 40.26), 218: (56.13, 25.12, 69.4, 34.79), 219: (32.53, 23.39, 42.14, 34.82),
    220: (48.24, 26.62, 60.7, 37.36), 221: (37.52, 17.89, 52.08, 30.85), 301: (66.44, 60.92, 85.99, 73.49),
    302: (59.09, 45.81, 92.55, 68), 303: (76.18, 46.47, 113.72, 73.62), 304: (103.29, 52.23, 141.43, 73.76),
    305: (87.85, 45.66, 99.01, 51.05), 401: (112.91, 33.33, 129.49, 48.29), 402: (107.61, 40.34, 141.37, 55.87),
    403: (102.33, 18.21, 114.7, 26.76), 404: (95.98, 32.29, 119.79, 42.66), 405: (113.46, 21.98, 122.56, 37.82),
    406: (90.61, 19.92, 120.38, 35.84), 407: (91.19, 1.28, 104.2, 32.73), 408: (93.93, 8.59, 109.43, 33.79),
    409: (73.42, 17.06, 97.66, 31.39), 410: (72.83, 5.94, 81.87, 19.62), 411: (67.63, 15.87, 86.72, 25.33),
    412: (65.99, 23.21, 82.37, 37.03), 413: (123.77, 24.27, 150.32, 54.33), 414: (80.33, 34.82, 99.16, 43.78),
    415: (73.26, 34.38, 90.43, 42.42), 416: (72.55, 42.32, 84.91, 49.07), 417: (57.69, 45.45, 67.4, 51.73),
    418: (60.75, 39.42, 79.14, 49.34), 419: (58.2, 34.54, 75.07, 46.34), 420: (107.87, 41.14, 116.44, 47.21),
    421: (80.11, 39.37, 100.2, 47.02), 422: (97.45, 37.87, 109.08, 47.6), 423: (80.53, 29.77, 92.13, 36.37),
    501: (95.23, -10.87, 131.66, 5.62), 502: (105.77, -6.06, 126.87, 19.37), 503: (126, -11.83, 162.22, 2.58),
    504: (113.33, -30.96, 120.87, -20.2), 505: (125.91, -34.91, 138.74, -22.24),
    506: (119.02, -23.06, 131.77, -13.89), 507: (142.83, -43.55, 153.59, -19.96),
    508: (136.65, -38.46, 152.29, -24.73), 509: (114.98, -35.05, 126.42, -26.78),
    510: (128.48, -23.14, 139.61, -11.32), 511: (119.81, -30.04, 131.53, -21.42),
    512: (139.1, -24.33, 148.58, -10.9), 513: (131.67, -30.81, 146.63, -19.09),
    514: (166.43, -47.13, 178.49, -34.49), 601: (-79.2, 1.64, -67, 12.41), 602: (-62.83, 0.43, -49.9, 8.63),
    603: (-74.86, 1.54, -59.44, 14.67), 604: (-55.56, -14.86, -48.79, 0.3), 605: (-79.56, -20.42, -50.04, 5.21),
    606: (-55.36, -17.98, -45.9, -0.27), 607: (-47.56, -23.77, -36.33, -7.3), 608: (-48.03, -10.87, -34.8, -0.65),
    609: (-68.95, -34.56, -43.64, -14.22), 610: (-59.86, -35.72, -45.7, -23.63),
    611: (-73.5, -52.81, -57.78, -36.25), 612: (-70.51, -40.95, -56.71, -27.67),
    613: (-75.56, -55.38, -64.64, -14.12), 614: (-91.39, -18.36, -69.82, 9.33), 701: (-115.99, 7.27, -79.4, 32.64),
    702: (-124.4, 35.19, -114.79, 44.24), 703: (-120.87, 30.55, -105.68, 43.43),
    704: (-128.33, 41.24, -109.88, 56.09), 705: (-137.96, 51.22, -124.61, 59.84),
    706: (-117.2, 45.57, -89.96, 59.41), 707: (-101.38, 47.32, -78.67, 60.16), 708: (-74.34, 43.56, -52.69, 54.2),
    709: (-93.14, 40.49, -70.35, 54.18), 710: (-73.49, 51.63, -57.56, 60.43), 711: (-80.55, 47.84, -67.91, 62.54),
    712: (-80.45, 34.65, -66.24, 46.41), 713: (-89.79, 25.15, -77.23, 36.61), 714: (-113.83, 29.08, -77.89, 50.57),
    715: (-108.97, 24.88, -93.21, 38.4), 716: (-97.63, 8.55, -79.3, 21.59), 717: (-106.25, 18.76, -96.04, 27.47),
    718: (-84.92, 15.3, -61.25, 26.72), 801: (-164.71, 58.96, -129.34, 69.03),
    802: (-171.48, 62.7, -142.93, 71.17), 803: (-167.05, 54.54, -136.48, 64.33),
    804: (-143.35, 52.22, -103.02, 70.41),
}


# ── HTTP seams (tests replace these three) ───────────────────────────────────

_CLIENTS: dict[str, Any] = {}


def _client(kind: str) -> Any:
    """One :class:`~aquascope.utils.http_client.CachedHTTPClient` per host role, made on first use."""
    if kind not in _CLIENTS:
        from aquascope.utils.cache import cache_dir
        from aquascope.utils.http_client import CachedHTTPClient

        where = cache_dir() / "geoglows"
        if kind == "api":
            # The REST API reads a Zarr store per request and takes 10 to 20 s; a day's cache is plenty.
            _CLIENTS[kind] = CachedHTTPClient(timeout=120.0, retries=2, cache_dir=where, cache_ttl_seconds=86_400)
        else:
            # The routing tables change only with a model version.
            _CLIENTS[kind] = CachedHTTPClient(timeout=90.0, retries=2, cache_dir=where,
                                              cache_ttl_seconds=30 * 86_400)
    return _CLIENTS[kind]


def _fetch_range(url: str, start: int, length: int) -> bytes:
    """``length`` bytes of ``url`` from ``start`` (an HTTP range request)."""
    return _client("s3").get_bytes(url, start=int(start), end=int(start) + int(length) - 1)


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    # The GEOGLOWS API reads a Zarr store per request and now and then answers 500 to a request the next one
    # serves (seen 2026-10-08); in a browser that error page carries no CORS header and arrives as a bare
    # network error. The browser client does not retry on its own, so one more try happens here.
    try:
        return _client("api").get_json(url, params=params)
    except Exception as exc:  # noqa: BLE001 - RuntimeError from the client, a JsException in Pyodide
        logger.info("GEOGLOWS API failed once, trying again: %s", exc)
        return _client("api").get_json(url, params=params)


def _fetch_text(url: str) -> str:
    return _client("s3").get_text(url)


# ── PMTiles v3, read in place ────────────────────────────────────────────────


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return result, i


def _decompress(buf: bytes, codec: int) -> bytes:
    if codec in (0, 1):  # unknown (assume none), none
        return buf
    if codec == 2:
        return gzip.decompress(buf)
    raise ValueError(f"PMTiles compression {codec} is not supported (only none and gzip)")


def _parse_directory(buf: bytes) -> list[tuple[int, int, int, int]]:
    """A PMTiles directory: ``(tile_id, run_length, length, offset)`` per entry."""
    n, i = _varint(buf, 0)
    ids: list[int] = []
    last = 0
    for _ in range(n):
        v, i = _varint(buf, i)
        last += v
        ids.append(last)
    runs, lengths, offsets = [], [], []
    for _ in range(n):
        v, i = _varint(buf, i)
        runs.append(v)
    for _ in range(n):
        v, i = _varint(buf, i)
        lengths.append(v)
    for k in range(n):
        v, i = _varint(buf, i)
        offsets.append(offsets[k - 1] + lengths[k - 1] if v == 0 and k > 0 else v - 1)
    return list(zip(ids, runs, lengths, offsets))


def _zxy_to_tile_id(z: int, x: int, y: int) -> int:
    """The Hilbert-curve tile id of the PMTiles spec."""
    acc = sum(4 ** t for t in range(z))
    d = 0
    tx, ty = x, y
    s = (1 << z) // 2
    while s > 0:
        rx = 1 if tx & s else 0
        ry = 1 if ty & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                tx, ty = s - 1 - tx, s - 1 - ty
            tx, ty = ty, tx
        s //= 2
    return acc + d


def _find_entry(entries: list[tuple[int, int, int, int]], tile_id: int) -> tuple[int, int, int, int] | None:
    lo, hi = 0, len(entries) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        cmp = tile_id - entries[mid][0]
        if cmp > 0:
            lo = mid + 1
        elif cmp < 0:
            hi = mid - 1
        else:
            return entries[mid]
    if hi >= 0:
        e = entries[hi]
        if e[1] == 0 or tile_id - e[0] < e[1]:
            return e
    return None


class _PMTiles:
    """Just enough of a PMTiles v3 reader: the header, the root directory, leaf directories on demand."""

    def __init__(self, url: str):
        self.url = url
        head = _fetch_range(url, 0, 127)
        if head[:7] != b"PMTiles" or head[7] != 3:
            raise ValueError(f"{url} is not a PMTiles v3 archive")
        (root_off, root_len, _meta_off, _meta_len, self.leaf_off, _leaf_len,
         self.data_off, _data_len) = struct.unpack_from("<8Q", head, 8)
        self.internal_codec, self.tile_codec = head[97], head[98]
        self.max_zoom = head[101]
        self.root = _parse_directory(_decompress(_fetch_range(url, root_off, root_len), self.internal_codec))
        self._leaves: OrderedDict[tuple[int, int], list[tuple[int, int, int, int]]] = OrderedDict()

    def _leaf(self, offset: int, length: int) -> list[tuple[int, int, int, int]]:
        key = (offset, length)
        if key in self._leaves:
            self._leaves.move_to_end(key)
            return self._leaves[key]
        entries = _parse_directory(_decompress(_fetch_range(self.url, self.leaf_off + offset, length),
                                               self.internal_codec))
        self._leaves[key] = entries
        while len(self._leaves) > 64:
            self._leaves.popitem(last=False)
        return entries

    def tile(self, z: int, x: int, y: int) -> bytes | None:
        tile_id = _zxy_to_tile_id(z, x, y)
        entries = self.root
        for _depth in range(4):
            e = _find_entry(entries, tile_id)
            if e is None:
                return None
            _tid, run, length, offset = e
            if run > 0:
                return _decompress(_fetch_range(self.url, self.data_off + offset, length), self.tile_codec)
            entries = self._leaf(offset, length)
        return None


# ── Mapbox Vector Tiles, the two attributes the stream layer carries ────────


def _pb_fields(buf: bytes):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(buf, i)
            yield field, v
        elif wire == 2:
            ln, i = _varint(buf, i)
            yield field, buf[i:i + ln]
            i += ln
        elif wire == 1:
            yield field, buf[i:i + 8]
            i += 8
        elif wire == 5:
            yield field, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _packed(buf: bytes) -> list[int]:
    out, i = [], 0
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _mvt_value(buf: bytes) -> Any:
    for field, v in _pb_fields(buf):
        if field == 1:
            return bytes(v).decode("utf-8", errors="replace")
        if field == 2:
            return struct.unpack("<f", v)[0]
        if field == 3:
            return struct.unpack("<d", v)[0]
        if field in (4, 5):
            return int(v)
        if field == 6:
            return _zigzag(int(v))
        if field == 7:
            return bool(v)
    return None


def decode_streams_tile(data: bytes, z: int, x: int, y: int) -> dict[int, dict[str, Any]]:
    """The reaches in one stream tile: ``river_id -> {"order", "lines"}``, each line a list of ``(gx, gy)``
    world coordinates at zoom ``z`` (tile pixel + tile origin, 4096 units a tile), in the archive's order:
    TDX-Hydro reaches are drawn from the downstream end to the upstream end."""
    out: dict[int, dict[str, Any]] = {}
    for field, layer in _pb_fields(data):
        if field != 3:
            continue
        keys: list[str] = []
        values: list[Any] = []
        features: list[bytes] = []
        extent = 4096
        for f, v in _pb_fields(layer):
            if f == 3:
                keys.append(bytes(v).decode("utf-8", errors="replace"))
            elif f == 4:
                values.append(_mvt_value(v))
            elif f == 5:
                extent = int(v)
            elif f == 2:
                features.append(v)
        scale = 4096 / extent
        ox, oy = x * 4096, y * 4096
        for fb in features:
            tags: list[int] = []
            geom: list[int] = []
            for f, v in _pb_fields(fb):
                if f == 2:
                    tags = _packed(v)
                elif f == 4:
                    geom = _packed(v)
            props = {keys[tags[k]]: values[tags[k + 1]] for k in range(0, len(tags) - 1, 2)
                     if tags[k] < len(keys) and tags[k + 1] < len(values)}
            rid = props.get("riverId")
            if rid is None:
                continue
            lines: list[list[tuple[float, float]]] = []
            cx = cy = 0
            i = 0
            cur: list[tuple[float, float]] | None = None
            while i < len(geom):
                cmd, count = geom[i] & 7, geom[i] >> 3
                i += 1
                for _ in range(count):
                    if cmd in (1, 2):
                        cx += _zigzag(geom[i])
                        cy += _zigzag(geom[i + 1])
                        i += 2
                        pt = (ox + cx * scale, oy + cy * scale)
                        if cmd == 1:
                            cur = [pt]
                            lines.append(cur)
                        elif cur is not None:
                            cur.append(pt)
            entry = out.setdefault(int(rid), {"order": None, "lines": []})
            if props.get("strahlerOrder") is not None:
                entry["order"] = int(props["strahlerOrder"])
            entry["lines"].extend(ln for ln in lines if len(ln) >= 2)
    return out


_ARCHIVE: dict[str, _PMTiles] = {}
_TILES: OrderedDict[tuple[int, int, int], dict[int, dict[str, Any]]] = OrderedDict()
_MAX_TILES = 256


def _streams() -> _PMTiles:
    if STREAMS_PMTILES not in _ARCHIVE:
        _ARCHIVE[STREAMS_PMTILES] = _PMTiles(STREAMS_PMTILES)
    return _ARCHIVE[STREAMS_PMTILES]


def _tile_reaches(z: int, x: int, y: int) -> dict[int, dict[str, Any]]:
    n = 1 << z
    if not (0 <= y < n):
        return {}
    x %= n
    key = (z, x, y)
    if key in _TILES:
        _TILES.move_to_end(key)
        return _TILES[key]
    data = _streams().tile(z, x, y)
    reaches = decode_streams_tile(data, z, x, y) if data else {}
    _TILES[key] = reaches
    while len(_TILES) > _MAX_TILES:
        _TILES.popitem(last=False)
    return reaches


# ── geometry ─────────────────────────────────────────────────────────────────

_EARTH_CIRCUMFERENCE_M = 40_075_016.686
_MAX_LAT = 85.05112878


def _world(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Web-Mercator world coordinates at zoom ``z``, 4096 units a tile."""
    n = (1 << z) * 4096
    lat = max(-_MAX_LAT, min(_MAX_LAT, lat))
    x = (lon + 180.0) / 360.0 * n
    r = math.radians(lat)
    y = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
    return x, y


def _lonlat(gx: float, gy: float, z: int) -> tuple[float, float]:
    n = (1 << z) * 4096
    lon = gx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * gy / n))))
    return lon, lat


def _metres_per_unit(lat: float, z: int) -> float:
    return _EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / ((1 << z) * 4096)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_008.8 * math.asin(math.sqrt(min(1.0, a)))


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float
                      ) -> tuple[float, float, float, float]:
    """Distance from P to segment AB, the parameter t along it, and the nearest point."""
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    t = 0.0 if ll == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), t, qx, qy


def _clip_segment(a: tuple[float, float], b: tuple[float, float], box: tuple[float, float, float, float]
                  ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky: the part of AB inside the box, or None."""
    x0, y0 = a
    dx, dy = b[0] - x0, b[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def _clip_polyline(pts: list[tuple[float, float]], box: tuple[float, float, float, float]
                   ) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for a, b in zip(pts, pts[1:]):
        seg = _clip_segment(a, b, box)
        if seg is None:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
            continue
        p, q = seg
        if not cur or cur[-1] != p:
            if len(cur) >= 2:
                out.append(cur)
            cur = [p]
        cur.append(q)
        if q != b:
            out.append(cur)
            cur = []
    if len(cur) >= 2:
        out.append(cur)
    return out


def _tiles_around(gx: float, gy: float, radius_units: float, z: int, cap: int = 16) -> list[tuple[int, int]]:
    n = 1 << z
    x0, x1 = int((gx - radius_units) // 4096), int((gx + radius_units) // 4096)
    y0, y1 = max(0, int((gy - radius_units) // 4096)), min(n - 1, int((gy + radius_units) // 4096))
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def gap(t: tuple[int, int]) -> float:
        cx = min(max(gx, t[0] * 4096), (t[0] + 1) * 4096)
        cy = min(max(gy, t[1] * 4096), (t[1] + 1) * 4096)
        return math.hypot(gx - cx, gy - cy)

    tiles.sort(key=gap)
    return [t for t in tiles if gap(t) <= radius_units][:cap]


# ── snap ─────────────────────────────────────────────────────────────────────


def _fmt_distance(m: float) -> str:
    """740 m, 1 km, 1.4 km."""
    if m < 1000:
        return f"{m:,.0f} m"
    km = f"{m / 1000:,.1f}"
    return f"{km[:-2] if km.endswith('.0') else km} km"


def _check_point(lat: Any, lon: Any) -> tuple[float, float]:
    lat_f, lon_f = float(lat), float(lon)
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        raise ValueError(f"not a point on Earth: lat {lat}, lon {lon}")
    return lat_f, lon_f


_Candidate = tuple[float, Any, float, float]  # distance (zoom-12 units), Strahler order, nearest gx, gy


def _candidates(lat: float, lon: float, radius_m: float) -> tuple[dict[int, _Candidate], float]:
    """Every reach with a line within ``radius_m`` of the point: ``river_id -> (distance in zoom-12 units, Strahler
    order, nearest gx, nearest gy)``, and the metres per unit at this latitude."""
    z = SNAP_ZOOM
    gx, gy = _world(lon, lat, z)
    mpu = _metres_per_unit(lat, z)
    best: dict[int, _Candidate] = {}
    for tx, ty in _tiles_around(gx, gy, radius_m / mpu, z):
        for rid, reach in _tile_reaches(z, tx, ty).items():
            for line in reach["lines"]:
                for a, b in zip(line, line[1:]):
                    d, _t, qx, qy = _segment_distance(gx, gy, a[0], a[1], b[0], b[1])
                    if d * mpu <= radius_m and (rid not in best or d < best[rid][0]):
                        best[rid] = (d, reach.get("order"), qx, qy)
    return best, mpu


def _as_reach(rid: int, item: _Candidate, mpu: float) -> dict[str, Any]:
    d, order, qx, qy = item
    qlon, qlat = _lonlat(qx, qy, SNAP_ZOOM)
    return {"river_id": rid, "strahler_order": order, "distance_m": round(d * mpu, 1),
            "lat": round(qlat, 6), "lon": round(qlon, 6)}


def main_channel(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The main stem among candidate reaches: the highest Strahler order, the nearer one on a tie.

    A click beside a big river is often nearer a small tributary than the river's mapped centreline (a braided
    river like the Jamuna is kilometres wide), and the nearest line then stands for a stream of a few m3/s.
    Stream order is the one size the tile archive carries, so it ranks; distance breaks ties."""
    if not candidates:
        return None
    return min(candidates, key=lambda r: (-(r.get("strahler_order") or 0), float(r.get("distance_m") or 0.0)))


def match_by_area(candidates: list[dict[str, Any]], area_km2: float | None) -> dict[str, Any] | None:
    """The candidate reach whose upstream area is closest (on a log scale) to a gauge's catchment area.

    Each reach's area is summed from its processing unit's routing tables (:func:`upstream_area`); a reach whose
    unit cannot be read is skipped. Returns ``{"reach", "upstream_area_km2", "area_ratio"}``, or None without an
    area or when no candidate's area could be read. The evidence ladder (:mod:`aquascope.evidence`) and a gauge's
    snap both use it, so a main-stem gauge gets the main stem."""
    try:
        target = float(area_km2) if area_km2 is not None else 0.0
    except (TypeError, ValueError):
        return None
    if not target > 0 or not candidates:
        return None
    scored = []
    for r in candidates:
        try:
            up = upstream_area(r["river_id"], lat=r.get("lat"), lon=r.get("lon"))["upstream_area_km2"]
        except Exception as exc:  # noqa: BLE001 - one reach whose unit cannot be read does not stop the others
            logger.info("upstream area unavailable for reach %s: %s", r["river_id"], exc)
            continue
        if up and float(up) > 0:
            scored.append((abs(math.log(float(up) / target)), float(r.get("distance_m") or 0.0), r, float(up)))
    if not scored:
        return None
    _gap, _d, r, up = min(scored, key=lambda x: (x[0], x[1]))
    return {"reach": r, "upstream_area_km2": round(up, 1), "area_ratio": round(up / target, 3)}


def snap_to_river(lat: float, lon: float, *, max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
                  prefer: str = "main", area_km2: float | None = None) -> dict[str, Any]:
    """The GEOGLOWS v2 river reach a point stands for, and how far it is.

    Reads the stream network tiles (zoom 12, where every reach is present) around the point and measures the
    distance to each reach's line. Every reach within ``max_distance_m`` is a candidate, and the main channel
    wins (``prefer="main"``, :func:`main_channel`): the highest Strahler order, the nearer on a tie. When a
    smaller stream was nearer, ``nearer`` names it and ``message`` says so, and ``prefer="nearest"`` takes it
    instead. With ``area_km2`` (a gauge's catchment area) the candidate whose upstream area matches it wins
    (:func:`match_by_area`), and ``choice`` is ``"area"``; the match is made only when the reaches in reach
    differ in stream order, since one river's neighbouring reaches need no telling apart.

    ``snapped`` is True and ``river_id``, ``strahler_order``, ``distance_m`` and the snapped position
    (``snap_lat``, ``snap_lon``) describe the chosen reach; ``choice`` says how it was chosen (``"nearest"``,
    ``"main_channel"`` or ``"area"``), ``n_candidates`` how many reaches were in reach and ``mixed_orders``
    whether they differ in stream order (where a gauge's area would decide). Beyond the tolerance
    the answer says there is no stream within that distance, and ``nearest`` still names the closest reach found
    (searched out to three times the tolerance, at least 3 km), so a caller can offer it rather than treat a
    hillside as a river.

    A braided river can be kilometres wide, so a click on its water may be more than the tolerance from the
    mapped centreline and close to a small stream. When a river at least :data:`LARGER_ORDER_GAP` orders bigger
    than the chosen (or nearest) reach lies beyond the tolerance but within the search, ``larger`` names it and
    ``message`` says how far it is, so a caller can offer it; it is never taken silently.

    Returns plain JSON, with ``message`` a sentence that says what happened.
    """
    lat, lon = _check_point(lat, lon)
    if prefer not in ("main", "nearest"):
        raise ValueError(f"prefer is 'main' or 'nearest', not {prefer!r}")
    max_d = max(1.0, float(max_distance_m))
    search_m = max(3.0 * max_d, 3000.0)
    found, mpu = _candidates(lat, lon, search_m)
    out: dict[str, Any] = {
        "lat": round(lat, 6), "lon": round(lon, 6), "max_distance_m": max_d, "searched_m": search_m,
        "snapped": False, "river_id": None, "strahler_order": None, "distance_m": None,
        "snap_lat": None, "snap_lon": None, "nearest": None, "nearer": None, "larger": None, "choice": None,
        "n_candidates": 0, "mixed_orders": False,
        "source": "GEOGLOWS v2 stream network (TDX-Hydro), streams.pmtiles",
        "attribution": ATTRIBUTION, "licence": LICENCE,
    }
    if not found:
        out["message"] = (f"No stream within {_fmt_distance(search_m)} of this point "
                          "(open water, or no river mapped here).")
        return out
    reaches = sorted((_as_reach(rid, item, mpu) for rid, item in found.items()), key=lambda r: r["distance_m"])
    nearest = reaches[0]
    out["nearest"] = nearest
    within = [r for r in reaches if r["distance_m"] <= max_d]
    out["n_candidates"] = len(within)
    out["mixed_orders"] = len({r["strahler_order"] for r in within}) > 1
    if not within:
        out["larger"] = _larger_beyond(reaches, max_d, nearest)
        out["message"] = (f"No stream within {_fmt_distance(max_d)} of this point. The nearest mapped reach is "
                          f"{nearest['river_id']}, {_fmt_distance(nearest['distance_m'])} away."
                          + _larger_sentence(out["larger"]))
        return out
    chosen, choice, matched = nearest, "nearest", None
    # Reaches of one order in reach are one river's neighbouring reaches (or rivers of a size): the area match
    # (which reads the unit's routing tables, up to about 30 MB) is spent only where the sizes differ.
    if area_km2 is not None and out["mixed_orders"]:
        matched = match_by_area(within, area_km2)
        if matched is not None:
            chosen, choice = matched["reach"], "area"
    if matched is None and prefer == "main":
        chosen = main_channel(within) or nearest
        if chosen["river_id"] != nearest["river_id"]:
            choice = "main_channel"
    out.update(snapped=True, river_id=chosen["river_id"], strahler_order=chosen["strahler_order"],
               distance_m=chosen["distance_m"], snap_lat=chosen["lat"], snap_lon=chosen["lon"], choice=choice)
    if matched is not None:
        out.update(upstream_area_km2=matched["upstream_area_km2"], area_ratio=matched["area_ratio"])
    if chosen["river_id"] != nearest["river_id"]:
        out["nearer"] = nearest
    # A gauge matched by its catchment area has its river; a bigger one further off is not offered.
    out["larger"] = None if matched is not None else _larger_beyond(reaches, max_d, chosen)
    order = chosen["strahler_order"]
    d = _fmt_distance(chosen["distance_m"])
    if choice == "main_channel":
        msg = (f"Snapped {d} to the main channel (river reach {chosen['river_id']}, order {order}); a smaller "
               f"stream is {_fmt_distance(nearest['distance_m'])} away.")
    elif choice == "area":
        msg = (f"Snapped {d} to river reach {chosen['river_id']}, the one whose upstream area "
               f"({matched['upstream_area_km2']:,.0f} km2) matches the catchment's")
        msg += (f"; the nearest line is {_fmt_distance(nearest['distance_m'])} away." if out["nearer"] else ".")
    else:
        msg = f"Snapped {d} to river reach {chosen['river_id']}" + (f", Strahler order {order}." if order is not None
                                                                     else ".")
    out["message"] = msg + _larger_sentence(out["larger"])
    return out


def _larger_beyond(reaches: list[dict[str, Any]], max_d: float, ref: dict[str, Any]) -> dict[str, Any] | None:
    """The main channel beyond the tolerance, when it is at least :data:`LARGER_ORDER_GAP` orders bigger than
    ``ref`` (the reach chosen, or the nearest one when none was in reach)."""
    beyond = main_channel([r for r in reaches if r["distance_m"] > max_d])
    if beyond is None or beyond.get("strahler_order") is None:
        return None
    if int(beyond["strahler_order"]) < int(ref.get("strahler_order") or 0) + LARGER_ORDER_GAP:
        return None
    return beyond


def _larger_sentence(larger: dict[str, Any] | None) -> str:
    if not larger:
        return ""
    return (f" A larger river (reach {larger['river_id']}, order {larger['strahler_order']}) is "
            f"{_fmt_distance(larger['distance_m'])} away.")


def reaches_near(lat: float, lon: float, *, max_distance_m: float = 2000.0, limit: int = 6) -> list[dict[str, Any]]:
    """The distinct river reaches within ``max_distance_m`` of a point, nearest first (at most ``limit``).

    The same tile read as :func:`snap_to_river`, keeping every reach rather than the chosen one: a gauge on a
    main stem sits a few hundred metres from its tributaries too, and the reach that matches it is the one whose
    upstream area matches the gauge's catchment (:func:`match_by_area`), not always the nearest line.
    Each item is ``{"river_id", "strahler_order", "distance_m", "lat", "lon"}``.
    """
    lat, lon = _check_point(lat, lon)
    found, mpu = _candidates(lat, lon, max(1.0, float(max_distance_m)))
    reaches = sorted((_as_reach(rid, item, mpu) for rid, item in found.items()), key=lambda r: r["distance_m"])
    return reaches[: max(1, int(limit))]


# ── the reach record ─────────────────────────────────────────────────────────


def _river_id(river_id: Any) -> int:
    try:
        rid = int(str(river_id).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not a GEOGLOWS river_id: {river_id!r}") from exc
    if rid // 10_000_000 not in _VPU_BY_PREFIX:
        raise ValueError(f"not a GEOGLOWS v2 river_id: {river_id!r} (expected a 9-digit TDX-Hydro LINKNO)")
    return rid


def reach_record(river_id: int | str | None = None, *, lat: float | None = None, lon: float | None = None,
                 max_distance_m: float = DEFAULT_MAX_DISTANCE_M, years: int | None = None,
                 return_periods: list[float] | None = None, store: dict[str, Any] | None = None) -> dict[str, Any]:
    """A river reach's simulated daily discharge since 1940, analysed the way a gauge is.

    Fetches the GEOGLOWS v2 retrospective (daily, m3/s) for ``river_id`` and runs
    :func:`aquascope.explore.analyze_series` on it: the hydrograph, annual maxima, return periods (GEV by
    L-moments and Log-Pearson III with 90 % confidence limits), the flow-duration curve and the Mann-Kendall
    trend. Adds the monthly regime (mean and the 10th and 90th percentiles of daily flow for each calendar
    month). ``years`` keeps only the last N years. Every payload says ``modelled: True``: this is a model, not a
    measurement. ``store`` (a dict) receives the full daily series under ``"series"``.

    With ``lat``/``lon`` instead of ``river_id`` the point is snapped first (:func:`snap_to_river`); the snap
    travels with the answer under ``snap``, and a point with no stream within ``max_distance_m`` gets an
    ``error`` that says so rather than the record of a river it is not on.
    """
    import pandas as pd

    from aquascope.explore import METHODS, analyze_series

    snap: dict[str, Any] | None = None
    if river_id is None:
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon to snap to the nearest reach")
        snap = snap_to_river(lat, lon, max_distance_m=max_distance_m)
        if not snap["snapped"]:
            return {"river_id": None, "modelled": True, "label": "modelled", "snap": snap, "error": snap["message"],
                    "attribution": ATTRIBUTION, "licence": LICENCE["discharge"], "notes": [MODELLED_NOTE]}
        river_id = snap["river_id"]
    rid = _river_id(river_id)
    params: dict[str, Any] = {"format": "json"}
    if years:
        start = datetime.now(timezone.utc).date() - timedelta(days=int(float(years) * 365.25))
        params["start_date"] = start.strftime("%Y%m%d")
    url = f"{GEOGLOWS_API}/retrospectivedaily/{rid}"
    data = _fetch_json(url, params)
    base: dict[str, Any] = {"river_id": rid, "modelled": True, "label": "modelled",
                            "source": "GEOGLOWS v2 retrospective simulation", "url": f"{url}?format=json",
                            "attribution": ATTRIBUTION, "licence": LICENCE["discharge"]}
    values = data.get(str(rid)) if isinstance(data, dict) else None
    times = data.get("datetime") if isinstance(data, dict) else None
    if not values or not times or len(values) != len(times):
        return {**base, "error": "GEOGLOWS returned no simulated flow for this reach.", "notes": [MODELLED_NOTE]}
    idx = pd.to_datetime(list(times), utc=True).tz_convert(None)
    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float), index=idx)
    s = s.where(s >= 0).dropna()
    res = analyze_series(s, "discharge", "m3/s", return_periods=return_periods)
    if len(s):
        by_month = s.groupby(s.index.month)
        res["monthly_regime"] = {
            "month": list(range(1, 13)),
            "mean": [_num(by_month.mean().get(m)) for m in range(1, 13)],
            "p10": [_num(by_month.quantile(0.10).get(m)) for m in range(1, 13)],
            "p90": [_num(by_month.quantile(0.90).get(m)) for m in range(1, 13)],
        }
    res.update(base)
    if snap is not None:
        res["snap"] = snap
    res["notes"] = [MODELLED_NOTE, *res.get("notes", [])]
    res["methods"] = [METHODS["geoglows"], *res.get("methods", [])]
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if meta.get("gen_date"):
        res["generated"] = meta["gen_date"]
    if store is not None:
        store["series"] = s
    return res


def reach_summary(river_id: int | str | None = None, *, lat: float | None = None, lon: float | None = None,
                  years: int | None = None, return_periods: list[float] | None = None,
                  max_distance_m: float = DEFAULT_MAX_DISTANCE_M) -> dict[str, Any]:
    """:func:`reach_record` without the daily arrays: the statistics, the return periods, the flow-duration
    percentiles, the monthly regime and the annual maxima. What the MCP server, the Analyst and the Studio
    quote; the full record stays with :func:`reach_record`."""
    res = reach_record(river_id, lat=lat, lon=lon, years=int(years) if years else None,
                       return_periods=return_periods, max_distance_m=max_distance_m)
    res.pop("series", None)
    res.pop("series_downsampled", None)
    if isinstance(res.get("fdc"), dict):
        res["fdc"] = {k: res["fdc"][k] for k in ("q95", "q50", "q10") if k in res["fdc"]}
    return res


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if math.isfinite(v) else None


def forecast_stats(river_id: int | str) -> dict[str, Any]:
    """The GEOGLOWS 15-day ensemble forecast statistics for a reach (hourly then 3-hourly), modelled.

    A thin fetch: ``datetime`` and the ensemble ``flow_avg``, ``flow_med``, ``flow_25p``, ``flow_75p``,
    ``flow_min``, ``flow_max`` and ``high_res`` lists in m3/s, gaps as ``None``.
    """
    rid = _river_id(river_id)
    url = f"{GEOGLOWS_API}/forecaststats/{rid}"
    data = _fetch_json(url, {"format": "json"})
    out: dict[str, Any] = {"river_id": rid, "modelled": True, "label": "modelled",
                           "source": "GEOGLOWS v2 forecast (ECMWF ensemble)", "url": f"{url}?format=json",
                           "attribution": ATTRIBUTION, "licence": LICENCE["discharge"], "unit": "m3/s"}
    if not isinstance(data, dict) or not data.get("datetime"):
        return {**out, "error": "GEOGLOWS returned no forecast for this reach."}
    out["datetime"] = list(data["datetime"])
    for key in ("flow_avg", "flow_med", "flow_25p", "flow_75p", "flow_min", "flow_max", "high_res"):
        if isinstance(data.get(key), list):
            out[key] = [_num(v) if v not in ("", None) else None for v in data[key]]
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    if meta.get("gen_date"):
        out["generated"] = meta["gen_date"]
    return out


# ── the network: topology and unit-catchment areas, one VPU at a time ───────


class _Network:
    """One VPU's reaches: ids, the downstream id of each, unit-catchment area and an approximate centre."""

    def __init__(self, vpu: int, ids: Any, ds: Any, area_m2: Any, lon: Any, lat: Any):
        import numpy as np

        self.vpu = vpu
        self.ids = np.asarray(ids, dtype=np.int64)
        self.ds = np.asarray(ds, dtype=np.int64)
        self.area_m2 = np.asarray(area_m2, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.lat = np.asarray(lat, dtype=float)
        self._order = np.argsort(self.ids, kind="stable")
        self._sorted = self.ids[self._order]
        self._ds_order = np.argsort(self.ds, kind="stable")
        self._ds_sorted = self.ds[self._ds_order]

    def index(self, rid: int) -> int | None:
        import numpy as np

        k = int(np.searchsorted(self._sorted, rid))
        if k < len(self._sorted) and int(self._sorted[k]) == rid:
            return int(self._order[k])
        return None

    def __contains__(self, rid: int) -> bool:
        return self.index(rid) is not None

    def children(self, i: int) -> list[int]:
        import numpy as np

        rid = self.ids[i]
        lo = int(np.searchsorted(self._ds_sorted, rid, side="left"))
        hi = int(np.searchsorted(self._ds_sorted, rid, side="right"))
        return [int(j) for j in self._ds_order[lo:hi]]

    def upstream(self, i: int, cap: int = 2_000_000) -> list[int]:
        out, stack = [], [i]
        while stack and len(out) < cap:
            j = stack.pop()
            out.append(j)
            stack.extend(self.children(j))
        return out


_NETWORKS: OrderedDict[int, _Network] = OrderedDict()
_UPSTREAM: OrderedDict[tuple[int, int], list[int]] = OrderedDict()


def _upstream_of(net: _Network, i: int) -> list[int]:
    """The reaches upstream of reach index ``i`` (its own included), kept for the last few asked: the trace, the
    upstream area and the upstream dams all walk the same basin."""
    key = (net.vpu, int(i))
    if key in _UPSTREAM:
        _UPSTREAM.move_to_end(key)
        return _UPSTREAM[key]
    ups = net.upstream(i)
    _UPSTREAM[key] = ups
    while len(_UPSTREAM) > 4:
        _UPSTREAM.popitem(last=False)
    return ups


def _network(vpu: int) -> _Network:
    import numpy as np
    import pandas as pd

    if vpu in _NETWORKS:
        _NETWORKS.move_to_end(vpu)
        return _NETWORKS[vpu]
    connect = pd.read_csv(io.StringIO(_fetch_text(ROUTING_CONFIG.format(vpu=vpu, name="rapid_connect.csv"))),
                          header=None, usecols=[0, 1])
    ids = connect.iloc[:, 0].to_numpy(dtype=np.int64)
    ds = connect.iloc[:, 1].to_numpy(dtype=np.int64)
    weights = pd.read_csv(io.StringIO(_fetch_text(ROUTING_CONFIG.format(vpu=vpu, name="weight_era5_721x1440.csv"))),
                          usecols=["LINKNO", "lon", "lat", "area_sqm"])
    weights["wlon"] = weights["lon"] * weights["area_sqm"]
    weights["wlat"] = weights["lat"] * weights["area_sqm"]
    g = weights.groupby("LINKNO")[["area_sqm", "wlon", "wlat"]].sum()
    g = g.reindex(ids)
    area = g["area_sqm"].fillna(0.0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = (g["wlon"] / g["area_sqm"]).to_numpy()
        lat = (g["wlat"] / g["area_sqm"]).to_numpy()
    net = _Network(vpu, ids, ds, area, lon, lat)
    _NETWORKS[vpu] = net
    while len(_NETWORKS) > 3:
        _NETWORKS.popitem(last=False)
    return net


def _vpu_candidates(rid: int, lat: float | None = None, lon: float | None = None) -> list[int]:
    cands = list(_VPU_BY_PREFIX.get(rid // 10_000_000, ()))
    if lat is None or lon is None:
        return cands

    def gap(v: int) -> float:
        w, s, e, n = _VPU_BBOX.get(v, (-180.0, -90.0, 180.0, 90.0))
        dx = max(w - lon, 0.0, lon - e)
        dy = max(s - lat, 0.0, lat - n)
        return math.hypot(dx, dy)

    return sorted(cands, key=gap)


def _network_for(rid: int, lat: float | None = None, lon: float | None = None) -> _Network:
    for vpu in _vpu_candidates(rid, lat, lon):
        net = _network(vpu)
        if rid in net:
            return net
    raise LookupError(f"river_id {rid} is not in any GEOGLOWS v2 processing unit")


AREA_METHOD = ("The unit-catchment areas of every reach upstream, added up: the area_sqm column of GEOGLOWS's "
               "ERA5 weight table for the processing unit, walked up its rapid_connect topology.")
AREA_NOTE = ("Checked on 2026-10-08 against the catchment areas four agencies publish for their gauges (USGS 07010000 "
             "Mississippi at St. Louis, NRFA 39001 Thames at Kingston, FOEN 2135 Aare at Bern, FOEN 2289 Rhine at "
             "Basel): from 3.7 % under to 6.6 % over, so read it as an approximate area.")


def upstream_area(river_id: int | str, *, lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    """The area that drains to a reach's downstream end, in km2, and how many reaches lie upstream.

    Adds up the GEOGLOWS v2 unit catchments upstream of ``river_id`` (the reach's own included), from the
    processing unit's routing tables (``rapid_connect.csv`` and the ERA5 weight table, a few MB each, read
    once and kept). ``lat``/``lon`` (where the reach is, roughly) only decide which unit is read first.
    """
    rid = _river_id(river_id)
    net = _network_for(rid, lat, lon)
    i = net.index(rid)
    assert i is not None
    ups = _upstream_of(net, i)
    total = float(net.area_m2[ups].sum())
    return {"river_id": rid, "vpu": net.vpu, "upstream_area_km2": round(total / 1e6, 1),
            "unit_area_km2": round(float(net.area_m2[i]) / 1e6, 2), "n_reaches_upstream": len(ups),
            "method": AREA_METHOD, "note": AREA_NOTE, "modelled": True, "attribution": ATTRIBUTION}


def upstream_dams(river_id: int | str | None = None, **kwargs: Any) -> dict[str, Any]:
    """The Global Dam Watch dams upstream of a reach and the degree of regulation there; see
    :func:`aquascope.river_path.upstream_dams`."""
    from aquascope.river_path import upstream_dams as _upstream_dams

    res: dict[str, Any] = _upstream_dams(river_id, **kwargs)
    return res


# ── the network around a reach, as ids: what the map lights up (#545) ───────


def _upstream_tree(net: _Network, i: int, cap: int = 3_000_000) -> tuple[Any, Any, list[int]]:
    """Breadth-first up the network from reach index ``i``, a whole level at a time (numpy, no per-reach loop:
    the Mississippi at St. Louis, 123,896 reaches, in about 0.05 s): the reach indices in visiting order, the
    position of each one's downstream neighbour in that order (-1 for the start), and where each level begins."""
    import numpy as np

    nodes = [np.array([i], dtype=np.int64)]
    parents = [np.array([-1], dtype=np.int64)]
    starts = [0]
    frontier = nodes[0]
    first = 0  # position of the frontier's first node in the whole order
    total = 1
    while frontier.size and total < cap:
        rids = net.ids[frontier]
        lo = np.searchsorted(net._ds_sorted, rids, side="left")
        hi = np.searchsorted(net._ds_sorted, rids, side="right")
        counts = hi - lo
        n = int(counts.sum())
        if n == 0:
            break
        offsets = np.repeat(lo - (np.cumsum(counts) - counts), counts)
        children = net._ds_order[offsets + np.arange(n)].astype(np.int64)
        parent_pos = np.repeat(np.arange(first, first + frontier.size, dtype=np.int64), counts)
        starts.append(total)
        nodes.append(children)
        parents.append(parent_pos)
        first = total
        total += n
        frontier = children
    return np.concatenate(nodes), np.concatenate(parents), starts


UPSTREAM_RULE = ("Every reach that drains to this one, from GEOGLOWS's rapid_connect topology; when there are more "
                 "than max_n, the max_n with the largest drainage area (the main stems and big tributaries) are kept.")


def upstream_ids(river_id: int | str, max_n: int = 20_000, *, lat: float | None = None,
                 lon: float | None = None) -> dict[str, Any]:
    """The river_ids of the reaches upstream of a reach (its own first), for lighting the network up on a map.

    Reads the processing unit's routing tables like :func:`upstream_area`. A big basin has hundreds of thousands
    of reaches; past ``max_n`` the ones with the largest drainage area are kept, so the trunk and the big
    tributaries stay and the smallest headwaters go, and ``min_area_km2`` says where the cut fell. ``ids`` are in
    breadth-first order from the reach upward. ``lat``/``lon`` only decide which unit is read first.
    """
    import numpy as np

    rid = _river_id(river_id)
    max_n = max(1, int(max_n))
    net = _network_for(rid, lat, lon)
    i = net.index(rid)
    assert i is not None
    order, parent, starts = _upstream_tree(net, i)
    acc = net.area_m2[order].astype(float)
    acc[~np.isfinite(acc)] = 0.0
    # Drainage area of every reach in the tree: each level, deepest first, added into its downstream neighbour.
    bounds = [*starts, len(order)]
    for k in range(len(starts) - 1, 0, -1):
        sl = slice(bounds[k], bounds[k + 1])
        np.add.at(acc, parent[sl], acc[sl])
    n = len(order)
    keep = np.arange(n)
    truncated = n > max_n
    min_area = None
    if truncated:
        keep = np.sort(np.argpartition(-acc, max_n - 1)[:max_n])
        min_area = round(float(acc[keep].min()) / 1e6, 1)
    ids = [int(x) for x in net.ids[order[keep]]]
    area_km2 = round(float(acc[0]) / 1e6, 1)
    message = (f"{n:,} reaches drain to river reach {rid} ({area_km2:,.0f} km2)"
               + (f"; the {max_n:,} largest, each draining at least {min_area:,.0f} km2, are listed." if truncated
                  else "."))
    return {"river_id": rid, "vpu": net.vpu, "ids": ids, "n_upstream": n, "n_ids": len(ids),
            "truncated": truncated, "min_area_km2": min_area, "upstream_area_km2": area_km2,
            "message": message, "method": UPSTREAM_RULE, "modelled": True, "attribution": ATTRIBUTION}


def downstream_ids(river_id: int | str, max_n: int = 5_000, *, lat: float | None = None,
                   lon: float | None = None) -> dict[str, Any]:
    """The river_ids from a reach down to its outlet (the reach first, the outlet last): the path to the sea, or to
    an inland sink, as GEOGLOWS's routing tables have it. Stops after ``max_n`` reaches and says so."""
    rid = _river_id(river_id)
    max_n = max(1, int(max_n))
    net = _network_for(rid, lat, lon)
    i = net.index(rid)
    assert i is not None
    path = [rid]
    seen = {i}
    while len(path) < max_n:
        nxt = net.index(int(net.ds[i]))
        if nxt is None or nxt in seen:
            break
        seen.add(nxt)
        i = nxt
        path.append(int(net.ids[i]))
    nxt = net.index(int(net.ds[i]))
    truncated = nxt is not None and nxt not in seen
    message = (f"{len(path):,} reaches from river reach {rid} to "
               + ("where the list stops; the river goes on." if truncated
                  else f"the outlet, reach {path[-1]} (the sea, or an inland sink)."))
    return {"river_id": rid, "vpu": net.vpu, "ids": path, "n_ids": len(path), "outlet_id": path[-1],
            "truncated": truncated, "message": message, "modelled": True, "attribution": ATTRIBUTION}


# ── the trace to the outlet ──────────────────────────────────────────────────


def _zooms_for(order: int | None) -> list[int]:
    """Coarse tiles for big rivers (fewer reads), fine ones for streams; zoom 12 is the last resort."""
    if order is not None and order >= 4:
        return [8, 10, 12]
    return [10, 12]


def _collect_pieces(rid: int, lon: float, lat: float, z: int, max_tiles: int = 48
                    ) -> tuple[list[list[tuple[float, float]]], int | None]:
    """Every piece of reach ``rid`` at zoom ``z``, each clipped to its own tile, starting near (lon, lat) and
    following the line into the neighbouring tiles it runs into."""
    gx, gy = _world(lon, lat, z)
    seed = (int(gx // 4096), int(gy // 4096))
    seeds = [seed] + [(seed[0] + dx, seed[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    pieces: list[list[tuple[float, float]]] = []
    order: int | None = None
    seen: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []
    for s in seeds:
        if s in seen:
            continue
        queue.append(s)
        while queue and len(seen) < max_tiles:
            t = queue.pop()
            if t in seen:
                continue
            seen.add(t)
            reach = _tile_reaches(z, t[0], t[1]).get(rid)
            if not reach:
                continue
            order = reach.get("order") if reach.get("order") is not None else order
            box = (t[0] * 4096.0, t[1] * 4096.0, (t[0] + 1) * 4096.0, (t[1] + 1) * 4096.0)
            for line in reach["lines"]:
                pieces.extend(_clip_polyline(line, box))
                for px, py in line:
                    if not (box[0] <= px <= box[2] and box[1] <= py <= box[3]):
                        nxt = (int(px // 4096), int(py // 4096))
                        if nxt not in seen:
                            queue.append(nxt)
        if pieces:
            break
    return pieces, order


def _chain(pieces: list[list[tuple[float, float]]], tol: float = 6.0) -> list[tuple[float, float]]:
    """Join a reach's pieces into one line, keeping the downstream-to-upstream direction they carry."""
    def gap(a: tuple[float, float], b: tuple[float, float]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rest = [p for p in pieces if len(p) >= 2]
    if not rest:
        return []
    start = next((p for p in rest if not any(q is not p and gap(q[-1], p[0]) <= tol for q in rest)), rest[0])
    rest.remove(start)
    line = list(start)
    while rest:
        nxt = min(rest, key=lambda p: gap(line[-1], p[0]))
        rest.remove(nxt)
        line.extend(nxt[1:] if gap(line[-1], nxt[0]) <= tol else nxt)
    return line


def _reach_line(rid: int, lon: float, lat: float, order_hint: int | None, *, first_zoom_only: bool = False
                ) -> tuple[list[tuple[float, float]], int | None]:
    """The reach as ``(lon, lat)`` points from its downstream end to its upstream end, and its Strahler order."""
    zooms = _zooms_for(order_hint)
    for z in zooms[:1] if first_zoom_only else zooms:
        pieces, order = _collect_pieces(rid, lon, lat, z)
        if pieces:
            return [_lonlat(px, py, z) for px, py in _chain(pieces)], order
    return [], None


def _line_km(pts: list[tuple[float, float]]) -> float:
    return sum(_haversine_m(a[1], a[0], b[1], b[0]) for a, b in zip(pts, pts[1:])) / 1000.0


class PathIndex:
    """Where points lie relative to a polyline of ``[lon, lat]``: how far off it and how far along it, in km.

    A coarse grid of the segments keeps each lookup to the few segments nearby. The gauges and the dams along a
    trace are both placed with it.
    """

    def __init__(self, coords: list[list[float]], max_km: float):
        self.coords = coords
        self.max_km = float(max_km)
        self.cell = c = max(0.02, self.max_km / 111.0)
        self.grid: dict[tuple[int, int], list[int]] = {}
        for k, (a, b) in enumerate(zip(coords, coords[1:])):
            for cx in range(int(math.floor(min(a[0], b[0]) / c)), int(math.floor(max(a[0], b[0]) / c)) + 1):
                for cy in range(int(math.floor(min(a[1], b[1]) / c)), int(math.floor(max(a[1], b[1]) / c)) + 1):
                    self.grid.setdefault((cx, cy), []).append(k)
        self.cum = [0.0]
        for a, b in zip(coords, coords[1:]):
            self.cum.append(self.cum[-1] + _haversine_m(a[1], a[0], b[1], b[0]) / 1000.0)
        lons = [p[0] for p in coords] or [0.0]
        lats = [p[1] for p in coords] or [0.0]
        pad_lat = self.max_km / 111.0
        pad_lon = pad_lat / max(0.1, math.cos(math.radians(max(abs(min(lats)), abs(max(lats))))))
        self.bbox = (min(lons) - pad_lon, min(lats) - pad_lat, max(lons) + pad_lon, max(lats) + pad_lat)

    @property
    def length_km(self) -> float:
        return self.cum[-1]

    def locate(self, lat: float, lon: float) -> tuple[float, float, int] | None:
        """``(distance_km, along_km, segment)`` of the nearest point on the path, or None beyond ``max_km``."""
        if len(self.coords) < 2:
            return None
        west, south, east, north = self.bbox
        if not (south <= lat <= north and west <= lon <= east):
            return None
        c = self.cell
        reach_cells = int(math.ceil(self.max_km / (111.0 * max(0.05, math.cos(math.radians(lat)))) / c))
        gcx, gcy = int(math.floor(lon / c)), int(math.floor(lat / c))
        segs: set[int] = set()
        for cx in range(gcx - reach_cells, gcx + reach_cells + 1):
            for cy in range(gcy - 1, gcy + 2):
                segs.update(self.grid.get((cx, cy), ()))
        kx = 111.320 * math.cos(math.radians(lat))
        best: tuple[float, float, int] | None = None
        for k in segs:
            a, b = self.coords[k], self.coords[k + 1]
            d, t, _qx, _qy = _segment_distance(0.0, 0.0, (a[0] - lon) * kx, (a[1] - lat) * 110.574,
                                               (b[0] - lon) * kx, (b[1] - lat) * 110.574)
            if best is None or d < best[0]:
                best = (d, self.cum[k] + t * (self.cum[k + 1] - self.cum[k]), k)
        if best is None or best[0] > self.max_km:
            return None
        return best


def _gauges_along(coords: list[list[float]], rows: list[dict[str, Any]], max_km: float, limit: int = 60
                  ) -> list[dict[str, Any]]:
    """Catalog stations within ``max_km`` of the path, in the order the path reaches them."""
    if len(coords) < 2:
        return []
    index = PathIndex(coords, max_km)
    found: list[dict[str, Any]] = []
    for r in rows:
        try:
            glat, glon = float(r.get("latitude")), float(r.get("longitude"))
        except (TypeError, ValueError):
            continue
        loc = index.locate(glat, glon)
        if loc is None:
            continue
        found.append({"source": r.get("source"), "station_id": r.get("station_id"), "name": r.get("name"),
                      "latitude": glat, "longitude": glon, "variables": list(r.get("variables") or []),
                      "distance_km": round(loc[0], 2), "along_km": round(loc[1], 1)})
    found.sort(key=lambda g: (g["along_km"], g["distance_km"]))
    return found[:limit]


def _thin(coords: list[list[float]], cap: int) -> list[list[float]]:
    if len(coords) <= cap:
        return coords
    step = math.ceil(len(coords) / cap)
    out = coords[::step]
    if out[-1] != coords[-1]:
        out.append(coords[-1])
    return out


def trace_downstream(
    river_id: int | str | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    gauge_km: float = 2.0,
    max_reaches: int = 5000,
    stations: list[dict[str, Any]] | None = None,
    max_points: int = 6000,
    dam_km: float = 2.0,
    path_context: bool = True,
    upstream_cells: int = 24,
) -> dict[str, Any]:
    """Follow a reach down the network to its outlet: the path, its length, the gauges, dams and countries on it.

    Give ``river_id``, or ``lat``/``lon`` to snap first (:func:`snap_to_river`). The topology is GEOGLOWS's own
    (``rapid_connect.csv`` of the reach's processing unit); the geometry is read from the stream tiles reach by
    reach, so the path can be drawn on a map. Catalog stations within ``gauge_km`` of the path are listed in the
    order the water reaches them (``stations`` overrides the catalog). The answer also carries the upstream area
    of the first reach.

    With ``path_context`` (the default) it adds what the river passes, from :mod:`aquascope.river_path`: the
    Global Dam Watch dams within ``dam_km`` of the path (``dams``, with capacity, purpose, the degree of
    regulation where GDW gives it and the km along the path), the countries crossed (``countries``, Natural
    Earth admin-0) and the dams upstream of the first reach (``upstream_dams``, searched when the basin spans at
    most ``upstream_cells`` 2-degree cells). Each part says when its data is not available and the rest stands.

    The geometry is TDX-Hydro (CC BY-SA 4.0): return it for display, do not republish it as a product.
    """
    snap: dict[str, Any] | None = None
    start_lonlat: tuple[float, float] | None = None
    if river_id is None:
        if lat is None or lon is None:
            raise ValueError("give a river_id, or lat and lon to snap to the nearest reach")
        snap = snap_to_river(lat, lon, max_distance_m=max_distance_m)
        if not snap["snapped"]:
            return {"snapped": False, "snap": snap, "message": snap["message"], "attribution": ATTRIBUTION}
        river_id = snap["river_id"]
        start_lonlat = (float(snap["snap_lon"]), float(snap["snap_lat"]))
    rid = _river_id(river_id)
    net = _network_for(rid, lat, lon)
    i0 = net.index(rid)
    assert i0 is not None
    if start_lonlat is None:
        if lat is not None and lon is not None:
            start_lonlat = (float(lon), float(lat))
        elif math.isfinite(float(net.lon[i0])) and math.isfinite(float(net.lat[i0])):
            start_lonlat = (float(net.lon[i0]), float(net.lat[i0]))
    notes: list[str] = []
    if start_lonlat is None:
        notes.append("No location for the reach, so the path is not drawn.")

    path = [rid]
    i = i0
    while len(path) < max_reaches:
        nxt = net.index(int(net.ds[i]))
        if nxt is None:
            break
        i = nxt
        path.append(int(net.ids[i]))
    truncated = len(path) >= max_reaches and net.index(int(net.ds[i])) is not None
    if truncated:
        notes.append(f"Stopped after {max_reaches} reaches; the river goes on.")

    coords: list[list[float]] = []
    coord_reach: list[int] = []  # the reach each vertex belongs to
    reaches: list[dict[str, Any]] = []
    missing = 0
    run = 0  # reaches in a row with no line of their own
    here = start_lonlat
    order_hint = snap.get("strahler_order") if snap else None
    for k, r in enumerate(path):
        line: list[tuple[float, float]] = []
        order: int | None = None
        if here is not None:
            # The stream tiles are map-optimised: a reach with no line of its own is usually drawn as part of the
            # next one down, so a miss at the usual zoom is accepted and the walk goes on from the same place. A
            # long run of misses, or the first reach, gets the finer zooms too.
            line, order = _reach_line(r, here[0], here[1], order_hint, first_zoom_only=k > 0 and run < 8)
        if not line and (here is None or k == 0):
            j = net.index(r)
            if j is not None and math.isfinite(float(net.lon[j])) and math.isfinite(float(net.lat[j])):
                line, order = _reach_line(r, float(net.lon[j]), float(net.lat[j]), order_hint)
        if not line:
            missing += 1
            run += 1
            reaches.append({"river_id": r, "strahler_order": None, "length_km": None, "drawn": False})
            continue
        run = 0
        down = list(reversed(line))  # upstream end first: the direction the water goes
        if k == 0 and snap is not None and here is not None:
            # start at the snapped point, not at the top of the reach
            j = min(range(len(down)), key=lambda m: _haversine_m(here[1], here[0], down[m][1], down[m][0]))
            down = [here, *down[j + 1:]] if j + 1 < len(down) else [here, down[-1]]
        reaches.append({"river_id": r, "strahler_order": order, "length_km": round(_line_km(down), 2),
                        "drawn": True})
        for pt in down:
            p = [round(pt[0], 5), round(pt[1], 5)]
            if not coords or coords[-1] != p:
                coords.append(p)
                coord_reach.append(r)
        here = down[-1]
        order_hint = order if order is not None else order_hint
    geometry_complete = bool(coords) and reaches[-1].get("drawn", False)
    if missing:
        notes.append(f"{missing} of {len(path)} reaches have no line of their own in the stream tiles: the map "
                     "draws most of them as part of the next reach down.")
    notes.append("The length is measured along the drawn line, which is simplified, so it runs a little short "
                 "of the river's.")

    length_km = round(sum(x["length_km"] or 0.0 for x in reaches), 1)
    gauges: list[dict[str, Any]] = []
    if coords:
        rows = stations
        if rows is None:
            try:
                from aquascope.archive.catalog import load_stations

                rows = load_stations()
            except Exception as exc:  # noqa: BLE001 - the path stands without the gauges
                rows = []
                notes.append(f"Gauges along the path unavailable: {exc}")
        gauges = _gauges_along(coords, rows or [], float(gauge_km))
    area: dict[str, Any] | None = None
    try:
        area = upstream_area(rid, lat=lat, lon=lon)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Upstream area unavailable: {exc}")
    dams: dict[str, Any] | None = None
    countries: dict[str, Any] | None = None
    up_dams: dict[str, Any] | None = None
    if path_context:
        from aquascope import river_path

        if coords:
            dams = river_path.dams_along(coords, reach_of_segment=coord_reach[1:], dam_km=dam_km,
                                         min_catchment_km2=area.get("upstream_area_km2") if area else None)
            countries = river_path.countries_along(coords)
            if dams.get("note") and not dams.get("available"):
                notes.append(dams["note"])
        try:
            up_dams = river_path.upstream_dams(rid, lat=lat, lon=lon, with_flow=False, limit=5,
                                               max_cells=upstream_cells, max_tiles=upstream_cells)
        except Exception as exc:  # noqa: BLE001 - the path stands without them
            up_dams = {"available": False, "summary": f"Dams upstream unavailable: {exc}"}
    last_drawn = bool(reaches) and bool(reaches[-1].get("drawn"))
    outlet = {"river_id": path[-1], "lon": coords[-1][0] if coords and last_drawn else None,
              "lat": coords[-1][1] if coords and last_drawn else None}
    where = (f" at {outlet['lat']:.3f}, {outlet['lon']:.3f}" if outlet["lat"] is not None else "")
    message = (f"{len(path)} reaches, {length_km:,.0f} km to the outlet{where}"
               + ("" if truncated else ", where the network ends (the sea, or an inland sink)") + ". "
               + (f"{len(gauges)} gauge{'s' if len(gauges) != 1 else ''} within {gauge_km:g} km of the path"
                  + (f", {dams['n_dams']} dam{'s' if dams['n_dams'] != 1 else ''}"
                     if dams and dams.get("available") else "") + "."
                  if coords else "The path could not be drawn."))
    return {
        "snapped": True, "river_id": rid, "vpu": net.vpu, "snap": snap,
        "n_reaches": len(path), "length_km": length_km, "truncated": truncated,
        "outlet": outlet, "reaches": reaches,
        "geometry": {"type": "LineString", "coordinates": _thin(coords, max_points)},
        "geometry_complete": geometry_complete,
        "geometry_licence": "TDX-Hydro, CC BY-SA 4.0: for display; a product derived from it is share-alike.",
        "gauges": gauges, "gauge_km": float(gauge_km),
        "dams": (dams or {}).get("dams", []),
        "dams_info": {k: v for k, v in dams.items() if k != "dams"} if dams else None,
        "countries": (countries or {}).get("countries", []),
        "countries_info": {k: v for k, v in countries.items() if k != "countries"} if countries else None,
        "upstream_dams": up_dams,
        "upstream_area_km2": area.get("upstream_area_km2") if area else None,
        "upstream": area,
        "message": message, "notes": notes, "attribution": ATTRIBUTION, "licence": LICENCE,
    }
