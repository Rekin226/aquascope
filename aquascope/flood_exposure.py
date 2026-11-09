"""What gets flooded: buildings, roads and people under a flood model's water, step by step (#622). Experimental.

The question after "how deep?" is "what is under it?". Given a :class:`aquascope.io.mesh.Mesh` and the
buildings, roads and population around it, :func:`exposure` counts, at every model step and at the peak:

* **buildings**: a footprint counts as flooded when the model's depth passes the threshold in a cell under
  its centre or one of its corners;
* **roads**: each stretch of road is cut where it crosses from one cell to the next, and a stretch counts as
  flooded when its cell is; the flooded length is the sum of those stretches;
* **people**: each population grid square is split into 3 x 3 parts that share its people evenly, and a part
  counts when the cell under it is flooded.

The threshold defaults to the readers' own dry depth (:data:`aquascope.io.mesh.DRY_DEPTH`): any water. A
rain-on-grid model wets every cell with a film, so :data:`THRESHOLDS_M` offers the two depths the Environment
Agency's surface water map explains ("What is the updated Flood Map for Surface Water?", November 2013,
https://www.cumbria.gov.uk/eLibrary/view.asp?ID=61733): at 0.15 m flooding typically exceeds kerb height and
can flood property in some areas; at 0.30 m it is likely to flood property, from average threshold levels.

The open sources, all read from the browser over HTTPS range requests (no key, CORS open, checked
2026-10-25 from the Explorer's origin):

* buildings and roads: Overture Maps (https://overturemaps.org), the ``buildings/building`` and
  ``transportation/segment`` GeoParquet files of the latest release on ``s3://overturemaps-us-west-2``, found
  through Overture's STAC catalogue (https://stac.overturemaps.org) and read only for the row groups whose
  bounding boxes meet the model's. Both themes are ODbL 1.0; attribution "© OpenStreetMap contributors,
  Overture Maps Foundation" (https://docs.overturemaps.org/attribution/). Roads are the segments of subtype
  ``road`` that carry vehicles: the classes in :data:`PATH_CLASSES` (footways, sidewalks, paths, steps,
  pedestrian streets, cycleways, bridleways) are left out, or a city's mapped sidewalks would double its roads.
* population: GHS-POP R2023A (European Commission JRC; Schiavina et al. 2023,
  doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE), the 2025 epoch at 3 arc-seconds in WGS84, CC BY 4.0, as
  Cloud-Optimized GeoTIFF tiles published by WFP on Source Cooperative
  (https://source.coop/wfp/jrc-ghsl-pop).

The counts are estimates and say so: footprints are mapped or machine-detected outlines (some missing, some
misplaced), the model's cell depth is not the depth at a door, a bridge above the water still counts as a
flooded road, and the population grid is a model of where residents live, spread evenly within each square.

:func:`exposure` is numpy only; the readers need pyarrow (Overture's GeoParquet), loaded by the Explorer's
mesh worker only when "What gets flooded" is asked.
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
from typing import Any

import numpy as np

from aquascope.flood_section import MeshIndex, _bed, _typical_cell_m, hours_since_start
from aquascope.io.mesh import DRY_DEPTH, Mesh, MeshError

__all__ = [
    "LIMITS", "PATH_CLASSES", "SOURCES", "THRESHOLDS_M", "exposure", "exposure_for_mesh", "exposure_payload",
    "ghsl_tile_url", "overture_files", "read_overture", "read_population", "read_sources", "wkb_parts",
]

IS_EMSCRIPTEN = sys.platform == "emscripten"

STAC_ROOT = "https://stac.overturemaps.org/catalog.json"
GHSL_BASE = "https://data.source.coop/wfp/jrc-ghsl-pop/r2023/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0"
# The GHSL WGS84 tiles are 10 degrees square, row 1 starting at 89.0996 N and column 1 at 180.0079 W, half a
# kilometre off the round degrees (the tiles' own geotransforms: R5_C20 spans 9.99208 to 19.99208 E and
# 39.09958 to 49.09958 N, R13_C13 60.00792 to 50.00792 W and 30.90042 to 40.90042 S).
GHSL_NORTH, GHSL_WEST, GHSL_TILE_DEG = 89.099583, -180.007917, 10.0

#: Where each count comes from, with its licence and what to credit.
SOURCES: dict[str, dict[str, str]] = {
    "buildings": {
        "name": "Overture Maps buildings", "licence": "ODbL 1.0",
        "credit": "© OpenStreetMap contributors, Overture Maps Foundation",
        "link": "https://docs.overturemaps.org/attribution/#buildings",
    },
    "roads": {
        "name": "Overture Maps transportation (road segments, no footways, paths or cycleways)", "licence": "ODbL 1.0",
        "credit": "© OpenStreetMap contributors, Overture Maps Foundation",
        "link": "https://docs.overturemaps.org/attribution/#transportation",
    },
    "population": {
        "name": "GHS-POP R2023A, 2025, 3 arc-seconds (JRC)", "licence": "CC BY 4.0",
        "credit": "European Commission, Joint Research Centre (Schiavina et al. 2023), hosted by WFP on Source "
                  "Cooperative",
        "link": "https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE",
    },
}

#: Depth thresholds a reader can choose, in metres, with the words for each and where they come from.
THRESHOLDS_M: dict[float, str] = {
    DRY_DEPTH: "any water",
    0.15: "over the kerb: can flood property in some areas (Environment Agency, 2013)",
    0.30: "likely to flood property, from average threshold levels (Environment Agency, 2013)",
}
THRESHOLD_SOURCE = "https://www.cumbria.gov.uk/eLibrary/view.asp?ID=61733"

#: What each estimate cannot tell.
LIMITS: dict[str, str] = {
    "buildings": "A building counts when the model's water in a cell under its centre or a corner passes the "
                 "threshold. Footprints are mapped or machine-detected outlines, some missing or misplaced, and "
                 "a cell's depth is not the depth at the door: HEC-RAS measures it from the lowest ground in the "
                 "cell, so a building on a rise beside a ditch can count as wet.",
    "roads": "Roads that carry vehicles, service roads and tracks included; footways, sidewalks, paths and "
             "cycleways are left out. Each stretch counts by the cell it crosses: a bridge or an embankment above "
             "the water still counts, because the model's cells do not know it is there.",
    "population": "A model of where residents live in 2025, at about 90 m, spread evenly within each grid square: "
                  "people in the flooded area, not people at risk, and nothing of who was there at the time.",
}

#: The most features a read keeps, so a dense city cannot stall a phone.
MAX_BUILDINGS = 60_000
MAX_ROADS = 40_000
#: The largest model box the readers will fetch for, in square kilometres.
MAX_AREA_KM2 = 900.0
POP_SPLIT = 3
#: Overture road classes that are not for vehicles, left out of the road count (Overture's transportation guide,
#: https://docs.overturemaps.org/guides/transportation/, lists the classes; sidewalks and crossings are footways).
PATH_CLASSES = frozenset({"footway", "sidewalk", "crosswalk", "path", "steps", "pedestrian", "cycleway",
                          "bridleway"})


# ── geometry: WKB ───────────────────────────────────────────────────────────


def wkb_parts(blob: bytes) -> tuple[str, list[list[np.ndarray]]]:
    """A WKB geometry as its kind ("point", "line" or "polygon") and its parts, each a list of rings or
    lines as (n, 2) arrays of longitude and latitude. Z and M are dropped."""
    out: list[list[np.ndarray]] = []
    kinds: set[str] = set()

    def read(at: int) -> int:
        bo = "<" if blob[at] == 1 else ">"
        (code,) = struct.unpack(bo + "I", blob[at + 1:at + 5])
        at += 5
        dims = 2
        if code & 0x20000000:   # EWKB SRID
            at += 4
        if code & 0x80000000 or code & 0x40000000:   # EWKB Z / M
            dims += int(bool(code & 0x80000000)) + int(bool(code & 0x40000000))
        code &= 0x0FFFFFFF
        if code >= 1000:   # ISO Z (1000), M (2000), ZM (3000)
            dims = 2 + {1: 1, 2: 1, 3: 2}[code // 1000]
            code %= 1000

        def coords(at: int) -> tuple[np.ndarray, int]:
            (n,) = struct.unpack(bo + "I", blob[at:at + 4])
            at += 4
            arr = np.frombuffer(blob, dtype=np.dtype(bo + "f8"), count=n * dims, offset=at).reshape(n, dims)
            return np.array(arr[:, :2], dtype="f8"), at + 8 * n * dims

        if code == 1:
            arr = np.frombuffer(blob, dtype=np.dtype(bo + "f8"), count=dims, offset=at)
            out.append([np.array([arr[:2]], dtype="f8")])
            kinds.add("point")
            return at + 8 * dims
        if code == 2:
            line, at = coords(at)
            out.append([line])
            kinds.add("line")
            return at
        if code == 3:
            (nr,) = struct.unpack(bo + "I", blob[at:at + 4])
            at += 4
            rings = []
            for _ in range(nr):
                ring, at = coords(at)
                rings.append(ring)
            out.append(rings)
            kinds.add("polygon")
            return at
        if code in (4, 5, 6, 7):
            (n,) = struct.unpack(bo + "I", blob[at:at + 4])
            at += 4
            for _ in range(n):
                at = read(at)
            return at
        raise ValueError(f"WKB geometry type {code} is not supported")

    read(0)
    kind = "polygon" if "polygon" in kinds else "line" if "line" in kinds else "point"
    return kind, out


# ── reading the open sources ────────────────────────────────────────────────


def _get(url: str, *, timeout: float = 60.0) -> bytes:
    """The body of ``url`` (CPython: httpx; the browser: urllib through pyodide-http)."""
    if IS_EMSCRIPTEN:  # pragma: no cover - only under Pyodide
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - https data hosts
            return resp.read()
    from aquascope.utils.cog import _httpx_client

    resp = _httpx_client().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _get_json(url: str) -> Any:
    return json.loads(_get(url).decode("utf-8"))


def _meets(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def overture_files(bbox: list[float], theme: str, kind: str, *, release: str | None = None,
                   get_json: Any = None) -> dict[str, Any]:
    """The GeoParquet files of an Overture ``theme``/``kind`` (``buildings``/``building``,
    ``transportation``/``segment``) whose bounding boxes meet ``bbox``, from the STAC catalogue: the latest
    release unless ``release`` is given. Returns ``{"release", "files": [url, ...]}``."""
    gj = get_json or _get_json
    if release is None:
        root = gj(STAC_ROOT)
        release = root.get("latest") or [lk for lk in root["links"] if lk.get("latest")][0]["title"].split()[0]
    base = f"https://stac.overturemaps.org/{release}/{theme}/{kind}"
    col = gj(f"{base}/collection.json")
    boxes = col["extent"]["spatial"]["bbox"][1:]
    items = [lk["href"] for lk in col["links"] if lk.get("rel") == "item"]
    if len(items) != len(boxes):
        raise MeshError(f"Overture's catalogue for {theme}/{kind} lists {len(items)} files but {len(boxes)} boxes.")
    files = []
    for href, box in zip(items, boxes):
        if _meets(box, bbox):
            item = gj(href)
            files.append(item["assets"]["aws"]["href"])
    return {"release": release, "files": files}


class _RangeFile(io.RawIOBase):
    """A remote file read with HTTP range requests, holding the byte ranges read ahead (:meth:`prefetch`), so
    pyarrow finds the column chunks it asks for already in memory."""

    def __init__(self, url: str, size: int, fetch: Any):
        self.url, self.size, self._fetch, self.pos = url, int(size), fetch, 0
        self.held: list[tuple[int, bytes]] = []
        self.requests = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        for at, blob in self.held:
            if at <= start and end <= at + len(blob):
                return blob[start - at:end - at]
        data = self._fetch(self.url, start, end - 1)
        self.requests += 1
        self.bytes_read += len(data)
        self.held.append((start, data))
        return data

    def prefetch(self, spans: list[tuple[int, int]], gap: int = 1 << 16) -> None:
        """Read the (start, end) spans, joining those closer than ``gap`` bytes, in as few requests as that."""
        merged: list[list[int]] = []
        for start, end in sorted(spans):
            if merged and start - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            self._get(start, min(end, self.size))

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._get(self.pos, self.pos + n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _size_of(url: str, fetch: Any) -> int:
    """A remote file's size, from the Content-Range of a one-byte read (or the fetch's own ``size_of``)."""
    known = getattr(fetch, "size_of", None)
    if callable(known):
        return int(known(url))
    if IS_EMSCRIPTEN:  # pragma: no cover - only under Pyodide
        import urllib.request

        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - https data hosts
            cr = resp.headers.get("Content-Range") or ""
    else:
        from aquascope.utils.cog import _httpx_client

        cr = _httpx_client().get(url, headers={"Range": "bytes=0-0"}).headers.get("content-range") or ""
    if "/" not in cr:
        raise MeshError(f"{url} did not say its size (no Content-Range).")
    return int(cr.rsplit("/", 1)[1])


def _row_groups(md: Any, bbox: list[float]) -> list[int]:
    names = [md.schema.column(i).path for i in range(md.num_columns)]
    ix = {n: i for i, n in enumerate(names)}
    need = ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")
    if not all(n in ix for n in need):
        return list(range(md.num_row_groups))
    w, s, e, n = bbox
    out = []
    for r in range(md.num_row_groups):
        rg = md.row_group(r)
        st = [rg.column(ix[k]).statistics for k in need]
        if any(x is None or not x.has_min_max for x in st):
            out.append(r)
            continue
        if st[0].min <= e and st[1].max >= w and st[2].min <= n and st[3].max >= s:
            out.append(r)
    return out


def _chunk_spans(md: Any, groups: list[int], columns: list[str]) -> list[tuple[int, int]]:
    spans = []
    for r in groups:
        rg = md.row_group(r)
        for i in range(rg.num_columns):
            c = rg.column(i)
            if c.path_in_schema.split(".")[0] not in columns:
                continue
            dic = c.dictionary_page_offset if c.has_dictionary_page else None
            start = dic if dic else c.data_page_offset
            spans.append((int(start), int(start) + int(c.total_compressed_size)))
    return spans


def read_overture(bbox: list[float], what: str, *, release: str | None = None, fetch: Any = None,
                  files: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    """Buildings (``what="buildings"``) or roads (``"roads"``) inside ``bbox`` from Overture Maps.

    Reads only the row groups whose bounding boxes meet ``bbox`` and only the columns it needs. Returns
    ``{"features": [{"id", "class", "kind", "parts"}], "release", "files", "requests", "bytes", "truncated"}``;
    ``parts`` are lists of (n, 2) coordinate arrays (outer rings first for a building).
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow is in the archive extra and the browser loads it
        raise MeshError("Reading Overture's buildings and roads needs pyarrow (pip install pyarrow).") from exc
    from aquascope.utils.cog import range_get

    fetch = fetch or range_get
    theme, kind = ("buildings", "building") if what == "buildings" else ("transportation", "segment")
    cap = limit or (MAX_BUILDINGS if what == "buildings" else MAX_ROADS)
    found = {"release": release, "files": files} if files is not None else overture_files(bbox, theme, kind,
                                                                                         release=release)
    cols = ["id", "geometry", "bbox", "class"] + (["subtype"] if what == "roads" else ["subtype", "height"])
    feats: list[dict[str, Any]] = []
    requests = nbytes = 0
    truncated = False
    w, s, e, n = bbox
    for url in found["files"]:
        f = _RangeFile(url, _size_of(url, fetch), fetch)
        # The footer: its length sits in the last 8 bytes; read it with them in one request.
        tail = f._get(f.size - 8, f.size)
        flen = struct.unpack("<I", tail[:4])[0]
        f.prefetch([(f.size - 8 - flen, f.size)])
        pf = pq.ParquetFile(f)
        md = pf.metadata
        have = {md.schema.column(i).path.split(".")[0] for i in range(md.num_columns)}
        use = [c for c in cols if c in have]
        groups = _row_groups(md, bbox)
        if not groups:
            continue
        f.prefetch(_chunk_spans(md, groups, use))
        table = pf.read_row_groups(groups, columns=use)
        bb = table.column("bbox")
        xmin, xmax = pc.struct_field(bb, "xmin"), pc.struct_field(bb, "xmax")
        ymin, ymax = pc.struct_field(bb, "ymin"), pc.struct_field(bb, "ymax")
        mask = pc.and_(pc.and_(pc.less_equal(xmin, e), pc.greater_equal(xmax, w)),
                       pc.and_(pc.less_equal(ymin, n), pc.greater_equal(ymax, s)))
        if what == "roads" and "subtype" in use:
            mask = pc.and_(mask, pc.equal(table.column("subtype"), "road"))
            if "class" in use:   # vehicles only: a null class (unknown) stays
                paths = pc.is_in(table.column("class"), value_set=pa.array(sorted(PATH_CLASSES)))
                mask = pc.and_(mask, pc.invert(pc.fill_null(paths, False)))
        table = table.filter(mask)
        ids = table.column("id").to_pylist()
        geoms = table.column("geometry").to_pylist()
        classes = table.column("class").to_pylist() if "class" in use else [None] * len(ids)
        for fid, g, c in zip(ids, geoms, classes):
            if g is None:
                continue
            if len(feats) >= cap:
                truncated = True
                break
            try:
                k, parts = wkb_parts(g)
            except (ValueError, struct.error):
                continue
            feats.append({"id": fid, "class": c, "kind": k, "parts": parts})
        requests += f.requests
        nbytes += f.bytes_read
        if truncated:
            break
    return {"features": feats, "release": found["release"], "files": found["files"], "requests": requests,
            "bytes": nbytes, "truncated": truncated}


def ghsl_tile_url(lon: float, lat: float) -> str:
    """The GHS-POP 2025 tile (10 x 10 degrees) holding a point."""
    row = int(math.floor((GHSL_NORTH - lat) / GHSL_TILE_DEG)) + 1
    col = int(math.floor((lon - GHSL_WEST) / GHSL_TILE_DEG)) + 1
    return f"{GHSL_BASE}_R{row}_C{col}.tif"


def read_population(bbox: list[float], *, fetch: Any = None) -> dict[str, Any]:
    """The GHS-POP 2025 grid inside ``bbox``: ``lon``/``lat`` of each square's centre, ``people`` in it, and the
    square's size in degrees, from the tiles that meet the box."""
    from aquascope.utils.cog import COGNotFound, open_cog

    w, s, e, n = bbox
    urls = sorted({ghsl_tile_url(x, y) for x in (w, e) for y in (s, n)})
    lon_all, lat_all, pop_all = [], [], []
    step = None
    read = []
    for url in urls:
        try:
            cog = open_cog(url, fetch=fetch)
        except COGNotFound:
            continue   # the open ocean has no tile
        x0, dx, _, y0, _, dy = cog.transform
        step = (dx, dy)
        c0, c1 = int(math.floor((w - x0) / dx)), int(math.floor((e - x0) / dx))
        r0, r1 = int(math.floor((n - y0) / dy)), int(math.floor((s - y0) / dy))
        win = cog.read_window(c0, r0, c1, r1)
        if not win.size:
            continue
        c0, r0 = max(0, c0), max(0, r0)
        cols = x0 + (np.arange(c0, c0 + win.shape[1]) + 0.5) * dx
        rows = y0 + (np.arange(r0, r0 + win.shape[0]) + 0.5) * dy
        gx, gy = np.meshgrid(cols, rows)
        ok = np.isfinite(win) & (win > 0)
        lon_all.append(gx[ok])
        lat_all.append(gy[ok])
        pop_all.append(win[ok])
        read.append(url)
    if not read:
        return {"lon": np.zeros(0), "lat": np.zeros(0), "people": np.zeros(0), "step": step, "tiles": []}
    return {"lon": np.concatenate(lon_all), "lat": np.concatenate(lat_all), "people": np.concatenate(pop_all),
            "step": step, "tiles": read}


# ── the counts ──────────────────────────────────────────────────────────────


def _ft(m: Mesh) -> bool:
    return str(m.units.get("depth") or "").lower().startswith("ft")


def _to_m(m: Mesh, x: float) -> float:
    return x * 0.3048 if _ft(m) else x


def _threshold(m: Mesh, threshold: float | None, threshold_m: float | None) -> float:
    """The depth that counts as flooded, in the file's own length unit (``threshold``), or given in metres."""
    if threshold is not None:
        return float(threshold)
    if threshold_m is not None:
        return float(threshold_m) / 0.3048 if _ft(m) else float(threshold_m)
    return DRY_DEPTH


def _wet_matrix(m: Mesh, cells: np.ndarray, threshold: float) -> np.ndarray:
    """(steps, len(cells)) True where the depth in that cell passes the threshold."""
    n = len(m.times)
    out = np.zeros((n, cells.size), dtype=bool)
    if not cells.size or not n:
        return out
    for i in range(0, n, 48):
        block = m.values_block(i, min(i + 48, n))[:, cells]
        out[i:i + block.shape[0]] = np.isfinite(block) & (block > threshold)
    return out


def _depth_matrix(m: Mesh, cells: np.ndarray) -> np.ndarray:
    n = len(m.times)
    out = np.full((n, cells.size), np.nan)
    for i in range(0, n, 48):
        out[i:i + 48] = m.values_block(i, min(i + 48, n))[:, cells]
    return out


def _bits(row: np.ndarray) -> str:
    return "".join("1" if x else "0" for x in row)


def _peak(series: np.ndarray, times: list[str]) -> dict[str, Any]:
    if not series.size:
        return {"value": 0, "step": None, "time": None}
    k = int(np.argmax(series))
    return {"value": float(series[k]), "step": k, "time": times[k]}


def _road_pieces(roads: list[dict[str, Any]], idx: MeshIndex, spacing_m: float) -> list[dict[str, Any]]:
    """Each road cut into stretches, one per cell it crosses: points every ``spacing_m`` along the line, and
    where two neighbouring points fall in different cells, the crossing found by bisection (to a thousandth of
    the spacing), so a stretch's length is its length inside its cell."""
    lines = [(r, np.asarray(ln, dtype="f8")) for r, road in enumerate(roads) for part in road["parts"] for ln in part]
    dens = []   # (road, lon, lat, distance)
    for r, line in lines:
        if line.ndim != 2 or line.shape[0] < 2:
            continue
        lat0 = math.radians(float(line[:, 1].mean()))
        x = np.radians(line[:, 0]) * math.cos(lat0) * 6371008.8
        y = np.radians(line[:, 1]) * 6371008.8
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        total = float(cum[-1])
        if total <= 0:
            continue
        # the line's own vertices and points every spacing_m between them
        d = np.union1d(cum, np.linspace(0.0, total, max(2, int(math.ceil(total / spacing_m)) + 1)))
        dens.append((r, line, cum, d))
    if not dens:
        return []
    lon = np.concatenate([np.interp(d, cum, ln[:, 0]) for _, ln, cum, d in dens])
    lat = np.concatenate([np.interp(d, cum, ln[:, 1]) for _, ln, cum, d in dens])
    cells = idx.locate(lon, lat)
    sizes = np.array([d.size for *_, d in dens])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # crossings: neighbouring points of the same line in different cells
    same_line = np.ones(lon.size, bool)
    same_line[starts[1:] - 1] = False
    same_line[-1] = False
    k = np.nonzero(same_line & (cells != np.roll(cells, -1)))[0]
    owner = np.searchsorted(starts, k, side="right") - 1
    d_all = np.concatenate([d for *_, d in dens])
    lo, hi = d_all[k].copy(), d_all[k + 1].copy()
    cell_lo = cells[k]
    for _ in range(10):
        mid = (lo + hi) / 2
        mlon = np.array([np.interp(mid[j], dens[o][2], dens[o][1][:, 0]) for j, o in enumerate(owner)])
        mlat = np.array([np.interp(mid[j], dens[o][2], dens[o][1][:, 1]) for j, o in enumerate(owner)])
        same = idx.locate(mlon, mlat) == cell_lo if mid.size else np.zeros(0, bool)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    cut = (lo + hi) / 2
    cuts_of: dict[int, list[float]] = {}
    for o, c in zip(owner.tolist(), cut.tolist()):
        cuts_of.setdefault(o, []).append(c)
    pieces: list[dict[str, Any]] = []
    for j, (r, line, cum, d) in enumerate(dens):
        c = cells[starts[j]:starts[j] + d.size]
        bounds = [0.0] + cuts_of.get(j, []) + [float(d[-1])]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a <= 0:
                continue
            at = int(np.searchsorted(d, (a + b) / 2, side="right") - 1)
            cell = int(c[min(at, c.size - 1)])
            if cell < 0:
                continue
            inner = d[(d > a) & (d < b)]
            dd = np.concatenate([[a], inner, [b]])
            coords = np.stack([np.interp(dd, cum, line[:, 0]), np.interp(dd, cum, line[:, 1])], 1)
            pieces.append({"road": r, "cell": cell, "length_m": float(b - a), "coords": coords})
    return pieces


def exposure(m: Mesh, *, buildings: list[dict[str, Any]] | None = None, roads: list[dict[str, Any]] | None = None,
             population: dict[str, Any] | None = None, threshold: float | None = None,
             threshold_m: float | None = None) -> dict[str, Any]:
    """Buildings, road length and people under the model's water at every step and at the peak.

    ``buildings`` and ``roads`` are features as :func:`read_overture` returns them (``parts`` of coordinate
    arrays), ``population`` a grid as :func:`read_population` returns it. Each count is a list with one value
    per model step, its peak (value, step, time) and the number ever flooded; each feature on the mesh comes
    back with ``wet``, a string of one character per step ("1" flooded), for the map to show step by step.
    """
    if m.variable != "depth" or _bed(m) is None:
        raise MeshError("What gets flooded needs depths, and this file has no bed levels: only the water surface.")
    if not m.times:
        raise MeshError("This file has no time steps, only its maxima.")
    thr = _threshold(m, threshold, threshold_m)
    idx = MeshIndex(m)
    times = list(m.times)
    n_steps = len(times)
    out: dict[str, Any] = {"times": times, "hours": [round(float(h), 4) for h in hours_since_start(times)],
                           "threshold": thr, "threshold_m": round(_to_m(m, thr), 4),
                           "units": {"depth": m.units.get("depth"), "length": "m"},
                           "name": m.name, "limits": dict(LIMITS),
                           "sources": {}}

    if buildings is not None:
        pts_lon, pts_lat, owner = [], [], []
        for b, f in enumerate(buildings):
            if not f.get("parts"):
                continue
            ring = f["parts"][0][0]   # the first polygon's outer ring
            if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            pts = np.vstack([ring.mean(axis=0, keepdims=True), ring])
            pts_lon.append(pts[:, 0])
            pts_lat.append(pts[:, 1])
            owner.append(np.full(pts.shape[0], b, dtype=np.intp))
        if pts_lon:
            cells = idx.locate(np.concatenate(pts_lon), np.concatenate(pts_lat))
            own = np.concatenate(owner)
            on = cells >= 0
            cells, own = cells[on], own[on]
        else:
            cells = own = np.zeros(0, dtype=np.intp)
        uniq, inv = np.unique(cells, return_inverse=True)
        depth = _depth_matrix(m, uniq) if uniq.size else np.zeros((n_steps, 0))
        on_mesh = np.unique(own)
        pos = np.searchsorted(on_mesh, own)
        # the deepest water under each building at each step
        deep = np.full((n_steps, on_mesh.size), -np.inf)
        if own.size:
            vals = np.where(np.isfinite(depth[:, inv]), depth[:, inv], -np.inf)
            for t in range(n_steps):
                np.maximum.at(deep[t], pos, vals[t])
        wet = deep > thr
        count = wet.sum(axis=1).astype(int)
        ever = wet.any(axis=0)
        feats = []
        for j, b in enumerate(on_mesh):
            f = buildings[int(b)]
            mx = float(np.max(deep[:, j]))
            feats.append({"id": f.get("id"), "class": f.get("class"), "ring": f["parts"][0][0],
                          "wet": _bits(wet[:, j]), "max_depth": round(mx, 3) if math.isfinite(mx) else None})
        out["buildings"] = {"per_step": count.tolist(), "peak": _peak(count, times), "ever": int(ever.sum()),
                            "on_mesh": int(on_mesh.size), "read": len(buildings), "features": feats}

    if roads is not None:
        pieces = _road_pieces(roads, idx, max(_typical_cell_m(m) / 3.0, 2.0))
        cells = np.array([p["cell"] for p in pieces], dtype=np.intp)
        uniq, inv = np.unique(cells, return_inverse=True)
        wet = _wet_matrix(m, uniq, thr)[:, inv] if pieces else np.zeros((n_steps, 0), bool)
        length = np.array([p["length_m"] for p in pieces])
        km = (wet * length).sum(axis=1) / 1000.0 if pieces else np.zeros(n_steps)
        flooded_roads = np.zeros((n_steps, len(roads)), bool)
        for j, p in enumerate(pieces):
            flooded_roads[:, p["road"]] |= wet[:, j]
        segs = []
        for j, p in enumerate(pieces):
            if not wet[:, j].any():
                continue
            f = roads[p["road"]]
            segs.append({"id": f.get("id"), "class": f.get("class"), "coords": p["coords"],
                         "wet": _bits(wet[:, j]), "length_m": round(p["length_m"], 1)})
        ever_km = float(length[wet.any(axis=0)].sum() / 1000.0) if pieces else 0.0
        peak = _peak(km, times)
        peak["value"] = round(peak["value"], 3)
        out["roads"] = {"per_step_km": [round(float(x), 3) for x in km], "peak": peak, "ever_km": round(ever_km, 3),
                        "on_mesh_km": round(float(length.sum() / 1000.0), 3), "read": len(roads),
                        "segments_flooded": int(flooded_roads.any(axis=0).sum()), "pieces": segs}

    if population is not None:
        lon, lat, ppl = (np.asarray(population[k], dtype="f8") for k in ("lon", "lat", "people"))
        st = population.get("step") or (0.0, 0.0)
        dx, dy = abs(float(st[0])), abs(float(st[1]))
        if lon.size:
            k = POP_SPLIT
            off = (np.arange(k) + 0.5) / k - 0.5
            ox, oy = np.meshgrid(off * dx, off * dy)
            plon = (lon[:, None] + ox.reshape(1, -1)).reshape(-1)
            plat = (lat[:, None] + oy.reshape(1, -1)).reshape(-1)
            share = np.repeat(ppl / (k * k), k * k)
            cells = idx.locate(plon, plat)
            on = cells >= 0
            uniq, inv = np.unique(cells[on], return_inverse=True)
            wet = _wet_matrix(m, uniq, thr)[:, inv]
            people = (wet * share[on]).sum(axis=1)
            ever = float(share[on][wet.any(axis=0)].sum())
            on_mesh = float(share[on].sum())
        else:
            people, ever, on_mesh = np.zeros(n_steps), 0.0, 0.0
        peak = _peak(people, times)
        peak["value"] = round(peak["value"], 1)
        out["population"] = {"per_step": [round(float(x), 1) for x in people], "peak": peak,
                             "ever": round(ever, 1), "on_mesh": round(on_mesh, 1),
                             "square_deg": [dx, dy]}
    return out


def _check_area(bbox: list[float]) -> None:
    w, s, e, n = bbox
    km2 = (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * (n - s) * 110.57
    if km2 > MAX_AREA_KM2:
        raise MeshError(f"The model covers about {km2:,.0f} km², more than the {MAX_AREA_KM2:,.0f} km² the browser "
                        "reads buildings and roads for.")


def _round_coords(a: np.ndarray, nd: int = 6) -> list[list[float]]:
    return [[round(float(x), nd), round(float(y), nd)] for x, y in a]


def _plain_parts(parts: list[list[np.ndarray]]) -> list[list[list[list[float]]]]:
    return [[_round_coords(np.asarray(r)) for r in part] for part in parts]


def read_sources(bbox: list[float], *, what: tuple[str, ...] = ("buildings", "roads", "population"),
                 release: str | None = None) -> dict[str, Any]:
    """Read the open sources inside ``bbox`` as plain JSON types, for :func:`exposure_payload`: the Explorer reads
    them in a light worker and counts in the mesh worker, beside the file. A source that cannot be read is left
    out, with the reason in ``errors``."""
    bbox = [float(x) for x in bbox]
    _check_area(bbox)
    out: dict[str, Any] = {"bbox": bbox, "info": {}, "errors": {}}
    for w in what:
        try:
            if w == "population":
                pop = read_population(bbox)
                out["population"] = {"lon": [round(float(x), 7) for x in pop["lon"]],
                                     "lat": [round(float(x), 7) for x in pop["lat"]],
                                     "people": [round(float(x), 4) for x in pop["people"]], "step": pop["step"]}
                out["info"]["population"] = {"tiles": pop["tiles"]}
            elif w in ("buildings", "roads"):
                r = read_overture(bbox, w, release=release)
                out[w] = [{"id": f["id"], "class": f["class"], "parts": _plain_parts(f["parts"])}
                          for f in r["features"]]
                out["info"][w] = {"release": r["release"], "files": len(r["files"]), "requests": r["requests"],
                                  "bytes": r["bytes"], "truncated": r["truncated"]}
        except Exception as exc:  # noqa: BLE001 - any failure of one source is said and the rest go on
            out["errors"][w] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def _arrays(feats: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if feats is None:
        return None
    return [{**f, "parts": [[np.asarray(r, dtype="f8").reshape(-1, 2) for r in part] for part in f["parts"]]}
            for f in feats]


def exposure_payload(m: Mesh, read: dict[str, Any], *, threshold: float | None = None,
                     threshold_m: float | None = None) -> dict[str, Any]:
    """:func:`exposure` over what :func:`read_sources` read, with each source's name, licence and credit, the
    limits of each count and the reading errors, as plain JSON types ready to draw."""
    pop = read.get("population")
    if pop is not None:
        pop = {**pop, "lon": np.asarray(pop["lon"], "f8"), "lat": np.asarray(pop["lat"], "f8"),
               "people": np.asarray(pop["people"], "f8")}
    res = exposure(m, buildings=_arrays(read.get("buildings")), roads=_arrays(read.get("roads")), population=pop,
                   threshold=threshold, threshold_m=threshold_m)
    res["threshold_source"] = THRESHOLD_SOURCE
    res["thresholds_m"] = [{"m": k, "words": v} for k, v in THRESHOLDS_M.items()]
    for w in ("buildings", "roads", "population"):
        if w in res:
            res["sources"][w] = {**SOURCES[w], **read.get("info", {}).get(w, {})}
    if "buildings" in res:
        for f in res["buildings"]["features"]:
            f["ring"] = _round_coords(f["ring"])
    if "roads" in res:
        for p in res["roads"]["pieces"]:
            p["coords"] = _round_coords(p["coords"])
    res["errors"] = dict(read.get("errors", {}))
    return res


def exposure_for_mesh(m: Mesh, *, threshold: float | None = None, what: tuple[str, ...] = ("buildings", "roads",
                      "population"), release: str | None = None) -> dict[str, Any]:
    """Read the open sources around the mesh and count what its water reaches, in one call (the CLI and MCP
    faces; the Explorer splits it in two across its workers)."""
    return exposure_payload(m, read_sources(m.bbox(), what=what, release=release), threshold=threshold)
