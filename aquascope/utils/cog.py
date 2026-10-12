"""A small Cloud-Optimized GeoTIFF reader for point values, in pure Python.

Reads one pixel of a remote GeoTIFF with HTTP range requests: the header (the
IFDs sit at the front of a COG), one entry of the tile offset and byte count
arrays, then the one tile or strip that holds the pixel. No GDAL, no rasterio,
so it runs in the Explorer's Pyodide worker as well as in CPython (#520).

What it understands:

* classic TIFF and BigTIFF, little and big endian;
* tiled and stripped layouts, chunky or planar samples;
* no compression, LZW, Deflate (zlib) and PackBits, with the horizontal
  (2) and floating-point (3) predictors;
* the geotransform from ModelPixelScale + ModelTiepoint or from
  ModelTransformation (north-up only), PixelIsPoint rasters, GDAL's nodata
  and scale/offset tags, and the reduced-resolution IFDs (overviews).

What it does not: JPEG, LERC, ZSTD or WebP tiles (an error says which),
rotated grids, reprojection. Callers project their coordinates first.

HTTP goes through httpx in CPython and through ``urllib`` (patched by
pyodide-http) in the browser, where httpx cannot open sockets.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import sys
import threading
import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "COG",
    "COGError",
    "COGNotFound",
    "Image",
    "lzw_decode",
    "open_cog",
    "range_get",
    "value_at",
]

IS_EMSCRIPTEN = sys.platform == "emscripten"

#: How much of the file the first request reads. GDAL writes every IFD and the small tag arrays here.
HEADER_BYTES = 65_536
#: What one read past the header fetches, so the IFDs and tag values of a plain GTiff come in one go.
BLOCK_BYTES = 16_384
#: Decoded blocks kept per open file (least recently used dropped first).
MAX_CACHED_BYTES = 24 << 20
#: The widest byte span :meth:`COG.prefetch` reads in one request.
MAX_PREFETCH_SPAN = 4 << 20

Fetch = Callable[[str, int, int], bytes]


class COGError(RuntimeError):
    """The file is not a GeoTIFF this reader can decode."""


class COGNotFound(COGError, FileNotFoundError):  # noqa: N818 - "not found" reads better than NotFoundError
    """The server answered 404 (or 403 for a missing object on a public bucket)."""


# ── HTTP ──────────────────────────────────────────────────────────────────────

_client_lock = threading.Lock()
_client: Any = None


def _httpx_client() -> Any:
    global _client
    with _client_lock:
        if _client is None:
            import httpx

            _client = httpx.Client(timeout=60.0, follow_redirects=True,
                                   headers={"User-Agent": "aquascope (https://github.com/Rekin226/aquascope)"})
        return _client


def range_get(url: str, start: int, end: int) -> bytes:
    """Bytes ``start`` to ``end`` inclusive of ``url``. A server that ignores the range is sliced locally."""
    header = {"Range": f"bytes={start}-{end}"}
    if IS_EMSCRIPTEN:  # pragma: no cover - only under Pyodide
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, headers=header)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - https data hosts
                status = int(getattr(resp, "status", 206) or 0)
                data = resp.read()
        except urllib.error.HTTPError as exc:
            status, data = exc.code, b""
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise COGError(f"Browser fetch failed for {url} (the host may not allow cross-origin reads): {exc}") \
                from exc
        # pyodide-http answers every status as a response rather than raising
        if status in (403, 404):
            raise COGNotFound(f"{url}: HTTP {status}")
        if status == 0 or status >= 400:
            raise COGError(f"{url}: HTTP {status or 'blocked (CORS or network)'}")
    else:
        resp = _httpx_client().get(url, headers=header)
        if resp.status_code in (403, 404):
            raise COGNotFound(f"{url}: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise COGError(f"{url}: HTTP {resp.status_code}")
        status, data = resp.status_code, resp.content
    if status == 200 and len(data) > end - start + 1:
        data = data[start:end + 1]
    return data


# ── decoders ─────────────────────────────────────────────────────────────────


def lzw_decode(data: bytes) -> bytes:
    """TIFF LZW (MSB-first codes of 9 to 12 bits, with the early change)."""
    out = bytearray()
    table: list[bytes] = [bytes([i]) for i in range(256)] + [b"", b""]
    width = 9
    bitbuf = 0
    nbits = 0
    prev: bytes | None = None
    pos = 0
    n = len(data)
    while True:
        while nbits < width:
            if pos >= n:
                return bytes(out)
            bitbuf = (bitbuf << 8) | data[pos]
            pos += 1
            nbits += 8
        nbits -= width
        code = (bitbuf >> nbits) & ((1 << width) - 1)
        bitbuf &= (1 << nbits) - 1
        if code == 257:  # end of information
            break
        if code == 256:  # clear
            table = table[:258]
            width = 9
            prev = None
            continue
        if prev is None:
            entry = table[code]
            out += entry
            prev = entry
            continue
        if code < len(table):
            entry = table[code]
            table.append(prev + entry[:1])
        elif code == len(table):
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise COGError(f"corrupt LZW stream (code {code}, table {len(table)})")
        out += entry
        prev = entry
        # The early change: the code width grows one entry before the table fills it.
        if len(table) + 1 >= (1 << width) and width < 12:
            width += 1
    return bytes(out)


def _packbits_decode(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        h = data[i]
        i += 1
        if h < 128:
            out += data[i:i + h + 1]
            i += h + 1
        elif h > 128:
            out += bytes([data[i]]) * (257 - h)
            i += 1
    return bytes(out)


def _decompress(raw: bytes, compression: int) -> bytes:
    if compression == 1:
        return raw
    if compression == 5:
        return lzw_decode(raw)
    if compression in (8, 32946):
        return zlib.decompress(raw)
    if compression == 32773:
        return _packbits_decode(raw)
    names = {7: "JPEG", 34887: "LERC", 50000: "ZSTD", 50001: "WebP", 34925: "LZMA"}
    raise COGError(f"compression {names.get(compression, compression)} is not supported by the pure-Python reader")


# ── TIFF structure ───────────────────────────────────────────────────────────

_TYPES: dict[int, tuple[str, int]] = {
    1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1), 7: ("B", 1),
    8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8), 17: ("q", 8), 18: ("Q", 8),
}

_LAZY_TAGS = {273, 279, 324, 325}  # strip/tile offsets and byte counts: read one entry at a time


@dataclass
class _Tag:
    type: int
    count: int
    offset: int  # absolute file offset of the value bytes
    value: Any = None  # decoded, for everything but the lazy arrays


@dataclass
class Image:
    """One image (IFD) of the file: the full resolution, or an overview."""

    width: int
    height: int
    tile_width: int | None
    tile_height: int | None
    rows_per_strip: int
    samples: int
    planar: int
    bits: int
    sample_format: int
    compression: int
    predictor: int
    subfile_type: int
    offsets: _Tag
    counts: _Tag
    #: GDAL geotransform (x0, dx, 0, y0, 0, dy); overviews derive theirs from the full image.
    transform: tuple[float, float, float, float, float, float] | None = None
    tags: dict[int, _Tag] = field(default_factory=dict)

    @property
    def dtype(self) -> np.dtype:
        kind = {1: "u", 2: "i", 3: "f"}.get(self.sample_format, "u")
        if self.bits not in (8, 16, 32, 64):
            raise COGError(f"{self.bits}-bit samples are not supported")
        return np.dtype(f"{kind}{self.bits // 8}")

    @property
    def tiled(self) -> bool:
        return bool(self.tile_width and self.tile_height)


class COG:
    """A remote GeoTIFF read pixel by pixel. Open with :func:`open_cog` to share the parsed header."""

    def __init__(self, url: str, *, fetch: Fetch | None = None, header_bytes: int = HEADER_BYTES):
        self.url = url
        self._fetch = fetch or range_get
        self._head = self._fetch(url, 0, header_bytes - 1)
        if len(self._head) < 16:
            raise COGError(f"{url}: too short to be a TIFF")
        order = self._head[:2]
        if order == b"II":
            self._bo = "<"
        elif order == b"MM":
            self._bo = ">"
        else:
            raise COGError(f"{url}: not a TIFF (byte order {order!r})")
        magic = struct.unpack(self._bo + "H", self._head[2:4])[0]
        if magic == 42:
            self.bigtiff = False
            first = struct.unpack(self._bo + "I", self._head[4:8])[0]
        elif magic == 43:
            self.bigtiff = True
            first = struct.unpack(self._bo + "Q", self._head[8:16])[0]
        else:
            raise COGError(f"{url}: not a TIFF (magic {magic})")
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._images: list[Image] = []
        self._seen: set[int] = set()
        self._next_ifd = first
        if not self._parse_next():
            raise COGError(f"{url}: no image")
        base = self._images[0]
        tags = base.tags
        self.nodata = _parse_nodata(tags.get(42113))
        self.scale, self.offset = _parse_scale_offset(tags.get(42112))
        base.transform = self._geotransform(tags, base)
        self.epsg = _epsg(tags.get(34735))
        self._tiles: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    # ── raw bytes ──
    def _bytes(self, offset: int, n: int) -> bytes:
        """``n`` bytes at ``offset``: from the header, from a block read earlier, or one new block read.

        Files written as plain tiled GTiff (not by GDAL's COG driver) keep IFDs and tag values past the
        header, often at the end; reading a block around each miss turns dozens of tiny requests into one.
        """
        if offset + n <= len(self._head):
            return self._head[offset:offset + n]
        for start, blob in self._blocks.items():
            if start <= offset and offset + n <= start + len(blob):
                self._blocks.move_to_end(start)
                return blob[offset - start:offset - start + n]
        blob = self._fetch(self.url, offset, offset + max(n, BLOCK_BYTES) - 1)
        if len(blob) < n:
            raise COGError(f"{self.url}: short read at {offset} ({len(blob)} of {n} bytes)")
        self._blocks[offset] = blob
        while len(self._blocks) > 32:
            self._blocks.popitem(last=False)
        return blob[:n]

    def _parse_next(self) -> bool:
        """Parse one more image from the IFD chain; False at the end of it."""
        while self._next_ifd and self._next_ifd not in self._seen and len(self._seen) < 64:
            self._seen.add(self._next_ifd)
            img, self._next_ifd = self._read_ifd(self._next_ifd)
            if img is None:
                continue
            if self._images:
                base = self._images[0]
                if base.transform is not None:
                    x0, dx, _, y0, _, dy = base.transform
                    img.transform = (x0, dx * base.width / img.width, 0.0, y0, 0.0, dy * base.height / img.height)
            self._images.append(img)
            return True
        return False

    def image(self, level: int = 0) -> Image:
        """The full-resolution image (0) or overview ``level``, parsed on first use."""
        while len(self._images) <= level:
            if not self._parse_next():
                raise IndexError(f"{self.url} has {len(self._images)} image(s), no level {level}")
        return self._images[level]

    @property
    def images(self) -> list[Image]:
        """Every image in the file (parses the whole IFD chain)."""
        while self._parse_next():
            pass
        return self._images

    def _read_ifd(self, at: int) -> tuple[Image | None, int]:
        bo = self._bo
        if self.bigtiff:
            count = struct.unpack(bo + "Q", self._bytes(at, 8))[0]
            entry_size, start, inline = 20, at + 8, 8
        else:
            count = struct.unpack(bo + "H", self._bytes(at, 2))[0]
            entry_size, start, inline = 12, at + 2, 4
        block = self._bytes(start, count * entry_size + (8 if self.bigtiff else 4))
        tags: dict[int, _Tag] = {}
        for i in range(count):
            e = block[i * entry_size:(i + 1) * entry_size]
            if self.bigtiff:
                tag, typ, cnt = struct.unpack(bo + "HHQ", e[:12])
                value_bytes = e[12:20]
            else:
                tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
                value_bytes = e[8:12]
            if typ not in _TYPES:
                continue
            size = _TYPES[typ][1] * cnt
            if size <= inline:
                off = start + i * entry_size + (12 if self.bigtiff else 8)
            else:
                off = struct.unpack(bo + ("Q" if self.bigtiff else "I"), value_bytes)[0]
            t = _Tag(typ, cnt, off)
            if tag not in _LAZY_TAGS or size <= inline:
                t.value = self._decode_value(t)
            tags[tag] = t
        nxt_raw = block[count * entry_size:count * entry_size + (8 if self.bigtiff else 4)]
        nxt = struct.unpack(bo + ("Q" if self.bigtiff else "I"), nxt_raw)[0] if len(nxt_raw) else 0

        def one(tag: int, default: Any = None) -> Any:
            t = tags.get(tag)
            if t is None or t.value is None:
                return default
            v = t.value
            return v[0] if isinstance(v, (tuple, list)) else v

        subfile = int(one(254, 0) or 0)
        if subfile & 4:  # a transparency mask, not data
            return None, nxt
        tiled = 324 in tags
        offsets = tags.get(324 if tiled else 273)
        counts = tags.get(325 if tiled else 279)
        if offsets is None or counts is None:
            raise COGError(f"{self.url}: an image without tile or strip offsets")
        width, height = int(one(256)), int(one(257))
        img = Image(
            width=width, height=height,
            tile_width=int(one(322)) if tiled else None, tile_height=int(one(323)) if tiled else None,
            rows_per_strip=int(one(278, height) or height), samples=int(one(277, 1) or 1),
            planar=int(one(284, 1) or 1), bits=int(one(258, 8) or 8), sample_format=int(one(339, 1) or 1),
            compression=int(one(259, 1) or 1), predictor=int(one(317, 1) or 1), subfile_type=subfile,
            offsets=offsets, counts=counts, tags=tags,
        )
        return img, nxt

    def _decode_value(self, t: _Tag) -> Any:
        fmt, size = _TYPES[t.type]
        raw = self._bytes(t.offset, size * t.count)
        if t.type == 2:
            return raw.split(b"\x00", 1)[0].decode("latin-1")
        if t.type in (5, 10):
            vals = struct.unpack(self._bo + fmt[0] * (2 * t.count), raw)
            return tuple(vals[i] / vals[i + 1] if vals[i + 1] else math.nan for i in range(0, len(vals), 2))
        return struct.unpack(self._bo + fmt * t.count, raw)

    def _array_item(self, t: _Tag, i: int) -> int:
        if i >= t.count:
            raise COGError(f"{self.url}: block {i} is past the end of the offsets ({t.count})")
        if t.value is not None:
            return int(t.value[i])
        fmt, size = _TYPES[t.type]
        return int(struct.unpack(self._bo + fmt, self._bytes(t.offset + i * size, size))[0])

    def _geotransform(self, tags: dict[int, _Tag], img: Image) -> tuple[float, ...] | None:
        point = _raster_type(tags.get(34735)) == 2  # RasterPixelIsPoint
        if 34264 in tags and tags[34264].value:
            m = tags[34264].value
            if m[1] or m[4]:
                raise COGError(f"{self.url}: rotated rasters are not supported")
            x0, dx, y0, dy = m[3], m[0], m[7], m[5]
        elif 33550 in tags and 33922 in tags:
            sx, sy = tags[33550].value[0], tags[33550].value[1]
            i, j, _k, x, y = tags[33922].value[:5]
            x0, dx, y0, dy = x - i * sx, sx, y + j * sy, -sy
        else:
            return None
        if point:
            x0, y0 = x0 - dx / 2.0, y0 - dy / 2.0
        return (float(x0), float(dx), 0.0, float(y0), 0.0, float(dy))

    # ── pixels ──
    @property
    def transform(self) -> tuple[float, ...] | None:
        return self._images[0].transform

    @property
    def width(self) -> int:
        return self._images[0].width

    @property
    def height(self) -> int:
        return self._images[0].height

    @property
    def dtype(self) -> np.dtype:
        return self._images[0].dtype

    def bounds(self) -> tuple[float, float, float, float] | None:
        """(west, south, east, north) in the raster's own CRS."""
        t = self.transform
        if t is None:
            return None
        x0, dx, _, y0, _, dy = t
        xs = (x0, x0 + dx * self.width)
        ys = (y0, y0 + dy * self.height)
        return (min(xs), min(ys), max(xs), max(ys))

    def index(self, x: float, y: float, level: int = 0) -> tuple[int, int] | None:
        """(col, row) of the pixel holding (x, y) in the raster's CRS, or None outside it."""
        img = self.image(level)
        if img.transform is None:
            raise COGError(f"{self.url}: no georeferencing")
        x0, dx, _, y0, _, dy = img.transform
        col = math.floor((x - x0) / dx)
        row = math.floor((y - y0) / dy)
        if 0 <= col < img.width and 0 <= row < img.height:
            return col, row
        return None

    def _remember(self, key: tuple[int, int], arr: np.ndarray) -> None:
        """Keep a decoded block, dropping the least recently used ones past MAX_CACHED_BYTES."""
        self._tiles[key] = arr
        self._tiles.move_to_end(key)
        total = sum(a.nbytes for a in self._tiles.values())
        while total > MAX_CACHED_BYTES and len(self._tiles) > 1:
            _, old = self._tiles.popitem(last=False)
            total -= old.nbytes

    def _block(self, level: int, index: int) -> np.ndarray:
        key = (level, index)
        hit = self._tiles.get(key)
        if hit is not None:
            self._tiles.move_to_end(key)
            return hit
        img = self.image(level)
        off = self._array_item(img.offsets, index)
        n = self._array_item(img.counts, index)
        if n == 0:
            arr = np.array([], dtype=img.dtype)
        else:
            raw = self._fetch(self.url, off, off + n - 1)
            arr = self._unpack(img, _decompress(raw, img.compression), index)
        self._remember(key, arr)
        return arr

    def block_index(self, col: int, row: int, *, band: int = 0, level: int = 0) -> int:
        """Which tile or strip holds pixel (col, row)."""
        img = self.image(level)
        if img.tiled:
            tw, th = int(img.tile_width or 0), int(img.tile_height or 0)
            across = math.ceil(img.width / tw)
            index = (row // th) * across + (col // tw)
            if img.planar == 2:
                index += band * across * math.ceil(img.height / th)
            return index
        index = row // img.rows_per_strip
        if img.planar == 2:
            index += band * math.ceil(img.height / img.rows_per_strip)
        return index

    def prefetch(self, pixels: list[tuple[int, int]], *, band: int = 0, level: int = 0,
                 max_span: int = MAX_PREFETCH_SPAN) -> int:
        """Read the blocks holding ``pixels`` (col, row) in one range request when they lie close together.

        A neighbourhood in a striped file is one strip per row: fetched one by one that is a round trip
        each, fetched as one span it is a single request. Returns how many blocks were read (0 when the
        span is too wide, and the reads then happen one by one as usual).
        """
        img = self.image(level)
        wanted = sorted({self.block_index(c, r, band=band, level=level) for c, r in pixels
                         if 0 <= c < img.width and 0 <= r < img.height})
        wanted = [i for i in wanted if (level, i) not in self._tiles]
        if len(wanted) < 2:
            return 0
        spans = [(self._array_item(img.offsets, i), self._array_item(img.counts, i), i) for i in wanted]
        spans = [t for t in spans if t[1] > 0]
        if not spans:
            return 0
        lo = min(o for o, _, _ in spans)
        hi = max(o + n for o, n, _ in spans)
        if hi - lo > max_span:
            return 0
        blob = self._fetch(self.url, lo, hi - 1)
        for off, n, i in spans:
            self._remember((level, i), self._unpack(img, _decompress(blob[off - lo:off - lo + n], img.compression), i))
        return len(spans)

    def _unpack(self, img: Image, data: bytes, index: int) -> np.ndarray:
        spp = img.samples if img.planar == 1 else 1
        if img.tiled:
            w, h = int(img.tile_width or 0), int(img.tile_height or 0)
        else:
            w = img.width
            strips_per_band = math.ceil(img.height / img.rows_per_strip)
            first_row = (index % strips_per_band) * img.rows_per_strip
            h = min(img.rows_per_strip, img.height - first_row)
        itemsize = img.dtype.itemsize
        need = w * h * spp * itemsize
        if len(data) < need:  # a short final strip, or a writer that trimmed padding
            h = len(data) // (w * spp * itemsize)
            need = w * h * spp * itemsize
        buf = np.frombuffer(data[:need], dtype=np.uint8)
        if img.predictor == 3:
            # libtiff undoes the byte differencing with a stride of one pixel (``spp`` bytes)
            rows = buf.reshape(h, w * itemsize, spp)
            rows = np.cumsum(rows, axis=1, dtype=np.uint8).reshape(h, w * spp * itemsize)
            # bytes are stored most significant first, one plane per byte position
            planes = rows.reshape(h, itemsize, w * spp).transpose(0, 2, 1)
            arr = np.ascontiguousarray(planes).view(img.dtype.newbyteorder(">")).reshape(h, w, spp)
            return arr.astype(img.dtype.newbyteorder("="), copy=False)
        arr = buf.view(img.dtype.newbyteorder(self._bo)).reshape(h, w, spp).astype(img.dtype.newbyteorder("="))
        if img.predictor == 2:
            # libtiff adds the differences as unsigned words of the sample size (wrapping), whatever the type
            words = arr.view(np.dtype(f"u{itemsize}"))
            with np.errstate(over="ignore"):
                words = np.cumsum(words, axis=1, dtype=words.dtype)
            arr = words.view(arr.dtype)
        return arr

    def read_pixel(self, col: int, row: int, *, band: int = 0, level: int = 0) -> Any:
        """The raw stored value at (col, row), before nodata and scale."""
        img = self.image(level)
        if not (0 <= col < img.width and 0 <= row < img.height):
            raise IndexError(f"pixel ({col}, {row}) is outside {img.width} x {img.height}")
        if band >= img.samples:
            raise IndexError(f"band {band} of {img.samples}")
        block = self._block(level, self.block_index(col, row, band=band, level=level))
        if img.tiled:
            r, c = row % int(img.tile_height or 1), col % int(img.tile_width or 1)
        else:
            r, c = row % img.rows_per_strip, col
        if block.size == 0 or r >= block.shape[0]:
            return None
        v = block[r, c, 0 if img.planar == 2 else band]
        return v.item()

    def value_at(self, x: float, y: float, *, band: int = 0, level: int = 0) -> float | None:
        """The scaled value at (x, y) in the raster's CRS; None outside the raster or on nodata."""
        ij = self.index(x, y, level)
        if ij is None:
            return None
        raw = self.read_pixel(ij[0], ij[1], band=band, level=level)
        if raw is None:
            return None
        if self.nodata is not None and (raw == self.nodata or (math.isnan(self.nodata) and raw != raw)):
            return None
        if isinstance(raw, float) and math.isnan(raw):
            return None
        return raw * self.scale + self.offset if (self.scale != 1.0 or self.offset != 0.0) else raw


def _parse_nodata(tag: _Tag | None) -> float | None:
    if tag is None or tag.value is None:
        return None
    text = str(tag.value).strip()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_scale_offset(tag: _Tag | None) -> tuple[float, float]:
    if tag is None or not isinstance(tag.value, str):
        return 1.0, 0.0
    scale, offset = 1.0, 0.0
    for m in re.finditer(r"<Item\s+([^>]*)>([^<]*)</Item>", tag.value):
        attrs, text = m.group(1), m.group(2).strip()
        role = re.search(r'role="(\w+)"', attrs)
        sample = re.search(r'sample="(\d+)"', attrs)
        if sample and sample.group(1) != "0":
            continue
        try:
            if role and role.group(1) == "scale":
                scale = float(text)
            elif role and role.group(1) == "offset":
                offset = float(text)
        except ValueError:
            continue
    return scale, offset


def _geokeys(tag: _Tag | None) -> dict[int, int]:
    if tag is None or not tag.value:
        return {}
    v = tag.value
    out: dict[int, int] = {}
    n = int(v[3]) if len(v) > 3 else 0
    for k in range(n):
        key, loc, _cnt, val = v[4 + 4 * k:8 + 4 * k]
        if loc == 0:
            out[int(key)] = int(val)
    return out


def _raster_type(tag: _Tag | None) -> int:
    return _geokeys(tag).get(1025, 1)


def _epsg(tag: _Tag | None) -> int | None:
    keys = _geokeys(tag)
    code = keys.get(3072) or keys.get(2048)
    return code if code and code != 32767 else None


# ── a shared cache of opened files ───────────────────────────────────────────

_OPEN: OrderedDict[str, COG] = OrderedDict()
_OPEN_LOCK = threading.Lock()
_MAX_OPEN = 48


def open_cog(url: str, *, fetch: Fetch | None = None) -> COG:
    """A :class:`COG` for ``url``, reusing the parsed header when the same file was opened before."""
    if fetch is not None:
        return COG(url, fetch=fetch)
    with _OPEN_LOCK:
        hit = _OPEN.get(url)
        if hit is not None:
            _OPEN.move_to_end(url)
            return hit
    cog = COG(url)
    with _OPEN_LOCK:
        _OPEN[url] = cog
        while len(_OPEN) > _MAX_OPEN:
            _OPEN.popitem(last=False)
    return cog


def value_at(url: str, x: float, y: float, *, band: int = 0, fetch: Fetch | None = None) -> float | None:
    """One value of a remote GeoTIFF at (x, y) in its own CRS (None outside it or on nodata)."""
    return open_cog(url, fetch=fetch).value_at(x, y, band=band)
