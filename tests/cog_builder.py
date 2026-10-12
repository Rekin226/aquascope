"""Build small GeoTIFFs in memory for the COG reader and the context-layer tests.

Hand-written on purpose: the tests must not depend on GDAL, and writing the
bytes ourselves is what proves the reader parses the format rather than one
writer's habits. Covers classic and BigTIFF, both byte orders, tiles and
strips, no compression, LZW and Deflate, predictors 2 and 3, the GeoTIFF
georeferencing tags, GDAL nodata and scale/offset, and overviews.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np


def lzw_encode(data: bytes) -> bytes:
    """TIFF LZW, MSB-first, with the early change (the inverse of aquascope.utils.cog.lzw_decode)."""
    out_bits: list[tuple[int, int]] = []
    table = {bytes([i]): i for i in range(256)}
    next_code = 258
    width = 9
    out_bits.append((256, width))
    w = b""
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
            continue
        out_bits.append((table[w], width))
        table[wc] = next_code
        next_code += 1
        # libtiff's order: a full table emits a clear code, otherwise the width grows past the last code
        if next_code >= 4094:
            out_bits.append((256, width))
            table = {bytes([i]): i for i in range(256)}
            next_code = 258
            width = 9
        elif next_code >= (1 << width) and width < 12:
            width += 1
        w = bytes([byte])
    if w:
        out_bits.append((table[w], width))
        # the decoder appends an entry on this code too, so the width may grow before the end code
        next_code += 1
        if next_code >= (1 << width) and width < 12:
            width += 1
    out_bits.append((257, width))
    acc = 0
    nbits = 0
    out = bytearray()
    for code, nb in out_bits:
        acc = (acc << nb) | code
        nbits += nb
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _predict(block: np.ndarray, predictor: int, bo: str) -> bytes:
    """block is (h, w) in its own dtype; returns the bytes as a TIFF writer would store them."""
    if predictor == 2:
        diff = block.copy()
        diff[:, 1:] = block[:, 1:] - block[:, :-1]
        return diff.astype(block.dtype.newbyteorder(bo)).tobytes()
    if predictor == 3:
        h, w = block.shape
        size = block.dtype.itemsize
        big = block.astype(block.dtype.newbyteorder(">")).view(np.uint8).reshape(h, w, size)
        planes = big.transpose(0, 2, 1).reshape(h, w * size)
        diff = planes.copy()
        diff[:, 1:] = planes[:, 1:] - planes[:, :-1]
        return diff.astype(np.uint8).tobytes()
    return block.astype(block.dtype.newbyteorder(bo)).tobytes()


def _compress(raw: bytes, compression: int) -> bytes:
    if compression == 1:
        return raw
    if compression == 5:
        return lzw_encode(raw)
    if compression == 8:
        return zlib.compress(raw)
    raise ValueError(compression)


@dataclass
class _Entry:
    tag: int
    typ: int
    values: list


_FMT = {1: "B", 2: "s", 3: "H", 4: "I", 11: "f", 12: "d", 16: "Q"}
_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 11: 4, 12: 8, 16: 8}


def build_tiff(
    arr: np.ndarray,
    *,
    tile: tuple[int, int] | None = (16, 16),
    rows_per_strip: int = 4,
    compression: int = 1,
    predictor: int = 1,
    bigtiff: bool = False,
    byteorder: str = "<",
    geotransform: tuple[float, float, float, float] | None = (0.0, 1.0, 0.0, -1.0),
    model_transformation: bool = False,
    pixel_is_point: bool = False,
    nodata: float | None = None,
    scale: float | None = None,
    offset: float | None = None,
    epsg: int | None = 4326,
    overview: np.ndarray | None = None,
) -> bytes:
    """A GeoTIFF of ``arr`` (2-D). ``geotransform`` is (x0, dx, y0, dy) of the top-left corner."""
    images = [arr] + ([overview] if overview is not None else [])
    bo = byteorder
    blobs: list[list[bytes]] = []
    for im in images:
        h, w = im.shape
        chunks: list[bytes] = []
        if tile:
            tw, th = tile
            for ty in range(0, h, th):
                for tx in range(0, w, tw):
                    block = np.zeros((th, tw), dtype=im.dtype)
                    part = im[ty:ty + th, tx:tx + tw]
                    block[:part.shape[0], :part.shape[1]] = part
                    chunks.append(_compress(_predict(block, predictor, bo), compression))
        else:
            for ry in range(0, h, rows_per_strip):
                block = im[ry:ry + rows_per_strip]
                chunks.append(_compress(_predict(block, predictor, bo), compression))
        blobs.append(chunks)

    kind = arr.dtype.kind
    sample_format = {"u": 1, "i": 2, "f": 3}[kind]
    bits = arr.dtype.itemsize * 8

    def entries_for(level: int, offsets: list[int], counts: list[int]) -> list[_Entry]:
        im = images[level]
        h, w = im.shape
        off_type = 16 if bigtiff else 4
        e = [
            _Entry(254, 4, [1 if level else 0]),
            _Entry(256, 4, [w]), _Entry(257, 4, [h]), _Entry(258, 3, [bits]), _Entry(259, 3, [compression]),
            _Entry(262, 3, [1]), _Entry(277, 3, [1]), _Entry(284, 3, [1]), _Entry(339, 3, [sample_format]),
        ]
        if predictor != 1:
            e.append(_Entry(317, 3, [predictor]))
        if tile:
            e += [_Entry(322, 3, [tile[0]]), _Entry(323, 3, [tile[1]]),
                  _Entry(324, off_type, offsets), _Entry(325, off_type, counts)]
        else:
            e += [_Entry(273, off_type, offsets), _Entry(278, 3, [rows_per_strip]), _Entry(279, off_type, counts)]
        if level == 0:
            if geotransform is not None:
                x0, dx, y0, dy = geotransform
                if model_transformation:
                    e.append(_Entry(34264, 12, [dx, 0, 0, x0, 0, dy, 0, y0, 0, 0, 0, 0, 0, 0, 0, 1]))
                else:
                    e.append(_Entry(33550, 12, [dx, -dy, 0.0]))
                    tie_x, tie_y = x0, y0
                    if pixel_is_point:
                        tie_x, tie_y = x0 + dx / 2, y0 + dy / 2
                    e.append(_Entry(33922, 12, [0.0, 0.0, 0.0, tie_x, tie_y, 0.0]))
            keys = [1, 1, 0, 0]
            body: list[int] = []
            body += [1025, 0, 1, 2 if pixel_is_point else 1]
            if epsg:
                body += [2048 if epsg == 4326 else 3072, 0, 1, epsg]
            keys[3] = len(body) // 4
            e.append(_Entry(34735, 3, keys + body))
            if scale is not None or offset is not None:
                xml = "<GDALMetadata>"
                if scale is not None:
                    xml += f'<Item name="SCALE" sample="0" role="scale">{scale}</Item>'
                if offset is not None:
                    xml += f'<Item name="OFFSET" sample="0" role="offset">{offset}</Item>'
                xml += "</GDALMetadata>"
                e.append(_Entry(42112, 2, [xml]))
            if nodata is not None:
                e.append(_Entry(42113, 2, [f"{nodata:g}"]))
        e.sort(key=lambda x: x.tag)
        return e

    head_size = 16 if bigtiff else 8
    entry_size = 20 if bigtiff else 12
    inline = 8 if bigtiff else 4

    def value_bytes(ent: _Entry) -> bytes:
        if ent.typ == 2:
            return ent.values[0].encode("latin-1") + b"\x00"
        return struct.pack(bo + _FMT[ent.typ] * len(ent.values), *ent.values)

    def count_of(ent: _Entry) -> int:
        return len(value_bytes(ent)) if ent.typ == 2 else len(ent.values)

    # layout: header, IFDs with their out-of-line values, then the image data
    def ifd_size(ents: list[_Entry]) -> int:
        size = (8 if bigtiff else 2) + len(ents) * entry_size + (8 if bigtiff else 4)
        for ent in ents:
            n = len(value_bytes(ent))
            if n > inline:
                size += n + (n % 2)
        return size

    placeholder = [entries_for(i, [0] * len(blobs[i]), [0] * len(blobs[i])) for i in range(len(images))]
    data_start = head_size + sum(ifd_size(p) for p in placeholder)
    offsets_all: list[list[int]] = []
    pos = data_start
    for chunks in blobs:
        offs = []
        for c in chunks:
            offs.append(pos)
            pos += len(c)
        offsets_all.append(offs)

    out = bytearray()
    out += (b"II" if bo == "<" else b"MM")
    if bigtiff:
        out += struct.pack(bo + "HHHQ", 43, 8, 0, head_size)
    else:
        out += struct.pack(bo + "HI", 42, head_size)
    for level in range(len(images)):
        ents = entries_for(level, offsets_all[level], [len(c) for c in blobs[level]])
        start = len(out)
        extra_at = start + (8 if bigtiff else 2) + len(ents) * entry_size + (8 if bigtiff else 4)
        table = bytearray()
        extra = bytearray()
        table += struct.pack(bo + ("Q" if bigtiff else "H"), len(ents))
        for ent in ents:
            vb = value_bytes(ent)
            if bigtiff:
                table += struct.pack(bo + "HHQ", ent.tag, ent.typ, count_of(ent))
            else:
                table += struct.pack(bo + "HHI", ent.tag, ent.typ, count_of(ent))
            if len(vb) <= inline:
                table += vb + b"\x00" * (inline - len(vb))
            else:
                table += struct.pack(bo + ("Q" if bigtiff else "I"), extra_at + len(extra))
                extra += vb + (b"\x00" if len(vb) % 2 else b"")
        is_last = level == len(images) - 1
        nxt = 0 if is_last else extra_at + len(extra)
        table += struct.pack(bo + ("Q" if bigtiff else "I"), nxt)
        out += table + extra
    assert len(out) == data_start, (len(out), data_start)
    for chunks in blobs:
        for c in chunks:
            out += c
    return bytes(out)


def fetcher(files: dict[str, bytes], calls: list | None = None):
    """A ``fetch(url, start, end)`` over in-memory files, recording each call; unknown URLs are a 404."""
    from aquascope.utils.cog import COGNotFound

    def fetch(url: str, start: int, end: int) -> bytes:
        if calls is not None:
            calls.append((url, start, end))
        if url not in files:
            raise COGNotFound(f"{url}: HTTP 404")
        return files[url][start:end + 1]

    return fetch
