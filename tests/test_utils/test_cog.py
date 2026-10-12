"""The pure-Python COG point reader (#520) against GeoTIFFs built in the test, never the network."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from aquascope.utils import cog
from tests.cog_builder import build_tiff, fetcher, lzw_encode

URL = "https://example.test/raster.tif"


def _open(data: bytes, calls: list | None = None, **kw) -> cog.COG:
    return cog.COG(URL, fetch=fetcher({URL: data}, calls), **kw)


def _grid(h=40, w=37, dtype="int16"):
    return (np.arange(h * w).reshape(h, w) % 997 - 300).astype(dtype)


def test_lzw_round_trip_including_the_width_changes_and_a_full_table():
    rng = np.random.default_rng(0)
    for data in (b"", b"a", b"TOBEORNOTTOBEORTOBEORNOT" * 50, rng.integers(0, 256, 20_000, dtype=np.uint8).tobytes(),
                 bytes(range(256)) * 40):
        assert cog.lzw_decode(lzw_encode(data)) == data


@pytest.mark.parametrize("compression", [1, 5, 8])
@pytest.mark.parametrize("predictor", [1, 2])
@pytest.mark.parametrize("tiled", [True, False])
def test_every_pixel_reads_back(compression, predictor, tiled):
    arr = _grid()
    data = build_tiff(arr, tile=(16, 16) if tiled else None, rows_per_strip=7, compression=compression,
                      predictor=predictor)
    c = _open(data)
    assert (c.width, c.height) == (37, 40)
    for row in range(0, 40, 3):
        for col in range(0, 37, 5):
            assert c.read_pixel(col, row) == int(arr[row, col])


@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_bigtiff_and_both_byte_orders(bigtiff, byteorder):
    arr = _grid(dtype="uint16")
    data = build_tiff(arr, bigtiff=bigtiff, byteorder=byteorder, compression=8, predictor=2)
    c = _open(data)
    assert c.bigtiff is bigtiff
    assert c.read_pixel(36, 39) == int(arr[39, 36])
    assert c.read_pixel(0, 0) == int(arr[0, 0])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_floating_point_predictor(dtype):
    arr = (np.linspace(-5, 5, 40 * 37).reshape(40, 37) ** 3).astype(dtype)
    c = _open(build_tiff(arr, compression=8, predictor=3))
    assert c.read_pixel(20, 11) == pytest.approx(float(arr[11, 20]))
    assert c.read_pixel(36, 39) == pytest.approx(float(arr[39, 36]))


def test_geotransform_nodata_and_scale():
    arr = np.full((20, 30), 7, dtype="int32")
    arr[5, 10] = -9999
    data = build_tiff(arr, geotransform=(10.0, 0.5, 60.0, -0.5), nodata=-9999, scale=0.1, offset=2.0)
    c = _open(data)
    assert c.transform == (10.0, 0.5, 0.0, 60.0, 0.0, -0.5)
    assert c.nodata == -9999 and c.scale == 0.1 and c.offset == 2.0 and c.epsg == 4326
    assert c.bounds() == (10.0, 50.0, 25.0, 60.0)
    # pixel (col 10, row 5) covers x 15.0-15.5, y 57.5-57.0
    assert c.value_at(15.25, 57.25) is None
    assert c.value_at(15.75, 57.25) == pytest.approx(7 * 0.1 + 2.0)
    assert c.value_at(9.99, 55.0) is None and c.value_at(26.0, 55.0) is None


def test_model_transformation_and_pixel_is_point():
    arr = np.arange(100, dtype="uint8").reshape(10, 10)
    a = _open(build_tiff(arr, geotransform=(0.0, 1.0, 10.0, -1.0), model_transformation=True))
    assert a.value_at(3.5, 6.5) == arr[3, 3]
    b = _open(build_tiff(arr, geotransform=(0.0, 1.0, 10.0, -1.0), pixel_is_point=True))
    assert b.transform == pytest.approx((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
    assert b.value_at(3.5, 6.5) == arr[3, 3]


def test_only_the_header_one_offset_and_one_tile_are_fetched():
    arr = _grid(200, 200)
    data = build_tiff(arr, tile=(32, 32), compression=8)
    calls: list = []
    c = _open(data, calls, header_bytes=64)  # a tiny header read forces the lazy offset reads
    first = len(calls)
    assert c.read_pixel(150, 170) == int(arr[170, 150])
    tile_reads = calls[first:]
    # one offset entry, one byte count entry, one tile; never the whole offsets array
    assert len(tile_reads) <= 3
    assert all(end - start < cog.BLOCK_BYTES for _, start, end in tile_reads)
    before = len(calls)
    assert c.read_pixel(151, 171) == int(arr[171, 151])  # same tile: served from the cache
    assert len(calls) == before


def test_overviews_carry_a_derived_transform():
    arr = np.arange(64 * 64, dtype="uint16").reshape(64, 64)
    ov = arr[::2, ::2].copy()
    c = _open(build_tiff(arr, geotransform=(0.0, 1.0, 64.0, -1.0), overview=ov))
    assert len(c.images) == 2
    assert c.images[1].transform == (0.0, 2.0, 0.0, 64.0, 0.0, -2.0)
    assert c.value_at(10.5, 50.5, level=1) == int(ov[int((64 - 50.5) // 2), int(10.5 // 2)])


def test_unsupported_compression_and_not_a_tiff():
    arr = _grid(8, 8)
    data = bytearray(build_tiff(arr, tile=None, rows_per_strip=8, compression=1))
    # patch the Compression tag value (259) to JPEG (7)
    i = data.find((259).to_bytes(2, "little") + (3).to_bytes(2, "little"))
    data[i + 8:i + 10] = (7).to_bytes(2, "little")
    c = _open(bytes(data))
    with pytest.raises(cog.COGError, match="JPEG"):
        c.read_pixel(1, 1)
    with pytest.raises(cog.COGError, match="not a TIFF"):
        _open(b"PK\x03\x04" + b"\x00" * 64)


def test_open_cog_shares_the_parsed_header(monkeypatch):
    data = build_tiff(_grid(), compression=5)
    calls: list = []
    monkeypatch.setattr(cog, "range_get", fetcher({URL: data}, calls))
    cog._OPEN.clear()
    try:
        a = cog.open_cog(URL)
        b = cog.open_cog(URL)
        assert a is b
        assert cog.value_at(URL, 3.5, -2.5) == int(_grid()[2, 3])
    finally:
        cog._OPEN.clear()


def test_a_missing_file_is_cog_not_found():
    with pytest.raises(cog.COGNotFound):
        cog.COG("https://example.test/missing.tif", fetch=fetcher({}))


def test_range_get_slices_a_server_that_ignores_the_range(monkeypatch):
    class Resp:
        status_code = 200
        content = b"0123456789"

    class Client:
        def get(self, url, headers=None):
            assert headers == {"Range": "bytes=2-4"}
            return Resp()

    monkeypatch.setattr(cog, "_httpx_client", lambda: Client())
    assert cog.range_get("https://example.test/x", 2, 4) == b"234"


def test_range_get_maps_404_to_not_found(monkeypatch):
    class Resp:
        status_code = 404
        content = b""

    monkeypatch.setattr(cog, "_httpx_client", lambda: type("C", (), {"get": lambda self, u, headers=None: Resp()})())
    with pytest.raises(cog.COGNotFound):
        cog.range_get("https://example.test/x", 0, 1)


def test_agrees_with_tifffile_when_it_is_installed():
    tifffile = pytest.importorskip("tifffile")
    arr = (np.arange(50 * 70).reshape(50, 70) % 251).astype("uint8")
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, tile=(16, 16), compression="zlib", predictor=True,
                     extratags=[(33550, 12, 3, (0.25, 0.25, 0.0), False),
                                (33922, 12, 6, (0.0, 0.0, 0.0, 100.0, 40.0, 0.0), False)])
    c = _open(buf.getvalue())
    assert c.transform == (100.0, 0.25, 0.0, 40.0, 0.0, -0.25)
    for row, col in ((0, 0), (49, 69), (17, 33)):
        assert c.read_pixel(col, row) == int(arr[row, col])
    assert c.value_at(100.0 + 33.5 * 0.25, 40.0 - 17.5 * 0.25) == int(arr[17, 33])
    assert not math.isnan(float(c.read_pixel(5, 5)))


def test_prefetch_reads_neighbouring_strips_in_one_request():
    arr = _grid(60, 30, dtype="uint8")
    data = build_tiff(arr, tile=None, rows_per_strip=1, compression=5, predictor=2)
    calls: list = []
    c = _open(data, calls)
    before = len(calls)
    pixels = [(5, r) for r in range(10, 31, 2)]
    assert c.prefetch(pixels) == 11
    assert len(calls) - before == 1
    after = len(calls)
    assert [c.read_pixel(col, row) for col, row in pixels] == [int(arr[row, col]) for col, row in pixels]
    assert len(calls) == after  # every strip came from the prefetch
    assert c.prefetch(pixels) == 0  # nothing left to fetch
    assert c.prefetch([(5, 40), (5, 50)], max_span=1) == 0  # too wide a span: left to the one-by-one reads


def test_the_block_cache_is_bounded_by_bytes(monkeypatch):
    monkeypatch.setattr(cog, "MAX_CACHED_BYTES", 600)
    arr = _grid(64, 64, dtype="uint16")
    c = _open(build_tiff(arr, tile=(16, 16), compression=8))
    for row in range(0, 64, 16):
        for col in range(0, 64, 16):
            assert c.read_pixel(col, row) == int(arr[row, col])
    # a 16 x 16 uint16 tile is 512 bytes: only the latest one fits
    assert len(c._tiles) == 1


def _fp_encode(block: np.ndarray) -> bytes:
    """libtiff's floating-point predictor for an (h, w, spp) block: big-endian byte planes, then each byte
    minus the one a pixel (``spp`` bytes) earlier."""
    h, w, spp = block.shape
    size = block.dtype.itemsize
    big = block.astype(block.dtype.newbyteorder(">")).view(np.uint8).reshape(h, w * spp, size)
    rows = big.transpose(0, 2, 1).reshape(h, w * size, spp)
    diff = rows.copy()
    diff[:, 1:] = rows[:, 1:] - rows[:, :-1]
    return diff.astype(np.uint8).tobytes()


@pytest.mark.parametrize("spp", [1, 2, 3])
def test_floating_point_predictor_with_several_samples_per_pixel(spp):
    arr = (np.linspace(-5, 5, 6 * 9 * spp).reshape(6, 9, spp) ** 3).astype("float32")
    c = object.__new__(cog.COG)
    c._bo = "<"
    img = cog.Image(width=9, height=6, tile_width=None, tile_height=None, rows_per_strip=6, samples=spp, planar=1,
                    bits=32, sample_format=3, compression=1, predictor=3, subfile_type=0,
                    offsets=cog._Tag(4, 1, 0, (0,)), counts=cog._Tag(4, 1, 0, (0,)))
    out = c._unpack(img, _fp_encode(arr), 0)
    assert out.shape == (6, 9, spp)
    np.testing.assert_array_equal(out, arr)


def test_floating_point_predictor_agrees_with_tifffile_and_imagecodecs():
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    arr = (np.linspace(-5, 5, 24 * 20 * 2).reshape(24, 20, 2) ** 3).astype("float32")
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, tile=(16, 16), compression="zlib", predictor=3, photometric="minisblack",
                     planarconfig="contig")
    c = _open(buf.getvalue())
    for row, col, band in ((0, 0, 0), (23, 19, 1), (11, 7, 1), (5, 17, 0)):
        assert c.read_pixel(col, row, band=band) == float(arr[row, col, band])
