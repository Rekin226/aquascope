"""aquascope.rivers: snap, reach record, upstream area and the trace to the sea, all with mocked HTTP (#516)."""

from __future__ import annotations

import gzip
import json
import math
import struct

import numpy as np
import pandas as pd
import pytest

from aquascope import rivers

#: The real API seam, captured at import, before tests/conftest.py takes GEOGLOWS off the network per test.
_REAL_FETCH_JSON = rivers._fetch_json

# ── tiny encoders: a PMTiles v3 archive holding one Mapbox Vector Tile ───────


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field(num: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((num << 3) | wire)
    if wire == 0:
        return key + _varint(int(payload))
    return key + _varint(len(payload)) + payload


def _zz(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _line_geometry(points: list[tuple[int, int]]) -> list[int]:
    out = [(1 & 7) | (1 << 3), _zz(points[0][0]), _zz(points[0][1])]
    out.append((2 & 7) | ((len(points) - 1) << 3))
    x, y = points[0]
    for px, py in points[1:]:
        out += [_zz(px - x), _zz(py - y)]
        x, y = px, py
    return out


def _mvt(features: list[tuple[int, int, list[tuple[int, int]]]]) -> bytes:
    """A tile with one 'streams' layer: (riverId, strahlerOrder, tile-pixel line) per feature."""
    values: list[int] = []
    feats = b""
    for rid, order, pts in features:
        values += [rid, order]
        tags = b"".join(_varint(v) for v in (0, len(values) - 2, 1, len(values) - 1))
        geom = b"".join(_varint(v) for v in _line_geometry(pts))
        feats += _field(2, 2, _field(2, 2, tags) + _field(3, 0, 2) + _field(4, 2, geom))
    layer = (_field(15, 0, 2) + _field(1, 2, b"streams") + feats + _field(3, 2, b"riverId")
             + _field(3, 2, b"strahlerOrder") + b"".join(_field(4, 2, _field(4, 0, v)) for v in values)
             + _field(5, 0, 4096))
    return _field(3, 2, layer)


def _pmtiles(tiles: dict[tuple[int, int, int], bytes]) -> bytes:
    """A PMTiles v3 archive: uncompressed directories, gzip tiles, everything in the root directory."""
    entries = sorted(((rivers._zxy_to_tile_id(*k), gzip.compress(v)) for k, v in tiles.items()))
    data = b""
    rows = []
    for tid, blob in entries:
        rows.append((tid, len(data), len(blob)))
        data += blob
    d = _varint(len(rows))
    last = 0
    for tid, _o, _ln in rows:
        d += _varint(tid - last)
        last = tid
    d += b"".join(_varint(1) for _ in rows)
    d += b"".join(_varint(ln) for _t, _o, ln in rows)
    d += b"".join(_varint(o + 1) for _t, o, _ln in rows)
    root_off = 127
    data_off = root_off + len(d)
    head = bytearray(127)
    head[:7] = b"PMTiles"
    head[7] = 3
    struct.pack_into("<8Q", head, 8, root_off, len(d), 0, 0, data_off, 0, data_off, len(data))
    head[97], head[98], head[99], head[100], head[101] = 1, 2, 1, 0, 12
    return bytes(head) + d + data


@pytest.fixture(autouse=True)
def _fresh_caches():
    for cache in (rivers._ARCHIVE, rivers._TILES, rivers._NETWORKS, rivers._UPSTREAM):
        cache.clear()
    yield
    for cache in (rivers._ARCHIVE, rivers._TILES, rivers._NETWORKS, rivers._UPSTREAM):
        cache.clear()


LAT, LON = 46.948, 7.452


@pytest.fixture
def one_river(monkeypatch):
    """A tile at zoom 12 with reach 230260670 passing 10 tile units (about 16 m) south of (LAT, LON)."""
    z = rivers.SNAP_ZOOM
    gx, gy = rivers._world(LON, LAT, z)
    tx, ty = int(gx // 4096), int(gy // 4096)
    px, py = int(gx - tx * 4096), int(gy - ty * 4096)
    tile = _mvt([(230260670, 5, [(px + 100, py + 10), (px - 100, py + 10)])])
    archive = _pmtiles({(z, tx, ty): tile})
    calls: list[tuple[int, int]] = []

    def fake_range(url, start, length):
        assert url == rivers.STREAMS_PMTILES
        calls.append((start, length))
        return archive[start:start + length]

    monkeypatch.setattr(rivers, "_fetch_range", fake_range)
    return calls


# ── PMTiles and MVT ──────────────────────────────────────────────────────────


def test_tile_ids_follow_the_pmtiles_hilbert_curve():
    assert rivers._zxy_to_tile_id(0, 0, 0) == 0
    assert [rivers._zxy_to_tile_id(1, x, y) for x, y in ((0, 0), (0, 1), (1, 1), (1, 0))] == [1, 2, 3, 4]
    assert rivers._zxy_to_tile_id(2, 0, 0) == 5


def test_decode_streams_tile_reads_the_two_attributes_and_the_line():
    data = _mvt([(230000001, 3, [(10, 20), (30, 20), (30, 50)])])
    out = rivers.decode_streams_tile(data, 1, 1, 0)
    assert list(out) == [230000001]
    assert out[230000001]["order"] == 3
    assert out[230000001]["lines"] == [[(4096 + 10, 20), (4096 + 30, 20), (4096 + 30, 50)]]


# ── snap ─────────────────────────────────────────────────────────────────────


def test_snap_to_river_finds_the_reach_and_the_distance(one_river):
    res = rivers.snap_to_river(LAT, LON)
    assert res["snapped"] is True
    assert res["river_id"] == 230260670 and res["strahler_order"] == 5
    assert 5 < res["distance_m"] < 40
    assert res["message"].startswith("Snapped") and "230260670" in res["message"]
    assert abs(res["snap_lat"] - LAT) < 0.001 and abs(res["snap_lon"] - LON) < 0.001
    assert "CC BY" in res["attribution"]


def test_snap_to_river_says_when_no_stream_is_within_the_tolerance(one_river):
    res = rivers.snap_to_river(LAT, LON, max_distance_m=5)
    assert res["snapped"] is False and res["river_id"] is None
    assert res["message"].startswith("No stream within 5 m")
    assert res["nearest"]["river_id"] == 230260670


def test_snap_to_river_with_no_tile_at_all(monkeypatch, one_river):
    res = rivers.snap_to_river(-30.0, -20.0)  # mid-Atlantic: not in the archive
    assert res["snapped"] is False and res["nearest"] is None
    assert "No stream within 3 km" in res["message"]


@pytest.fixture
def two_rivers(monkeypatch):
    """Reach 230260670 about 16 m south of (LAT, LON) and reach 230260671 about 160 m north."""
    z = rivers.SNAP_ZOOM
    gx, gy = rivers._world(LON, LAT, z)
    tx, ty = int(gx // 4096), int(gy // 4096)
    px, py = int(gx - tx * 4096), int(gy - ty * 4096)
    tile = _mvt([(230260670, 3, [(px + 100, py + 10), (px - 100, py + 10)]),
                 (230260671, 7, [(px + 100, py - 100), (px - 100, py - 100)])])
    archive = _pmtiles({(z, tx, ty): tile})
    monkeypatch.setattr(rivers, "_fetch_range", lambda url, start, length: archive[start:start + length])


def test_reaches_near_lists_every_reach_within_the_distance_nearest_first(two_rivers):
    out = rivers.reaches_near(LAT, LON, max_distance_m=500)
    assert [r["river_id"] for r in out] == [230260670, 230260671]
    assert out[0]["distance_m"] < out[1]["distance_m"] < 500 and out[1]["strahler_order"] == 7
    assert [r["river_id"] for r in rivers.reaches_near(LAT, LON, max_distance_m=50)] == [230260670]
    assert len(rivers.reaches_near(LAT, LON, max_distance_m=500, limit=1)) == 1


def test_snap_prefers_the_main_channel_and_says_a_smaller_stream_was_nearer(two_rivers):
    res = rivers.snap_to_river(LAT, LON)
    assert res["snapped"] is True and res["choice"] == "main_channel"
    assert res["river_id"] == 230260671 and res["strahler_order"] == 7
    assert res["nearer"]["river_id"] == 230260670 and res["nearer"]["distance_m"] < res["distance_m"]
    assert res["n_candidates"] == 2 and res["mixed_orders"] is True
    assert "to the main channel" in res["message"] and "order 7" in res["message"]
    assert "a smaller stream is" in res["message"]


def test_snap_can_still_take_the_nearest_line(two_rivers):
    res = rivers.snap_to_river(LAT, LON, prefer="nearest")
    assert res["river_id"] == 230260670 and res["choice"] == "nearest" and res["nearer"] is None
    with pytest.raises(ValueError):
        rivers.snap_to_river(LAT, LON, prefer="widest")


def test_snap_keeps_the_nearest_when_the_main_channel_is_beyond_the_tolerance(two_rivers, monkeypatch):
    res = rivers.snap_to_river(LAT, LON, max_distance_m=50)
    assert res["river_id"] == 230260670 and res["choice"] == "nearest" and res["n_candidates"] == 1
    assert res["mixed_orders"] is False
    # one reach in reach: the gauge's area has nothing to decide, so the routing tables are not read
    monkeypatch.setattr(rivers, "upstream_area", lambda *a, **k: pytest.fail("no area match with one candidate"))
    assert rivers.snap_to_river(LAT, LON, max_distance_m=50, area_km2=3000.0)["choice"] == "nearest"
    # ...and names the bigger river beyond it (a braided river's centreline can be far from its water).
    assert res["larger"]["river_id"] == 230260671 and res["larger"]["strahler_order"] == 7
    assert "A larger river (reach 230260671, order 7) is" in res["message"]


def test_snap_names_no_larger_river_when_the_chosen_one_is_the_biggest(two_rivers):
    assert rivers.snap_to_river(LAT, LON)["larger"] is None
    assert rivers.snap_to_river(LAT, LON, max_distance_m=50, prefer="nearest")["larger"]["river_id"] == 230260671


def test_main_channel_ranks_by_order_then_distance():
    cands = [{"river_id": 1, "strahler_order": 2, "distance_m": 60.0},
             {"river_id": 2, "strahler_order": 9, "distance_m": 380.0},
             {"river_id": 3, "strahler_order": 9, "distance_m": 200.0},
             {"river_id": 4, "strahler_order": None, "distance_m": 5.0}]
    assert rivers.main_channel(cands)["river_id"] == 3
    assert rivers.main_channel([]) is None


def test_a_gauge_with_a_known_area_snaps_to_the_reach_whose_area_matches(two_rivers, monkeypatch):
    areas = {230260670: 30.0, 230260671: 2900.0}
    monkeypatch.setattr(rivers, "upstream_area", lambda rid, lat=None, lon=None: {"upstream_area_km2": areas[rid]})
    small = rivers.snap_to_river(LAT, LON, area_km2=25.0)
    assert small["choice"] == "area" and small["river_id"] == 230260670 and small["area_ratio"] == 1.2
    big = rivers.snap_to_river(LAT, LON, area_km2=3000.0)
    assert big["choice"] == "area" and big["river_id"] == 230260671 and big["nearer"]["river_id"] == 230260670
    assert "upstream area" in big["message"] and "the nearest line is" in big["message"]


def test_match_by_area_skips_unreadable_units_and_needs_an_area(monkeypatch):
    cands = [{"river_id": 1, "distance_m": 10.0}, {"river_id": 2, "distance_m": 90.0}]

    def area(rid, lat=None, lon=None):
        if rid == 1:
            raise LookupError("no unit")
        return {"upstream_area_km2": 500.0}

    monkeypatch.setattr(rivers, "upstream_area", area)
    assert rivers.match_by_area(cands, 480.0)["reach"]["river_id"] == 2
    assert rivers.match_by_area(cands, None) is None
    assert rivers.match_by_area(cands, 0) is None


def test_snap_rejects_a_point_off_the_earth():
    with pytest.raises(ValueError):
        rivers.snap_to_river(95, 0)


# ── the reach record ─────────────────────────────────────────────────────────


def _retro(rid: int, years: int = 30) -> dict:
    idx = pd.date_range("1990-01-01", periods=int(years * 365.25), freq="D")
    rng = np.random.default_rng(7)
    seasonal = 50 + 30 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    flow = seasonal + rng.gamma(2.0, 10.0, len(idx))
    return {str(rid): [round(float(v), 2) for v in flow],
            "datetime": [d.strftime("%Y-%m-%dT%H:%M:%S") for d in idx],
            "metadata": {"gen_date": "2026-10-01"}}


def test_reach_record_analyses_the_simulated_flow_like_a_gauge(monkeypatch):
    seen = {}

    def fake_json(url, params=None):
        seen["url"], seen["params"] = url, params
        return _retro(230260670)

    monkeypatch.setattr(rivers, "_fetch_json", fake_json)
    store: dict = {}
    res = rivers.reach_record(230260670, store=store)
    assert seen["url"].endswith("/retrospectivedaily/230260670") and seen["params"]["format"] == "json"
    assert res["modelled"] is True and res["label"] == "modelled"
    assert res["licence"] == "CC BY 4.0" and "GEOGLOWS" in res["attribution"]
    assert res["notes"][0].startswith("Simulated by the GEOGLOWS v2")
    assert res["methods"][0]["name"].startswith("GEOGLOWS v2")
    assert res["ffa"]["n_years"] >= 29
    lp3 = res["ffa"]["fits"]["lp3"]
    assert len(lp3["ci"]) == len(res["ffa"]["return_periods"])
    assert res["fdc"]["q95"] < res["fdc"]["q50"] < res["fdc"]["q10"]
    assert len(res["monthly_regime"]["mean"]) == 12
    assert res["generated"] == "2026-10-01"
    assert len(store["series"]) == len(_retro(230260670)["datetime"])
    json.dumps(res, default=str)


def test_reach_record_years_asks_for_the_last_n_years(monkeypatch):
    seen = {}

    def fake_json(url, params=None):
        seen.update(params or {})
        return _retro(230260670, years=12)

    monkeypatch.setattr(rivers, "_fetch_json", fake_json)
    rivers.reach_record(230260670, years=10)
    assert len(seen["start_date"]) == 8 and seen["start_date"].isdigit()


def test_reach_record_from_a_point_snaps_first(monkeypatch, one_river):
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: _retro(230260670))
    res = rivers.reach_record(lat=LAT, lon=LON)
    assert res["river_id"] == 230260670 and res["snap"]["snapped"] is True


def test_reach_record_on_a_hillside_says_so_instead_of_a_record(monkeypatch, one_river):
    monkeypatch.setattr(rivers, "_fetch_json", lambda *a, **k: pytest.fail("no record for a hillside"))
    res = rivers.reach_record(lat=LAT, lon=LON, max_distance_m=5)
    assert res["river_id"] is None and res["error"].startswith("No stream within 5 m")


def test_reach_record_with_an_empty_answer(monkeypatch):
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: {"datetime": []})
    res = rivers.reach_record("230260670")
    assert res["error"] and res["modelled"] is True


def test_reach_summary_drops_the_daily_arrays(monkeypatch):
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: _retro(230260670))
    res = rivers.reach_summary(230260670)
    assert "series" not in res and set(res["fdc"]) == {"q95", "q50", "q10"}
    assert res["ffa"]["fits"]["gev_lmoments"]["q"]


@pytest.mark.parametrize("bad", ["abc", 12, 999999999999])
def test_a_river_id_must_be_a_geoglows_one(bad):
    with pytest.raises(ValueError):
        rivers.reach_record(bad)


def test_forecast_stats_is_a_thin_modelled_fetch(monkeypatch):
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: {
        "datetime": ["2026-10-08T00:00:00", "2026-10-08T03:00:00"], "flow_avg": [1.5, ""], "flow_max": [2, 3],
        "metadata": {"gen_date": "2026-10-08"}})
    res = rivers.forecast_stats(230260670)
    assert res["modelled"] is True and res["flow_avg"] == [1.5, None] and res["flow_max"] == [2.0, 3.0]


# ── network: upstream area and the trace ─────────────────────────────────────

# Three reaches in a line flowing east at lat 47 (230000001 -> 230000002 -> 230000003 -> outlet), and a
# tributary 230000004 joining 230000002.
NET = {230000001: 230000002, 230000002: 230000003, 230000003: 0, 230000004: 230000002}
AREA_KM2 = {230000001: 10.0, 230000002: 5.0, 230000003: 2.0, 230000004: 3.0}
SPAN = {230000001: (8.00, 8.01), 230000002: (8.01, 8.02), 230000003: (8.02, 8.03), 230000004: (8.01, 8.01)}


@pytest.fixture
def small_network(monkeypatch):
    connect = "\n".join(f"{rid},{ds},1,0" for rid, ds in NET.items())
    rows = ["LINKNO,lon,lat,area_sqm"]
    for rid, (a, b) in SPAN.items():
        half = AREA_KM2[rid] * 1e6 / 2
        rows += [f"{rid},{a},47.0,{half}", f"{rid},{b},47.0,{half}"]
    texts = {"rapid_connect.csv": connect, "weight_era5_721x1440.csv": "\n".join(rows)}

    def fake_text(url):
        return texts[url.rsplit("/", 1)[1]]

    def fake_tile(z, x, y):
        out = {}
        for rid, (a, b) in SPAN.items():
            if rid == 230000004:
                continue
            down = rivers._world(b, 47.0, z)
            up = rivers._world(a, 47.0, z)
            if int(down[0] // 4096) == x and int(down[1] // 4096) == y:
                out[rid] = {"order": 2, "lines": [[down, up]]}  # TDX-Hydro draws downstream to upstream
        return out

    monkeypatch.setattr(rivers, "_fetch_text", fake_text)
    monkeypatch.setattr(rivers, "_tile_reaches", fake_tile)


def test_upstream_area_adds_the_unit_catchments(small_network):
    res = rivers.upstream_area(230000002)
    assert res["upstream_area_km2"] == pytest.approx(18.0)
    assert res["n_reaches_upstream"] == 3 and res["unit_area_km2"] == pytest.approx(5.0)
    assert res["modelled"] is True and "approximate" in res["note"]


def test_trace_downstream_follows_the_network_to_the_outlet(small_network):
    gauges = [
        {"source": "x", "station_id": "near", "name": "On the river", "latitude": 47.0005, "longitude": 8.015},
        {"source": "x", "station_id": "far", "name": "Elsewhere", "latitude": 48.0, "longitude": 9.0},
        {"source": "x", "station_id": "bad", "latitude": None, "longitude": 8.0},
    ]
    res = rivers.trace_downstream(230000001, stations=gauges)
    assert res["n_reaches"] == 3 and [r["river_id"] for r in res["reaches"]] == [230000001, 230000002, 230000003]
    assert not res["truncated"]
    expected_km = 0.03 * 111.32 * math.cos(math.radians(47.0))
    assert res["length_km"] == pytest.approx(expected_km, rel=0.02)
    coords = res["geometry"]["coordinates"]
    assert coords[0][0] == pytest.approx(8.00, abs=1e-4) and coords[-1][0] == pytest.approx(8.03, abs=1e-4)
    assert res["outlet"]["river_id"] == 230000003 and res["outlet"]["lon"] == pytest.approx(8.03, abs=1e-4)
    assert [g["station_id"] for g in res["gauges"]] == ["near"]
    assert res["gauges"][0]["along_km"] == pytest.approx(0.015 * 111.32 * math.cos(math.radians(47.0)), rel=0.05)
    assert res["upstream_area_km2"] == pytest.approx(10.0)
    assert "CC BY-SA" in res["geometry_licence"]
    assert "3 reaches" in res["message"] and "1 gauge within 2 km" in res["message"]


def test_trace_downstream_from_a_hillside_does_not_trace(monkeypatch):
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000: {
        "snapped": False, "message": "No stream within 1,000 m of this point."})
    res = rivers.trace_downstream(lat=47.0, lon=8.0)
    assert res["snapped"] is False and res["message"].startswith("No stream")


def test_trace_downstream_stops_at_max_reaches(small_network):
    res = rivers.trace_downstream(230000001, stations=[], max_reaches=2)
    assert res["truncated"] is True and res["n_reaches"] == 2
    assert any("Stopped after 2 reaches" in n for n in res["notes"])


def test_upstream_ids_lists_every_reach_that_drains_to_it(small_network):
    res = rivers.upstream_ids(230000003)
    assert res["ids"][0] == 230000003 and set(res["ids"]) == set(NET)
    assert res["n_upstream"] == 4 and res["truncated"] is False and res["min_area_km2"] is None
    assert res["upstream_area_km2"] == pytest.approx(20.0) and res["modelled"] is True
    head = rivers.upstream_ids(230000001)
    assert head["ids"] == [230000001] and head["n_upstream"] == 1


def test_upstream_ids_keeps_the_biggest_drainage_when_capped(small_network):
    # Drainage areas: 230000003 20, 230000002 18, 230000001 10, 230000004 3 km2.
    res = rivers.upstream_ids(230000003, max_n=3)
    assert res["truncated"] is True and res["n_upstream"] == 4 and res["n_ids"] == 3
    assert res["ids"] == [230000003, 230000002, 230000001]  # breadth-first, the small tributary dropped
    assert res["min_area_km2"] == pytest.approx(10.0) and "3 largest" in res["message"]


def test_upstream_ids_agrees_with_upstream_area(small_network):
    assert rivers.upstream_ids(230000002)["upstream_area_km2"] == rivers.upstream_area(230000002)["upstream_area_km2"]


def test_downstream_ids_walk_to_the_outlet(small_network):
    res = rivers.downstream_ids(230000004)
    assert res["ids"] == [230000004, 230000002, 230000003] and res["outlet_id"] == 230000003
    assert res["truncated"] is False and "outlet" in res["message"]
    short = rivers.downstream_ids(230000001, max_n=2)
    assert short["ids"] == [230000001, 230000002] and short["truncated"] is True


def test_trace_needs_a_river_or_a_point():
    with pytest.raises(ValueError):
        rivers.trace_downstream()


def test_vpu_candidates_put_the_nearest_unit_first():
    assert rivers._vpu_candidates(230260670, 46.9, 7.4)[0] == 209
    assert set(rivers._vpu_candidates(230260670)) == {207, 208, 209}


# ── the HTTP seam ────────────────────────────────────────────────────────────


def test_get_bytes_sends_a_range_and_trims_a_full_answer(tmp_path):
    import httpx

    from aquascope.utils.http_client import CachedHTTPClient

    body = bytes(range(256)) * 4
    seen = []

    def handler(request):
        seen.append(request.headers.get("range"))
        if request.url.path.endswith("/ranged"):
            return httpx.Response(206, content=body[10:20])
        return httpx.Response(200, content=body)  # a server that ignores Range

    client = CachedHTTPClient(cache_dir=tmp_path, retries=1)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.get_bytes("https://example.test/ranged", start=10, end=19) == body[10:20]
    assert client.get_bytes("https://example.test/full", start=10, end=19) == body[10:20]
    assert client.get_bytes("https://example.test/full") == body
    assert seen == ["bytes=10-19", "bytes=10-19", None]


def test_the_api_seam_tries_once_more_after_a_failure(monkeypatch):
    tries = []

    class Flaky:
        def get_json(self, url, params=None):
            tries.append(url)
            if len(tries) == 1:
                raise RuntimeError("HTTP 500")
            return {"ok": True}

    monkeypatch.setattr(rivers, "_client", lambda kind: Flaky())
    monkeypatch.setattr(rivers, "_fetch_json", _REAL_FETCH_JSON)  # tests/conftest.py took it offline
    assert rivers._fetch_json("https://x.test/a") == {"ok": True} and len(tries) == 2
