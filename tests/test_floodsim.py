"""Flood it (#658): the local-inertial solver against still water, the volume budget, Manning's normal depth and
Ritter's dam break; the PNG and Terrarium decoding; the channel; the agreement measure."""

from __future__ import annotations

import math
import struct
import zlib

import numpy as np
import pytest

from aquascope import floodsim as fs


def test_still_water_stays_still():
    """Well-balanced: a lake at rest over a bumpy bed, with dry islands, does not move (closed or open edges)."""
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 3, size=(40, 50))
    z[10:14, 10:14] = 5.0          # an island above the water
    level = 2.5
    h0 = np.maximum(level - z, 0.0)
    # Raise the edges above the lake so the open-edge case has nothing to drain.
    z[0, :] = z[-1, :] = z[:, 0] = z[:, -1] = 6.0
    h0[0, :] = h0[-1, :] = h0[:, 0] = h0[:, -1] = 0.0
    for closed in (True, False):
        r = fs.simulate(z, 10.0, duration_s=600.0, h0=h0, n=0.03, closed=closed)
        assert r["steps"] > 50
        assert np.abs(r["h"] - h0).max() < 1e-12
        assert r["volume"]["outflow_m3"] == 0.0


def test_mass_conservation_closed_and_open():
    rng = np.random.default_rng(2)
    z = np.cumsum(rng.normal(0, 0.2, size=(60, 60)), axis=0) + rng.uniform(0, 1, size=(60, 60))
    # Closed: everything poured in stays in.
    r = fs.simulate(z, 20.0, duration_s=3600.0, n=0.05, sources=[(30, 30, 50.0)], closed=True)
    v = r["volume"]
    assert v["inflow_m3"] == pytest.approx(50.0 * 3600.0)
    assert v["outflow_m3"] == 0.0
    assert abs(v["error_rel"]) < 1e-12
    assert r["h"].min() >= 0.0
    # Open: what leaves at the edges is counted, and the budget still closes.
    r = fs.simulate(z, 20.0, duration_s=6 * 3600.0, n=0.05, sources=[(30, 30, 50.0)])
    v = r["volume"]
    assert v["outflow_m3"] > 0.0
    assert abs(v["error_rel"]) < 1e-12


def test_normal_depth_on_a_slope():
    """Steady uniform flow down a plane reaches Manning's normal depth, h = (n q / sqrt(S))^(3/5)."""
    s, n, q, dx, nx = 1e-3, 0.05, 2.0, 10.0, 400
    z = -(np.arange(nx) * dx * s)[None, :]
    r = fs.simulate(z, dx, duration_s=6 * 3600.0, n=n, sources=[(0, 5, q * dx)], closed="nsw")
    expected = (n * q / math.sqrt(s)) ** 0.6
    mid = r["h"][0, 150:250]
    assert np.abs(mid / expected - 1).max() < 0.02


def test_ritter_dam_break_within_the_stated_tolerance():
    """Ritter (1892): dry, flat, frictionless bed. The local-inertial scheme has no advection term, so it cannot match
    the supercritical front (the literature says so); the stated tolerance is a relative L1 depth error under 0.20
    at 30 s, which holds as the grid is refined (0.177 to 0.185 for dx = 4 m to 0.5 m), with the volume exact."""
    h0, dx, t = 1.0, 1.0, 30.0
    nx = 600
    x = (np.arange(nx) + 0.5) * dx - 200.0
    z = np.zeros((1, nx))
    h = np.where(x < 0, h0, 0.0)[None, :]
    r = fs.simulate(z, dx, duration_s=t, h0=h, n=0.0, closed=True)
    hm, ha = r["h"][0], fs.ritter(x, r["t_s"], h0)
    l1 = np.abs(hm - ha).sum() / ha.sum()
    assert l1 < 0.20
    # Upstream of the rarefaction (beyond a 20 m numerical spread) the water has not moved yet.
    c0 = math.sqrt(fs.G * h0)
    calm = x < -c0 * t - 20
    assert np.abs(hm[calm] - h0).max() < 1e-6
    # The front lags Ritter's 2 c0 t (the known under-estimate of the scheme) but has moved.
    front = x[hm > 0.01].max()
    assert 0.25 * 2 * c0 * t < front < 2 * c0 * t
    assert abs(r["volume"]["error_rel"]) < 1e-12


def test_ritter_formula():
    c0 = math.sqrt(fs.G * 2.0)
    assert fs.ritter([-100.0], 1.0, 2.0)[0] == 2.0
    assert fs.ritter([0.0], 10.0, 2.0)[0] == pytest.approx(4 / 9 * 2.0)
    assert fs.ritter([2 * c0 * 10 + 1], 10.0, 2.0)[0] == 0.0


def _png(rows: list[list[tuple[int, int, int]]], filters: list[int]) -> bytes:
    """A tiny RGB PNG with the given per-row filters (0 none, 1 sub, 2 up)."""
    h, w = len(rows), len(rows[0])
    raw = b""
    prev = bytes(3 * w)
    for row, f in zip(rows, filters):
        line = bytes(v for px in row for v in px)
        if f == 1:
            enc = bytes((line[i] - (line[i - 3] if i >= 3 else 0)) & 255 for i in range(len(line)))
        elif f == 2:
            enc = bytes((line[i] - prev[i]) & 255 for i in range(len(line)))
        else:
            enc = line
        raw += bytes([f]) + enc
        prev = line

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_png_and_terrarium_decoding():
    rows = [[(128, 0, 0), (128, 10, 128)], [(127, 255, 0), (129, 2, 64)], [(128, 100, 200), (128, 1, 1)]]
    rgb = fs.decode_png_rgb(_png(rows, [0, 1, 2]))
    assert rgb.tolist() == [[list(p) for p in r] for r in rows]
    z = fs.terrarium_elevation(rgb)
    assert z[0, 0] == 0.0                      # 128*256 - 32768
    assert z[0, 1] == pytest.approx(10.5)
    assert z[1, 0] == pytest.approx(-1.0)
    assert z[1, 1] == pytest.approx(258.25)


def test_channel_depth_and_burning():
    # d = (n Q / (W sqrt S))^(3/5)
    assert fs.channel_depth(100.0, 25.0, 1e-3) == pytest.approx((0.035 * 100 / (25 * math.sqrt(1e-3))) ** 0.6)
    # The slope is kept within the range.
    assert fs.channel_depth(100.0, 25.0, 0.0) == fs.channel_depth(100.0, 25.0, fs.SLOPE_RANGE[0])
    z = np.array([[10.0, 9.0, 9.5, 8.0, 8.2]])
    out = fs.burn_channel(z, [(0, i) for i in range(5)], 2.0)
    # The bank is a running minimum, so the bed never rises downstream.
    assert out[0].tolist() == [8.0, 7.0, 7.0, 6.0, 6.0]


def test_pixel_size_and_cells():
    assert fs.pixel_size_m(0.0, 0) == pytest.approx(fs.EARTH_CIRCUMFERENCE_M / 256)
    grid = {"zoom": 12, "x0": 0, "y0": 0, "size": 512}
    x, y = fs._world_px(-2.2, 52.06, 12)
    grid["x0"], grid["y0"] = int(x) - 256, int(y) - 256
    assert fs.cell_of(grid, -2.2, 52.06) == (256, 256)
    lon, lat = fs.cell_lonlat(grid, 256, 256)
    assert abs(float(lon) + 2.2) < 3e-4 and abs(float(lat) - 52.06) < 3e-4


def test_compare_depths():
    m = np.array([[0.0, 1.0, 2.0], [0.5, 0.0, 0.0]])
    ref = np.array([[0.0, 2.0, 0.0], [1.0, 1.0, np.nan]])
    c = fs.compare_depths(m, ref)
    assert (c["wet_both"], c["model_only"], c["reference_only"]) == (2, 1, 1)
    assert c["csi"] == 0.5
    assert c["depth_diff_m"]["mean"] == pytest.approx(-0.75)


def test_matches_the_javascript_twin_fixture():
    """explorer/tests/floodsim-core.test.mjs runs the same grid in JS against this fixture; here it is regenerated
    and checked so the two cannot drift apart silently."""
    import json
    from pathlib import Path

    fx = json.loads((Path(__file__).parent / "fixtures" / "floodsim_twin.json").read_text())
    r = fs.simulate(np.array(fx["z"]), fx["dx"], duration_s=fx["duration_s"], n=np.array(fx["n"]),
                    sources=[tuple(fx["source"])])
    assert r["steps"] == fx["steps"]
    # Round-off differs between numpy builds and CPUs (1.5e-8 m seen on CI): within a micrometre, as the JS twin.
    assert np.abs(r["h"] - np.array(fx["h"])).max() < 1e-6


# ── the full feature: edges, inflows, the steady state, the grid from a reach, the files ────────────────────


def test_matches_the_javascript_twin_with_edges_and_tables():
    """The second fixture: walls, a normal-depth outfall, a free outfall face and two inflows, one a table."""
    import json
    from pathlib import Path

    fx = json.loads((Path(__file__).parent / "fixtures" / "floodsim_twin_edges.json").read_text())
    edges = {k: np.array([np.nan if x is None else x for x in v]) for k, v in fx["edges"].items()}
    srcs = [(a, b, c) for a, b, c in fx["sources"]]
    r = fs.simulate(np.array(fx["z"]), fx["dx"], duration_s=fx["duration_s"], n=np.array(fx["n"]), sources=srcs,
                    edges=edges, table_dt_s=fx["table_dt_s"])
    assert r["steps"] == fx["steps"]
    assert np.abs(r["h"] - np.array(fx["h"])).max() < 1e-6
    assert abs(r["volume"]["error_rel"]) < 1e-12


def test_normal_depth_outfall_passes_the_flow_at_steady_state():
    """A channel walled everywhere but a normal-depth outfall at its foot: the inflow leaves there, and the depth at
    the outfall is Manning's normal depth for the outfall's slope."""
    s, n, q, dx, nx = 1e-3, 0.05, 2.0, 10.0, 200
    z = -(np.arange(nx) * dx * s)[None, :]
    r = fs.simulate(z, dx, duration_s=8 * 3600.0, n=n, sources=[(0, 2, q * dx)], edges={"n": 0.0, "s": 0.0, "w": 0.0,
                                                                                      "e": s})
    flux_out = r["qx"][0, -1] * dx
    assert flux_out == pytest.approx(q * dx, rel=0.01)
    assert r["h"][0, -1] == pytest.approx((n * q / math.sqrt(s)) ** 0.6, rel=0.02)
    # Walls hold everything.
    r = fs.simulate(z, dx, duration_s=3600.0, n=n, sources=[(0, 2, q * dx)], edges={"n": 0, "s": 0, "w": 0, "e": 0})
    assert r["volume"]["outflow_m3"] == 0.0
    assert abs(r["volume"]["error_rel"]) < 1e-12


def test_table_inflows_and_the_steady_state_rule():
    assert fs.flow_at(5.0, 1e9, 3600.0) == 5.0
    assert fs.flow_at([0.0, 10.0, 4.0], 1800.0, 3600.0) == 5.0
    assert fs.flow_at([0.0, 10.0, 4.0], 1e9, 3600.0) == 4.0
    z = (-0.002 * np.arange(30) * 20)[None, :] + 0.2 * np.abs(np.arange(9) - 4)[:, None]
    r = fs.simulate(z, 20.0, duration_s=48 * 3600.0, n=0.05, sources=[(4, 1, 3.0)], steady_tol=fs.STEADY_TOL,
                    edges={"n": 0, "s": 0, "w": 0, "e": 0.002}, frames_every_s=fs.frame_times(48))
    assert r["steady_at_s"] is not None and r["steady_at_s"] % 3600 == 0 and r["steady_at_s"] < 48 * 3600
    last = r["checks"][-1]
    assert last["d_area"] < fs.STEADY_TOL and last["d_volume"] < fs.STEADY_TOL
    assert r["times_s"][-1] == pytest.approx(r["steady_at_s"])
    assert fs.frame_times(5) == [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9000, 9900, 10800, 14400,
                                 18000]


def _curve(q2: float) -> dict:
    return {"T": list(fs.FIT_RETURN_PERIODS), "q": [q2 * k for k in (1.0, 1.5, 1.8, 2.2, 2.5, 2.8, 3.1, 3.5)]}


def test_flow_curves_and_the_same_return_period():
    years = list(range(1940, 2025))
    rng = np.random.default_rng(3)
    am = np.exp(rng.normal(np.log(100), 0.35, len(years)))
    c = fs.flow_curve(years, am)
    assert c["n_years"] == 85 and c["T"] == list(fs.FIT_RETURN_PERIODS)
    assert all(b >= a for a, b in zip(c["q"], c["q"][1:]))
    assert fs.flow_curve(years[:5], am[:5])["error"]
    a, b = _curve(100.0), _curve(10.0)
    assert fs.flow_for(a, 100) == pytest.approx(280.0)
    assert fs.flow_for(a, 1000) == pytest.approx(350.0)        # held at 500 years
    assert fs.return_period_for(a, 280.0) == pytest.approx(100.0)
    assert fs.return_period_for(a, 50.0) is None
    assert fs.matched_flow(a, 280.0, b) == pytest.approx(28.0)
    assert fs.matched_flow(a, 50.0, b) == pytest.approx(5.0)   # half the 2-year flow: half the other's
    assert fs.matched_flow(a, 700.0, b) == pytest.approx(70.0)


class _Net:
    """A small GEOGLOWS-like network: a main river 1 -> 2 -> 3 -> 4 (4 leaves), a tributary 10 joining 3, a
    headwater 11 above 2, and 5 above 1 (outside the grid)."""

    def __init__(self):
        self.vpu = 0
        self.ids = np.array([5, 1, 2, 3, 4, 10, 11])
        self.ds = np.array([1, 2, 3, 4, -1, 3, 2])
        self.area_m2 = np.array([5e7, 1e7, 1e7, 1e7, 1e7, 2e7, 1e6])
        self.lat = np.full(7, 52.0)
        self.lon = np.full(7, -2.0)

    def index(self, rid):
        hit = np.nonzero(self.ids == int(rid))[0]
        return int(hit[0]) if hit.size else None

    def children(self, i):
        return [int(j) for j in np.nonzero(self.ds == self.ids[i])[0]]

    def upstream(self, i):
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(self.children(j))
        return out


def _valley_plan():
    """A straight valley running east, 300 x 120 cells at zoom 13, the main river along its middle row."""
    zoom = 13
    x0, y0 = fs._world_px(-2.05, 52.01, zoom)
    x0, y0 = int(x0), int(y0)
    nx, ny = 300, 120
    lon = lambda c: fs._px_lonlat(x0 + c + 0.5, y0, zoom)[0]  # noqa: E731
    lat = lambda r: fs._px_lonlat(x0, y0 + r + 0.5, zoom)[1]  # noqa: E731
    mid = lat(60)
    reaches = {
        5: {"order": 4, "lines": [[(lon(-40), mid), (lon(-1), mid)]]},
        1: {"order": 4, "lines": [[(lon(-1), mid), (lon(80), mid)]]},
        2: {"order": 4, "lines": [[(lon(80), mid), (lon(160), mid)]]},
        3: {"order": 4, "lines": [[(lon(160), mid), (lon(230), mid)]]},
        4: {"order": 4, "lines": [[(lon(230), mid), (lon(301), mid)]]},
        10: {"order": 3, "lines": [[(lon(160), lat(-5)), (lon(160), mid)]]},
        11: {"order": 2, "lines": [[(lon(100), lat(30)), (lon(100), mid)]]},
    }
    curves = {r: _curve(q2) for r, q2 in {5: 50.0, 1: 52.0, 2: 54.0, 3: 75.0, 4: 76.0, 10: 20.0, 11: 1.0}.items()}
    rows = np.arange(ny)[:, None]
    cols = np.arange(nx)[None, :]
    z = 20.0 - 0.002 * cols + 0.05 * np.abs(rows - 60) + np.where(np.abs(rows - 60) > 40, 30.0, 0.0)
    plan = fs.plan_window(2, lat(60), lon(120), net=_Net(), reaches=reaches, curves=curves, zoom=zoom, km=50)
    plan["window"] = fs.window(x0, y0, nx, ny, zoom)    # the valley test sets its own window
    for r in list(plan["reaches"]):
        if int(r) == 5:
            del plan["reaches"][r]
    plan["reaches"]["1"]["outside"] = [5]
    plan["reaches"]["1"]["outside_curves"] = {"5": curves[5]}
    return plan, z.astype("f8")


def test_corridor_and_links_follow_the_network():
    net = _Net()
    reaches = {r: {"order": 4, "lines": [[(-2.0, 52.0), (-1.99, 52.0)]]} for r in (5, 1, 2, 3, 4, 10, 11)}
    assert fs.corridor(net, 2, reaches, km=0.5) == [1, 2, 3]
    assert fs.corridor(net, 2, reaches, km=50.0) == [5, 1, 2, 3, 4]
    ups, outside, ds = fs._links(net, 3, drawn={2, 3, 4}, known={2, 3, 4, 10})
    assert (sorted(ups), outside, ds) == ([2], [10], 4)


def test_prepare_grid_feeds_every_river_and_opens_the_outlet():
    plan, z = _valley_plan()
    g = fs.prepare_grid(plan, z, {"return_period": 100})
    # The valley cut: the plateau 30 m up on either side is left out.
    assert g["ny"] < 120 and g["nx"] == 300
    by = {s["river_id"]: s for s in g["sources"]}
    # The main river's reach 1 crosses the west edge and brings its own flow in there, the tributary 10 enters from
    # the north edge, the headwater 11 starts inside; reaches 2, 3 and 4 have rivers above them in the grid.
    assert set(by) == {1, 10, 11}
    assert by[1]["enters"] and by[1]["cell"][1] <= fs.SNAP_CELLS
    assert by[1]["flow"] == pytest.approx(fs.flow_for(_curve(52.0), 100))
    assert by[10]["enters"] and by[10]["cell"][0] <= fs.SNAP_CELLS
    assert not by[11]["enters"]
    assert g["click"]["flow_cms"] == pytest.approx(fs.flow_for(_curve(54.0), 100))
    # The outlet: a normal-depth outfall on the east edge across the valley floor; walls elsewhere.
    assert g["exits"] and g["exits"][0]["side"] == "e"
    assert (g["edges"]["e"] > 0).sum() >= 3 and (g["edges"]["w"] == 0).all() and (g["edges"]["n"] == 0).all()
    # Channels burnt, never upward downstream, with Manning's n.
    row = int(np.argmin(g["z_burnt"][:, 150]))
    bed = g["z_burnt"][row, :]
    assert (np.diff(bed[5:-5]) <= 1e-9).all()
    assert g["n"][row, 150] == fs.N_CHANNEL
    # A hydrograph at the clicked reach becomes a table at every inflow, at the same return period hour by hour.
    q2c = _curve(54.0)["q"][0]
    h = fs.prepare_grid(plan, z, {"series": {"t_s": [0, 7200, 14400], "q": [q2c, fs.flow_for(_curve(54.0), 10), q2c]},
                                  "label": "a test hydrograph"})
    s1 = next(s for s in h["sources"] if s["river_id"] == 1)
    assert len(s1["flow"]) == 5 and s1["flow"][2] == pytest.approx(fs.flow_for(_curve(52.0), 10), rel=1e-3)
    assert h["hours"] == 4.0 and h["label"] == "a test hydrograph"


def test_a_run_on_the_valley_reaches_steady_state_and_writes_its_files(tmp_path):
    from aquascope.io.mesh import read_mesh
    from aquascope.mesh_views import mesh_lenses

    plan, z = _valley_plan()
    g = fs.prepare_grid(plan, z, {"return_period": 10}, coarse=True)
    assert g["coarse"] and g["window"]["zoom"] == 12
    res = fs.run_grid(g, cap_hours=24)
    s = fs.run_summary(g, res)
    assert s["steady_at_h"] is not None and s["wet_km2"] > 0 and abs(s["volume"]["error_rel"]) < 1e-9
    assert s["volume"]["outflow_m3"] > 0
    out = fs.to_ugrid(str(tmp_path / "run.nc"), g, res, start="2026-10-27", title="valley test")
    m = read_mesh(out["path"])
    assert m.n_cells == out["faces"] and m.variable == "depth" and m.has_velocity
    assert len(m.times) == len(res["times_s"]) and m.times[0] == "2026-10-27T00:00:00"
    # The file's depth is the water above the terrain (the channel's cut is the model's, not the ground's).
    over = fs.depth_over_terrain(g, res["h_max"])
    framed = max(float(fs.depth_over_terrain(g, f).max()) for f in res["frames"])   # the file keeps the frames
    assert np.nanmax(m.max_value) == pytest.approx(framed, rel=1e-3)
    lenses = mesh_lenses(m)
    assert {"hazard", "arrival"} <= set(lenses.meta["available"])
    tif = fs.to_geotiff(str(tmp_path / "depth.tif"), g, over)
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(tif["path"]) as src:
        assert src.crs.to_epsg() == 3857 and src.width == g["nx"] and src.height == g["ny"]
        a = src.read(1)
        assert np.nanmax(np.where(a == src.nodata, np.nan, a)) == pytest.approx(float(over.max()), rel=1e-5)
        w, s_, e, n_ = g["window"]["bounds"]
        from pyproj import Transformer

        lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(src.bounds.left, src.bounds.top)
        assert lon == pytest.approx(w, abs=1e-6) and lat == pytest.approx(n_, abs=1e-6)


def test_the_annual_maxima_store_is_searched_by_id():
    """The store reader on a fake uncompressed store: a binary search over the id chunks, then one column."""
    import json as _json

    ids = np.array([10, 20, 30, 40, 50, 60, 70], dtype="<i4")
    q = np.arange(3 * 7, dtype="<f4").reshape(3, 7)
    meta = {"metadata": {
        "river_id/.zarray": {"shape": [7], "chunks": [3], "dtype": "<i4", "compressor": None, "order": "C"},
        "daily/.zarray": {"shape": [3, 7], "chunks": [3, 4], "dtype": "<f4", "compressor": None, "order": "C"},
        "time/.zarray": {"shape": [3], "chunks": [3], "dtype": "<i8", "compressor": None, "order": "C"},
        "time/.zattrs": {"units": "days since 1940-01-01"}}}
    files = {".zmetadata": _json.dumps(meta).encode(), "time/0": np.array([0, 366, 731], dtype="<i8").tobytes()}
    for i in range(3):
        chunk = np.zeros(3, dtype="<i4")
        part = ids[i * 3:i * 3 + 3]
        chunk[:part.size] = part
        files[f"river_id/{i}"] = chunk.tobytes()
    for j in range(2):
        block = np.zeros((3, 4), dtype="<f4")
        part = q[:, j * 4:j * 4 + 4]
        block[:, :part.shape[1]] = part
        files[f"daily/0.{j}"] = block.tobytes()
    fetch = lambda url: files.get(url.split("maximums.zarr/")[1])  # noqa: E731
    years, got = fs.annual_maxima([50, 20, 99], fetch=fetch)
    assert years == [1940, 1941, 1942]
    assert set(got) == {20, 50}
    assert got[20].tolist() == [1.0, 8.0, 15.0] and got[50].tolist() == [4.0, 11.0, 18.0]


def test_channel_cells_follow_the_valley_and_carry_water_through_faces():
    z = np.full((7, 9), 10.0)
    z[4, :] = 5.0                                  # the valley floor runs along row 4
    line = [(2, c) for c in range(9)]              # the drawn line runs two cells off it, over the bank
    snapped = fs.snap_chain(line, z)
    assert all(r == 4 for r, _ in snapped) and len(snapped) == 9
    # A diagonal step gets the lower of the two cells that close it, so water can pass through faces.
    zz = np.zeros((3, 3))
    zz[0, 1] = 9.0
    assert fs.four_connected([(0, 0), (1, 1)], zz) == [(0, 0), (1, 0), (1, 1)]
    # A cut is never deeper than the cap below the terrain plus the channel's own depth.
    hill = np.array([[5.0, 30.0, 4.0]])
    bed = fs.burn_channel(hill, [(0, 0), (0, 1), (0, 2)], 1.0, max_cut_m=10.0)
    assert bed[0].tolist() == [4.0, 19.0, 3.0]
    assert fs.burn_channel(hill, [(0, 0), (0, 1), (0, 2)], 1.0)[0, 1] == 4.0


def test_stream_tile_pieces_are_cut_to_their_tiles_and_joined_end_to_end():
    parts = fs._clip_line([(-10.0, 5.0), (5.0, 5.0), (5.0, 20.0)], (0.0, 0.0, 10.0, 10.0))
    assert parts == [[(0.0, 5.0), (5.0, 5.0), (5.0, 10.0)]]
    assert fs._clip_line([(20.0, 20.0), (30.0, 30.0)], (0.0, 0.0, 10.0, 10.0)) == []
    zoom = 14
    x0, y0 = fs._world_px(-2.0, 52.0, zoom)
    win = fs.window(int(x0), int(y0), 60, 20, zoom)
    lon = lambda c: fs._px_lonlat(int(x0) + c, int(y0), zoom)[0]  # noqa: E731
    lat = fs._px_lonlat(int(x0), int(y0) + 10.5, zoom)[1]
    # Three pieces of one line, given out of order and one reversed: the walk puts them back in order.
    pieces = [[(lon(40), lat), (lon(58), lat)], [(lon(2), lat), (lon(20), lat)], [(lon(40), lat), (lon(20), lat)]]
    cells = fs._reach_cells(win, pieces)
    cols = [c for _, c in cells]
    assert cols == sorted(cols) or cols == sorted(cols, reverse=True)
    assert len(cells) >= 55


# ── the three faces: the CLI, the MCP tool and Ask the map ─────────────────


def _offline(monkeypatch):
    plan, z = _valley_plan()
    monkeypatch.setattr(fs, "plan_window", lambda *a, **k: plan)
    monkeypatch.setattr(fs, "terrain_window", lambda *a, **k: {"z": z})
    return plan


def test_cli_flood_writes_the_depth_and_the_run(tmp_path, monkeypatch, capsys):
    from aquascope import cli

    _offline(monkeypatch)
    monkeypatch.setattr("sys.argv", ["aquascope", "flood", "2", "--rp", "10", "--hours", "12", "--out-dir",
                                     str(tmp_path)])
    cli.main()
    out = capsys.readouterr().out
    assert "the 10-year flow" in out and "km² under water" in out
    for name in ("flood-2-max-depth.tif", "flood-2.nc", "flood-2.json"):
        assert (tmp_path / name).exists(), name
    import json

    s = json.loads((tmp_path / "flood-2.json").read_text())
    assert s["river_id"] == 2 and s["wet_km2"] > 0 and s["licences"]["flows"].startswith("GEOGLOWS")


def test_mcp_flood_reach(tmp_path, monkeypatch):
    from aquascope.mcp_server import flood_reach

    _offline(monkeypatch)
    s = flood_reach(2, return_period=25, hours=12, out_dir=str(tmp_path))
    assert s["flow"] == "the 25-year flow" and s["sources"] and s["wet_km2"] > 0
    assert set(s["files"]) == {"depth_geotiff", "ugrid_netcdf"}
    assert "screening flood" in s["label"]


def test_ask_the_map_flood_reach_is_read_checked_and_said():
    from aquascope import map_commands as mc

    res = mc.parse_command("flood the Severn at Tewkesbury with 500 m3/s", today="2026-10-27")
    assert res["actions"] == [{"type": "flood_reach", "place": "Severn at Tewkesbury", "flow_cms": 500.0}]
    assert mc.parse_command("flood here", today="2026-10-27")["actions"] == [
        {"type": "flood_reach", "at": "here", "return_period": 100.0}]
    assert mc.parse_command("flood this river with the forecast peak", today="2026-10-27")["actions"][0]["flow"] == \
        "forecast_peak"
    assert mc.parse_command("flood depth on", today="2026-10-27")["actions"][0]["type"] == "set_layer"
    ok, why = mc.validate_action({"type": "flood_reach", "lat": 52.06, "lon": -2.21})
    assert why is None and ok["return_period"] == 100.0
    assert "return_period" in mc.validate_action({"type": "flood_reach", "at": "here", "return_period": 1000})[1]
    assert "flow must be" in mc.validate_action({"type": "flood_reach", "at": "here", "flow": "guess"})[1]
    assert mc.validate_action({"type": "flood_reach"})[1].startswith("flood_reach needs")
    assert mc.describe_action({"type": "flood_reach", "at": "here", "return_period": 10}) == \
        "Flood the selected river at the 10-year flow"
    assert "flood_reach" in mc.model_prompt("flood it")["system"] if isinstance(mc.model_prompt("flood it"), dict) \
        else "flood_reach" in str(mc.model_prompt("flood it"))


def test_design_flow_twin_fixture():
    """explorer/tests/design-flow-core.test.mjs reads the same curve and flows: the JS reads a flow off the curve
    and words the method exactly as floodsim.flow_for and flow_words do."""
    import json
    from pathlib import Path

    fx = json.loads((Path(__file__).parent / "fixtures" / "design_flow_twin.json").read_text(encoding="utf-8"))
    years = list(range(1940, 2025))
    am = np.exp(np.random.default_rng(7).normal(np.log(300), 0.4, len(years)))
    c = fs.flow_curve(years, am)
    assert c == fx["curve"]
    assert [round(fs.flow_for(c, t), 6) for t in fx["T"]] == fx["q"]
    assert fs.flow_words(c) == fx["words"]


def test_design_flow_reads_the_reach_curve(monkeypatch):
    years = list(range(1940, 2025))
    am = np.exp(np.random.default_rng(7).normal(np.log(300), 0.4, len(years)))
    monkeypatch.setattr(fs, "annual_maxima", lambda rids, fetch=None: (years, {int(r): am for r in rids if r == 7}))
    d = fs.design_flow(7, 100)
    assert d["flow_cms"] == pytest.approx(fs.flow_for(fs.flow_curve(years, am), 100), abs=1e-3)
    assert d["method"] == "Log-Pearson III" and d["words"] == fs.flow_words(fs.flow_curve(years, am))
    assert d["licence"] == fs.MAXIMA_LICENCE and d["source"] == fs.MAXIMA_ZARR
    with pytest.raises(ValueError):
        fs.design_flow(7, 1.5)
    with pytest.raises(RuntimeError):
        fs.design_flow(8, 100)


WALL_WORDS_N = ("The water reaches the grid's north edge along 120 m. The valley goes on past it, and the edge holds "
                "the water like a wall, so the flood there is deeper and narrower than it would be. Flood a reach "
                "nearer that edge to see it.")


def test_the_wall_is_said_only_when_the_flood_is_held_there():
    """The same grid and rule as explorer/tests/floodsim-core.test.mjs's wallContact test."""
    nx, ny = 12, 10
    e = np.where((np.arange(ny) >= 3) & (np.arange(ny) <= 6), 0.001, 0.0)
    edges = fs.edge_codes(ny, nx, closed=True, edges={"e": e})
    src = [{"cell": (5, 0)}]
    h = np.zeros((ny, nx))
    h[0, 6:10] = 0.4
    w = fs.wall_contact(h, edges, src, near=2)
    assert w["held"] and w["sides"]["n"] == 4 and w["faces"] == 4
    assert fs.wall_words(w, 30.0) == WALL_WORDS_N
    h[:] = 0
    h[0, 6:10] = 0.06
    assert not fs.wall_contact(h, edges, src, near=2)["held"]
    h[:] = 0
    h[2:8, -1] = 1.0
    assert not fs.wall_contact(h, edges, src, near=2)["held"]
    h[:] = 0
    h[3:7, 0] = 1.0
    assert not fs.wall_contact(h, edges, src, near=2)["held"]
    assert fs.wall_words({"held": False}, 30.0) == ""
