"""Flood views of a mesh result (#619, #620): smooth depth, contours, hazard, arrival, largest depth and speed.

The model is the synthetic valley of the Explorer's test models (aquascope_promo/test_models), made here: a 2.4 km
valley of 50 m cells near Linz with a channel, bed levels, and a 24-hour flood wave peaking at 4.8 m at 12:00 that
arrives 5 minutes later per cell downstream. Everything in it is made up.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")
pytest.importorskip("scipy")

from aquascope import mesh_views as V  # noqa: E402, N812
from aquascope.io import mesh as M  # noqa: E402, N812

NX, NY, DX = 48, 28, 50.0
X0, Y0 = 447_000.0, 5_349_000.0
STEPS = 25
CHANNEL_HALF = 2
MID = (NY - 1) / 2.0


def _grid():
    pts = np.array([[X0 + i * DX, Y0 + j * DX] for j in range(NY + 1) for i in range(NX + 1)], dtype="f8")
    pid = lambda i, j: j * (NX + 1) + i  # noqa: E731
    faces = [(pid(i, j), pid(i + 1, j)) for j in range(NY + 1) for i in range(NX)]
    faces += [(pid(i, j), pid(i, j + 1)) for j in range(NY) for i in range(NX + 1)]
    faces = np.array(faces, dtype="i4")
    fidx = {tuple(sorted(f)): k for k, f in enumerate(faces.tolist())}
    cells, centres, rings, ij = [], [], [], []
    for j in range(NY):
        for i in range(NX):
            c = [pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1)]
            cells.append([fidx[tuple(sorted((c[k], c[(k + 1) % 4])))] for k in range(4)])
            centres.append([X0 + (i + 0.5) * DX, Y0 + (j + 0.5) * DX])
            rings.append(c)
            ij.append((i, j))
    return pts, faces, cells, np.array(centres), rings, np.array(ij)


def make_valley(path, *, bed_levels=True, velocity=True, units=b"SI Units"):
    """The synthetic valley (make_synthetic_models.py), with the bed, the velocities and the units optional."""
    from pyproj import CRS

    pts, faces, cells, centres, rings, ij = _grid()
    n = len(cells)
    ab = pts[faces[:, 1]] - pts[faces[:, 0]]
    length = np.hypot(ab[:, 0], ab[:, 1])
    normals = np.stack([ab[:, 1] / length, -ab[:, 0] / length, length], 1)
    i_idx, j_idx = ij[:, 0], ij[:, 1]
    off = np.abs(j_idx - MID)
    slope_drop = 0.0015 * (i_idx * DX)
    bed = 260.0 - slope_drop + np.where(off <= CHANNEL_HALF, 0.0, 2.0 + 0.35 * (off - CHANNEL_HALF))
    bed = bed + 0.15 * np.sin(i_idx / 3.0) * (off > CHANNEL_HALF)
    lag = i_idx * (5.0 / 60.0)
    ws = np.empty((STEPS, n))
    for k in range(STEPS):
        d = 1.2 + 3.6 * np.exp(-((k - lag - 12.0) / 4.0) ** 2)
        ws[k] = np.maximum(260.0 - slope_drop + d, bed)
    depth = ws - bed
    vx = np.where(off <= CHANNEL_HALF, 0.6 + 0.35 * depth, 0.25 * np.sqrt(np.maximum(depth, 0)))
    vx = np.where(depth > 0.05, vx, 0.0)
    vy = -0.04 * np.sign(j_idx - MID) * (depth > 0.05)
    times = [f"15MAR2026 {h:02d}:00:00:000" if h < 24 else "16MAR2026 00:00:00:000" for h in range(STEPS)]
    with h5py.File(path, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Results"
        f.attrs["File Version"] = b"HEC-RAS 6.5 February 2024"
        f.attrs["Units System"] = units
        f.attrs["Projection"] = CRS.from_user_input("EPSG:32633").to_wkt("WKT1_ESRI").encode()
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
        f.create_dataset("Geometry/2D Flow Areas/Attributes", data=np.array([(b"Valley", n)], dtype=dt))
        g = f.create_group("Geometry/2D Flow Areas/Valley")
        g["FacePoints Coordinate"] = pts
        g["Faces FacePoint Indexes"] = faces
        info, vals = [], []
        for c in cells + [[0]]:
            info.append([len(vals), len(c)])
            vals.extend([[fc, 1] for fc in c])
        g["Cells Face and Orientation Info"] = np.array(info, dtype="i4")
        g["Cells Face and Orientation Values"] = np.array(vals, dtype="i4")
        g["Cells Center Coordinate"] = np.vstack([centres, np.zeros((1, 2))])
        g["Cells FacePoint Indexes"] = np.array(rings + [[-1] * 4], dtype="i4")
        g["Faces NormalUnitVector and Length"] = normals.astype("f4")
        if bed_levels:
            g["Cells Minimum Elevation"] = np.concatenate([bed, [0.0]]).astype("f4")
        f.create_group("Plan Data/Plan Information").attrs["Plan Title"] = b"SYNTHETIC valley flood (test data)"
        ts = f.create_group(M.RAS_TS)
        ts["Time Date Stamp (ms)"] = np.array([s.encode() for s in times], dtype="S22")
        area = ts.create_group("2D Flow Areas/Valley")
        area["Water Surface"] = np.hstack([ws, np.zeros((STEPS, 1))]).astype("f4")
        if velocity:
            area[M.RAS_CELL_VX] = np.hstack([vx, np.zeros((STEPS, 1))]).astype("f4")
            area[M.RAS_CELL_VY] = np.hstack([vy, np.zeros((STEPS, 1))]).astype("f4")
        summ = f.create_group(f"{M.RAS_SUMMARY}/2D Flow Areas/Valley")
        wsm = np.concatenate([ws.max(axis=0), [0.0]])
        summ["Maximum Water Surface"] = np.vstack([wsm, np.zeros_like(wsm)]).astype("f4")
    return path


@pytest.fixture(scope="module")
def valley(tmp_path_factory):
    m = M.read_ras_plan(make_valley(tmp_path_factory.mktemp("valley") / "SyntheticValley.p01.hdf"))
    yield m
    m.close()


@pytest.fixture(scope="module")
def lenses(valley):
    return V.mesh_lenses(valley)


@pytest.fixture(scope="module")
def grid(valley):
    return V.field_grid(valley)


def _cell(i, j):
    return j * NX + i


# ── the hazard curves (AIDR Guideline 7-3, Table 2) ─────────────────────────


@pytest.mark.parametrize(("d", "v", "cls"), [
    (0.2, 1.0, 1), (0.3, 1.0, 1),        # H1: D x V <= 0.3, D <= 0.3, V <= 2
    (0.4, 0.5, 2), (0.5, 1.2, 2),        # H2: D x V <= 0.6, D <= 0.5
    (1.0, 0.5, 3), (1.2, 0.5, 3),        # H3: D x V <= 0.6, D <= 1.2
    (1.5, 0.6, 4), (0.1, 2.0, 1), (2.0, 0.5, 4),   # H4: D x V <= 1.0, D <= 2.0
    (0.1, 2.5, 5), (3.0, 1.0, 5), (4.0, 1.0, 5),   # H5: D x V <= 4.0, D <= 4.0, V <= 4.0
    (4.5, 0.1, 6), (0.5, 4.5, 6), (2.5, 2.0, 6),   # H6: beyond any H5 limit
    (0.005, 1.0, 0), (float("nan"), 1.0, 0), (1.0, float("nan"), 0),
])
def test_hazard_class_follows_table_2(d, v, cls):
    assert int(V.hazard_class(np.array([d]), np.array([v]))[0]) == cls


def test_hazard_classes_carry_table_1_words_and_the_source():
    assert [c["code"] for c in V.HAZARD_CLASSES] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert V.HAZARD_CLASSES[3]["words"] == "Unsafe for vehicles and people."
    assert "7-3" in V.HAZARD_SOURCE["name"] and V.HAZARD_SOURCE["url"].startswith("https://knowledge.aidr.org.au/")


# ── the lenses on the synthetic valley ──────────────────────────────────────


def test_largest_depth_is_the_known_peak(valley, lenses):
    assert lenses.meta["available"] == ["depth", "hazard", "max_depth", "max_speed", "arrival"]
    assert float(np.nanmax(lenses.max_depth)) == pytest.approx(4.8, abs=0.01)
    # the peak passes the top of the valley at 12:00 and comes later downstream
    top, bottom = _cell(0, 13), _cell(47, 13)
    assert valley.times[lenses.max_depth_step[top]] == "2026-03-15T12:00:00"
    assert lenses.max_depth_step[bottom] > lenses.max_depth_step[top]


def test_arrival_comes_later_downstream(lenses):
    a = lenses.arrival_h
    assert a[_cell(10, 13)] == 0 and lenses.arrival_step[_cell(10, 13)] == 0   # the channel is wet from the start
    row = np.array([a[_cell(i, 19)] for i in range(NX)])   # a floodplain row, 6 cells off the channel's centre
    assert np.isfinite(row).all()
    assert row[-1] > row[0]
    assert np.all(np.diff(row[::6]) >= 0)    # step by step it never comes earlier downstream
    assert lenses.meta["arrival"]["rule"] == "the first output step with at least 0.1 m of water"
    assert lenses.meta["arrival"]["wet_at_start"] == 4 * NX   # the channel's four rows of cells


def test_hazard_is_classified_at_every_step(valley, lenses):
    ch = _cell(20, 13)
    assert lenses.hazard[ch] == 6            # 4.8 m moving at 2.3 m/s at the peak
    assert lenses.hazard_dv[ch] == pytest.approx(lenses.hazard_depth[ch] * lenses.hazard_speed[ch], rel=1e-5)
    assert lenses.hazard_depth[ch] > 4.0
    k = int(lenses.hazard_step[ch])
    f = valley.frame(k)
    assert V.hazard_class(f["value"][ch:ch + 1], np.hypot(f["u"], f["v"])[ch:ch + 1])[0] == 6
    side = _cell(20, 19)                      # the floodplain gets shallower, slower water
    assert 1 <= lenses.hazard[side] < lenses.hazard[ch]
    assert lenses.hazard[_cell(20, 0)] == 0   # the valley's rim stays dry
    h = lenses.meta["hazard"]
    assert h["from"] == "step" and sum(c["cells"] for c in h["classes"]) == int((lenses.hazard > 0).sum())
    assert sum(c["area_m2"] for c in h["classes"]) == pytest.approx(2500.0 * (lenses.hazard > 0).sum(), rel=0.02)


def test_hazard_falls_back_to_the_largest_values_on_a_long_run(valley):
    big = V.mesh_lenses(valley, budget=10)
    assert big.meta["hazard"]["from"] == "largest"
    assert any("overstate" in n for n in big.meta["notes"])
    assert big.hazard[_cell(20, 13)] == 6


def test_a_file_in_feet_is_classified_in_metres(tmp_path):
    m = M.read_ras_plan(make_valley(tmp_path / "ft.p01.hdf", units=b"US Customary"))
    try:
        assert m.units["depth"] == "ft"
        L = V.mesh_lenses(m)  # noqa: N806
        ch = _cell(20, 13)
        assert L.hazard_depth[ch] == pytest.approx(L.max_depth[ch] * 0.3048, rel=0.01)
        assert L.meta["arrival"]["depth"] == pytest.approx(0.1 / 0.3048, rel=1e-3)
    finally:
        m.close()


def test_without_velocities_or_bed_levels_the_lenses_say_what_is_missing(tmp_path):
    m = M.read_ras_plan(make_valley(tmp_path / "nov.p01.hdf", velocity=False))
    try:
        L = V.mesh_lenses(m)  # noqa: N806
        assert "hazard" not in L.meta["available"] and "arrival" in L.meta["available"]
        assert any("no velocities" in n for n in L.meta["notes"])
    finally:
        m.close()
    m = M.read_ras_plan(make_valley(tmp_path / "nobed.p01.hdf", bed_levels=False))
    try:
        L = V.mesh_lenses(m)  # noqa: N806
        assert L.meta["available"] == ["depth", "max_depth", "max_speed"]
        assert L.hazard is None and L.arrival_h is None
    finally:
        m.close()


# ── the smooth field and the lines ──────────────────────────────────────────


def test_the_grid_covers_the_mesh_and_interpolates_linearly(valley, grid):
    assert grid.width * grid.height <= 1_000_000
    assert grid.pix.size > 0.85 * grid.width * grid.height   # the valley is a rectangle: nearly every pixel is in it
    # a linear field (longitude) comes back as itself inside the centres' hull
    img = V.smooth_field(grid, valley.lon - 14.0, kind="strict")
    rows, cols = np.indices(img.shape)
    lon, _ = grid.lonlat(cols + 0.5, rows + 0.5)
    ok = np.isfinite(img)
    assert ok.sum() > 0.7 * img.size
    inner = np.zeros_like(ok)   # half a cell in from the edge, pixels sit between centres, not beyond them
    inner[img.shape[0] // 10: -img.shape[0] // 10, img.shape[1] // 20: -img.shape[1] // 20] = True
    ok &= inner
    assert np.nanmax(np.abs(img[ok] - (lon[ok] - 14.0))) < 2e-5


def test_smooth_depth_peaks_at_the_peak_and_is_dry_outside(valley, grid):
    img = V.smooth_field(grid, valley.max_value, kind="depth")
    assert np.nanmax(img) == pytest.approx(4.8, abs=0.02)
    assert np.isnan(img[0, 0]) or img[0, 0] >= M.DRY_DEPTH
    early = V.smooth_field(grid, valley.frame(0)["value"], kind="depth")
    assert np.isfinite(early).sum() < np.isfinite(img).sum()   # at 00:00 only the channel holds water


def test_contour_levels_are_round():
    assert V.contour_levels(0, 4.8) == [1.0, 2.0, 3.0, 4.0]
    assert V.contour_levels(0, 0.45) == [0.1, 0.2, 0.3, 0.4]
    assert V.contour_levels(0, 16, most=8, hours=True) == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    assert V.contour_levels(3, 3) == [] and V.contour_levels(0, float("nan")) == []


def test_blur_keeps_holes_where_they_are():
    a = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    b = V.blur(a, 1)
    assert math.isnan(b[0, 2]) and b[0, 0] == pytest.approx(3.0) and b[1, 2] == pytest.approx(13 / 3)


def test_depth_contours_and_the_wet_edge(valley, grid, lenses):
    pytest.importorskip("contourpy")
    out = V.field_payload(valley, grid, lenses, "depth", None)
    fc = json.loads(out["lines"])
    labels = [f["properties"]["label"] for f in fc["features"] if f["properties"]["kind"] == "depth"]
    assert labels == ["1 m", "2 m", "3 m", "4 m"]
    assert any(f["properties"]["kind"] == "edge" for f in fc["features"])
    w, s, e, n = valley.bbox()
    for f in fc["features"]:
        for part in f["geometry"]["coordinates"]:
            xy = np.array(part)
            assert (xy[:, 0] >= w - 1e-3).all() and (xy[:, 0] <= e + 1e-3).all()
            assert (xy[:, 1] >= s - 1e-3).all() and (xy[:, 1] <= n + 1e-3).all()
    assert out["image"].dtype == np.float32 and out["image"].size == grid.width * grid.height


def test_arrival_lines_and_other_pictures(valley, grid, lenses):
    pytest.importorskip("contourpy")
    fc = json.loads(V.field_payload(valley, grid, lenses, "arrival")["lines"])
    # The channel is wet from the start and the floodplain first floods at 8 h: 2, 4 and 6 h would all lie on the
    # bank, on top of each other, so they are not drawn.
    assert [f["properties"]["label"] for f in fc["features"]] == ["8 h", "10 h", "12 h", "14 h"]
    sp = V.field_payload(valley, grid, lenses, "max_speed")["image"]
    assert np.nanmax(sp) == pytest.approx(float(np.nanmax(lenses.max_speed)), rel=0.05)
    step = V.field_payload(valley, grid, lenses, "depth", 0)["image"]
    assert np.isfinite(step).sum() < np.isfinite(V.field_payload(valley, grid, lenses, "max_depth")["image"]).sum()
    with pytest.raises(M.MeshError):
        V.field_payload(valley, grid, lenses, "nonsense")


def test_views_payload_for_the_browser(valley, grid, lenses):
    meta, arrays = V.views_payload(valley, lenses, grid)
    json.dumps(meta)
    assert meta["grid"]["width"] == grid.width and len(meta["grid"]["corners"]) == 4
    assert meta["depth_levels"] == [1.0, 2.0, 3.0, 4.0]
    assert arrays["hazard"].dtype == np.int8 and arrays["arrival_step"].dtype == np.int32
    assert arrays["max_speed"].dtype == np.float32 and arrays["hazard"].size == valley.n_cells


# ── the faces: CLI and MCP ──────────────────────────────────────────────────


def test_lens_summary_cli_and_mcp(tmp_path, capsys, monkeypatch):
    path = make_valley(tmp_path / "v.p01.hdf")
    res = V.lens_summary(str(path))
    assert res["ok"] and res["max_depth"]["value"] == pytest.approx(4.8, abs=0.01)
    assert [c["code"] for c in res["hazard"]["classes"]][-1] == "H6"
    assert res["arrival"]["wet_at_start"] == 4 * NX
    assert V.lens_summary(str(tmp_path / "missing.p01.hdf"))["ok"] is False

    from aquascope import mcp_server

    assert mcp_server.mesh_lenses(str(path))["hazard"]["source"].startswith("AIDR Guideline 7-3")

    from aquascope import cli

    monkeypatch.setattr("sys.argv", ["aquascope", "mesh", "lenses", str(path)])
    cli.main()
    out = capsys.readouterr().out
    assert "Flood hazard" in out and "H6" in out and "Largest depth: 4.8 m" in out


def test_the_module_is_experimental():
    from aquascope import experimental

    assert experimental.is_experimental("aquascope.mesh_views")
    assert "mesh_lenses" in experimental.MCP_TOOLS
