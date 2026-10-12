"""Model results on a mesh (#555): the HEC-RAS plan and UGRID readers, on small synthetic files made here."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")

from aquascope.io import mesh as M  # noqa: E402, N812

# A 3 x 2 grid of 100 m cells in UTM zone 33N, on the zone's central meridian (15 E), so grid north is true north.
UTM = "EPSG:32633"
X0, Y0, DX = 500_000.0, 5_000_000.0, 100.0
NX, NY = 3, 2
TIMES = ["01JAN2020 00:00:00:000", "01JAN2020 01:00:00:000", "01JAN2020 02:00:00:000"]
FLOW = (1.0, 0.5)   # a uniform current, m/s east and north


def _grid():
    """Face points, faces (A, B), cells (their faces) and centres of the grid."""
    pts = np.array([[X0 + i * DX, Y0 + j * DX] for j in range(NY + 1) for i in range(NX + 1)], dtype="f8")
    pid = lambda i, j: j * (NX + 1) + i  # noqa: E731
    faces = []
    for j in range(NY + 1):
        for i in range(NX):
            faces.append((pid(i, j), pid(i + 1, j)))       # horizontal
    for j in range(NY):
        for i in range(NX + 1):
            faces.append((pid(i, j), pid(i, j + 1)))       # vertical
    faces = np.array(faces, dtype="i4")
    fidx = {tuple(sorted(f)): k for k, f in enumerate(faces.tolist())}
    cells, centres, rings = [], [], []
    for j in range(NY):
        for i in range(NX):
            corners = [pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1)]
            cells.append([fidx[tuple(sorted((corners[k], corners[(k + 1) % 4])))] for k in range(4)])
            centres.append([X0 + (i + 0.5) * DX, Y0 + (j + 0.5) * DX])
            rings.append(corners)
    return pts, faces, cells, np.array(centres), rings


def _normals(pts, faces):
    ab = pts[faces[:, 1]] - pts[faces[:, 0]]
    length = np.hypot(ab[:, 0], ab[:, 1])
    return np.stack([ab[:, 1] / length, -ab[:, 0] / length, length], 1)


def make_ras(path, *, bed=True, normals=True, ordered=True, projection=True, series=True, ghost=1,
             cell_velocity=None, face_velocity=True):
    pts, faces, cells, centres, rings = _grid()
    n = len(cells)
    nl = _normals(pts, faces)
    with h5py.File(path, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Results"
        f.attrs["File Version"] = b"HEC-RAS 6.5 February 2024"
        f.attrs["Units System"] = b"SI Units"
        if projection:
            from pyproj import CRS

            f.attrs["Projection"] = CRS.from_user_input(UTM).to_wkt("WKT1_ESRI").encode()
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
        f.create_dataset("Geometry/2D Flow Areas/Attributes", data=np.array([(b"Grid", n)], dtype=dt))
        g = f.create_group("Geometry/2D Flow Areas/Grid")
        g["FacePoints Coordinate"] = pts
        g["Faces FacePoint Indexes"] = faces
        info, vals = [], []
        for c in cells + [[0]] * ghost:     # ghost cells after the real ones, as RAS writes them
            info.append([len(vals), len(c)])
            vals.extend([[fc, 1] for fc in c])
        g["Cells Face and Orientation Info"] = np.array(info, dtype="i4")
        g["Cells Face and Orientation Values"] = np.array(vals, dtype="i4")
        g["Cells Center Coordinate"] = np.vstack([centres, np.zeros((ghost, 2))])
        if ordered:
            g["Cells FacePoint Indexes"] = np.array(rings + [[-1] * 4] * ghost, dtype="i4")
        if normals:
            g["Faces NormalUnitVector and Length"] = nl.astype("f4")
        if bed:
            g["Cells Minimum Elevation"] = np.array([10.0, 10.5, 11.0, 12.0, 12.5, 30.0] + [0.0] * ghost, dtype="f4")
        pinfo = f.create_group("Plan Data/Plan Information")
        pinfo.attrs["Plan Title"] = b"Synthetic run"
        pinfo.attrs["Project Title"] = b"Test grid"
        if series:
            ts = f.create_group(M.RAS_TS)
            ts["Time Date Stamp (ms)"] = np.array([t.encode() for t in TIMES], dtype="S22")
            ws = np.array([[12.0, 12.0, 12.0, 12.0, 12.0, 30.0],
                           [13.0, 13.0, 13.0, 13.0, 13.0, 30.0],
                           [12.5, 12.5, 12.5, 12.5, 12.5, 30.0]], dtype="f4")
            ws = np.hstack([ws, np.zeros((3, ghost), "f4")])
            area = ts.create_group("2D Flow Areas/Grid")
            area["Water Surface"] = ws
            vn = (nl[:, :2] @ np.array(FLOW)).astype("f4")
            if face_velocity:
                area["Face Velocity"] = np.vstack([vn, 2 * vn, vn])
            if cell_velocity is not None:   # RAS's own cell velocity (grid x, y), as 6.x writes it when asked
                k = np.array([1.0, 2.0, 1.0])[:, None]
                area["Cell Velocity - Velocity X"] = np.hstack([k * cell_velocity[0] * np.ones((1, n)),
                                                               np.zeros((3, ghost))]).astype("f4")
                area["Cell Velocity - Velocity Y"] = np.hstack([k * cell_velocity[1] * np.ones((1, n)),
                                                               np.zeros((3, ghost))]).astype("f4")
            summ = f.create_group(f"{M.RAS_SUMMARY}/2D Flow Areas/Grid")
            summ["Maximum Water Surface"] = np.vstack([ws.max(axis=0), np.zeros(ws.shape[1])]).astype("f4")
    return path


def test_ras_plan_reads_cells_times_depth_and_place(tmp_path):
    m = M.read_ras_plan(make_ras(tmp_path / "run.p01.hdf"))
    assert m.format == "hec-ras" and m.n_cells == 6
    assert m.areas == [{"name": "Grid", "cells": 6, "start": 0}]
    assert m.times == ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T02:00:00"]
    assert m.variable == "depth" and m.units == {"depth": "m", "water_surface": "m", "velocity": "m/s"}
    assert "EPSG:32633" in m.crs["label"] and m.crs["from"] == "file"
    assert m.title == "Synthetic run (Test grid)"
    # max depth = max water surface - bed; the last cell never gets wet
    np.testing.assert_allclose(m.max_value[:5], [3.0, 2.5, 2.0, 1.0, 0.5])
    assert math.isnan(m.max_value[5])
    # the grid sits on 15 E, about 45.1 N
    assert m.lon.min() > 14.99 and m.lon.max() < 15.01 and 45.0 < m.lat.mean() < 45.2
    assert list(m.ring_offsets) == [0, 4, 8, 12, 16, 20, 24]
    np.testing.assert_allclose(m.cell_areas(), 10_000, rtol=2e-3)   # 100 m squares (UTM scale is 0.9996)
    m.close()


def test_ras_cell_velocity_is_rebuilt_from_face_velocities(tmp_path):
    m = M.read_ras_plan(make_ras(tmp_path / "run.p01.hdf"))
    fr = m.frame(1)
    np.testing.assert_allclose(fr["u"][:5], 2 * FLOW[0], atol=1e-3)
    np.testing.assert_allclose(fr["v"][:5], 2 * FLOW[1], atol=1e-3)
    assert math.isnan(fr["u"][5])   # dry: no arrow
    np.testing.assert_allclose(m.max_speed[:5], 2 * math.hypot(*FLOW), atol=1e-3)
    np.testing.assert_allclose(fr["value"][:5], [3.0, 2.5, 2.0, 1.0, 0.5])
    s = m.series(0)
    np.testing.assert_allclose(s["value"], [2.0, 3.0, 2.5])
    np.testing.assert_allclose(s["speed"], np.array([1, 2, 1]) * np.hypot(*FLOW), atol=1e-3)
    m.close()


@pytest.mark.parametrize("faces", [True, False])
def test_ras_uses_the_models_own_cell_velocity_where_written(tmp_path, faces):
    # A real 6.5 plan (docs/assessments/hec-ras.md) writes Cell Velocity X/Y; the reader takes them over the
    # least-squares rebuild from faces (here they differ on purpose), and a run with no face velocity still has arrows.
    own = (-0.3, 0.8)
    m = M.read_ras_plan(make_ras(tmp_path / "run.p01.hdf", cell_velocity=own, face_velocity=faces))
    assert m.has_velocity
    fr = m.frame(1)
    np.testing.assert_allclose(fr["u"][:5], 2 * own[0], atol=1e-3)
    np.testing.assert_allclose(fr["v"][:5], 2 * own[1], atol=1e-3)
    assert math.isnan(fr["u"][5])   # dry: no arrow
    np.testing.assert_allclose(m.max_speed[:5], 2 * math.hypot(*own), atol=1e-3)
    np.testing.assert_allclose(m.series(0)["speed"], np.array([1, 2, 1]) * math.hypot(*own), atol=1e-3)
    m.close()


def test_ras_largest_own_cell_velocity_leaves_out_dry_steps(tmp_path):
    # RAS writes a small velocity in dry cells (up to 0.24 m/s on the real 6.5 plan); cell 4 is dry at step 0,
    # so a speed written there must not become its largest, as the frame and the series leave it out too.
    own = (-0.3, 0.8)
    path = make_ras(tmp_path / "run.p01.hdf", cell_velocity=own)
    with h5py.File(path, "r+") as f:
        f[f"{M.RAS_TS}/2D Flow Areas/Grid/Cell Velocity - Velocity X"][0, 4] = 10.0
    m = M.read_ras_plan(path)
    np.testing.assert_allclose(m.max_speed[4], 2 * math.hypot(*own), atol=1e-3)
    m.close()


def test_ras_without_normals_or_ordered_points_gives_the_same_mesh(tmp_path):
    full = M.read_ras_plan(make_ras(tmp_path / "a.hdf"))
    bare = M.read_ras_plan(make_ras(tmp_path / "b.hdf", normals=False, ordered=False))
    for i in range(6):
        a = {(round(x, 7), round(y, 7)) for x, y in zip(full.ring_lon[full.ring_offsets[i]:full.ring_offsets[i + 1]],
                                                          full.ring_lat[full.ring_offsets[i]:full.ring_offsets[i + 1]])}
        b = {(round(x, 7), round(y, 7)) for x, y in zip(bare.ring_lon[bare.ring_offsets[i]:bare.ring_offsets[i + 1]],
                                                          bare.ring_lat[bare.ring_offsets[i]:bare.ring_offsets[i + 1]])}
        assert a == b
    np.testing.assert_allclose(bare.frame(1)["u"][:5], 2 * FLOW[0], atol=1e-3)
    full.close()
    bare.close()


def test_ras_without_bed_levels_shows_the_water_surface_and_says_so(tmp_path):
    m = M.read_ras_plan(make_ras(tmp_path / "run.hdf", bed=False))
    assert m.variable == "water_surface"
    assert any("so no depth" in n for n in m.notes)
    np.testing.assert_allclose(m.max_value, [13, 13, 13, 13, 13, 30])
    m.close()


def test_ras_needs_a_coordinate_system(tmp_path):
    path = make_ras(tmp_path / "run.hdf", projection=False)
    with pytest.raises(M.MeshError, match="EPSG"):
        M.read_ras_plan(path)
    m = M.read_ras_plan(path, crs=32633)
    assert m.crs["from"] == "given" and "EPSG:32633" in m.crs["label"]
    m.close()


def test_ras_limits_and_wrong_files(tmp_path, monkeypatch):
    path = make_ras(tmp_path / "run.hdf")
    with pytest.raises(M.MeshError, match="draws up to 3"):
        M.read_ras_plan(path, max_cells=3)
    monkeypatch.setattr(M, "MAX_FILE_BYTES", 100)
    with pytest.raises(M.MeshError, match="reads files up to"):
        M.read_mesh(path)
    monkeypatch.undo()
    other = tmp_path / "notes.txt"
    other.write_text("not a model")
    with pytest.raises(M.MeshError, match="neither"):
        M.read_mesh(other)
    geom = tmp_path / "g.hdf"
    with h5py.File(geom, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Geometry"
    with pytest.raises(M.MeshError, match="geometry file"):
        M.read_ras_plan(geom)
    empty = tmp_path / "e.hdf"
    with h5py.File(empty, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Results"
    with pytest.raises(M.MeshError, match="no 2D flow areas"):
        M.read_ras_plan(empty)


def test_ras_file_with_maxima_only(tmp_path):
    m = M.read_ras_plan(make_ras(tmp_path / "run.hdf", series=False))
    assert m.times == [] and not m.has_velocity
    assert any("only the maxima" in n for n in m.notes) or np.isnan(m.max_value).all()
    with pytest.raises(M.MeshError, match="no time steps"):
        m.frame(0)
    m.close()


@pytest.mark.parametrize("text, iso", [
    ("01JAN1999 12:00:00:000", "1999-01-01T12:00:00"),
    ("31Dec2000 24:00", "2001-01-01T00:00:00"),
    (b"15mar2021 06:30:15", "2021-03-15T06:30:15"),
])
def test_parse_ras_time(text, iso):
    assert M._iso(M.parse_ras_time(text)) == iso


# ── UGRID ───────────────────────────────────────────────────────────────────

def _ugrid_arrays():
    # 2 x 2 quads over 3 x 3 nodes
    xs, ys = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    nx_, ny_ = xs.ravel(), ys.ravel()
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype="i4")
    t = np.array([0.0, 3600.0, 7200.0])
    depth = np.array([[0.5, 1.0, 0.0, 2.0], [1.5, 2.0, 0.0, 3.0], [1.0, 1.5, 0.0, 2.5]])
    ucx = np.full((3, 4), 0.3)
    ucy = np.full((3, 4), -0.4)
    return nx_, ny_, faces, t, depth, ucx, ucy


def make_ugrid_nc3(path, *, degrees=True, node_depth=False, level=False):
    from scipy.io import netcdf_file

    nx_, ny_, faces, t, depth, ucx, ucy = _ugrid_arrays()
    if degrees:
        nx_, ny_ = 4.0 + nx_ * 0.01, 52.0 + ny_ * 0.01
    with netcdf_file(path, "w") as f:
        f.title = b"Synthetic estuary"
        f.source = b"Synthetic D-Flow FM"
        f.createDimension("time", 3)
        f.createDimension("nmesh2d_node", nx_.size)
        f.createDimension("nmesh2d_face", 4)
        f.createDimension("max_nodes", 4)
        topo = f.createVariable("mesh2d", "i", ())
        topo.cf_role = b"mesh_topology"
        topo.topology_dimension = 2
        topo.node_coordinates = b"mesh2d_node_x mesh2d_node_y"
        topo.face_node_connectivity = b"mesh2d_face_nodes"
        x = f.createVariable("mesh2d_node_x", "d", ("nmesh2d_node",))
        y = f.createVariable("mesh2d_node_y", "d", ("nmesh2d_node",))
        x[:], y[:] = nx_, ny_
        if degrees:
            x.units, y.units = b"degrees_east", b"degrees_north"
        else:
            crs = f.createVariable("projected_coordinate_system", "i", ())
            crs.epsg = 28992
            crs.grid_mapping_name = b"Unknown projected"
        c = f.createVariable("mesh2d_face_nodes", "i", ("nmesh2d_face", "max_nodes"))
        c[:] = faces + 1
        c.start_index = 1
        c._FillValue = -999
        tv = f.createVariable("time", "d", ("time",))
        tv[:] = t
        tv.units = b"seconds since 2020-06-01 00:00:00 +00:00"
        if level:
            s1 = f.createVariable("mesh2d_s1", "d", ("time", "nmesh2d_face"))
            s1[:] = depth - 5.0
            s1.mesh = b"mesh2d"
            s1.units = b"m"
            bl = f.createVariable("mesh2d_flowelem_bl", "d", ("nmesh2d_face",))
            bl[:] = -5.0
            bl.mesh = b"mesh2d"
        elif node_depth:
            d = f.createVariable("depth", "d", ("time", "nmesh2d_node"))
            d[:] = np.tile(np.arange(9.0), (3, 1)) * np.array([[1.0], [2.0], [1.0]])
        else:
            d = f.createVariable("mesh2d_waterdepth", "d", ("time", "nmesh2d_face"))
            d[:] = depth
            d.mesh = b"mesh2d"
            d.location = b"face"
            d.units = b"m"
            d.standard_name = b"sea_floor_depth_below_sea_surface"
        u = f.createVariable("mesh2d_ucx", "d", ("time", "nmesh2d_face"))
        v = f.createVariable("mesh2d_ucy", "d", ("time", "nmesh2d_face"))
        u[:], v[:] = ucx, ucy
        u.units = v.units = b"m s-1"
        u.mesh = v.mesh = b"mesh2d"
    return path


def test_ugrid_netcdf3_in_degrees(tmp_path):
    m = M.read_ugrid(make_ugrid_nc3(tmp_path / "map.nc"))
    assert m.format == "ugrid" and m.n_cells == 4 and m.crs["geographic"]
    assert m.model == "Synthetic D-Flow FM" and m.title == "Synthetic estuary"
    assert m.times == ["2020-06-01T00:00:00", "2020-06-01T01:00:00", "2020-06-01T02:00:00"]
    assert m.units["depth"] == "m" and m.units["velocity"] == "m s-1"
    np.testing.assert_allclose(m.max_value[[0, 1, 3]], [1.5, 2.0, 3.0])
    assert math.isnan(m.max_value[2])        # never wet
    np.testing.assert_allclose(m.lon, [4.005, 4.015, 4.005, 4.015])
    fr = m.frame(1)
    np.testing.assert_allclose(fr["u"][[0, 1, 3]], 0.3)   # degrees: already east and north
    assert math.isnan(fr["v"][2])
    s = m.series(3)
    np.testing.assert_allclose(s["value"], [2.0, 3.0, 2.5])
    np.testing.assert_allclose(s["speed"], 0.5)
    m.close()


def test_ugrid_node_values_are_averaged_onto_faces(tmp_path):
    m = M.read_ugrid(make_ugrid_nc3(tmp_path / "map.nc", node_depth=True))
    # face 0 has nodes 0, 1, 4, 3: (0 + 1 + 4 + 3) / 4 = 2 at step 0, 4 at step 1
    np.testing.assert_allclose(m.frame(0)["value"], [2.0, 3.0, 5.0, 6.0])
    np.testing.assert_allclose(m.max_value, [4.0, 6.0, 10.0, 12.0])
    np.testing.assert_allclose(m.series(0)["value"], [2.0, 4.0, 2.0])
    m.close()


def test_ugrid_water_level_less_bed_is_depth(tmp_path):
    m = M.read_ugrid(make_ugrid_nc3(tmp_path / "map.nc", level=True))
    assert m.variable == "depth"
    np.testing.assert_allclose(m.max_value[[0, 1, 3]], [1.5, 2.0, 3.0])
    m.close()


def test_ugrid_projected_epsg_code_and_rotation(tmp_path):
    path = make_ugrid_nc3(tmp_path / "map.nc", degrees=False)
    # RD New coordinates near Amersfoort: nodes 0..2 m from (155000, 463000)
    from scipy.io import netcdf_file

    with netcdf_file(path, "a") as f:
        f.variables["mesh2d_node_x"][:] = 155_000 + f.variables["mesh2d_node_x"][:] * 100
        f.variables["mesh2d_node_y"][:] = 463_000 + f.variables["mesh2d_node_y"][:] * 100
    m = M.read_ugrid(path)
    assert "28992" in m.crs["label"] and not m.crs["geographic"]
    assert 5.3 < m.lon.mean() < 5.5 and 52.1 < m.lat.mean() < 52.2
    fr = m.frame(0)
    # RD grid north is close to true north at Amersfoort (its origin): the current keeps its speed
    np.testing.assert_allclose(np.hypot(fr["u"][0], fr["v"][0]), 0.5, atol=1e-6)
    m.close()


def test_ugrid_netcdf4(tmp_path):
    nc4 = pytest.importorskip("netCDF4")
    nx_, ny_, faces, t, depth, ucx, ucy = _ugrid_arrays()
    path = tmp_path / "map4.nc"
    with nc4.Dataset(path, "w") as f:
        f.createDimension("time", None)
        f.createDimension("node", 9)
        f.createDimension("face", 4)
        f.createDimension("nmax", 4)
        topo = f.createVariable("mesh", "i4")
        topo.cf_role = "mesh_topology"
        topo.topology_dimension = 2
        topo.node_coordinates = "node_x node_y"
        topo.face_node_connectivity = "face_nodes"
        x = f.createVariable("node_x", "f8", ("node",))
        y = f.createVariable("node_y", "f8", ("node",))
        x[:], y[:] = 500_000 + nx_ * 50, 5_000_000 + ny_ * 50
        crs = f.createVariable("crs", "i4")
        from pyproj import CRS

        crs.crs_wkt = CRS.from_user_input(UTM).to_wkt()
        c = f.createVariable("face_nodes", "i4", ("face", "nmax"), fill_value=-1)
        c[:, :] = faces
        c.start_index = 0
        tv = f.createVariable("time", "f8", ("time",))
        tv.units = "hours since 2021-02-03"
        tv[:] = [0, 1, 2]
        d = f.createVariable("water_depth", "f4", ("time", "face"), fill_value=-999.0)
        d[:, :] = depth
        d.units = "m"
    m = M.read_ugrid(path)
    assert "32633" in m.crs["label"]
    assert m.times[-1] == "2021-02-03T02:00:00"
    np.testing.assert_allclose(m.max_value[[0, 1, 3]], [1.5, 2.0, 3.0])
    assert not m.has_velocity
    m.close()


def test_ugrid_errors(tmp_path):
    path = make_ugrid_nc3(tmp_path / "map.nc", degrees=False)
    from scipy.io import netcdf_file

    with netcdf_file(path, "a") as f:   # take the EPSG code away
        f.variables["projected_coordinate_system"].epsg = 0
    with pytest.raises(M.MeshError, match="coordinate system"):
        M.read_ugrid(path)
    plain = tmp_path / "plain.nc"
    with netcdf_file(plain, "w") as f:
        f.createDimension("x", 2)
        f.createVariable("x", "d", ("x",))[:] = [1, 2]
    with pytest.raises(M.MeshError, match="no UGRID 2D mesh"):
        M.read_mesh(plain)


# ── the faces: summary, payloads, MCP, CLI ─────────────────────────────────

def test_mesh_summary_numbers_come_from_the_file(tmp_path):
    s = M.mesh_summary(make_ras(tmp_path / "run.hdf"))
    assert s["ok"] and s["format"] == "hec-ras" and s["cells"] == 6 and s["time_steps"] == 3
    assert s["max"]["value"] == 3.0 and s["max"]["cell"] == 0 and s["max"]["unit"] == "m"
    assert s["wet_cells"] == 5
    assert s["max_speed"]["value"] == round(2 * math.hypot(*FLOW), 2)
    assert "wkt" not in s["crs"]
    json.dumps(s)
    bad = M.mesh_summary(tmp_path / "missing.hdf")
    assert bad == {"ok": False, "error": f"No such file: {tmp_path / 'missing.hdf'}"}


def test_browser_payloads(tmp_path):
    m = M.read_ras_plan(make_ras(tmp_path / "run.hdf"))
    meta, arrays = M.browser_payload(m)
    json.dumps(meta)
    assert meta["value_range"] == [0.5, 3.0] and meta["peak_step"] == 1
    assert meta["speed_max"] == pytest.approx(2 * math.hypot(*FLOW), abs=1e-3)
    assert arrays["ring_offsets"].dtype == np.int32 and arrays["max_value"].dtype == np.float32
    assert arrays["lon"].size == 6 and arrays["ring_lon"].size == 24
    fr = M.frame_payload(m, 2)
    assert set(fr) == {"value", "u", "v"} and fr["value"].dtype == np.float32
    s = M.series_payload(m, 5)
    assert s["value"] == [None, None, None] and s["variable"] == "depth"
    json.dumps(s)
    m.close()


def test_mcp_tool(tmp_path):
    pytest.importorskip("mcp")
    from aquascope import mcp_server

    s = mcp_server.mesh_summary(str(make_ras(tmp_path / "run.hdf")))
    assert s["ok"] and s["cells"] == 6


def test_cli_mesh(tmp_path, capsys, monkeypatch):
    import sys

    from aquascope import cli

    path = make_ras(tmp_path / "run.hdf", projection=False)
    monkeypatch.setattr(sys, "argv", ["aquascope", "mesh", str(path), "--crs", "32633"])
    cli.main()
    out = capsys.readouterr().out
    assert "6 cells" in out and "Largest depth: 3.0 m" in out and "3 time steps" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "mesh", str(path)])
    with pytest.raises(SystemExit):
        cli.main()
    assert "--crs" in capsys.readouterr().out


def test_importing_the_mesh_reader_stays_light():
    import subprocess
    import sys

    code = "import sys, aquascope.io.mesh; print('pandas' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# ── a stripped plan file: the results without the mesh (#555 follow-up) ─────


def strip_geometry(full, path, *, keep_attrs=False, drop=M.RAS_GEOM.split("/")[0]):
    """A copy of a plan file without its geometry (as a *.p01.timeseries.hdf holds it): ``drop`` and, unless
    ``keep_attrs``, the file's own attributes (RAS Mapper's projection lives with the geometry)."""
    with h5py.File(full, "r") as src, h5py.File(path, "w") as dst:
        for key in src:
            src.copy(src[key], dst, name=key)
        if keep_attrs:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
        del dst[drop]
    return path


def make_geometry(full, path, *, rename=None):
    """A HEC-RAS geometry file (*.g01.hdf) of the same mesh: the plan's Geometry group and projection."""
    with h5py.File(full, "r") as src, h5py.File(path, "w") as dst:
        src.copy(src["Geometry"], dst, name="Geometry")
        dst.attrs["File Type"] = b"HEC-RAS Geometry"
        dst.attrs["Projection"] = src.attrs["Projection"]
        if rename:
            dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
            count = int(dst[f"{M.RAS_GEOM}/Attributes"][0]["Cell Count"])
            del dst[f"{M.RAS_GEOM}/Attributes"]
            dst.create_dataset(f"{M.RAS_GEOM}/Attributes", data=np.array([(rename.encode(), count)], dtype=dt))
            dst.move(f"{M.RAS_GEOM}/Grid", f"{M.RAS_GEOM}/{rename}")
    return path


def test_a_plan_without_its_geometry_says_what_is_missing_and_how_to_fix_it(tmp_path):
    stripped = strip_geometry(make_ras(tmp_path / "run.p01.hdf"), tmp_path / "run.p01.timeseries.hdf")
    for read in (M.read_ras_plan, M.read_mesh):   # read_mesh knows it for a RAS file by its results alone
        with pytest.raises(M.MeshError) as err:
            read(stripped)
        msg = str(err.value)
        assert msg.startswith("run.p01.timeseries.hdf holds model results but not the mesh they sit on")
        assert "Geometry/2D Flow Areas is missing" in msg
        assert "full plan file (.p01.hdf)" in msg and "geometry file the plan was run with (.g01.hdf)" in msg
        assert "\n" not in msg
    assert M.mesh_summary(stripped) == {"ok": False, "error": msg}


def test_a_plan_with_results_but_no_2d_areas_is_called_a_hec_ras_file(tmp_path):
    # A plan holding only 1D results or boundary lines (rashdf's LowerKanawha.p01.bclines.hdf): a HEC-RAS file
    # with no 2D flow areas, not "no UGRID mesh".
    path = tmp_path / "lines.p01.hdf"
    with h5py.File(path, "w") as fh:
        fh.create_group("Geometry/Boundary Condition Lines")
        fh.create_group("Results/Unsteady/Output")
    with pytest.raises(M.MeshError, match="has no 2D flow areas") as err:
        M.read_mesh(path)
    assert "UGRID" not in str(err.value)


@pytest.mark.parametrize("drop, missing", [
    (f"{M.RAS_GEOM}/Grid", f"{M.RAS_GEOM}/Grid is missing"),
    (f"{M.RAS_GEOM}/Grid/FacePoints Coordinate", "Grid/FacePoints Coordinate is missing"),
    (f"{M.RAS_GEOM}/Grid/Cells Face and Orientation Info", "Grid/Cells Face and Orientation Info is missing"),
])
def test_a_plan_with_part_of_its_geometry_names_the_part(tmp_path, drop, missing):
    stripped = strip_geometry(make_ras(tmp_path / "run.p01.hdf"), tmp_path / "part.p01.hdf", keep_attrs=True,
                              drop=drop)
    with pytest.raises(M.MeshError, match="not the mesh they sit on") as err:
        M.read_ras_plan(stripped)
    assert missing in str(err.value)


def test_a_stripped_plan_reads_with_its_geometry_file(tmp_path):
    full = make_ras(tmp_path / "run.p01.hdf")
    stripped = strip_geometry(full, tmp_path / "run.p01.timeseries.hdf")
    geom = make_geometry(full, tmp_path / "run.g01.hdf")
    a, b = M.read_ras_plan(full), M.read_mesh(stripped, geometry=geom)
    try:
        assert b.n_cells == a.n_cells == 6 and b.times == a.times
        assert b.crs["from"] == "file" and "EPSG:32633" in b.crs["label"]   # the projection from the geometry file
        np.testing.assert_allclose(b.lon, a.lon)
        np.testing.assert_allclose(b.max_value, a.max_value)
        np.testing.assert_allclose(b.max_speed, a.max_speed)
        np.testing.assert_allclose(b.frame(1)["value"], a.frame(1)["value"])
        assert b.title == a.title   # the plan's own data, not the geometry's
    finally:
        a.close()
        b.close()
    # A full plan with a geometry file given anyway reads its own mesh.
    c = M.read_ras_plan(full, geometry=geom)
    assert c.n_cells == 6
    c.close()
    s = M.mesh_summary(stripped, geometry=geom)
    assert s["ok"] and s["cells"] == 6 and s["max"]["value"] == 3.0
    assert not any("summary output" in n for n in s["notes"])   # this one kept RAS's summary maxima
    bare = strip_geometry(full, tmp_path / "bare.p01.hdf", keep_attrs=True, drop=M.RAS_SUMMARY)
    s = M.mesh_summary(bare)
    assert s["max"]["value"] == 3.0 and any("over the saved time steps" in n for n in s["notes"])


def test_a_geometry_file_that_does_not_fit_is_refused(tmp_path):
    full = make_ras(tmp_path / "run.p01.hdf")
    stripped = strip_geometry(full, tmp_path / "run.p01.timeseries.hdf")
    other = make_geometry(full, tmp_path / "other.g01.hdf", rename="Elsewhere")
    with pytest.raises(M.MeshError, match="'Elsewhere' that the plan has no results for"):
        M.read_ras_plan(stripped, geometry=other)
    bigger = make_geometry(full, tmp_path / "big.g01.hdf")
    with h5py.File(bigger, "a") as f:
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
        del f[f"{M.RAS_GEOM}/Attributes"]
        f.create_dataset(f"{M.RAS_GEOM}/Attributes", data=np.array([(b"Grid", 99)], dtype=dt))
    with pytest.raises(M.MeshError, match="99 cells, more than the plan's results hold"):
        M.read_ras_plan(stripped, geometry=bigger)
    with pytest.raises(M.MeshError, match="no 2D mesh to give the plan"):
        M.read_ras_plan(stripped, geometry=stripped)
    text = tmp_path / "run.g01.hdf"
    text.write_text("not hdf5")
    with pytest.raises(M.MeshError, match="not an HDF5 file"):
        M.read_ras_plan(stripped, geometry=text)


def test_cli_mesh_and_review_say_it_without_a_traceback(tmp_path, capsys, monkeypatch):
    import sys

    from aquascope import cli

    full = make_ras(tmp_path / "run.p01.hdf")
    stripped = strip_geometry(full, tmp_path / "run.p01.timeseries.hdf")
    geom = make_geometry(full, tmp_path / "run.g01.hdf")
    for argv in (["mesh", str(stripped)], ["mesh", "review", str(stripped), "--offline"]):
        monkeypatch.setattr(sys, "argv", ["aquascope", *argv])
        with pytest.raises(SystemExit) as ex:
            cli.main()
        assert ex.value.code == 1
        out, errout = capsys.readouterr()
        assert "holds model results but not the mesh they sit on" in out and "--geometry" in out
        assert "Traceback" not in out + errout and "KeyError" not in out + errout
    monkeypatch.setattr(sys, "argv", ["aquascope", "mesh", str(stripped), "--geometry", str(geom)])
    cli.main()
    assert "6 cells" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["aquascope", "mesh", "review", str(stripped), "--offline", "--geometry",
                                      str(geom)])
    cli.main()
    assert "run.p01.timeseries.hdf: 6 cells, 3 time steps" in capsys.readouterr().out


def test_the_explorer_knows_the_words_of_the_no_geometry_error(tmp_path):
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "explorer" / "src" / "mesh-core.js").read_text(encoding="utf-8")
    phrase = re.search(r'export const NO_GEOMETRY = "([^"]+)"', js).group(1)
    stripped = strip_geometry(make_ras(tmp_path / "run.p01.hdf"), tmp_path / "run.p01.timeseries.hdf")
    err = M.mesh_summary(stripped)["error"]
    assert phrase in err and re.search(r": (.+?) is missing\.", err)
    worker = (Path(__file__).resolve().parents[2] / "explorer" / "src" / "mesh-worker.js").read_text(encoding="utf-8")
    assert "geometry=_geom or None" in worker
