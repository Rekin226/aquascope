"""Groundwater surface and MODFLOW heads on the map (#558). Every fixture here is small and synthetic."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from aquascope.groundwater import surface as gs

# ── synthetic MODFLOW files ─────────────────────────────────────────────────


def hds_bytes(frames, *, double=False, text="HEAD"):
    """frames: [(kstp, kper, totim, layer, 2-D array)] -> a MODFLOW binary head file."""
    f = "d" if double else "f"
    out = b""
    for kstp, kper, totim, lay, arr in frames:
        arr = np.asarray(arr, dtype=float)
        nrow, ncol = arr.shape
        out += struct.pack(f"<ii{f}{f}", kstp, kper, totim, totim) + f"{text:>16}".encode()
        out += struct.pack("<iii", ncol, nrow, lay) + arr.astype("<f8" if double else "<f4").tobytes()
    return out


def grb_bytes(nrow, ncol, delr, delc, xorigin=0.0, yorigin=0.0, angrot=0.0):
    defs = [("NCELLS", "INTEGER", []), ("NLAY", "INTEGER", []), ("NROW", "INTEGER", []), ("NCOL", "INTEGER", []),
            ("XORIGIN", "DOUBLE", []), ("YORIGIN", "DOUBLE", []), ("ANGROT", "DOUBLE", []),
            ("DELR", "DOUBLE", [ncol]), ("DELC", "DOUBLE", [nrow])]
    head = b"".join(f"{t:<49}\n".encode() for t in ("GRID DIS", "VERSION 1", f"NTXT {len(defs)}", "LENTXT 100"))
    for name, typ, dims in defs:
        head += f"{name} {typ} NDIM {len(dims)} {' '.join(map(str, dims))}".ljust(99).encode() + b"\n"
    data = struct.pack("<iiii", nrow * ncol, 1, nrow, ncol) + struct.pack("<ddd", xorigin, yorigin, angrot)
    data += np.asarray(delr, "<f8").tobytes() + np.asarray(delc, "<f8").tobytes()
    return head + data


MF6_DIS = """BEGIN OPTIONS
  LENGTH_UNITS meters
  XORIGIN 500000.0
  YORIGIN 4000000.0
  ANGROT 30.0
END OPTIONS
BEGIN DIMENSIONS
  NLAY 2
  NROW 3
  NCOL 4
END DIMENSIONS
BEGIN GRIDDATA
  DELR
    INTERNAL FACTOR 1.0
    100.0 100.0 200.0 200.0
  DELC
    CONSTANT 150.0
  TOP
    CONSTANT 10.0
  BOTM LAYERED
    CONSTANT 0.0
    CONSTANT -10.0
END GRIDDATA
"""

MF2005_DIS = """# Discretization file
#xul:1000; yul:2450; rotation:0; epsg:32631; start_datetime:2020-01-01
    2    3    4    2    4    2
 0 0
INTERNAL 1.0 (FREE) -1 #delr
 100.0 100.0 200.0 200.0
CONSTANT 150.0 #delc
CONSTANT 10.0
"""

MF6_DISV = """BEGIN OPTIONS
  XORIGIN 10.0
  YORIGIN 50.0
END OPTIONS
BEGIN DIMENSIONS
  NLAY 1
  NCPL 2
  NVERT 6
END DIMENSIONS
BEGIN VERTICES
  1 0.0 1.0
  2 1.0 1.0
  3 2.0 1.0
  4 0.0 0.0
  5 1.0 0.0
  6 2.0 0.0
END VERTICES
BEGIN CELL2D
  1 0.5 0.5 4 1 2 5 4
  2 1.5 0.5 4 2 3 6 5
END CELL2D
"""


def heads_2x3x4():
    a = np.arange(12, dtype=float).reshape(3, 4)
    return [(1, 1, 1.0, 1, a), (1, 1, 1.0, 2, a + 100), (1, 2, 31.0, 1, a + 1), (1, 2, 31.0, 2, a + 101)]


# ── the head file ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("double", [False, True])
def test_read_heads_both_precisions(double):
    h = gs.read_heads(hds_bytes(heads_2x3x4(), double=double))
    assert h["precision"] == ("double" if double else "single")
    assert (h["nrow"], h["ncol"], h["layers"], h["times"]) == (3, 4, [1, 2], [1.0, 31.0])
    assert h["kstpkper"] == [[1, 1], [1, 2]]
    assert h["heads"].shape == (2, 2, 12)
    assert h["heads"][1, 1, 5] == pytest.approx(106.0)


def test_inactive_and_dry_cells_have_no_head():
    a = np.ones((2, 2))
    a[0, 0], a[0, 1], a[1, 0] = 1e30, -999.99, -1e30
    h = gs.read_heads(hds_bytes([(1, 1, 1.0, 1, a)]))
    assert np.isnan(h["heads"][0, 0, :3]).all() and h["heads"][0, 0, 3] == 1.0


def test_not_a_head_file_says_so():
    with pytest.raises(ValueError, match="not a MODFLOW binary head file"):
        gs.read_heads(b"BEGIN OPTIONS\nEND OPTIONS\n" * 4)
    with pytest.raises(ValueError, match="truncated"):
        gs.read_heads(hds_bytes(heads_2x3x4())[:-10])


def test_reader_matches_flopy(tmp_path):
    flopy = pytest.importorskip("flopy")
    for double in (False, True):
        p = tmp_path / f"m{int(double)}.hds"
        p.write_bytes(hds_bytes(heads_2x3x4(), double=double))
        mine = gs.read_heads(p)
        hf = flopy.utils.HeadFile(str(p))
        assert mine["times"] == pytest.approx(hf.get_times())
        for k, t in enumerate(hf.get_times()):
            assert np.allclose(mine["heads"][k], hf.get_data(totim=t).reshape(2, -1))


# ── grids ───────────────────────────────────────────────────────────────────


def test_mf6_dis_text():
    g = gs.read_grid(MF6_DIS.encode())
    assert (g["type"], g["nrow"], g["ncol"], g["nlay"]) == ("dis", 3, 4, 2)
    assert list(g["delr"]) == [100, 100, 200, 200] and list(g["delc"]) == [150] * 3
    assert (g["xorigin"], g["yorigin"], g["angrot"], g["length_units"]) == (500000.0, 4000000.0, 30.0, "meters")


def test_mf2005_dis_with_flopy_georeference():
    g = gs.read_grid(MF2005_DIS.encode())
    assert (g["nrow"], g["ncol"], g["crs"], g["time_units"], g["length_units"]) == (3, 4, "EPSG:32631", "days",
                                                                                   "meters")
    # FloPy's xul/yul is the upper-left corner; the origin is the lower-left one (450 m of rows below it).
    assert (g["xorigin"], g["yorigin"]) == (1000.0, 2000.0)
    assert g["start_datetime"] == "2020-01-01"


def test_binary_grid_file():
    g = gs.read_grid(grb_bytes(3, 4, [100, 100, 200, 200], [150] * 3, 7.0, 8.0, 0.0))
    assert (g["type"], g["nrow"], g["ncol"], g["xorigin"], g["yorigin"]) == ("dis", 3, 4, 7.0, 8.0)


def test_disv_text():
    g = gs.read_grid(MF6_DISV.encode())
    assert g["type"] == "disv" and g["ncpl"] == 2 and list(g["cell_offsets"]) == [0, 4, 8]


def test_external_arrays_are_refused_with_a_way_out():
    text = MF6_DIS.replace("CONSTANT 150.0", "OPEN/CLOSE delc.txt")
    with pytest.raises(ValueError, match="dis.grb"):
        gs.read_grid(text.encode())


# ── placing heads on the map ────────────────────────────────────────────────


def test_heads_on_a_lonlat_grid_without_pyproj():
    res = gs.read_modflow_heads(hds_bytes(heads_2x3x4()), cell_size=0.01, origin=[10.0, 45.0], crs="EPSG:4326",
                                start_date="2024-01-01")
    assert res["grid"]["placed_by"] == "you"
    assert res["dates"] == ["2024-01-02", "2024-02-01"]
    rings, off = res["rings"], res["ring_offsets"]
    assert len(off) == 13
    # Row 1 is the northern row: cell 1's top-left corner is the origin plus three rows of 0.01 degrees.
    assert rings[0] == pytest.approx([10.0, 45.03])
    assert rings[off[11]:off[12]][2] == pytest.approx([10.04, 45.0])
    assert gs.modflow_frame(res, -1, 2)[0] == pytest.approx(101.0)


def test_a_grid_that_is_nowhere_asks_where_it_is():
    res = gs.read_modflow_heads(hds_bytes(heads_2x3x4()), grb_bytes(3, 4, [100] * 4, [150] * 3))
    assert res["error"] and "crs" in res["needs"]
    assert gs.read_modflow_heads(hds_bytes(heads_2x3x4()))["needs"] == ["grid", "origin", "crs"]


def test_mismatched_grid_is_refused():
    with pytest.raises(ValueError, match="same model"):
        gs.read_modflow_heads(hds_bytes(heads_2x3x4()), grb_bytes(2, 4, [100] * 4, [150] * 2), crs=4326)


def test_projected_grid_matches_pyproj_and_rotation():
    pyproj = pytest.importorskip("pyproj")
    res = gs.read_modflow_heads(hds_bytes(heads_2x3x4()), MF6_DIS.encode(), crs=32633)
    tr = pyproj.Transformer.from_crs(32633, 4326, always_xy=True)
    # The lower-left corner of the grid (bottom-left of cell 9) is the origin itself.
    lon, lat = tr.transform(500000.0, 4000000.0)
    assert res["rings"][res["ring_offsets"][8] + 3] == pytest.approx([lon, lat], abs=1e-9)
    # The far corner sits 600 m along the 30-degree-rotated x axis and 450 m along y.
    a = math.radians(30)
    x = 500000 + 600 * math.cos(a) - 450 * math.sin(a)
    y = 4000000 + 600 * math.sin(a) + 450 * math.cos(a)
    lon, lat = tr.transform(x, y)
    assert res["rings"][res["ring_offsets"][3] + 1] == pytest.approx([lon, lat], abs=1e-9)
    assert res["grid"]["crs"] == "EPSG:32633" and res["grid"]["rotation"] == 30.0


def test_sidecar_name_file_places_a_binary_grid():
    pytest.importorskip("pyproj")
    nam = b"# Name file\n#xul:500000; yul:4000450; rotation:0; epsg:32633; start_datetime:1/1/2015\nLIST 2 m.list\n"
    res = gs.read_modflow_heads(hds_bytes(heads_2x3x4()), grb_bytes(3, 4, [100, 100, 200, 200], [150] * 3),
                                sidecars=[nam])
    assert res["grid"]["placed_by"] == "the model's files"
    assert res["grid"]["origin"] == pytest.approx([500000.0, 4000000.0])
    assert res["dates"][0] == "2015-01-02"


# The first two lines of FloPy's freyberg_multilayer_transient/freyberg.nam (public domain), as the Explorer's
# MODFLOW sample drops them (#591): the upper-left corner, a 15-degree rotation and a proj4 CRS.
FREYBERG_NAM = (b"# Name file for mf2005; generated by Flopy.\n"
                b"#xul:619653; yul:3353277; rotation:15; proj4_str:+proj=utm +zone=14 +ellps=WGS84 +datum=WGS84 "
                b"+units=m +no_defs; start_datetime:1/1/2015\nLIST 2 freyberg.list\n")


def test_a_flopy_name_file_with_proj4_places_the_model():
    pytest.importorskip("pyproj")
    frames = [(1, 1, 1.0, 1, np.zeros((40, 20)))]
    grid = grb_bytes(40, 20, [250.0] * 20, [250.0] * 40)
    res = gs.read_modflow_heads(hds_bytes(frames), grid, sidecars=[FREYBERG_NAM])
    assert res["grid"]["placed_by"] == "the model's files" and res["grid"]["approximate"] is False
    assert res["grid"]["crs"].startswith("+proj=utm +zone=14")
    # xul, yul is the upper-left corner: the lower-left one is 40 rows of 250 m down the rotated y axis.
    a = math.radians(15)
    assert res["grid"]["origin"] == pytest.approx([619653 + 10000 * math.sin(a), 3353277 - 10000 * math.cos(a)])
    assert res["grid"]["rotation"] == 15.0 and res["dates"][0] == "2015-01-02"


def test_a_model_with_no_georeference_is_placed_approximately_when_asked():
    """Never a wall (#591): the map's centre, north up, at the model's true size, and the result says so."""
    hds = hds_bytes(heads_2x3x4())
    grid = grb_bytes(3, 4, [100] * 4, [150] * 3)
    assert gs.read_modflow_heads(hds, grid)["needs"] == ["crs"]
    res = gs.read_modflow_heads(hds, grid, approximate_at=[10.0, 50.0])
    g = res["grid"]
    assert g["approximate"] is True and g["origin"] is None and g["crs"] is None and g["rotation"] == 0.0
    assert any(n.startswith("Placed approximately") for n in res["notes"])
    assert any("read as metres" in n for n in res["notes"])   # a binary grid file names no length unit
    assert "approximately" in res["summary"]
    lon, lat = res["rings"][:, 0], res["rings"][:, 1]
    # The middle of the model is the point asked for; 400 m wide and 450 m tall on the ground.
    assert (lon.min() + lon.max()) / 2 == pytest.approx(10.0) and (lat.min() + lat.max()) / 2 == pytest.approx(50.0)
    width_m = math.radians(lon.max() - lon.min()) * gs.EARTH_RADIUS_M * math.cos(math.radians(50.0))
    height_m = math.radians(lat.max() - lat.min()) * gs.EARTH_RADIUS_M
    assert width_m == pytest.approx(400.0, rel=1e-6) and height_m == pytest.approx(450.0, rel=1e-6)
    # Row 1 is still the northern row, north up.
    assert res["rings"][0][1] == pytest.approx(lat.max())


def test_approximate_placement_keeps_the_model_length_unit():
    frames = [(1, 1, 1.0, 1, np.zeros((3, 4)))]
    res = gs.read_modflow_heads(hds_bytes(frames), cell_size=100, length_units="feet", approximate_at=[0.0, 0.0])
    lon = res["rings"][:, 0]
    width_m = math.radians(lon.max() - lon.min()) * gs.EARTH_RADIUS_M
    assert width_m == pytest.approx(400 * 0.3048, rel=1e-6)
    assert any("(in feet)" in n for n in res["notes"])
    with pytest.raises(ValueError, match="lon, lat"):
        gs.read_modflow_heads(hds_bytes(frames), cell_size=100, approximate_at=[0.0, 89.0])


def test_disv_heads_placed_approximately():
    frames = [(1, 1, 1.0, 1, np.array([[5.0, 6.0]]))]
    res = gs.read_modflow_heads(hds_bytes(frames), MF6_DISV.encode(), approximate_at=[-3.0, 40.0])
    lon, lat = res["rings"][:, 0], res["rings"][:, 1]
    assert (lon.min() + lon.max()) / 2 == pytest.approx(-3.0) and (lat.min() + lat.max()) / 2 == pytest.approx(40.0)


def test_disv_heads():
    frames = [(1, 1, 1.0, 1, np.array([[5.0, 6.0]]))]
    res = gs.read_modflow_heads(hds_bytes(frames), MF6_DISV.encode(), crs=4326)
    assert list(res["ring_offsets"]) == [0, 4, 8]
    assert res["rings"][0] == pytest.approx([10.0, 51.0])
    gj = gs.modflow_geojson(res)
    assert [f["properties"]["head"] for f in gj["features"]] == [5.0, 6.0]
    assert gj["features"][0]["geometry"]["coordinates"][0][0] == gj["features"][0]["geometry"]["coordinates"][0][-1]


def test_big_grids_are_drawn_in_blocks():
    a = np.arange(400, dtype=float).reshape(20, 20)
    res = gs.read_modflow_heads(hds_bytes([(1, 1, 1.0, 1, a)]), cell_size=0.001, origin=[0, 0], crs=4326,
                                max_cells=100)
    assert res["grid"]["block"] == 2 and len(res["ring_offsets"]) == 101
    assert gs.modflow_frame(res)[0] == pytest.approx(np.mean([0, 1, 20, 21]))
    assert any("blocks" in n for n in res["notes"])


def test_summary_is_json_safe():
    import json

    res = gs.read_modflow_heads(hds_bytes(heads_2x3x4()), cell_size=0.01, origin=[10.0, 45.0], crs=4326)
    out = gs.modflow_summary(res, time_index=0, layer=2)
    json.dumps(out)
    assert out["frame"]["layer"] == 2 and out["frame"]["min"] == 100.0
    assert "heads" not in out and "rings" not in out


# ── the surface from wells ──────────────────────────────────────────────────


def plane_wells(n=6, spacing=0.08, lon0=-1.0, lat0=52.0):
    """A small square of wells whose head is a plane: 100 m + 50 m per degree east."""
    wells, ids, dates, values = [], [], [], []
    for i in range(n):
        for j in range(n):
            wid = f"w{i}-{j}"
            lon, lat = lon0 + j * spacing, lat0 + i * spacing
            wells.append({"id": wid, "lon": lon, "lat": lat, "name": wid})
            for d, rise in (("2024-03-10", 0.0), ("2025-03-12", 0.5)):
                ids.append(wid)
                dates.append(d)
                values.append(100 + 50 * (lon - lon0) + rise)
    return wells, {"id": ids, "date": dates, "value": values}


def test_month_values_mean_then_nearest():
    well = np.array([0, 0, 1, 2])
    day = gs._days(["2025-03-02", "2025-03-30", "2025-04-10", "2025-06-01"])
    vals = np.array([1.0, 3.0, 7.0, 9.0])
    out = gs.month_values(well, day, vals, "2025-03", 3)
    assert out[0] == {"value": 2.0, "how": "month mean", "n": 2}
    assert out[1]["how"] == "nearest" and out[1]["date"] == "2025-04-10"
    assert out[2] is None


def test_pick_month_falls_back_when_few_wells_report():
    well = np.repeat(np.arange(10), 3)
    day = gs._days([d for _ in range(10) for d in ("2025-01-15", "2025-02-15", "2025-03-15")])
    keep = ~((day == gs._days(["2025-03-15"])[0]) & (well >= 2))  # only 2 of 10 wells report in March
    ym, note = gs.pick_month(well[keep], day[keep], "2025-03")
    assert ym == (2025, 2) and "so this is February 2025" in note


def test_level_surface_recovers_a_plane_and_masks_sparse_areas():
    wells, readings = plane_wells()
    wells.append({"id": "lonely", "lon": 1.5, "lat": 52.2})
    readings["id"].append("lonely")
    readings["date"].append("2025-03-05")
    readings["value"].append(10.0)
    res = gs.surface_from_readings(wells, readings, month="2025-03", mode="level", source="uk_ea")
    assert res["month"] == "2025-03" and res["datum"].startswith("mAOD")
    g = res["grid"]
    vals = np.array([np.nan if v is None else v for v in g["values"]]).reshape(g["ny"], g["nx"])
    st = np.array(g["status"]).reshape(g["ny"], g["nx"])
    assert (st == 2).any() and (st == 1).any() and (st == 0).any()
    assert np.isnan(vals[st != 2]).all()
    # Inside the square of wells the surface follows the plane (IDW flattens it towards the edges).
    drawn = np.nonzero(st == 2)
    lon_c = g["west"] + (drawn[1] + 0.5) * (g["east"] - g["west"]) / g["nx"]
    inside = (lon_c > -0.95) & (lon_c < -0.65)
    truth = 100.5 + 50 * (lon_c[inside] + 1.0)
    got = vals[drawn][inside]
    assert np.corrcoef(got, truth)[0, 1] > 0.97 and np.abs(got - truth).mean() < 2.0
    assert res["contours"]["levels"] and res["contours"]["step"] > 0
    assert "within 20 km" in res["rule"]


def test_change_surface_and_its_shares():
    wells, readings = plane_wells()
    res = gs.surface_from_readings(wells, readings, month="2025-03", mode="change")
    assert res["compare"] == "2024-03" and res["label"] == "March 2024 to March 2025"
    assert res["range"][0] == -res["range"][1]
    assert res["rising_share"] == 1.0 and res["falling_share"] == 0.0
    assert all(w["value"] == pytest.approx(0.5) for w in res["wells"])
    assert "contours" not in res


def test_level_needs_a_common_datum():
    wells, readings = plane_wells()
    res = gs.surface_from_readings(wells, readings, month="2025-03", mode="level", source="elsewhere")
    assert "no common datum" in res["error"]
    assert not gs.surface_from_readings(wells, readings, month="2025-03", mode="change", source="elsewhere").get(
        "error")


def test_bbox_limits_wells_listed_and_grid():
    wells, readings = plane_wells()
    res = gs.surface_from_readings(wells, readings, month="2025-03", bbox=[-1.01, 51.99, -0.85, 52.15])
    assert 0 < res["n_wells_in_view"] < len(wells)
    assert res["grid"]["west"] >= -1.01 and res["grid"]["east"] <= -0.85
    # The wells just outside the view that the mask counted come back too, flagged, so the map card's
    # "N wells within 20 km" agrees with the mask at the view's edge.
    inside = [w for w in res["wells"] if w.get("in_view", True)]
    outside = [w for w in res["wells"] if w.get("in_view") is False]
    assert len(inside) == res["n_wells_in_view"] and outside
    assert all(not (-1.01 <= w["lon"] <= -0.85 and 51.99 <= w["lat"] <= 52.15) for w in outside)


def test_rows_format_and_geojson_and_summary():
    import json

    wells, readings = plane_wells(n=3)
    rows = list(zip(readings["id"], readings["date"], readings["value"]))
    res = gs.surface_from_readings(wells, rows, month="2025-03")
    gj = gs.surface_geojson(res)
    kinds = {f["properties"]["layer"] for f in gj["features"]}
    assert {"surface", "well"} <= kinds
    s = gs.surface_summary(res)
    json.dumps(s)
    assert "values" not in s["grid"]


def test_groundwater_surface_reads_the_archive(monkeypatch):
    import pandas as pd

    wells, readings = plane_wells(n=3)
    rows = [{"source": "uk_ea", "station_id": w["id"], "name": w["name"], "longitude": w["lon"],
             "latitude": w["lat"], "variables": ["groundwater_level"]} for w in wells]
    rows.append({"source": "usgs", "station_id": "x", "longitude": 0, "latitude": 0, "variables": ["discharge"]})
    monkeypatch.setattr("aquascope.archive.catalog.load_stations", lambda **k: rows)
    df = pd.DataFrame({"station_id": readings["id"], "date": readings["date"], "value": readings["value"]})
    monkeypatch.setattr("aquascope.archive.bundles.load_observations", lambda *a, **k: df)
    res = gs.groundwater_surface([-1.5, 51.5, 0.0, 52.5], month="2025-03", mode="change")
    assert res["n_wells_reporting"] == 9 and res["mode"] == "change" and res["grid"]
    with pytest.raises(ValueError, match="uk_ea"):
        gs.groundwater_surface(source="usgs")


# ── the faces: CLI and MCP ──────────────────────────────────────────────────


def test_cli_modflow_prints_and_writes_geojson(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from aquascope import cli

    hds = tmp_path / "m.hds"
    hds.write_bytes(hds_bytes(heads_2x3x4()))
    out = tmp_path / "frame.geojson"
    monkeypatch.setattr(sys, "argv", ["aquascope", "modflow", str(hds), "--cell-size", "0.01", "--origin", "10", "45",
                                      "--crs", "EPSG:4326", "--layer", "2", "--start-date", "2024-01-01",
                                      "--geojson", str(out)])
    cli.main()
    text = capsys.readouterr().out
    assert "layer 2, 2024-02-01: heads 101 to 112 on 12 cells" in text
    assert len(json.loads(out.read_text())["features"]) == 12


def test_cli_modflow_says_what_it_needs(tmp_path, monkeypatch, capsys):
    import sys

    from aquascope import cli

    hds = tmp_path / "m.hds"
    hds.write_bytes(hds_bytes(heads_2x3x4()))
    monkeypatch.setattr(sys, "argv", ["aquascope", "modflow", str(hds)])
    with pytest.raises(SystemExit):
        cli.main()
    assert "Where is this model?" in capsys.readouterr().out


def test_cli_groundwater_surface(monkeypatch, capsys):
    import sys

    from aquascope import cli

    wells, readings = plane_wells(n=3)
    res = gs.surface_from_readings(wells, readings, month="2025-03", mode="change")
    seen = {}

    def fake(bbox, **kw):
        seen.update(kw, bbox=bbox)
        return res

    monkeypatch.setattr("aquascope.groundwater.surface.groundwater_surface", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "groundwater", "surface", "--mode", "change", "--month", "2025-03",
                                      "--bbox", "-2", "51", "0", "53"])
    cli.main()
    text = capsys.readouterr().out
    assert seen == {"bbox": [-2.0, 51.0, 0.0, 53.0], "month": "2025-03", "mode": "change", "compare": None}
    assert "March 2024 to March 2025" in text and "within 20 km" in text and "Open Government Licence" in text


def test_cli_groundwater_still_needs_analysis_without_surface(monkeypatch):
    import sys

    from aquascope import cli

    monkeypatch.setattr(sys, "argv", ["aquascope", "groundwater", "--analysis", "trend"])
    with pytest.raises(SystemExit):
        cli.main()


def test_mcp_tools(tmp_path, monkeypatch):
    from aquascope import mcp_server as m

    hds = tmp_path / "m.hds"
    hds.write_bytes(hds_bytes(heads_2x3x4()))
    out = m.modflow_heads(str(hds), cell_size=0.01, origin_x=10, origin_y=45, crs="4326", geojson=True)
    assert out["frame"]["cells_with_head"] == 12 and out["geojson"]["type"] == "FeatureCollection"
    assert m.modflow_heads(str(hds))["needs"] == ["grid", "origin", "crs"]
    wells, readings = plane_wells(n=3)
    res = gs.surface_from_readings(wells, readings, month="2025-03")
    monkeypatch.setattr("aquascope.groundwater.surface.groundwater_surface", lambda *a, **k: res)
    assert "values" not in m.groundwater_surface()["grid"]
    assert "values" in m.groundwater_surface(grid=True)["grid"]
