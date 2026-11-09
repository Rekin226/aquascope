"""Groundwater views (#623, #624): contours, arrows, drawdown, wells, pathlines and capture zones."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from aquascope.groundwater import views as gv
from aquascope.groundwater.surface import read_modflow_heads

R = gv.EARTH_RADIUS_M


def plane_from_metres(x, y, lat0=45.0, lon0=5.0):
    """A Plane whose local metres are exactly x, y (the inverse of Plane's own projection)."""
    k = math.cos(math.radians(lat0))
    lat = lat0 + np.degrees(np.asarray(y) / R)
    lon = lon0 + np.degrees(np.asarray(x) / (R * k))
    # Plane re-centres on the lattice's mean, so build the lattice symmetric about (lon0, lat0)
    return gv.Plane(lon, lat)


def rotated_lattice(nrow, ncol, cell, angle_deg):
    a = math.radians(angle_deg)
    j, i = np.meshgrid(np.arange(ncol + 1), np.arange(nrow + 1))
    u = (j - ncol / 2) * cell            # along the columns
    v = (nrow / 2 - i) * cell            # up the rows (row 0 is the top)
    return u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)


def centres(plane):
    fi, fj = np.meshgrid(np.arange(plane.nrow, dtype=float), np.arange(plane.ncol, dtype=float), indexing="ij")
    return plane.xy(fi, fj)


GX, GY = -0.003, 0.001                     # dh/dx (east) and dh/dy (north): flow to the east-south-east
FLOW_BEARING = math.degrees(math.atan2(-GX, -GY)) % 360


@pytest.fixture(params=[0.0, 30.0])
def analytic(request):
    x, y = rotated_lattice(24, 30, 100.0, request.param)
    plane = plane_from_metres(x, y)
    cx, cy = centres(plane)
    head = 100.0 + GX * cx + GY * cy
    return plane, head


# ── the analytic plane: a constant gradient ─────────────────────────────────


def test_gradient_and_arrows_of_a_plane(analytic):
    plane, head = analytic
    f = gv.flow_field(head, plane)
    assert np.allclose(f["gx"], GX, atol=2e-5)
    assert np.allclose(f["gy"], GY, atol=2e-5)
    a = gv.flow_arrows(f, plane)
    assert a["arrows"]
    assert all(abs((x["bearing"] - FLOW_BEARING + 180) % 360 - 180) < 0.6 for x in a["arrows"])
    assert a["towards"] == "east-south-east"
    assert a["gradient"] == pytest.approx(math.hypot(GX, GY), rel=1e-2)


def test_pathlines_on_a_plane_are_straight_and_down_the_gradient(analytic):
    plane, head = analytic
    f = gv.flow_field(head, plane)
    out = gv.track(f, [[12.0, 6.0], [5.0, 3.0]])
    for pth, end in zip(out["paths"], out["end"]):
        assert end == "edge" and len(pth) > 20
        x, y = plane.xy(pth[:, 0], pth[:, 1])
        d = np.array([x[-1] - x[0], y[-1] - y[0]])
        d /= np.hypot(*d)
        assert math.degrees(math.atan2(d[0], d[1])) % 360 == pytest.approx(FLOW_BEARING, abs=0.5)
        off = np.abs((x - x[0]) * d[1] - (y - y[0]) * d[0])     # distance from the straight line
        assert off.max() < 0.5                                   # metres, on a 100 m grid


def test_backward_tracking_goes_up_the_gradient(analytic):
    plane, head = analytic
    f = gv.flow_field(head, plane)
    pth = gv.track(f, [[12.0, 15.0]], backward=True)["paths"][0]
    x, y = plane.xy(pth[:, 0], pth[:, 1])
    up = math.degrees(math.atan2(x[-1] - x[0], y[-1] - y[0])) % 360
    assert up == pytest.approx((FLOW_BEARING + 180) % 360, abs=0.5)


def test_contours_of_a_plane_are_straight_parallel_and_evenly_spaced(analytic):
    plane, head = analytic
    c = gv.head_contours(head, plane, step=0.5)
    assert c["step"] == 0.5 and len(c["levels"]) >= 3
    k = math.cos(math.radians(plane.lat0))
    g = math.hypot(GX, GY)
    for lv in c["levels"]:
        assert len(lv["lines"]) == 1                             # joined into one polyline
        pts = np.array(lv["lines"][0])
        x = np.radians(pts[:, 0] - plane.lon0) * R * k
        y = np.radians(pts[:, 1] - plane.lat0) * R
        assert np.allclose(100.0 + GX * x + GY * y, lv["level"], atol=2e-3)   # every vertex is on the level
    # neighbouring levels lie step / |gradient| apart
    a, b = (np.array(lv["lines"][0])[0] for lv in c["levels"][:2])
    xa, ya = np.radians(a[0] - plane.lon0) * R * k, np.radians(a[1] - plane.lat0) * R
    hb = c["levels"][1]["level"]
    dist = abs(100.0 + GX * xa + GY * ya - hb) / g
    assert dist == pytest.approx(0.5 / g, rel=1e-3)


# ── an analytic cone: a well alone, on a rotated grid ───────────────────────


def test_arrows_point_into_a_cone_and_its_contours_close_around_it():
    x, y = rotated_lattice(31, 31, 100.0, 30.0)
    plane = plane_from_metres(x, y)
    cx, cy = centres(plane)
    wx, wy = cx[15, 15], cy[15, 15]
    r = np.hypot(cx - wx, cy - wy) + 50.0
    head = 50.0 - 2.0 * np.log(2000.0 / r)                     # Thiem-like: log of the distance to the well
    f = gv.flow_field(head, plane)
    arrows = gv.flow_arrows(f, plane, across=10)["arrows"]
    wlon, wlat = plane.lonlat(np.array([15.0]), np.array([15.0]))
    k = math.cos(math.radians(plane.lat0))
    checked = 0
    for a in arrows:
        dist = math.hypot((a["lon"] - wlon[0]) * k, a["lat"] - wlat[0]) * math.pi / 180 * R
        if dist < 150:
            continue
        to_well = gv._bearing(np.array([a["lon"]]), np.array([a["lat"]]), wlon, wlat)[0]
        assert abs((a["bearing"] - to_well + 180) % 360 - 180) < 6
        checked += 1
    assert checked >= 40
    c = gv.head_contours(head, plane, step=0.5)
    inside = [lv for lv in c["levels"] if 2000.0 * math.exp((lv["level"] - 50.0) / 2.0) - 50.0 < 1300]
    assert len(inside) >= 4
    for lv in inside:
        assert len(lv["lines"]) == 1
        pts = np.array(lv["lines"][0])
        assert np.allclose(pts[0], pts[-1])                       # a closed ring
        px = np.radians(pts[:, 0] - plane.lon0) * R * k
        py = np.radians(pts[:, 1] - plane.lat0) * R
        want = 2000.0 * math.exp((lv["level"] - 50.0) / 2.0) - 50.0
        assert np.allclose(np.hypot(px - wx, py - wy), want, rtol=0.08, atol=15)


# ── the synthetic pumping well (a copy of aquascope_promo/test_models/make_synthetic_models.py) ─────────────


NROW, NCOL, MONTHS, CELL = 30, 40, 24, 100.0
WELL = (14, 22)                                                   # 0-based: row 15, column 23


def hds_bytes(frames):
    out = b""
    for kstp, kper, totim, lay, arr in frames:
        nrow, ncol = arr.shape
        out += struct.pack("<iiff", kstp, kper, totim, totim) + f"{'HEAD':>16}".encode()
        out += struct.pack("<iii", ncol, nrow, lay) + arr.astype("<f4").tobytes()
    return out


def synthetic_model(pumping=None):
    r, c = np.mgrid[0:NROW, 0:NCOL]
    x = (c + 0.5) * CELL
    y = (NROW - r - 0.5) * CELL
    regional = 52.0 - 0.004 * x + 0.002 * y
    dist = np.hypot((c - WELL[1]) * CELL, (r - WELL[0]) * CELL) + 50.0
    inactive = (r + c < 5) | ((r > 25) & (c > 35))
    frames = []
    for m in range(MONTHS):
        p = pumping(m) if pumping else (1.0 if m < 14 else max(0.0, 1.0 - (m - 13) / 8.0))
        seasonal = 0.6 * np.sin(2 * np.pi * (m - 2) / 12.0)
        for lay in (1, 2):
            dd = p * (4.5 if lay == 2 else 3.2) * np.log(1500.0 / np.minimum(dist, 1500.0)) / np.log(30.0)
            h = regional + seasonal - (0.3 if lay == 2 else 0.0) - dd
            frames.append((1, m + 1, 30.4375 * (m + 1), lay, np.where(inactive, 1.0e30, h)))
    delr = " ".join(["100.0"] * NCOL)
    delc = " ".join(["100.0"] * NROW)
    dis = (f"# synthetic\n    2{NROW:5d}{NCOL:5d}{MONTHS:5d}    4    2\n 0 0\n"
           f"INTERNAL 1.0 (FREE) -1 #delr\n {delr}\nINTERNAL 1.0 (FREE) -1 #delc\n {delc}\n"
           "CONSTANT 60.0\nCONSTANT 20.0\nCONSTANT -40.0\n" + " 30.4375 1 1.0 TR\n" * MONTHS)
    return hds_bytes(frames), dis.encode()


@pytest.fixture(scope="module")
def synthetic():
    hds, dis = synthetic_model()
    return read_modflow_heads(hds, dis, start_date="2024-01-01", approximate_at=[2.35, 48.85])


def test_the_pumping_well_is_found_at_row_15_column_23(synthetic):
    w = gv.model_wells(synthetic, 1)
    assert w["from"] == "found"
    assert [(x["row"], x["col"]) for x in w["wells"]] == [(15, 23)]
    assert "local maximum" in w["how"]


def test_a_single_step_finds_the_head_minimum():
    hds, dis = synthetic_model()
    res = read_modflow_heads(hds, dis, approximate_at=[2.35, 48.85])
    one = gv.find_wells(np.asarray(res["heads"])[0, 0].reshape(NROW, NCOL))
    assert [(x["row"], x["col"]) for x in one["wells"]] == [(15, 23)]
    assert "lower than all" in one["how"]


def test_drawdown_cone_is_centred_on_the_well(synthetic):
    v = gv.model_views(synthetic, 5, 1, reference="highest")
    heads = np.asarray(synthetic["heads"])
    dd = gv.drawdown(heads[5, 0], heads[v["drawdown"]["reference"], 0])
    dd = dd.reshape(NROW, NCOL)
    assert np.unravel_index(np.nanargmax(dd), dd.shape) == WELL
    assert v["wells"][0]["drawdown"] == pytest.approx(float(dd[WELL]), abs=1e-3)
    # the drawdown rings are closed loops around the well
    assert v["drawdown"]["levels"] and v["drawdown"]["step"] > 0
    assert v["drawdown"]["reference"] >= 21                       # the recovered months have the highest heads


def test_the_first_step_is_the_default_reference_and_a_better_one_is_suggested(synthetic):
    v = gv.model_views(synthetic, 5, 1)
    assert v["drawdown"]["reference"] == 0
    assert v["drawdown"]["suggested_reference"] >= 21


def test_capture_zone_matches_forward_tracking(synthetic):
    v = gv.model_views(synthetic, 3, 1)
    w = v["wells"][0]
    assert w["pumping"] and v["capture"] and v["capture"][0]["area_km2"] > 0.2
    heads = np.asarray(synthetic["heads"])[3, 0].reshape(NROW, NCOL)
    plane = gv.Plane(*gv.lattice_from_rings(synthetic["rings"], NROW, NCOL))
    f = gv.flow_field(heads, plane)
    zone = gv.capture_zones(f, [WELL], plane)[0]["mask"]
    assert zone[WELL]
    fw = gv.pathlines(f, [WELL], every=1)
    ok = np.isfinite(heads)
    ends = np.zeros((NROW, NCOL), bool)
    idx = np.argwhere(ok)
    for (i, j), well in zip(idx, fw["well"]):
        ends[i, j] = well == 1
    inter = (zone & ends).sum()
    union = (zone | ends).sum()
    assert inter / union > 0.75                                   # the backward outline is the forward catchment
    # water from the far up-gradient side reaches the well; from far down-gradient it does not
    assert not ends[25, 34]


def test_forward_pathlines_end_in_the_well(synthetic):
    v = gv.model_views(synthetic, 3, 1)
    paths = v["capture"][0]["paths"]
    assert paths
    wlon, wlat = v["wells"][0]["lon"], v["wells"][0]["lat"]
    for p in paths:
        assert p[-1] == [wlon, wlat]


def test_no_capture_once_the_pump_stops(synthetic):
    v = gv.model_views(synthetic, 23, 1)
    assert not v["wells"][0]["pumping"] and v["capture"] == []
    assert any("No well pumps" in n for n in v["notes"])


def test_views_are_json_and_follow_the_layer(synthetic):
    v1 = gv.model_views(synthetic, 3, 1)
    v2 = gv.model_views(synthetic, 3, 2)
    json.dumps(v1)
    assert v2["layer"] == 2 and v1["head"]["levels"] != v2["head"]["levels"]
    assert v1["flow"]["towards"] in ("east-south-east", "south-east", "east")
    s = gv.views_summary(v1)
    assert "capture" in s and isinstance(s["head"]["levels"], int) and "rings" not in json.dumps(s)


def test_views_need_a_structured_grid():
    v = gv.model_views({"grid": {"type": "disv"}, "times": [1.0], "layers": [1]})
    assert "structured" in v["error"]


# ── well files ──────────────────────────────────────────────────────────────


MF2005_WEL = """# Well file for MODFLOW, generated by Flopy.
         2         0
         1         0 # stress period 1
         1        15        23 -0.000000
         2         0 # stress period 2
         1        15        23 -500.0
         2         4         5 120.0
        -1         0 # stress period 3: reuse
"""

MF6_WEL = """BEGIN OPTIONS
  BOUNDNAMES
END OPTIONS
BEGIN DIMENSIONS
  MAXBOUND 2
END DIMENSIONS
BEGIN PERIOD 2
  1 15 23 -250.0 supply
END PERIOD
BEGIN PERIOD 4
  1 15 23 0.0
  2 4 5 -10.0
END PERIOD
"""


def test_read_mf2005_wel():
    w = gv.read_wells(MF2005_WEL.encode())
    assert w["format"] == "MODFLOW-2005 WEL" and w["periods"] == 3
    by = {(x["layer"], x["row"], x["col"]): x["q"] for x in w["wells"]}
    assert by[(1, 15, 23)] == [0.0, -500.0, -500.0]
    assert by[(2, 4, 5)] == [0.0, 120.0, 120.0]


def test_read_mf6_wel_fills_periods_forward():
    w = gv.read_wells(MF6_WEL.encode())
    assert w["format"] == "MODFLOW 6 WEL" and w["periods"] == 4
    by = {(x["layer"], x["row"], x["col"]): x["q"] for x in w["wells"]}
    assert by[(1, 15, 23)] == [0.0, -250.0, -250.0, 0.0]
    assert by[(2, 4, 5)] == [0.0, 0.0, 0.0, -10.0]


def test_external_lists_are_noted():
    w = gv.read_wells(b"5 0\n2 0\nOPEN/CLOSE wel_sp1.txt\n")
    assert w["notes"] and "OPEN/CLOSE" in w["notes"][0]


def test_a_wel_file_decides_which_wells_pump(synthetic):
    wel = "1 0\n1 0\n1 15 23 -800.0\n" + "".join("-1 0\n" for _ in range(MONTHS - 1))
    v = gv.model_views(synthetic, 3, 1, wells=wel.encode())
    assert v["wells_from"] == "wel" and v["wells"][0]["pumping"] and v["wells"][0]["q"] == -800.0
    assert v["capture"]
    v2 = gv.model_views(synthetic, 3, 2, wells=wel.encode())
    assert not v2["wells"][0]["pumping"] and not v2["wells"][0]["in_layer"]


def test_experimental():
    from aquascope.experimental import is_experimental

    assert is_experimental("aquascope.groundwater.views")


def test_cli_modflow_views(tmp_path, monkeypatch, capsys):
    import sys

    from aquascope import cli

    hds, dis = synthetic_model()
    (tmp_path / "m.hds").write_bytes(hds)
    (tmp_path / "m.dis").write_bytes(dis)
    monkeypatch.setattr(sys, "argv", ["aquascope", "modflow", str(tmp_path / "m.hds"),
                                      "--grid", str(tmp_path / "m.dis"), "--origin", "448000", "5411000",
                                      "--crs", "EPSG:32631", "--time", "3",
                                      "--views", "--reference", "highest", "--json"])
    pytest.importorskip("pyproj")
    cli.main()
    out = json.loads(capsys.readouterr().out)
    v = out["views"]
    assert v["wells"][0]["model_row"] == 15 and v["wells"][0]["model_col"] == 23
    assert v["flow"]["towards"] in ("east-south-east", "south-east", "east")
    assert v["capture"][0]["area_km2"] > 0.2 and v["drawdown"]["reference"] >= 21
