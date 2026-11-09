"""Review my model results (#564): each check on a small synthetic mesh with one planted problem.

The clean run floods the west half of a 10 x 8 grid of 1 km cells for eight days and drains it once. Every test
plants one problem and asserts that its check, and only its check, flags it.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")

from aquascope import mesh_review as R  # noqa: E402, N812
from aquascope.io import mesh as M  # noqa: E402, N812

UTM = "EPSG:32633"
X0, Y0, DX = 480_000.0, 5_000_000.0, 1000.0
NX, NY = 10, 8
N = NX * NY
STEPS = 8
DAY0 = date(2020, 3, 1)
TIMES = [(DAY0 + timedelta(days=k)).strftime("%d%b%Y").upper() + " 00:00:00:000" for k in range(STEPS)]
BED = 10.0
# The flood: wet from day 2, highest on day 4, still wet at the end (one change from dry to wet).
SHAPE = np.array([0.0, 0.5, 1.0, 1.6, 1.2, 0.8, 0.5, 0.3])
WET_COLS = 5


def cell(row: int, col: int) -> int:
    return row * NX + col


def clean_depths() -> np.ndarray:
    d = np.zeros((STEPS, N))
    for r in range(NY):
        for c in range(WET_COLS):
            d[:, cell(r, c)] = SHAPE * (1 + 0.1 * c + 0.01 * r)
    return d


def make_run(path, depth=None, *, bed=True, volume_pct=0.2, ws_override=None, area_volume_pct=None):
    """A HEC-RAS plan file of the grid: water surface = bed + depth (a dry cell holds the bed, as RAS writes it)."""
    depth = clean_depths() if depth is None else depth
    pid = lambda i, j: j * (NX + 1) + i  # noqa: E731
    pts = np.array([[X0 + i * DX, Y0 + j * DX] for j in range(NY + 1) for i in range(NX + 1)], dtype="f8")
    faces, fidx = [], {}
    rings, cells, centres = [], [], []
    for j in range(NY):
        for i in range(NX):
            corners = [pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1)]
            cf = []
            for k in range(4):
                key = tuple(sorted((corners[k], corners[(k + 1) % 4])))
                if key not in fidx:
                    fidx[key] = len(faces)
                    faces.append(key)
                cf.append(fidx[key])
            cells.append(cf)
            rings.append(corners)
            centres.append([X0 + (i + 0.5) * DX, Y0 + (j + 0.5) * DX])
    faces = np.array(faces, dtype="i4")
    ws = BED + depth if ws_override is None else ws_override
    with h5py.File(path, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Results"
        f.attrs["File Version"] = b"HEC-RAS 6.5 February 2024"
        f.attrs["Units System"] = b"SI Units"
        from pyproj import CRS

        f.attrs["Projection"] = CRS.from_user_input(UTM).to_wkt("WKT1_ESRI").encode()
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
        f.create_dataset("Geometry/2D Flow Areas/Attributes", data=np.array([(b"Valley", N)], dtype=dt))
        g = f.create_group("Geometry/2D Flow Areas/Valley")
        g["FacePoints Coordinate"] = pts
        g["Faces FacePoint Indexes"] = faces
        info, vals = [], []
        for c in cells:
            info.append([len(vals), len(c)])
            vals.extend([[fc, 1] for fc in c])
        g["Cells Face and Orientation Info"] = np.array(info, dtype="i4")
        g["Cells Face and Orientation Values"] = np.array(vals, dtype="i4")
        g["Cells Center Coordinate"] = np.array(centres)
        g["Cells FacePoint Indexes"] = np.array(rings, dtype="i4")
        if bed:
            g["Cells Minimum Elevation"] = np.full(N, BED, dtype="f4")
        ts = f.create_group(M.RAS_TS)
        ts["Time Date Stamp (ms)"] = np.array([t.encode() for t in TIMES], dtype="S22")
        f[f"{M.RAS_TS}/2D Flow Areas/Valley/Water Surface"] = ws.astype("f4")
        f[f"{M.RAS_SUMMARY}/2D Flow Areas/Valley/Maximum Water Surface"] = np.vstack(
            [ws.max(axis=0), np.zeros(N)]).astype("f4")
        if volume_pct is not None:
            va = f.create_group("Results/Unsteady/Summary/Volume Accounting")
            va.attrs["Error Percent"] = np.float32(volume_pct)
            va.attrs["Error"] = np.float32(volume_pct * 10)
            va.attrs["Vol Accounting in"] = b"1000 m3"
            va.attrs["Volume Starting"] = np.float32(0.0)
            va.attrs["Volume Ending"] = np.float32(1000.0)
            if area_volume_pct is not None:   # each 2D area's own, with the names HEC-RAS 6.5 gives them
                per = va.create_group("Volume Accounting 2D/Valley")
                per.attrs["Error Percent"] = np.float32(area_volume_pct)
                per.attrs["Error"] = np.float32(-0.4319)
                per.attrs["Vol Accounting in"] = b"1000 m^3"
                per.attrs["Vol Starting"] = np.float32(2787.7)
                per.attrs["Vol Ending"] = np.float32(3956.2)
                per.attrs["Cum Inflow"] = np.float32(69082.1)
                per.attrs["Cum Outflow"] = np.float32(67913.2)
    return path


@pytest.fixture
def mesh_of(tmp_path):
    opened = []

    def open_(depth=None, **kw):
        m = M.read_mesh(make_run(tmp_path / f"run{len(opened)}.p01.hdf", depth, **kw))
        opened.append(m)
        return m

    yield open_
    for m in opened:
        m.close()


def radar_square(m, i):
    """The centre of the 0.05 degree radar square holding cell i."""
    r = math.floor((m.lat[i] + 90) / R.RADAR_DEG + 1e-9)
    c = math.floor((m.lon[i] + 180) / R.RADAR_DEG + 1e-9)
    return round(-90 + (r + 0.5) * R.RADAR_DEG, 3), round(-180 + (c + 0.5) * R.RADAR_DEG, 3)


def clean_context(m, **over):
    """Observed floods and a gauge that agree with the clean run."""
    wet_cell = cell(3, 1)
    lat, lon = radar_square(m, wet_cell)
    box = lambda i: dict(lat=float(m.lat[i]), lon=float(m.lon[i]), west=float(m.lon[i]) - 0.003,  # noqa: E731
                         east=float(m.lon[i]) + 0.003, south=float(m.lat[i]) - 0.003, north=float(m.lat[i]) + 0.003)
    ctx = {
        "floods": {
            "radar": {"rows": [[lat, lon, "2020-03", 40]], "period": ["2014-10", "2024-09"]},
            "news": {"events": [{"start": "2020-03-04", "end": "2020-03-05", "area_km2": 0.5, **box(wet_cell)}],
                     "period": ["2000-01", "2026-02"]},
        },
        "gauges": [gauge(m, cell(4, 2), peak_day=3)],
    }
    ctx.update(over)
    return ctx


def gauge(m, i, peak_day):
    vals = [10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 15.0, 12.0]
    vals[peak_day] = 99.0
    return {"source": "usgs", "station_id": "TEST-1", "name": "Test Creek", "lat": float(m.lat[i]),
            "lon": float(m.lon[i]), "variable": "discharge", "unit": "m3/s",
            "series": [[(DAY0 + timedelta(days=k)).isoformat(), v] for k, v in enumerate(vals)],
            "cell": int(R.cells_at(m, m.lon[i], m.lat[i])[0])}


def flagged(res):
    return {c["check"] for c in res["checks"] if c["status"] == "flagged"}


def found(res, check):
    return [f for f in res["findings"] if f["check"] == check]


# ── the clean run ───────────────────────────────────────────────────────────


def test_the_clean_run_flags_nothing(mesh_of):
    m = mesh_of()
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == set()
    assert {f["severity"] for f in res["findings"]} == {"info"}
    assert res["counts"]["problem"] == res["counts"]["warning"] == 0
    assert res["summary"].startswith("Nothing flagged")
    by = {c["check"]: c for c in res["checks"]}
    assert by["floods_radar"]["status"] == "ok" and "the model wets cells in 1 of them" in by["floods_radar"]["line"]
    assert by["gauges"]["status"] == "ok"
    on_time = found(res, "gauges")
    assert len(on_time) == 1 and on_time[0]["title"] == "Peak on time at Test Creek"
    json.dumps(res)   # plain JSON for the browser and the MCP tool


def test_the_deepest_cells_carry_depth_time_and_place(mesh_of):
    m = mesh_of()
    res = R.review(m, context=clean_context(m))
    deep = found(res, "deepest")
    assert deep and all(f["severity"] == "info" for f in deep)
    top = deep[0]
    # The deepest cell is the far edge of the flood (column 4, top row): 1.6 x 1.47 m on day 4.
    assert top["value"] == pytest.approx(1.6 * (1 + 0.4 + 0.07), abs=1e-3)
    assert top["unit"] == "m" and top["when"] == "2020-03-04T00:00:00"
    assert top["facts"][0]["when"] == "4 Mar 2020 00:00"
    assert top["cells"] == [cell(7, 4)]
    assert (top["lat"], top["lon"]) == (pytest.approx(m.lat[cell(7, 4)], abs=1e-4),
                                        pytest.approx(m.lon[cell(7, 4)], abs=1e-4))
    assert top["source"].startswith("Your model file")
    # Pins are spread out, not five cells side by side.
    assert len({f["cells"][0] for f in deep}) == len(deep)


# ── one planted problem each ────────────────────────────────────────────────


def test_negative_depth_is_found_and_nothing_else(mesh_of):
    d = clean_depths()
    d[3, cell(5, 8)] = -0.5   # the water surface half a metre under the bed of a dry cell
    m = mesh_of(d)
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == {"negative"}
    f = found(res, "negative")[0]
    assert f["severity"] == "problem" and f["rank"] == 1
    assert f["value"] == pytest.approx(-0.5, abs=1e-4) and f["cells"] == [cell(5, 8)]


def test_a_film_of_water_is_found_and_nothing_else(mesh_of):
    d = clean_depths()
    for r in (1, 2, 3):
        d[3, cell(r, WET_COLS)] = 0.02   # next to the flood, so not isolated
    m = mesh_of(d)
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == {"tiny"}
    films = found(res, "tiny")
    assert sorted(films[0]["cells"]) == [cell(r, WET_COLS) for r in (1, 2, 3)]
    assert films[0]["value"] == pytest.approx(0.02, abs=1e-4)
    assert any(x["label"] == "Cells wetted only by a film" and x["value"] == 3 for x in films[0]["facts"])


def test_an_isolated_wet_cell_is_found_and_nothing_else(mesh_of):
    d = clean_depths()
    d[2:6, cell(7, 9)] = 0.5   # a puddle in the far corner, no wet neighbour
    m = mesh_of(d)
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == {"isolated"}
    f = found(res, "isolated")[0]
    assert f["cells"] == [cell(7, 9)] and f["severity"] == "warning"


def test_wet_dry_flicker_is_found_and_nothing_else(mesh_of):
    d = clean_depths()
    d[:, cell(0, WET_COLS)] = [0, 0.3, 0, 0.3, 0, 0.3, 0, 0.3]   # beside the flood: wet, dry, wet...
    m = mesh_of(d)
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == {"flicker"}
    f = found(res, "flicker")[0]
    assert f["cells"] == [cell(0, WET_COLS)] and f["value"] == 7


def test_volume_error_is_found_and_nothing_else(mesh_of):
    m = mesh_of(volume_pct=3.2)
    res = R.review(m, context=clean_context(m))
    assert flagged(res) == {"mass_balance"}
    f = found(res, "mass_balance")[0]
    assert f["severity"] == "warning" and f["value"] == pytest.approx(3.2, abs=1e-4) and f["unit"] == "%"
    assert R.review(mesh_of(volume_pct=-7.5), context=clean_context(m))["findings"][0]["severity"] == "problem"
    assert R.check_volume(m)["worst"]["unit"] == "1000 m3"


def test_each_2d_areas_volume_accounting_is_read_with_its_own_names(mesh_of):
    # HEC-RAS 6.5 (a real plan, docs/assessments/hec-ras.md) names the per-area volumes apart from the plan's
    m = mesh_of(volume_pct=0.0006, area_volume_pct=1.8)
    va = R.check_volume(m)
    w = va["worst"]
    assert w["where"].endswith("Valley") and w["error_percent"] == pytest.approx(1.8)
    assert (w["start"], w["end"], w["in"], w["out"]) == pytest.approx((2787.7, 3956.2, 69082.1, 67913.2), rel=1e-6)
    assert w["unit"] == "1000 m^3"
    plan = next(p for p in va["parts"] if p["where"] == "plan")
    assert (plan["start"], plan["end"]) == pytest.approx((0.0, 1000.0))
    res = R.review(m, context=clean_context(m))
    f = found(res, "mass_balance")[0]
    assert f["severity"] == "warning" and "(Valley)" in f["reason"]


def test_a_film_share_under_one_percent_is_not_shown_as_nought():
    from types import SimpleNamespace

    n = 2000
    hi = np.full(n, 1.5)
    hi[:3] = 0.02                      # 3 films among 2,000 wet cells: 0.15%
    m = SimpleNamespace(variable="depth", lat=np.linspace(45, 45.1, n), lon=np.linspace(15, 15.1, n))
    out, status, _ = R._tiny(m, {"hi": hi}, R._Ids(), "m", 1.0)
    share = next(x for x in out[0]["facts"] if x["label"] == "Share of the cells that get wet")
    assert status == "flagged" and share["shown"] == "0.15%"


def test_radar_flooding_where_the_model_is_dry_is_found_and_nothing_else(mesh_of):
    m = mesh_of()
    ctx = clean_context(m)
    east = cell(4, NX - 1)
    lat, lon = radar_square(m, east)
    assert not any(radar_square(m, cell(r, c)) == (lat, lon) for r in range(NY) for c in range(WET_COLS))
    ctx["floods"]["radar"]["rows"].append([lat, lon, "2020-03", 7])
    res = R.review(m, context=ctx)
    assert flagged(res) == {"floods_radar"}
    f = found(res, "floods_radar")[0]
    assert f["severity"] == "warning" and f["period"] == "same" and f["value"] == 7
    assert east in f["cells"] and "during the run" in f["reason"]


def test_a_flood_in_the_news_over_dry_cells_is_found_and_nothing_else(mesh_of):
    m = mesh_of()
    ctx = clean_context(m)
    i = cell(6, 8)
    ctx["floods"]["news"]["events"].append({"start": "2020-03-05", "end": None, "lat": float(m.lat[i]),
                                            "lon": float(m.lon[i]), "west": float(m.lon[i]) - 0.002,
                                            "east": float(m.lon[i]) + 0.002, "south": float(m.lat[i]) - 0.002,
                                            "north": float(m.lat[i]) + 0.002, "area_km2": 0.2})
    # One outside the run's days is left out.
    ctx["floods"]["news"]["events"].append({**ctx["floods"]["news"]["events"][-1], "start": "2020-05-01"})
    res = R.review(m, context=ctx)
    assert flagged(res) == {"floods_news"}
    f = found(res, "floods_news")
    assert len(f) == 1 and f[0]["cells"] == [i] and f[0]["severity"] == "warning" and f[0]["when"] == "2020-03-05"


def test_a_gauge_peak_days_off_is_found_and_nothing_else(mesh_of):
    m = mesh_of()
    res = R.review(m, context=clean_context(m, gauges=[gauge(m, cell(4, 2), peak_day=6)]))
    assert flagged(res) == {"gauges"}
    f = found(res, "gauges")[0]
    assert f["severity"] == "warning" and f["value"] == 3 and f["title"] == "Peak 3 days off at Test Creek"
    assert "Timing only" in f["reason"]


def test_more_gauges_than_are_checked_are_counted(mesh_of):
    m = mesh_of()
    many = [{**gauge(m, cell(4, 2), peak_day=3), "station_id": f"TEST-{k}"} for k in range(R.MAX_GAUGES + 2)]
    rows = [{k: g[k] for k in ("source", "station_id", "name", "lat", "lon")} for g in many]
    assert len(R.gauges_inside(m, rows)) == R.MAX_GAUGES + 2   # every gauge inside, none dropped unsaid
    res = R.review(m, context=clean_context(m, gauges=many))
    line = next(c for c in res["checks"] if c["check"] == "gauges")["line"]
    assert line.startswith(f"{R.MAX_GAUGES + 2} gauges inside the mesh, the first {R.MAX_GAUGES} checked")
    assert len(found(res, "gauges")) == R.MAX_GAUGES


def test_one_cell_reads_as_one_cell(mesh_of):
    d = clean_depths()
    d[2:6, cell(7, 9)] = 0.5
    m = mesh_of(d, volume_pct=3.2)
    res = R.review(m, context=clean_context(m))
    assert "1 such cell in all" in found(res, "isolated")[0]["reason"]
    assert "32 × 1000 m3" in found(res, "mass_balance")[0]["reason"]


# ── what the review says when it cannot check ───────────────────────────────


def test_no_gauge_inside_the_mesh_is_said(mesh_of):
    m = mesh_of()
    res = R.review(m, context=clean_context(m, gauges=[]))
    line = next(c for c in res["checks"] if c["check"] == "gauges")
    assert line["status"] == "ok" and "No Archive gauge stands inside the mesh" in line["line"]
    assert not found(res, "gauges")


def test_periods_that_do_not_overlap_are_said(mesh_of):
    m = mesh_of()
    ctx = clean_context(m)
    ctx["floods"]["radar"]["period"] = ["2021-01", "2024-09"]
    ctx["floods"]["radar"]["rows"] = [[*radar_square(m, cell(4, NX - 1)), "2022-07", 5]]
    ctx["floods"]["news"]["period"] = ["2021-01", "2026-02"]
    res = R.review(m, context=ctx)
    by = {c["check"]: c for c in res["checks"]}
    assert "outside the radar's record" in by["floods_radar"]["line"]
    assert "outside the news record" in by["floods_news"]["line"]
    f = found(res, "floods_radar")[0]
    # Floods seen in other months are information, not a fault in this run.
    assert f["severity"] == "info" and f["period"] == "record" and f["reason"].startswith("Not this event")
    assert "floods_radar" not in flagged(res)


def test_without_context_the_outside_checks_say_they_were_skipped(mesh_of):
    res = R.review(mesh_of())
    by = {c["check"]: c["status"] for c in res["checks"]}
    assert by["floods_radar"] == by["floods_news"] == by["gauges"] == "skipped"
    lines = {c["check"]: c["line"] for c in R.review(mesh_of())["checks"]}
    assert "not looked up" in lines["floods_radar"] and "not looked up" in lines["floods_news"]
    assert by["negative"] == "ok" and by["mass_balance"] == "ok"


def test_a_file_without_bed_levels_reads_rises_and_skips_depth_checks(mesh_of):
    m = mesh_of(bed=False)
    assert m.variable == "water_surface"
    res = R.review(m, context=clean_context(m))
    by = {c["check"]: c for c in res["checks"]}
    for check in ("negative", "tiny", "flicker"):
        assert by[check]["status"] == "skipped" and "no bed levels" in by[check]["line"]
    deep = found(res, "deepest")
    assert deep[0]["title"].startswith("Largest rise") and "no bed levels" in deep[0]["reason"]
    assert deep[0]["value"] == pytest.approx(1.6 * 1.47, abs=1e-3)


def test_a_ugrid_depth_below_zero_is_read_raw(tmp_path):
    from scipy.io import netcdf_file

    path = tmp_path / "flat.nc"
    # Two triangles in degrees; the depth goes below zero in the first at the second step.
    with netcdf_file(path, "w") as f:
        f.createDimension("time", 3)
        f.createDimension("node", 4)
        f.createDimension("face", 2)
        f.createDimension("three", 3)
        topo = f.createVariable("mesh", "i", ())
        topo.cf_role = b"mesh_topology"
        topo.topology_dimension = 2
        topo.node_coordinates = b"x y"
        topo.face_node_connectivity = b"faces"
        x = f.createVariable("x", "d", ("node",))
        y = f.createVariable("y", "d", ("node",))
        x[:], y[:] = [4.0, 4.01, 4.01, 4.0], [52.0, 52.0, 52.01, 52.01]
        x.units, y.units = b"degrees_east", b"degrees_north"
        c = f.createVariable("faces", "i", ("face", "three"))
        c[:] = [[0, 1, 2], [0, 2, 3]]
        t = f.createVariable("time", "d", ("time",))
        t[:] = [0, 3600, 7200]
        t.units = b"seconds since 2020-06-01 00:00:00"
        d = f.createVariable("waterdepth", "d", ("time", "face"))
        d[:] = [[1.0, 1.0], [-0.2, 1.2], [0.8, 1.1]]
        d.mesh = b"mesh"
        d.units = b"m"
    m = M.read_mesh(path)
    try:
        assert m.values_block(1, 2, raw=True)[0, 0] == pytest.approx(-0.2)
        assert math.isnan(m.values_block(1, 2)[0, 0])
        res = R.review(m)
        assert {c["check"] for c in res["checks"] if c["status"] == "flagged"} == {"negative"}
        assert found(res, "negative")[0]["value"] == pytest.approx(-0.2)
    finally:
        m.close()


# ── the mesh helpers ────────────────────────────────────────────────────────


def test_neighbours_and_cell_lookup(mesh_of):
    m = mesh_of()
    i, j = R.neighbours(m)
    pairs = {tuple(sorted(p)) for p in zip(i.tolist(), j.tolist())}
    # A 10 x 8 grid has 9 x 8 east-west and 10 x 7 north-south shared edges.
    assert len(pairs) == 9 * 8 + 10 * 7
    assert (cell(0, 0), cell(0, 1)) in pairs and (cell(0, 0), cell(1, 0)) in pairs
    assert (cell(0, 0), cell(1, 1)) not in pairs
    got = R.cells_at(m, [m.lon[cell(3, 4)], 0.0], [m.lat[cell(3, 4)], 0.0])
    assert got.tolist() == [cell(3, 4), -1]
    rows = [{"source": "x", "station_id": "1", "lat": float(m.lat[5]), "lon": float(m.lon[5])},
            {"source": "x", "station_id": "2", "lat": 10.0, "lon": 10.0}]
    inside = R.gauges_inside(m, rows)
    assert [g["station_id"] for g in inside] == ["1"] and inside[0]["cell"] == 5


# ── the model path: the model picks, the code writes (#590, round 3) ─────────


def _review_with_problems(mesh_of):
    d = clean_depths()
    d[3, cell(5, 8)] = -0.5
    m = mesh_of(d, volume_pct=3.2)
    return R.review(m, context=clean_context(m))


def test_a_model_picks_the_pins_their_order_and_a_variant_and_the_package_writes_them(mesh_of):
    res = _review_with_problems(mesh_of)
    neg, vol = found(res, "negative")[0], found(res, "mass_balance")[0]
    deep = found(res, "deepest")[0]
    reply = json.dumps({"pins": [{"id": deep["id"], "variant": "c"}, {"id": vol["id"], "variant": "b"}]})
    out = R.apply_words(res["findings"], reply, by="device")
    ids = [f["id"] for f in out["findings"]]
    # Its order; the problem it left out is pinned all the same, after its picks.
    assert ids == [deep["id"], vol["id"], neg["id"]] and out["ordered"] and out["refused"] == []
    first, second, third = out["findings"]
    assert first["by"] == "device" and first["variant"] == "c"
    assert first["title"] == deep["rules_title"]
    assert first["reason"] == deep["rules_reason"].split(". Worth")[0] + "."      # the first sentence, the package's
    assert second["title"] == "Volume balance" and second["reason"] == vol["rules_reason"]
    assert third["by"] == "rules" and third["title"] == neg["rules_title"] and third["severity"] == "problem"
    # The findings it left out (not problems) are listed, not pinned; values, places and cells never move.
    assert set(out["left_out"]) == {f["id"] for f in res["findings"]} - set(ids)
    assert first["value"] == deep["value"] and first["cells"] == deep["cells"]
    # The counts are the rules' counts: the review's own, untouched.
    assert res["summary"].startswith("1 problem")


def test_every_variant_is_the_packages_words_held_to_its_facts(mesh_of):
    from aquascope.map_explain import _numbers

    res = _review_with_problems(mesh_of)
    for f in res["findings"]:
        rules = f"{f['rules_title']} {f['rules_reason']}"
        for v, (title, reason) in R.review_variants(f).items():
            assert title and reason, (f["id"], v)
            # The second line of defence holds every variant: no number the rules' words do not say, no "not"
            # dropped.
            assert _numbers(f"{title} {reason}") <= _numbers(rules), (f["id"], v, reason)
            assert R.negation_flip(f"{f['rules_title']}. {f['rules_reason']}", reason) is None, (f["id"], v)


@pytest.mark.parametrize("reply, why", [
    # The old free-text reply shape: refused whole, nothing of it shown.
    ({"order": ["r1"], "notes": [{"id": "r1", "title": "Worst cell", "reason": "Depth falls to {f1} m."}]},
     "reply has no pins"),
    ({"pins": [{"id": "r99", "variant": "a"}]}, "reply.pins[0].id 'r99' is not one of"),
    ({"pins": [{"id": "r1", "variant": "d"}]}, "reply.pins[0].variant 'd' is not one of a, b, c"),
    ({"pins": [{"id": "r1", "variant": "a", "reason": "Two cells flicker."}]}, "which the schema does not hold"),
    ({"pins": []}, "fewer than 1 items"),
    ("not json", "the model did not reply with JSON"),
])
def test_a_reply_outside_the_schema_is_refused_whole(mesh_of, reply, why):
    res = _review_with_problems(mesh_of)
    out = R.apply_words(res["findings"], reply, by="device")
    assert len(out["refused"]) == 1 and why in out["refused"][0], out["refused"]
    assert [f["id"] for f in out["findings"]] == [f["id"] for f in res["findings"]]
    assert all(f["by"] == "rules" and f["title"] == f["rules_title"] for f in out["findings"])
    assert not out["ordered"]


@pytest.mark.parametrize("title, reason, why", [
    ("Worst cell", "Depth falls to {F1} m.", "ranking"),
    ("Below the bed", "Depth falls to {f999} m.", "not one of its facts"),
    ("Below the bed", "Twelve cells go below zero.", "number"),
    ("Below the bed", "Depth falls to {F1.when}.", "has no time"),
    ("", "Depth falls.", "empty"),
    ("Below the bed", "Depth falls to -{F1} m.", "sign or a digit"),
    ("Below the bed", "Depth falls to {F1}0 m.", "sign or a digit"),
    ("Below the bed", "Depth falls to minus {F1} m.", "sign or a digit"),
    ("Below the bed", "Depth falls to {F1}.5 m.", "sign or a digit"),
    ("Below the bed", "Depth falls to \u2212{F1} m.", "sign or a digit"),
])
def test_the_older_lock_still_refuses(mesh_of, title, reason, why):
    # The free-text lock is no longer on the model path; it stays as a second line of defence.
    res = _review_with_problems(mesh_of)
    neg = found(res, "negative")[0]
    reason = reason.replace("F1", neg["facts"][0]["id"])
    got, err = R._locked(neg, title, reason)
    assert got is None and why in err


@pytest.mark.parametrize("reason", [
    # Gemini Nano on the HEC-RAS Portsmouth run, 2026-10-25 and 2026-10-26, on the deepest-water pins. The rules
    # say "Worth a look if it is not a channel or a pit you expect."
    "Deepest water is {F1} at {F1.when}; investigate if it's an expected feature.",
    "Deepest water of {F1}: check if expected (e.g. channel or pit).",
])
def test_the_older_lock_still_catches_a_dropped_not(mesh_of, reason):
    m = mesh_of()
    res = R.review(m, context=clean_context(m))
    deep = found(res, "deepest")[0]
    got, err = R._locked(deep, "Deepest water", reason.replace("F1", deep["facts"][0]["id"]))
    assert got is None and err == "drops the 'not' of the rules wording ('if it is not a channel or a pit you expect')"


def test_a_negated_statement_said_without_its_not_is_refused():
    rules = "Gauge Kingston: no record for the run. It has no observations during the run."
    assert R.negation_flip(rules, "The gauge has a record for the run.") == "no record for the run"
    assert R.negation_flip(rules, "No record at this gauge covers the run.") is None
    assert R.negation_flip(rules, "The gauge has records outside the run period.") is None   # other words
    film = "This cell never gets deeper than 0.02 m, under 0.03 m."
    assert R.negation_flip(film, "This cell gets deeper than {f1}.") == "never gets deeper than 0.02 m"
    assert R.negation_flip(film, "A thin film, at most {f1} deep.") is None


def test_the_prompt_offers_ids_and_variants_and_no_free_text(mesh_of):
    res = _review_with_problems(mesh_of)
    p = R.review_prompt(res["findings"], context="Portsmouth")
    ids = [f["id"] for f in res["findings"]]
    item = p["schema"]["properties"]["pins"]["items"]
    assert item["properties"]["id"]["enum"] == ids and item["properties"]["variant"]["enum"] == ["a", "b", "c"]
    assert item["additionalProperties"] is False and set(item["required"]) == {"id", "variant"}
    # No string field the model could fill with words of its own.
    assert all("enum" in v for v in item["properties"].values())
    for f in res["findings"]:
        assert f"{f['id']} ({f['severity']}" in p["system"] and f["rules_reason"] in p["system"]
    assert "never write a word of a pin" in p["system"] and "Portsmouth" in p["system"]


@pytest.mark.parametrize("question", [
    # The two questions of eight in which the model dropped both flicker findings (ai-on-the-map.md, round four),
    # and the words a reader uses instead.
    "are there any numerical instabilities in this run?",
    "is the model stable?",
    "does the depth flicker at the edge?",
    "any wet and dry problems?",
    "the water surface oscillates - is that a problem?",
])
def test_the_question_names_the_flicker(question):
    findings = [{"id": "r1", "check": "deepest"}, {"id": "r2", "check": "flicker"}]
    assert R.named_findings(findings, question) == ["r2"]
    # The ids follow the review's order, not the words' order.
    assert R.named_findings([{"id": "r1", "check": "flicker"}, {"id": "r2", "check": "deepest"}], question) == ["r1"]


@pytest.mark.parametrize("question", [
    "review my model results", "is anything wrong with my flood model?", "does this simulation look plausible?",
    "are the depths in my flood model realistic?", "check my model", "is the volume balance right?", "",
])
def test_a_question_that_names_nothing_pins_nothing_extra(question):
    assert R.named_findings([{"id": "r1", "check": "flicker"}, {"id": "r2", "check": "mass_balance"}],
                            question) == []


def _review_with_flicker(mesh_of):
    d = clean_depths()
    d[:, cell(0, WET_COLS)] = [0, 0.3, 0, 0.3, 0, 0.3, 0, 0.3]   # beside the flood: wet, dry, wet...
    m = mesh_of(d)
    return R.review(m, context=clean_context(m))


def test_a_question_about_stability_pins_the_flicker_the_model_left_out(mesh_of):
    res = _review_with_flicker(mesh_of)
    flick = found(res, "flicker")[0]
    deep = found(res, "deepest")[0]
    reply = json.dumps({"pins": [{"id": deep["id"], "variant": "a"}]})
    # Its order first, then the finding the question names, in the package's wording, not in the left-out list.
    out = R.apply_words(res["findings"], reply, by="device", question="is the model stable?")
    ids = [f["id"] for f in out["findings"]]
    assert ids[:2] == [deep["id"], flick["id"]] and flick["id"] not in out["left_out"]
    pinned = {f["id"]: f for f in out["findings"]}
    assert pinned[flick["id"]]["by"] == "rules" and pinned[flick["id"]]["title"] == flick["rules_title"]
    # The same reply with no question leaves the warning out, as before.
    plain = R.apply_words(res["findings"], reply, by="device")
    assert flick["id"] not in [f["id"] for f in plain["findings"]] and flick["id"] in plain["left_out"]


def test_the_prompt_keeps_the_question_verbatim_and_says_what_it_names(mesh_of):
    res = _review_with_flicker(mesh_of)
    flick = found(res, "flicker")[0]
    q = "are there any numerical instabilities in this run?"
    p = R.review_prompt(res["findings"], context="Portsmouth", question=q)
    assert f"The modeller asks: {q}" in p["system"]
    assert f"The findings that answer it are always pinned: {flick['id']}." in p["system"]
    assert "Portsmouth" in p["system"]
    # A question that names nothing still reaches the prompt, and promises no extra pin.
    plain = R.review_prompt(res["findings"], question="review my model results")
    assert "The modeller asks: review my model results" in plain["system"]
    assert "always pinned:" not in plain["system"]


class _Client:
    def __init__(self, text):
        self.text = text
        self.chat = self
        self.completions = self
        self.sent = None

    def create(self, **kw):
        self.sent = kw
        msg = type("M", (), {"content": self.text})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_model_words_with_a_client(mesh_of):
    res = _review_with_problems(mesh_of)
    neg = found(res, "negative")[0]
    client = _Client(json.dumps({"pins": [{"id": neg["id"], "variant": "b"}]}))
    out = R.model_words(res["findings"], client=client, model="m", provider="p")
    assert out["findings"][0]["title"] == "Negative depths" and out["model"] == "m via p"
    assert out["findings"][0]["reason"] == neg["rules_reason"]
    assert client.sent["messages"][1]["content"] == "Choose the pins."


def test_model_words_keeps_the_question_and_pins_what_it_names(mesh_of):
    res = _review_with_flicker(mesh_of)
    flick = found(res, "flicker")[0]
    deep = found(res, "deepest")[0]
    client = _Client(json.dumps({"pins": [{"id": deep["id"], "variant": "b"}]}))
    out = R.model_words(res["findings"], client=client, model="m", provider="p", question="is the model stable?")
    assert "is the model stable?" in client.sent["messages"][0]["content"]
    assert flick["id"] in [f["id"] for f in out["findings"]]
    assert flick["id"] not in out["left_out"]


# ── the faces: file, CLI and MCP ────────────────────────────────────────────


def test_review_file_offline(tmp_path):
    path = make_run(tmp_path / "run.p01.hdf", volume_pct=2.0)
    res = R.review_file(str(path), context=False)
    assert res["ok"] and res["by"] == "rules" and res["mesh"]["cells"] == N
    assert {c["check"] for c in res["checks"] if c["status"] == "flagged"} == {"mass_balance"}
    assert R.review_file(str(tmp_path / "missing.hdf"))["ok"] is False


def test_cli_mesh_review(tmp_path, capsys, monkeypatch):
    from aquascope import cli

    path = make_run(tmp_path / "run.p01.hdf", volume_pct=2.0)
    monkeypatch.setattr("sys.argv", ["aquascope", "mesh", "review", str(path), "--offline"])
    cli.main()
    out = capsys.readouterr().out
    assert "Volume error 2%" in out and "Volume balance" in out and "skipped" in out
    monkeypatch.setattr("sys.argv", ["aquascope", "mesh", "review", str(path), "--offline", "--json"])
    cli.main()
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] and data["findings"][0]["check"] == "mass_balance"
    # `aquascope mesh FILE` still answers as before.
    monkeypatch.setattr("sys.argv", ["aquascope", "mesh", str(path), "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["cells"] == N


def test_mcp_tool(tmp_path):
    from aquascope import mcp_server

    path = make_run(tmp_path / "run.p01.hdf", volume_pct=2.0)
    res = mcp_server.mesh_review(str(path), offline=True)
    assert res["ok"] and res["findings"][0]["check"] == "mass_balance"
    json.dumps(res)


def test_importing_the_review_stays_light():
    import subprocess
    import sys

    code = ("import sys; import aquascope.mesh_review; "
            "bad = [m for m in ('pandas', 'httpx', 'pyarrow', 'scipy') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


# ── the Explorer agrees with the package ────────────────────────────────────


def test_explorer_check_labels_and_worker_ops_match():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "explorer"
    js = (root / "src" / "mesh-review-core.js").read_text(encoding="utf-8")
    block = re.search(r"export const CHECK_LABELS = \{(.*?)\};", js, re.S).group(1)
    assert dict(re.findall(r'(\w+): "([^"]+)"', block)) == R.CHECKS
    sev = re.search(r"export const SEVERITY = \{(.*?)\n\};", js, re.S).group(1)
    assert re.findall(r"^  (\w+):", sev, re.M) == list(R.SEVERITIES)
    mesh_worker = (root / "src" / "mesh-worker.js").read_text(encoding="utf-8")
    for call in ("gauges_inside", "_review.review(", "review_prompt", "apply_words"):
        assert call in mesh_worker
    worker = (root / "worker.js").read_text(encoding="utf-8")
    assert '"mesh_review"' in re.search(r"const LITE_TYPES = new Set\(\[([^\]]*)\]\)", worker).group(1)
    for call in ("gather_floods", "gauge_series", "model_words"):
        assert f"_mr.{call}(" in worker


def test_a_count_is_said_under_its_own_threshold(mesh_of):
    # On the device the review read "8 cells ... (< 0.0269 m)": both numbers are facts, but the 8 cells are those
    # under 0.03 m and 0.0269 m is one cell's largest depth (#590). Now the words are the package's; the older
    # lock still catches the wrong bound.
    d = clean_depths()
    for r in (1, 2, 3):
        d[3, cell(r, WET_COLS)] = 0.02
    m = mesh_of(d)
    res = R.review(m, context=clean_context(m))
    film = found(res, "tiny")[0]
    ids = {x["label"]: x["id"] for x in film["facts"]}
    largest, count, limit = ids["Largest depth"], ids["Cells wetted only by a film"], ids["A film is shallower than"]
    assert next(x for x in film["facts"] if x["id"] == limit)["shown"] == "0.03"
    got, err = R._locked(film, "Only a film", f"{{{count}}} cells are wetted only by a film (< {{{largest}}} m).")
    assert got is None and "the count is of those under 0.03" in err
    for _v, (title, reason) in R.review_variants(film).items():
        assert "3 cells in all are like it" in reason or reason.startswith("This cell never gets deeper")
        assert "under 0.03" in reason
