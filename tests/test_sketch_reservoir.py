"""Draw a reservoir (#662): level-pool routing, checked against NRCS NEH 630 Chapter 17 Example 17-2, the storage
from the terrain inside the area, the dam on the river, and the reservoir inside Route it."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aquascope import floodsim as fs
from aquascope import rivers, routing
from aquascope.sketch import reservoir as rv

# ── the method, against the book ────────────────────────────────────────────


def test_level_pool_reproduces_neh_630_chapter_17_example_17_2():
    """Table 17-12's inflow through Table 17-11's working curve, the book's steps (0.5 day, then 0.1 day from 4.5 to
    6.0 days, then 0.5 day): every outflow within 1 ft3/s of the book's (which rounds each step to a whole ft3/s),
    the peak 364 ft3/s on day 5.3 as in the book."""
    ex = rv.NEH_17_2
    res = rv.level_pool(ex["inflow_cfs"], ex["storage_cfs_days"], ex["outflow_cfs"], times=ex["time_days"])
    got = res["outflow"]
    book = np.asarray(ex["outflow_book_cfs"], dtype=float)
    diff = np.abs(got - book)
    assert diff.max() < 1.0
    assert diff.max() == pytest.approx(rv.CHECK["max_abs_diff_cfs"], abs=0.005)
    j = int(np.argmax(got))
    assert ex["time_days"][j] == rv.CHECK["peak_day"] and round(got[j]) == rv.CHECK["book_peak_cfs"]
    assert round(float(got[j]), 1) == rv.CHECK["peak_cfs"]
    # Mass: what came in is what went out plus what is still stored.
    t = np.asarray(ex["time_days"])
    vin = np.sum((np.asarray(ex["inflow_cfs"])[1:] + np.asarray(ex["inflow_cfs"])[:-1]) / 2 * np.diff(t))
    vout = np.sum((got[1:] + got[:-1]) / 2 * np.diff(t))
    assert vin == pytest.approx(vout + res["storage"][-1], rel=1e-9)


def test_level_pool_with_a_linear_reservoir_is_the_trapezoidal_recursion():
    """S = K O, a linear reservoir: the storage-indication step is then exactly
    O2 = ((2K - dt) O1 + dt (I1 + I2)) / (2K + dt) (Chow, Maidment and Mays 1988, section 8.2's equation, S linear)."""
    k, dt = 5.0, 1.0
    inflow = [0, 40, 100, 160, 120, 80, 40, 10, 0, 0, 0, 0]
    o = np.linspace(0, 1000, 11)
    got = rv.level_pool(inflow, k * o, o, dt=dt)["outflow"]
    want = [0.0]
    for j in range(len(inflow) - 1):
        want.append(((2 * k - dt) * want[-1] + dt * (inflow[j] + inflow[j + 1])) / (2 * k + dt))
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_level_pool_refuses_a_bad_relation():
    with pytest.raises(ValueError):
        rv.level_pool([0, 1, 2], [0, 0, 1], [0, 1, 2], dt=1)
    with pytest.raises(ValueError):
        rv.level_pool([0, 1, 2], [0, 1, 2], [0, 1], dt=1)
    with pytest.raises(ValueError):
        rv.level_pool([0, 1, 2], [0, 1, 2], [0, 1, 2])


def test_the_spillway_is_a_broad_crested_weir():
    q = rv.spillway([10.0, 11.0, 12.0], crest=11.0, width_m=20.0)
    assert q[0] == 0 and q[1] == 0
    assert q[2] == pytest.approx(fs.WEIR_C * 20.0 * 1.0 ** 1.5)
    assert fs.WEIR_C == pytest.approx((2 / 3) ** 1.5 * math.sqrt(9.81), rel=1e-3)
    assert rv.spillway([13.0], 11.0, 10.0, coefficient=1.5)[0] == pytest.approx(1.5 * 10 * 2 ** 1.5)


# ── the storage from the terrain ────────────────────────────────────────────


def _bowl(zoom=14, nx=60, ny=60, lon=-3.4, lat=52.5, depth=None):
    x0, y0 = fs._world_px(lon, lat, zoom)
    win = fs.window(int(x0), int(y0), nx, ny, zoom)
    z = np.full((ny, nx), 100.0) if depth is None else depth
    return win, z


def test_a_flat_floor_holds_its_area_times_the_depth():
    win, z = _bowl()
    w, s, e, n = win["bounds"]
    ring = [[w + 0.2 * (e - w), s + 0.2 * (n - s)], [w + 0.8 * (e - w), s + 0.2 * (n - s)],
            [w + 0.8 * (e - w), s + 0.8 * (n - s)], [w + 0.2 * (e - w), s + 0.8 * (n - s)]]
    z = z.copy()
    z[0, 0] = 90.0   # outside the area: not counted
    cur = rv.stage_storage(z + np.where(np.arange(z.size).reshape(z.shape) == 0, 0, 0), win, ring, top=110.0,
                           n_levels=11)
    assert cur["bed_m"] == 100.0
    a = cur["area_drawn_m2"]
    # one level above the floor: every cell is under water
    assert cur["storage_m3"][-1] == pytest.approx(a * 10.0, rel=1e-9)
    assert cur["area_m2"][-1] == pytest.approx(a, rel=1e-9)
    assert cur["edge_below_pct"][-1] == pytest.approx(100.0)
    # the area is the cells inside: within a cell's width round the edge of the drawn box
    side = 0.6 * win["nx"] * cur["cell_m"]
    assert a == pytest.approx(side * side, rel=0.06)


def test_a_valley_fills_from_its_lowest_point():
    win, _ = _bowl()
    ny, nx = 60, 60
    cols = np.arange(nx)
    z = np.tile(100.0 + 0.5 * np.abs(cols - nx // 2), (ny, 1))   # a V across the window, 0.5 m a cell
    w, s, e, n = win["bounds"]
    ring = [[w, s], [e, s], [e, n], [w, n]]
    cur = rv.stage_storage(z, win, ring, top=105.0, n_levels=11)
    assert cur["bed_m"] == 100.0 and cur["storage_m3"][0] == 0
    assert np.all(np.diff(cur["storage_m3"]) > 0) and np.all(np.diff(cur["area_m2"]) >= 0)
    # at 5 m the water is 19 cells wide (|c - 30| * 0.5 < 5): 19 columns of 60 cells
    assert cur["area_m2"][-1] == pytest.approx(19 * 60 * cur["cell_m"] ** 2, rel=0.02)
    with pytest.raises(ValueError):
        rv.stage_storage(z, win, ring, top=99.0)


def test_the_terrain_window_covers_the_area_at_the_finest_zoom_it_can():
    ring = [[-3.47, 52.47], [-3.40, 52.47], [-3.40, 52.50], [-3.47, 52.50]]
    win = rv.terrain_for(ring)
    assert win["zoom"] == 14 and win["nx"] * win["ny"] <= rv.MAX_CELLS
    w, s, e, n = win["bounds"]
    assert w <= -3.47 and e >= -3.40 and s <= 52.47 and n >= 52.50
    big = [[-4.5, 51.5], [-2.0, 51.5], [-2.0, 53.0], [-4.5, 53.0]]
    assert rv.terrain_for(big)["zoom"] < 14


# ── the dam on the river, and the reservoir inside Route it ─────────────────

TOY = [("H1", "A", 3600.0), ("H2", "B", 7200.0), ("A", "B", 10800.0), ("B", "C", 18000.0), ("C", None, 3600.0)]
IDS = {name: 230000100 + i for i, (name, _, _) in enumerate(TOY)}


@pytest.fixture
def toy(monkeypatch):
    names = [t[0] for t in TOY]
    ids = [IDS[n] for n in names]
    ds = [IDS[d] if d else -1 for _, d, _ in TOY]
    net = rivers._Network(231, ids, ds, [100e6, 50e6, 80e6, 60e6, 40e6], [8.0, 8.1, 8.2, 8.3, 8.4],
                          [47.0, 47.1, 47.0, 47.1, 47.2])
    monkeypatch.setattr(rivers, "_network_for", lambda rid, lat=None, lon=None: net)
    monkeypatch.setattr(rivers, "_network", lambda vpu: net)
    k = np.array([t[2] for t in TOY])
    x = np.full(len(TOY), 0.36)
    ds_index = np.array([names.index(d) if d else -1 for _, d, _ in TOY])
    monkeypatch.setattr(routing, "_params", lambda n: (k, x, ds_index))
    return net


def test_the_dam_is_at_the_downstream_end_of_the_biggest_river_in_the_lake(toy):
    ring = [[8.0, 47.0], [8.2, 47.0], [8.2, 47.2], [8.0, 47.2]]
    reaches = {
        IDS["H1"]: {"order": 1, "lines": [[(8.05, 47.05), (8.08, 47.08)]]},
        IDS["A"]: {"order": 3, "lines": [[(8.08, 47.08), (8.12, 47.1)]]},
        IDS["B"]: {"order": 3, "lines": [[(8.12, 47.1), (8.18, 47.15), (8.3, 47.2)]]},
        IDS["C"]: {"order": 4, "lines": [[(8.3, 47.2), (8.5, 47.3)]]},
    }
    p = rv.plan(ring, reaches=reaches)
    assert p["ok"] and p["dam"]["river_id"] == IDS["B"] and p["dam"]["order"] == 3
    assert p["dam"]["downstream_id"] == IDS["C"]
    assert (p["dam"]["lon"], p["dam"]["lat"]) == (8.18, 47.15)   # the last point of B inside, nearest C
    assert p["river_reaches"] == sorted([IDS["A"], IDS["B"]])
    assert rv.plan([[9, 48], [9.1, 48], [9.1, 48.1]], reaches=reaches)["ok"] is False


def test_route_network_passes_a_reach_through_a_function(toy):
    k, x, ds_index = routing._params(toy)
    q_in = routing.hydrograph(100.0, 6, 40)
    half = routing.route_network([2, 3, 4], ds_index, k, x, 40, outflow={2: q_in}, through={3: lambda q: q / 2})
    full = routing.route_network([2, 3, 4], ds_index, k, x, 40, outflow={2: q_in})
    assert half[3] == pytest.approx(full[3] / 2)
    assert half[4] == pytest.approx(full[4] / 2)   # linear: half in, half out
    assert half[2] == pytest.approx(full[2])


def _flat_lake(monkeypatch):
    """A lake about 3 km square (30 cells of about 105 m) with a flat floor at 100 m, walls round it at 200 m:
    storage = area x depth."""
    zoom = 10
    x0, y0 = fs._world_px(8.05, 47.15, zoom)
    win = fs.window(int(x0), int(y0), 40, 40, zoom)
    z = np.full((40, 40), 200.0)
    z[5:35, 5:35] = 100.0
    # the outline runs through the walls' cells (4 and 35), so the shore is the walls, as a reader draws a lake
    lon, lat = fs.cell_lonlat(win, np.array([3.1, 35.9]), np.array([3.1, 35.9]))
    ring = [[lon[0], lat[1]], [lon[1], lat[1]], [lon[1], lat[0]], [lon[0], lat[0]]]
    return win, z, ring


def test_a_reservoir_flattens_the_flood_and_routes_its_outflow_on(toy, monkeypatch):
    win, z, ring = _flat_lake(monkeypatch)
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    res = rv.route_reservoir(ring, z=z, window=win, dam=dam, crest_m=5.0, top_m=12.0, width_m=10.0,
                             peak_cms=600.0, duration_h=12, from_river_id=IDS["A"], hours=200)
    assert res["ok"]
    r = res["reservoir"]
    assert r["peak_out_cms"] < r["peak_in_cms"] and 0 < r["reduction_pct"] < 100 and r["delay_h"] > 0
    assert r["without"]["peak_at_dam_cms"] == pytest.approx(r["peak_in_cms"], rel=0.01)
    # downstream of the dam the wave is the reservoir's outflow routed on: lower than without it
    rows = {row["river_id"]: row for row in res["reaches"]}
    assert rows[IDS["C"]]["peak_cms"] < r["without"]["furthest_peak_cms"]
    # mass: what came in left or is still held above the crest (the series have three significant figures)
    qi, qo = np.asarray(r["inflow_cms"]), np.asarray(r["outflow_cms"])
    vin = np.sum((qi[1:] + qi[:-1]) / 2) * 3600
    vout = np.sum((qo[1:] + qo[:-1]) / 2) * 3600
    assert vout + r["stored_end_m3"] == pytest.approx(vin, rel=0.01) and r["flood_storage_m3"] > 0
    # the highest level: the flat floor's storage over the crest is area x head
    a = float(r["curve"]["area_m2"][-1])
    assert r["max_depth_m"] == pytest.approx(5.0 + r["flood_storage_m3"] / a, abs=0.05)
    assert not r["overtopped"] and r["edge_below_top_pct"] == 0
    assert "NEH" in r["check"] and res["message"].startswith("A reservoir of")
    json.dumps(res)


def test_a_small_spillway_overtops_the_dam(toy, monkeypatch):
    win, z, ring = _flat_lake(monkeypatch)
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    res = rv.route_reservoir(ring, z=z, window=win, dam=dam, crest_m=5.0, top_m=5.5, width_m=1.0,
                             peak_cms=5000.0, duration_h=48, from_river_id=IDS["B"], hours=120)
    assert res["reservoir"]["overtopped"] is True
    assert any("overtopped" in n for n in res["notes"])


def test_a_storm_can_fill_the_reservoir(toy, monkeypatch):
    win, z, ring = _flat_lake(monkeypatch)
    monkeypatch.setattr(rivers, "_VPU_BBOX", {231: (7.0, 46.0, 9.0, 48.0)})
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    storm = {"polygon": [[7.95, 46.95], [8.15, 46.95], [8.15, 47.15], [7.95, 47.15]], "rain_mm": 80,
             "duration_h": 6, "runoff_coefficient": 0.4}
    res = rv.route_reservoir(ring, z=z, window=win, dam=dam, crest_m=5.0, top_m=12.0, width_m=10.0, storm=storm,
                             hours=200)
    assert res["ok"] and res["reservoir"]["peak_out_cms"] < res["reservoir"]["peak_in_cms"]
    assert "storm" in res["reservoir"]["what"]
    w = res["reservoir"]["without"]
    assert w["furthest_river_id"] == IDS["C"] and w["furthest_peak_with_cms"] < w["furthest_peak_cms"]


def test_a_flood_that_misses_the_dam_says_so(toy, monkeypatch):
    win, z, ring = _flat_lake(monkeypatch)
    dam = {"river_id": IDS["A"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["B"]}
    res = rv.route_reservoir(ring, z=z, window=win, dam=dam, crest_m=5.0, top_m=12.0, width_m=10.0,
                             peak_cms=100.0, duration_h=6, from_river_id=IDS["H2"], hours=60)
    assert res["ok"] is False and "does not reach the dam" in res["error"]


def test_the_reservoir_and_flood_it_read_the_same_design_flow(toy, monkeypatch):
    """The same reach and the same T give the same m3/s in Draw a reservoir and Flood it: both read
    floodsim.design_flow's Log-Pearson III curve on GEOGLOWS's annual maxima (#664 follow-up)."""
    years = list(range(1940, 2025))
    rng = np.random.default_rng(11)
    am = {rid: np.exp(rng.normal(np.log(50.0 * (i + 1)), 0.35, len(years))) for i, rid in enumerate(IDS.values())}
    monkeypatch.setattr(fs, "annual_maxima", lambda rids, fetch=None: (years, {int(r): am[int(r)] for r in rids
                                                                              if int(r) in am}))
    win, z, ring = _flat_lake(monkeypatch)
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    reaches = {IDS["A"]: {"order": 3, "lines": [[(8.2, 47.0), (8.25, 47.05)]]},
               IDS["B"]: {"order": 3, "lines": [[(8.25, 47.05), (8.3, 47.1)]]},
               IDS["C"]: {"order": 4, "lines": [[(8.3, 47.1), (8.4, 47.2)]]}}
    for t in (10, 100):
        res = rv.route_reservoir(ring, z=z, window=win, dam=dam, crest_m=5.0, top_m=12.0, width_m=10.0,
                                 return_period=t, duration_h=24, hours=120)
        assert res["ok"]
        plan = fs.plan_window(IDS["B"], net=toy, reaches=reaches, zoom=12, km=20)
        curve = (plan["reaches"].get(IDS["B"]) or plan["reaches"][str(IDS["B"])])["curve"]
        flood_it = fs.flow_for(curve, t)
        design = fs.design_flow(IDS["B"], t)
        assert res["inflow"]["peak_cms"] == pytest.approx(flood_it, abs=0.01)
        assert res["reservoir"]["peak_in_cms"] == pytest.approx(flood_it, rel=0.01)
        assert design["flow_cms"] == pytest.approx(flood_it, abs=0.001)
        assert design["words"] in res["reservoir"]["what"] and curve["method"] == design["method"]


# ── the CLI and MCP faces ────────────────────────────────────────────────────


def test_cli_prints_the_check_against_the_book(capsys, monkeypatch):
    from aquascope import cli

    monkeypatch.setattr("sys.argv", ["aquascope", "route", "reservoir", "--check"])
    cli.main()
    out = capsys.readouterr().out
    assert "Example 17-2" in out and "Largest difference 0.51 ft3/s; peak 364.2 ft3/s on day 5.3" in out


def test_cli_and_mcp_route_a_drawn_reservoir(toy, monkeypatch, capsys):
    from aquascope import cli
    from aquascope.mcp_server import route_reservoir

    win, z, ring = _flat_lake(monkeypatch)
    dam = {"river_id": IDS["B"], "order": 3, "lon": 8.1, "lat": 47.1, "downstream_id": IDS["C"]}
    monkeypatch.setattr(rv, "plan", lambda polygon, reaches=None: {"ok": True, "dam": dam, "window": win})
    monkeypatch.setattr(fs, "terrain_window", lambda x0, y0, nx, ny, zoom, fetch=None: {"z": z})
    res = route_reservoir(ring, crest_m=5.0, top_m=12.0, width_m=10.0, peak_cms=600.0, duration_h=12,
                          from_river_id=IDS["A"], hours=200)
    assert res["ok"] and "frames" not in res and "curve" not in res["reservoir"]
    monkeypatch.setattr("sys.argv", ["aquascope", "route", "reservoir", "--polygon", json.dumps(ring), "--crest", "5",
                                     "--top", "12", "--width", "10", "--peak", "600", "--duration", "12",
                                     "--from", str(IDS["A"]), "--hours", "200"])
    cli.main()
    out = capsys.readouterr().out
    assert "At the dam: in " in out and "% lower" in out and "NEH" in out


# ── Ask the map's draw_reservoir ────────────────────────────────────────────


def test_the_map_reads_a_reservoir_and_checks_it():
    from aquascope import map_commands as mc

    assert mc.RESERVOIR_DEFAULTS["crest_m"] == rv.DEFAULTS["crest_m"]
    assert mc.RESERVOIR_DEFAULTS["rise_m"] == rv.DEFAULTS["rise_m"]
    assert mc.RESERVOIR_DEFAULTS["width_m"] == rv.DEFAULTS["width_m"]
    for text, want in [
        ("draw a reservoir here", {"area": "drawn", "crest_m": 10.0, "top_m": 15.0, "width_m": 30.0,
                                   "return_period": 100.0}),
        ("draw a reservoir with a 40 m spillway at 12 m", {"crest_m": 12.0, "top_m": 17.0, "width_m": 40.0}),
        ("put a dam here and route the 100-year flood through it", {"return_period": 100.0}),
        ("draw a reservoir and send the wave through it", {"inflow": "wave"}),
        ("build a dam 25 m high with a 60 m wide spillway and route 300 m3/s through it",
         {"top_m": 25.0, "width_m": 60.0, "peak_cms": 300.0}),
    ]:
        res = mc.parse_command(text, today="2026-10-27")
        assert res["matched"], text
        got = res["actions"][0]
        assert got["type"] == "draw_reservoir" and {k: got[k] for k in want} == want, text
    assert mc.validate_action({"type": "draw_reservoir", "crest_m": 20, "top_m": 15})[1].endswith("above its crest_m")
    assert "3 to 200" in mc.validate_action({"type": "draw_reservoir", "polygon": [[1, 2], [2, 3]]})[1]
    ok, why = mc.validate_action({"type": "draw_reservoir", "inflow": "wave", "return_period": 10})
    assert why is None and "return_period" not in ok
    assert mc.describe_action(mc.parse_command("draw a reservoir here")["actions"][0]) == \
        ("A reservoir on the lake drawn on the map (crest 10 m, top 15 m, spillway 30 m wide), and route the "
         "100-year flood through it")
    assert "draw_reservoir" in mc.model_prompt()["system"]
