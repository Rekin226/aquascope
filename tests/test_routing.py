"""Route it (#656): Muskingum routing down GEOGLOWS's network, its faces and the map action."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aquascope import map_commands as mc
from aquascope import rivers, routing

# ── the Muskingum equation ──────────────────────────────────────────────────


@pytest.mark.parametrize("k", [20.0, 754.0, 3600.0, 17430.0, 95937.0])
def test_every_reach_gets_a_plan_with_no_negative_coefficient(k):
    n, m = routing.plan([k], [0.36], 3600)
    n, m = int(n[0]), int(m[0])
    c1, c2, c3 = routing.coefficients(k / n, 0.36, 3600 / m)
    assert min(c1, c2, c3) >= -1e-12
    assert 2 * 0.36 * k / n <= 3600 / m + 1e-9 <= 2 * 0.64 * k / n + 2e-9


def test_the_plan_is_the_fewest_steps():
    # dt = k: one step, whole. A short reach takes sub-steps, a long one sub-reaches (river-route's rule).
    assert [int(v[0]) for v in routing.plan([3600.0], [0.36])] == [1, 1]
    n, m = routing.plan([20.0], [0.36])
    assert int(n[0]) == 1 and int(m[0]) == math.ceil(3600 / (2 * 0.64 * 20))
    n, m = routing.plan([95937.0], [0.36])
    assert int(m[0]) == 1 and int(n[0]) == math.ceil(2 * 0.36 * 95937 / 3600)


@pytest.mark.parametrize(("k", "n", "m"), [(3600.0, 1, 1), (754.0, 1, 4), (20.0, 1, 141), (17430.0, 4, 1)])
def test_folded_coefficients_conserve_mass(k, n, m):
    c1, c2, c3, c4 = routing.folded(k, 0.36, 3600, n, m)
    assert c1 + c2 + c3 == pytest.approx(1.0, abs=1e-12)
    assert c4 == pytest.approx(c1 + c2, abs=1e-12)
    assert min(c1, c2, c3, c4) >= -1e-12


def test_one_step_unfolded_matches_the_textbook():
    # McCarthy's coefficients for dt/k = 1, x = 0.2: c1 = 0.6/2.6, c2 = 1.4/2.6, c3 = 0.6/2.6.
    c1, c2, c3 = routing.coefficients(3600, 0.2, 3600)
    assert (c1, c2, c3) == pytest.approx((0.6 / 2.6, 1.4 / 2.6, 0.6 / 2.6))


def test_a_reach_keeps_the_volume_delays_the_peak_and_flattens_it():
    n_steps = 400
    q_in = routing.hydrograph(1000.0, 24, n_steps)
    k = np.array([20000.0])
    x = np.array([0.2])
    out = routing.route_network([0, 1], np.array([1, -1]), np.array([0.0, 20000.0]), np.array([0.36, 0.2]),
                                n_steps, outflow={0: q_in})[1]
    assert out.sum() == pytest.approx(q_in.sum(), rel=1e-6)
    assert int(np.argmax(out)) > int(np.argmax(q_in))
    assert out.max() < q_in.max()
    assert out.min() >= -1e-9
    # Travel time: the centroid moves on by k (a Muskingum reach delays the centre of mass by exactly k).
    t = np.arange(n_steps + 1)
    lag_h = (t * out).sum() / out.sum() - (t * q_in).sum() / q_in.sum()
    assert lag_h == pytest.approx(float(k[0]) / 3600, abs=0.05)
    assert x[0] == 0.2


def test_steady_flow_stays_steady():
    n_steps = 48
    q = np.full(n_steps + 1, 250.0)
    out = routing.route_network([0, 1, 2], np.array([1, 2, -1]), np.array([0.0, 5000.0, 30000.0]),
                                np.array([0.36, 0.36, 0.36]), n_steps, outflow={0: q}, q0={1: 250.0, 2: 250.0})
    assert np.allclose(out[2], 250.0)


def test_lateral_runoff_is_added_and_routed():
    n_steps = 100
    lat = np.where(np.arange(n_steps) < 6, 10.0, 0.0)
    out = routing.route_network([0], np.array([-1]), np.array([7200.0]), np.array([0.36]), n_steps, lateral={0: lat})
    assert out[0][1:].sum() == pytest.approx(lat.sum(), rel=1e-6)


def test_hydrograph_shapes():
    tri = routing.hydrograph(300.0, 24, 48)
    block = routing.hydrograph(300.0, 24, 48, shape="block")
    assert tri.max() == pytest.approx(300.0) and int(np.argmax(tri)) == 8
    assert block[1:25].tolist() == [300.0] * 24 and block[25] == 0 and block[0] == 0
    with pytest.raises(ValueError):
        routing.hydrograph(300.0, 24, 48, shape="square")


# ── a toy network, as GEOGLOWS's tables have it ─────────────────────────────

# H1 -> A -> B -> C (outlet), H2 -> B: row order upstream first, like rapid_connect.csv.
TOY = [("H1", "A", 3600.0), ("H2", "B", 7200.0), ("A", "B", 10800.0), ("B", "C", 18000.0), ("C", None, 3600.0)]
TOY_IDS = {name: 230000100 + i for i, (name, _, _) in enumerate(TOY)}


@pytest.fixture
def toy(monkeypatch):
    names = [t[0] for t in TOY]
    ids = [TOY_IDS[n] for n in names]
    ds = [TOY_IDS[d] if d else -1 for _, d, _ in TOY]
    net = rivers._Network(231, ids, ds, [100e6, 50e6, 80e6, 60e6, 40e6], [8.0, 8.1, 8.2, 8.3, 8.4],
                          [47.0, 47.1, 47.0, 47.1, 47.2])
    monkeypatch.setattr(rivers, "_network_for", lambda rid, lat=None, lon=None: net)
    monkeypatch.setattr(rivers, "_network", lambda vpu: net)
    k = np.array([t[2] for t in TOY])
    x = np.full(len(TOY), 0.36)
    ds_index = np.array([names.index(d) if d else -1 for _, d, _ in TOY])
    monkeypatch.setattr(routing, "_params", lambda n: (k, x, ds_index))
    return net


def test_route_follows_the_path_and_reports_each_reach(toy):
    res = routing.route(TOY_IDS["A"], peak_cms=600.0, duration_h=12, along=False)
    assert res["ok"] and [r["river_id"] for r in res["reaches"]] == [TOY_IDS["A"], TOY_IDS["B"], TOY_IDS["C"]]
    first, last = res["reaches"][0], res["reaches"][-1]
    assert first["peak_cms"] == 600.0 and first["attenuation_pct"] == 0.0
    assert last["peak_h"] > first["peak_h"] and 0 < last["attenuation_pct"] < 100
    assert last["front_h"] <= last["peak_h"]
    # the time routed covers the path: the peak passes the outlet before the end
    assert last["peak_h"] < res["hours"]
    assert res["inflow"]["volume_m3"] == pytest.approx(600.0 * 12 * 3600 / 2, rel=0.01)
    fr = res["frames"]
    assert fr["ids"] == [r["river_id"] for r in res["reaches"]]
    assert len(fr["share"]) == len(fr["t_h"]) and all(0 <= v <= 100 for row in fr["share"] for v in row)
    assert max(row[0] for row in fr["share"]) == 100
    assert "linear" in res["notes"][0] and res["modelled"] is True
    json.dumps(res)


def test_a_release_is_a_block(toy):
    res = routing.route(TOY_IDS["B"], peak_cms=100.0, duration_h=2, shape="block", hours=60, along=False)
    assert res["hours"] == 60 and res["inflow"]["shape"] == "block"
    assert res["reaches"][-1]["peak_cms"] < 100.0


def test_route_needs_a_size_and_holds_its_bounds(toy):
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], along=False)
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], peak_cms=-5, along=False)
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], peak_cms=5, duration_h=9999, along=False)


def test_the_return_period_flow_is_flood_its_design_flow(toy, monkeypatch):
    """Route it's T-year flow is floodsim.design_flow: Log-Pearson III on GEOGLOWS's annual maxima, as Flood it."""
    from aquascope import floodsim

    years = list(range(1940, 2025))
    am = np.exp(np.random.default_rng(5).normal(np.log(400), 0.4, len(years)))
    monkeypatch.setattr(floodsim, "annual_maxima", lambda rids, fetch=None: (years, {int(r): am for r in rids}))
    res = routing.route(TOY_IDS["A"], return_period=10, along=False)
    want = floodsim.flow_for(floodsim.flow_curve(years, am), 10)
    assert res["inflow"]["peak_cms"] == pytest.approx(want, abs=0.01)
    d = res["inflow"]["design"]
    assert d["method"] == "Log-Pearson III" and d["n_years"] == 85 and d["words"] in res["inflow"]["source"]
    assert "10-year flow" in res["inflow"]["source"] and "flows" in res["licence"]
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], return_period=1000, along=False)


def test_places_on_the_path_get_the_wave_where_they_stand(toy):
    res = routing.route(TOY_IDS["A"], peak_cms=600.0, along=False)
    res["geometry"] = {"type": "LineString", "coordinates": [[8.0, 47.0], [8.1, 47.0], [8.2, 47.0], [8.3, 47.0]]}
    for r, km in zip(res["reaches"], (7.6, 15.2, 22.8)):
        r["along_km"] = km
    places = [{"name": "Town", "lat": 47.001, "lon": 8.15}, {"name": "Far", "lat": 48, "lon": 8}]
    got = routing.places_on_path(res, places)
    assert [p["name"] for p in got] == ["Town"]
    assert got[0]["river_id"] == TOY_IDS["B"] and got[0]["peak_h"] == res["reaches"][1]["peak_h"]


def test_a_storm_runs_off_and_is_routed(toy, monkeypatch):
    monkeypatch.setattr(rivers, "_VPU_BBOX", {231: (7.0, 46.0, 9.0, 48.0)})
    ring = [[7.95, 46.95], [8.15, 46.95], [8.15, 47.15], [7.95, 47.15]]   # over H1 and H2
    res = routing.route_storm(ring, rain_mm=50, duration_h=6, runoff_coefficient=0.4)
    assert res["ok"] and res["storm"]["n_reaches"] == 2
    assert res["storm"]["runoff_m3"] == pytest.approx(0.4 * 0.05 * 150e6, rel=0.01)
    outlet = res["outlets"][0]
    assert outlet["river_id"] == TOY_IDS["C"] and outlet["peak_cms"] > 0
    with pytest.raises(ValueError):
        routing.route_storm(ring, rain_mm=50, runoff_coefficient=1.5)


def test_a_storm_gives_the_card_its_hydrographs_and_balances_its_water(toy, monkeypatch):
    """Draw a storm (#670): the card's two hydrographs are the reaches' own routed series, the box covers the reaches
    the wave reaches, and every m3 of runoff leaves by the outlets once the wave has passed (Muskingum keeps mass)."""
    monkeypatch.setattr(rivers, "_VPU_BBOX", {231: (7.0, 46.0, 9.0, 48.0)})
    ring = [[7.95, 46.95], [8.15, 46.95], [8.15, 47.15], [7.95, 47.15]]
    res = routing.route_storm(ring, rain_mm=50, duration_h=6, runoff_coefficient=0.4, hours=400)
    hyd = {h["role"]: h for h in res["hydrographs"]}
    assert hyd["biggest"]["river_id"] == res["biggest"]["river_id"]
    assert max(hyd["biggest"]["cms"]) == pytest.approx(res["biggest"]["peak_cms"], rel=0.01)
    assert len(hyd["biggest"]["cms"]) == res["hours"] + 1 and hyd["biggest"]["dt_h"] == 1.0
    assert res["storm"]["out_m3"] == pytest.approx(res["storm"]["runoff_m3"], rel=0.01)
    assert res["storm"]["out_pct"] == pytest.approx(100.0, abs=1.0)
    w, s, e, n = res["bbox"]
    assert w <= 8.0 and e >= 8.2 and s <= 47.0 and n >= 47.1
    assert res["polygon"] == ring
    json.dumps(res)


def test_the_map_reads_a_storm_and_checks_it():
    """Ask the map (#670): "50 mm of rain over this area in 6 hours" is a storm over the area drawn on the map, with
    route_storm's own defaults for what the words leave out."""
    import inspect

    defaults = inspect.signature(routing.route_storm).parameters
    assert mc.STORM_DEFAULTS == {k: defaults[k].default for k in ("duration_h", "runoff_coefficient")}
    for text, want in [
        ("50 mm of rain over this area in 6 hours", {"area": "drawn", "rain_mm": 50.0, "duration_h": 6.0}),
        ("draw a storm of 80 mm over 12 h", {"area": "drawn", "rain_mm": 80.0, "duration_h": 12.0}),
        ("rain 100 mm on Wales in 2 days with a runoff coefficient of 0.5",
         {"place": "Wales", "rain_mm": 100.0, "duration_h": 48.0, "runoff_coefficient": 0.5}),
        ("what if 120 mm of rain fell here", {"area": "drawn", "rain_mm": 120.0, "duration_h": 6.0}),
    ]:
        res = mc.parse_command(text, today="2026-10-27")
        assert res["matched"], text
        got = res["actions"][0]
        assert got["type"] == "draw_storm" and {k: got[k] for k in want} == want, text
    ok, why = mc.validate_action({"type": "draw_storm", "rain_mm": 50, "bbox": [7, 46, 8, 47]})
    assert why is None and ok["polygon"] == [[7, 46], [8, 46], [8, 47], [7, 47]]
    assert "needs rain_mm" in mc.validate_action({"type": "draw_storm", "area": "drawn"})[1]
    assert "runoff_coefficient" in mc.validate_action({"type": "draw_storm", "rain_mm": 5, "runoff_coefficient": 2})[1]
    assert "3 to 200" in mc.validate_action({"type": "draw_storm", "rain_mm": 5, "polygon": [[7, 46], [8, 46]]})[1]
    assert mc.describe_action(mc.parse_command("50 mm of rain over this area in 6 hours")["actions"][0]) == \
        "Rain 50 mm in 6 h over the area drawn on the map, 0.3 of it running off, and route it"
    assert "draw_storm" in mc.model_prompt()["system"]
    hits = {"wales": [{"geometry": {"coordinates": [-3.7, 52.3]}, "properties": {
        "name": "Wales", "osm_key": "place", "osm_value": "state", "country": "GB",
        "extent": [-5.3, 53.4, -2.6, 51.4]}}]}
    out = mc.resolve_actions([{"type": "draw_storm", "place": "Wales", "rain_mm": 100.0, "duration_h": 48.0,
                               "runoff_coefficient": 0.3}], fetch=lambda q, limit=5: hits.get(q.lower(), []))
    assert out["actions"] and len(out["actions"][0]["polygon"]) == 4


# ── the map's action (aquascope.map_commands) ────────────────────────────────

@pytest.mark.parametrize(("text", "expected"), [
    ("route a flood of 500 m3/s from Basel", {"type": "route", "place": "Basel", "peak_cms": 500.0,
                                               "duration_h": 24.0, "shape": "triangle"}),
    ("release 800 m3/s for 2 days from here", {"type": "route", "at": "here", "peak_cms": 800.0,
                                                "duration_h": 48.0, "shape": "block"}),
    ("route the 100-year flood down the Rhine from Basel", {"type": "route", "place": "Rhine at Basel",
                                                             "return_period": 100.0, "duration_h": 24.0,
                                                             "shape": "triangle"}),
    ("route a flood from here", {"type": "route", "at": "here", "return_period": 10.0, "duration_h": 24.0,
                                 "shape": "triangle"}),
])
def test_the_keyless_rules_read_a_route(text, expected):
    res = mc.parse_command(text, today="2026-10-27")
    assert res["matched"] and res["actions"] == [expected]
    assert res["said"][0].startswith("Route ")


def test_route_is_checked_like_every_action():
    ok, why = mc.validate_action({"type": "route", "lat": 47.5, "lon": 7.6, "peak_cms": 300, "return_period": 5})
    assert why is None and ok["peak_cms"] == 300 and "return_period" not in ok
    assert mc.validate_action({"type": "route", "peak_cms": 300})[1].startswith("route needs")
    assert "peak_cms" in mc.validate_action({"type": "route", "at": "here", "peak_cms": -3})[1]
    assert "shape" in mc.validate_action({"type": "route", "at": "here", "shape": "square"})[1]
    assert "route" in mc.action_schema()["properties"]["actions"]["items"]["properties"]["type"]["enum"]
    assert mc.describe_action({"type": "route", "at": "here", "return_period": 10, "duration_h": 48}) == \
        "Route the 10-year flood over 2 days from the selected river"


def test_a_place_resolves_to_a_point_to_start_from():
    hits = {"basel": [{"geometry": {"coordinates": [7.59, 47.56]},
                       "properties": {"name": "Basel", "osm_key": "place", "osm_value": "city", "country": "CH"}}]}

    def fetch(query, limit=5):
        return hits.get(query.lower(), [])

    out = mc.resolve_actions([{"type": "route", "place": "Basel", "peak_cms": 500.0, "duration_h": 24.0,
                               "shape": "triangle"}], fetch=fetch)
    assert out["actions"] and out["actions"][0]["lat"] == pytest.approx(47.56)


# ── the CLI and MCP faces ────────────────────────────────────────────────────


def test_cli_route_flood(toy, capsys, monkeypatch):
    from aquascope import cli

    monkeypatch.setattr("sys.argv", ["aquascope", "route", "flood", str(TOY_IDS["A"]), "--peak", "600",
                                     "--no-places"])
    cli.main()
    out = capsys.readouterr().out
    assert "A triangle of 600 m³/s over 24 h" in out


def test_mcp_route_flood(toy):
    from aquascope.mcp_server import route_flood

    res = route_flood(TOY_IDS["A"], peak_cms=600.0, places=False)
    assert res["ok"] and "frames" not in res and "geometry" not in res


def test_the_check_against_the_retrospective_meets_its_stated_tolerance():
    cases = routing.CHECK["cases"]
    assert len(cases) >= 3
    for c in cases:
        assert c["nse"] >= 0.85 and abs(c["peak_error_pct"]) <= 10 and abs(c["peak_timing_h"]) <= 12, c["reach"]
        assert c["nse"] > c["nse_no_routing"], c["reach"]


def test_the_retro_check_routes_what_it_reads(toy):
    # A fake retrospective: the upstream reach H1 sends a pulse, H2 a steady flow; the downstream reach C is what
    # this module itself routes from them, so the check scores it perfectly.
    n = 400
    t = np.arange(n)
    q_h1 = 50 + 400 * np.exp(-0.5 * ((t - 200) / 12.0) ** 2)
    q_h2 = np.full(n, 20.0)

    class Fake:
        def times(self):
            return np.datetime64("2020-01-01T00:00") + np.arange(n).astype("timedelta64[h]")

        def columns(self, rids, t0, t1):
            k, x, ds = routing._params(None)
            out = routing.route_network([0, 1, 2, 3, 4], ds, k, x, n - 1,
                                        outflow={0: q_h1, 1: q_h2}, q0={2: 50.0, 3: 70.0, 4: 70.0})
            series = {TOY_IDS["H1"]: q_h1, TOY_IDS["H2"]: q_h2, TOY_IDS["C"]: out[4]}
            return {r: series[r][t0:t1] for r in rids}

    res = routing.retro_check(TOY_IDS["H1"], TOY_IDS["C"], "2020-01-06", "2020-01-17", spin_up_h=0, retro=Fake(),
                              lateral="none", min_share=0.0)
    assert res["n_path"] == 3 and res["n_tributaries_read"] == 1
    assert res["routed"]["nse"] > 0.999 and abs(res["routed"]["peak_timing_h"]) <= 1
