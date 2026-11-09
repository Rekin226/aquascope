"""Drop a well (#660): aquascope.sketch.drawdown, its CLI verb, its MCP tool and the drop_well map action.

The checks are published numbers:

* Fetter, C.W. (2001), Applied Hydrogeology, 4th ed., the Theis worked example: K = 14.9 m/day, b = 20.1 m (so
  T = 299 m2/day as printed), S = 0.0051, Q = 2725 m3/day, r = 7.0 m, t = 1 day; printed: 5.7 m.
* Lohman, S.W. (1972), USGS Professional Paper 708, Table 5, p. 16: W(u), rows N = 1.0 and 2.0, four decimals.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aquascope import map_commands as mc
from aquascope.groundwater.aquifer import _well_function, theis_drawdown
from aquascope.sketch import drawdown as dd

ONE = [{"lon": 2.35, "lat": 48.85}]


def test_fetter_worked_example():
    f = dd.FETTER
    s = dd.drawdown_at(0.0, 45.0, [{"lon": 0.0, "lat": 45.0 + f["radius_m"] / 111_194.9, "rate_m3_day": 2725.0}],
                       days=1.0, transmissivity=299.0, storativity=0.0051)["drawdown_m"]
    assert round(s, 1) == 5.7                     # as printed
    assert s == pytest.approx(5.7269, abs=2e-3)   # the same inputs without the table's rounding of u


def test_pp708_table5():
    # the well function the map uses is the aquifer module's
    for n, row in dd.PP708_TABLE5.items():
        for k, printed in zip(range(-15, 1), row):
            assert float(_well_function(np.array(n * 10.0 ** k))) == pytest.approx(printed, abs=1.01e-4)


def test_check_passes():
    res = dd.check()
    assert res["ok"], res["rows"]
    assert len(res["rows"]) == 3
    assert "pubs.usgs.gov/pp/0708" in " ".join(res["sources"])


def test_one_engine_with_the_aquifer_module():
    # the map's drawdown is the aquifer module's Theis, point for point
    w = {"lon": 10.0, "lat": 50.0, "rate_m3_day": 1800.0}
    for metres in (5.0, 120.0, 3000.0):
        got = dd.drawdown_at(10.0, 50.0 + metres / 111_194.9, [w], days=12.0, transmissivity=250.0,
                             storativity=3e-4)
        want = float(theis_drawdown(250.0, 3e-4, 1800.0, got["wells"][0]["distance_m"], 12.0)[0])
        assert got["drawdown_m"] == pytest.approx(want, abs=1e-4)


def test_superposition_symmetry_and_reciprocity():
    sup = dd.check()["superposition"]
    assert sup["mirror_m"][0] == pytest.approx(sup["mirror_m"][1], abs=1e-4)
    assert sup["reciprocal_m"][0] == sup["reciprocal_m"][1]
    assert sup["together_m"] == pytest.approx(sup["alone_added_m"], abs=2e-4)
    # order does not matter
    a, b = {"lon": 0.0, "lat": 45.0, "rate_m3_day": 900.0}, {"lon": 0.01, "lat": 45.002, "rate_m3_day": 2500.0}
    one = dd.drawdown_map([a, b], days=5.0, image=False)
    two = dd.drawdown_map([b, a], days=5.0, image=False)
    assert sorted(w["at_well_m"] for w in one["wells"]) == sorted(w["at_well_m"] for w in two["wells"])


def test_map_shape_and_contours_are_circles_for_one_well():
    res = dd.drawdown_map(ONE, frames=True)
    assert res["ok"] and res["wells"][0]["rate_l_s"] == pytest.approx(11.57, abs=0.01)
    assert res["levels"][0] == dd.EDGE_M and len(res["levels"]) <= dd.MOST_LEVELS
    assert all(c["label"].endswith(" m") for c in res["contours"])
    # the 0.5 m line sits where Theis says, all the way round
    half = next(c for c in res["contours"] if c["level"] == 0.5)
    pts = np.array([p for line in half["lines"] for p in line])
    k = math.cos(math.radians(48.85))
    r = np.hypot((pts[:, 0] - 2.35) * 111_194.9 * k, (pts[:, 1] - 48.85) * 111_194.9)
    want = dd._reach(1000.0, 500.0, 1e-4, 30.0, 0.5)
    assert np.allclose(r, want, rtol=0.02)
    # the image, the bbox and the frames
    assert res["image"]["nx"] * res["image"]["ny"] == len(res["image"]["values"])
    w, s, e, n = res["bbox"]
    assert w < 2.35 < e and s < 48.85 < n
    days = [f["days"] for f in res["frames"]]
    assert days == sorted(days) and days[-1] == 30.0
    assert [f["max_m"] for f in res["frames"]] == sorted(f["max_m"] for f in res["frames"])
    json.dumps(res)


def test_defaults_ranges_and_their_source():
    assert dd.DEFAULTS["transmissivity_m2_day"] == 500.0 and dd.DEFAULTS["storativity"] == 1e-4
    lo, hi = dd.RANGES["storativity"]
    assert lo <= dd.DEFAULTS["storativity"] <= hi
    lo, hi = dd.RANGES["transmissivity_m2_day"]
    assert lo <= dd.DEFAULTS["transmissivity_m2_day"] <= hi
    assert "Water-Supply Paper 2220" in dd.RANGES_SOURCE


@pytest.mark.parametrize("bad", [
    [], [{"lon": 0}], [{"lon": 0, "lat": 0, "rate_m3_day": -5}], [{"lon": 500, "lat": 0}], [[1, 2]] * 11,
])
def test_bad_wells(bad):
    with pytest.raises(ValueError):
        dd.drawdown_map(bad)


def test_bad_aquifer():
    with pytest.raises(ValueError):
        dd.drawdown_map(ONE, transmissivity=0)
    with pytest.raises(ValueError):
        dd.drawdown_map(ONE, storativity=True)


def test_map_command_limits_match_the_package():
    assert mc.WELL_LIMITS["rate_m3_day"] == dd.LIMITS["rate_m3_day"]
    assert mc.WELL_LIMITS["days"] == dd.LIMITS["days"]
    assert mc.WELL_LIMITS["transmissivity_m2_day"] == dd.LIMITS["transmissivity_m2_day"]
    assert mc.WELL_LIMITS["storativity"] == dd.LIMITS["storativity"]
    assert mc.WELL_DEFAULT_RATE == dd.DEFAULTS["rate_m3_day"]


@pytest.mark.parametrize("text,want", [
    ("drop a well here", {"type": "drop_well", "at": "center", "rate_m3_day": 1000.0}),
    ("drop a well at Lyon pumping 2000 m3/day", {"type": "drop_well", "place": "Lyon", "rate_m3_day": 2000.0}),
    ("drill a borehole near Lille pumping 20 l/s for 90 days",
     {"type": "drop_well", "place": "Lille", "rate_m3_day": 1728.0, "days": 90.0}),
    ("pump 1500 m3/day here for a year", {"type": "drop_well", "at": "center", "rate_m3_day": 1500.0,
                                          "days": 365.0}),
])
def test_drop_well_phrases(text, want):
    res = mc.parse_command(text, "2026-10-27")
    assert res["actions"] == [want], res
    assert mc.describe_action(res["actions"][0]).startswith("Drop a well at ")


def test_drop_well_validation():
    ok, err = mc.validate_action({"type": "drop_well", "lat": 45.0, "lon": 4.8, "rate_m3_day": 500})
    assert err is None and ok == {"type": "drop_well", "lat": 45.0, "lon": 4.8, "rate_m3_day": 500.0}
    _, err = mc.validate_action({"type": "drop_well", "at": "center", "storativity": 3})
    assert "storativity" in err
    assert "drop_well" in json.dumps(mc.action_schema())
    assert "drop_well" in mc.model_prompt()["system"]


def test_drop_well_resolves_a_place():
    def fetch(query):
        return [{"geometry": {"coordinates": [4.835, 45.758]},
                 "properties": {"name": "Lyon", "type": "city", "countrycode": "FR", "country": "France"}}]

    res = mc.resolve_actions([{"type": "drop_well", "place": "Lyon", "rate_m3_day": 2000.0}], fetch=fetch)
    a = res["actions"][0]
    assert a["lat"] == pytest.approx(45.758) and a["lon"] == pytest.approx(4.835) and a["label"].startswith("Lyon")


def test_cli_and_mcp(capsys, tmp_path):
    from aquascope.cli import main
    from aquascope.mcp_server import drop_well

    out = drop_well(lon=2.35, lat=48.85, rate_m3_day=1500.0, days=60)
    assert out["ok"] and out["wells"][0]["rate_m3_day"] == 1500.0 and "image" not in out
    assert "error" in drop_well()
    import sys
    gj = tmp_path / "cone.geojson"
    argv = sys.argv
    try:
        sys.argv = ["aquascope", "well", "--at", "2.35", "48.85", "--rate", "1500", "--days", "60",
                    "--geojson", str(gj)]
        main()
        sys.argv = ["aquascope", "well", "--check"]
        main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "draws the water down" in text and "Fetter" in text and "PP 708" in text
    fc = json.loads(gj.read_text(encoding="utf-8"))
    assert fc["features"][-1]["geometry"]["type"] == "Point"


def test_explorer_defaults_are_the_package_defaults():
    from pathlib import Path

    core = (Path(__file__).resolve().parents[1] / "explorer" / "src" / "well-core.js").read_text(encoding="utf-8")
    d = dd.DEFAULTS
    want = (f"export const DEFAULTS = {{ rate: {d['rate_m3_day']:g}, T: {d['transmissivity_m2_day']:g}, "
            f"S: 1e-4, days: {d['days']:g} }};")
    assert dd.DEFAULTS["storativity"] == 1e-4 and want in core
    html = (Path(__file__).resolve().parents[1] / "explorer" / "index.html").read_text(encoding="utf-8")
    assert 'id="wl-rate-new" type="number" min="1" step="any" inputmode="decimal" value="1000"' in html
