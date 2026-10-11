"""What if, on the map (#565): the scenario words, the walk downstream and the run, on a toy network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope import advanced as adv
from aquascope import rivers
from aquascope import whatif_map as wm

# ── the scenario in words ────────────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("20% less rain and 2 degrees warmer", {"dp_pct": -20.0, "dt_c": 2.0}),
    ("What if it rains 20% less?", {"dp_pct": -20.0}),
    ("take 5 m3/s for irrigation", {"demand_cms": 5.0}),
    ("rain -10%, +1.5 °C", {"dp_pct": -10.0, "dt_c": 1.5}),
    ("a fifth less rain", {"dp_pct": -20.0}),
    ("2C warmer", {"dt_c": 2.0}),
    ("300 l/s abstraction", {"demand_cms": 0.3}),
    ("it gets 3 degrees cooler and 10 percent wetter", {"dp_pct": 10.0, "dt_c": -3.0}),
])
def test_the_rules_read_a_scenario(text, expected):
    res = wm.parse_scenario(text)
    assert res["matched"], res
    for key, value in expected.items():
        assert res["scenario"][key] == pytest.approx(value)
    assert res["said"] == wm.scenario_label(res["scenario"])


def test_the_rules_leave_what_they_cannot_read():
    assert not wm.parse_scenario("show me the Nile")["matched"]
    half = wm.parse_scenario("a wetter climate, rain up 10%")
    assert not half["matched"] and half["unknown"] == ["a wetter climate"]
    assert not wm.parse_scenario("20% rain")["matched"]   # more or less?


def test_a_scenario_is_held_to_its_limits():
    sc, errors = wm.check_scenario({"dp_pct": -95, "dt_c": "2", "demand_cms": "lots"})
    assert sc["dp_pct"] == -80.0 and sc["dt_c"] == 2.0 and sc["demand_cms"] == 0.0
    assert any("outside" in e for e in errors) and any("not a number" in e for e in errors)
    assert wm.scenario_label(sc) == "80% less rain, 2 °C warmer"


def test_the_claim_lock_refuses_a_flipped_sign_a_reused_number_and_a_clipped_one():
    flipped = wm.parse_scenario_reply({"dp_pct": -20, "dt_c": None, "dpet_pct": None, "demand_cms": None},
                                      "20% more rain")
    assert flipped["scenario"] is None and "20% more rain" in flipped["errors"][0]
    reused = wm.parse_scenario_reply({"dp_pct": 20, "dt_c": 20, "dpet_pct": None, "demand_cms": None},
                                     "20% more rain")
    assert reused["scenario"]["dp_pct"] == 20 and reused["scenario"]["dt_c"] == 0
    assert any("dt_c" in e for e in reused["errors"])
    twice = wm.parse_scenario_reply({"dp_pct": -2, "dt_c": 2, "dpet_pct": None, "demand_cms": None},
                                    "rain falls by 2% and it is 2 degrees warmer")
    assert twice["scenario"]["dp_pct"] == -2 and twice["scenario"]["dt_c"] == 2 and not twice["errors"]
    clipped = wm.parse_scenario_reply({"dp_pct": None, "dt_c": 12, "dpet_pct": None, "demand_cms": None},
                                      "12 degrees warmer")
    assert clipped["scenario"] is None and "outside" in clipped["errors"][0]


def test_the_claim_lock_refuses_a_number_the_reader_did_not_say():
    res = wm.parse_scenario_reply('```json\n{"dp_pct": -20, "dt_c": 2, "dpet_pct": null, "demand_cms": 7}\n```',
                                  "20% less rain and 2 degrees warmer")
    assert res["scenario"]["dp_pct"] == -20.0 and res["scenario"]["demand_cms"] == 0.0
    assert res["errors"] == ["refused demand_cms 7: that number is not in the words"]
    words = wm.parse_scenario_reply({"dp_pct": -20, "dt_c": None, "dpet_pct": None, "demand_cms": None},
                                    "a fifth less rain")
    assert words["scenario"]["dp_pct"] == -20.0 and not words["errors"]
    assert wm.parse_scenario_reply("no idea", "x")["scenario"] is None


def test_the_prompt_shows_the_rules_own_answers():
    p = wm.scenario_prompt("river reach 230000002")
    assert '"dp_pct": -20.0' in p["system"] and "river reach 230000002" in p["system"]
    assert set(p["schema"]["required"]) == set(wm.LIMITS)


class _Fake:
    def __init__(self, content):
        self.content = content
        self.chat = self
        self.completions = self

    def create(self, **kw):
        msg = type("M", (), {"content": self.content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_model_scenario_with_a_client():
    res = wm.model_scenario("rain 15% lower", client=_Fake('{"dp_pct": -15, "dt_c": null}'), model="m", provider="p")
    assert res["scenario"]["dp_pct"] == -15.0 and res["model"] == "m via p"


# ── the way down ─────────────────────────────────────────────────────────────

# A toy basin: A -> B (clicked) -> C <- T1 (major), C -> D <- S (minor), D -> E <- T2 (major),
# E -> F <- T3 (major), F -> G (outlet).
IDS = {"A": 230000001, "B": 230000002, "C": 230000003, "T1": 230000004, "D": 230000005, "S": 230000006,
       "E": 230000007, "T2": 230000008, "F": 230000009, "T3": 230000010, "G": 230000011}
DOWN = {"A": "B", "B": "C", "T1": "C", "C": "D", "S": "D", "D": "E", "T2": "E", "E": "F", "T3": "F", "F": "G",
        "G": None}
UNIT_KM2 = {"A": 300, "B": 200, "C": 50, "T1": 400, "D": 20, "S": 30, "E": 10, "T2": 600, "F": 10, "T3": 1000, "G": 5}


@pytest.fixture
def toy_network(monkeypatch):
    names = list(IDS)
    net = rivers._Network(
        231, [IDS[n] for n in names], [IDS[DOWN[n]] if DOWN[n] else 0 for n in names],
        [UNIT_KM2[n] * 1e6 for n in names], [8.0 + 0.05 * k for k in range(len(names))],
        [47.0 + 0.02 * k for k in range(len(names))])
    monkeypatch.setattr(rivers, "_network_for", lambda rid, lat=None, lon=None: net)
    return net


def test_the_walk_stops_at_the_third_major_confluence(toy_network):
    res = wm.downstream_shares(toy_network, toy_network.index(IDS["B"]))
    ids = [r["river_id"] for r in res["reaches"]]
    assert ids == [IDS[n] for n in ("B", "C", "D", "E", "F")] and res["stopped"] == "confluences"
    shares = {r["river_id"]: r["share"] for r in res["reaches"]}
    assert res["basin_km2"] == 500.0
    assert shares[IDS["C"]] == pytest.approx(500 / 950, abs=1e-4)
    assert shares[IDS["D"]] == pytest.approx(500 / 1000, abs=1e-4)
    majors = [r["confluence"]["n"] for r in res["reaches"] if r["confluence"]]
    assert majors == [1, 2, 3]   # S (30 km2 against 950) is not major
    one = wm.downstream_shares(toy_network, toy_network.index(IDS["B"]), confluences=1)
    assert [r["river_id"] for r in one["reaches"]] == [IDS["B"], IDS["C"]]


def test_the_walk_stops_where_the_change_has_faded(toy_network):
    res = wm.downstream_shares(toy_network, toy_network.index(IDS["A"]), confluences=9, min_share=0.3)
    assert res["stopped"] == "faded" and res["reaches"][-1]["share"] < 0.3


# ── the run ──────────────────────────────────────────────────────────────────


def _forcing(idx: pd.DatetimeIndex, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    full = pd.date_range(idx[0] - pd.Timedelta(days=400), idx[-1], freq="D")
    doy = full.dayofyear.to_numpy()
    return pd.DataFrame({"p": rng.gamma(0.6, 6.0, len(full)),
                         "pet": np.clip(2.5 + 2.0 * np.sin(2 * np.pi * (doy - 80) / 365.25), 0.2, None),
                         "t": 13 + 8 * np.sin(2 * np.pi * (doy - 110) / 365.25)}, index=full)


@pytest.fixture
def toy_model(monkeypatch, toy_network):
    from aquascope.models.rainfall_runoff import GR4J

    idx = pd.date_range("2004-01-01", "2023-12-31", freq="D")
    f = _forcing(idx)
    truth = GR4J(x1=300, x2=0.5, x3=80, x4=1.8).simulate(f["p"], f["pet"], warmup_days=365).streamflow
    q_cms = (truth * 500.0 / 86.4).loc[idx]
    adv._RECORDS.clear()
    seen = {}

    def fetch(source, station_id, years, variable):
        seen.update(source=source, station_id=station_id, years=years)
        return {"series": q_cms, "variable": "discharge", "unit": "m3/s", "note": "synthetic"}

    monkeypatch.setattr(adv, "_fetch", fetch)
    monkeypatch.setattr(adv, "_forcing", lambda lat, lon, start, end: (seen.update(at=(lat, lon)) or
                                                                       f.loc[str(start):str(end)]))
    return seen


def test_whatif_runs_the_model_and_paints_the_way_down(toy_model):
    res = wm.whatif(IDS["B"], dp_pct=-20, dt_c=2, members=8, maxiter=8)
    assert res["ok"], res.get("error")
    assert toy_model["source"] == "geoglows" and toy_model["station_id"] == str(IDS["B"])
    assert res["model"]["calibrated_on"] == "modelled" and res["model"]["validation"]["kge"] > 0.8
    # forcing at the basin's area-weighted centre (A 300 km2, B 200 km2)
    lat_c = (300 * 47.0 + 200 * 47.02) / 500
    assert toy_model["at"][0] == pytest.approx(lat_c, abs=1e-3)
    o = res["outlet"]
    assert o["mean"]["change_pct"] < 0 and o["low"]["change_pct"] < 0 and o["flood"]["rp"] == 10.0
    assert o["mean"]["range"][0] <= o["mean"]["change_pct"] <= o["mean"]["range"][1]
    assert o["mean"]["after"] < o["mean"]["before"]
    assert res["model"]["ensemble"]["kept"] >= 1
    # the map's numbers are the outlet's, scaled by each reach's share
    first, conf1 = res["reaches"][0], res["reaches"][1]
    assert first["river_id"] == IDS["B"]
    assert first["mean_change_pct"] == pytest.approx(o["mean"]["change_pct"], abs=0.01)
    assert conf1["mean_change_pct"] == pytest.approx(o["mean"]["change_pct"] * 500 / 950, abs=0.01)
    assert conf1["confluence"]["n"] == 1
    assert set(res["basin"]["ids"]) == {IDS["A"], IDS["B"]} and res["basin"]["area_km2"] == 500.0
    assert res["basin"]["bbox"] == [8.0, 47.0, 8.05, 47.02]
    assert "20% less rain, 2 °C warmer" in res["verdict"] and "confluence 1" in res["verdict"]
    assert res["scenario"]["label"] == "20% less rain, 2 °C warmer"
    assert any(m["name"].startswith("Area-weighted") for m in res["methods"])


def test_whatif_reads_words_and_takes_a_demand(toy_model):
    res = wm.whatif(IDS["B"], text="take 2 m3/s for irrigation", members=0, maxiter=6)
    assert res["ok"] and res["read"]["by"] == "rules" and res["scenario"]["demand_cms"] == 2.0
    assert res["outlet"]["low"]["change_pct"] < res["outlet"]["mean"]["change_pct"] < 0
    assert res["outlet"]["mean"]["range"] is None   # no members, no range


def test_whatif_says_no_to_nothing_and_to_a_huge_basin(toy_model, monkeypatch):
    assert "nothing to change" in wm.whatif(IDS["B"])["error"]
    assert "could not read" in wm.whatif(IDS["B"], text="show me the Nile")["error"]
    monkeypatch.setattr(wm, "MAX_BASIN_KM2", 100.0)
    res = wm.whatif(IDS["B"], dp_pct=-10)
    assert not res["ok"] and "tributary" in res["error"]


def test_the_catchment_model_keeps_its_old_answer_without_the_new_options(toy_model):
    out = adv.catchment_model("usgs", "m", 47.0, 8.0, area_km2=500.0, maxiter=6,
                              scenarios=[{"label": "drier", "dp_pct": -10}])
    assert "flood" not in out["baseline_simulated"] and "range" not in out["scenarios"][0]
    assert "ensemble" not in out and "flood_change_pct" not in out["scenarios"][0]


def test_a_second_scenario_reuses_the_calibration_and_agrees(toy_model):
    first = wm.whatif(IDS["B"], dp_pct=-10, members=4, maxiter=6)
    again = wm.whatif(IDS["B"], dp_pct=-10, members=4, maxiter=6, params=first["model"]["params"])
    assert again["ok"] and again["model"].get("params") == first["model"]["params"]
    for key in ("mean", "low", "flood"):   # the parameters come back rounded to 6 decimals: the same to the display
        a, b = again["outlet"][key], first["outlet"][key]
        assert a["change_pct"] == pytest.approx(b["change_pct"], abs=0.01)
        assert a["before"] == pytest.approx(b["before"], rel=1e-5) and a["after"] == pytest.approx(b["after"], rel=1e-5)
    assert [r["mean_change_pct"] for r in again["reaches"]] == pytest.approx(
        [r["mean_change_pct"] for r in first["reaches"]], abs=0.01)
    assert adv._given_params({"X1": 1, "X2": "x"}) is None and adv._given_params({"X1": 1}) is None
