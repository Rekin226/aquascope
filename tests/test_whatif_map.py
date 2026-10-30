"""What if, on the map (#565): the scenario words, the walk downstream and the run, on a toy network."""

from __future__ import annotations

import json

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


def _reply(*clauses):
    """A model's parse in the round-3 shape (#590): (change, number, unit, season) per clause."""
    return {"clauses": [{"change": c[0], "number": c[1], "unit": c[2], "season": c[3] if len(c) > 3 else "year"}
                        for c in clauses]}


def test_the_code_refuses_a_flipped_sign_a_reused_number_and_a_clipped_one():
    flipped = wm.parse_scenario_reply(_reply(("less_rain", "20", "percent")), "20% more rain")
    assert flipped["scenario"] is None and "20% more rain" in flipped["errors"][0]
    reused = wm.parse_scenario_reply(_reply(("more_rain", "20", "percent"), ("warmer", "20", "percent")),
                                     "20% more rain")
    assert reused["scenario"]["dp_pct"] == 20 and reused["scenario"]["dt_c"] == 0
    assert any("warmer 20" in e and "as often as it is used" in e for e in reused["errors"])
    twice = wm.parse_scenario_reply(_reply(("less_rain", "2", "percent"), ("warmer", "2", "degC")),
                                    "rain falls by 2% and it is 2 degrees warmer")
    assert twice["scenario"]["dp_pct"] == -2 and twice["scenario"]["dt_c"] == 2 and not twice["errors"]
    clipped = wm.parse_scenario_reply(_reply(("warmer", "12", "degC")), "12 degrees warmer")
    assert clipped["scenario"] is None and "outside" in clipped["errors"][0]


def test_a_number_outside_the_words_is_refused_by_the_schema():
    # The number field is an enum of the words' own number tokens: 7 was never said.
    res = wm.parse_scenario_reply('```json\n{"clauses": [{"change": "take_water", "number": "7", "unit": "m3/s", '
                                  '"season": "year"}]}\n```', "20% less rain and 2 degrees warmer")
    assert res["scenario"] is None and res["errors"][0].startswith("the reply was refused: ")
    assert "'7' is not one of 20, 2, none" in res["errors"][0]
    words = wm.parse_scenario_reply(_reply(("less_rain", "a fifth", "fraction")), "a fifth less rain")
    assert words["scenario"]["dp_pct"] == -20.0 and not words["errors"]
    assert wm.parse_scenario_reply("no idea", "x")["scenario"] is None
    # The old reply shape (the four numbers, #565 to #642) is refused whole.
    old = wm.parse_scenario_reply({"dp_pct": -20, "dt_c": None, "dpet_pct": None, "demand_cms": None}, "20% less rain")
    assert old["scenario"] is None and "reply has no clauses" in old["errors"][0]


def test_the_schema_is_built_from_the_words():
    p = wm.scenario_prompt("river reach 230000002", "a brewery draws 7.5 l/s and it warms by one and a half degrees")
    item = p["schema"]["properties"]["clauses"]["items"]["properties"]
    assert item["number"]["enum"] == ["7.5", "one and a half", "1.5", "none"]
    assert item["change"]["enum"] == list(wm.CHANGES) and item["unit"]["enum"] == list(wm.UNITS)
    assert item["season"]["enum"] == list(wm.SEASONS)
    assert all("enum" in v for v in item.values())         # no free text
    assert "river reach 230000002" in p["system"] and '"one and a half"' in p["system"]
    assert wm.scenario_schema("a much wetter future")["properties"]["clauses"]["items"]["properties"]["number"][
        "enum"] == ["none"]


class _Fake:
    def __init__(self, content):
        self.content = content
        self.chat = self
        self.completions = self

    def create(self, **kw):
        msg = type("M", (), {"content": self.content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_model_scenario_with_a_client():
    reply = '{"clauses": [{"change": "less_rain", "number": "15", "unit": "percent", "season": "year"}]}'
    res = wm.model_scenario("rain 15% lower", client=_Fake(reply), model="m", provider="p")
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
    # the parameters come back rounded to 6 decimals, so a percentage (rounded to 0.01) may move by one
    # display step: -18.9 against -18.91 differs by 0.0100000000000016 in floating point (#600 CI)
    step = 0.015
    for key in ("mean", "low", "flood"):
        a, b = again["outlet"][key], first["outlet"][key]
        assert a["change_pct"] == pytest.approx(b["change_pct"], abs=step)
        assert a["before"] == pytest.approx(b["before"], rel=1e-5) and a["after"] == pytest.approx(b["after"], rel=1e-5)
    assert [r["mean_change_pct"] for r in again["reaches"]] == pytest.approx(
        [r["mean_change_pct"] for r in first["reaches"]], abs=step)
    assert adv._given_params({"X1": 1, "X2": "x"}) is None and adv._given_params({"X1": 1}) is None


# ── on the device (#590): the model parses, the code reads ──────────────────


@pytest.mark.parametrize("line, clauses, want, said", [
    # The intent of Gemini Nano's recorded replies (2026-10-25 and 10-26), in the round-3 shape.
    ("what if the town pumps 300 litres per second out of the river", [("take_water", "300", "l/s")],
     {"demand_cms": 0.3}, "0.3 m³/s taken"),
    ("summers get much hotter, maybe three degrees", [("warmer", "three", "degC", "summer")], {"dt_c": 3.0},
     "3 °C warmer"),
    ("drought: precipitation down by a quarter with 1.5 C of warming",
     [("less_rain", "a quarter", "fraction"), ("warmer", "1.5", "degC")], {"dp_pct": -25.0, "dt_c": 1.5},
     "25% less rain, 1.5 °C warmer"),
    # Check 3, escape 2: "draws 7.5 l/s" ran as 7.5 m³/s taken. The code converts: 0.0075 m³/s.
    ("a brewery upstream draws 7.5 l/s and the climate warms by one and a half degrees",
     [("take_water", "7.5", "l/s"), ("warmer", "one and a half", "degC")], {"demand_cms": 0.0075, "dt_c": 1.5},
     "1.5 °C warmer, 0.0075 m³/s taken"),
    ("what if winters get two degrees warmer and there is 15 percent less rain",
     [("warmer", "two", "degC", "winter"), ("less_rain", "15", "percent")], {"dt_c": 2.0, "dp_pct": -15.0},
     "15% less rain, 2 °C warmer"),
    ("what if the factory takes twenty litres a second", [("take_water", "20", "l/s")], {"demand_cms": 0.02},
     "0.02 m³/s taken"),
])
def test_recorded_intents_read_right(line, clauses, want, said):
    out = wm.parse_scenario_reply(json.dumps(_reply(*clauses)), line)
    assert out["scenario"] and not out["errors"], out
    for key, value in want.items():
        assert out["scenario"][key] == pytest.approx(value)
    assert out["said"] == said


def test_a_unit_the_words_do_not_give_is_refused():
    line = "a brewery upstream draws 7.5 l/s"
    out = wm.parse_scenario_reply(_reply(("take_water", "7.5", "m3/s")), line)
    assert out["scenario"] is None and out["errors"] == ["take water 7.5: the words say l/s, not m3/s"]
    out = wm.parse_scenario_reply(_reply(("less_rain", "10", "mm")), "10 mm less rain")
    assert out["scenario"] is None and "a change in mm cannot be run" in out["errors"][0]
    out = wm.parse_scenario_reply(_reply(("warmer", "2", "degF")), "2 degrees F warmer")
    assert out["scenario"]["dt_c"] == pytest.approx(1.111) and not out["errors"]
    out = wm.parse_scenario_reply(_reply(("take_water", "86400", "m3/day")), "take 86400 m3/day")
    assert out["scenario"]["demand_cms"] == pytest.approx(1.0)


def test_what_the_words_do_not_say_is_refused():
    # "a much wetter future" holds no number: the model can only say "none", and nothing is read.
    out = wm.parse_scenario_reply(_reply(("more_rain", "none", "percent")), "a much wetter future")
    assert out["scenario"] is None and out["errors"] == ["more rain: the words give no number for it"]
    # Nano's reading of that line (rain +10, 2026-10-26) cannot be written: 10 is no token of the words.
    out = wm.parse_scenario_reply(_reply(("more_rain", "10", "percent")), "a much wetter future")
    assert out["scenario"] is None and out["errors"][0].startswith("the reply was refused")
    # "m3/s" holds a 3 that nobody said; "a hotter one" holds no unit for its one.
    assert [t["token"] for t in wm.number_tokens("take 5 m3/s")] == ["5"]
    assert wm.parse_scenario_reply(_reply(("warmer", "one", "degC")), "a hotter one")["errors"] == [
        "warmer one: the words give no unit for it"]
    assert wm.parse_scenario("a hotter one")["scenario"] is None


def test_number_tokens_carry_the_unit_the_words_give():
    toks = wm.number_tokens("rain falls by a fifth, 300 litres a second, twenty-five percent and 2 °C")
    assert [(t["token"], t["value"], t["unit"]) for t in toks] == [
        ("a fifth", 0.2, "fraction"), ("300", 300.0, "l/s"), ("twenty-five", 25.0, "percent"), ("2", 2.0, "degC")]


@pytest.mark.parametrize("line, token, unit, value", [
    ("take half a cubic metre per second", "half", "m3/s", 0.5),
    ("take a quarter of a litre per second", "a quarter", "l/s", 0.00025),
    ("take half a megalitre per day", "half", "ML/day", 500 / 86400),
    ("take a half cubic metre per second", "a half", "m3/s", 0.5),
    ("take a half litre per second", "a half", "l/s", 0.0005),
])
def test_fractional_quantities_keep_their_explicit_units(line, token, unit, value):
    result = wm.parse_scenario_reply(_reply(("take_water", token, unit)), line)
    assert not result["errors"], result
    assert result["scenario"]["demand_cms"] == pytest.approx(value, abs=1e-6)


def test_fractional_quantity_does_not_weaken_unit_checks_or_shares():
    line = "take half a cubic metre per second"
    bad = wm.parse_scenario_reply(_reply(("take_water", "half", "l/s")), line)
    assert bad["scenario"] is None and "words say m3/s, not l/s" in bad["errors"][0]
    share = wm.parse_scenario_reply(_reply(("less_rain", "a quarter", "fraction")), "a quarter less rain")
    assert not share["errors"] and share["scenario"]["dp_pct"] == -25


@pytest.mark.parametrize("phrase", ["a third", "one third", "a quarter", "half"])
def test_fractional_quantity_schema_aliases_round_trip(phrase):
    line = f"take {phrase} of a litre per second"
    token = wm.number_tokens(line)[0]
    numbers = wm.scenario_schema(line)["properties"]["clauses"]["items"]["properties"]["number"]["enum"]
    assert phrase in numbers
    if "third" in phrase:
        assert "0.333" not in numbers
    for number in numbers:
        if number == "none":
            continue
        result = wm.parse_scenario_reply(_reply(("take_water", number, "l/s")), line)
        assert not result["errors"], (number, result)
        assert result["scenario"]["demand_cms"] == pytest.approx(token["value"] / 1000, abs=1e-6)


@pytest.mark.parametrize("phrase", ["a third", "one third"])
def test_third_share_and_token_reuse(phrase):
    share = wm.parse_scenario_reply(_reply(("less_rain", phrase, "fraction")), f"{phrase} less rain")
    assert not share["errors"] and share["scenario"]["dp_pct"] == pytest.approx(-100 / 3, abs=0.001)
    quantity = wm.parse_scenario_reply(_reply(("take_water", phrase, "m3/s")),
                                       f"take {phrase} of a cubic metre per second")
    assert not quantity["errors"] and quantity["scenario"]["demand_cms"] == pytest.approx(1 / 3, abs=1e-6)
    line = f"take {phrase} of a litre per second"
    reused = wm.parse_scenario_reply(_reply(("take_water", phrase, "l/s"), ("take_water", phrase, "l/s")), line)
    assert reused["scenario"]["demand_cms"] == pytest.approx(1 / 3000, abs=1e-6)
    assert any("as often as it is used" in error for error in reused["errors"])


def test_rounded_digit_alias_is_not_offered_for_a_precise_number():
    line = "take 0.3333 litres per second"
    numbers = wm.scenario_schema(line)["properties"]["clauses"]["items"]["properties"]["number"]["enum"]
    assert numbers == ["0.3333", "none"]
    result = wm.parse_scenario_reply(_reply(("take_water", "0.3333", "l/s")), line)
    assert not result["errors"]
    assert result["scenario"]["demand_cms"] == pytest.approx(0.000333, abs=1e-6)


@pytest.mark.parametrize("line, key, value", [
    ("take half a cubic metre per second", "demand_cms", 0.5),
    ("take a quarter of a litre per second", "demand_cms", 0.00025),
    ("half a degree warmer", "dt_c", 0.5),
    ("half a percent less rain", "dp_pct", -0.5),
    ("a quarter less rain", "dp_pct", -25.0),
    ("a half cubic metre per second", "demand_cms", 0.5),
    ("a half litre per second", "demand_cms", 0.0005),
    ("a half degree warmer", "dt_c", 0.5),
    ("a half percent less rain", "dp_pct", -0.5),
])
def test_rules_distinguish_fractional_quantities_from_proportions(line, key, value):
    result = wm.parse_scenario(line)
    assert result["matched"], result
    assert result["scenario"][key] == pytest.approx(value)


@pytest.mark.parametrize("line, key, value", [
    ("take 300 litres per second", "demand_cms", 0.3),
    ("take 300 liters a second", "demand_cms", 0.3),
    ("take 300 L/s", "demand_cms", 0.3),
    ("take five cubic metres per second", "demand_cms", 5.0),
    ("three degrees warmer", "dt_c", 3.0),
    ("one degree warmer", "dt_c", 1.0),
    ("twenty percent less rain", "dp_pct", -20.0),
    ("rain falls by one fifth", "dp_pct", -20.0),
    ("one and a half degrees warmer", "dt_c", 1.5),
    ("two and a half degrees warmer", "dt_c", 2.5),
    ("take 7.5 l/s", "demand_cms", 0.0075),
])
def test_units_and_numbers_in_words_are_read(line, key, value):
    out = wm.parse_scenario(line)
    assert out["matched"] and out["scenario"][key] == pytest.approx(value), out


def test_every_clause_is_read_or_named():
    # The rules read the evaporation clause; the page used to drop it (no slider of its own).
    out = wm.parse_scenario("rain falls by a fifth and evaporation goes up 10 percent")
    assert out["matched"] and out["scenario"]["dp_pct"] == -20 and out["scenario"]["dpet_pct"] == 10
    assert out["said"] == "20% less rain, 10% more evaporation"
    # A word the rules cannot read is named, so the page can say it and does not run on half the line.
    out = wm.parse_scenario("drought: precipitation down by a quarter with 1.5 C of warming")
    assert not out["matched"] and out["unknown"] == ["drought"] and out["said"] == "25% less rain, 1.5 °C warmer"


def test_a_basin_over_the_limit_says_why_before_a_run():
    assert wm._too_large(1_379.0) is None
    why = wm._too_large(143_851.0)
    assert why.startswith("143,851 km2 drain to this reach") and "100,000 km2" in why


# ── on the device, round 2 (#590): nothing reads as nought, no season widened in silence ─────────────────────


def test_a_small_demand_keeps_its_figures():
    # Gemini Nano read "the factory takes twenty litres a second" as 0.02 m³/s, right; the status line and the map
    # log said "0 m³/s taken".
    out = wm.parse_scenario_reply(_reply(("take_water", "twenty", "l/s")),
                                  "what if the factory takes twenty litres a second")
    assert out["scenario"]["demand_cms"] == pytest.approx(0.02) and not out["errors"], out
    assert out["said"] == "0.02 m³/s taken"
    assert wm.scenario_label({"demand_cms": 0.001}) == "0.001 m³/s taken"
    assert wm.scenario_label({"demand_cms": 0.0075}) == "0.0075 m³/s taken"
    assert wm.scenario_label({"dp_pct": -100 / 3, "dt_c": 1.25}) == "33.3% less rain, 1.25 °C warmer"
    assert wm.scenario_label({"demand_cms": 1234.5, "dp_pct": 20}) == "20% more rain, 1235 m³/s taken"
    assert wm.scenario_label({}) == "no change"


def test_a_season_the_words_name_is_said_not_widened():
    # Nano, 2026-10-26: "summers ... three degrees" read as +3 °C and run over the whole year without a word.
    out = wm.parse_scenario_reply(_reply(("warmer", "three", "degC", "summer")),
                                  "summers get much hotter, maybe three degrees")
    assert out["scenario"]["dt_c"] == 3.0 and out["season"] == ["summers"] and not out["errors"]
    # A season the words do not name is not said.
    out = wm.parse_scenario_reply(_reply(("warmer", "three", "degC", "winter")), "three degrees warmer")
    assert out["scenario"]["dt_c"] == 3.0 and out["season"] == [] and "name no winter" in out["errors"][0]
    assert wm.parse_scenario("20% less rain in the dry season")["season"] == ["dry season"]
    assert wm.parse_scenario("20% less rain and 2 degrees warmer")["season"] == []
    # Words that only look like a season are not one: "rain may fall", "rain falls by a fifth".
    assert wm.season_words("rain may fall 20%, and it falls by a fifth") == []
    assert wm.season_words("in May rain falls 10%, less in the fall") == ["in May", "in the fall"]
