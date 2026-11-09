"""On-device Ask (#590): the compact tool result a small model is shown, and the lock on its answer.

The replies quoted here are the ones Gemini Nano wrote in Chrome 155 on 2026-10-25 (docs/assessments/
ai-on-the-map.md); the results they are held to are fixtures shaped like the tools' real payloads.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope.ai_engine.device_ask import MAX_CHARS, compact_result, facts, lock_answer
from aquascope.explore import analyze_series

POTOMAC_WORDS = "Summarise the record at this gauge: period, mean, trend, and the flood frequency."


def _potomac_payload() -> dict:
    """An analyze_station payload as the MCP face returns it, with the Potomac's recorded figures."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    res = analyze_series(pd.Series(np.exp(rng.normal(5, 1, len(idx))), index=idx), "discharge", "m3/s")
    res.pop("series", None)
    res["fdc"] = {k: res["fdc"][k] for k in ("q95", "q50", "q10")}
    res.update(source="usgs", station_id="01646500", start="1930-03-01", end="2026-10-24",
               stats={"mean": 320.7738, "median": 182.0, "min": 3.43, "max": 12100.0})
    res["trend"] = {"on": "annual mean", "p_value": 0.82, "tau": 0.016, "trend": "no trend",
                    "sens_slope_per_year": 0.059, "n_years": 96}
    return res


def _result(name: str, payload: dict, arguments: dict | None = None) -> dict:
    return {"name": name, "arguments": arguments or {}, "compact": compact_result(name, payload), "ok": True}


# ── the compact result ─────────────────────────────────────────────────────────


def test_the_compact_result_keeps_the_trend_and_floods_the_3000_character_cut_lost() -> None:
    payload = _potomac_payload()
    full = json.dumps(payload, default=str)
    assert "p_value" not in full[:3000], "the fixture reproduces the cut: the trend lies past 3,000 characters"
    c = compact_result("analyze_station", payload)
    text = json.dumps(c)
    assert len(text) <= MAX_CHARS
    assert c["trend_annual_mean"]["p_value"] == 0.82 and c["trend_annual_mean"]["significant"] is False
    assert "no significant trend" in c["trend_annual_mean"]["reading"] and "+0.059" in c["trend_annual_mean"]["reading"]
    assert {r["T_years"] for r in c["flood_frequency"]["return_levels"]} >= {2, 10, 100}
    assert set(c["flow_duration"]) >= {"q95", "q50", "q10"}
    assert c["daily_values"]["max"] == 12100.0          # the key says these are daily values, not annual ones
    assert "annual_max" not in c and "methods" not in c  # the series and the citations stay out


def test_find_stations_and_anywhere_compact_to_what_an_answer_needs() -> None:
    stations = {"n_catalog": 60000, "n_returned": 12, "limit": 12, "stations": [
        {"source": "uk_ea", "station_id": f"E{i:05d}A", "name": f"SITE {i}", "latitude": 51.4, "longitude": -0.3,
         "license": "OGL", "url": "https://x", "variables": ["discharge"]} for i in range(12)]}
    c = compact_result("find_stations", stations)
    assert len(c["stations"]) == 10 and "license" not in c["stations"][0]
    here = {"latitude": 25.05, "longitude": 121.55, "climate": {
        "precipitation_mm_per_year": 1733.4111, "temperature_mean_c": 22.7201, "aridity_index": 1.4838,
        "aridity_class": "humid", "monthly_precipitation_mm": list(range(12)),
        "annual_precipitation": {"year": list(range(2006, 2026)), "mm": list(range(20))}},
        "glofas": {"unit": "m3/s", "start": "2006-01-01", "end": "2026-10-24",
                   "stats": {"mean": 5.6336, "min": 1.35, "max": 214.28}, "modelled": True}}
    c = compact_result("anywhere", here)
    assert "monthly_precipitation_mm" not in c["climate"]
    assert c["glofas"]["daily_values"] == {"mean": 5.6336, "min": 1.35, "max": 214.28}
    assert any("1733" in line or "1,733" in line for line in facts("anywhere", c))


def test_a_huge_unknown_result_is_shrunk_never_cut_mid_value() -> None:
    payload = {"rows": [{"id": i, "note": "x" * 300, "values": list(range(50))} for i in range(200)]}
    c = compact_result("describe_catchment", payload)
    assert len(json.dumps(c)) <= MAX_CHARS
    assert isinstance(c, dict)


# ── the lock: recorded device answers ─────────────────────────────────────────


def test_the_recorded_increasing_trend_claim_is_refused_against_p_082() -> None:
    res = [_result("analyze_station", _potomac_payload(), {"source": "usgs", "station_id": "01646500"})]
    answer = ("The Potomac at Little Falls (USGS 01646500) has a record from 1930 to 2026 with a mean of "
              "320.7738 m3/s, a minimum of 3.43 and a maximum of 12100. The trend in annual mean flow is increasing.")
    out = lock_answer(answer, res, POTOMAC_WORDS)
    assert not out["ok"]
    assert any("trend wording" in r and "p = 0.82" in r for r in out["refused"])
    assert any("no significant trend" in line for line in out["facts"])


def test_the_correct_reading_passes() -> None:
    res = [_result("analyze_station", _potomac_payload())]
    for answer in (
        "USGS 01646500 runs from 1930-03-01 to 2026-10-24, mean 320.77 m3/s. There is no significant trend in the "
        "annual mean (Mann-Kendall p = 0.82; Sen's slope +0.059 m3/s per year).",
        "No, the Potomac at Little Falls is not getting drier: the annual-mean trend is not significant (p = 0.82).",
        "The annual means rise slightly (Sen's slope 0.059 m3/s per year), but the trend is not significant.",
    ):
        out = lock_answer(answer, res, POTOMAC_WORDS)
        assert out["ok"], (answer, out["refused"])


def test_numbers_from_an_earlier_question_are_refused_when_no_tool_ran() -> None:
    # Recorded: no tool call, the figures came from the previous question's result still in the session.
    answer = ("The 100-year flood (Q10) is not directly available from the data, which covers a record from "
              "1930-03-01.")
    out = lock_answer(answer, [], "What is the 100-year flood at this gauge?")
    assert not out["ok"]
    joined = " ".join(out["refused"])
    assert "1930-03-01" in joined and "Q10" in joined
    assert "100" not in joined.replace("1930", "")  # the 100 is the reader's own


def test_a_trend_with_no_test_behind_it_is_refused() -> None:
    answer = ("The Potomac at Little Falls shows an increasing trend in annual mean flow. The Mann-Kendall test "
              "indicates an upward trend.")
    out = lock_answer(answer, [], "Is the Potomac at Little Falls getting drier? Use the annual-mean trend.")
    assert not out["ok"]
    assert any("no trend was computed" in r for r in out["refused"])


KINGSTON = {"n_catalog": 60000, "n_returned": 3, "limit": 5, "stations": [
    {"source": "uk_ea", "station_id": "E02102A", "name": "KINGSTON VALE", "latitude": 51.43, "longitude": -0.26},
    {"source": "uk_ea", "station_id": "3400TH", "name": "HOGSMILL AT EWELL", "latitude": 51.35, "longitude": -0.25},
    {"source": "uk_ea", "station_id": "3990TH", "name": "KINGSTON BRIDGE", "latitude": 51.41, "longitude": -0.31},
]}


def test_gauge_ids_swapped_between_sites_are_refused() -> None:
    res = [_result("find_stations", KINGSTON, {"near": [51.1, 0.0], "limit": 5})]
    words = "Find the five gauges nearest to Kingston upon Thames"
    swapped = "KINGSTON VALE (3400TH), HOGSMILL AT EWELL (E02102A) and KINGSTON BRIDGE (3990TH)."
    out = lock_answer(swapped, res, words)
    assert not out["ok"]
    assert any("3400TH is the id of HOGSMILL AT EWELL" in r for r in out["refused"])
    right = "KINGSTON VALE (E02102A), HOGSMILL AT EWELL (3400TH) and KINGSTON BRIDGE (3990TH)."
    ok = lock_answer(right, res, words)
    assert ok["ok"], ok["refused"]
    # The search point was the model's own (about 35 km from Kingston): said, not hidden.
    assert ok["notes"] and "51.1" in ok["notes"][0]


def test_an_id_that_no_result_holds_is_refused() -> None:
    res = [_result("find_stations", KINGSTON)]
    out = lock_answer("The nearest is KINGSTON VALE (E02103B).", res, "nearest gauge to Kingston")
    assert not out["ok"] and any("E02103B" in r for r in out["refused"])


def test_the_recorded_anywhere_answer_passes_on_its_numbers() -> None:
    here = {"latitude": 25.05, "longitude": 121.55, "start": "2016-10-24", "end": "2026-10-24",
            "climate": {"precipitation_mm_per_year": 1733.4111, "temperature_mean_c": 22.7201,
                        "aridity_index": 1.4838, "aridity_class": "humid"},
            "glofas": {"unit": "m3/s", "start": "2006-01-01", "end": "2026-10-24",
                       "stats": {"mean": 5.6336, "min": 1.35, "max": 214.28}, "modelled": True}}
    res = [_result("anywhere", here, {"lat": 25.05, "lon": 121.55})]
    answer = ("At 25.05, 121.55 the climate is humid: 1733.4111 mm of rain a year, 22.7201 °C, aridity 1.4838. "
              "GloFAS gives a mean discharge of 5.6336 m3/s from 2006 to 2026, with a minimum of 1.35 and a "
              "maximum of 214.28 m3/s.")
    out = lock_answer(answer, res, "What is the climate and modelled discharge at 25.05, 121.55?")
    assert out["ok"], out["refused"]


@pytest.mark.parametrize("answer", [
    "The mean flow is 999 m3/s.",
    "The 100-year flood is 4,500 m3/s.",
    "The record starts in 1950.",
])
def test_an_invented_number_is_refused(answer: str) -> None:
    res = [_result("analyze_station", _potomac_payload())]
    assert not lock_answer(answer, res, POTOMAC_WORDS)["ok"]


def test_rounding_and_list_markers_are_not_refused() -> None:
    res = [_result("analyze_station", _potomac_payload())]
    answer = "1. Mean flow: 320.8 m3/s.\n2. Lowest day: 3.4 m3/s.\n3. Highest day: 12,100 m3/s."
    out = lock_answer(answer, res, POTOMAC_WORDS)
    assert out["ok"], out["refused"]


# ── labels tied to their own numbers (recheck of 2026-10-26) ──────────────────
# The recorded figures: the Potomac's daily minimum 3.43, Q50 186, Q95 35.7, GEV 100-year 11247.39, LP3 100-year
# 10795.57, LP3 50-year 9150.06; the Thames at Kingston's mean 65.48, Q50 39.9, Q95 7.5, maximum 800 in 1894.
# The other return levels and the intervals are fixture values, shaped like the tool's.


def _potomac_ffa_payload() -> dict:
    return {
        "source": "usgs", "station_id": "01646500", "name": "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA",
        "river": "Potomac River", "unit": "m3/s", "variable": "discharge", "start": "1930-03-01", "end": "2026-10-24",
        "stats": {"mean": 320.7738, "median": 182.0, "min": 3.43, "max": 12100.0},
        "fdc": {"q95": 35.7, "q50": 186.0, "q10": 702.0},
        "ffa": {"n_years": 96, "return_periods": [2, 5, 10, 25, 50, 100],
                "record_max": {"value": 12100.0, "year": 1936},
                "fits": {"gev_lmoments": {"q_by_T": {"2": 2620.4, "5": 4100.2, "10": 5230.8, "25": 6890.1,
                                                     "50": 8300.5, "100": 11247.39}},
                         "lp3": {"q_by_T": {"2": 2580.0, "5": 4050.7, "10": 5180.3, "25": 6930.9,
                                            "50": 9150.06, "100": 10795.57},
                                 "ci": [[2400.0, 2760.0], [3800.0, 4320.0], [4850.0, 5560.0], [6350.0, 7600.0],
                                        [8200.0, 10400.0], [9500.0, 12600.0]]}}},
        "trend": {"on": "annual mean", "p_value": 0.82, "trend": "no trend", "sens_slope_per_year": 0.059,
                  "n_years": 96},
    }


def _thames_payload() -> dict:
    return {
        "source": "uk_ea", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston",
        "river": "Thames", "unit": "m3/s", "variable": "discharge", "start": "1883-10-01", "end": "2026-10-09",
        "stats": {"mean": 65.48, "median": 40.1, "min": 0.92, "max": 800.0},
        "fdc": {"q95": 7.5, "q50": 39.9, "q10": 160.0},
        "ffa": {"record_max": {"value": 800.0, "year": 1894}},
    }


COMPARE_WORDS = "compare the mean flow of the Potomac (USGS-01646500) with the Thames at Kingston"

#: Recorded on the device, 2026-10-26: right means, periods and maxima, and the Potomac's minimum given as its Q95.
COMPARE_ANSWER = (
    "The mean flow of the Thames at Kingston (uk_ea 8496ce69-482c-406a-a2f0-ac418ef8f099) is 65.48 m3/s over the "
    "period 1883-10-01 to 2026-10-09 whereas the mean flow of the Potomac at Little Falls (USGS-01646500) is "
    "320.77 m3/s over the period 1930-03-01 to 2026-10-24. The Thames's daily flow duration shows a q50 of 39.9 "
    "m3/s and a q95 of 7.5 m3/s, while the Potomac's daily flow duration shows a q50 of 186 m3/s and a q95 of "
    "3.43 m3/s. The Thames has a maximum recorded value of 800 m3/s in 1894, while the Potomac's maximum "
    "recorded value is 12100 m3/s in 1936.")


def _two_gauges() -> list[dict]:
    return [_result("analyze_station", _thames_payload(), {"source": "uk_ea"}),
            _result("analyze_station", _potomac_ffa_payload(), {"source": "usgs", "station_id": "01646500"})]


def test_the_recorded_q95_under_the_wrong_label_is_refused() -> None:
    out = lock_answer(COMPARE_ANSWER, _two_gauges(), COMPARE_WORDS)
    assert not out["ok"]
    assert any(r.startswith("3.43 is not the Q95 of") and "POTOMAC" in r and "35.7" in r for r in out["refused"]), \
        out["refused"]
    assert len(out["refused"]) == 1, out["refused"]     # the means, periods, Q50s and maxima were right


def test_the_same_comparison_with_the_right_q95_passes() -> None:
    out = lock_answer(COMPARE_ANSWER.replace("q95 of 3.43", "q95 of 35.7"), _two_gauges(), COMPARE_WORDS)
    assert out["ok"], out["refused"]


@pytest.mark.parametrize(("swap", "said"), [
    # The Thames's Q95 put on the Potomac.
    (("q95 of 3.43", "q95 of 7.5"), "7.5 is the Q95 of Kingston"),
    # The Thames's record dates put on the Potomac.
    (("320.77 m3/s over the period 1930-03-01 to 2026-10-24", "320.77 m3/s over the period 1883-10-01 to 2026-10-09"),
     "1883-10-01 to 2026-10-09 is the record of Kingston"),
    # The Thames's mean given as the Potomac's.
    (("Potomac at Little Falls (USGS-01646500) is 320.77", "Potomac at Little Falls (USGS-01646500) is 65.48"),
     "65.48 is the daily mean of Kingston"),
])
def test_a_number_said_of_the_other_gauge_is_refused(swap: tuple[str, str], said: str) -> None:
    answer = COMPARE_ANSWER.replace(*swap).replace("q95 of 3.43", "q95 of 35.7")
    out = lock_answer(answer, _two_gauges(), COMPARE_WORDS)
    assert not out["ok"]
    assert any(said in r for r in out["refused"]), out["refused"]


def test_the_comparison_in_another_order_still_passes() -> None:
    answer = ("The mean is 65.48 m3/s at the Thames and 320.77 m3/s at the Potomac; the Q95 is 7.5 m3/s at the "
              "Thames and 35.7 m3/s at the Potomac.")
    out = lock_answer(answer, _two_gauges(), COMPARE_WORDS)
    assert out["ok"], out["refused"]


def test_the_recorded_return_period_written_as_a_flow_is_refused() -> None:
    res = [_result("analyze_station", _potomac_ffa_payload())]
    # Recorded (direct run, 2026-10-26); the elided middle of the reply is filled with the same pattern.
    answer = ("The return period for a flood of magnitude 2 m3/s is 2 years, and for a flood of magnitude 10 m3/s "
              "is 25 years.")
    out = lock_answer(answer, res, "What are the flood return periods at this gauge?")
    assert not out["ok"]
    joined = " ".join(out["refused"])
    assert "2 m3/s is not a flow" in joined and "10 m3/s is not a flow" in joined


@pytest.mark.parametrize(("answer", "ok", "said"), [
    # Recorded: the LP3 value, no method named, while the page's figures are GEV.
    ("The 100-year flood at the Potomac at Little Falls is 10795.57 m3/s.", False,
     "10795.6 is the LP3 100-year flood, and the answer does not say LP3"),
    ("The 100-year flood (LP3) at the Potomac at Little Falls is 10795.57 m3/s.", True, ""),
    ("The 100-year flood (GEV) is 11,247 m3/s.", True, ""),
    ("The 100-year flood is 11247.39 m3/s.", True, ""),                      # the page's own method
    ("The 100-year flood by Log-Pearson III is 11247.39 m3/s.", False, "is the GEV 100-year flood, not the LP3"),
    ("The 100-year flood (GEV) is 8300.5 m3/s.", False, "8300.5 is the 50-year flood, not the 100-year"),
    ("The 50-year flood (LP3) is 9150.06 m3/s, with a 90 % interval of 8200 to 10400 m3/s.", True, ""),
    ("The 1-in-100 flood is 4,500 m3/s.", False, "not the 100-year flood"),
])
def test_a_return_level_is_tied_to_its_own_period_and_method(answer: str, ok: bool, said: str) -> None:
    res = [_result("flood_frequency", _potomac_ffa_payload())]
    out = lock_answer(answer, res, "What is the 100-year flood at this gauge?")
    assert out["ok"] is ok, out["refused"]
    if said:
        assert any(said in r for r in out["refused"]), out["refused"]


def test_the_compact_result_names_both_flood_methods() -> None:
    c = compact_result("analyze_station", _potomac_ffa_payload())
    method = c["flood_frequency"]["method"]
    assert "GEV" in method and "Log-Pearson III" in method and "page" in method
    row = next(r for r in c["flood_frequency"]["return_levels"] if r["T_years"] == 100)
    assert row["gev"] == 11247.39 and row["lp3"] == 10795.57


def test_a_mean_minimum_or_maximum_is_its_own_field() -> None:
    res = [_result("analyze_station", _potomac_ffa_payload())]
    assert lock_answer("The daily minimum is 3.43 m3/s and the maximum is 12,100 m3/s.", res, POTOMAC_WORDS)["ok"]
    bad = lock_answer("The mean flow is 186 m3/s.", res, POTOMAC_WORDS)        # 186 is the Q50
    assert not bad["ok"] and any("186 is not the daily mean" in r for r in bad["refused"]), bad["refused"]
    bad = lock_answer("The minimum flow is 35.7 m3/s.", res, POTOMAC_WORDS)    # 35.7 is the Q95
    assert not bad["ok"] and any("35.7 is not the daily minimum" in r for r in bad["refused"]), bad["refused"]
    # A record start said as the end, and the reverse.
    bad = lock_answer("The record runs from 2026-10-24 to 1930-03-01.", res, POTOMAC_WORDS)
    assert not bad["ok"] and any("runs from 1930-03-01 to 2026-10-24" in r for r in bad["refused"]), bad["refused"]


def test_the_trend_and_climate_wording_is_not_bound_to_a_label() -> None:
    res = [_result("analyze_station", _potomac_ffa_payload())]
    answer = ("There is no significant trend in the annual mean (Mann-Kendall p = 0.82; Sen's slope +0.059 m3/s per "
              "year). The mean is 320.77 m3/s.")
    assert lock_answer(answer, res, POTOMAC_WORDS)["ok"]
    here = {"latitude": 25.05, "longitude": 121.55, "climate": {"precipitation_mm_per_year": 1733.4111,
                                                                "temperature_mean_c": 22.7201},
            "glofas": {"unit": "m3/s", "start": "2006-01-01", "end": "2026-10-24",
                       "stats": {"mean": 5.6336, "min": 1.35, "max": 214.28}, "modelled": True}}
    res = [_result("anywhere", here, {"lat": 25.05, "lon": 121.55})]
    answer = ("Rain averages an average of 1733.41 mm a year, with a mean of 22.72 °C. GloFAS gives a mean of "
              "5.63 m3/s.")
    out = lock_answer(answer, res, "What is the climate and modelled discharge at 25.05, 121.55?")
    assert out["ok"], out["refused"]


FIVE = {"n_catalog": 60000, "n_returned": 5, "limit": 5, "stations": [
    {"source": "uk_ea", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston", "river": "Thames",
     "latitude": 51.415, "longitude": -0.308},
    {"source": "uk_ea", "station_id": "E02102A", "name": "KINGSTON VALE", "latitude": 51.43, "longitude": -0.26},
    {"source": "uk_ea", "station_id": "3400TH", "name": "HOGSMILL AT EWELL", "latitude": 51.35, "longitude": -0.25},
    {"source": "uk_ea", "station_id": "3990TH", "name": "KINGSTON BRIDGE", "latitude": 51.41, "longitude": -0.31},
    {"source": "uk_ea", "station_id": "E21778", "name": "TEDDINGTON", "latitude": 51.43, "longitude": -0.32},
]}


def test_five_gauges_asked_and_one_analysed_is_refused_and_the_list_shown() -> None:
    words = "Find the five gauges nearest to Kingston upon Thames"
    res = [_result("find_stations", FIVE, {"query": "Kingston", "limit": 5}),
           _result("analyze_station", _thames_payload(), {"source": "uk_ea"})]
    # Recorded shape: the device analysed one gauge and answered about it.
    one = ("The Thames at Kingston (8496ce69-482c-406a-a2f0-ac418ef8f099) has a mean flow of 65.48 m3/s from "
           "1883-10-01 to 2026-10-09.")
    out = lock_answer(one, res, words + "\n\nContext: The user is looking at a map centred on 51.41, -0.30.")
    assert not out["ok"]
    assert any("asks for 5 gauges and the answer names 1" in r for r in out["refused"]), out["refused"]
    assert sum(".where" not in line and "(uk_ea" in line for line in out["facts"]) >= 5   # the search's list is shown
    listed = "; ".join(f"{s['name']} ({s['station_id']})" for s in FIVE["stations"])
    assert lock_answer(f"The five nearest are {listed}.", res, words)["ok"]


def test_wanted_list_reads_the_count() -> None:
    from aquascope.ai_engine.device_ask import wanted_list

    assert wanted_list("Find the five gauges nearest to Kingston upon Thames") == 5
    assert wanted_list("the 3 nearest discharge stations to Paris") == 3
    assert wanted_list("Which gauges are on the Rhine?") == 2
    assert wanted_list("What is the 100-year flood at this gauge?") == 0
