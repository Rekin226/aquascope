"""On-device Ask, round 3 (#590): the model picks a tool or fact keys, the package writes every sentence.

The replies replayed here are the ones Gemini Nano wrote in Chrome 155 on 2026-10-26 (the third check,
tests/fixtures/ondevice_ask_check3.json, docs/assessments/ai-on-the-map.md). The results they meet are fixtures
with the figures that check recorded: the Potomac at Little Falls (USGS-01646500), the point 25.05, 121.55 and
the Kingston search.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquascope.ai_engine.device_ask import (
    CONNECTIVES,
    MAX_PICKS,
    check_call,
    compact_result,
    display,
    fact_table,
    lock_answer,
    render_answer,
    step_budget,
    step_schema,
)
from aquascope.map_pick import schema_errors

RECORDS = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "ondevice_ask_check3.json")
                     .read_text(encoding="utf-8"))["records"]
ASK = [r for r in RECORDS if r["role"] == "ask"]

POTOMAC_NAME = "POTOMAC RIVER NEAR WASH, DC LITTLE FALLS PUMP STA"
#: The context line the page added in the check: the selected gauge, off screen, and the map's centre.
CONTEXT = (f"The user is looking at the gauge {POTOMAC_NAME} (source usgs, id USGS-01646500, 38.950, -77.127), "
           "selected earlier and now off screen; a map centred on 51.41, -0.30 at zoom 9.0.")


def _words(question: str) -> str:
    return f"{question}\n\nContext: {CONTEXT}"


def _potomac() -> dict:
    """analyze_station for USGS-01646500 with the figures the device saw (return levels, trends, FDC)."""
    gev = {"2": 2837.86, "5": 4420.08, "10": 5693.05, "25": 7613.84, "50": 9304.97, "100": 11247.39}
    lp3 = {"2": 2850.66, "5": 4520.46, "10": 5805.92, "25": 7635.82, "50": 9150.06, "100": 10795.57}
    return {
        "source": "usgs", "station_id": "USGS-01646500", "name": POTOMAC_NAME, "river": "Potomac River",
        "unit": "m3/s", "variable": "discharge", "start": "1930-03-01", "end": "2026-10-24",
        "stats": {"mean": 320.7738, "median": 185.0, "min": 3.43, "max": 12100.0},
        "fdc": {"q95": 35.7, "q50": 186.0, "q10": 702.0},
        "trend": {"on": "annual mean", "p_value": 0.9528, "trend": "no trend", "sens_slope_per_year": -0.0365,
                  "n_years": 96},
        "ffa": {"n_years": 96, "return_periods": [2, 5, 10, 25, 50, 100],
                "record_max": {"value": 12100.0, "year": 1936},
                "amax_trend": {"on": "annual maxima", "p_value": 0.9602, "trend": "no trend",
                               "sens_slope_per_year": -0.0543, "n_years": 96},
                "fits": {"gev_lmoments": {"q_by_T": gev},
                         "lp3": {"q_by_T": lp3, "ci": [[2600.0, 3100.0], [4100.0, 4950.0], [5250.0, 6400.0],
                                                       [6800.0, 8600.0], [8000.0, 10500.0], [9300.0, 12500.0]]}}},
    }


def _anywhere() -> dict:
    return {
        "latitude": 25.05, "longitude": 121.55, "start": "2006-10-18", "end": "2026-10-18",
        "climate": {"source": "ERA5", "precipitation_mm_per_year": 1733.4111, "et0_mm_per_year": 1168.2433,
                    "temperature_mean_c": 22.7201, "aridity_index": 1.4838, "aridity_class": "humid"},
        "glofas": {"unit": "m3/s", "variable": "discharge", "start": "2006-10-18", "end": "2026-10-18",
                   "modelled": True, "stats": {"mean": 5.6336, "min": 1.35, "max": 214.28},
                   "trend": {"on": "annual mean", "p_value": 0.184, "sens_slope_per_year": 0.031},
                   # Check3's escape 1: "significant trend in annual maxima with a p = 0.0689". It is not.
                   "ffa": {"amax_trend": {"on": "annual maxima", "p_value": 0.0689, "sens_slope_per_year": 1.92}}},
    }


KINGSTON = {"n_catalog": 60000, "n_returned": 5, "limit": 5, "stations": [
    {"source": "uk_ea", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston", "river": "Thames",
     "latitude": 51.4155, "longitude": -0.3076},
    {"source": "uk_ea", "station_id": "E02102A", "name": "KINGSTON VALE", "latitude": 51.43, "longitude": -0.26},
    {"source": "uk_ea", "station_id": "3400TH", "name": "HOGSMILL AT EWELL", "latitude": 51.35, "longitude": -0.25},
    {"source": "uk_ea", "station_id": "3990TH", "name": "KINGSTON BRIDGE", "latitude": 51.41, "longitude": -0.31},
    {"source": "uk_ea", "station_id": "E21778", "name": "TEDDINGTON", "latitude": 51.43, "longitude": -0.32},
]}


def _thames() -> dict:
    return {"source": "uk_ea", "station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099", "name": "Kingston",
            "river": "Thames", "unit": "m3/s", "variable": "discharge", "start": "1883-10-01", "end": "2026-10-09",
            "stats": {"mean": 65.48, "median": 40.1, "min": 0.92, "max": 800.0},
            "fdc": {"q95": 7.5, "q50": 39.9, "q10": 160.0}}


CATCHMENT = {"latitude": 51.4155, "longitude": -0.3076, "attributes": {
    "area_km2": 9948.0, "elevation_m": {"value": 112.4, "unit": "m", "label": "mean elevation"},
    "urban_pct": {"value": 14.2, "unit": "%", "label": "urban extent"},
    "population": {"value": 1250000.0, "unit": "people", "label": "population count"}}}


def _tool(name: str, args: dict) -> dict:
    """The tools, as fixtures: what each returned on the device for these arguments."""
    if name == "analyze_station":
        return _potomac() if args["station_id"] == "USGS-01646500" else _thames()
    if name == "anywhere":
        return _anywhere()
    if name == "find_stations":
        return KINGSTON
    return CATCHMENT


def _parse(text: str) -> dict:
    raw = str(text or "").strip()
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        return {"answer": raw}


def drive(replies: list, words: str, run=_tool) -> dict:
    """The page's loop (explorer/src/ask-local-core.js askLoop) over recorded replies, with the package's ops."""
    results: list[dict] = []
    notes: list[str] = []
    force, budget = False, None
    queue = list(replies)
    calls = []
    for step in range(1, 9):
        plan = step_schema(results, words, step=step, budget=budget, force_answer=force, notes=notes)
        budget = plan["budget"]
        if plan["mode"] == "none":
            return {**render_answer(None, results, words, why="nothing in your words to look up"), "calls": calls,
                    "plans": step}
        if not queue:
            return {**render_answer(None, results, words, why="the recording ends here"), "calls": calls,
                    "plans": step}
        reply = queue.pop(0)
        reply = _parse(reply) if isinstance(reply, str) else reply
        if plan["mode"] == "answer" or ("answer" in reply and not reply.get("tool")):
            return {**render_answer(reply, results, words), "calls": calls, "plans": step}
        chk = check_call(reply, results, words)
        calls.append(chk)
        if chk.get("repeat"):
            force = True
            notes.append(chk["refused"])
            continue
        if not chk["ok"]:
            notes.append(chk["refused"])
            continue
        results.append({"name": chk["name"], "arguments": chk["arguments"], "ok": True,
                        "compact": compact_result(chk["name"], run(chk["name"], chk["arguments"]))})
    return {**render_answer(None, results, words, why="steps used"), "calls": calls, "plans": 8}


def _written(results: list[dict]) -> set[str]:
    """Every sentence the package could write for these results, with either form of each gauge's name."""
    table = fact_table(results)
    out = set(CONNECTIVES.values())
    for f in table["facts"]:
        for name in table["gauges"].get(f.get("gauge"), [""]) if f.get("gauge") else [""]:
            line = f["text"].replace("{who}", name)
            out.add(line[:1].upper() + line[1:])
    return out


def _results_of(out: dict) -> list[dict]:
    return [{"name": c["name"], "arguments": c["arguments"], "ok": True,
             "compact": compact_result(c["name"], _tool(c["name"], c["arguments"]))}
            for c in out["calls"] if c["ok"]]


# ── the recorded replies, replayed ────────────────────────────────────────────


@pytest.mark.parametrize("rec", ASK, ids=[r["tag"] for r in ASK])
def test_no_recorded_reply_reaches_the_page_and_every_question_shows_facts(rec: dict) -> None:
    out = drive(rec["replies"], _words(rec["request"]))
    shown = out["sentences"] if out["ok"] else out["facts"]
    allowed = _written(_results_of(out))
    for line in shown:
        assert line in allowed, (rec["tag"], line)   # only the package's sentences, never the model's text
    # The recorded replies are the old shape: a free-text answer is refused before display, and a loop of
    # repeated calls ends on the facts gathered, not on a stock line with nothing under it.
    assert not out["ok"] and out["facts"], (rec["tag"], out)
    for reply in rec["replies"]:
        text = _parse(reply).get("answer")
        if isinstance(text, str):
            assert text not in " ".join(shown)


def test_the_example_id_is_never_run() -> None:
    rec = next(r for r in ASK if r["tag"] == "ask-compare")
    words = _words(rec["request"])
    out = check_call(_parse(rec["replies"][1]), [], words)
    assert not out["ok"] and "'abc'" in out["refused"]
    plan = step_schema([], words)
    assert "abc" not in json.dumps(plan["schema"]) and "abc" not in plan["prompt"]


def test_a_repeated_call_is_not_run_and_the_next_step_answers() -> None:
    rec = next(r for r in ASK if r["tag"] == "ask-summary-1")   # analyze_station three times
    words = _words(rec["request"])
    out = drive(rec["replies"], words)
    assert sum(1 for c in out["calls"] if c["ok"]) == 1
    assert out["calls"][1]["repeat"] is True
    # After the repeat the only reply the schema allows is an answer.
    results = _results_of(out)
    plan = step_schema(results, words, step=3, force_answer=True)
    assert plan["mode"] == "answer" and set(plan["schema"]["properties"]) == {"answer"}
    # The recorded third reply is a tool call where the answer had to be: refused, and the facts are shown.
    assert "tool call" in out["refused"][0]
    assert any("no significant trend" in f for f in out["facts"])


# ── a correct intent gives a correct sentence ─────────────────────────────────

P = "USGS-01646500"
AT = "climate@25.05,121.55"
GL = "glofas@25.05,121.55"
ANALYSE = {"tool": "analyze_station", "arguments": {"source": "usgs", "station_id": P}}

#: Each recorded question with the reply the round-3 schema allows for what the device meant to say.
INTENTS = {
    "ask-summary-2": ([ANALYSE, {"answer": [f"{P}.record", f"{P}.mean", f"{P}.trend.mean", f"{P}.trend.max",
                                            f"{P}.fdc.q95", f"{P}.gev.100", f"{P}.lp3.100", "say.floods"]}],
                      ["no significant trend in the annual mean", "100-year flood", "11,247", "10,796"]),
    "ask-100yr-1": ([ANALYSE, {"answer": [f"{P}.gev.100", f"{P}.lp3.100", f"{P}.amax"]}],
                    ["11,247 m3/s (GEV", "10,796 m3/s by Log-Pearson III", "12,100 m3/s, in 1936"]),
    "ask-trend-1": ([ANALYSE, {"answer": [f"{P}.trend.mean"]}],
                    ["no significant trend in the annual mean (Mann-Kendall p = 0.953; Sen's slope -0.0365"]),
    "ask-trend-2": ([ANALYSE, {"answer": [f"{P}.trend.mean", f"{P}.trend.max"]}],
                    ["-0.0365 m3/s per year", "-0.0543 m3/s per year"]),
    "ask-anywhere": ([{"tool": "anywhere", "arguments": {"lat": 25.05, "lon": 121.55}},
                      {"answer": [f"{AT}.rain", f"{AT}.temp", f"{GL}.record", f"{GL}.mean", f"{GL}.trend.mean",
                                  f"{GL}.trend.max", "say.modelled"]}],
                     ["no significant trend in the annual maxima (Mann-Kendall p = 0.0689", "1,733 mm a year",
                      "at 25.05, 121.55"]),
    "ask-new-q95": ([ANALYSE, {"answer": [f"{P}.fdc.q95", f"{P}.median"]}], ["35.7 m3/s", "185 m3/s"]),
    "ask-new-report": ([ANALYSE, {"answer": [f"{P}.record", f"{P}.mean", f"{P}.amax", f"{P}.trend.mean"]}],
                       ["1930-03-01 to 2026-10-24", "321 m3/s", "12,100"]),
    "ask-trend-max": ([ANALYSE, {"answer": [f"{P}.trend.max"]}], ["no significant trend in the annual maxima"]),
    "ask-50yr": ([ANALYSE, {"answer": [f"{P}.gev.50", f"{P}.lp3.50"]}], ["9,305", "9,150"]),
    "ask-kingston": ([{"tool": "find_stations", "arguments": {"query": "Kingston", "limit": 5}},
                      {"answer": ["search.count", "8496ce69-482c-406a-a2f0-ac418ef8f099.where", "E02102A.where",
                                  "3400TH.where", "3990TH.where", "E21778.where"]}],
                     ["returned 5 gauges", "KINGSTON VALE (uk_ea E02102A), at 51.43, -0.26",
                      "TEDDINGTON (uk_ea E21778)"]),
    "ask-compare": ([ANALYSE, {"tool": "find_stations", "arguments": {"query": "Thames", "limit": 5}},
                     {"tool": "analyze_station", "arguments": {"station_id": "8496ce69-482c-406a-a2f0-ac418ef8f099"}},
                     {"answer": [f"{P}.mean", "8496ce69-482c-406a-a2f0-ac418ef8f099.mean"]}],
                    ["321 m3/s", "65.5 m3/s"]),
}


@pytest.mark.parametrize("tag", sorted(INTENTS))
def test_the_recorded_intent_is_written_right(tag: str) -> None:
    rec = next(r for r in ASK if r["tag"] == tag)
    replies, says = INTENTS[tag]
    out = drive(replies, _words(rec["request"]))
    assert out["ok"], (tag, out["refused"])
    text = out["text"]
    for bit in says:
        assert bit in text, (tag, bit, text)
    assert "e+0" not in text and "{" not in text
    assert all(line in _written(_results_of(out)) for line in out["sentences"])


def test_escape_1_cannot_be_written() -> None:
    """Check3 blocking 1: "significant trend in annual maxima with a p = 0.0689". No key says that."""
    words = _words("What is the climate and modelled discharge at 25.05, 121.55?")
    out = drive(INTENTS["ask-anywhere"][0], words)
    assert "a significant" not in out["text"]
    every = " ".join(_written(_results_of(out)))
    assert "a significant" not in every   # no sentence the package could write for these results claims it


def test_the_label_false_refusals_of_check3_are_gone() -> None:
    """"100-year" read as a gauge id and a p-value or Sen's slope read as the daily mean: the recorded right
    answers now pass the old lock too, which stays as the second line."""
    res = [{"name": "analyze_station", "arguments": {"source": "usgs", "station_id": P}, "ok": True,
            "compact": compact_result("analyze_station", _potomac())}]
    wrongly = {"ask-summary-2": 1, "ask-summary-3": 1, "ask-trend-1": 1, "ask-trend-2": 1, "ask-trend-max": 1}
    for tag in wrongly:
        rec = next(r for r in ASK if r["tag"] == tag)
        answer = _parse(rec["replies"][1])["answer"]
        out = lock_answer(answer, res, _words(rec["request"]))
        assert not any("100-year" in r or "is not the daily mean" in r for r in out["refused"]), (tag, out["refused"])


# ── the schema ────────────────────────────────────────────────────────────────


def test_each_step_offers_only_enums_built_from_this_question() -> None:
    words = _words("Summarise the record at this gauge: period, mean, trend, and the flood frequency")
    first = step_schema([], words)
    assert first["mode"] == "tool" and "answer" not in first["schema"]["properties"]
    assert first["schema"]["required"] == ["tool", "arguments"]
    args = first["schema"]["properties"]["arguments"]["properties"]
    assert args["station_id"]["enum"] == [P] and args["source"]["enum"] == ["usgs"]
    assert set(first["schema"]["properties"]["tool"]["enum"]) <= {"find_stations", "analyze_station", "anywhere",
                                                                  "describe_catchment"}
    assert "51.41" not in first["prompt"].split("Tools you may call")[1]   # the map's centre is no place to look up
    res = [{"name": "analyze_station", "arguments": {"source": "usgs", "station_id": P}, "ok": True,
            "compact": compact_result("analyze_station", _potomac())}]
    mid = step_schema(res, words, step=2)
    assert mid["mode"] == "either"
    keys = mid["schema"]["properties"]["answer"]["items"]["enum"]
    assert set(keys) == {f["key"] for f in fact_table(res)["facts"]} | set(CONNECTIVES)
    assert {f"{P}.trend.mean", f"{P}.fdc.q95", f"{P}.gev.100", f"{P}.record"} <= set(keys)
    assert mid["schema"]["properties"]["answer"]["maxItems"] == MAX_PICKS
    # The gauge already analysed is no longer offered.
    assert "station_id" not in mid["schema"]["properties"]["arguments"]["properties"]
    last = step_schema(res, words, step=3)
    assert last["mode"] == "answer" and last["schema"]["required"] == ["answer"]
    assert set(last["schema"]["properties"]) == {"answer"}


def test_the_step_budget() -> None:
    assert step_budget("Summarise the record at this gauge") == 3
    assert step_budget("Is the Potomac at Little Falls getting drier? Use the annual-mean trend.") == 3
    assert step_budget("Compare the mean flow of the Potomac (USGS-01646500) with the Thames at Kingston") == 5
    assert step_budget("Find the five gauges nearest to Kingston upon Thames") == 5


def test_nothing_to_look_up_says_so() -> None:
    plan = step_schema([], "What is a flood?")
    assert plan["mode"] == "none"


@pytest.mark.parametrize("reply, why", [
    ({"answer": "The 100-year flood is 11247 m3/s."}, "free text"),
    ({"answer": [f"{P}.gev.1000"]}, "not among the facts"),
    ({"answer": ["say.floods"]}, "no fact"),
    ({"answer": []}, "no facts"),
    ({"tool": "analyze_station", "arguments": {}}, "tool call"),
    (None, "no facts"),
])
def test_anything_outside_the_enums_is_refused_before_display(reply, why: str) -> None:
    res = [{"name": "analyze_station", "arguments": {}, "ok": True,
            "compact": compact_result("analyze_station", _potomac())}]
    out = render_answer(reply, res, _words("What is the 100-year flood at this gauge?"))
    assert not out["ok"] and why in out["refused"][0]
    assert out["sentences"] == [] and out["facts"]


@pytest.mark.parametrize("reply, ok", [
    ({"tool": "analyze_station", "arguments": {"source": "nrfa", "station_id": "usgs-01646500"}}, True),
    ({"tool": "analyze_station", "arguments": {"source": "usgs", "station_id": "100-year"}}, False),
    ({"tool": "anywhere", "arguments": {"lat": 25.05, "lon": 121.55}}, False),       # not in these words
    ({"tool": "anywhere", "arguments": {"lat": 38.95, "lon": -77.127}}, True),       # the selected gauge
    ({"tool": "find_stations", "arguments": {"query": "Nile"}}, False),
    ({"tool": "find_stations", "arguments": {"near": [51.1, 0.0]}}, False),           # check1's made-up point
    ({"tool": "describe_catchment", "arguments": {"lat": "x"}}, False),
    ({"tool": "run_python", "arguments": {}}, False),
])
def test_tool_arguments_come_from_the_words_or_the_results(reply: dict, ok: bool) -> None:
    out = check_call(reply, [], _words("What is the 100-year flood at this gauge?"))
    assert out["ok"] is ok, out
    if ok and reply["tool"] == "analyze_station":
        assert out["arguments"] == {"source": "usgs", "station_id": P}   # the context's source, not the model's


def test_a_search_result_gives_the_ids_and_points_the_next_call_may_use() -> None:
    words = "Find the five gauges nearest to Kingston upon Thames"
    res = [{"name": "find_stations", "arguments": {"query": "Kingston", "limit": 5}, "ok": True,
            "compact": compact_result("find_stations", KINGSTON)}]
    assert check_call({"tool": "analyze_station", "arguments": {"station_id": "E02102A"}}, res, words)["ok"]
    assert check_call({"tool": "describe_catchment", "arguments": {"lat": 51.4155, "lon": -0.3076}}, res, words)["ok"]
    assert check_call({"tool": "find_stations", "arguments": {"query": "Kingston upon Thames"}}, [], words)["ok"]


# ── the package's own sentences ───────────────────────────────────────────────


def _all_results() -> list[dict]:
    return [
        {"name": "analyze_station", "arguments": {"source": "usgs", "station_id": P}, "ok": True,
         "compact": compact_result("analyze_station", _potomac())},
        {"name": "anywhere", "arguments": {"lat": 25.05, "lon": 121.55}, "ok": True,
         "compact": compact_result("anywhere", _anywhere())},
        {"name": "find_stations", "arguments": {"query": "Kingston", "limit": 5}, "ok": True,
         "compact": compact_result("find_stations", KINGSTON)},
        {"name": "describe_catchment", "arguments": {"lat": 51.4155, "lon": -0.3076}, "ok": True,
         "compact": compact_result("describe_catchment", CATCHMENT)},
    ]


def test_every_fact_names_its_gauge_value_and_unit_and_passes_the_second_line() -> None:
    res = _all_results()
    words = "What is the climate at 25.05, 121.55, and the record of USGS-01646500, and the gauges near Kingston?"
    table = fact_table(res)
    assert len(table["facts"]) > 40
    for f in table["facts"]:
        out = render_answer({"answer": [f["key"]]}, res, words)
        assert out["ok"], (f["key"], out["refused"])
        line = out["text"]
        assert "e+" not in line and "{" not in line and "None" not in line, line
        if f["key"].startswith(P) and f["key"] != f"{P}.record":
            assert "USGS-01646500" in line and "m3/s" in line, line
        if ".gev." in f["key"]:
            assert "GEV" in line
        if ".lp3." in f["key"]:
            assert "LP3" in line


def test_the_gauge_is_named_in_full_once_then_by_its_id() -> None:
    res = _all_results()[:1]
    out = render_answer({"answer": [f"{P}.mean", f"{P}.fdc.q95"]}, res, "")
    assert out["sentences"][0].count(POTOMAC_NAME) == 1
    assert POTOMAC_NAME not in out["sentences"][1] and P in out["sentences"][1]


@pytest.mark.parametrize("x, shown", [
    (11247.39, "11,247"), (320.7738, "321"), (35.7, "35.7"), (3.43, "3.43"), (-0.0365, "-0.0365"),
    (0.000012, "0.000012"), (1733.4111, "1,733"), (0, "0"), (5.6336, "5.63"), (96, "96"),
])
def test_display_never_writes_an_exponent(x: float, shown: str) -> None:
    assert display(x) == shown


@pytest.mark.parametrize("token", ["100-year", "100yr", "1930s", "7-day"])
def test_a_period_in_the_words_is_not_a_gauge_id(token: str) -> None:
    words = f"What is the {token} flood at E02102A since 1930-03-01?"
    assert not check_call({"tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": token}}, [],
                          words)["ok"]
    assert check_call({"tool": "analyze_station", "arguments": {"source": "uk_ea", "station_id": "E02102A"}}, [],
                      words)["ok"]


# ── the fourth check (2026-10-26, tests/fixtures/ondevice_ask_check4.json): a closed step schema ──────────────

CHECK4 = [r for r in json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "ondevice_ask_check4.json")
                                .read_text(encoding="utf-8"))["records"] if r["role"] == "ask"]


def _closed(schema: dict, path: str = "schema") -> list[str]:
    """Where a schema is open: an object without additionalProperties false, or a value with no enum."""
    bad = []
    kind = schema.get("type")
    if kind == "object":
        if schema.get("additionalProperties") is not False:
            bad.append(f"{path} takes any key")
        for k, sub in schema.get("properties", {}).items():
            bad += _closed(sub, f"{path}.{k}")
    elif kind == "array":
        bad += _closed(schema["items"], f"{path}[]")
    elif "enum" not in schema:
        bad.append(f"{path} is a free {kind}")
    return bad


def _plans(rec: dict) -> list[dict]:
    """Every step's plan for one recorded question, as the page's loop builds them over the recorded replies."""
    words, results, notes, plans = _words(rec["request"]), [], [], []
    for step, raw in enumerate(rec["replies"], start=1):
        plan = step_schema(results, words, step=step, notes=notes)
        plans.append(plan)
        reply = _parse(raw)
        if plan["mode"] in ("none", "answer") or not reply.get("tool"):
            break
        chk = check_call(reply, results, words)
        if chk["ok"]:
            results.append({"name": chk["name"], "arguments": chk["arguments"], "ok": True,
                            "compact": compact_result(chk["name"], _tool(chk["name"], chk["arguments"]))})
        else:
            notes.append(chk["refused"])
    return plans


@pytest.mark.parametrize("rec", CHECK4 + ASK, ids=[f"check4-{r['tag']}" for r in CHECK4] + [r["tag"] for r in ASK])
def test_every_step_schema_is_closed(rec: dict) -> None:
    for plan in _plans(rec):
        assert _closed(plan["schema"]) == [], (rec["tag"], plan["mode"], _closed(plan["schema"]))


def test_each_tool_takes_only_its_own_arguments_and_points_only_known_pairs() -> None:
    words = _words("What is the climate and modelled discharge at 25.05, 121.55?")
    args = step_schema([], words)["schema"]["properties"]["arguments"]["properties"]
    assert args["lat"]["enum"] == [25.05, 38.95] and args["lon"]["enum"] == [121.55, -77.127]
    assert args["near"]["items"]["enum"] == [25.05, 121.55, 38.95, -77.127]
    assert args["limit"]["enum"] == list(range(1, 11))
    # No point in the words or results: no lat, lon or near at all.
    only_ids = step_schema([], "Summarise the record at USGS-01646500")["schema"]["properties"]["arguments"]
    assert set(only_ids["properties"]) == {"station_id"}
    # A point already looked up by both point tools is not offered to them again (the catchment's own point is).
    res = [{"name": t, "arguments": {"lat": 25.05, "lon": 121.55}, "ok": True,
            "compact": compact_result(t, _anywhere() if t == "anywhere" else CATCHMENT)}
           for t in ("anywhere", "describe_catchment")]
    later = step_schema(res, "What is the climate at 25.05, 121.55?", step=2)
    assert later["schema"]["properties"]["arguments"]["properties"]["lat"]["enum"] == [51.4155]


def test_check4_kingston_calls_cannot_be_written_under_the_closed_schema() -> None:
    """Four searches, each at a made-up point with an invented key ("limit__", "limittypicode"): the schema the
    step now gives allows none of them, and the call the words hold still fits."""
    rec = next(r for r in CHECK4 if r["tag"] == "ask-kingston")
    plan = step_schema([], _words(rec["request"]))
    for raw in rec["replies"]:
        assert schema_errors(_parse(raw), plan["schema"]), raw
    assert schema_errors({"tool": "find_stations", "arguments": {"query": "Kingston upon Thames", "limit": 5}},
                         plan["schema"]) == []
    assert "near one of [38.95, -77.127]" in plan["prompt"]


def test_check4_right_calls_still_fit_the_schema() -> None:
    for rec in CHECK4:
        if rec["tag"] in ("ask-kingston", "ask-compare", "ask-100yr-2", "ask-new-list-thames"):
            continue
        plan = step_schema([], _words(rec["request"]))
        assert schema_errors(_parse(rec["replies"][0]), plan["schema"]) == [], rec["tag"]


@pytest.mark.parametrize("question, bad", [
    ("List the gauges on the Thames near Kingston and their mean flows", {"List"}),
    ("Compare the mean flow of the Potomac (USGS-01646500) with the Thames at Kingston", {"USGS-", "Potomac USGS-"}),
    ("Is the Potomac at Little Falls getting drier? Use the annual-mean trend.", {"Use"}),
])
def test_search_phrases_are_names_never_an_opening_verb_or_an_ids_letters(question: str, bad: set) -> None:
    plan = step_schema([], _words(question))
    offered = set(plan["schema"]["properties"]["arguments"]["properties"]["query"]["enum"])
    assert not offered & bad, offered
    assert offered & {"Thames near Kingston", "Thames at Kingston", "Potomac at Little Falls"}


@pytest.mark.parametrize("rec", CHECK4, ids=[r["tag"] for r in CHECK4])
def test_check4_replies_replayed_reach_the_page_only_as_the_packages_sentences(rec: dict) -> None:
    out = drive(rec["replies"], _words(rec["request"]))
    shown = out["sentences"] if out["ok"] else out["facts"]
    allowed = _written(_results_of(out))
    for line in shown:
        assert line in allowed, (rec["tag"], line)


def test_check4_the_packages_own_trend_connective_is_no_claim() -> None:
    """ask-summary-2: the right picks with say.daily, say.floods and say.trend were refused because the second
    line read "a p-value below 0.05 is called significant" as a claim. Now the answer is written."""
    rec = next(r for r in CHECK4 if r["tag"] == "ask-summary-2")
    out = drive(rec["replies"], _words(rec["request"]))
    assert out["ok"], out["refused"]
    assert CONNECTIVES["say.trend"] in out["sentences"]
    assert any("no significant trend in the annual mean" in s for s in out["sentences"])
    # A claim next to it is still held: the trend is not significant, so "a significant trend" is refused.
    res = _results_of(out)
    wrong = CONNECTIVES["say.trend"] + " There is a significant upward trend in the annual mean."
    assert not lock_answer(wrong, res, _words(rec["request"]))["ok"]
