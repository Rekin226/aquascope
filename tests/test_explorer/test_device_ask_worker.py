"""The worker's on-device Ask Python (#590), run in CPython with a fake ``js`` module.

It is the exact code string worker.js hands to Pyodide: op "plan" is a step's schema of enums, op "tool" checks
the model's call and runs one analyst tool, returning only the compact result, op "render" writes the answer
from the fact keys the model picked, and op "lock" (the second line) holds free text to the results. The map's
reply check gets the request's words, so a box the model made up can be refused.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from aquascope.ai_engine import analyst

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "explorer" / "worker.js"


def _block(fn: str) -> str:
    text = WORKER.read_text(encoding="utf-8")
    m = re.search(r"async function " + fn + r"\(.*?const code = \(k\) => `(.*?)`;", text, re.S)
    assert m, f"{fn}'s Python block is missing from worker.js"
    # runPy hands the code its call key as a Python string literal
    return m.group(1).replace("${k}", '"k"')


def _runner(monkeypatch, fn: str):
    code = _block(fn)

    def call(payload: dict):
        fake = types.ModuleType("js")
        args = json.dumps(payload)
        fake.__aqCall = lambda key, name: args if (key, name) == ("k", "args") else None
        monkeypatch.setitem(sys.modules, "js", fake)
        ns: dict = {}
        lines = code.strip().splitlines()
        exec("\n".join(lines[:-1]), ns)  # noqa: S102 - the worker's own code, under test
        return json.loads(eval(lines[-1], ns))  # noqa: S307 - the block's last expression, as Pyodide returns it

    return call


@pytest.fixture
def device(monkeypatch):
    return _runner(monkeypatch, "deviceAsk")


WORDS = ("What is the 100-year flood at this gauge?\n\nContext: The user is looking at the gauge POTOMAC (source "
         "usgs, id 01646500, 38.950, -77.127).")
CALL = {"tool": "analyze_station", "arguments": {"source": "usgs", "station_id": "01646500"}}


def test_the_tool_op_returns_the_compact_result_not_the_payload(device, monkeypatch):
    payload = {"source": "usgs", "station_id": "01646500", "unit": "m3/s", "start": "1930-03-01",
               "end": "2026-10-24", "stats": {"mean": 320.7738, "min": 3.43, "max": 12100.0},
               "annual_max": {"year": list(range(1930, 2026)), "v": [1.0] * 96},
               "methods": [{"name": "x", "text": "y" * 500, "citation": "z"}],
               "trend": {"on": "annual mean", "p_value": 0.82, "trend": "no trend", "sens_slope_per_year": 0.059}}
    seen = {}
    spec = analyst.ToolSpec("analyze_station", "d", {"type": "object"}, lambda **kw: seen.update(kw) or payload)
    monkeypatch.setattr(analyst, "_tool_specs", lambda: [spec])
    out = device({"op": "tool", "reply": CALL, "results": [], "words": WORDS})
    assert out["ok"] is True and out["arguments"] == {"source": "usgs", "station_id": "01646500"}
    assert seen == out["arguments"]
    assert "annual_max" not in out["compact"] and "methods" not in out["compact"]
    assert out["compact"]["trend_annual_mean"]["p_value"] == 0.82
    # The plan op then offers the facts by key, and the render op writes the ones the model picks.
    res = [{"name": out["name"], "arguments": out["arguments"], "compact": out["compact"], "ok": True}]
    plan = device({"op": "plan", "results": res, "words": WORDS, "step": 2})
    assert plan["mode"] == "either" and "01646500.trend.mean" in plan["schema"]["properties"]["answer"]["items"]["enum"]
    said = device({"op": "render", "reply": {"answer": ["01646500.trend.mean"]}, "results": res, "words": WORDS})
    assert said["ok"] and "no significant trend in the annual mean" in said["text"]


def test_a_call_the_words_do_not_hold_is_not_run(device, monkeypatch):
    def boom(**kw):
        raise AssertionError("must not run")

    monkeypatch.setattr(analyst, "_tool_specs", lambda: [analyst.ToolSpec("analyze_station", "d", {}, boom)])
    out = device({"op": "tool", "reply": {"tool": "analyze_station", "arguments": {"station_id": "abc"}},
                  "results": [], "words": WORDS})
    assert out["ok"] is False and "'abc'" in out["refused"] and "compact" not in out


def test_a_failing_tool_comes_back_as_data(device, monkeypatch):
    def boom(**kw):
        raise ValueError("no such station")

    monkeypatch.setattr(analyst, "_tool_specs", lambda: [analyst.ToolSpec("analyze_station", "d", {}, boom)])
    out = device({"op": "tool", "reply": CALL, "results": [], "words": WORDS})
    assert out["ok"] is False and "no such station" in out["compact"]["error"]


def test_the_lock_op_refuses_the_recorded_trend_with_no_test(device):
    out = device({"op": "lock", "answer": "The Mann-Kendall test indicates an upward trend.", "results": [],
                  "words": "Is the Potomac at Little Falls getting drier? Use the annual-mean trend."})
    assert out["ok"] is False and out["refused"]


def test_the_map_reply_op_hands_the_words_to_the_check(monkeypatch):
    run = _runner(monkeypatch, "mapCommand")
    reply = json.dumps({"actions": [{"type": "fly_to", "bbox": [15, 20, 30, 40]}]})
    out = run({"op": "reply", "reply": reply, "text": "where is it driest in Africa right now? zoom there",
               "today": "2026-10-25", "context": "", "actions": [], "llm": {}})
    assert out["actions"][0]["region"] == "africa"
    assert out["actions"][0]["bbox"] == [-18.0, -35.0, 52.0, 37.5]


def test_the_page_sends_what_the_worker_reads():
    ask = (ROOT / "explorer" / "src" / "ask.js").read_text(encoding="utf-8")
    cmd = (ROOT / "explorer" / "src" / "map-command.js").read_text(encoding="utf-8")
    studio = (ROOT / "explorer" / "src" / "studio.js").read_text(encoding="utf-8")
    assert 'call("device_ask", { op, ...extra })' in ask
    for op in ('ask("plan", { results, words, step, budget, force, notes })', 'ask("tool", { reply, results, words })',
               'ask("render", { reply, results, words, why })'):
        assert op in ask
    assert '{ op: "reply", reply, text, today: todayIso() }' in cmd
    assert "parseBriefReply(reply, text)" in studio
    worker = WORKER.read_text(encoding="utf-8")
    assert 'if (m.type === "device_ask") return await deviceAsk(m);' in worker


def test_the_map_prompt_op_gives_the_requests_own_schema(monkeypatch):
    run = _runner(monkeypatch, "mapCommand")
    out = run({"op": "prompt", "text": "can you show me how the Danube is doing compared with normal",
               "today": "2026-10-26", "context": "", "reply": None, "actions": [], "llm": {}})
    item = out["schema"]["properties"]["actions"]["items"]["properties"]
    assert item["place"]["enum"] == ["Danube"] and item["layer"]["enum"] == ["status"]
    plain = run({"op": "prompt", "text": "", "today": "2026-10-26", "context": "", "reply": None, "actions": [],
                 "llm": {}})
    assert "enum" not in plain["schema"]["properties"]["actions"]["items"]["properties"]["place"]
