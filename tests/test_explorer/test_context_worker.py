"""The worker's place-context Python (#520), run in CPython with a fake ``js`` module.

It is the exact code string worker.js hands to Pyodide, so this checks the message contract the Context card
relies on: one layer at a point, one layer over a box, and errors that come back as data.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from aquascope import context

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "explorer" / "worker.js"


def _code() -> str:
    text = WORKER.read_text(encoding="utf-8")
    m = re.search(r"async function placeContext\(.*?const code = \(k\) => `(.*?)`;", text, re.S)
    assert m, "placeContext's Python block is missing from worker.js"
    # runPy hands the code its call key as a Python string literal
    return m.group(1).replace("${k}", '"k"')


@pytest.fixture
def run(monkeypatch):
    code = _code()

    def call(payload: dict):
        fake = types.ModuleType("js")
        args = json.dumps({"op": "point", "name": "", "lat": None, "lon": None, "bbox": None, **payload})
        fake.__aqCall = lambda key, name: args if (key, name) == ("k", "args") else None
        monkeypatch.setitem(sys.modules, "js", fake)
        ns: dict = {}
        lines = code.strip().splitlines()
        exec("\n".join(lines[:-1]), ns)  # noqa: S102 - the worker's own code, under test
        return json.loads(eval(lines[-1], ns))  # noqa: S307 - the block's last expression, as Pyodide returns it

    return call


def test_one_layer_at_a_point(run, monkeypatch):
    seen = {}

    def fake_dams(lat, lon, **kw):
        seen["dams"] = (lat, lon)
        return {"layer": "dams", "summary": "No dams.", "n_dams": 0}

    monkeypatch.setitem(context.LAYERS, "dams", fake_dams)
    out = run({"op": "point", "name": "dams", "lat": 45.1, "lon": 5.1})
    assert out == {"layer": "dams", "summary": "No dams.", "n_dams": 0} and seen["dams"] == (45.1, 5.1)


def test_one_layer_over_a_box_and_errors_as_data(run, monkeypatch):
    monkeypatch.setitem(context.AREA_LAYERS, "soil", lambda w, s, e, n: {"layer": "soil", "bbox": [w, s, e, n]})
    assert run({"op": "area", "name": "soil", "bbox": [4, 44, 6, 46]})["bbox"] == [4, 44, 6, 46]
    assert "west, south, east, north" in run({"op": "area", "name": "soil", "bbox": [6, 44, 4, 46]})["error"]
    assert "unknown context layer" in run({"op": "point", "name": "rivers", "lat": 1, "lon": 1})["error"]
    assert run({"op": "nope"}) == {"error": "unknown op"}


def test_the_page_asks_for_every_layer_the_engine_has():
    view = (ROOT / "explorer" / "src" / "context-view.js").read_text(encoding="utf-8")
    ids = re.findall(r'\{ id: "([a-z_]+)", label:', view)
    assert ids == list(context.LAYERS) and set(ids) == set(context.AREA_LAYERS)
