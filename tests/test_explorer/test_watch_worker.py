"""The worker's watch Python (#521), run in CPython with a fake ``js`` module.

It is the exact code string worker.js hands to Pyodide, so this checks the message contract the "Since you were
here" panel relies on: one item's digest from what the page sends, a summary, and errors that come back as data.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from aquascope import watch

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "explorer" / "worker.js"


def _code() -> str:
    text = WORKER.read_text(encoding="utf-8")
    m = re.search(r"async function watchDigest\(.*?const code = \(k\) => `(.*?)`;", text, re.S)
    assert m, "watchDigest's Python block is missing from worker.js"
    # runPy hands the code its call key as a Python string literal
    return m.group(1).replace("${k}", '"k"')


@pytest.fixture
def run(monkeypatch):
    code = _code()

    def call(payload: dict):
        fake = types.ModuleType("js")
        args = json.dumps({"op": "digest", "items": [], "last_seen": {}, "snapshot": [], "issued": [],
                           "today": None, **payload})
        fake.__aqCall = lambda key, name: args if (key, name) == ("k", "args") else None
        monkeypatch.setitem(sys.modules, "js", fake)
        ns: dict = {}
        lines = code.strip().splitlines()
        exec("\n".join(lines[:-1]), ns)  # noqa: S102 - the worker's own code, under test
        return json.loads(eval(lines[-1], ns))  # noqa: S307 - the block's last expression, as Pyodide returns it

    return call


def test_one_item_digest_from_what_the_page_sends(run, monkeypatch):
    seen = {}

    def fake(items, last_seen, **kw):
        seen.update(items=items, last_seen=last_seen, **kw)
        return {"items": [{"id": "usgs/1", "line": "x"}], "summary": "s"}

    monkeypatch.setattr(watch, "watch_digest", fake)
    snap = [{"source": "usgs", "station_id": "1", "class": "normal"}]
    out = run({"items": [{"id": "usgs/1", "kind": "gauge", "source": "usgs", "station_id": "1"}],
               "last_seen": {"usgs/1": {"date": "2026-10-01"}}, "snapshot": snap, "today": "2026-10-08"})
    assert out["items"][0]["line"] == "x"
    assert seen["snapshot"] == snap and seen["issued"] == [] and seen["archive"] is False
    assert seen["today"] == "2026-10-08" and seen["last_seen"] == {"usgs/1": {"date": "2026-10-01"}}


def test_the_real_engine_answers_through_the_worker_code(run, monkeypatch):
    monkeypatch.setattr("aquascope.context.events.flood_history_area", lambda *a, **k: {
        "news": {"available": True, "recent": [{"start": "2026-10-03"}], "latest": "2026-10-03"}})
    out = run({"items": [{"id": "area:-77.5,38.1,-76.8,39", "kind": "area", "bbox": [-77.5, 38.1, -76.8, 39]}],
               "last_seen": {"area:-77.5,38.1,-76.8,39": {"date": "2026-10-01"}}, "today": "2026-10-08"})
    item = out["items"][0]
    assert item["id"] == "area:-77.5,38.1,-76.8,39" and item["floods"]["n_new"] == 1
    assert item["seen"]["date"] == "2026-10-08"


def test_summary_and_errors_as_data(run):
    out = run({"op": "summary", "items": [{"id": "a", "name": "A", "changed": True, "alerts": ["x"],
                                           "since": "2026-10-01"}], "today": "2026-10-08"})
    assert out["summary"].startswith("Since 1 October: 1 of 1 watched place changed")
    assert run({"op": "nope"}) == {"error": "unknown op"}
    out = run({"items": ["nonsense"]})
    assert out["errors"][0]["item"] == "nonsense" and out["items"] == []
