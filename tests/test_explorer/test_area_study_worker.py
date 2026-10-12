"""The worker's area-study Python, run in CPython with a fake ``js`` module and a mocked Archive.

It is the exact code string worker.js hands to Pyodide, so this checks the
message contract the page relies on: run (with progress events), areas, csv,
xlsx and an unknown op, over one persistent namespace as in the worker.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquascope import area_study

WORKER = Path(__file__).resolve().parents[2] / "explorer" / "worker.js"


def _code() -> str:
    text = WORKER.read_text(encoding="utf-8")
    m = re.search(r"async function areaStudy\(.*?const code = `(.*?)`;", text, re.S)
    assert m, "areaStudy's Python block is missing from worker.js"
    return m.group(1)


def _daily(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-01", "2019-12-31", freq="D")
    s = pd.Series(rng.random(len(idx)), index=idx)
    for y in range(1980, 2020):
        s[pd.Timestamp(f"{y}-03-01")] = 10 * (1 + 0.3 * rng.gumbel())
    return s


@pytest.fixture
def run(monkeypatch):
    code = _code()
    ns: dict = {}
    events: list = []
    monkeypatch.setattr(area_study, "_archive_reader", lambda src, sid, var: _daily(int(sid)))

    def call(payload: dict):
        fake = types.ModuleType("js")
        fake.__aqArea = json.dumps({"stations": [], "question": None, "max_live": None, "areas": {}, **payload})
        fake.__aqAreaEvent = lambda text: events.append(json.loads(text))
        monkeypatch.setitem(sys.modules, "js", fake)
        lines = code.strip().splitlines()
        exec("\n".join(lines[:-1]), ns)  # noqa: S102 - the worker's own code, under test
        return json.loads(eval(lines[-1], ns))  # noqa: S307 - the block's last expression, as Pyodide returns it

    call.events = events
    return call


def test_run_then_areas_then_downloads(run):
    stations = [{"source": "usgs", "station_id": str(i), "latitude": 40 + i / 10, "longitude": -75,
                 "variables": ["discharge"], "period_start": "1980-01-01"} for i in range(4)]
    res = run({"op": "run", "stations": stations})
    assert res["summary"]["n_archive"] == 4 and res["regional_frequency"]["n_sites"] == 4
    assert {e["phase"] for e in run.events} >= {"fetch", "regional"}

    res2 = run({"op": "areas", "areas": {"usgs/0": 500.0}})
    row = next(r for r in res2["sites"] if r["key"] == "usgs/0")
    assert row["q100_per_km2"] == pytest.approx(row["q100"] / 500.0, rel=1e-3)

    csv = run({"op": "csv"})
    assert csv.startswith("source,station_id") and "500" in csv
    pytest.importorskip("openpyxl")
    assert base64.b64decode(run({"op": "xlsx"}))[:2] == b"PK"
    assert run({"op": "nope"}) == {"error": "unknown op"}
