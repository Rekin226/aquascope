"""The worker's steering face: every reply carries the plan in plain words and the step controls, and the steer
op reruns one step and its dependants on the worker's copy. Run in CPython like the rest of the studio face."""

from __future__ import annotations

import json

import pytest

from tests.test_explorer.test_studio_worker import _face, _run
from tests.test_studio.conftest import PROBLEM, fake_tools


@pytest.fixture
def face():
    return _face()


def test_replies_carry_the_plain_plan(face) -> None:
    out = _run(face, {"op": "start", "lat": 51.415, "lon": -0.308, "text": PROBLEM}, tools=fake_tools([]))
    plain = out["plain"]
    steps = {s["id"]: s for s in plain["steps"]}
    ws_steps = {s["id"]: s for s in out["workspace"]["study"]["steps"]}
    assert set(steps) == set(ws_steps)
    s3 = steps["s3"]
    assert s3["tool"] == "flood_frequency" and s3["gates"] == ws_steps["s3"]["expects"]
    assert "the 100-year estimate may not exceed 3x the record length" in s3["checks"]
    assert not any("max_return_period_factor" in c for c in s3["checks"]), "no raw gate names in the sentences"
    assert {c["param"] for c in s3["controls"]} >= {"return_period", "years"}
    assert json.dumps(out)


def test_steer_reruns_the_step_and_its_dependants(face) -> None:
    calls: list = []
    tools = fake_tools(calls)
    out = _run(face, {"op": "start", "lat": 51.415, "lon": -0.308, "text": PROBLEM}, tools=tools)
    done = _run(face, {"op": "approve", "workspace": out["workspace"]}, tools=tools)
    assert done["status"] == "done"
    calls.clear()
    res = _run(face, {"op": "steer", "workspace": done["workspace"], "step_id": "s3",
                      "changes": {"return_period": 50}}, tools=tools)
    assert res["reply"]["kind"] == "report" and res["status"] == "done"
    assert [c[0] for c in calls] == ["flood_frequency", "anywhere"]
    assert res["workspace"]["study"]["plan"]["steering"][-1]["changes"]["return_period"] == {"from": 100, "to": 50}
    assert res["plain"]["steering"] and json.dumps(res)
    bad = _run(face, {"op": "steer", "workspace": res["workspace"], "step_id": "s3",
                      "changes": {"distribution": "lp3"}}, tools=tools)
    assert bad["reply"]["kind"] == "answer" and "not accepted" in bad["reply"]["text"]
