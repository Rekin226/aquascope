"""Study links in the Explorer: the worker ops, a shared plan rerun end to end, and the page wiring.

The pure half (explorer/src/study-link.js) is a node:test suite under
explorer/tests/; the codec and the check are aquascope.study_link
(tests/test_study_link.py). This checks the worker's "link" and "open_link"
ops in CPython, that a link made at the end of one study reruns as the same
plan in a fresh one (plan_used "proposed", keyless), and the hooks in
studio.js, url.js and app.js.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_explorer.test_studio_worker import _face, _run
from tests.test_studio.conftest import PROBLEM, fake_tools

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"


def _read(*parts: str) -> str:
    return (EXPLORER / Path(*parts)).read_text(encoding="utf-8")


@pytest.fixture
def face():
    return _face()


def test_a_link_from_a_finished_study_reruns_the_same_plan_keyless(face) -> None:
    calls: list = []
    tools = fake_tools(calls)
    out = _run(face, {"op": "start", "lat": 51.415, "lon": -0.308, "text": PROBLEM}, tools=tools)
    done = _run(face, {"op": "approve", "workspace": out["workspace"]}, tools=tools)
    assert done["status"] == "done"
    first = [c[0] for c in calls]

    made = _run(face, {"op": "link", "workspace": done["workspace"], "name": "Kingston"}, tools=tools)
    assert made["ok"] and made["token"].startswith("z1.")
    opened = _run(face, {"op": "open_link", "token": made["token"]}, tools=tools)
    assert opened["ok"], opened["errors"]
    shared = opened["study"]
    assert shared["name"] == "Kingston" and (shared["lat"], shared["lon"]) == (51.415, -0.308)

    # the recipient: a fresh worker, the same brief at the same place, the shared plan approved
    face["_STUDIO"].clear()
    calls.clear()
    out2 = _run(face, {"op": "start", "lat": shared["lat"], "lon": shared["lon"], "text": shared["text"],
                       "intake": shared["intake"]}, tools=tools)
    ws2 = out2["workspace"]
    if out2["status"] == "intake":
        ws2 = _run(face, {"op": "say", "workspace": ws2, "text": "just go"}, tools=tools)["workspace"]
    assert ws2["status"] == "review"
    ran = _run(face, {"op": "approve", "workspace": ws2, "plan": {**shared["plan"], "source": "shared"}}, tools=tools)
    assert ran["status"] == "done"
    assert ran["reply"]["payload"]["plan_used"] == "proposed", ran["reply"]["payload"].get("plan_errors")
    assert ran["workspace"]["study"]["author"] == "shared"
    assert [c[0] for c in calls] == first, "the same tools, in the same order"


def test_a_tampered_link_is_refused_by_the_worker(face) -> None:
    from aquascope import study_link

    bad = {"v": 1, "q": "q", "site": {"lat": 1, "lon": 2},
           "steps": [{"id": "s1", "tool": "run_python", "arguments": {"code": "import os"}}]}
    out = _run(face, {"op": "open_link", "token": study_link.encode(bad)}, tools={})
    assert out == {"ok": False, "errors": ["step s1: unknown tool 'run_python'"]}
    assert _run(face, {"op": "open_link", "token": "z1.%%%"}, tools={})["ok"] is False


def test_the_page_wiring() -> None:
    studio = _read("src", "studio.js")
    assert 'from "./study-link.js?v=__BUILD__"' in studio
    done = studio[studio.index("function doneHtml("):studio.index("function recordedDoneHtml(")]
    assert done.index('data-act="bundle"') < done.index('data-act="copy-link"') < done.index('data-act="again"')
    assert 'if (S.shared && !S.ws) return "shared";' in studio
    assert "BOARDS.shared = () => sharedBoardHtml(S.shared, { escapeHtml, stepHtml });" in studio
    assert 'else if (what === "copy-link") copyStudyLink();' in studio
    assert 'else if (what === "run-shared") runShared();' in studio
    run = studio[studio.index("async function runShared("):studio.index("// ── open, wire")]
    assert 'plan: { ...sh.plan, source: "shared" }' in run and 'callStudio("say", { text: "just go" })' in run
    assert 'op: "open_link"' in studio and 'op: "link"' in studio
    assert "sharedPlanLine({ used: p.plan_used" in studio

    url = _read("src", "url.js")
    assert "out.studyLink = v" in url and "state.study.link" in url
    app = _read("app.js")
    assert "actions.openSharedStudy({ link: url.studyLink })" in app
    assert "studyUrlParam(location.search)" in app

    worker = _read("worker.js")
    assert 'if op in ("link", "open_link"):' in worker and "from aquascope.study_link import studio_op" in worker


def test_the_study_link_surface_writes_with_plain_hyphens() -> None:
    for parts in (("src", "study-link.js"), ("tests", "study-link.test.mjs")):
        text = _read(*parts)
        assert "—" not in text and "–" not in text, parts[-1]
    text = (ROOT / "aquascope" / "study_link.py").read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text
