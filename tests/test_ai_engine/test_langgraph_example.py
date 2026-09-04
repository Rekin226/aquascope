"""examples/langgraph_team.py: the team's roles as graph nodes, with LangGraph absent.

LangGraph is not a dependency, so the example has to import without it, say
what to install, and keep its node functions runnable (they are plain calls
into the package). None of this needs langgraph installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.test_ai_engine.test_team import RECON, _tools

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "langgraph_team.py"


@pytest.fixture
def example(monkeypatch):
    """The example module loaded with every langgraph import failing."""
    for name in ("langgraph", "langgraph.graph", "langgraph.types", "langgraph.checkpoint",
                 "langgraph.checkpoint.memory"):
        monkeypatch.setitem(sys.modules, name, None)
    spec = importlib.util.spec_from_file_location("langgraph_team_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_without_langgraph_the_module_imports_and_says_what_to_install(example, capsys):
    assert example.HAVE_LANGGRAPH is False
    assert "pip install langgraph langchain-core" in example.LANGGRAPH_MISSING
    with pytest.raises(ImportError, match="pip install langgraph langchain-core"):
        example.build_graph()
    assert example.main(["--lat", "51.415", "--lon", "-0.308"]) == 2
    assert "pip install langgraph langchain-core" in capsys.readouterr().err


def test_the_nodes_run_as_plain_functions_keyless(example, monkeypatch):
    """scout -> plan -> review -> run -> report, by hand, on the Kingston fixture and fake tools."""
    monkeypatch.setattr("aquascope.explore.assess_site", lambda lat, lon, **kw: dict(RECON, asked=kw))
    state = {"problem": "Design flow for a road crossing, 50-year return period", "lat": 51.415, "lon": -0.308,
             "playbook": "flood_risk", "intake": {"return_period": 50}}
    state.update(example.scout_node(state))
    assert state["recon"]["asked"] == {"problem": "flood_risk", "return_period": 50.0}

    state.update(example.plan_node(state))
    assert state["declined"] is None
    assert state["study"]["plan"]["branch"] == "at_site" and state["study"]["problem"]["params"]["return_period"] == 50
    assert example._after_plan(state) == "review"
    text = example.plan_text(state["study"])
    assert "branch at_site, 3 step(s)" in text and "gate spread_within 0.25" in text

    # no langgraph: the review is the callback; approve as is
    state.update(example.review_node(state))
    assert state["approved"] is True and example._after_review(state) == "run"

    calls: list = []
    state.update(example.run_node(state, tools=_tools(calls)))
    assert [c[0] for c in calls] == ["describe_catchment", "analyze_station", "flood_frequency"]
    assert all(g["passed"] for g in state["result"]["gates"])

    state.update(example.report_node(state))
    assert "## Steps and gates" in state["report"] and "gate spread_within: passed" in state["report"]


def test_a_review_that_declines_goes_straight_to_the_report(example, monkeypatch):
    monkeypatch.setattr(example, "REVIEW", lambda study: None)
    state = {"study": {"steps": [{"tool": "anywhere"}], "plan": {"playbook": "flood_risk"}}}
    state.update(example.review_node(state))
    assert state["approved"] is False and example._after_review(state) == "report"
    assert example.report_node(state)["report"].startswith("Declined: The plan was declined at review.")
    # an edited study coming back from the reviewer replaces the plan
    monkeypatch.setattr(example, "REVIEW", lambda study: {"steps": [{"tool": "describe_catchment"}]})
    out = example.review_node(state)
    assert out["approved"] is True and out["study"]["steps"][0]["tool"] == "describe_catchment"
