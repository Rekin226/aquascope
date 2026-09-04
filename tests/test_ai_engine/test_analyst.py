"""The Analyst: a tool loop over aquascope's functions with deterministic provenance."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aquascope.ai_engine import analyst


class FakeChat:
    """Scripted OpenAI-compatible client: a list of turns, each either tool calls or a final text."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            msg = SimpleNamespace(content=turn, tool_calls=None)
        else:
            calls = [
                SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                for i, (name, args) in enumerate(turn)
            ]
            msg = SimpleNamespace(content="", tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


FAKE_FF = {
    "source": "uk_ea", "station_id": "abc", "agency": "EA", "license": "OGL-UK-3.0",
    "attribution": "Environment Agency", "unit": "m3/s", "start": "1986-08-17", "end": "2026-08-15",
    "years": 40.0, "n": 14555,
    "ffa": {"n_years": 39, "return_periods": [2, 5, 10, 25, 50, 100],
            "fits": {"gev_lmoments": {"q": [1, 2, 3, 4, 5, 6]}}},
    "notes": [], "methods": [{"name": "GEV fitted by L-moments", "text": "t", "citation": "Hosking 1990"}],
}


def test_ask_runs_tools_and_builds_cited_report():
    client = FakeChat([
        [("find_stations", {"query": "kingston", "variable": "discharge"})],
        [("flood_frequency", {"source": "uk_ea", "station_id": "abc"})],
        "The 100-year flow at Kingston is about 6 m3/s (GEV L-moments, 39 annual maxima).",
    ])
    found = {"n_returned": 1, "stations": [{"source": "uk_ea", "station_id": "abc"}]}
    with patch("aquascope.mcp_server.find_stations", return_value=found), \
         patch("aquascope.mcp_server.flood_frequency", return_value=FAKE_FF):
        res = analyst.ask("What is the 100-year flood at Kingston?", client=client, model="fake")
    assert res.steps == 3 and [c.name for c in res.tool_calls] == ["find_stations", "flood_frequency"]
    assert all(c.ok for c in res.tool_calls)
    assert res.answer.startswith("The 100-year flow")
    md = res.to_markdown()
    assert "## Methods and citations" in md and "Hosking 1990" in md
    assert "## Data" in md and "uk_ea / abc" in md and "OGL-UK-3.0" in md
    assert "aquascope ask" in md and "fake" in md
    # tool schemas were offered on every request; results were fed back as tool messages
    assert all("tools" in r for r in client.requests)
    tool_msgs = [m for m in client.requests[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and json.loads(tool_msgs[1]["content"])["ffa"]["n_years"] == 39


def test_ask_reports_tool_errors_and_unknown_tools():
    client = FakeChat([[("nope", {}), ("flood_frequency", {"source": "x", "station_id": "1"})], "done"])
    with patch("aquascope.mcp_server.flood_frequency", side_effect=RuntimeError("boom")):
        res = analyst.ask("q", client=client)
    assert [c.ok for c in res.tool_calls] == [False, False]
    assert "boom" in res.tool_calls[1].summary


def test_ask_stops_at_max_steps():
    client = FakeChat([[("describe_methods", {})]] * 5)
    res = analyst.ask("q", client=client, max_steps=2)
    assert res.steps == 2 and "ran out of tool-call steps" in res.answer


def test_resolve_llm_env_and_errors(monkeypatch):
    for k in ("OPENAI_API_KEY", "GROQ_API_KEY", "HF_TOKEN", "AQUASCOPE_LLM_API_KEY", "AQUASCOPE_LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="No LLM configured"):
        analyst.resolve_llm()
    monkeypatch.setenv("GROQ_API_KEY", "g")
    cfg = analyst.resolve_llm()
    assert cfg["provider"] == "groq" and cfg["base_url"].startswith("https://api.groq.com") and cfg["api_key"] == "g"
    monkeypatch.setenv("AQUASCOPE_LLM_API_KEY", "hosted")
    cfg = analyst.resolve_llm()
    assert cfg["provider"] == "custom" and cfg["api_key"] == "hosted"
    assert analyst.resolve_llm(provider="ollama")["api_key"] == "ollama"
    with pytest.raises(ValueError):
        analyst.resolve_llm(provider="nope", api_key="k")


def test_tool_specs_cover_the_mcp_surface():
    names = {s.name for s in analyst._tool_specs()}
    assert names == {"list_sources", "find_stations", "analyze_station", "flood_frequency", "get_timeseries",
                     "anywhere", "describe_catchment", "similar_basins", "regionalize_signatures",
                     # sampled water-quality parameters at a station (#62)
                     "water_quality_samples",
                     # reconnaissance before analysis: what the record here supports (#306)
                     "assess_site",
                     # the workbench: analyses of the user's own table (#235)
                     "list_analyses", "analyse_table", "describe_methods",
                     # code for the questions no fixed tool covers (#234)
                     "run_python",
                     # Ask hands a problem at a place to Solve (#307, #308)
                     "list_playbooks", "describe_playbook", "solve_plan", "solve_run",
                     # the site-level tools of the drought, supply and irrigation playbooks (#309)
                     "drought_indices", "drought_propagation", "low_flow_context", "supply_reliability",
                     "crop_water_demand"}
    tools = analyst._openai_tools(analyst._tool_specs())
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in tools)


def test_the_prompt_forbids_facts_from_memory_and_names_the_label() -> None:
    """#324: "the 2014 winter floods" and "heavily abstracted upstream" came from no tool."""
    assert "from memory" in analyst.SYSTEM_PROMPT
    assert "from general knowledge, not from the data" in analyst.SYSTEM_PROMPT


def test_the_years_argument_is_described_as_an_optional_cap() -> None:
    """#270: the model should know that leaving years out asks for the full record."""
    specs = {s.name: s for s in analyst._tool_specs()}
    for name in ("analyze_station", "flood_frequency"):
        years = specs[name].parameters["properties"]["years"]
        assert years["type"] == "integer" and "full record" in years["description"]
