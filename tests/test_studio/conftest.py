"""Fixtures for the Studio tests: the recon sites of the playbook tests, realistic tool payloads, a recording
set of fake tools, a scripted model client that answers per role, and a Studio wired to all of them.
Nothing here touches the network."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import aquascope.explore
from tests.test_ai_engine.test_team import CATCHMENT, FLOW, RECON
from tests.test_playbooks import LONG, RAIN_MARGINAL, RAIN_ONLY, RICH, SHORT, UNGAUGED, WELL, recon

__all__ = ["CATCHMENT", "FLOW", "LONG", "RAIN_MARGINAL", "RAIN_ONLY", "RECON", "RICH", "SHORT", "UNGAUGED",
           "WELL", "recon"]

PROBLEM = "Design flow for a road crossing, 100-year return period"

DROUGHT = {
    "latitude": 51.415, "longitude": -0.308, "precipitation_source": "uk_ea R1 (gauge)",
    "station": {"source": "uk_ea", "station_id": "R1", "variable": "precipitation", "unit": "mm",
                "start": "1990-01-01", "end": "2026-08-01", "years": 36.6, "n_months": 439},
    "timescales": [1, 3, 12], "months": 439, "start": "1990-01", "end": "2026-08", "years": 36.6,
    "pet_method": "thornthwaite",
    "current": {"date": "2026-08", "spi": {"1": -0.42, "3": -1.13, "12": -0.87},
                "spei": {"1": -0.61, "3": -1.36, "12": -1.05}},
    "headline_index": "spei", "headline_timescale": 12, "threshold": -1.0, "in_drought": True,
    "status": "moderately_dry",
    "indices": [{"timescale": 12, "spi": {"worst": -2.31, "worst_date": "1997-03", "events": 6},
                 "spei": {"worst": -2.55, "worst_date": "2022-09", "events": 7},
                 "divergence": {"mean_last_10y": -0.21, "months_spei_drier_pct": 63.4, "correlation": 0.91}}],
    "temperature": {"mean_c": 11.2, "trend_c_per_decade": 0.31, "p_value": 0.004, "n_years": 36},
    "unit": "index",
    "methods": [{"name": "Standardized Precipitation Index", "text": "t", "citation": "McKee et al. 1993"}],
    "attribution": "Open-Meteo",
}

LOW_FLOW = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "unit": "m3/s", "years": 39.9,
            "start": "1986-08-17", "end": "2026-08-15", "fdc": {"q95": 12.3, "q50": 38.7, "q10": 141.0},
            "bfi": 0.71, "low_flow": {"7q10": 9.8},
            "recent": {"end": "2026-08-15", "last_30d_mean": 14.1, "last_30d_exceedance_pct": 88.2},
            "attribution": "Environment Agency", "license": "OGL-UK-3.0"}

DONORS = {"k": 5, "method": "combined",
          "stations": [{"source": "usgs", "station_id": str(i), "name": f"D{i}"} for i in range(5)],
          "attribution": "BasinATLAS"}

SIGNATURES = {"latitude": 51.415, "longitude": -0.308, "method": "similarity", "n_donors_available": 5,
              "estimates": {"q95_mm": {"label": "Q95", "value": 0.12, "unit": "mm/d", "low": 0.08, "high": 0.2},
                            "q_mean_mm": {"label": "mean flow", "value": 0.61, "unit": "mm/d", "low": 0.5,
                                          "high": 0.8}},
              "skill": {"by_signature": {"q95_mm": {"nse": 0.55}}}, "license": "CC BY 4.0"}

ANYWHERE = {"latitude": 51.415, "longitude": -0.308, "start": "2006-01-01", "end": "2026-01-01",
            "climate": {"precipitation_mm_per_year": 700.0, "et0_mm_per_year": 600.0, "aridity_index": 1.17,
                        "aridity_class": "humid"},
            "glofas": {"stats": {"mean": 60.0},
                       "ffa": {"return_periods": [2, 5, 10, 25, 50, 100],
                               "fits": {"gev_lmoments": {"q": [260, 340, 395, 460, 505, 560],
                                                         "q_by_T": {"2": 260, "5": 340, "10": 395, "25": 460,
                                                                    "50": 505, "100": 560}}}}},
            "attribution": "Open-Meteo"}

SUPPLY = {"mode": "gauged", "source": "uk_ea", "station_id": "3400TH", "years": 39.9, "start": "1986-08-17",
          "end": "2026-08-15", "unit": "m3/s", "fdc": {"q95": 12.3, "q50": 38.7, "q10": 141.0}, "bfi": 0.71,
          "demand_m3s": 2.0, "reserve_rule": "q95", "reserve_m3s": 12.3, "share": 0.1,
          "required_flow_m3s": 32.3, "status": "ok", "verdict": "reliable",
          "reliability": {"daily": 0.61, "annual": 0.2, "volumetric": 0.97, "days_short_per_year": 142.0,
                          "worst_year": {"year": 2011, "days_short": 270}},
          "methods": [{"name": "Flow-duration screening", "text": "t", "citation": "Vogel & Fennessey 1994"}]}

FLOW_FDC = {**FLOW, "fdc": {"q95": 12.3, "q50": 38.7, "q10": 141.0}}

# aquascope.rivers.reach_summary at the site: the GEOGLOWS v2 reach and its simulated record (#516).
REACH = {"river_id": 230399750, "modelled": True, "label": "modelled", "unit": "m3/s", "variable": "discharge",
         "start": "1940-01-01", "end": "2026-09-30", "years": 86.7,
         "snap": {"snapped": True, "river_id": 230399750, "distance_m": 18.0},
         "stats": {"mean": 70.1, "max": 690.0}, "fdc": {"q95": 14.0, "q50": 48.0, "q10": 160.0},
         "ffa": {"n_years": 86, "return_periods": [2, 5, 10, 25, 50, 100],
                 "fits": {"gev_lmoments": {"q": [300, 390, 450, 520, 570, 620],
                                           "q_by_T": {"2": 300, "5": 390, "10": 450, "25": 520, "50": 570,
                                                      "100": 620}}}},
         "attribution": "GEOGLOWS v2, CC BY 4.0", "licence": "CC BY 4.0"}

PAYLOADS: dict[str, Any] = {
    "describe_catchment": CATCHMENT, "analyze_station": FLOW_FDC, "flood_frequency": FLOW, "similar_basins": DONORS,
    "regionalize_signatures": SIGNATURES, "anywhere": ANYWHERE, "drought_indices": DROUGHT,
    "low_flow_context": LOW_FLOW, "supply_reliability": SUPPLY, "reach_record": REACH,
    "get_timeseries": {"source": "uk_ea", "station_id": "3400TH", "unit": "m3/s", "n_points": 120,
                       "resample": "M", "years": 10.0, "series": {"t": ["2016-01-31", "2016-02-29"],
                                                                   "v": [40.0, 55.0]}},
}

SERIES_CSV = "\n".join(["date,flow_m3s"] + [f"20{10 + i // 365:02d}-{(i % 365) // 31 + 1:02d}-{i % 28 + 1:02d},"
                                            f"{20 + (i * 7) % 90}" for i in range(0, 365 * 12, 3)])
SAMPLES_CSV = "station,parameter,value,unit\nA,nitrate,3.1,mg/L\nA,nitrate,2.7,mg/L\nA,pH,7.9,\n"


def fake_tools(calls: list, **overrides: Any) -> dict[str, Any]:
    """Every tool of PAYLOADS as a recording callable; ``overrides`` replace a payload (or a callable)."""
    table = {**PAYLOADS, **overrides}

    def make(name: str):
        def f(**kw: Any) -> Any:
            calls.append((name, kw))
            value = table[name]
            return value(**kw) if callable(value) else json.loads(json.dumps(value))
        return f

    return {name: make(name) for name in table}


class FakeModel:
    """A scripted OpenAI-shaped client that answers per role.

    ``script`` maps a role (consultant, methodologist, analyst, critic, author) to a list of replies (dicts are
    sent as JSON, strings as they are) consumed in order, or to a callable taking the context dict. The role
    is read off the system prompt. Every request is recorded with its role and parsed context.
    """

    ROLE_WORDS = (("Consultant", "consultant"), ("Methodologist", "methodologist"), ("specialist", "analyst"),
                  ("Interpreter", "interpreter"), ("Critic", "critic"), ("Author", "author"))

    def __init__(self, script: dict[str, Any]):
        self.script = {k: (list(v) if isinstance(v, list) else v) for k, v in script.items()}
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @classmethod
    def role_of(cls, system: str) -> str:
        found = [(system.index(word), role) for word, role in cls.ROLE_WORDS if word in system]
        return min(found)[1] if found else "unknown"

    def _create(self, **kwargs: Any):
        messages = kwargs["messages"]
        system = messages[0]["content"]
        role = self.role_of(system)
        try:
            context = json.loads(messages[1]["content"])
        except (json.JSONDecodeError, IndexError, KeyError):
            context = {}
        self.requests.append({"role": role, "system": system, "context": context, "model": kwargs.get("model")})
        entry = self.script.get(role)
        if callable(entry):
            reply = entry(context)
        elif isinstance(entry, list) and entry:
            reply = entry.pop(0)
        else:
            reply = None
        text = json.dumps(reply) if isinstance(reply, dict) else (reply or "")
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30))

    def calls(self, role: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["role"] == role]


def workbench_tools() -> dict[str, Any]:
    """The workbench analyses as the runner wraps them: pure functions over a frame, no network."""
    from aquascope import workbench
    from aquascope.study import _workbench_tool

    return {name: _workbench_tool(name) for name in workbench.TOOLS}


@contextmanager
def patched(recon_value: dict[str, Any] = RECON, tools: dict[str, Any] | None = None, calls: list | None = None):
    """assess_site and the runner's tools replaced for the block: the fakes over the workbench's pure tools,
    so a plan over an attached table runs while every network tool stays faked (or unknown)."""
    tools = tools if tools is not None else fake_tools(calls if calls is not None else [])
    tools = {**workbench_tools(), **tools}
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=recon_value), \
         patch("aquascope.study._tools", return_value=tools):
        yield tools


@pytest.fixture
def no_deliverables(monkeypatch):
    """Make the deliverables package unimportable even when a sibling worktree installed it."""
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables", None)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.figures", None)
    monkeypatch.setitem(sys.modules, "aquascope.studio.deliverables.tables", None)
    yield


@pytest.fixture
def studio_factory(no_deliverables):
    """``make(problem=..., recon=..., tools=..., client=..., **kwargs) -> (Studio, calls)`` inside the patches."""
    from aquascope.studio import Studio

    patchers: list = []

    def make(*, recon_value: dict[str, Any] = RECON, tools: dict[str, Any] | None = None,
             client: Any = None, lat: float | None = 51.415, lon: float | None = -0.308, **kwargs: Any):
        calls: list = []
        tools = tools if tools is not None else fake_tools(calls)
        if client is not None:
            kwargs.setdefault("model", "fake")
            kwargs.setdefault("provider", "custom")
        # assess_site is looked up on aquascope.explore at call time; patch it for the test's life.
        patcher = patch.object(aquascope.explore, "assess_site", create=True, return_value=recon_value)
        patcher.start()
        patchers.append(patcher)
        s = Studio(lat=lat, lon=lon, client=client, tools=tools, **kwargs)
        return s, calls

    yield make
    for patcher in reversed(patchers):      # last started, first stopped: each restores what it found
        patcher.stop()
