"""The Analyst: ask a water question in plain language, get a cited answer built from real data.

Not an agent that replaces the hydrologist. A tool-using assistant over the
same functions the MCP server exposes (``find_stations``, ``analyze_station``,
``flood_frequency``, ``get_timeseries``, ``list_sources``, ``describe_methods``,
``anywhere``): the model decides which to call, aquascope does the work, and
the report ends with a "Data" and a "Methods and citations" section that are
assembled deterministically from what the tools returned, never from the
model's memory.

Works with any OpenAI-compatible chat endpoint that supports tool calling
(OpenAI, Groq, Hugging Face router, Mistral, OpenRouter, Ollama, ...) and with
Anthropic's Messages API (``--provider anthropic``), which the transport
translates. Configuration, in order: explicit arguments,
``AQUASCOPE_LLM_API_KEY`` / ``AQUASCOPE_LLM_BASE_URL`` / ``AQUASCOPE_LLM_MODEL``,
then ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GROQ_API_KEY``, ``HF_TOKEN``.
Uses the provider's SDK when installed and a built-in ``urllib`` client
otherwise (``aquascope.ai_engine.llm_transport``), which is also what runs
inside the Explorer's browser worker.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aquascope import __version__
from aquascope.ai_engine.providers import ENV_SCAN_ORDER
from aquascope.ai_engine.providers import PROVIDERS as _REGISTRY

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 14_000

#: Providers with a large window raise both caps through the registry
#: (``context_chars``): a single result may then take up to half the window.
#: The whole conversation's budget, in characters (roughly four to a token).
#: A per-result cap is not enough on a small window: three results at the cap
#: exceed a free tier's 8,000 tokens a minute on their own, and the provider
#: answers 413 "Request too large", which no amount of retrying will fix.
#: 24,000 characters is about 6,000 tokens, leaving room for the reply.
MAX_CONTEXT_CHARS = 24_000

#: Below this there is no room for a useful tool result, so a 413 here is about
#: something other than the conversation and should surface rather than loop.
MIN_CONTEXT_CHARS = 3_000

#: Said back to a model whose tool call the provider could not parse. Short on
#: purpose: it is added to a conversation that may already be near the limit.
TOOL_JSON_REMINDER = (
    "Your last tool call could not be parsed. The arguments must be a single JSON object, "
    'with any code as one JSON string: {"code": "import numpy as np\\nresult = 1"}. '
    "Newlines and quotes inside it have to be escaped. Please make that call again."
)

# One registry for the whole project (aquascope.ai_engine.providers); this dict
# is the shape the loop already used, kept so callers and tests do not change.
PROVIDERS: dict[str, dict[str, str | None]] = {
    p.id: {"base_url": None if p.id == "openai" else p.base_url, "model": p.model, "env": p.env}
    for p in _REGISTRY.values()
}

SYSTEM_PROMPT = """You are AquaScope's analyst, a careful hydrologist's assistant.
You answer questions about rivers, gauges, floods, rainfall and water resources ONLY from tool results.
Rules:
- For a question about a place or a station, call assess_site(lat, lon) first and respect its sufficiency
  table: do not run a method it marks not_defensible, say why, and offer what it marks defensible instead.
- Find stations with find_stations before analysing; prefer stations with long records for flood questions.
- Use analyze_station / flood_frequency for numbers; use anywhere(lat, lon) when the user names a place with no gauge.
- Use describe_catchment(lat, lon) for the catchment itself: area, elevation, climate, land cover, soils, dams.
- For an ungauged place, use similar_basins to find donor gauges, then analyze_station on the best donors.
- For "what flow to expect" at an ungauged place, use regionalize_signatures (mm/d estimates with a band and
  the leave-one-out skill); always quote the band and the skill, never a bare number.
- Never invent values, station ids, or citations. If a tool returns an error or an empty record, say so.
- Do not add facts about the site from memory: historic floods or droughts, dams, regulation or abstraction
  upstream, geography, land use. Everything about the site comes from a tool result. If a fact did not, leave
  it out or label it in the same sentence: "from general knowledge, not from the data".
- Report units. Quote return levels with their confidence intervals when available.
- Say which record (station, source, period, number of years) each number comes from.
- Keep the answer under 300 words unless the user asks for a full report; the tool outputs already carry
  licences and citations, they will be appended automatically.
"""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]



# Data the sandbox tool can see (the record on screen, an uploaded table). The
# caller fills this in for one ask() and it is cleared afterwards.
_SANDBOX_DATA: dict[str, Any] = {}


def _run_python_tool(code: str) -> dict[str, Any]:
    """The run_python tool: execute a snippet against aquascope and the current data."""
    from aquascope.ai_engine.sandbox import SandboxError, run_python

    try:
        return run_python(code, data=_SANDBOX_DATA).to_dict()
    except SandboxError as exc:
        return {"ok": False, "error": str(exc)}


def _tool_specs() -> list[ToolSpec]:
    from aquascope import mcp_server as t
    from aquascope.explore import anywhere

    num = {"type": "number"}
    years_cap = "Optional cap on the record: the last N years. Leave it out for the full record (the default)."
    return [
        ToolSpec(
            "list_sources", "Every data source with agency, country, variables and licence.",
            {"type": "object", "properties": {}}, t.list_sources,
        ),
        ToolSpec(
            "find_stations",
            "Search the world station catalog (no agency call). Use query for a name/id, near=[lat, lon] for the "
            "nearest gauges, bbox=[west, south, east, north] for an area, variable to filter (discharge, "
            "water_level, precipitation, ...).",
            {"type": "object", "properties": {
                "query": {"type": "string"}, "bbox": {"type": "array", "items": num, "minItems": 4, "maxItems": 4},
                "near": {"type": "array", "items": num, "minItems": 2, "maxItems": 2}, "variable": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer"}}},
            t.find_stations,
        ),
        ToolSpec(
            "assess_site",
            "What can be answered at a place, before any analysis: the gauges within reach (catalog record spans, "
            "no agency call), the catchment, and a sufficiency table marking every method defensible, marginal or "
            "not_defensible here, with the reason and the station it would use. Call it first for a question about "
            "a place or a station. problem narrows the table: flood_risk, ungauged_flow, drought, "
            "groundwater_decline, supply_reliability, climate_change, irrigation, water_quality.",
            {"type": "object", "properties": {"lat": num, "lon": num, "radius_km": num, "problem": {"type": "string"},
                                              "return_period": num},
             "required": ["lat", "lon"]},
            t.assess_site,
        ),
        ToolSpec(
            "analyze_station",
            "Fetch one station's record and compute summary, annual maxima, flood frequency (GEV, LP3 with CI), "
            "FDC percentiles and trend. variable: discharge (default), water_level, precipitation or "
            "groundwater_level for stations that have several. The full record is requested by default; "
            "fetch_note in the result says what was asked for and what the agency served.",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "years": {"type": "integer", "description": years_cap},
                                              "bootstrap_ci": {"type": "boolean"},
                                              "variable": {"type": "string"}},
             "required": ["source", "station_id"]},
            t.analyze_station,
        ),
        ToolSpec(
            "flood_frequency",
            "Return levels for T = 2..100 years at a station (subset of analyze_station).",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "years": {"type": "integer", "description": years_cap},
                                              "bootstrap_ci": {"type": "boolean"}},
             "required": ["source", "station_id"]},
            t.flood_frequency,
        ),
        ToolSpec(
            "get_timeseries",
            "The observed record for a station, resampled (D/W/M/Y) and thinned; use for values, not for statistics.",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "years": {"type": "integer"}, "resample": {"type": "string"},
                                              "max_points": {"type": "integer"}, "variable": {"type": "string"}},
             "required": ["source", "station_id"]},
            t.get_timeseries,
        ),
        ToolSpec(
            "water_quality_samples",
            "Sampled water-quality parameters at a station (USGS daily values: temperature, conductivity, dissolved "
            "oxygen, pH; Water Quality Portal discrete samples): tidy rows with per-parameter counts, units, period, "
            "licence and attribution. A screening: the last 5 years and a short parameter list by default. Run "
            "analyse_table with wqi, iwqi or who_screen on the rows.",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "years": {"type": "integer"},
                                              "parameters": {"type": "array", "items": {"type": "string"}},
                                              "use": {"type": "string"}},
             "required": ["source", "station_id"]},
            t.water_quality_samples,
        ),
        ToolSpec(
            "anywhere",
            "Climate and modelled discharge for a point with no gauge: ERA5 rainfall/temperature, FAO-56 ET0, "
            "aridity, GloFAS.",
            {"type": "object", "properties": {"lat": num, "lon": num, "years": {"type": "integer"}},
             "required": ["lat", "lon"]},
            lambda lat, lon, years=10: anywhere(lat, lon, years=years),
        ),
        ToolSpec(
            "describe_catchment",
            "The catchment of a point from BasinATLAS (HydroATLAS): sub-basin, upstream area and area-weighted "
            "attributes (elevation, precipitation, PET, aridity, snow, runoff, land cover, soils, population, dams). "
            "upstream=false for the local sub-basin only.",
            {"type": "object", "properties": {"lat": num, "lon": num, "upstream": {"type": "boolean"}},
             "required": ["lat", "lon"]},
            t.describe_catchment,
        ),
        ToolSpec(
            "similar_basins",
            "Gauged basins whose catchments most resemble a point's (lat, lon) or a station's (source, station_id): "
            "donor selection for ungauged sites. method: similarity | proximity | combined.",
            {"type": "object", "properties": {"lat": num, "lon": num, "source": {"type": "string"},
                                              "station_id": {"type": "string"}, "k": {"type": "integer"},
                                              "method": {"type": "string"},
                                              "sources": {"type": "array", "items": {"type": "string"}}}},
            t.similar_basins,
        ),
        ToolSpec(
            "regionalize_signatures",
            "Estimated flow regime of an ungauged point (mean/median/Q95/Q05 flow in mm/d, annual max, runoff ratio, "
            "baseflow index, FDC slope, flow frequencies, seasonality, flashiness) transferred from similar gauged "
            "donors, with an uncertainty band and the leave-one-out skill. method: similarity | regression | both.",
            {"type": "object", "properties": {"lat": num, "lon": num, "k": {"type": "integer"},
                                              "method": {"type": "string"}},
             "required": ["lat", "lon"]},
            t.regionalize_signatures,
        ),
        ToolSpec(
            "drought_indices",
            "Drought status at a place: SPI and SPEI at 1, 3 and 12 months (or timescales) with their divergence, "
            "from a rain gauge (source + station_id, its whole record) or the ERA5 cell (lat, lon, last years). "
            "pet: thornthwaite (default, from ERA5 temperature) | fao56 | none. Prefer SPEI under warming.",
            {"type": "object", "properties": {"lat": num, "lon": num, "years": {"type": "integer"},
                                              "timescales": {"type": "array", "items": {"type": "integer"}},
                                              "source": {"type": "string"}, "station_id": {"type": "string"},
                                              "pet": {"type": "string"}},
             "required": ["lat", "lon"]},
            t.drought_indices,
        ),
        ToolSpec(
            "drought_propagation",
            "Groundwater drought at a well (Standardised Groundwater Index) and the SPI timescale and lag in months "
            "at which rainfall deficits reach it (cross-correlation, Bloomfield and Marchant 2013).",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "lat": num, "lon": num, "years": {"type": "integer"},
                                              "max_lag": {"type": "integer"}},
             "required": ["source", "station_id", "lat", "lon"]},
            t.drought_propagation,
        ),
        ToolSpec(
            "low_flow_context",
            "How low is low at a gauge and is it low now: Q95/Q50/Q10, baseflow index, 7Q10, and where the last 30 "
            "and 90 days sit in the record.",
            {"type": "object", "properties": {"source": {"type": "string"}, "station_id": {"type": "string"},
                                              "years": {"type": "integer"}},
             "required": ["source", "station_id"]},
            t.low_flow_context,
        ),
        ToolSpec(
            "supply_reliability",
            "Can a river supply a demand (demand_m3s or demand_ml_day): a run-of-river screening keeping a reserve "
            "(q95 by default) in the river and taking at most share of the flow. Gauged with source + station_id "
            "(fraction of days, years and volume met, over the year or over months), ungauged with lat + lon "
            "(reliability band from donor-transferred Q95/median/Q05). Not a storage-yield analysis.",
            {"type": "object", "properties": {"demand_m3s": num, "demand_ml_day": num, "source": {"type": "string"},
                                              "station_id": {"type": "string"}, "lat": num, "lon": num,
                                              "share": num, "reserve": {"type": "string"},
                                              "months": {"type": "array", "items": {"type": "integer"}}}},
            t.supply_reliability,
        ),
        ToolSpec(
            "crop_water_demand",
            "A crop's seasonal irrigation demand at a point (FAO-56 single Kc on ERA5 ET0, effective rain "
            "subtracted, divided by efficiency): mm, m3 over area_ha, mean and peak-month m3/s, the season's "
            "months. crop is a FAO-56 Table 12 key (maize, wheat_winter, rice_paddy, ...). Supply not checked.",
            {"type": "object", "properties": {"lat": num, "lon": num, "crop": {"type": "string"}, "area_ha": num,
                                              "planting_month": {"type": "integer"}, "efficiency": num,
                                              "years": {"type": "integer"}},
             "required": ["lat", "lon", "crop", "area_ha", "planting_month"]},
            t.crop_water_demand,
        ),
        ToolSpec(
            "run_python",
            "Run a short Python snippet with aquascope, workbench, pandas (pd) and numpy (np) already imported, "
            "plus any data the page passed (for example df, the record on screen). Leave what you want back in a "
            "variable called result. Use this when no other tool fits: decadal statistics, a ratio between two "
            "records, the same analysis over several donors. Imports outside the standard scientific set are refused.",
            {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
            _run_python_tool,
        ),
        ToolSpec(
            "list_analyses",
            "The analyses available for a table of the user's own data (the workbench): what each needs and does.",
            {"type": "object", "properties": {}}, t.list_analyses,
        ),
        ToolSpec(
            "analyse_table",
            "Run one workbench analysis on a table the user supplied as CSV text: eda, quality, who_screen, "
            "wqi (CCME WQI 1.0 against WHO drinking-water, FAO 29 irrigation or CCME aquatic-life guidelines, "
            "plus the NSF WQI; params use, variant), iwqi (FAO 29 irrigation suitability), flow_duration, "
            "baseflow, recession, flood_frequency, signatures, return_periods, spei (precipitation with a "
            "pet_column or temperature_column + latitude), sgi_drought, recharge, aquifer_drawdown. params "
            "carries the analysis's own options.",
            {"type": "object", "properties": {"csv": {"type": "string"}, "analysis": {"type": "string"},
                                              "params": {"type": "object"}},
             "required": ["analysis"]},
            t.analyse_table,
        ),
        ToolSpec(
            "describe_methods", "What each analysis computes and the reference to cite.",
            {"type": "object", "properties": {}}, t.describe_methods,
        ),
        ToolSpec(
            "list_playbooks",
            "The problem playbooks (flood risk, ungauged flow, groundwater decline, drought status, supply "
            "reliability, irrigation feasibility, water quality): the method chain aquascope follows for the data "
            "that exists at a site. Use solve_plan when the user has a problem at a place rather than a question "
            "about a station.",
            {"type": "object", "properties": {}}, t.list_playbooks,
        ),
        ToolSpec(
            "describe_playbook", "One playbook in full: intake, branches, steps with gates, declines, caveats.",
            {"type": "object", "properties": {"playbook": {"type": "string"}}, "required": ["playbook"]},
            t.describe_playbook,
        ),
        ToolSpec(
            "solve_plan",
            "Plan (not run) a problem at a point with a playbook: reconnaissance, branch, and the study with its "
            "gates, with zero model calls. Show the plan to the user, then call solve_run with the study.",
            {"type": "object", "properties": {"problem": {"type": "string"}, "lat": num, "lon": num,
                                              "playbook": {"type": "string"}, "intake": {"type": "object"}},
             "required": ["problem", "lat", "lon"]},
            t.solve_plan,
        ),
        ToolSpec(
            "solve_run",
            "Execute a study from solve_plan with its gates (a failed gate runs the fallback once or stops with "
            "the reason); returns the gate outcomes, the report and the re-runnable study.",
            {"type": "object", "properties": {"study": {"type": "object"}}, "required": ["study"]},
            t.solve_run,
        ),
    ]


def _openai_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
        for s in specs
    ]


def resolve_llm(
    provider: str | None = None, model: str | None = None, api_key: str | None = None, base_url: str | None = None
) -> dict[str, str | None]:
    """Pick provider/model/key/base_url from arguments and the environment; raise if no key can be found."""
    if os.environ.get("AQUASCOPE_LLM_API_KEY") and not api_key and not provider:
        return {
            "provider": "custom",
            "api_key": os.environ["AQUASCOPE_LLM_API_KEY"],
            "base_url": base_url or os.environ.get("AQUASCOPE_LLM_BASE_URL") or PROVIDERS["huggingface"]["base_url"],
            "model": model or os.environ.get("AQUASCOPE_LLM_MODEL") or PROVIDERS["huggingface"]["model"],
        }
    if provider is None:
        for name in ENV_SCAN_ORDER:
            env = PROVIDERS[name]["env"]
            if env and os.environ.get(env):
                provider = name
                break
        else:
            provider = "ollama" if base_url or os.environ.get("AQUASCOPE_LLM_BASE_URL") else None
    if provider is None:
        raise RuntimeError(
            "No LLM configured. Set OPENAI_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY or HF_TOKEN, or "
            "AQUASCOPE_LLM_API_KEY with AQUASCOPE_LLM_BASE_URL/AQUASCOPE_LLM_MODEL, or pass --provider ollama "
            "for a local model."
        )
    if provider not in PROVIDERS and provider != "custom":
        raise ValueError(f"Unknown provider {provider!r}; choose from {list(PROVIDERS)}")
    cfg = PROVIDERS.get(provider, {"base_url": None, "model": None, "env": None})
    key = api_key or (os.environ.get(cfg["env"]) if cfg["env"] else None)
    if not key and provider == "ollama":
        key = "ollama"
    if not key:
        raise RuntimeError(f"No API key for {provider}: pass --api-key or set {cfg['env']}.")
    return {
        "provider": provider, "api_key": key, "base_url": base_url or cfg["base_url"], "model": model or cfg["model"],
    }


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


@dataclass
class AskResult:
    question: str
    answer: str
    model: str
    provider: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    methods: list[dict[str, str]] = field(default_factory=list)
    data_used: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    #: Deterministic checks over the answer and the tool results (verify.py).
    checks: list[dict[str, Any]] = field(default_factory=list)
    verified: bool = True
    #: The steps behind the answer as a study file, so it can be run again.
    study: str = ""
    #: Model calls and tokens, summed from the provider's usage fields when it reports them.
    usage: dict[str, int] = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", "", self.answer.strip(), ""]
        unmet = [c for c in self.checks if not c.get("passed")]
        if unmet:
            lines += ["## What this answer does not establish", ""]
            lines += [f"- {c.get('detail') or c.get('name')}" for c in unmet]
            lines += [""]
        if self.data_used:
            lines += ["## Data", ""]
            for d in self.data_used:
                bits = [f"**{d.get('label')}**"]
                if d.get("period"):
                    bits.append(d["period"])
                if d.get("license"):
                    bits.append(f"licence {d['license']}")
                if d.get("attribution"):
                    bits.append(d["attribution"])
                lines.append("- " + " · ".join(bits))
            lines.append("")
        if self.methods:
            lines += ["## Methods and citations", ""]
            for i, m in enumerate(self.methods, 1):
                lines.append(f"{i}. **{m['name']}.** {m['text']} _{m['citation']}_")
            lines.append("")
        lines += [
            "---",
            f"Produced by aquascope {__version__} (`aquascope ask`), model {self.model} via {self.provider}, "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Tools called: "
            + (", ".join(f"{c.name}" for c in self.tool_calls) or "none")
            + ". Numbers come from the tool results; the model wrote the prose.",
        ]
        return "\n".join(lines)


def _harvest_provenance(name: str, args: dict[str, Any], result: Any, res: AskResult) -> None:
    """Pull citations and data provenance out of tool results, deterministically."""
    if not isinstance(result, dict):
        return
    seen = {m["name"] for m in res.methods}
    for m in result.get("methods", []) or []:
        if isinstance(m, dict) and m.get("name") and m["name"] not in seen:
            res.methods.append({k: str(m.get(k, "")) for k in ("name", "text", "citation")})
            seen.add(m["name"])
    record_tools = ("analyze_station", "flood_frequency", "get_timeseries", "water_quality_samples")
    if name in record_tools and result.get("source"):
        label = f"{result.get('source')} / {result.get('station_id')}"
        period = None
        if result.get("start") and result.get("end"):
            period = f"{result['start']} to {result['end']}"
            if result.get("years"):
                period += f" ({result['years']} years)"
        if not any(d.get("label") == label for d in res.data_used):
            res.data_used.append({
                "label": label, "period": period, "license": result.get("license"),
                "attribution": result.get("attribution"),
            })
    if name == "similar_basins" and result.get("stations"):
        label = "similar-basins search"
        if not any(d.get("label") == label for d in res.data_used):
            res.data_used.append({"label": label, "period": None, "license": result.get("license"),
                                  "attribution": result.get("attribution")})
    if name == "regionalize_signatures" and result.get("estimates"):
        label = f"regionalised signatures at {result.get('latitude')}, {result.get('longitude')}"
        if not any(d.get("label") == label for d in res.data_used):
            res.data_used.append({"label": label, "period": None, "license": result.get("license"),
                                  "attribution": "donor gauges (per-source licences); BasinATLAS (CC BY 4.0)"})
    if name == "describe_catchment" and result.get("sub_basin"):
        label = f"catchment at {result.get('latitude')}, {result.get('longitude')}"
        if not any(d.get("label") == label for d in res.data_used):
            res.data_used.append({"label": label, "period": None, "license": result.get("license"),
                                  "attribution": result.get("attribution")})
    if name == "anywhere" and "latitude" in result:
        label = f"point {result['latitude']}, {result['longitude']}"
        if not any(d.get("label") == label for d in res.data_used):
            res.data_used.append({
                "label": label, "period": f"{result.get('start')} to {result.get('end')}",
                "license": "CC-BY-4.0", "attribution": result.get("attribution"),
            })


def _add_usage(total: dict[str, int], usage: Any) -> None:
    """Add one response's usage (an object or a dict, fields missing on some providers) to ``total``."""
    total["calls"] = total.get("calls", 0) + 1
    for key in ("prompt_tokens", "completion_tokens"):
        v = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total[key] = total.get(key, 0) + int(v)


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f'... [truncated {len(text) - limit} chars]'


def _conversation_size(messages: list[dict[str, Any]]) -> int:
    """Everything that goes on the wire, not just the text a human would read.

    An assistant turn carries its ``tool_calls`` with the arguments the model
    wrote, which for a ``run_python`` call is a whole snippet. Counting only
    ``content`` under-reads the request badly, which is how a conversation
    budgeted at 6,000 tokens arrived as 9,300.
    """
    try:
        return len(json.dumps(messages, ensure_ascii=False, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return sum(len(str(m.get("content") or "")) for m in messages)


def fit_context(messages: list[dict[str, Any]], budget: int = MAX_CONTEXT_CHARS) -> list[dict[str, Any]]:
    """Shrink the oldest tool results until the conversation fits the budget.

    The model needs the last result in full to answer; the older ones it has
    usually already summarised into its own reasoning. So the oldest are cut
    first, each keeping a readable head and saying what was removed, and the
    most recent is left alone. Nothing else is touched: the system prompt, the
    question and the assistant's own turns stay whole.
    """
    if _conversation_size(messages) <= budget:
        return messages
    out = [dict(m) for m in messages]
    tool_indexes = [i for i, m in enumerate(out) if m.get("role") == "tool"]
    for i in tool_indexes[:-1] if len(tool_indexes) > 1 else []:
        if _conversation_size(out) <= budget:
            break
        content = str(out[i].get("content") or "")
        if len(content) <= 400:
            continue
        out[i]["content"] = content[:400] + f"... [trimmed {len(content) - 400} chars to fit the context]"
    # Still over: the newest result is itself too big, so cut that too. The
    # note about the cut is part of the message, so leave room for it, or the
    # conversation comes out just over the budget it was supposed to fit.
    if _conversation_size(out) > budget and tool_indexes:
        i = tool_indexes[-1]
        content = str(out[i].get("content") or "")
        note_allowance = 80
        room = max(200, budget - (_conversation_size(out) - len(content)) - note_allowance)
        if len(content) > room:
            out[i]["content"] = content[:room] + f"... [trimmed {len(content) - room} chars to fit the context]"
    return out


def context_budget(provider: str | None) -> int:
    """How much conversation to keep before trimming, in characters.

    The default is sized for free tiers that meter tokens per minute. A
    provider with a large window says so in the registry (``context_chars``)
    and gets to keep its tool results whole.
    """
    spec = _REGISTRY.get(provider or "")
    return spec.context_chars if spec is not None and spec.context_chars else MAX_CONTEXT_CHARS


def ask(
    question: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_steps: int = 8,
    client: Any | None = None,
    on_event: Callable[[str], None] | None = None,
    data: dict[str, Any] | None = None,
    verify_answer: bool = True,
    context_chars: int | None = None,
) -> AskResult:
    """Answer ``question`` with tool calls over aquascope; returns an :class:`AskResult`.

    ``client`` lets tests (or callers with their own SDK setup) pass an
    OpenAI-compatible client; otherwise one is built from ``resolve_llm``
    (the ``openai`` SDK if installed, else the built-in ``urllib`` client).
    With a client, ``provider`` still names the provider so its context
    budget applies.

    ``data`` is put in reach of the ``run_python`` tool (the Explorer passes the
    record on screen). ``verify_answer`` runs the deterministic checks in
    :mod:`aquascope.ai_engine.verify` and reports what the answer does not
    establish, rather than leaving it to the reader to notice.
    ``context_chars`` caps how much conversation is kept (the registry's
    budget for the provider otherwise); a benchmark uses it to bound spend.
    """
    cfg = {"provider": provider or "custom", "model": model or "test", "api_key": None, "base_url": base_url}
    if client is None:
        from aquascope.ai_engine.llm_transport import make_client

        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
    specs = {s.name: s for s in _tool_specs()}
    tools = _openai_tools(list(specs.values()))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = AskResult(question=question, answer="", model=str(cfg["model"]), provider=str(cfg["provider"]))
    say = on_event or (lambda _m: None)
    _SANDBOX_DATA.clear()
    _SANDBOX_DATA.update(data or {})
    seen: list[dict[str, Any]] = []          # what each tool actually returned, for the checks

    budget = int(context_chars) if context_chars else context_budget(cfg["provider"])
    result_cap = max(MAX_TOOL_RESULT_CHARS, budget // 2)
    for step in range(1, max_steps + 1):
        result.steps = step
        messages = fit_context(messages, budget)
        # A 413 is the provider saying this request cannot fit its window, at
        # any speed, so retrying it unchanged is pointless. Halve the budget and
        # go again: the window belongs to the provider and is not ours to guess.
        from aquascope.ai_engine.llm_transport import LLMHTTPError
        malformed = 0
        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=cfg["model"], messages=messages, tools=tools, tool_choice="auto"
                )
                break
            except LLMHTTPError as exc:
                body = (exc.body or "").lower()
                too_large = exc.status == 413 or "too large" in body or "too long" in body
                # Some providers reject the whole request when the model's own
                # tool call will not parse as JSON. The model wrote it, so the
                # model can write it again: say what was wrong and resample.
                bad_call = exc.status == 400 and "tool_use_failed" in body
                if too_large and attempt < 5 and budget > MIN_CONTEXT_CHARS:
                    budget = max(MIN_CONTEXT_CHARS, budget // 2)
                    say(f"the request was too large for the model's window, retrying within {budget} characters")
                    messages = fit_context(messages, budget)
                elif bad_call and malformed < 2:
                    malformed += 1
                    say("the model's tool call was not valid JSON, asking it to write that call again")
                    messages = [*messages, {"role": "user", "content": TOOL_JSON_REMINDER}]
                else:
                    raise
        _add_usage(result.usage, getattr(response, "usage", None))
        choice = response.choices[0]
        msg = choice.message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            result.answer = (msg.content or "").strip()
            break
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls],
        })
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            spec = specs.get(name)
            say(f"tool {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            if spec is None:
                payload: Any = {"error": f"unknown tool {name}"}
                ok = False
            else:
                try:
                    payload = spec.func(**args)
                    ok = not (isinstance(payload, dict) and payload.get("error"))
                except Exception as exc:  # noqa: BLE001 - the model gets to see the failure and try again
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                    ok = False
            _harvest_provenance(name, args, payload, result)
            seen.append({"name": name, "arguments": args, "payload": payload, "ok": ok})
            text = json.dumps(payload, ensure_ascii=False, default=str)
            summary = text[:160]
            result.tool_calls.append(ToolCallRecord(name=name, arguments=args, ok=ok, summary=summary))
            messages.append({"role": "tool", "tool_call_id": c.id, "name": name,
                             "content": _truncate(text, result_cap)})
    else:
        result.answer = (result.answer or "").strip() or (
            "I ran out of tool-call steps before finishing. Here is what the tools returned so far; "
            "ask a narrower question or raise --max-steps."
        )
    if not result.answer:
        result.answer = "The model returned no answer."
    if verify_answer:
        from aquascope.ai_engine.verify import verify as _verify

        checks = _verify(result.answer, seen, question=question)
        result.checks = checks.to_dict()["checks"]
        result.verified = checks.ok
    # The steps that produced this answer, written down so they can be run again
    # without a model (aquascope run study.yaml).
    from aquascope.study import study_from_calls

    result.study = study_from_calls(question, result.tool_calls, model=str(cfg["model"])).to_yaml()
    _SANDBOX_DATA.clear()
    return result
