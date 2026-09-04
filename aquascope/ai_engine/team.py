"""The team: from "a problem at a location" to a verified answer (#308).

The plan-first Analyst is a team of roles, each a prompt plus a bounded tool
set, sharing one study as the blackboard:

    intake -> recon -> plan -> review -> execute -> report
              Scout    Coordinator  (you)   Reviewer   Narrator
                                            + Specialist on a failed gate

Every role call is a stateless subcall: fresh messages, the study and its own
inputs as compact JSON, never the whole transcript. Keyless, the Scout,
the tree-filled Coordinator, the Reviewer and a template Narrator give a
complete run with zero model calls. With a key, the Coordinator adds a
rationale (and settles an ambiguous problem), the Specialist proposes one
fallback step when a gate fails, and the Narrator writes prose. No agent
framework: plain Python, the same code in the CLI, the MCP server and the
browser worker.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aquascope import __version__
from aquascope.study import Study, StudyRun, run_study

logger = logging.getLogger(__name__)

__all__ = ["SolveResult", "choose_playbook", "intake_hints", "run_reviewed", "solve"]

MAX_CONTEXT_CHARS = 12_000

#: Keyword rules the Coordinator applies before any model call; a playbook's
#: score is the number of distinct patterns that match the problem text.
KEYWORDS: dict[str, list[str]] = {
    "flood_risk": [
        r"\bflood", r"design (flow|flood|discharge|storm)", r"return period", r"\d+\s*-?\s*(year|yr)\b",
        r"\b(culvert|bridge|crossing|spillway|levee|embankment|weir)\b", r"\b(inundat|overtop)", r"peak flow",
        r"\b1\s*-?\s*in\s*-?\s*\d+\b",
    ],
    "ungauged_flow": [
        r"ungauged|no gauge|not gauged|unmonitored",
        r"what flow|how much (water|flow)|flow (to expect|regime|available)",
        r"\b(q95|low[- ]flow|baseflow|mean flow|yield)\b",
        r"offtake|abstraction|hydro-?power|micro-?hydro|irrigation|environmental flow|e-?flow",
        r"\b(stream|creek|brook)\b",
    ],
    "groundwater_decline": [
        r"ground ?water|aquifer|water[- ]table|borehole|\bwells?\b|piezometer", r"declin|falling|dropping|depletion|"
        r"drawdown|recover|lowering", r"\bsgi\b|groundwater drought", r"recharge",
    ],
    "water_quality": [
        r"water quality|quality of (the |this )?(river |stream |well )?water", r"safe to drink|drinkab|potab|drinking",
        r"\bwqi\b|pollut|contaminat|guideline",
        r"\b(nitrate|arsenic|e\.?\s*coli|coliform|turbidity|dissolved oxygen)\b",
        r"\bph\b|salinity|\bsar\b|sodium|irrigation (water )?(quality|suitab)", r"aquatic life|fish kill",
    ],
    "drought_status": [
        r"\bdrought", r"\bspi\b|\bspei\b", r"\bdry (spell|period|year|season)|how dry|\bdryness",
        r"(rain|rainfall|precipitation) (deficit|shortfall|anomal)", r"in drought|drought (status|index|indices)",
    ],
    "supply_reliability": [
        r"\breliab", r"\bsupply\b|\bsupplies\b|\bsupplying\b", r"\bdemand\b",
        r"\d\s*(m3|m³|cumec|cubic met|ML|megalit)", r"\btown\b|municipal|drinking water|water works",
        r"withdraw|divert|take (out|from) the river",
    ],
    "irrigation_feasibility": [
        r"irrigat", r"\bcrops?\b|\bfield\b|\bfarm", r"\bhectares?\b|\bha\b",
        r"\b(maize|wheat|rice|paddy|soy(bean)?|cotton|sugar ?cane|tomato|potato|grape|vine|citrus|olive|sunflower|"
        r"barley|alfalfa|onion|cabbage|pepper|banana|coffee|tea|sorghum|groundnut|sugar ?beet)\b",
        r"crop water|water requirement|planting|growing season",
    ],
}

_RETURN_PERIOD = re.compile(r"(\d{1,4})\s*-?\s*(?:year|yr)\b", re.I)
_ONE_IN = re.compile(r"\b1\s*-?\s*in\s*-?\s*(\d{1,4})\b", re.I)

COORDINATOR_PROMPT = """You are the Coordinator of AquaScope's plan-first Analyst.
You are given a problem in plain language, the reconnaissance of the site (which records exist, for how long),
and the playbooks available (each with its intake fields). Answer with ONE JSON object and nothing else:
{"playbook": "<id or null>", "intake": {<field>: <value>}, "reason": "<one sentence>"}
Pick the playbook whose problem class matches; null when none does. Fill only intake fields the text supports."""

RATIONALE_PROMPT = """You are the Coordinator of AquaScope's plan-first Analyst.
Write ONE paragraph (under 120 words) explaining why this plan fits this site: which record it rests on and for
how long, why this branch and not another, and what the gates will check. Use only the facts in the JSON given.
Do not invent numbers, station names or citations. Plain prose, no headings, no lists."""

SPECIALIST_PROMPTS: dict[str, str] = {
    "default": "You are a hydrologist on AquaScope's team. A step of a study failed its gate.",
    "drought_status": (
        "You are the drought specialist on AquaScope's team. A step of a drought study failed its gate. Sound "
        "fallbacks: drought_indices for the ERA5 cell instead of a short rain gauge (the record must reach 20 "
        "years), a shorter set of timescales, low_flow_context on a nearby discharge gauge, or the SGI alone when "
        "the propagation lag cannot be read. Never quote a sub-monthly index; this team sees monthly droughts."
    ),
    "supply_reliability": (
        "You are the water-supply specialist on AquaScope's team. A step of a supply study failed its gate. Sound "
        "fallbacks: supply_reliability for the point from donors (lat, lon) when the gauge record is too short, "
        "similar_basins for a regional cross-check, low_flow_context for the low-flow statistics. Never propose a "
        "storage-yield analysis; the playbook screens run-of-river abstraction."
    ),
    "irrigation_feasibility": (
        "You are the irrigation specialist on AquaScope's team. A step of an irrigation study failed its gate. "
        "Sound fallbacks: crop_water_demand over a longer ERA5 window, anywhere for the climate context, "
        "supply_reliability against a gauge within reach. Keep to FAO-56 single Kc; say when supply was not "
        "checked."
    ),
    "flood_risk": (
        "You are the flood-frequency specialist on AquaScope's team. A step of a flood study failed its gate. "
        "Sound fallbacks: donor gauges (similar_basins) for a regional cross-check, regionalize_signatures for a "
        "transferred mean annual maximum, anywhere for GloFAS, or the same station over a different record "
        "length. Never propose a nonstationary fit."
    ),
    "ungauged_flow": (
        "You are the ungauged-basins specialist on AquaScope's team. A step of a regionalisation study failed "
        "its gate. Sound fallbacks: similar_basins with a larger k or method 'proximity', regionalize_signatures "
        "with method 'both', anywhere for GloFAS."
    ),
    "groundwater_decline": (
        "You are the groundwater specialist on AquaScope's team. A step of a well study failed its gate. Sound "
        "fallbacks: analyze_station over a longer record, get_timeseries with a different resample, anywhere "
        "for the regional water balance. Never attribute the cause."
    ),
    "water_quality": (
        "You are the water-quality specialist on AquaScope's team. A step of a water-quality study failed its "
        "gate. Sound fallbacks: water_quality_samples over a longer window (years) or with a different parameter "
        "list, wqi with another guideline set, who_screen alone. Never turn an index over sampled parameters "
        "into a health verdict."
    ),
}

SPECIALIST_RULES = """Propose exactly ONE fallback step as a JSON object and nothing else:
{"tool": "<one of the tools listed>", "arguments": {...}, "rationale": "<one sentence>",
 "expects": [<optional gates, same vocabulary as the failed step>]}
The arguments must be valid for the tool. Use only station ids, coordinates and values that appear in the
context. If no fallback is defensible, answer {"tool": null, "rationale": "<why>"}."""

NARRATOR_RULES = """
You are the Narrator of a plan-first study. The engine ran a plan (a playbook branch) and evaluated a gate after
each step; you are given the executed steps, their gate outcomes and their results as JSON.
Write the answer as prose under 300 words, from these results only:
- Answer the problem first, in one or two sentences, with the number(s) and their units and intervals.
- Say which record each number comes from (station, source, period, years) and which method produced it.
- Confidence intervals in the results are 90 % bands (Log-Pearson III analytical, GEV bootstrap) unless a result
  says otherwise; quote them as 90 %, and write plain ASCII digits, hyphens and units (m3/s).
- Report the gate outcomes that matter (a failed gate, a fallback that ran, a spread between fits).
- Do not write headings or lists; do not add a Data or Methods section, the engine appends them.
- Do not invent a number, a station, a citation or a cause. If something did not run, say so.
"""


# ── the result ───────────────────────────────────────────────────────────────


@dataclass
class SolveResult:
    """What the team produced: the executed study, the prose, the checks and who did what."""

    problem: dict[str, Any]
    study: Study
    recon: dict[str, Any]
    run: StudyRun | None = None
    answer: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    declined: bool = False
    declined_reason: str | None = None
    cost: dict[str, dict[str, int]] = field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    methods: list[dict[str, str]] = field(default_factory=list)
    data_used: list[dict[str, Any]] = field(default_factory=list)
    not_established: list[str] = field(default_factory=list)
    finished: str = ""

    @property
    def caveats(self) -> list[str]:
        return list((self.study.plan or {}).get("caveats") or [])

    @property
    def citations(self) -> list[str]:
        return list((self.study.plan or {}).get("citations") or [])

    @property
    def gates(self) -> list[dict[str, Any]]:
        return self.run.gates if self.run else []

    @property
    def ok(self) -> bool:
        return bool(self.run and self.run.ok and not self.declined)

    @property
    def study_yaml(self) -> str:
        return self.study.to_yaml()

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "declined": self.declined,
            "declined_reason": self.declined_reason,
            "study": self.study.to_dict(),
            "study_yaml": self.study_yaml,
            "recon": self.recon,
            "run": self.run.manifest() if self.run else None,
            "gates": self.gates,
            "checks": self.checks,
            "not_established": self.not_established,
            "caveats": self.caveats,
            "citations": self.citations,
            "methods": self.methods,
            "data_used": self.data_used,
            "timeline": self.timeline,
            "cost": self.cost,
            "model": self.model,
            "provider": self.provider,
            "report": self.to_markdown(),
        }

    def to_markdown(self) -> str:
        plan = self.study.plan or {}
        text = self.problem.get("text") or self.study.question
        lines = [f"# {text}", ""]
        if self.declined:
            lines += ["**Declined.** " + (self.declined_reason or ""), ""]
        if self.answer:
            lines += [self.answer.strip(), ""]
        head = ", ".join(f"{k} {plan[k]}" for k in ("playbook", "branch") if plan.get(k))
        if head or plan.get("rationale"):
            lines += ["## Plan", ""]
            if head:
                lines.append(f"{head[0].upper() + head[1:]}.")
            if plan.get("rationale"):
                lines.append(str(plan["rationale"]))
            for n in (plan.get("recon_notes") or []) + (plan.get("notes") or []):
                lines.append(f"- {n}")
            lines.append("")
        if self.study.steps:
            lines += ["## Steps and gates", ""]
            results = {r.get("id"): r for r in (self.run.results if self.run else [])}
            for i, s in enumerate(self.study.steps, 1):
                args = ", ".join(f"{k}={v!r}" for k, v in s.arguments.items())
                r = results.get(s.id)
                state = "not run" if r is None else ("ok" if r["ok"] else f"failed: {r.get('error')}")
                lines.append(f"{i}. `{s.tool}({args})`: {state}")
                if s.rationale:
                    lines.append(f"   {s.rationale}")
                for g in (r or {}).get("gates") or []:
                    verdict = "passed" if g["passed"] else "FAILED"
                    lines.append(f"   - gate {g['check']}: {verdict}, {g.get('detail', '')}")
                fb = (r or {}).get("fallback")
                if r and r.get("fallback_used") and isinstance(fb, dict):
                    fargs = ", ".join(f"{k}={v!r}" for k, v in (fb.get("arguments") or {}).items())
                    fstate = "ok" if fb.get("ok") else f"failed: {fb.get('error')}"
                    lines.append(f"   - fallback `{fb.get('tool')}({fargs})`: {fstate}")
                    for g in fb.get("gates") or []:
                        verdict = "passed" if g["passed"] else "FAILED"
                        lines.append(f"     - gate {g['check']}: {verdict}, {g.get('detail', '')}")
            if self.run and self.run.stop_reason:
                lines.append(f"\n**Stopped at {self.run.stopped_at}:** {self.run.stop_reason}")
            lines.append("")
        if self.not_established:
            lines += ["## What this answer does not establish", ""]
            lines += [f"- {c}" for c in self.not_established] + [""]
        if self.caveats:
            lines += ["## Caveats", ""] + [f"- {c}" for c in self.caveats] + [""]
        if self.data_used:
            lines += ["## Data", ""]
            for d in self.data_used:
                bits = [f"**{d.get('label')}**"]
                if d.get("period"):
                    bits.append(str(d["period"]))
                if d.get("license"):
                    bits.append(f"licence {d['license']}")
                if d.get("attribution"):
                    bits.append(str(d["attribution"]))
                lines.append("- " + " · ".join(bits))
            lines.append("")
        if self.methods or self.citations:
            lines += ["## Methods and citations", ""]
            n = 0
            for m in self.methods:
                n += 1
                lines.append(f"{n}. **{m['name']}.** {m['text']} _{m['citation']}_")
            for c in self.citations:
                n += 1
                lines.append(f"{n}. {c}")
            lines.append("")
        who = (f"model {self.model} via {self.provider}" if self.model
               else "no model: the playbook tree filled the plan and a template wrote the prose")
        tokens = sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0) for v in self.cost.values())
        calls = sum(v.get("calls", 0) for v in self.cost.values())
        lines += [
            "---",
            f"Produced by aquascope {__version__} (`aquascope solve`), playbook {plan.get('playbook') or 'none'}"
            f"{', branch ' + plan['branch'] if plan.get('branch') else ''}, {who}, "
            f"{self.finished or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
            f"Model calls: {calls}" + (f" ({tokens} tokens)" if tokens else "") + ". "
            "Numbers come from the tool results and were checked by the gates; the study file re-runs the same "
            "steps with `aquascope run`.",
        ]
        return "\n".join(lines) + "\n"


# ── the model, when there is one ────────────────────────────────────────────


class _Model:
    """A stateless subcall per role: fresh messages, compact JSON context, tokens counted per role."""

    def __init__(self, client: Any, model: str, provider: str, cost: dict[str, dict[str, int]],
                 timeline: list[dict[str, Any]], say: Callable[[dict[str, Any]], None]):
        self.client = client
        self.model = model
        self.provider = provider
        self.cost = cost
        self.timeline = timeline
        self.say = say

    def call(self, role: str, system: str, context: dict[str, Any], *, step: str | None = None) -> str | None:
        user = json.dumps(context, ensure_ascii=False, default=str)
        if len(user) > MAX_CONTEXT_CHARS:
            user = user[:MAX_CONTEXT_CHARS] + '... [truncated]"}'
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages)
        except Exception as exc:  # noqa: BLE001 - the role falls back to its keyless behaviour
            self._event(role, step, "model_error", f"{type(exc).__name__}: {exc}")
            return None
        usage = getattr(response, "usage", None)
        entry = self.cost.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        entry["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens"):
            n = _usage_field(usage, key)
            if n:
                entry[key] += int(n)
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            text = ""
        self._event(role, step, "model_call", f"{len(user)} chars in, {len(text)} out")
        return text.strip() or None

    def _event(self, role: str, step: str | None, event: str, detail: str) -> None:
        self.say({"role": role, "step": step, "event": event, "detail": detail})


def _usage_field(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        v = usage.get(key)
    else:
        v = getattr(usage, key, None)
    return int(v) if isinstance(v, (int, float)) else None


def _json_block(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


# ── the Coordinator's keyword rules ─────────────────────────────────────────


def choose_playbook(text: str) -> tuple[str | None, bool]:
    """The playbook the keyword rules pick for ``text`` and whether the choice is ambiguous."""
    scores = {
        pid: sum(1 for pat in pats if re.search(pat, text or "", re.I))
        for pid, pats in KEYWORDS.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, top = ranked[0]
    if top == 0:
        return None, True
    tie = len(ranked) > 1 and ranked[1][1] == top
    return best, tie


def intake_hints(text: str, playbook: str | None = None) -> dict[str, Any]:
    """Intake values the problem text states outright (a return period, a decision, a cause question)."""
    text = text or ""
    out: dict[str, Any] = {}
    m = _RETURN_PERIOD.search(text) or _ONE_IN.search(text)
    if m:
        out["return_period"] = int(m.group(1))
    if re.search(r"inundat|flood (map|extent)|how deep|which (streets|houses|fields)", text, re.I):
        out["decision"] = "inundation extent"
    elif re.search(r"insur", text, re.I):
        out["decision"] = "insurance"
    elif re.search(r"screen|how risky|is it at risk|risk", text, re.I) and not re.search(r"design", text, re.I):
        out["decision"] = "risk screening"
    if re.search(r"\bwhy\b|\bcause|because of|due to|attribut|pumping|abstraction", text, re.I) and (
        playbook == "groundwater_decline" or re.search(r"ground ?water|aquifer|well", text, re.I)
    ):
        out["attribute_cause"] = True
    if re.search(r"\bq95\b|low[- ]flow|dry season|minimum flow", text, re.I):
        out["statistic"] = "Q95"
    elif re.search(r"\bq05\b|high[- ]flow|peak", text, re.I) and playbook == "ungauged_flow":
        out["statistic"] = "Q05"
    elif re.search(r"mean flow|average flow", text, re.I):
        out["statistic"] = "mean"
    for pat, purpose in ((r"irrigat", "irrigation offtake"), (r"environmental|e-?flow|ecolog", "environmental flow"),
                         (r"hydro-?power|turbine|micro-?hydro", "hydropower screening"),
                         (r"water supply|drinking|abstraction", "water supply")):
        if re.search(pat, text, re.I):
            out["purpose"] = purpose
            break
    for pat, concern in ((r"subsid", "subsidence"), (r"supply|drinking|well running dry", "supply"),
                         (r"wetland|spring|ecosystem|river baseflow", "ecosystem")):
        if re.search(pat, text, re.I):
            out["concern"] = concern
            break
    if playbook == "water_quality" or re.search(r"water quality|safe to drink|drinkab|potab", text, re.I):
        for pat, use in ((r"irrigat|crop|farm|salinity|\bsar\b", "irrigation"),
                         (r"aquatic|fish|ecolog|ecosystem|habitat", "aquatic life"),
                         (r"drink|potab|tap|household|safe", "drinking")):
            if re.search(pat, text, re.I):
                out["use"] = use
                break
    if playbook == "drought_status" or re.search(r"\bdrought", text, re.I):
        if re.search(r"flash[- ]drought|(this|last|past|next) (week|fortnight)|\bweeks?\b", text, re.I):
            out["flash_drought"] = True
        for pat, concern in ((r"crop|farm|irrigat|agricultur|harvest", "agriculture"),
                             (r"ground ?water|aquifer|well|borehole", "groundwater"),
                             (r"supply|reservoir|drinking|town|municipal", "water supply")):
            if re.search(pat, text, re.I):
                out["drought_concern"] = concern
                break
    m = _DEMAND_M3S.search(text)
    if m:
        out["demand_m3s"] = float(m.group(1))
    m = _DEMAND_ML.search(text)
    if m and "demand_m3s" not in out:
        out["demand_ml_day"] = float(m.group(1))
    if playbook == "supply_reliability" or "demand_m3s" in out or "demand_ml_day" in out:
        if re.search(r"reservoir|\bdam\b|storage|impound", text, re.I):
            out["storage"] = True
        for pat, use in ((r"\btown\b|city|municipal|drinking|household|domestic", "municipal"),
                         (r"irrigat|farm|crop", "irrigation"),
                         (r"factory|industr|plant|mine|mill|brewery", "industrial")):
            if re.search(pat, text, re.I):
                out["use"] = use
                break
    if re.search(r"daily (irrigation )?schedul|irrigation schedul|when to irrigate|how much (water )?each (day|time)",
                 text, re.I) and (playbook == "irrigation_feasibility" or re.search(r"irrigat", text, re.I)):
        out["decision"] = "daily schedule"
    m = _AREA_HA.search(text)
    if m:
        out["area_ha"] = float(m.group(1))
    m = _CROP.search(text)
    if m:
        out["crop"] = _CROP_KEYS.get(m.group(1).lower().replace(" ", "").replace("-", ""), m.group(1).lower())
    m = _PLANT_MONTH.search(text)
    if m:
        out["planting_month"] = _MONTHS[m.group(1).lower()[:3]]
    return out


_DEMAND_M3S = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m3|m³|cubic met(?:er|re)s?|cumecs?)\s*(?:/|per|a)\s*s(?:ec(?:ond)?)?\b|"
                         r"(\d+(?:\.\d+)?)\s*cumecs?\b", re.I)
_DEMAND_ML = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ML|megalit(?:er|re)s?)\s*(?:/|per|a)\s*(?:d(?:ay)?)\b", re.I)
_AREA_HA = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ha\b|hectares?)", re.I)
_CROP = re.compile(r"\b(winter wheat|wheat|maize|corn|rice|paddy|soy(?:bean)?s?|cotton|sugar ?cane|tomato(?:es)?|"
                   r"potato(?:es)?|grapes?|vines?|citrus|olives?|sunflowers?|barley|alfalfa|onions?|cabbages?|"
                   r"peppers?|bananas?|coffee|tea|sorghum|groundnuts?|peanuts?|sugar ?beet)\b", re.I)
_CROP_KEYS = {
    "winterwheat": "wheat_winter", "wheat": "wheat_winter", "maize": "maize", "corn": "maize", "rice": "rice_paddy",
    "paddy": "rice_paddy", "soy": "soybean", "soys": "soybean", "soybean": "soybean", "soybeans": "soybean",
    "cotton": "cotton", "sugarcane": "sugarcane", "tomato": "tomato", "tomatoes": "tomato", "potato": "potato",
    "potatoes": "potato", "grape": "grape", "grapes": "grape", "vine": "grape", "vines": "grape", "citrus": "citrus",
    "olive": "olive", "olives": "olive", "sunflower": "sunflower", "sunflowers": "sunflower", "barley": "barley",
    "alfalfa": "alfalfa", "onion": "onion", "onions": "onion", "cabbage": "cabbage", "cabbages": "cabbage",
    "pepper": "pepper", "peppers": "pepper", "banana": "banana", "bananas": "banana", "coffee": "coffee", "tea": "tea",
    "sorghum": "sorghum", "groundnut": "groundnut", "groundnuts": "groundnut", "peanut": "groundnut",
    "peanuts": "groundnut", "sugarbeet": "sugar_beet",
}
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                       "dec"), start=1)}
_PLANT_MONTH = re.compile(r"(?:plant(?:ed|ing)?|sow(?:n|ing)?)\s+(?:in|from|on)?\s*(jan(?:uary)?|feb(?:ruary)?|"
                          r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
                          r"nov(?:ember)?|dec(?:ember)?)\b", re.I)


# ── compacting tool payloads for a role's context ───────────────────────────

_BULKY = {"series", "points", "samples", "exceedance", "sgi", "annual_precipitation", "monthly_precipitation_mm",
          "monthly_et0_mm", "features", "target", "_meta"}


def _compact(obj: Any, *, depth: int = 0) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _BULKY:
                if isinstance(v, (list, dict)):
                    out[k] = f"[{len(v)} entries omitted]"
                continue
            out[k] = _compact(v, depth=depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > 12:
            nums = [x for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(nums) == len(obj):
                return {"n": len(obj), "min": min(nums), "max": max(nums), "first": obj[:3], "last": obj[-3:]}
            return [_compact(x, depth=depth + 1) for x in obj[:8]] + [f"[{len(obj) - 8} more omitted]"]
        return [_compact(x, depth=depth + 1) for x in obj]
    if isinstance(obj, str) and len(obj) > 400:
        return obj[:400] + "..."
    return obj


def _recon_summary(recon: dict[str, Any]) -> dict[str, Any]:
    ctx = recon.get("context") or {}
    stations = [
        {k: s.get(k) for k in ("source", "station_id", "name", "distance_km", "variables", "years")}
        for s in (recon.get("stations") or [])[:5] if isinstance(s, dict)
    ]
    return {
        "point": recon.get("point"),
        "stations": stations,
        "years_by_variable": ctx.get("years_by_variable"),
        "resolution_by_variable": ctx.get("resolution_by_variable"),
        "area_km2": ctx.get("area_km2") or (recon.get("catchment") or {}).get("upstream_area_km2"),
        "donors": ctx.get("donors"),
        "catchment": _compact(recon.get("catchment") or {}),
        "sufficiency": [{k: r.get(k) for k in ("method", "status", "reason")}
                        for r in (recon.get("sufficiency") or [])[:12] if isinstance(r, dict)],
        "notes": recon.get("notes"),
    }


# ── the template Narrator ───────────────────────────────────────────────────


def _fmt(x: Any, digits: int = 4) -> str:
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:,.{digits}g}"
    return str(x)


def _rp_index(payload: dict[str, Any], rp: Any) -> int | None:
    periods = (payload.get("ffa") or {}).get("return_periods") or []
    try:
        return [float(p) for p in periods].index(float(rp))
    except (ValueError, TypeError):
        return None


def _sentences_for(tool: str, payload: dict[str, Any], study: Study) -> list[str]:
    out: list[str] = []
    params = (study.problem or {}).get("params") or {}
    unit = payload.get("unit") or ""
    if tool in ("analyze_station", "flood_frequency", "get_timeseries"):
        who = " ".join(str(x) for x in (payload.get("name"),) if x) or ""
        label = f"{payload.get('source')} {payload.get('station_id')}"
        where = f"{who} ({label})" if who else f"station {label}"
        if payload.get("start") and payload.get("end"):
            out.append(f"The record at {where} runs from {payload['start']} to {payload['end']}"
                       f" ({_fmt(payload.get('years'))} years, {payload.get('variable') or 'values'} in {unit}).")
        stats = payload.get("stats") or {}
        if stats.get("mean") is not None and tool != "flood_frequency":
            out.append(f"Its mean is {_fmt(stats['mean'])} {unit} (min {_fmt(stats.get('min'))}, "
                       f"max {_fmt(stats.get('max'))} {unit}).")
        trend = payload.get("trend")
        if isinstance(trend, dict) and trend.get("p_value") is not None:
            p = trend["p_value"]
            verdict = "significant" if p < 0.05 else "not significant"
            out.append(f"Mann-Kendall on the {trend.get('on', 'annual mean')}: {verdict} at the 5 % level "
                       f"(p = {_fmt(p, 3)}, Sen's slope {_fmt(trend.get('sens_slope_per_year'))} {unit} per year "
                       f"over {trend.get('n_years')} years).")
        ffa = payload.get("ffa") or {}
        fits = ffa.get("fits") or {}
        rp = params.get("return_period") or 100
        idx = _rp_index(payload, rp)
        if fits and idx is not None:
            parts = []
            gev = fits.get("gev_lmoments") or {}
            lp3 = fits.get("lp3") or {}
            boot = fits.get("gev_bootstrap") or {}
            if gev.get("q"):
                parts.append(f"GEV by L-moments {_fmt(gev['q'][idx])} {unit}")
            if lp3.get("q"):
                ci = (lp3.get("ci") or [None] * 6)[idx]
                s = f"Log-Pearson III {_fmt(lp3['q'][idx])} {unit}"
                if isinstance(ci, (list, tuple)) and len(ci) == 2 and None not in ci:
                    s += f" (90 % confidence interval {_fmt(ci[0])} to {_fmt(ci[1])} {unit})"
                parts.append(s)
            if boot.get("q"):
                ci = (boot.get("ci") or [None] * 6)[idx]
                s = f"bootstrap GEV {_fmt(boot['q'][idx])} {unit}"
                if isinstance(ci, (list, tuple)) and len(ci) == 2 and None not in ci:
                    s += f" (90 % confidence interval {_fmt(ci[0])} to {_fmt(ci[1])} {unit})"
                parts.append(s)
            if parts:
                out.append(f"The {rp}-year return level from {ffa.get('n_years')} annual maxima: "
                           + "; ".join(parts) + ".")
            if gev.get("q") and lp3.get("q"):
                a, b = gev["q"][idx], lp3["q"][idx]
                if a and b:
                    out.append(f"The two fits differ by {abs(a - b) / ((a + b) / 2):.0%}.")
        fdc = payload.get("fdc") or {}
        if fdc.get("q95") is not None:
            out.append(f"Flow duration: the flow exceeded on 95 % of days is {_fmt(fdc['q95'])} {unit}, the median "
                       f"{_fmt(fdc.get('q50'))} {unit}, the flow exceeded on 10 % of days "
                       f"{_fmt(fdc.get('q10'))} {unit}.")
        if tool == "get_timeseries" and payload.get("n_points"):
            out.append(f"{payload['n_points']} {payload.get('resample', '')} points were taken for the next step.")
    elif tool == "describe_catchment":
        attrs = payload.get("attributes") or {}
        bits = []
        if attrs.get("upstream_area_km2"):
            bits.append(f"upstream area {_fmt(attrs['upstream_area_km2'])} km2")
        elif attrs.get("area_km2"):
            bits.append(f"area {_fmt(attrs['area_km2'])} km2")
        for key in ("elevation_m", "precipitation_mm", "aridity_index", "degree_of_regulation_pct"):
            v = attrs.get(key)
            if isinstance(v, dict) and v.get("value") is not None:
                bits.append(f"{v.get('label') or key} {_fmt(v['value'])} {v.get('unit') or ''}".strip())
        if bits:
            out.append("The catchment (BasinATLAS): " + ", ".join(bits) + ".")
    elif tool == "similar_basins":
        st = payload.get("stations") or []
        names = [f"{s.get('name') or s.get('station_id')} ({s.get('source')} {s.get('station_id')})" for s in st[:5]]
        if st:
            out.append(f"{payload.get('k', len(st))} donor gauges by {payload.get('method')}: "
                       + ", ".join(names) + ".")
    elif tool == "regionalize_signatures":
        est = payload.get("estimates") or {}
        skill = ((payload.get("skill") or {}).get("by_signature") or {})
        bits = []
        for key in ("q_mean_mm", "q95_mm", "q05_mm", "q_annual_max_mm", "runoff_ratio", "baseflow_index"):
            e = est.get(key)
            if not isinstance(e, dict) or e.get("value") is None:
                continue
            s = f"{e.get('label') or key} {_fmt(e['value'])} {e.get('unit') or ''}".strip()
            if e.get("low") is not None and e.get("high") is not None:
                s += f" (band {_fmt(e['low'])} to {_fmt(e['high'])})"
            sk = skill.get(key) or {}
            nse = sk.get("nse") if isinstance(sk, dict) else None
            if nse is not None:
                s += f", leave-one-out NSE {_fmt(nse, 2)}"
            bits.append(s)
        if bits:
            out.append(f"Signatures transferred from {payload.get('n_donors_available') or ''} donors "
                       f"({payload.get('method')}): " + "; ".join(bits) + ".")
    elif tool == "anywhere":
        cl = payload.get("climate") or {}
        if cl.get("precipitation_mm_per_year") is not None:
            out.append(f"ERA5 climate for the cell: precipitation {_fmt(cl['precipitation_mm_per_year'])} mm per year, "
                       f"reference evapotranspiration {_fmt(cl.get('et0_mm_per_year'))} mm per year, aridity index "
                       f"{_fmt(cl.get('aridity_index'), 2)} ({cl.get('aridity_class')}).")
        g = payload.get("glofas") or {}
        if g:
            mean = _fmt((g.get("stats") or {}).get("mean"))
            s = f"GloFAS modelled discharge (grid cell, indicative): mean {mean} m3/s"
            fits = (g.get("ffa") or {}).get("fits") or {}
            gev = fits.get("gev_lmoments") or {}
            idx = _rp_index(g, params.get("return_period") or 100)
            if gev.get("q") and idx is not None:
                s += f", {params.get('return_period') or 100}-year GEV {_fmt(gev['q'][idx])} m3/s"
            out.append(s + ".")
    elif tool == "sgi_drought":
        out.append(f"Standardised Groundwater Index: current {_fmt(payload.get('current'), 2)}, worst "
                   f"{_fmt(payload.get('worst'), 2)}, {len(payload.get('events') or [])} drought events below "
                   f"{payload.get('threshold')}.")
    elif tool == "drought_indices":
        out += _drought_sentences(payload)
    elif tool == "drought_propagation":
        sgi = payload.get("sgi") or {}
        label = f"{payload.get('source')} {payload.get('station_id')}"
        if sgi.get("current") is not None:
            state = "in groundwater drought" if sgi.get("in_drought") else "not in drought"
            out.append(f"Standardised Groundwater Index at well {label} ({_fmt(payload.get('years'))} years of "
                       f"levels in {unit}, to {sgi.get('date')}): {_fmt(sgi['current'], 2)} now ({state} at the "
                       f"{sgi.get('threshold')} threshold), worst {_fmt(sgi.get('worst'), 2)} in "
                       f"{sgi.get('worst_date')}, {sgi.get('events')} drought events.")
        best = (payload.get("propagation") or {}).get("best")
        if isinstance(best, dict):
            lag = _months(best.get("lag_months"))
            out.append(f"Drought propagation: SPI over {_months(best.get('timescale'))} on ERA5 precipitation "
                       f"correlates best with the SGI at a lag of {lag} (r = {_fmt(best.get('correlation'), 2)} over "
                       f"{best.get('n')} months), so a rainfall deficit takes about {lag} to reach the water table "
                       f"here.")
    elif tool == "low_flow_context":
        out += _low_flow_sentences(payload, unit)
    elif tool == "supply_reliability":
        out += _supply_sentences(payload)
    elif tool == "crop_water_demand":
        d = payload.get("demand") or {}
        if d.get("gross_irrigation_mm") is not None:
            rng_ = d.get("gross_irrigation_mm_range") or [None, None]
            out.append(f"Crop water demand for {str(payload.get('crop', '')).replace('_', ' ')} on "
                       f"{_fmt(payload.get('area_ha'))} ha planted on the first of month "
                       f"{payload.get('planting_month')} (FAO-56 single Kc on ERA5 reference ET0, "
                       f"{len(payload.get('years_used') or [])} seasons "
                       f"averaged): crop evapotranspiration {_fmt(d.get('etc_mm'))} mm, effective rain "
                       f"{_fmt(d.get('effective_rain_mm'))} mm, net irrigation {_fmt(d.get('net_irrigation_mm'))} mm, "
                       f"gross irrigation {_fmt(d.get('gross_irrigation_mm'))} mm at an efficiency of "
                       f"{_fmt(payload.get('efficiency'), 2)} (range {_fmt(rng_[0])} to {_fmt(rng_[1])} mm across "
                       f"seasons).")
            checked = any(s.tool == "supply_reliability" for s in study.steps)
            tail = ("; the supply check against the gauge follows" if checked
                    else "; supply was not checked, no gauge with a usable record is within reach")
            out.append(f"That is {_fmt(d.get('gross_m3'), 6)} m3 over the {payload.get('season_days')}-day season, a "
                       f"mean {_fmt(d.get('mean_m3s'), 3)} m3/s and {_fmt(d.get('peak_month_m3s'), 3)} m3/s in the "
                       f"peak month{tail}.")
    elif tool == "recharge":
        out.append(f"Water-table-fluctuation recharge: {_fmt(payload.get('value_mm_per_year'))} mm per year "
                   f"(specific yield as given).")
    elif tool == "assess_site":
        ctx = payload.get("context") or {}
        out.append(f"Reconnaissance: {len(payload.get('stations') or [])} stations within reach; records "
                   f"{json.dumps(ctx.get('years_by_variable') or {})}.")
    elif tool == "water_quality_samples":
        label = f"{payload.get('source')} {payload.get('station_id')}"
        per = payload.get("parameters") or {}
        bits = [f"{name.replace('_', ' ')} {int(p.get('n') or 0)} in {p.get('unit') or 'no unit'}"
                for name, p in list(per.items())[:8] if isinstance(p, dict)]
        span = (f" from {payload.get('start')} to {payload.get('end')}" if payload.get("start") and payload.get("end")
                else "")
        out.append(f"{payload.get('n_samples')} water-quality samples of {payload.get('n_parameters')} parameter(s) "
                   f"at station {label}{span}" + (": " + ", ".join(bits) if bits else "") + ".")
    elif tool == "who_screen":
        rows = payload.get("rows") or []
        if rows:
            flagged = [r for r in rows if r.get("status") != "OK"]
            out.append(f"WHO drinking-water screen over {len(rows)} parameter(s): {payload.get('n_alerts', 0)} alert(s)"
                       f" and {payload.get('n_warnings', 0)} warning(s).")
            for r in flagged[:5]:
                out.append(f"{str(r.get('parameter')).replace('_', ' ')}: {r.get('n_exceed')} of {r.get('n')} samples "
                           f"({_fmt(r.get('pct'))} %) outside the guideline of {r.get('rule')} ({r.get('status')}).")
        else:
            out.append("WHO drinking-water screen: none of the sampled parameters has a WHO guideline value.")
    elif tool == "wqi":
        ccme = payload.get("ccme") or {}
        if ccme.get("score") is not None:
            out.append(f"CCME Water Quality Index 1.0 against the {payload.get('guideline_set')} guidelines: "
                       f"{_fmt(ccme['score'])} out of 100, {ccme.get('category')}, over {ccme.get('n_variables')} "
                       f"parameter(s) and {ccme.get('n_tests')} tests of which {ccme.get('n_failed_tests')} failed "
                       f"(F1 {_fmt(ccme.get('f1'))}, F2 {_fmt(ccme.get('f2'))}, F3 {_fmt(ccme.get('f3'))}).")
            drivers = ccme.get("drivers") or []
            if drivers:
                out.append("Exceedances: " + "; ".join(
                    f"{str(d.get('parameter')).replace('_', ' ')} {d.get('n_failed')} of {d.get('n')} samples outside "
                    f"{d.get('guideline')}" for d in drivers[:5]) + ".")
            else:
                out.append("No sampled parameter exceeded its guideline.")
            if ccme.get("meets_minimum_design") is False:
                out.append("The sampling design is below the CCME minimum of four parameters sampled four times each.")
        elif "ccme" in payload:
            out.append("The CCME index was not computed: no sampled parameter has a guideline in this set.")
        nsf = payload.get("nsf") or {}
        if nsf.get("score") is not None:
            out.append(f"NSF Water Quality Index: {_fmt(nsf['score'])} out of 100, {nsf.get('category')}, over "
                       f"{nsf.get('n_parameters')} of its nine parameters"
                       + (" (weights renormalised)" if nsf.get("weights_renormalised") else "") + ".")
        elif nsf:
            out.append(f"The NSF index was not computed: {nsf.get('n_parameters', 0)} of its nine parameters present"
                       + (f" (missing {', '.join(str(m).replace('_', ' ') for m in nsf.get('missing') or [])})" if
                          nsf.get("missing") else "") + ".")
    elif tool == "iwqi":
        restriction = payload.get("restriction")
        if restriction is None:
            out.append("Irrigation suitability (FAO 29) was not judged: none of the parameters it reads was sampled.")
        else:
            idx = payload.get("indices") or {}
            bits = []
            for key, label, unit in (("sar", "SAR", ""), ("sodium_percent", "sodium percentage", " %"),
                                     ("rsc", "residual sodium carbonate", " meq/L")):
                e = idx.get(key) or {}
                if e.get("value") is not None:
                    bits.append(f"{label} {_fmt(e['value'])}{unit} ({e.get('class')})")
            if idx.get("ussl_class"):
                bits.append(f"USSL class {idx['ussl_class']}")
            drivers = ", ".join(str(d).replace("_", " ") for d in payload.get("drivers") or [])
            verdict = ("no restriction on use from the sampled parameters" if restriction == "none"
                       else f"{restriction} restriction on use")
            out.append(f"Irrigation suitability (FAO 29): {verdict}"
                       + (f", driven by {drivers}" if drivers else "") + (": " + "; ".join(bits) if bits else "") + ".")
            missing = payload.get("missing") or []
            if missing:
                names = ", ".join(str(m).replace("_", " ") for m in missing[:8])
                out.append(f"Not sampled, so not judged: {names}.")
    return out


def _months(n: Any) -> str:
    return f"{n} month" if n == 1 else f"{n} months"


def _drought_sentences(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    src = payload.get("precipitation_source") or "the record"
    st = payload.get("station") or {}
    where = (f"rain gauge {st.get('source')} {st.get('station_id')} ({_fmt(st.get('years'))} years of "
             f"{st.get('variable', 'precipitation')} in {st.get('unit', 'mm')})" if st else f"{src}")
    cur = payload.get("current") or {}
    spi = cur.get("spi") or {}
    spei = cur.get("spei") or {}
    scales = payload.get("timescales") or []
    if spi:
        parts = [f"{_fmt(spi.get(str(s)), 2)} at {_months(s)}" for s in scales if spi.get(str(s)) is not None]
        out.append(f"Drought indices on {where}, {payload.get('months')} months from {payload.get('start')} to "
                   f"{payload.get('end')} ({_fmt(payload.get('years'))} years), as of {cur.get('date')}: SPI "
                   + ", ".join(parts) + ".")
    if spei:
        parts = [f"{_fmt(spei.get(str(s)), 2)} at {_months(s)}" for s in scales if spei.get(str(s)) is not None]
        out.append(f"SPEI ({payload.get('pet_method')} PET): " + ", ".join(parts) + ".")
    head = payload.get("headline_timescale")
    state = "in drought" if payload.get("in_drought") else "not in drought"
    out.append(f"Status on the {str(payload.get('headline_index', 'index')).upper()} at {_months(head)}: "
               f"{str(payload.get('status', '')).replace('_', ' ')}, {state} at the {payload.get('threshold')} "
               f"threshold.")
    for row in payload.get("indices") or []:
        if row.get("timescale") != head:
            continue
        d = row.get("divergence") or {}
        lead = row.get("spei") or row.get("spi") or {}
        if lead.get("worst") is not None:
            out.append(f"The worst value at {_months(head)} on record was {_fmt(lead['worst'], 2)} in "
                       f"{lead.get('worst_date')}; {lead.get('events')} drought events at or below "
                       f"{payload.get('threshold')}.")
        if d.get("mean_last_10y") is not None:
            drier = "drier" if d["mean_last_10y"] < 0 else "wetter"
            out.append(f"SPI and SPEI diverge: over the last ten years SPEI minus SPI at {_months(head)} averages "
                       f"{_fmt(d['mean_last_10y'], 2)} (SPEI the {drier}), and SPEI is drier in "
                       f"{_fmt(d.get('months_spei_drier_pct'), 3)} % of months (correlation "
                       f"{_fmt(d.get('correlation'), 2)}).")
    temp = payload.get("temperature") or {}
    if temp.get("trend_c_per_decade") is not None:
        p = temp.get("p_value")
        sig = "significant" if isinstance(p, (int, float)) and p < 0.05 else "not significant"
        out.append(f"ERA5 temperature for the cell: mean {_fmt(temp.get('mean_c'), 3)} C, trend "
                   f"{_fmt(temp['trend_c_per_decade'], 2)} C per decade over {temp.get('n_years')} years "
                   f"(Mann-Kendall {sig} at the 5 % level, p = {_fmt(p, 3)}).")
    return out


def _low_flow_sentences(payload: dict[str, Any], unit: str) -> list[str]:
    out: list[str] = []
    fdc = payload.get("fdc") or {}
    label = f"{payload.get('source')} {payload.get('station_id')}"
    if fdc.get("q95") is not None:
        out.append(f"Low flows at gauge {label} ({_fmt(payload.get('years'))} years of discharge in {unit}, "
                   f"{payload.get('start')} to {payload.get('end')}): Q95 {_fmt(fdc['q95'])} {unit}, Q50 "
                   f"{_fmt(fdc.get('q50'))} {unit}, Q10 {_fmt(fdc.get('q10'))} {unit}"
                   + (f", baseflow index {_fmt(payload.get('bfi'), 2)}" if payload.get("bfi") is not None else "")
                   + (f", 7Q10 {_fmt((payload.get('low_flow') or {}).get('7q10'))} {unit}"
                      if (payload.get("low_flow") or {}).get("7q10") is not None else "") + ".")
    rec = payload.get("recent") or {}
    if rec.get("last_30d_mean") is not None:
        out.append(f"The last 30 days to {rec.get('end')} averaged {_fmt(rec['last_30d_mean'])} {unit}, a flow "
                   f"exceeded on {_fmt(rec.get('last_30d_exceedance_pct'), 3)} % of days in the record"
                   + (f"; the last three months {_fmt(rec.get('last_90d_mean'))} {unit}, exceeded on "
                      f"{_fmt(rec.get('last_90d_exceedance_pct'), 3)} % of days" if rec.get("last_90d_mean") is not None
                      else "") + ".")
    return out


def _supply_sentences(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    rel = payload.get("reliability") or {}
    demand = payload.get("demand_m3s")
    if payload.get("mode") == "gauged":
        label = f"{payload.get('source')} {payload.get('station_id')}"
        fdc = payload.get("fdc") or {}
        out.append(f"Supply screening at gauge {label} ({_fmt(payload.get('years'))} years of daily discharge in m3/s, "
                   f"{payload.get('start')} to {payload.get('end')}): Q95 {_fmt(fdc.get('q95'))} m3/s, Q50 "
                   f"{_fmt(fdc.get('q50'))} m3/s, Q10 {_fmt(fdc.get('q10'))} m3/s"
                   + (f", baseflow index {_fmt(payload.get('bfi'), 2)}" if payload.get("bfi") is not None else "")
                   + (f", 7Q10 {_fmt((payload.get('low_flow') or {}).get('7q10'))} m3/s"
                      if (payload.get("low_flow") or {}).get("7q10") is not None else "") + ".")
        months = payload.get("months")
        share_pct = _fmt(100 * float(payload.get("share") or 0), 3)
        out.append(f"A demand of {_fmt(demand)} m3/s with {payload.get('reserve_rule')} "
                   f"({_fmt(payload.get('reserve_m3s'))} m3/s) and at most {share_pct} % of the flow taken needs the "
                   f"river to carry {_fmt(payload.get('required_flow_m3s'))} m3/s: met on "
                   f"{_fmt(100 * float(rel.get('daily') or 0), 3)} % of days"
                   + (f" in months {months}" if months else "")
                   + (f", in {_fmt(100 * float(rel['annual']), 3)} % of years without a shortfall"
                      if rel.get("annual") is not None else "")
                   + (f", {_fmt(100 * float(rel['volumetric']), 3)} % of the volume"
                      if rel.get("volumetric") is not None else "") + f"; verdict: {payload.get('verdict')}.")
        worst = rel.get("worst_year") or {}
        if worst.get("year") is not None and rel.get("days_short_per_year") is not None:
            out.append(f"Shortfalls average {_fmt(rel['days_short_per_year'], 3)} days a year; the worst year was "
                       f"{worst['year']} with {worst.get('days_short')} days short.")
    elif payload.get("mode") == "regional":
        sig = payload.get("signatures_m3s") or {}
        q95 = sig.get("q95") or {}
        out.append(f"No gauge: from {payload.get('n_donors')} donor catchments "
                   f"({payload.get('regionalisation_method')}) over an upstream area of "
                   f"{_fmt(payload.get('area_km2'))} km2, the flow exceeded 95 % of the time is about "
                   f"{_fmt(q95.get('value'))} m3/s (band {_fmt(q95.get('low'))} to {_fmt(q95.get('high'))} m3/s"
                   + (f", leave-one-out NSE {_fmt(q95.get('loo_nse'), 2)}" if q95.get("loo_nse") is not None else "")
                   + f"), the median {_fmt((sig.get('q50') or {}).get('value'))} m3/s.")
        if rel.get("daily") is not None:
            out.append(f"A demand of {_fmt(demand)} m3/s with {payload.get('reserve_rule')} and at most "
                       f"{_fmt(100 * float(payload.get('share') or 0), 3)} % of the flow taken needs "
                       f"{_fmt(payload.get('required_flow_m3s'))} m3/s in the river, exceeded about "
                       f"{_fmt(100 * float(rel['daily']), 3)} % of the time (band "
                       f"{_fmt(100 * float(rel.get('low') or 0), 3)} to "
                       f"{_fmt(100 * float(rel.get('high') or 0), 3)} %); verdict: {payload.get('verdict')}.")
    return out


def _template_answer(study: Study, run: StudyRun | None) -> str:
    plan = study.plan or {}
    lines: list[str] = []
    #: Where a new paragraph starts (one per step, plus every standalone line).
    starts: set[int] = set()
    said: dict[str, int] = {}

    def para(text: str) -> None:
        starts.add(len(lines))
        lines.append(text)

    def add(sentences: list[str]) -> None:
        # Two steps on the same record say the same thing once. analyze_station
        # and flood_frequency open the same sentence ("The 100-year return level
        # from 140 annual maxima: ..."), so the key is the opening clause; when a
        # later step says it with more in it, the fuller version replaces the
        # earlier one. Each step's sentences form one paragraph.
        first = True
        for s in sentences:
            key = s.strip().lower()[:48]
            if key in said:
                idx = said[key]
                if len(s) > len(lines[idx]):
                    lines[idx] = s
                continue
            if first:
                starts.add(len(lines))
                first = False
            said[key] = len(lines)
            lines.append(s)

    if plan.get("branch"):
        para(f"Plan {plan.get('playbook')}, branch {plan['branch']}, executed with no model in the loop.")
    for r in (run.results if run else []):
        if r.get("skipped"):
            para(f"Step {r.get('id')} ({r.get('tool')}) was skipped: {r.get('error')}.")
            continue
        if not r.get("ok"):
            para(f"Step {r.get('id')} ({r.get('tool')}) failed: {r.get('error')}.")
            continue
        payload = r.get("result")
        if isinstance(payload, dict):
            add(_sentences_for(r["tool"], payload, study))
        failed = [g for g in r.get("gates") or [] if not g.get("passed")]
        for g in failed:
            para(f"Gate {g['check']} on step {r.get('id')} failed: {g.get('detail')}.")
        fb = r.get("fallback")
        if r.get("fallback_used") and isinstance(fb, dict):
            passed = " and passed its gates." if fb.get("ok") and fb.get("gates_passed") else "."
            para(f"The fallback {fb.get('tool')} ran" + passed)
            if isinstance(fb.get("result"), dict):
                add(_sentences_for(fb["tool"], fb["result"], study))
    if run and run.stop_reason:
        para(f"The study stopped at step {run.stopped_at}: {run.stop_reason}.")
    if not lines:
        return "No step ran."
    paragraphs: list[list[str]] = []
    for i, line in enumerate(lines):
        if i in starts or not paragraphs:
            paragraphs.append([line])
        else:
            paragraphs[-1].append(line)
    return "\n\n".join(" ".join(p) for p in paragraphs)


# ── the loop ────────────────────────────────────────────────────────────────


def _problem_parts(problem: str | dict[str, Any], playbook: str | None,
                   intake: dict[str, Any] | None) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(problem, dict):
        text = str(problem.get("text") or problem.get("question") or problem.get("problem") or "")
        playbook = playbook or problem.get("kind") or problem.get("playbook")
        merged = dict(problem.get("params") or {})
        merged.update(intake or {})
        return text, playbook, merged
    return str(problem or ""), playbook, dict(intake or {})


def _tools_for_specialist() -> list[dict[str, Any]]:
    from aquascope import workbench
    from aquascope.ai_engine.analyst import _tool_specs

    out = []
    for s in _tool_specs():
        if s.name in ("run_python", "list_sources", "describe_methods", "list_analyses", "solve_plan", "solve_run",
                      "list_playbooks", "describe_playbook"):
            continue
        out.append({"tool": s.name, "arguments": list((s.parameters.get("properties") or {}).keys()),
                    "about": s.description[:120]})
    out += [{"tool": name, "arguments": ["from_step", "..."], "about": spec["summary"]}
            for name, spec in workbench.TOOLS.items() if spec["needs"] == "frame"]
    return out


def solve(
    problem: str | dict[str, Any],
    *,
    lat: float,
    lon: float,
    playbook: str | None = None,
    intake: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    review: Callable[[Study], Study | None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    max_replans: int = 1,
    client: Any | None = None,
    recon: dict[str, Any] | None = None,
    execute: bool = True,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> SolveResult:
    """Solve ``problem`` at (``lat``, ``lon``): recon, plan, review, execute with gates, replan once, report.

    A model is used only when asked for (``provider``, ``model``, ``api_key``,
    ``base_url`` or a ready ``client``); otherwise no role calls one. ``review``
    sees the Study before execution and returns the Study to run (edited or
    not) or None to decline. ``recon`` lets a caller pass a reconnaissance it
    already has instead of calling the Scout. ``execute=False`` stops after
    the plan (and the review): the MCP ``solve_plan`` face returns that study
    for the assistant to look at, and ``solve_run`` executes it later.
    ``tools`` adds to or replaces the runner's tools by name (see ``run_study``).
    """
    from aquascope import playbooks as pbk

    timeline: list[dict[str, Any]] = []
    cost: dict[str, dict[str, int]] = {}

    def say(event: dict[str, Any]) -> None:
        timeline.append(event)
        if on_event:
            on_event(event)

    text, playbook, intake = _problem_parts(problem, playbook, intake)
    site = {"lat": float(lat), "lon": float(lon)}
    problem_dict: dict[str, Any] = {"text": text, "site": site, "kind": playbook, "params": dict(intake)}

    llm, cfg = _model_for(provider, model, api_key, base_url, client, cost, timeline, say)

    # Coordinator, first pass: the keyword rules.
    ambiguous = False
    if playbook is None:
        playbook, ambiguous = choose_playbook(text)
        say({"role": "coordinator", "step": None, "event": "keywords",
             "detail": f"rules pick {playbook or 'nothing'}" + (" (ambiguous)" if ambiguous else "")})
    known = {p["id"] for p in pbk.list_playbooks()}
    kind = playbook if playbook in known else None

    # Scout.
    hints = intake_hints(text, kind)
    for k, v in hints.items():
        intake.setdefault(k, v)
    if recon is None:
        recon = _scout(site, kind, intake, say)
    else:
        say({"role": "scout", "step": None, "event": "recon", "detail": "reconnaissance supplied by the caller"})

    def declined(reason: str, kind_: str, study: Study | None = None) -> SolveResult:
        say({"role": "coordinator", "step": None, "event": "declined", "detail": reason})
        problem_dict["params"] = dict(intake)
        st = study or Study(question=text or "declined", version=2, author="playbook",
                            problem={"kind": kind, "site": site, "params": dict(intake), "text": text},
                            plan={"playbook": kind, "declined": reason, "kind": kind_})
        res = SolveResult(problem=problem_dict, study=st, recon=recon or {}, declined=True, declined_reason=reason,
                          timeline=timeline, cost=cost, model=cfg.get("model"), provider=cfg.get("provider"),
                          not_established=[reason], answer="", finished=_now())
        res.checks = []
        return res

    # Coordinator, second pass: settle an ambiguous problem with the model, or decline.
    if kind is None or ambiguous:
        if llm is not None:
            picked = _json_block(llm.call("coordinator", COORDINATOR_PROMPT, {
                "problem": text, "site": site, "recon": _recon_summary(recon),
                "playbooks": [{k: p.get(k) for k in ("id", "title", "problem", "description", "intake")}
                              for p in pbk.list_playbooks()],
            }))
            if picked and picked.get("playbook") in known:
                kind = str(picked["playbook"])
                for k, v in (picked.get("intake") or {}).items():
                    intake.setdefault(k, v)
                say({"role": "coordinator", "step": None, "event": "playbook",
                     "detail": f"{kind}: {picked.get('reason') or 'chosen by the model'}"})
            elif picked is not None and kind is None:
                return declined(
                    f"No playbook covers this problem ({picked.get('reason') or 'the model found no match'}); "
                    f"the playbooks are {', '.join(sorted(known))}.", "no_playbook")
        if kind is None:
            return declined(
                f"No playbook covers this problem; say which of {', '.join(sorted(known))} it is with --playbook.",
                "no_playbook")
    problem_dict["kind"] = kind

    # Plan (the tree).
    try:
        pb = pbk.load(kind)
        study = pbk.plan(pb, recon, intake, problem_text=text)
    except pbk.Declined as exc:
        return declined(exc.reason, exc.kind)
    problem_dict["params"] = dict((study.problem or {}).get("params") or intake)
    say({"role": "coordinator", "step": None, "event": "plan",
         "detail": f"playbook {kind}, branch {study.plan['branch']}, {len(study.steps)} steps"})
    if llm is not None:
        prose = llm.call("coordinator", RATIONALE_PROMPT, {
            "problem": text, "playbook": kind, "branch": study.plan.get("branch"),
            "tree_rationale": study.plan.get("rationale"),
            "steps": [{"id": s.id, "tool": s.tool, "rationale": s.rationale,
                       "gates": [g.get("check") for g in s.expects]} for s in study.steps],
            "recon": _recon_summary(recon),
        })
        if prose:
            study.plan["tree_rationale"] = study.plan.get("rationale")
            study.plan["rationale"] = prose

    # Review.
    if review is not None:
        edited = review(study)
        if edited is None:
            return declined("The plan was declined at review.", "review", study)
        if edited is not study:
            say({"role": "coordinator", "step": None, "event": "review", "detail": "the reviewer edited the plan"})
        else:
            say({"role": "coordinator", "step": None, "event": "review", "detail": "the reviewer approved the plan"})
        study = edited
    if not execute:
        say({"role": "coordinator", "step": None, "event": "plan_ready", "detail": "returned without executing"})
        return SolveResult(problem=problem_dict, study=study, recon=recon, timeline=timeline, cost=cost,
                           model=cfg.get("model"), provider=cfg.get("provider"), finished=_now())

    return _execute(study, pb=pb, kind=kind, text=text, intake=intake, recon=recon, site=site,
                    problem_dict=problem_dict, llm=llm, cfg=cfg, timeline=timeline, cost=cost, say=say,
                    max_replans=max_replans, tools=tools)


def _model_for(
    provider: str | None, model: str | None, api_key: str | None, base_url: str | None, client: Any | None,
    cost: dict[str, dict[str, int]], timeline: list[dict[str, Any]], say: Callable[[dict[str, Any]], None],
) -> tuple[_Model | None, dict[str, Any]]:
    """The model the roles may call, or None: only when one was asked for, never from the environment alone."""
    cfg: dict[str, Any] = {"model": None, "provider": None}
    if client is None and not any((provider, model, api_key, base_url)):
        return None, cfg
    if client is None:
        from aquascope.ai_engine.analyst import resolve_llm
        from aquascope.ai_engine.llm_transport import make_client

        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
    else:
        cfg = {"model": model or "test", "provider": provider or "custom"}
    return _Model(client, str(cfg["model"]), str(cfg["provider"]), cost, timeline, say), cfg


def _execute(
    study: Study,
    *,
    pb: Any,
    kind: str | None,
    text: str,
    intake: dict[str, Any],
    recon: dict[str, Any],
    site: dict[str, Any],
    problem_dict: dict[str, Any],
    llm: _Model | None,
    cfg: dict[str, Any],
    timeline: list[dict[str, Any]],
    cost: dict[str, dict[str, int]],
    say: Callable[[dict[str, Any]], None],
    max_replans: int,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> SolveResult:
    """Run a reviewed study with the Reviewer's gates, replan once, then report: the half shared by
    ``solve`` and ``run_reviewed``. ``pb`` is the playbook a branch replan is filled from (None: no replan)."""
    from aquascope import playbooks as pbk

    run = run_study(study, on_event=say, tools=tools)
    replans = 0
    while run.stop_reason and replans < max_replans:
        replans += 1
        if run.replan:
            branch = run.replan["branch"]
            if pb is None:
                say({"role": "specialist", "step": run.stopped_at, "event": "replan_declined",
                     "detail": "no playbook to fill the branch from"})
                break
            try:
                new = pbk.plan(pb, recon, intake, branch=branch, problem_text=text)
            except pbk.Declined as exc:
                say({"role": "specialist", "step": run.stopped_at, "event": "replan_declined", "detail": exc.reason})
                break
            new.plan["replanned_from"] = {"branch": study.plan.get("branch"), "step": run.stopped_at,
                                          "reason": run.replan.get("reason")}
            if study.plan.get("tree_rationale"):
                new.plan["rationale"] = study.plan["rationale"]
            say({"role": "specialist", "step": run.stopped_at, "event": "replan",
                 "detail": f"branch {branch} after {run.stop_reason}"})
            study = new
            run = run_study(study, on_event=say, prior=run, tools=tools)
            continue
        if llm is None:
            break
        failed = next((r for r in run.results if r.get("id") == run.stopped_at), None)
        step = study.step_by_id(run.stopped_at or "")
        if failed is None or step is None:
            break
        proposal = _json_block(llm.call("specialist", f"{SPECIALIST_PROMPTS.get(kind, SPECIALIST_PROMPTS['default'])}\n"
                                        f"{SPECIALIST_RULES}", {
            "problem": text, "playbook": kind, "branch": study.plan.get("branch"),
            "failed_step": {"id": step.id, "tool": step.tool, "arguments": step.arguments,
                            "rationale": step.rationale},
            "failed_gates": [g for g in failed.get("gates") or [] if not g.get("passed")],
            "result": _compact(failed.get("result")),
            "earlier_fallback": _compact(failed.get("fallback")) if failed.get("fallback") else None,
            "recon": _recon_summary(recon),
            "tools": _tools_for_specialist(),
        }, step=step.id))
        from aquascope.study import tool_names

        if not proposal or not proposal.get("tool") or proposal["tool"] not in tool_names():
            say({"role": "specialist", "step": step.id, "event": "no_fallback",
                 "detail": (proposal or {}).get("rationale") or "the specialist proposed no usable fallback"})
            break
        fb_step = {"tool": str(proposal["tool"]), "arguments": dict(proposal.get("arguments") or {}),
                   "rationale": str(proposal.get("rationale") or "proposed by the specialist after the gate failed"),
                   "expects": [g for g in (proposal.get("expects") or []) if isinstance(g, dict)]}
        step.fallback = {"step": fb_step}
        study.plan.setdefault("replans", []).append({"step": step.id, "reason": run.stop_reason, "fallback": fb_step})
        say({"role": "specialist", "step": step.id, "event": "replan",
             "detail": f"fallback {fb_step['tool']}: {fb_step['rationale']}"})
        run = run_study(study, on_event=say, prior=run, tools=tools)

    # Reviewer: what the run established and what it did not.
    result = SolveResult(problem=problem_dict, study=study, recon=recon, run=run, timeline=timeline, cost=cost,
                         model=cfg.get("model"), provider=cfg.get("provider"))
    not_established: list[str] = []
    for g in run.failed_gates:
        not_established.append(f"Step {g['step']}, gate {g['check']}: {g.get('detail')}")
    for r in run.results:
        if not r.get("ok"):
            not_established.append(f"Step {r.get('id')} ({r.get('tool')}) did not run: {r.get('error')}")
    if run.stop_reason:
        not_established.append(f"The study stopped at {run.stopped_at}: {run.stop_reason}")
    for n in (study.plan or {}).get("notes") or []:
        not_established.append(n)
    from aquascope.ai_engine.analyst import _harvest_provenance

    seen: list[dict[str, Any]] = []
    for r in run.results:
        if r.get("skipped"):
            continue
        seen.append({"name": r["tool"], "arguments": r["arguments"], "payload": r.get("result"), "ok": r["ok"]})
        _harvest_provenance(r["tool"], r["arguments"], r.get("result"), result)
        fb = r.get("fallback")
        if isinstance(fb, dict) and fb.get("tool"):
            seen.append({"name": fb["tool"], "arguments": fb.get("arguments") or {}, "payload": fb.get("result"),
                         "ok": bool(fb.get("ok"))})
            _harvest_provenance(fb["tool"], fb.get("arguments") or {}, fb.get("result"), result)
    say({"role": "reviewer", "step": None, "event": "gates",
         "detail": f"{len(run.gates) - len(run.failed_gates)} of {len(run.gates)} gates passed"})

    # Narrator.
    answer = None
    if llm is not None:
        from aquascope.ai_engine.analyst import SYSTEM_PROMPT

        answer = llm.call("narrator", SYSTEM_PROMPT + NARRATOR_RULES, {
            "problem": text, "site": site, "intake": problem_dict["params"],
            "plan": {k: study.plan.get(k) for k in ("playbook", "branch", "rationale")},
            "steps": [{
                "id": r.get("id"), "tool": r["tool"], "arguments": r["arguments"], "ok": r["ok"],
                "error": r.get("error"), "gates": r.get("gates"), "fallback_used": r.get("fallback_used"),
                "result": _compact(r.get("result")),
                "fallback": _compact({k: v for k, v in (r.get("fallback") or {}).items()
                                      if k in ("tool", "arguments", "ok", "gates", "result")})
                if r.get("fallback") else None,
            } for r in run.results],
            "not_established": not_established,
            "caveats": result.caveats,
        })
    if not answer:
        answer = _template_answer(study, run)
        say({"role": "narrator", "step": None, "event": "template", "detail": f"{len(answer)} chars"})
    result.answer = answer

    from aquascope.ai_engine.verify import verify as _verify

    # The gates' own words ("142.9 years of record, 20 needed") and the plan are
    # legitimate sources for the prose too, so they join the pool the checks read.
    gate_pool = {"gates": [g for r in run.results for g in (r.get("gates") or [])], "plan": study.plan}
    checks = _verify(answer, [*seen, {"name": "gates", "arguments": {}, "payload": gate_pool, "ok": True}],
                     question=text)
    result.checks = checks.to_dict()["checks"]
    for c in checks.failed:
        not_established.append(c.detail or c.name)
    result.not_established = not_established
    say({"role": "reviewer", "step": None, "event": "checks",
         "detail": f"{len(checks.checks) - len(checks.failed)} of {len(checks.checks)} checks passed"})
    result.finished = _now()
    return result


def run_reviewed(
    study: Study | dict[str, Any] | str,
    *,
    recon: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    max_replans: int = 1,
    client: Any | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> SolveResult:
    """Execute a study planned earlier and reviewed elsewhere, and report as ``solve`` would.

    The page plans with ``solve(..., execute=False)``, shows the checklist, lets
    the reader edit it, and hands the study back here: the gates, one bounded
    replan, the Reviewer's "not established" list and the Narrator are the same
    code path. The study carries its problem, site, intake and playbook, so
    nothing is asked twice; ``recon`` is the reconnaissance the caller already
    has (the Scout runs otherwise); ``tools`` adds to or replaces the runner's
    tools by name (see ``run_study``). Accepts a Study, its dict or its YAML.
    """
    from aquascope import playbooks as pbk
    from aquascope.study import loads as _loads

    if isinstance(study, str):
        st = _loads(study)
    elif isinstance(study, dict):
        st = Study.from_dict(dict(study))
    else:
        st = study
    if not st.steps:
        raise ValueError("the study has no steps")

    timeline: list[dict[str, Any]] = []
    cost: dict[str, dict[str, int]] = {}

    def say(event: dict[str, Any]) -> None:
        timeline.append(event)
        if on_event:
            on_event(event)

    llm, cfg = _model_for(provider, model, api_key, base_url, client, cost, timeline, say)
    problem = dict(st.problem or {})
    plan = st.plan or {}
    text = str(problem.get("text") or st.question or "")
    kind = plan.get("playbook") or problem.get("kind")
    intake = dict(problem.get("params") or {})
    raw_site = problem.get("site") or {}
    site: dict[str, Any] = {}
    if isinstance(raw_site, dict) and raw_site.get("lat") is not None and raw_site.get("lon") is not None:
        site = {"lat": float(raw_site["lat"]), "lon": float(raw_site["lon"])}
    problem_dict: dict[str, Any] = {"text": text, "site": site, "kind": kind, "params": dict(intake)}
    known = {p["id"] for p in pbk.list_playbooks()}
    pb = pbk.load(kind) if kind in known else None
    if recon is not None:
        say({"role": "scout", "step": None, "event": "recon", "detail": "reconnaissance supplied by the caller"})
    elif site:
        recon = _scout(site, kind if kind in known else None, intake, say)
    else:
        recon = {"point": {}, "stations": [], "catchment": {}, "context": {"years_by_variable": {}},
                 "sufficiency": [], "notes": ["the study names no site: no reconnaissance"]}
        say({"role": "scout", "step": None, "event": "recon", "detail": "the study names no site; no reconnaissance"})
    say({"role": "coordinator", "step": None, "event": "review",
         "detail": f"running the reviewed plan: {len(st.steps)} step(s)"})
    return _execute(st, pb=pb, kind=kind, text=text, intake=intake, recon=recon, site=site,
                    problem_dict=problem_dict, llm=llm, cfg=cfg, timeline=timeline, cost=cost, say=say,
                    max_replans=max_replans, tools=tools)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _scout(site: dict[str, float], kind: str | None, intake: dict[str, Any],
           say: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    rp = intake.get("return_period")
    # The reconnaissance narrows its table by the registry's problem kind, which is the playbook's `problem`
    # (drought for drought_status), not its id.
    problem: str | None = None
    if kind:
        from aquascope import playbooks as pbk

        try:
            problem = pbk.load(kind).problem
        except pbk.PlaybookError:
            problem = kind
    try:
        from aquascope.explore import assess_site

        recon = assess_site(site["lat"], site["lon"], problem=problem,
                            return_period=float(rp) if isinstance(rp, (int, float)) else None)
    except Exception as exc:  # noqa: BLE001 - no reconnaissance is a fact the plan has to live with
        recon = {"point": dict(site), "stations": [], "catchment": {},
                 "context": {"years_by_variable": {}, "resolution_by_variable": {}, "ungauged": True},
                 "sufficiency": [], "notes": [f"reconnaissance unavailable: {type(exc).__name__}: {exc}"],
                 "error": f"{type(exc).__name__}: {exc}"}
        say({"role": "scout", "step": None, "event": "error", "detail": recon["error"]})
        return recon
    if not isinstance(recon, dict):
        recon = {"point": dict(site), "stations": [], "context": {"years_by_variable": {}}, "notes": []}
    ctx = recon.get("context") or {}
    years = ctx.get("years_by_variable") or {}
    detail = f"{len(recon.get('stations') or [])} stations within reach; " + (
        ", ".join(f"{k} {v:g} years" for k, v in years.items()) if years else "no usable record")
    say({"role": "scout", "step": None, "event": "recon", "detail": detail})
    return recon
