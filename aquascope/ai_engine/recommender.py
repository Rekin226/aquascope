"""
AI-powered research methodology recommender.

Works in two modes:
1. **Rule-based** (default, zero-cost): scores methodologies from the
   built-in knowledge base against the user's dataset profile.
2. **LLM-enhanced** (optional): sends the dataset summary + knowledge base
   to an LLM (OpenAI, Anthropic, or local Ollama) for nuanced reasoning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from aquascope.ai_engine.knowledge_base import (
    METHODOLOGIES,
    ResearchMethodology,
)

logger = logging.getLogger(__name__)


# ── Data profile helper ──────────────────────────────────────────────

@dataclass
class DatasetProfile:
    """Summary of a collected dataset, used as input to the recommender."""

    parameters: list[str] = field(default_factory=list)
    n_records: int = 0
    n_stations: int = 0
    time_span_years: float = 0.0
    geographic_scope: str = ""       # e.g. "Taiwan", "Global", "Tamsui River basin"
    data_sources: list[str] = field(default_factory=list)
    research_goal: str = ""          # free-text from the user
    keywords: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """A single methodology recommendation with a relevance score."""

    methodology: ResearchMethodology
    score: float           # 0-100
    rationale: str = ""


# ── Rule-based scorer ────────────────────────────────────────────────

def _score_methodology(method: ResearchMethodology, profile: DatasetProfile) -> float:
    """
    Heuristic scorer.  Returns 0-100.

    Scoring criteria (weights):
      - Parameter overlap      : 40 %
      - Data sufficiency       : 25 %
      - Scale match            : 15 %
      - Keyword / tag overlap  : 20 %
    """
    # 1. parameter overlap (Jaccard-like)
    if method.applicable_parameters:
        user_params = {p.lower() for p in profile.parameters}
        method_params = {p.lower() for p in method.applicable_parameters}
        overlap = len(user_params & method_params)
        param_score = (overlap / max(len(method_params), 1)) * 100
    else:
        param_score = 50  # neutral

    # 2. data sufficiency heuristic
    data_score = 50
    reqs = " ".join(method.data_requirements).lower()
    if "time-series" in reqs or "years" in reqs:
        if profile.time_span_years >= 5:
            data_score = 90
        elif profile.time_span_years >= 1:
            data_score = 60
        else:
            data_score = 20
    if "multi-site" in reqs:
        data_score = min(data_score, 90 if profile.n_stations >= 5 else 30)
    if profile.n_records >= 200:
        data_score = max(data_score, 70)

    # 3. scale match
    scale_map = {
        "lab": 1, "pilot": 2, "field": 3, "regional": 4, "global": 5,
    }
    scope_lower = profile.geographic_scope.lower()
    if "global" in scope_lower:
        user_scale = 5
    elif any(w in scope_lower for w in ("region", "basin", "national", "taiwan")):
        user_scale = 4
    elif any(w in scope_lower for w in ("field", "river", "station")):
        user_scale = 3
    elif "pilot" in scope_lower:
        user_scale = 2
    else:
        user_scale = 3

    method_scale = scale_map.get(method.typical_scale, 3)
    scale_score = max(0, 100 - abs(user_scale - method_scale) * 25)

    # 4. keyword / tag overlap
    user_kw = {k.lower() for k in profile.keywords}
    user_kw |= {w.lower() for w in profile.research_goal.split()}
    method_tags = {t.lower() for t in method.tags}
    if user_kw and method_tags:
        tag_score = (len(user_kw & method_tags) / max(len(method_tags), 1)) * 100
    else:
        tag_score = 30

    total = (
        param_score * 0.40
        + data_score * 0.25
        + scale_score * 0.15
        + tag_score * 0.20
    )
    return round(total, 1)


def _generate_rationale(method: ResearchMethodology, profile: DatasetProfile, score: float) -> str:
    parts = []
    user_params = {p.lower() for p in profile.parameters}
    matched = user_params & {p.lower() for p in method.applicable_parameters}
    if matched:
        parts.append(f"Your dataset includes {', '.join(sorted(matched))} which are key inputs for this method.")
    if profile.time_span_years >= 2 and "time-series" in " ".join(method.data_requirements).lower():
        parts.append(f"You have ~{profile.time_span_years:.0f} years of data, meeting the time-series requirement.")
    if profile.n_stations >= 5 and "multi-site" in " ".join(method.data_requirements).lower():
        parts.append(f"Your {profile.n_stations} stations satisfy the multi-site requirement.")
    if not parts:
        cat = method.category.replace('_', ' ')
        parts.append(f"This methodology is generally applicable to {cat} studies in the water domain.")
    return " ".join(parts)


# ── Public API ───────────────────────────────────────────────────────

def recommend(
    profile: DatasetProfile,
    top_k: int = 5,
    min_score: float = 20.0,
) -> list[Recommendation]:
    """
    Return the top-k methodology recommendations for the given dataset profile.

    Parameters
    ----------
    profile : DatasetProfile
    top_k : int
    min_score : float
        Minimum relevance score to include.

    Returns
    -------
    list[Recommendation]  sorted by descending score.
    """
    scored: list[Recommendation] = []
    for method in METHODOLOGIES:
        score = _score_methodology(method, profile)
        if score >= min_score:
            rationale = _generate_rationale(method, profile, score)
            scored.append(Recommendation(methodology=method, score=score, rationale=rationale))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


# ── Optional LLM-enhanced recommendation ─────────────────────────────

def recommend_with_llm(
    profile: DatasetProfile,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[Recommendation]:
    """
    Use an LLM to provide more nuanced methodology recommendations.

    Falls back to rule-based if the LLM call fails.

    Supports OpenAI-compatible APIs (OpenAI, Anthropic via proxy,
    local Ollama at http://localhost:11434/v1).
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; falling back to rule-based.")
        return recommend(profile, top_k=top_k)

    # Build a compact JSON of the knowledge base
    kb_json = json.dumps(
        [asdict(m) for m in METHODOLOGIES], indent=1, ensure_ascii=False
    )
    profile_json = json.dumps(asdict(profile), indent=1, ensure_ascii=False)

    system_prompt = (
        "You are an expert water-resources research advisor. "
        "Given a dataset profile and a catalogue of research methodologies, "
        "recommend the most suitable methodologies. "
        "For each recommendation, provide: methodology id, a relevance score (0-100), "
        "and a concise rationale explaining why it fits the dataset. "
        "Return valid JSON: a list of objects with keys 'id', 'score', 'rationale'. "
        f"Return at most {top_k} recommendations sorted by score descending."
    )

    user_prompt = (
        f"## Dataset Profile\n```json\n{profile_json}\n```\n\n"
        f"## Available Methodologies\n```json\n{kb_json}\n```"
    )

    try:
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_text = resp.choices[0].message.content or "[]"
        parsed = json.loads(raw_text)
        items = parsed if isinstance(parsed, list) else parsed.get("recommendations", [])

        results: list[Recommendation] = []
        for item in items[:top_k]:
            method = next((m for m in METHODOLOGIES if m.id == item["id"]), None)
            if method:
                results.append(
                    Recommendation(
                        methodology=method,
                        score=float(item.get("score", 50)),
                        rationale=item.get("rationale", ""),
                    )
                )
        return results

    except Exception as exc:
        logger.warning("LLM recommendation failed (%s); falling back to rule-based.", exc)
        return recommend(profile, top_k=top_k)
