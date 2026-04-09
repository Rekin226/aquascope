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

import httpx

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

_OPENAI_HOST = "api.openai.com"


def _build_prompts(profile: DatasetProfile, top_k: int) -> tuple[str, str]:
    """Build (system_prompt, user_prompt). Uses a compact KB to reduce token count."""
    profile_json = json.dumps(asdict(profile), ensure_ascii=False)

    # Compact KB — only the fields the LLM needs for matching.
    compact_kb = [
        {
            "id": m.id,
            "name": m.name,
            "category": m.category,
            "params": m.applicable_parameters,
            "scale": m.typical_scale,
            "tags": m.tags,
        }
        for m in METHODOLOGIES
    ]
    kb_json = json.dumps(compact_kb, ensure_ascii=False)

    system_prompt = (
        "You are a water-resources research advisor. "
        "Output ONLY a JSON array — no markdown, no explanation, no code fences. "
        f"Pick the top {top_k} methodologies from the catalogue that best fit the dataset. "
        "Each element must have exactly: "
        "\"id\" (string), \"score\" (integer 0-100), \"rationale\" (one sentence)."
    )
    user_prompt = (
        f"Dataset: {profile_json}\n\n"
        f"Catalogue: {kb_json}"
    )
    return system_prompt, user_prompt


def _parse_llm_output(raw_text: str, top_k: int) -> list[Recommendation]:
    import re as _re

    raw_text = _re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
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


def _call_ollama_native(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> str:
    """
    Call Ollama's native /api/chat endpoint via httpx.

    Using the native endpoint (instead of the OpenAI-compatible /v1/chat/completions)
    lets us pass Ollama-specific options like think=false (disables qwen3/deepseek
    chain-of-thought mode which would otherwise generate thousands of reasoning tokens
    and time out on every request).
    """
    import httpx

    # Derive native host from base_url, e.g.
    # "http://localhost:11434/v1" → "http://localhost:11434"
    native_base = base_url.rstrip("/")
    if native_base.endswith("/v1"):
        native_base = native_base[:-3]
    chat_url = f"{native_base}/api/chat"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,          # disable qwen3 / deepseek thinking mode
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,  # cap output tokens for local models
        },
    }

    resp = httpx.post(chat_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def recommend_with_llm(
    profile: DatasetProfile,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> list[Recommendation]:
    """
    Use an LLM to provide more nuanced methodology recommendations.

    Falls back to rule-based if the LLM call fails.

    Supports:
    - OpenAI API (base_url=None or contains api.openai.com): uses openai library.
    - Local Ollama (base_url like "http://localhost:11434/v1"): calls Ollama's
      native /api/chat endpoint directly via httpx, bypassing the openai client
      compatibility issues (broken timeout, unsupported parameters, qwen3 thinking
      mode causing runaway token generation).
    """
    is_openai = base_url is None or _OPENAI_HOST in base_url
    system_prompt, user_prompt = _build_prompts(profile, top_k)

    try:
        if is_openai:
            try:
                import httpx
                from openai import OpenAI
            except ImportError:
                logger.warning("openai package not installed; falling back to rule-based.")
                return recommend(profile, top_k=top_k)

            client = OpenAI(
                api_key=api_key or "openai",
                base_url=base_url or None,
                timeout=httpx.Timeout(timeout, connect=10.0),
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "[]"
        else:
            # Non-OpenAI provider: use Ollama's native API via httpx.
            raw_text = _call_ollama_native(
                base_url, model, system_prompt, user_prompt, timeout  # type: ignore[arg-type]
            )

        return _parse_llm_output(raw_text, top_k)

    except Exception as exc:
        logger.warning("LLM recommendation failed (%s); falling back to rule-based.", exc)
        return recommend(profile, top_k=top_k)
