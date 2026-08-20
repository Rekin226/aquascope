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
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from aquascope.ai_engine.knowledge_base import (
    METHODOLOGIES,
    ResearchMethodology,
)
from aquascope.ai_engine.providers import PROVIDERS as _REGISTRY

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


@dataclass
class RecommendationResult:
    """
    Recommendations plus how they were produced.

    ``mode`` is ``"llm"`` when a language model actually answered, and
    ``"rule_based"`` when the heuristic scorer produced the list.  When the
    LLM was requested but could not be used, ``error`` holds a human-readable
    reason so callers can show it instead of silently degrading.
    """

    recommendations: list[Recommendation]
    mode: str = "rule_based"      # "llm" | "rule_based"
    provider: str = ""
    model: str = ""
    error: str = ""

    @property
    def used_llm(self) -> bool:
        return self.mode == "llm"

    def __iter__(self):
        return iter(self.recommendations)

    def __len__(self) -> int:
        return len(self.recommendations)


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
_HUGGINGFACE_HOST = "huggingface.co"   # matches router.* and the retired api-inference.*
_GROQ_HOST = "api.groq.com"

# Public constants used by the dashboard UI. The registry
# (aquascope.ai_engine.providers) is the single source: this module used to keep
# its own copies, and they drifted from the analyst's until a retired Groq model
# broke both at once.
PROVIDER_BASE_URLS: dict[str, str | None] = {
    p.id: (None if p.id == "openai" else p.base_url) for p in _REGISTRY.values()
}

PROVIDER_MODELS: dict[str, list[str]] = {
    p.id: list(p.models or [p.model]) for p in _REGISTRY.values()
}

# ── Deployment-supplied ("hosted") credentials ───────────────────────
#
# A deployment can put ONE free API token in the environment so that visitors
# get LLM-backed recommendations without creating an account of their own.
# This is how the public HuggingFace Space demo works: the Space owner adds
# HF_TOKEN as a Space secret.  No key is ever bundled in the package.

HOSTED_HF_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
HOSTED_GENERIC_ENV_KEY = "AQUASCOPE_LLM_API_KEY"

# Small, fast, reliable at structured output, and cheap on a free quota.
HOSTED_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def hosted_llm_config() -> dict[str, Any] | None:
    """
    Return an LLM config supplied by the deployment's environment, or None.

    Resolution order:

    1. ``AQUASCOPE_LLM_API_KEY`` with optional ``AQUASCOPE_LLM_BASE_URL`` and
       ``AQUASCOPE_LLM_MODEL`` — any OpenAI-compatible provider.
    2. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` — HuggingFace serverless
       inference, the convention a HuggingFace Space secret already follows.

    Returns None when nothing is configured, which is the normal case for a
    plain ``pip install``: the recommender then stays rule-based unless the
    user supplies their own key in the UI.
    """
    base_url_override = os.environ.get("AQUASCOPE_LLM_BASE_URL", "").strip()
    model_override = os.environ.get("AQUASCOPE_LLM_MODEL", "").strip()

    generic_key = os.environ.get(HOSTED_GENERIC_ENV_KEY, "").strip()
    if generic_key:
        base_url = base_url_override or None
        return {
            "provider": _detect_provider(base_url),
            "model": model_override or "gpt-4o-mini",
            "api_key": generic_key,
            "base_url": base_url,
            "hosted": True,
        }

    hf_key = next(
        (os.environ[k].strip() for k in HOSTED_HF_ENV_KEYS if os.environ.get(k, "").strip()),
        "",
    )
    if hf_key:
        base_url = base_url_override or PROVIDER_BASE_URLS["huggingface"]
        return {
            "provider": _detect_provider(base_url),
            "model": model_override or HOSTED_DEFAULT_HF_MODEL,
            "api_key": hf_key,
            "base_url": base_url,
            "hosted": True,
        }

    return None


class LLMOutputError(ValueError):
    """The LLM replied, but the reply could not be turned into recommendations."""


class LLMDependencyError(ImportError):
    """A Python package required to reach the chosen provider is missing."""


def _detect_provider(base_url: str | None) -> str:
    """Map a base_url onto a provider name used for request tuning and messages."""
    if not base_url:
        return "openai"
    if _HUGGINGFACE_HOST in base_url:
        return "huggingface"
    if _GROQ_HOST in base_url:
        return "groq"
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "ollama"
    if _OPENAI_HOST in base_url:
        return "openai"
    return "openai-compatible"


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

    # A JSON *object* wrapper, not a bare array: providers that support
    # response_format={"type": "json_object"} reject a top-level array, and
    # _parse_llm_output accepts either shape anyway.
    system_prompt = (
        "You are a water-resources research advisor. "
        "Output ONLY a JSON object — no markdown, no explanation, no code fences. "
        f"Pick the top {top_k} methodologies from the catalogue that best fit the dataset. "
        'Reply with {"recommendations": [...]} where each element has exactly: '
        '"id" (string, copied verbatim from the catalogue), '
        '"score" (integer 0-100), "rationale" (one sentence).'
    )
    user_prompt = (
        f"Dataset: {profile_json}\n\n"
        f"Catalogue: {kb_json}"
    )
    return system_prompt, user_prompt


def _coerce_items(parsed: Any) -> list[Any]:
    """Pull the recommendation list out of whatever JSON shape the model returned."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Preferred key first, then any other list-of-objects value — models
        # routinely rename the wrapper to "methodologies", "results", "items"…
        preferred = parsed.get("recommendations")
        if isinstance(preferred, list):
            return preferred
        for value in parsed.values():
            if isinstance(value, list) and all(isinstance(v, dict) for v in value):
                return value
        # A single recommendation returned unwrapped.
        if "id" in parsed:
            return [parsed]
    return []


def _parse_llm_output(raw_text: str, top_k: int) -> list[Recommendation]:
    """
    Turn a raw model reply into Recommendations.

    Raises
    ------
    LLMOutputError
        If the reply is not JSON, or contains no id that matches the catalogue.
    """
    import re as _re

    text = _re.sub(r"```(?:json)?\s*", "", raw_text or "").strip().rstrip("`").strip()
    if not text:
        raise LLMOutputError("the model returned an empty reply")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Some models wrap JSON in prose; salvage the first object or array.
        match = _re.search(r"[\[{].*[\]}]", text, _re.DOTALL)
        if not match:
            raise LLMOutputError(
                f"the model did not return JSON (got: {text[:120]!r})"
            ) from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"the model returned malformed JSON ({exc})") from None

    items = _coerce_items(parsed)
    if not items:
        raise LLMOutputError("the model returned JSON with no recommendation list")

    by_id = {m.id: m for m in METHODOLOGIES}
    results: list[Recommendation] = []
    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue
        method = by_id.get(str(item.get("id", "")))
        if method is None:
            continue
        try:
            score = float(item.get("score", 50))
        except (TypeError, ValueError):
            score = 50.0
        results.append(
            Recommendation(
                methodology=method,
                score=score,
                rationale=str(item.get("rationale", "")),
            )
        )

    if not results:
        raise LLMOutputError(
            "the model returned no methodology id matching the catalogue"
        )
    return results


def _describe_llm_error(exc: BaseException, provider: str, model: str) -> str:
    """Turn an exception from the LLM path into a short, actionable sentence."""
    text = str(exc).strip()
    lowered = text.lower()
    label = provider or "the provider"

    if isinstance(exc, LLMDependencyError):
        return text
    if isinstance(exc, LLMOutputError):
        return f"{label} replied, but {text}."
    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return f"{label} rejected the API key (authentication failed)."
    if "403" in text or "forbidden" in lowered or "permission" in lowered:
        return f"{label} refused the request (check the key's permissions)."
    if "402" in text or "payment required" in lowered or "credits" in lowered:
        return f"{label} reports the account is out of inference credits."
    if "429" in text or "rate limit" in lowered or "quota" in lowered:
        return f"{label} rate limit or quota exceeded — retry later."
    if "404" in text or "does not exist" in lowered or "not found" in lowered:
        return f"{label} does not serve the model '{model}' — pick a different model."
    if "timeout" in lowered or "timed out" in lowered:
        return f"{label} did not respond before the timeout."
    if any(
        w in lowered
        for w in ("could not resolve", "name or service", "nodename", "connect", "refused", "unreachable")
    ):
        return f"Could not reach {label}: {text[:160]}"

    detail = text[:200] or type(exc).__name__
    return f"{label} call failed: {type(exc).__name__}: {detail}"


def _call_openai_compatible(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> str:
    """Call any OpenAI-compatible chat endpoint and return the raw reply text."""
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise LLMDependencyError(
            "The 'openai' package is not installed, so LLM providers are "
            "unavailable. Install it with:  pip install 'aquascope[llm]'"
        ) from exc

    client = OpenAI(
        api_key=api_key or "dummy",
        base_url=base_url or None,
        timeout=httpx.Timeout(timeout, connect=10.0),
    )
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    # json_object mode is reliable on OpenAI and Groq; other endpoints vary.
    if provider in ("openai", "groq"):
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


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


def recommend_with_llm_detailed(
    profile: DatasetProfile,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> RecommendationResult:
    """
    Use an LLM for nuanced methodology recommendations, reporting what happened.

    Unlike :func:`recommend_with_llm`, this never degrades silently: on any
    failure it still returns rule-based recommendations, but ``mode`` is
    ``"rule_based"`` and ``error`` explains why the LLM was not used, so a UI
    or CLI can show the reason to the user.

    Supported providers (detected from base_url):
    - OpenAI (base_url=None or api.openai.com)
    - HuggingFace serverless inference (router.huggingface.co) — free tier
    - Groq (api.groq.com) — free tier
    - Ollama local (localhost) — uses native /api/chat to avoid openai-client quirks
    - Any other OpenAI-compatible endpoint
    """
    provider = _detect_provider(base_url)
    system_prompt, user_prompt = _build_prompts(profile, top_k)

    try:
        if provider == "ollama":
            raw_text = _call_ollama_native(
                base_url or PROVIDER_BASE_URLS["ollama"],  # type: ignore[arg-type]
                model,
                system_prompt,
                user_prompt,
                timeout,
            )
        else:
            raw_text = _call_openai_compatible(
                provider, model, api_key, base_url, system_prompt, user_prompt, timeout
            )

        recs = _parse_llm_output(raw_text, top_k)
        return RecommendationResult(
            recommendations=recs, mode="llm", provider=provider, model=model
        )

    except Exception as exc:
        reason = _describe_llm_error(exc, provider, model)
        logger.warning("LLM recommendation failed (%s); falling back to rule-based.", reason)
        return RecommendationResult(
            recommendations=recommend(profile, top_k=top_k),
            mode="rule_based",
            provider=provider,
            model=model,
            error=reason,
        )


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

    Falls back to rule-based if the LLM call fails.  Use
    :func:`recommend_with_llm_detailed` when you need to know whether the LLM
    actually answered, and why it did not.
    """
    return recommend_with_llm_detailed(
        profile,
        top_k=top_k,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    ).recommendations
