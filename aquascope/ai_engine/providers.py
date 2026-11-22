"""The LLM providers AquaScope can talk to, in one place.

There used to be three lists that disagreed: :mod:`aquascope.ai_engine.analyst`
(the tool loop), :mod:`aquascope.ai_engine.recommender` (the dashboard picker)
and ``explorer/app.js`` (the browser). They drifted, and when Groq retired
``llama-3.3-70b-versatile`` on 2026-08-16 every default that still named it
broke at once.

So: one registry here, read by the Python side, and written out as JSON for the
Explorer by :func:`as_json` (``explorer/providers.json``, refreshed by
``python -m aquascope.ai_engine.providers``). A model id is edited once.

Every provider listed here supports tool calling on the model named below and
(except Ollama, which is local) allows a browser to call it directly, which is
what makes bring-your-own-key work on a static page with no server of ours.
Most speak the OpenAI chat-completions API; Anthropic speaks its own Messages
API, and ``api`` says which, so :func:`aquascope.ai_engine.llm_transport.make_client`
can pick the transport.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["PRICES", "PROVIDERS", "Provider", "as_json", "default_model", "env_var", "price_for", "provider_ids",
           "usd_for", "write_json"]

#: Characters per token used to turn a model's published window (tokens) into the
#: character ceiling a caller can compare a prompt with. Code and prose run close
#: to four, and under-counting the tokens keeps an early refusal from firing for a
#: prompt that would actually fit: the endpoint's own 413 stays the final judge.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Provider:
    """One LLM endpoint the Analyst can use."""

    id: str
    label: str
    base_url: str | None
    model: str
    env: str | None
    #: Models worth offering in a picker; the first is the default.
    models: list[str] = field(default_factory=list)
    #: Free tier a newcomer can actually get, if any.
    free: str | None = None
    signup: str | None = None
    #: Reachable from a browser (CORS), so the Explorer can offer it.
    browser: bool = True
    note: str | None = None
    #: The wire protocol: "openai" (chat completions) or "anthropic" (the Messages API).
    api: str = "openai"
    #: How much conversation (characters) the Analyst may keep before trimming old
    #: tool results. None means the conservative free-tier default in the loop.
    #: This is a trimming budget, NOT the model's context window: it is sized for
    #: what a free tier will accept in one request, and on every provider but
    #: Anthropic it is far below the window. Use ``context_window`` (and
    #: :func:`context_limit`) when the question is whether a prompt fits at all.
    context_chars: int | None = None
    #: The model's real context window, in tokens, when the id above publishes one.
    #: This is what decides whether a one-shot prompt (a repair, a benchmark run)
    #: can be sent at all. None means unknown: callers must not guess a number and
    #: must let the endpoint's 413 be the answer.
    context_window: int | None = None


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        id="groq",
        label="Groq (free tier, fast)",
        base_url="https://api.groq.com/openai/v1",
        # Groq retired llama-3.3-70b-versatile and llama-3.1-8b-instant on
        # 2026-08-16; these are its production chat models.
        model="openai/gpt-oss-120b",
        models=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        env="GROQ_API_KEY",
        free="Free tier: about 1,000 requests a day, 8k tokens a minute.",
        signup="https://console.groq.com/keys",
        context_window=131_072,
    ),
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic (Claude)",
        base_url="https://api.anthropic.com",
        model="claude-opus-5-5",
        models=["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
        env="ANTHROPIC_API_KEY",
        signup="https://console.anthropic.com/settings/keys",
        note="Pay as you go. In the browser use a key created for a single workspace: keys that span "
             "several need a workspace id header the page does not ask for.",
        api="anthropic",
        # Roughly 50k tokens: plenty for a tool loop, a fraction of the window,
        # and old tool results stop being trimmed to 400 characters. A trimming
        # budget, so it is deliberately smaller than the 200k-token window below.
        context_chars=200_000,
        context_window=200_000,
    ),
    "huggingface": Provider(
        id="huggingface",
        label="Hugging Face (free with an account)",
        base_url="https://router.huggingface.co/v1",
        model="Qwen/Qwen2.5-72B-Instruct",
        models=["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
        env="HF_TOKEN",
        free="Free accounts get a small monthly credit on Inference Providers.",
        signup="https://huggingface.co/settings/tokens",
        context_window=131_072,
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        models=["gpt-4o-mini", "gpt-4o"],
        env="OPENAI_API_KEY",
        signup="https://platform.openai.com/api-keys",
        context_window=128_000,
    ),
    "mistral": Provider(
        id="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        model="mistral-small-latest",
        models=["mistral-small-latest", "mistral-large-latest"],
        env="MISTRAL_API_KEY",
        signup="https://console.mistral.ai/api-keys/",
        context_window=128_000,
    ),
    "openrouter": Provider(
        id="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        models=["openai/gpt-4o-mini", "google/gemma-3-27b-it"],
        env="OPENROUTER_API_KEY",
        free="Models whose id ends in `:free` cost nothing, with a small daily cap.",
        signup="https://openrouter.ai/keys",
        context_window=128_000,
    ),
    "ollama": Provider(
        id="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        models=["qwen2.5:7b", "llama3.2", "mistral"],
        env=None,
        free="Runs on your own machine, so nothing leaves it.",
        browser=False,
        note="Needs `ollama serve` on this machine; a page served over HTTPS cannot reach it.",
        context_window=32_768,
    ),
    "nvidia": Provider(
        id="nvidia",
        label="NVIDIA Build (free trial credits)",
        base_url="https://integrate.api.nvidia.com/v1",
        model="openai/gpt-oss-120b",
        models=["openai/gpt-oss-120b", "nvidia/llama-3.1-nemotron-70b-instruct", "openai/gpt-oss-20b"],
        env="NVIDIA_API_KEY",
        free="1,000 API calls on signup, more on request; not a refilling quota.",
        signup="https://build.nvidia.com",
        browser=False,
        note="NVIDIA Build endpoints do not support browser CORS; use from the CLI or behind a proxy.",
        context_window=131_072,
    ),
}

#: The order the CLI scans the environment in when no provider was named.
ENV_SCAN_ORDER = ("anthropic", "openai", "groq", "nvidia", "huggingface", "mistral", "openrouter")

#: USD per million tokens (input, output) by model id: the one price table every ledger reads (the Studio, the
#: gym, the showcase). Anthropic's first-party list prices as of October 2026, current models first and then the
#: older ids that recorded results were run on, so those keep their cost; Groq's production models cost nothing
#: on its free tier. Claude Haiku 5.5 is listed at its rate for prompts up to 100k tokens ($0.50 / $2.50 beyond).
#: Cache and batch discounts are not modelled here. A model that is not listed is counted in tokens only:
#: :func:`usd_for` returns None for it and no spend ceiling can be enforced on it.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "openai/gpt-oss-120b": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
}


def price_for(model: str | None) -> tuple[float, float] | None:
    """The (input, output) USD per million tokens of a model id, or None when the table does not price it."""
    return PRICES.get(str(model or "").strip())


def usd_for(prompt_tokens: int, completion_tokens: int, model: str | None) -> float | None:
    """The cost of one call (or a ledger's worth of tokens) at the model's rate; None for an unpriced model."""
    rate = price_for(model)
    if rate is None:
        return None
    return round(int(prompt_tokens or 0) * rate[0] / 1e6 + int(completion_tokens or 0) * rate[1] / 1e6, 6)


def provider_ids(*, browser_only: bool = False) -> list[str]:
    return [p.id for p in PROVIDERS.values() if not browser_only or p.browser]


def context_limit(provider: str | None) -> int | None:
    """The model's context window in characters, or None when it is unknown.

    Deliberately not :func:`aquascope.ai_engine.analyst.context_budget`, which is
    how much conversation the Analyst *keeps* (a trimming budget sized for what a
    free tier meters per minute). This is how much the model can *take*: a 34k-
    character BOM repair fits an OpenAI, Groq or Hugging Face window and was only
    refused because the trimming budget was 24k characters. None means the window
    is not known here, so a caller must not guess: send the request and let the
    endpoint's 413 be the answer.
    """
    spec = PROVIDERS.get(provider or "")
    if spec is None or not spec.context_window:
        return None
    return int(spec.context_window * CHARS_PER_TOKEN)


def default_model(provider: str) -> str | None:
    p = PROVIDERS.get(provider)
    return p.model if p else None


def env_var(provider: str) -> str | None:
    p = PROVIDERS.get(provider)
    return p.env if p else None


def as_json(*, browser_only: bool = True) -> str:
    """The registry as JSON for the Explorer (browser-reachable providers plus `custom`)."""
    out = {
        "generated_by": "python -m aquascope.ai_engine.providers",
        "providers": [
            {k: v for k, v in asdict(p).items() if k != "env"}
            for p in PROVIDERS.values()
            if not browser_only or p.browser
        ],
    }
    out["providers"].append({
        "id": "custom", "label": "Custom OpenAI-compatible endpoint", "base_url": "", "model": "",
        "models": [], "free": None, "signup": None, "browser": True,
        "note": "Any endpoint that speaks /chat/completions with tool calling.",
        "api": "openai", "context_chars": None, "context_window": None,
    })
    return json.dumps(out, indent=2) + "\n"


def write_json(path: str | Path | None = None) -> Path:
    target = Path(path) if path else Path(__file__).resolve().parents[2] / "explorer" / "providers.json"
    # newline="\n": the committed file is LF everywhere, and Windows' default
    # translation would rewrite every line and hide the real change.
    target.write_text(as_json(), encoding="utf-8", newline="\n")
    return target


if __name__ == "__main__":  # pragma: no cover - a tiny maintenance command
    print(f"wrote {write_json()}")
