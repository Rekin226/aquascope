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

__all__ = ["PROVIDERS", "Provider", "as_json", "default_model", "env_var", "provider_ids", "write_json"]


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
    context_chars: int | None = None


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
    ),
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic (Claude)",
        base_url="https://api.anthropic.com",
        model="claude-opus-5",
        models=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
        env="ANTHROPIC_API_KEY",
        signup="https://console.anthropic.com/settings/keys",
        note="Pay as you go. In the browser use a key created for a single workspace: keys that span "
             "several need a workspace id header the page does not ask for.",
        api="anthropic",
        # Roughly 50k tokens: plenty for a tool loop, a fraction of the window,
        # and old tool results stop being trimmed to 400 characters.
        context_chars=200_000,
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
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        models=["gpt-4o-mini", "gpt-4o"],
        env="OPENAI_API_KEY",
        signup="https://platform.openai.com/api-keys",
    ),
    "mistral": Provider(
        id="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        model="mistral-small-latest",
        models=["mistral-small-latest", "mistral-large-latest"],
        env="MISTRAL_API_KEY",
        signup="https://console.mistral.ai/api-keys/",
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
    ),
}

#: The order the CLI scans the environment in when no provider was named.
ENV_SCAN_ORDER = ("anthropic", "openai", "groq", "nvidia", "huggingface", "mistral", "openrouter")


def provider_ids(*, browser_only: bool = False) -> list[str]:
    return [p.id for p in PROVIDERS.values() if not browser_only or p.browser]


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
        "api": "openai", "context_chars": None,
    })
    return json.dumps(out, indent=2) + "\n"


def write_json(path: str | Path | None = None) -> Path:
    target = Path(path) if path else Path(__file__).resolve().parents[2] / "explorer" / "providers.json"
    target.write_text(as_json(), encoding="utf-8")
    return target


if __name__ == "__main__":  # pragma: no cover - a tiny maintenance command
    print(f"wrote {write_json()}")
