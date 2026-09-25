"""A model key a person pastes: which provider it belongs to, whether it works, and remembering it if asked.

The Studio and Ask never read a key they were not given. This module is how a
person gives one without editing their shell profile: the CLI asks, the key is
checked with one tiny request (so a mistyped key is said at once rather than
turning the crew keyless in silence), and it is saved only when the person says
so, to a file only they can read (``~/.config/aquascope/keys.json``, mode 600).
A saved key is put in the environment for the process, under the provider's
usual variable, so every face finds it the same way.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

__all__ = ["check_key", "guess_provider", "keys_path", "load_saved_keys", "save_key"]

#: Key prefixes that name their provider. Order matters: the specific ``sk-`` forms come before OpenAI's.
_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sk-ant-", "anthropic"),
    ("sk-or-", "openrouter"),
    ("gsk_", "groq"),
    ("hf_", "huggingface"),
    ("nvapi-", "nvidia"),
    ("sk-", "openai"),
)


def guess_provider(key: str) -> str | None:
    """The provider a key's prefix names (``sk-ant-`` is Anthropic, ``gsk_`` Groq, ...), or None."""
    k = (key or "").strip()
    return next((p for prefix, p in _PREFIXES if k.startswith(prefix)), None)


def check_key(provider: str, key: str, *, model: str | None = None) -> tuple[bool, str]:
    """``(works, sentence)``: one request of a few tokens with the key. The sentence names the model on
    success, and the provider's refusal (a 401, a quota) on failure, never the key."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client
    from aquascope.ai_engine.providers import PROVIDERS

    try:
        cfg = resolve_llm(provider, model, key.strip(), None)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        client.chat.completions.create(model=cfg["model"], max_tokens=5,
                                       messages=[{"role": "user", "content": "Reply with the word OK."}])
    except Exception as exc:  # noqa: BLE001 - any failure is the answer: the key does not work here
        text = str(exc).replace(key.strip(), "***")
        status = re.search(r"\b(401|403|404|429)\b", text)
        why = {"401": "the key was refused (401)", "403": "the key has no access to this model (403)",
               "404": "the model was not found for this key (404)",
               "429": "the key is out of quota or rate-limited (429)"}.get(status.group(1) if status else "")
        return False, why or f"the check failed: {text[:160]}"
    label = PROVIDERS[provider].label if provider in PROVIDERS else provider
    return True, f"{label} key works (model {cfg['model']})"


def keys_path() -> Path:
    root = os.environ.get("AQUASCOPE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".config", "aquascope")
    return Path(root) / "keys.json"


def save_key(provider: str, key: str) -> Path:
    """Remember a key for this provider in :func:`keys_path`, readable by its owner only."""
    path = keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = _read(path)
    saved[provider] = key.strip()
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return path


def load_saved_keys() -> list[str]:
    """Put each saved key in the environment under its provider's variable, unless that variable is already
    set (the shell wins). Returns the providers loaded."""
    from aquascope.ai_engine.providers import env_var

    loaded = []
    for provider, key in _read(keys_path()).items():
        env = env_var(provider)
        if env and key and not os.environ.get(env):
            os.environ[env] = key
            loaded.append(provider)
    return loaded


def _read(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}
