"""A pasted model key: its provider from the prefix, a one-request check, and remembering it only if asked."""

from __future__ import annotations

import os
import stat

import pytest

from aquascope.ai_engine import keys


def test_the_prefix_names_the_provider():
    assert keys.guess_provider(" sk-ant-api03-x ") == "anthropic"
    assert keys.guess_provider("sk-or-v1-x") == "openrouter" and keys.guess_provider("sk-proj-x") == "openai"
    assert keys.guess_provider("gsk_x") == "groq" and keys.guess_provider("hf_x") == "huggingface"
    assert keys.guess_provider("nvapi-x") == "nvidia" and keys.guess_provider("abc") is None


def test_a_saved_key_is_private_and_the_shell_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AQUASCOPE_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    path = keys.save_key("groq", " gsk_saved ")
    keys.save_key("anthropic", "sk-ant-saved")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert keys.load_saved_keys() == ["groq"]
    assert os.environ["GROQ_API_KEY"] == "gsk_saved" and os.environ["ANTHROPIC_API_KEY"] == "from-the-shell"


def test_the_check_says_why_without_the_key(monkeypatch):
    import aquascope.ai_engine.llm_transport as transport

    class Refusing:
        class chat:  # noqa: N801 - the client's own shape
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise RuntimeError("https://api.groq.com: HTTP 401 invalid key gsk_secret")

    monkeypatch.setattr(transport, "make_client", lambda *a, **k: Refusing())
    works, said = keys.check_key("groq", "gsk_secret")
    assert not works and said == "the key was refused (401)"

    class Fine:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    assert kw["max_tokens"] == 5
                    return {}

    monkeypatch.setattr(transport, "make_client", lambda *a, **k: Fine())
    works, said = keys.check_key("groq", "gsk_secret")
    assert works and said.startswith("Groq") and "gsk_secret" not in said


@pytest.mark.parametrize("picks, pasted, works, want", [
    (["1"], [], True, None),                              # no key: keyless, nothing asked
    (["2", "1"], ["gsk_ok"], True, "groq"),               # paste, it works, don't remember
    (["2", "2"], ["gsk_bad"], False, None),               # paste, refused, run keyless
])
def test_the_studio_asks_for_a_key_and_checks_it(monkeypatch, tmp_path, picks, pasted, works, want):
    import argparse
    import builtins
    import getpass
    import sys

    from aquascope import cli

    monkeypatch.setitem(sys.modules, "questionary", None)
    monkeypatch.setenv("AQUASCOPE_CONFIG_DIR", str(tmp_path))
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "HF_TOKEN",
                "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    it, pastes = iter(picks), iter(pasted)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": next(pastes))
    monkeypatch.setattr(keys, "check_key", lambda p, k, model=None: (works, "Groq key works" if works else "refused"))
    args = argparse.Namespace(provider=None, model=None, api_key=None, base_url=None, max_usd=None)
    assert cli._studio_key_step(args)
    assert args.provider == want and (args.api_key == "gsk_ok") == (want == "groq")
    assert not keys.keys_path().exists(), "a key is saved only when the person asks"
