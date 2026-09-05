import json
from pathlib import Path

import pytest

from aquascope.ai_engine.analyst import resolve_llm
from aquascope.ai_engine.providers import (
    ENV_SCAN_ORDER,
    PROVIDERS,
    as_json,
    default_model,
    env_var,
    provider_ids,
    write_json,
)


def test_nvidia_provider_registered():
    assert "nvidia" in PROVIDERS
    p = PROVIDERS["nvidia"]
    assert p.id == "nvidia"
    assert p.label == "NVIDIA Build (free trial credits)"
    assert p.base_url == "https://integrate.api.nvidia.com/v1"
    assert p.model == "openai/gpt-oss-120b"
    assert "openai/gpt-oss-120b" in p.models
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in p.models
    assert p.env == "NVIDIA_API_KEY"
    assert p.signup == "https://build.nvidia.com"
    assert p.browser is False
    assert p.free is not None
    assert "1,000 API calls" in p.free
    assert p.note is not None
    assert "CORS" in p.note


def test_env_scan_order_includes_nvidia():
    assert "nvidia" in ENV_SCAN_ORDER
    assert ENV_SCAN_ORDER.index("nvidia") > ENV_SCAN_ORDER.index("groq")


def test_provider_ids():
    all_ids = provider_ids(browser_only=False)
    browser_ids = provider_ids(browser_only=True)
    assert "nvidia" in all_ids
    assert "nvidia" not in browser_ids
    assert "ollama" not in browser_ids
    assert "groq" in browser_ids


def test_default_model_and_env_var():
    assert default_model("nvidia") == "openai/gpt-oss-120b"
    assert env_var("nvidia") == "NVIDIA_API_KEY"
    assert default_model("unknown_provider") is None
    assert env_var("unknown_provider") is None


def test_as_json_roundtrip():
    raw_browser = as_json(browser_only=True)
    data_browser = json.loads(raw_browser)
    ids_browser = [p["id"] for p in data_browser["providers"]]
    assert "nvidia" not in ids_browser
    assert "ollama" not in ids_browser
    assert "groq" in ids_browser
    assert "custom" in ids_browser
    for p in data_browser["providers"]:
        assert "env" not in p

    raw_all = as_json(browser_only=False)
    data_all = json.loads(raw_all)
    ids_all = [p["id"] for p in data_all["providers"]]
    assert "nvidia" in ids_all
    assert "ollama" in ids_all
    assert "groq" in ids_all

    nvidia_entry = next(p for p in data_all["providers"] if p["id"] == "nvidia")
    assert nvidia_entry["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert nvidia_entry["model"] == "openai/gpt-oss-120b"
    assert nvidia_entry["browser"] is False
    assert "env" not in nvidia_entry


def test_write_json_custom_path(tmp_path: Path):
    target = tmp_path / "providers.json"
    written = write_json(target)
    assert written == target
    assert target.is_file()
    content = json.loads(target.read_text(encoding="utf-8"))
    assert "providers" in content
    ids = [p["id"] for p in content["providers"]]
    assert "nvidia" not in ids


def test_resolve_llm_nvidia_explicit():
    cfg = resolve_llm(provider="nvidia", api_key="nv-test-key")
    assert cfg["provider"] == "nvidia"
    assert cfg["api_key"] == "nv-test-key"
    assert cfg["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert cfg["model"] == "openai/gpt-oss-120b"


def test_resolve_llm_nvidia_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("AQUASCOPE_LLM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-env-key-1234")

    cfg = resolve_llm()
    assert cfg["provider"] == "nvidia"
    assert cfg["api_key"] == "nv-env-key-1234"
    assert cfg["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert cfg["model"] == "openai/gpt-oss-120b"
