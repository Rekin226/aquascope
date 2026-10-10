"""Prompt caching, per-role effort, server-side fallback and the call log, on the Claude path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aquascope.ai_engine import llm_transport as transport
from aquascope.ai_engine import team
from aquascope.ai_engine.llm_transport import AnthropicChatClient, UrllibChatClient, with_options
from aquascope.ai_engine.providers import usd_for


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _reply(text="ok", *, model="claude-opus-5-5", usage=None, stop="end_turn"):
    return {"id": "msg_1", "model": model, "stop_reason": stop, "content": [{"type": "text", "text": text}],
            "usage": usage or {"input_tokens": 12, "output_tokens": 3}}


def _capture(replies):
    bodies, headers = [], []

    def urlopen(req, timeout=0):
        bodies.append(json.loads(req.data))
        headers.append({k.lower(): v for k, v in req.header_items()})
        return _Resp(replies.pop(0) if replies else _reply())

    return urlopen, bodies, headers


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("AQUASCOPE_LLM_EFFORT", "AQUASCOPE_LLM_LOG", "AQUASCOPE_LLM_FALLBACKS"):
        monkeypatch.delenv(name, raising=False)


# ── caching ──────────────────────────────────────────────────────────────────


def test_a_one_off_call_caches_its_system_prompt_but_not_its_unique_tail():
    urlopen, bodies, _ = _capture([])
    client = AnthropicChatClient("k")
    with patch("urllib.request.urlopen", urlopen):
        client.chat.completions.create(model="claude-sonnet-5-5", messages=[
            {"role": "system", "content": "You are the Critic."}, {"role": "user", "content": "{...}"}])
    body = bodies[0]
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body, "a unique tail would only pay the write premium"


def test_cache_tokens_are_reported_apart_from_full_price_input():
    usage = {"input_tokens": 40, "output_tokens": 9, "cache_read_input_tokens": 3000,
             "cache_creation_input_tokens": 500}
    urlopen, _, _ = _capture([_reply(usage=usage)])
    with patch("urllib.request.urlopen", urlopen):
        resp = AnthropicChatClient("k").chat.completions.create(
            model="claude-opus-5-5", messages=[{"role": "user", "content": "q"}])
    assert resp.usage.prompt_tokens == 40
    assert resp.usage.cache_read_input_tokens == 3000 and resp.usage.cache_creation_input_tokens == 500


def test_cache_reads_and_writes_are_priced_at_their_own_rates():
    # Claude Opus 5.5: input $4, a read $0.20 (0.05x), a write $5 (1.25x), per million.
    assert usd_for(0, 0, "claude-opus-5-5", cache_read=1_000_000) == pytest.approx(0.20)
    assert usd_for(0, 0, "claude-opus-5-5", cache_write=1_000_000) == pytest.approx(5.0)
    # Everyone else reads at a tenth of the input price.
    assert usd_for(0, 0, "claude-sonnet-5-5", cache_read=1_000_000) == pytest.approx(0.20)
    assert usd_for(1_000_000, 1_000_000, "claude-haiku-5-5") == pytest.approx(0.60)
    assert usd_for(1, 1, "no-such-model", cache_read=5) is None


# ── effort ───────────────────────────────────────────────────────────────────


def test_each_role_asks_for_its_own_effort():
    urlopen, bodies, _ = _capture([])
    client = AnthropicChatClient("k")
    model = team._Model(client, "claude-opus-5-5", "anthropic", {}, [], lambda _e: None)
    with patch("urllib.request.urlopen", urlopen):
        model.call("critic", "sys", {"a": 1})
        model.call("consultant", "sys", {"a": 1})
        model.call("someone_new", "sys", {"a": 1})
    assert bodies[0]["output_config"] == {"effort": "high"}
    assert bodies[1]["output_config"] == {"effort": "low"}
    assert "output_config" not in bodies[2], "an unlisted role keeps the model's default"


def test_the_environment_overrides_every_role(monkeypatch):
    monkeypatch.setenv("AQUASCOPE_LLM_EFFORT", "max")
    urlopen, bodies, _ = _capture([])
    model = team._Model(AnthropicChatClient("k"), "claude-sonnet-5-5", "anthropic", {}, [], lambda _e: None)
    with patch("urllib.request.urlopen", urlopen):
        model.call("consultant", "sys", {"a": 1})
    assert bodies[0]["output_config"] == {"effort": "max"}


def test_models_that_reject_effort_never_get_it():
    assert transport.supports_effort("claude-opus-5-5") and transport.supports_effort("claude-haiku-5-5")
    assert not transport.supports_effort("claude-haiku-4-5") and not transport.supports_effort("gpt-4o-mini")
    urlopen, bodies, _ = _capture([])
    with patch("urllib.request.urlopen", urlopen):
        AnthropicChatClient("k").chat.completions.create(
            model="claude-haiku-4-5", messages=[{"role": "user", "content": "q"}],
            **with_options(AnthropicChatClient("k"), effort="high"))
    assert "output_config" not in bodies[0]


def test_options_never_reach_an_openai_compatible_endpoint_or_a_foreign_client():
    urlopen, bodies, _ = _capture([{"choices": [{"message": {"content": "hi"}}]}])
    client = UrllibChatClient("k", "https://api.groq.com/openai/v1")
    with patch("urllib.request.urlopen", urlopen):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}],
                                       **with_options(client, role="critic", effort="high"))
    assert transport.OPTIONS_KEY not in bodies[0] and "effort" not in bodies[0]
    foreign = SimpleNamespace(chat=None)
    assert with_options(foreign, role="critic", effort="high") == {}


# ── server-side fallback ─────────────────────────────────────────────────────


def test_fallback_is_asked_for_only_where_the_model_supports_it(monkeypatch):
    urlopen, bodies, headers = _capture([])
    client = AnthropicChatClient("k")
    with patch("urllib.request.urlopen", urlopen):
        for model in ("claude-opus-5-5", "claude-haiku-5-5"):
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "q"}])
        monkeypatch.setenv("AQUASCOPE_LLM_FALLBACKS", "0")
        client.chat.completions.create(model="claude-opus-5-5", messages=[{"role": "user", "content": "q"}])
    assert bodies[0]["fallbacks"] == "default" and headers[0]["anthropic-beta"] == transport.FALLBACK_BETA
    assert "fallbacks" not in bodies[1] and "anthropic-beta" not in headers[1]
    assert "fallbacks" not in bodies[2], "AQUASCOPE_LLM_FALLBACKS=0 turns it off"


def test_the_browser_path_waits_long_enough_for_a_long_reply():
    assert AnthropicChatClient("k").timeout == transport.ANTHROPIC_URLLIB_TIMEOUT
    assert AnthropicChatClient("k", timeout=30).timeout == 30


# ── the call log ─────────────────────────────────────────────────────────────


def test_every_call_writes_one_log_row_without_the_prompt(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("AQUASCOPE_LLM_LOG", str(log))
    usage = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 2000}
    urlopen, _, _ = _capture([_reply(usage=usage)])
    model = team._Model(AnthropicChatClient("k"), "claude-opus-5-5", "anthropic", {}, [], lambda _e: None)
    with patch("urllib.request.urlopen", urlopen):
        model.call("interpreter", "SECRET SYSTEM PROMPT", {"basin": "SECRET CONTEXT"})
    (row,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert row["role"] == "interpreter" and row["model"] == "claude-opus-5-5" and row["effort"] == "high"
    assert row["input_tokens"] == 100 and row["output_tokens"] == 20 and row["cache_read_tokens"] == 2000
    assert row["usd"] == pytest.approx(usd_for(100, 20, "claude-opus-5-5", cache_read=2000))
    assert row["finish_reason"] == "stop" and row["error"] is None and row["latency_s"] >= 0
    assert "SECRET" not in log.read_text()


def test_a_failed_call_is_logged_too(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("AQUASCOPE_LLM_LOG", str(log))
    client = UrllibChatClient("k", "https://example.test/v1", sleep=lambda _s: None)

    def urlopen(req, timeout=0):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", urlopen), pytest.raises(transport.LLMConnectionError):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}],
                                       **with_options(client, role="author"))
    (row,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert row["role"] == "author" and "LLMConnectionError" in row["error"]


def test_a_refusal_is_logged_with_its_category(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("AQUASCOPE_LLM_LOG", str(log))
    refusal = {"id": "m", "model": "claude-opus-5-5", "stop_reason": "refusal", "content": [],
               "stop_details": {"type": "refusal", "category": "bio", "explanation": "no"}, "usage": {}}
    urlopen, _, _ = _capture([refusal])
    with patch("urllib.request.urlopen", urlopen):
        AnthropicChatClient("k").chat.completions.create(model="claude-opus-5-5",
                                                         messages=[{"role": "user", "content": "q"}])
    row = json.loads(log.read_text())
    assert row["refusal"] == "bio" and row["finish_reason"] == "content_filter"


# ── the ledgers carry cache tokens ───────────────────────────────────────────


def test_a_role_ledger_counts_cache_tokens_and_prices_them():
    usage = {"input_tokens": 50, "output_tokens": 10, "cache_read_input_tokens": 4000,
             "cache_creation_input_tokens": 0}
    urlopen, _, _ = _capture([_reply(usage=usage)])
    cost: dict = {}
    model = team._Model(AnthropicChatClient("k"), "claude-opus-5-5", "anthropic", cost, [], lambda _e: None)
    with patch("urllib.request.urlopen", urlopen):
        model.call("author", "sys", {"a": 1})
    assert cost["author"]["prompt_tokens"] == 50 and cost["author"]["cache_read_tokens"] == 4000


def test_the_gym_estimate_includes_the_cache():
    from aquascope.gym.bench import estimate_cost

    plain = estimate_cost("claude-sonnet-5-5", 1000, 100)
    cached = estimate_cost("claude-sonnet-5-5", 1000, 100, cache_read=100_000)
    assert cached == pytest.approx(plain + 100_000 * 0.20 / 1e6)
    assert estimate_cost("unknown", 0, 0, cache_read=10) is None
