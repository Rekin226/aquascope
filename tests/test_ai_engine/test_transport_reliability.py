"""The transport's error policy: timeouts, both SDKs' errors, truncated replies, the provider, the price table."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aquascope.ai_engine import analyst, team
from aquascope.ai_engine import llm_transport as transport
from aquascope.ai_engine.llm_transport import (
    AnthropicChatClient,
    LLMConnectionError,
    LLMHTTPError,
    OpenAISDKChatClient,
    UrllibChatClient,
)
from aquascope.ai_engine.providers import PRICES, PROVIDERS


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── timeouts and dropped connections ─────────────────────────────────────────


def test_a_timeout_is_a_connection_error_and_is_retried_a_bounded_number_of_times():
    slept = []
    client = UrllibChatClient("k", "https://example.test/v1", sleep=slept.append)
    calls = {"n": 0}

    def urlopen(req, timeout=0):
        calls["n"] += 1
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", urlopen), pytest.raises(LLMConnectionError, match="timed out"):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert calls["n"] == 1 + transport.MAX_CONNECTION_RETRIES, "two more tries, not the four a 429 gets"
    assert slept == [2.0, 4.0]


def test_a_dropped_connection_that_recovers_returns_the_answer():
    client = UrllibChatClient("k", "https://example.test/v1", sleep=lambda _s: None)
    replies = [urllib.error.URLError("connection reset"), {"choices": [{"message": {"content": "ok"}}]}]

    def urlopen(req, timeout=0):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)

    with patch("urllib.request.urlopen", urlopen):
        out = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert out.choices[0].message.content == "ok"


def test_an_http_error_is_still_an_http_error_not_a_connection_error():
    client = UrllibChatClient("k", "https://example.test/v1", sleep=lambda _s: None)

    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)

    with patch("urllib.request.urlopen", urlopen), pytest.raises(LLMHTTPError) as ei:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert ei.value.status == 401


def test_the_anthropic_sdks_timeout_is_a_connection_error():
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    def create(**kw):
        raise anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))

    client = AnthropicChatClient("k", sdk_client=SimpleNamespace(messages=SimpleNamespace(create=create)),
                                 sleep=lambda _s: None)
    with pytest.raises(LLMConnectionError):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])


# ── the openai SDK behind the same error policy ──────────────────────────────


def test_the_openai_sdks_errors_reach_the_loop_as_ours():
    openai = pytest.importorskip("openai")
    import httpx

    def failing(**kw):
        response = httpx.Response(413, request=httpx.Request("POST", "https://api.groq.com/openai/v1/x"))
        raise openai.APIStatusError("too large", response=response, body={"error": {"message": "Request too large"}})

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=failing)))
    client = OpenAISDKChatClient("k", "https://api.groq.com/openai/v1", sdk_client=sdk)
    with pytest.raises(LLMHTTPError) as ei:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert ei.value.status == 413 and "too large" in ei.value.body.lower()


def test_the_openai_sdk_answer_reads_like_the_urllib_one():
    pytest.importorskip("openai")
    dump = {"choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    seen = []

    def create(**kw):
        seen.append(kw)
        return SimpleNamespace(model_dump=lambda **_: dump)

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = OpenAISDKChatClient("k", None, sdk_client=sdk).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "q"}], tools=None)
    assert out.choices[0].message.content == "hi" and out.usage.prompt_tokens == 3
    assert "tools" not in seen[0], "None arguments are not sent"


def test_make_client_wraps_the_openai_sdk_with_its_retries_off(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("AQUASCOPE_LLM_TRANSPORT", raising=False)
    c = transport.make_client("k", "https://api.groq.com/openai/v1", provider="groq")
    assert isinstance(c, OpenAISDKChatClient) and c._sdk.max_retries == 0


# ── replies cut off at the output limit ──────────────────────────────────────


def _reply(content, *, finish, calls=None):
    msg = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_a_role_whose_reply_was_cut_off_runs_keyless_and_says_why():
    reply = _reply('{"plan": [{"tool": "analyze_station", "ar', finish="length")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    events = []
    m = team._Model(client, "m", "anthropic", {}, [], events.append)
    assert m.call("methodologist", "sys", {"q": 1}) is None
    assert events[-1]["event"] == "model_truncated"


def test_an_answer_cut_off_mid_tool_call_does_not_run_the_call():
    call = SimpleNamespace(id="t1", function=SimpleNamespace(name="describe_methods", arguments='{"x": '))
    reply = _reply("Let me look", finish="length", calls=[call])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    res = analyst.ask("what methods?", provider="groq", model="m", client=client, verify_answer=False)
    assert res.tool_calls == []
    assert "cut off at its output limit" in res.answer and "not run" in res.answer


# ── the provider reaches the client in every caller ──────────────────────────


def test_repair_builds_the_client_for_the_resolved_provider(monkeypatch):
    from aquascope.maintenance import repair

    seen = {}

    def fake_make_client(api_key, base_url, provider=None):
        seen["provider"] = provider
        reply = _reply('{"action": "no_fix", "explanation": "x", "confidence": 0.1}', finish="stop")
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))

    monkeypatch.setattr(transport, "make_client", fake_make_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    evidence = SimpleNamespace(to_prompt=lambda: "evidence")
    repair.propose_repair(evidence, provider="anthropic")
    assert seen["provider"] == "anthropic"


def test_the_no_key_message_names_the_anthropic_key(monkeypatch):
    for p in PROVIDERS.values():
        if p.env:
            monkeypatch.delenv(p.env, raising=False)
    monkeypatch.delenv("AQUASCOPE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AQUASCOPE_LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        analyst.resolve_llm()


# ── one price table ──────────────────────────────────────────────────────────


def test_every_claude_model_the_picker_offers_is_priced():
    for model in PROVIDERS["anthropic"].models:
        assert model in PRICES, model
    assert PROVIDERS["anthropic"].model == PROVIDERS["anthropic"].models[0]


def test_the_gym_and_the_showcase_read_the_same_prices():
    from aquascope.gym.bench import PRICES_USD_PER_MTOK
    from aquascope.studio import showcase

    for model, rate in PRICES_USD_PER_MTOK.items():
        assert PRICES[model] == rate
    assert showcase.PRICES["claude-sonnet-5"] == PRICES["claude-sonnet-5"] == (2.0, 10.0)
    assert "claude-opus-5" in PRICES_USD_PER_MTOK, "older ids keep their price, so recorded runs keep their cost"
