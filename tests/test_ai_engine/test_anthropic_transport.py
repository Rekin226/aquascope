"""Claude behind the same chat.completions surface: the Messages API, translated both ways."""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aquascope.ai_engine import analyst
from aquascope.ai_engine import llm_transport as transport
from aquascope.ai_engine.llm_transport import AnthropicChatClient, LLMHTTPError, UrllibChatClient, make_client
from aquascope.ai_engine.providers import ENV_SCAN_ORDER, PROVIDERS, as_json


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_stations",
            "description": "Search the catalog.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


def _tool_use(name="find_stations", args=None, text=None, tool_id="toolu_1"):
    blocks = [{"type": "thinking", "thinking": "", "signature": "sig-1"}]
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.append({"type": "tool_use", "id": tool_id, "name": name, "input": args or {"query": "seine"}})
    return {
        "id": "msg_1",
        "model": "claude-opus-5",
        "stop_reason": "tool_use",
        "content": blocks,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _final(text):
    return {
        "id": "msg_2",
        "model": "claude-opus-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def _headers(req):
    return {k.lower(): v for k, v in req.header_items()}


def test_registry_lists_anthropic_with_its_protocol():
    p = PROVIDERS["anthropic"]
    assert p.api == "anthropic" and p.env == "ANTHROPIC_API_KEY" and p.model == "claude-opus-5"
    assert p.browser and p.context_chars and p.context_chars > analyst.MAX_CONTEXT_CHARS
    assert ENV_SCAN_ORDER[0] == "anthropic"
    assert PROVIDERS["groq"].api == "openai"
    data = json.loads(as_json())
    entry = next(e for e in data["providers"] if e["id"] == "anthropic")
    assert entry["api"] == "anthropic" and "env" not in entry
    assert all("api" in e for e in data["providers"])  # the custom entry too


def test_request_is_translated_to_the_messages_api():
    seen = {}

    def urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["headers"] = _headers(req)
        seen["body"] = json.loads(req.data)
        return _Resp(_tool_use())

    client = AnthropicChatClient("sk-ant-test", "https://api.anthropic.com/v1/")  # an OpenAI-style root, by habit
    messages = [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Seine?"}]
    with patch("urllib.request.urlopen", urlopen):
        resp = client.chat.completions.create(model="claude-opus-5", messages=messages, tools=TOOLS, tool_choice="auto")

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    h = seen["headers"]
    assert h["x-api-key"] == "sk-ant-test" and h["anthropic-version"] == transport.ANTHROPIC_VERSION
    assert "authorization" not in h and "anthropic-dangerous-direct-browser-access" not in h
    body = seen["body"]
    assert body["model"] == "claude-opus-5" and body["system"] == "Be careful."
    assert body["messages"] == [{"role": "user", "content": "Seine?"}]
    assert body["tools"] == [
        {
            "name": "find_stations",
            "description": "Search the catalog.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert body["tool_choice"] == {"type": "auto"}
    assert body["max_tokens"] == transport.ANTHROPIC_DEFAULT_MAX_TOKENS and "output_config" not in body

    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    call = choice.message.tool_calls[0]
    assert call.id == "toolu_1" and call.function.name == "find_stations"
    assert json.loads(call.function.arguments) == {"query": "seine"}
    assert resp.usage.prompt_tokens == 10 and resp.model == "claude-opus-5"


def test_assistant_turns_replay_the_models_own_blocks_and_results_share_one_message():
    bodies = []
    responses = [_tool_use(text="Looking it up."), _final("Done.")]

    def urlopen(req, timeout=0):
        bodies.append(json.loads(req.data))
        return _Resp(responses.pop(0))

    client = AnthropicChatClient("k")
    with patch("urllib.request.urlopen", urlopen):
        first = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], tools=TOOLS)
        msg = first.choices[0].message
        assert msg.content == "Looking it up."
        call = msg.tool_calls[0]
        # What the loop writes down, in chat-completions shape, before the next round.
        history = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "name": "find_stations", "content": '{"n": 1}'},
            {"role": "tool", "tool_call_id": "toolu_other", "name": "other", "content": "{}"},
        ]
        second = client.chat.completions.create(model="m", messages=history, tools=TOOLS)

    sent = bodies[1]["messages"]
    assert sent[0] == {"role": "user", "content": "q"}
    # The assistant turn is the model's own blocks, thinking included, not a reconstruction.
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert sent[1]["content"][-1]["type"] == "tool_use" and sent[1]["content"][-1]["id"] == "toolu_1"
    # Both results ride in one user message, in order.
    assert sent[2]["role"] == "user"
    assert [b["type"] for b in sent[2]["content"]] == ["tool_result", "tool_result"]
    assert sent[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"n": 1}'}
    assert second.choices[0].message.content == "Done." and second.choices[0].message.tool_calls is None
    assert second.choices[0].finish_reason == "stop"


def test_an_unseen_assistant_turn_is_rebuilt_from_text_and_calls():
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [
                {"id": "c9", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                {"id": "c10", "type": "function", "function": {"name": "g", "arguments": "not json"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c9", "content": "r"},
        {"role": "assistant", "content": "", "tool_calls": None},  # nothing to send
    ]
    system, msgs = transport._anthropic_messages(history, turns={})
    assert system == ""
    assert msgs[1]["content"] == [
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "c9", "name": "f", "input": {"a": 1}},
        {"type": "tool_use", "id": "c10", "name": "g", "input": {}},
    ]
    assert msgs[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c9", "content": "r"}]}
    assert len(msgs) == 3


def test_tool_choice_effort_and_refusal():
    assert transport._anthropic_tool_choice("required") == {"type": "any"}
    assert transport._anthropic_tool_choice("none") == {"type": "none"}
    assert transport._anthropic_tool_choice({"type": "function", "function": {"name": "f"}}) == {
        "type": "tool",
        "name": "f",
    }
    assert transport._anthropic_tool_choice(None) is None

    seen = {}

    def urlopen(req, timeout=0):
        seen["body"] = json.loads(req.data)
        return _Resp(
            {
                "id": "m",
                "model": "claude-opus-5",
                "stop_reason": "refusal",
                "content": [],
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": "nope"},
                "usage": {},
            }
        )

    client = AnthropicChatClient("k", effort="medium", max_tokens=2048)
    with patch("urllib.request.urlopen", urlopen):
        resp = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], tools=TOOLS)
    assert seen["body"]["output_config"] == {"effort": "medium"} and seen["body"]["max_tokens"] == 2048
    msg = resp.choices[0].message
    assert msg.tool_calls is None and "declined" in msg.content and "nope" in msg.content
    assert resp.choices[0].finish_reason == "content_filter"


def test_http_errors_are_explained_and_too_long_is_recognised():
    body = json.dumps(
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long: 250000 tokens > 200000 maximum"},
        }
    )

    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "bad request", None, io.BytesIO(body.encode()))

    client = AnthropicChatClient("k")
    with patch("urllib.request.urlopen", urlopen), pytest.raises(LLMHTTPError) as ei:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert ei.value.status == 400 and "too long" in ei.value.body
    assert "v1/messages" in str(ei.value)


def test_browser_header_only_in_pyodide():
    seen = {}

    def urlopen(req, timeout=0):
        seen["headers"] = _headers(req)
        return _Resp(_final("ok"))

    client = AnthropicChatClient("k")
    with patch("urllib.request.urlopen", urlopen), patch.object(transport, "in_pyodide", return_value=True):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert seen["headers"]["anthropic-dangerous-direct-browser-access"] == "true"


def test_make_client_dispatches_on_the_registry(monkeypatch):
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "urllib")
    monkeypatch.delenv("AQUASCOPE_LLM_EFFORT", raising=False)
    c = make_client("k", None, provider="anthropic")
    assert isinstance(c, AnthropicChatClient) and c._sdk is None and c.effort is None
    assert c.base_url == "https://api.anthropic.com"
    c2 = make_client("k", "https://api.groq.com/openai/v1", provider="groq")
    assert isinstance(c2, UrllibChatClient) and not isinstance(c2, AnthropicChatClient)
    assert isinstance(make_client("k", None), UrllibChatClient)  # no provider: OpenAI-compatible, as before
    monkeypatch.setenv("AQUASCOPE_LLM_EFFORT", "low")
    assert make_client("k", "https://api.anthropic.com/v1", provider="anthropic").effort == "low"


def test_make_client_uses_the_sdk_when_installed(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("AQUASCOPE_LLM_TRANSPORT", raising=False)
    c = make_client("sk-ant-x", None, provider="anthropic")
    assert isinstance(c, AnthropicChatClient) and c._sdk is not None
    assert c._sdk.max_retries == 0  # the loop already waits out 429s


def test_sdk_path_calls_messages_create_and_maps_errors():
    anthropic = pytest.importorskip("anthropic")
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(model_dump=lambda **_: _final("from sdk"))

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    client = AnthropicChatClient("k", sdk_client=sdk)
    resp = client.chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        tools=TOOLS,
        tool_choice="auto",
    )
    assert resp.choices[0].message.content == "from sdk"
    assert calls[0]["system"] == "s" and calls[0]["tools"][0]["input_schema"]["type"] == "object"

    import httpx2

    def failing(**kw):
        response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        raise anthropic.APIStatusError("rate limited", response=response, body={"error": {"type": "rate_limit_error"}})

    sleeps = []
    client = AnthropicChatClient(
        "k", sdk_client=SimpleNamespace(messages=SimpleNamespace(create=failing)), max_retries=1, sleep=sleeps.append
    )
    with pytest.raises(LLMHTTPError) as ei:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert ei.value.status == 429 and "rate_limit_error" in ei.value.body
    assert len(sleeps) == 1  # retried once, then gave up, like the OpenAI-compatible client


def test_context_budget_follows_the_registry():
    assert analyst.context_budget("anthropic") == PROVIDERS["anthropic"].context_chars
    assert analyst.context_budget("groq") == analyst.MAX_CONTEXT_CHARS
    assert analyst.context_budget(None) == analyst.MAX_CONTEXT_CHARS


def test_analyst_runs_end_to_end_over_a_scripted_messages_api(monkeypatch):
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "urllib")
    bodies = []
    responses = [_tool_use(name="describe_methods", args={}, tool_id="toolu_dm"), _final("Methods listed.")]

    def urlopen(req, timeout=0):
        bodies.append((req.full_url, _headers(req), json.loads(req.data)))
        return _Resp(responses.pop(0))

    long_result = {"methods": [{"name": "GEV", "text": "x" * 30_000, "citation": "Hosking 1990"}]}
    with (
        patch("urllib.request.urlopen", urlopen),
        patch("aquascope.mcp_server.describe_methods", return_value=long_result),
    ):
        res = analyst.ask("Which methods do you know?", provider="anthropic", api_key="sk-ant-test")

    assert res.provider == "anthropic" and res.model == "claude-opus-5"
    assert [c.name for c in res.tool_calls] == ["describe_methods"] and res.tool_calls[0].ok
    assert res.answer == "Methods listed." and res.steps == 2
    assert "Hosking 1990" in res.to_markdown()

    url, headers, first = bodies[0]
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "sk-ant-test"
    assert first["system"] == analyst.SYSTEM_PROMPT
    assert {t["name"] for t in first["tools"]} >= {"find_stations", "describe_methods", "run_python"}
    second = bodies[1][2]["messages"]
    assert second[1]["content"][0]["type"] == "thinking" and second[1]["content"][-1]["id"] == "toolu_dm"
    result_block = second[2]["content"][0]
    assert result_block["tool_use_id"] == "toolu_dm"
    # The window is large, so the 30k-character result went back whole instead of trimmed to 400.
    assert len(result_block["content"]) > 30_000 and "trimmed" not in result_block["content"]


def test_workspace_id_travels_as_a_header_on_both_paths(monkeypatch):
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "urllib")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01test")
    seen = {}

    def urlopen(req, timeout=0):
        seen["headers"] = _headers(req)
        return _Resp(_final("ok"))

    client = make_client("k", None, provider="anthropic")
    assert client.workspace_id == "wrkspc_01test"
    with patch("urllib.request.urlopen", urlopen):
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    assert seen["headers"]["anthropic-workspace-id"] == "wrkspc_01test"

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    assert make_client("k", None, provider="anthropic").workspace_id is None

    pytest.importorskip("anthropic")
    monkeypatch.delenv("AQUASCOPE_LLM_TRANSPORT")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01test")
    sdk_backed = make_client("sk-ant-x", None, provider="anthropic")
    assert sdk_backed._sdk.default_headers["anthropic-workspace-id"] == "wrkspc_01test"
