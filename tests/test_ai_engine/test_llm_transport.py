"""The urllib chat client: same protocol as the OpenAI SDK, no dependencies, runs in Pyodide."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from aquascope.ai_engine import analyst
from aquascope.ai_engine import llm_transport as transport
from aquascope.ai_engine.llm_transport import LLMHTTPError, UrllibChatClient, make_client


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_posts_openai_shape_and_wraps_response():
    seen = {}

    def urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = json.loads(req.data)
        return _Resp({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "find_stations", "arguments": '{"query": "seine"}'}}
        ]}}]})

    client = UrllibChatClient("sk-test", "https://api.groq.com/openai/v1/")
    with patch("urllib.request.urlopen", urlopen):
        resp = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}],
                                              tools=[{"type": "function"}], tool_choice="auto")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["body"]["model"] == "m" and seen["body"]["tool_choice"] == "auto"
    msg = resp.choices[0].message
    assert msg.content is None and msg.tool_calls[0].function.name == "find_stations"
    assert msg.tool_calls[0].id == "c1" and json.loads(msg.tool_calls[0].function.arguments) == {"query": "seine"}
    assert msg.missing_field is None  # optional SDK fields read as None


def test_http_errors_are_explained():
    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", None, io.BytesIO(b'{"error":"bad key"}'))

    client = UrllibChatClient("bad", "https://api.openai.com/v1")
    with patch("urllib.request.urlopen", urlopen), pytest.raises(LLMHTTPError) as ei:
        client.chat.completions.create(model="m", messages=[])
    assert ei.value.status == 401 and "rejected" in str(ei.value) and "bad key" in str(ei.value)


def test_ask_runs_the_full_loop_over_urllib(monkeypatch):
    """ask() with the urllib transport: two model turns (one tool call, then the answer)."""
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "urllib")
    turns = iter([
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "describe_methods", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"content": "GEV and LP3 are used, see methods."}}]},
    ])
    posted = []

    def urlopen(req, timeout=0):
        posted.append(json.loads(req.data))
        return _Resp(next(turns))

    with patch("urllib.request.urlopen", urlopen):
        res = analyst.ask("what methods?", provider="groq", api_key="k", model="llama")
    assert res.answer.startswith("GEV") and res.provider == "groq" and res.model == "llama"
    assert [c.name for c in res.tool_calls] == ["describe_methods"] and res.tool_calls[0].ok
    assert posted[1]["messages"][-1]["role"] == "tool" and posted[1]["messages"][-1]["tool_call_id"] == "t1"
    md = res.to_markdown()
    assert md.startswith("# what methods?") and "Tools called: describe_methods" in md


def test_make_client_falls_back_to_urllib_when_forced(monkeypatch):
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "urllib")
    assert isinstance(make_client("k", None), UrllibChatClient)
    monkeypatch.setenv("AQUASCOPE_LLM_TRANSPORT", "")
    with patch.dict("sys.modules", {"openai": None}):  # openai not installed
        assert isinstance(make_client("k", "http://x/v1"), UrllibChatClient)


def test_catalog_override_feeds_find_stations():
    from aquascope import mcp_server
    from aquascope.archive import catalog

    rows = [{"source": "usgs", "station_id": "USGS-1", "name": "Potomac River", "latitude": 38.9, "longitude": -77.1,
             "variables": ["discharge"]}]
    catalog.set_catalog(rows)
    try:
        out = mcp_server.find_stations(query="potomac")
        assert out["n_catalog"] == 1 and out["stations"][0]["station_id"] == "USGS-1"
    finally:
        catalog.set_catalog(None)


# ── waiting out a rate limit (#233) ─────────────────────────────────────────
#
# The first live showcase recording got 1 of 8: the free tier's per-minute token
# budget went on the first question and every later request came back 429, each
# one telling us exactly how long to wait ("Please try again in 14.5725s") while
# nothing waited. These cover the reading of that hint and the retry around it.

GROQ_429 = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in organization '
    '`org_x` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7945, '
    'Requested 1490. Please try again in 10.7625s. Need more tokens? Upgrade to Dev Tier"}}'
)


def test_the_wait_comes_from_what_the_provider_said() -> None:
    assert transport.retry_after(GROQ_429) == pytest.approx(11.2625)


def test_a_millisecond_hint_is_not_read_as_seconds() -> None:
    assert transport.retry_after("Please try again in 412.5ms.") == pytest.approx(0.9125)


def test_without_a_hint_it_backs_off() -> None:
    waits = [transport.retry_after("no idea", attempt) for attempt in range(4)]
    assert waits == [1.0, 2.0, 4.0, 8.0]


def test_a_wait_longer_than_the_cap_is_not_honoured() -> None:
    """An hour is a quota, not a window; waiting it out in CI helps nobody."""
    assert transport.retry_after("try again in 3600s") == transport.MAX_BACKOFF_SECONDS


def _client_that_fails(statuses, *, body=GROQ_429):
    """A client whose one HTTP call yields the given statuses in turn, then succeeds."""
    slept = []
    client = transport.UrllibChatClient("k", "https://example.test/v1", sleep=slept.append)
    calls = {"n": 0}

    def fake_once(_payload):
        i = calls["n"]
        calls["n"] += 1
        if i < len(statuses):
            raise transport.LLMHTTPError(statuses[i], body, "https://example.test/v1/chat/completions")
        return {"ok": True, "attempts": calls["n"]}

    client._request_once = fake_once
    return client, slept, calls


def test_a_rate_limited_request_is_retried_after_the_stated_wait() -> None:
    client, slept, _ = _client_that_fails([429, 429])
    out = client.request({"model": "m", "messages": []})
    assert out["ok"] and out["attempts"] == 3
    assert slept == [pytest.approx(11.2625), pytest.approx(11.2625)], "it waits what it was told, twice"


def test_a_rejected_key_is_not_retried() -> None:
    """401 will not improve with time, and retrying it just makes the failure slower."""
    client, slept, calls = _client_that_fails([401, 401, 401, 401, 401, 401])
    with pytest.raises(transport.LLMHTTPError):
        client.request({"model": "m", "messages": []})
    assert calls["n"] == 1 and slept == []


def test_a_server_error_is_retried() -> None:
    client, slept, _ = _client_that_fails([503], body="upstream hiccup")
    assert client.request({"model": "m", "messages": []})["ok"]
    assert slept == [1.0]


def test_retries_are_bounded_and_the_last_failure_is_raised() -> None:
    client, slept, calls = _client_that_fails([429] * 99)
    with pytest.raises(transport.LLMHTTPError, match="429"):
        client.request({"model": "m", "messages": []})
    assert calls["n"] == 5, "one try plus four retries, then it gives up rather than hanging"
    assert len(slept) == 4
