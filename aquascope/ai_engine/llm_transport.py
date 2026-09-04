"""A dependency-free OpenAI-compatible chat client (``urllib`` only).

Why: the ``openai`` SDK pulls in a compiled JSON parser and cannot be
installed in Pyodide, so the Explorer's browser worker could not run
:func:`aquascope.ai_engine.analyst.ask`. This client speaks the same
``/chat/completions`` protocol (messages, tools, tool_choice) through
``urllib.request``, which ``pyodide_http`` patches into a synchronous XHR
inside web workers, and which works unchanged in CPython. So the very same
tool loop, Data and Methods sections run in the CLI, the MCP server and the
browser. It also means ``aquascope ask`` works without the ``llm`` extra.

The response is wrapped in tiny attribute-access objects mirroring the SDK
shapes the analyst reads: ``response.choices[0].message.content`` and
``.tool_calls[i].id / .function.name / .function.arguments``.

:class:`AnthropicChatClient` gives Claude the same surface: Anthropic's
Messages API is a different protocol (content blocks, ``tool_use`` and
``tool_result``, ``input_schema``), so requests are translated on the way out
and responses on the way back, and the loop never knows. It uses the
``anthropic`` SDK when that is installed and we are not in a browser, and
``urllib`` against ``/v1/messages`` otherwise.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any

from aquascope import __version__


class _Attr:
    """Read-only attribute view over a dict (recursively), for SDK-shaped access."""

    def __init__(self, data: Any):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if isinstance(data, dict):
            return _wrap(data.get(name))  # missing keys read as None, like optional SDK fields
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        return _wrap(data.get(name, default)) if isinstance(data, dict) else default

    def to_dict(self) -> Any:
        return object.__getattribute__(self, "_data")

    def __repr__(self) -> str:
        return f"_Attr({object.__getattribute__(self, '_data')!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return _Attr(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class LLMHTTPError(RuntimeError):
    """The endpoint answered with an error status; ``status`` and ``body`` carry the details."""

    def __init__(self, status: int, body: str, url: str):
        self.status = status
        self.body = body
        self.url = url
        hint = {
            401: "the API key was rejected",
            403: "the API key has no access to this model or endpoint",
            404: "no such endpoint or model",
            429: "rate limit or quota exceeded",
        }.get(status, "the endpoint returned an error")
        super().__init__(f"HTTP {status} from {url}: {hint}. {body[:300]}")


#: Providers state the wait in the error body, in seconds ("try again in 14.5725s")
#: or as a bare number of milliseconds. Prefer what they say over a guess.
_RETRY_HINT = re.compile(r"try again in ([0-9.]+)\s*(ms|s\b|seconds?)", re.I)

MAX_BACKOFF_SECONDS = 30.0


def retry_after(body: str, attempt: int = 0) -> float:
    """How long to wait before retrying, from the provider's own words if it gave them.

    Falls back to exponential backoff (1, 2, 4, 8 s) when it did not, and never
    waits longer than ``MAX_BACKOFF_SECONDS``: a minute-window limit clears in
    seconds, and anything longer is a quota, which waiting will not fix.
    """
    m = _RETRY_HINT.search(body or "")
    if m:
        value = float(m.group(1))
        seconds = value / 1000 if m.group(2).lower() == "ms" else value
        # A hair more than asked: the window is a moving average, and coming
        # back at the exact boundary earns a second 429.
        return min(seconds + 0.5, MAX_BACKOFF_SECONDS)
    return min(2.0 ** attempt, MAX_BACKOFF_SECONDS)


class _Completions:
    def __init__(self, client: UrllibChatClient):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client.request(kwargs)


class _Chat:
    def __init__(self, client: UrllibChatClient):
        self.completions = _Completions(client)


class UrllibChatClient:
    """``client.chat.completions.create(model=, messages=, tools=, tool_choice=)`` over ``urllib``.

    ``base_url`` defaults to OpenAI's; pass any OpenAI-compatible root (Groq,
    Hugging Face router, Mistral, OpenRouter, Ollama, ...). ``extra_headers``
    are sent with every request.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        *,
        timeout: float = 120,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = 4,
        sleep: Any = None,
    ):
        self.api_key = api_key or ""
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = max_retries
        self._sleep = sleep or time.sleep
        self.chat = _Chat(self)

    def request(self, payload: dict[str, Any]) -> Any:
        """One completion, waiting out a rate limit rather than failing on it.

        Free tiers are per minute as much as per day, and a tool-calling loop
        spends a question's whole budget in a few seconds. Providers say how
        long to wait ("Please try again in 14.5725s"), so the honest thing is to
        wait that long and go again, up to ``max_retries``. Only 429 and 5xx are
        retried: a rejected key or a bad request will not improve with time.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._request_once(payload)
            except LLMHTTPError as exc:
                retryable = exc.status == 429 or 500 <= exc.status < 600
                if not retryable or attempt == self.max_retries:
                    raise
                self._sleep(retry_after(exc.body, attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _request_once(self, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"aquascope/{__version__}",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._post_json(url, payload, headers)
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"{url}: {msg}")
        return _wrap(data)

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """POST ``payload`` as JSON and return the decoded body; an error status raises :class:`LLMHTTPError`."""
        import urllib.error
        import urllib.request

        body = json.dumps({k: v for k, v in payload.items() if v is not None}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - caller-chosen https host
                raw = resp.read()
                status = getattr(resp, "status", None) or (resp.getcode() if hasattr(resp, "getcode") else 200)
            if status and int(status) >= 400:  # pyodide-http's urlopen returns error bodies instead of raising
                raise LLMHTTPError(int(status), raw.decode("utf-8", "replace"), url)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                detail = ""
            raise LLMHTTPError(exc.code, detail, url) from None
        return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Anthropic: the Messages API behind the same chat.completions surface
# ---------------------------------------------------------------------------

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 16_000

_FINISH_REASONS = {
    "end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
    "max_tokens": "length", "refusal": "content_filter",
}


def _anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI ``{"type": "function", "function": {...}}`` specs as Messages API tools."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if not fn.get("name"):
            continue
        out.append({
            "name": fn["name"],
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def _anthropic_tool_choice(choice: Any) -> dict[str, Any] | None:
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}.get(
            choice, {"type": "auto"}
        )
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        return {"type": "tool", "name": name} if name else {"type": "auto"}
    return None


def _anthropic_messages(
    messages: list[dict[str, Any]], turns: dict[str, list[dict[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    """Chat-completions messages as (system text, Messages API messages).

    ``tool`` messages become ``tool_result`` blocks, and every result answering
    one assistant turn goes into a single user message, which is what the API
    expects. An assistant turn the model itself produced goes back as the exact
    content blocks it returned (``turns`` remembers them by tool-use id), so its
    thinking blocks travel with it; one we never saw is rebuilt from the text
    and the tool calls.
    """
    system: list[str] = []
    out: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def flush() -> None:
        if results:
            out.append({"role": "user", "content": list(results)})
            results.clear()

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system.append(str(m["content"]))
        elif role == "user":
            flush()
            if m.get("content"):
                out.append({"role": "user", "content": str(m["content"])})
        elif role == "assistant":
            flush()
            calls = [c for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
            cached = next((turns[c["id"]] for c in calls if c.get("id") in turns), None)
            if cached is not None:
                out.append({"role": "assistant", "content": cached})
                continue
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": str(m["content"])})
            for c in calls:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": c.get("id"), "name": fn.get("name"), "input": args})
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            results.append({
                "type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": str(m.get("content") or ""),
            })
    flush()
    return "\n\n".join(system), out


def _from_anthropic(data: dict[str, Any]) -> dict[str, Any]:
    """A Messages API response in the chat-completions shape the analyst reads."""
    blocks = data.get("content") or []
    text = "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text").strip()
    calls = [
        {"id": b.get("id"), "type": "function",
         "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
        for b in blocks if b.get("type") == "tool_use"
    ]
    stop = data.get("stop_reason")
    if stop == "refusal":
        # Said out loud in the answer rather than surfacing as "no answer".
        details = data.get("stop_details") or {}
        text = text or (
            f"The model declined this request ({details.get('category') or 'safety'}): "
            f"{details.get('explanation') or 'no explanation given'}."
        )
        calls = []
    usage = data.get("usage") or {}
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "choices": [{
            "index": 0,
            "finish_reason": _FINISH_REASONS.get(stop, stop),
            "message": {"role": "assistant", "content": text or None, "tool_calls": calls or None},
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        },
    }


class AnthropicChatClient(UrllibChatClient):
    """``client.chat.completions.create(...)`` over Anthropic's Messages API.

    Same surface as :class:`UrllibChatClient`, so the analyst does not know the
    difference. With ``sdk_client`` (an ``anthropic.Anthropic``) the SDK makes
    the call; without one, ``urllib`` posts to ``/v1/messages``, which is what
    the Explorer's worker does, adding the header Anthropic requires before it
    answers a browser page directly. ``effort`` is passed through as
    ``output_config.effort`` when set; ``max_tokens`` is the per-reply ceiling.
    ``workspace_id`` (``wrkspc_...``) is sent as ``anthropic-workspace-id``,
    which identity-linked keys that span several workspaces require.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        *,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        effort: str | None = None,
        workspace_id: str | None = None,
        sdk_client: Any | None = None,
        **kwargs: Any,
    ):
        base = (base_url or "https://api.anthropic.com").rstrip("/")
        if base.endswith("/v1"):  # an OpenAI-style root, given by habit
            base = base[:-3]
        super().__init__(api_key, base, **kwargs)
        if workspace_id:
            self.extra_headers["anthropic-workspace-id"] = workspace_id
        self.workspace_id = workspace_id
        self.max_tokens = max_tokens
        self.effort = effort
        self._sdk = sdk_client
        #: The content blocks of every assistant turn that called a tool, by tool-use id.
        self._turns: dict[str, list[dict[str, Any]]] = {}

    def request(self, payload: dict[str, Any]) -> Any:
        system, messages = _anthropic_messages(payload.get("messages") or [], self._turns)
        body: dict[str, Any] = {
            "model": payload.get("model"),
            "max_tokens": int(payload.get("max_tokens") or self.max_tokens),
            "messages": messages,
        }
        if system:
            body["system"] = system
        tools = _anthropic_tools(payload.get("tools"))
        if tools:
            body["tools"] = tools
            choice = _anthropic_tool_choice(payload.get("tool_choice"))
            if choice:
                body["tool_choice"] = choice
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        data = super().request(body)  # the same 429 / 5xx patience as every other provider
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "tool_use" and b.get("id"):
                self._turns[b["id"]] = list(blocks)
        return _wrap(_from_anthropic(data))

    def _request_once(self, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}/v1/messages"
        if self._sdk is not None:
            return self._sdk_request(payload, url)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"aquascope/{__version__}",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            **self.extra_headers,
        }
        if in_pyodide():
            headers["anthropic-dangerous-direct-browser-access"] = "true"
        return self._post_json(url, payload, headers)

    def _sdk_request(self, payload: dict[str, Any], url: str) -> dict[str, Any]:
        import anthropic

        try:
            response = self._sdk.messages.create(**payload)
        except anthropic.APIStatusError as exc:
            body = exc.body if isinstance(exc.body, str) else json.dumps(exc.body, default=str)
            raise LLMHTTPError(exc.status_code, body or str(exc), url) from None
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"{url}: {exc}") from exc
        return response.model_dump(mode="json", exclude_none=True)


def in_pyodide() -> bool:
    return sys.platform == "emscripten" or "pyodide" in sys.modules


def make_client(api_key: str | None, base_url: str | None, provider: str | None = None) -> Any:
    """The client for ``provider``: an SDK when installed (and we are not in a browser), else ``urllib``.

    ``provider`` is looked up in the registry for its wire protocol; unknown
    or missing means OpenAI-compatible, which every provider but Anthropic is.
    """
    import os

    from aquascope.ai_engine.providers import PROVIDERS

    spec = PROVIDERS.get(provider or "")
    want_sdk = not in_pyodide() and os.environ.get("AQUASCOPE_LLM_TRANSPORT", "").lower() != "urllib"
    if spec is not None and spec.api == "anthropic":
        client = AnthropicChatClient(
            api_key, base_url,
            effort=os.environ.get("AQUASCOPE_LLM_EFFORT") or None,
            workspace_id=os.environ.get("ANTHROPIC_WORKSPACE_ID") or None,
        )
        if want_sdk:
            try:
                import anthropic

                # The loop already waits out 429s, so the SDK's own retries stay off.
                client._sdk = anthropic.Anthropic(
                    api_key=api_key, base_url=client.base_url, max_retries=0,
                    default_headers=dict(client.extra_headers) or None,
                )
            except ImportError:
                pass
        return client
    if want_sdk:
        try:
            from openai import OpenAI

            return OpenAI(api_key=api_key or "none", base_url=base_url)
        except ImportError:
            pass
    return UrllibChatClient(api_key, base_url)


__all__ = ["AnthropicChatClient", "LLMHTTPError", "UrllibChatClient", "in_pyodide", "make_client"]
