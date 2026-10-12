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
import os
import re
import sys
import time
from datetime import datetime, timezone
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


class LLMConnectionError(RuntimeError):
    """The request never got an answer: a timeout, a refused or dropped connection, a DNS failure.

    Worth one or two more tries (a long reply that timed out once often lands the second time), but not
    ``max_retries`` of them: a run that times out four times in a row spends half an hour saying nothing.
    """

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"{url}: no answer ({reason})")


#: How many times a timeout or dropped connection is tried again, separately from the 429/5xx budget.
MAX_CONNECTION_RETRIES = 2


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


#: The keyword a caller passes to ``chat.completions.create`` for aquascope's own options, which never go on the
#: wire: ``{"role": "critic", "effort": "high"}``. ``role`` names the call in the telemetry; ``effort`` is how hard
#: a Claude model should think (``low`` to ``max``) and is ignored by OpenAI-compatible providers. Only pass it to
#: this module's clients: an SDK client of a caller's own would reject the keyword.
OPTIONS_KEY = "aquascope"

#: Set this to a file path and every model call appends one JSON line to it: role, model, effort, tokens (uncached
#: input, output, cache read, cache write), latency, how the reply ended, a refusal's category, the error if any,
#: and the USD estimate. Prompts and replies are never written.
LOG_ENV = "AQUASCOPE_LLM_LOG"


def accepts_options(client: Any) -> bool:
    """Whether ``client`` is one of this module's clients, which take the :data:`OPTIONS_KEY` keyword."""
    return isinstance(client, UrllibChatClient)


def with_options(client: Any, **options: Any) -> dict[str, Any]:
    """The extra ``create`` keyword for ``client``: ``{OPTIONS_KEY: options}`` for ours, nothing for anyone else's."""
    opts = {k: v for k, v in options.items() if v is not None}
    return {OPTIONS_KEY: opts} if opts and accepts_options(client) else {}


def _log_call(row: dict[str, Any] | None) -> None:
    path = os.environ.get(LOG_ENV)
    if not path or row is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:  # telemetry never breaks a call
        pass


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
        """One completion: aquascope's options taken off the payload, the call made, one telemetry row written."""
        payload = dict(payload)
        opts = dict(payload.pop(OPTIONS_KEY, None) or {})
        started = time.monotonic()
        try:
            response = self._call(payload, opts)
        except Exception as exc:
            if os.environ.get(LOG_ENV):
                _log_call(self._log_row(payload, opts, started, None, exc))
            raise
        if os.environ.get(LOG_ENV):
            _log_call(self._log_row(payload, opts, started, response, None))
        return response

    def _call(self, payload: dict[str, Any], opts: dict[str, Any]) -> Any:
        """The call itself; OpenAI-compatible endpoints take no effort setting, so the options stop here."""
        return self._with_retries(payload)

    def _log_row(self, payload: dict[str, Any], opts: dict[str, Any], started: float, response: Any,
                 error: BaseException | None) -> dict[str, Any]:
        from aquascope.ai_engine.providers import usd_for

        data = response.to_dict() if isinstance(response, _Attr) else {}
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0] or {}

        def n(key: str) -> int:
            v = usage.get(key)
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

        tokens = {
            "input_tokens": n("prompt_tokens"), "output_tokens": n("completion_tokens"),
            "cache_read_tokens": n("cache_read_input_tokens"), "cache_write_tokens": n("cache_creation_input_tokens"),
        }
        model = payload.get("model")
        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "role": opts.get("role"),
            "endpoint": self.base_url,
            "model": model,
            "served_model": data.get("model"),
            "effort": opts.get("effort_sent"),
            **tokens,
            "latency_s": round(time.monotonic() - started, 2),
            "finish_reason": choice.get("finish_reason"),
            "refusal": (data.get("stop_details") or {}).get("category") if data.get("stop_details") else None,
            "error": f"{type(error).__name__}: {error}"[:300] if error is not None else None,
            "usd": usd_for(tokens["input_tokens"], tokens["output_tokens"], model,
                           cache_read=tokens["cache_read_tokens"], cache_write=tokens["cache_write_tokens"]),
        }

    def _with_retries(self, payload: dict[str, Any]) -> Any:
        """One completion, waiting out a rate limit rather than failing on it.

        Free tiers are per minute as much as per day, and a tool-calling loop
        spends a question's whole budget in a few seconds. Providers say how
        long to wait ("Please try again in 14.5725s"), so the honest thing is to
        wait that long and go again, up to ``max_retries``. Only 429 and 5xx are
        retried: a rejected key or a bad request will not improve with time. A
        timeout or a dropped connection is tried again too, but at most
        ``MAX_CONNECTION_RETRIES`` times, so a role that cannot be reached fails
        in minutes and says so instead of silently never running.
        """
        attempt = dropped = 0
        while True:
            try:
                return self._request_once(payload)
            except LLMHTTPError as exc:
                retryable = exc.status == 429 or 500 <= exc.status < 600
                if not retryable or attempt >= self.max_retries:
                    raise
                self._sleep(retry_after(exc.body, attempt))
                attempt += 1
            except LLMConnectionError:
                if dropped >= min(MAX_CONNECTION_RETRIES, self.max_retries):
                    raise
                dropped += 1
                self._sleep(min(2.0 ** dropped, MAX_BACKOFF_SECONDS))

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
        except OSError as exc:  # URLError, socket timeouts, resets: the request got no answer at all
            reason = getattr(exc, "reason", None) or exc
            raise LLMConnectionError(url, f"{type(exc).__name__}: {reason}") from None
        return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Anthropic: the Messages API behind the same chat.completions surface
# ---------------------------------------------------------------------------

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 16_000
#: A browser cannot stream a synchronous request, so a long reply needs a long wait; the SDK streams instead.
ANTHROPIC_URLLIB_TIMEOUT = 600

#: The models that take ``fallbacks: "default"``: when one declines a request on safety grounds, the API runs it
#: again on a model Anthropic picks for that kind of refusal, inside the same call. Not on the Batch API.
FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Effort levels the API knows. Claude Haiku 4.5, Sonnet 4.5 and older reject the parameter outright.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_NO_EFFORT = ("claude-haiku-4", "claude-sonnet-4-5", "claude-opus-4-1", "claude-opus-4-0", "claude-3")


def supports_effort(model: str | None) -> bool:
    """Whether ``model`` takes ``output_config.effort`` (every current Claude model does; the oldest do not)."""
    m = str(model or "")
    return m.startswith("claude-") and not m.startswith(_NO_EFFORT)

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
        # prompt_tokens is the input billed at the full rate; what the cache served or stored is counted apart,
        # because it is billed apart (a read at a tenth of the rate or less, a write at 1.25 times).
        "usage": {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        },
        "stop_details": data.get("stop_details") if stop == "refusal" else None,
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
        kwargs.setdefault("timeout", ANTHROPIC_URLLIB_TIMEOUT)
        super().__init__(api_key, base, **kwargs)
        if workspace_id:
            self.extra_headers["anthropic-workspace-id"] = workspace_id
        self.workspace_id = workspace_id
        self.max_tokens = max_tokens
        self.effort = effort
        self._sdk = sdk_client
        #: The content blocks of every assistant turn that called a tool, by tool-use id.
        self._turns: dict[str, list[dict[str, Any]]] = {}

    def _call(self, payload: dict[str, Any], opts: dict[str, Any]) -> Any:
        system, messages = _anthropic_messages(payload.get("messages") or [], self._turns)
        model = payload.get("model")
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": int(payload.get("max_tokens") or self.max_tokens),
            "messages": messages,
        }
        if system:
            # The system prompt (with the tools, which come before it) is the part every call of a role or every
            # step of a loop shares, so the cache breakpoint goes on its last block: the next call reads it back
            # at a tenth of the input price or less instead of paying for it again.
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        tools = _anthropic_tools(payload.get("tools"))
        if tools:
            body["tools"] = tools
            choice = _anthropic_tool_choice(payload.get("tool_choice"))
            if choice:
                body["tool_choice"] = choice
            # A tool loop resends the whole conversation each step, so the growing tail is cached too (automatic
            # caching moves the breakpoint forward as it grows). A one-off call's tail is unique to it, and caching
            # it would only pay the write premium, so it is not.
            body["cache_control"] = {"type": "ephemeral"}
        effort = os.environ.get("AQUASCOPE_LLM_EFFORT") or opts.get("effort") or self.effort
        if effort in EFFORT_LEVELS and supports_effort(model):
            body["output_config"] = {"effort": effort}
            opts["effort_sent"] = effort
        if model in FALLBACK_MODELS and os.environ.get("AQUASCOPE_LLM_FALLBACKS", "1") != "0":
            body["fallbacks"] = "default"
            body["_betas"] = [FALLBACK_BETA]
        data = self._with_retries(body)  # the same 429 / 5xx patience as every other provider
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "tool_use" and b.get("id"):
                self._turns[b["id"]] = list(blocks)
        return _wrap(_from_anthropic(data))

    def _request_once(self, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}/v1/messages"
        payload = dict(payload)
        betas = payload.pop("_betas", None) or []
        if self._sdk is not None:
            return self._sdk_request(payload, url, betas)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"aquascope/{__version__}",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            **self.extra_headers,
        }
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        if in_pyodide():
            headers["anthropic-dangerous-direct-browser-access"] = "true"
        return self._post_json(url, payload, headers)

    def _sdk_request(self, payload: dict[str, Any], url: str, betas: list[str] | None = None) -> dict[str, Any]:
        """One call through the SDK, streamed: a long reply arrives as it is written instead of tripping an idle
        timeout, and ``get_final_message`` hands back the same message ``create`` would have."""
        import anthropic

        try:
            if betas:
                stream = self._sdk.beta.messages.stream(**payload, betas=betas)
            else:
                stream = self._sdk.messages.stream(**payload)
            with stream as events:
                response = events.get_final_message()
        except anthropic.APIStatusError as exc:
            body = exc.body if isinstance(exc.body, str) else json.dumps(exc.body, default=str)
            raise LLMHTTPError(exc.status_code, body or str(exc), url) from None
        except anthropic.APIConnectionError as exc:  # APITimeoutError included
            raise LLMConnectionError(url, f"{type(exc).__name__}: {exc}") from None
        return response.model_dump(mode="json", exclude_none=True)


class OpenAISDKChatClient(UrllibChatClient):
    """The ``openai`` SDK behind the same surface and the same error policy as :class:`UrllibChatClient`.

    The SDK raises its own exception classes, so on its own a 413 or a 429 never reached the analyst's
    recovery (which reads :class:`LLMHTTPError`) and timeouts were not retried by our rules. Here the SDK
    makes the call, its retries stay off, and its errors are mapped exactly as the Anthropic SDK's are.
    """

    def __init__(self, api_key: str | None, base_url: str | None = None, *, sdk_client: Any, **kwargs: Any):
        super().__init__(api_key, base_url, **kwargs)
        self._sdk = sdk_client

    def _request_once(self, payload: dict[str, Any]) -> Any:
        import openai

        url = f"{self.base_url}/chat/completions"
        try:
            response = self._sdk.chat.completions.create(**{k: v for k, v in payload.items() if v is not None})
        except openai.APIStatusError as exc:
            body = exc.body if isinstance(exc.body, str) else json.dumps(exc.body, default=str)
            raise LLMHTTPError(exc.status_code, body or str(exc), url) from None
        except openai.APIConnectionError as exc:  # APITimeoutError included
            raise LLMConnectionError(url, f"{type(exc).__name__}: {exc}") from None
        return _wrap(response.model_dump(mode="json", exclude_none=True))


def in_pyodide() -> bool:
    return sys.platform == "emscripten" or "pyodide" in sys.modules


def make_client(api_key: str | None, base_url: str | None, provider: str | None = None) -> Any:
    """The client for ``provider``: an SDK when installed (and we are not in a browser), else ``urllib``.

    ``provider`` is looked up in the registry for its wire protocol; unknown
    or missing means OpenAI-compatible, which every provider but Anthropic is.
    """
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

            # Our loop owns the retries (429 waits, bounded timeouts), so the SDK's stay off.
            sdk = OpenAI(api_key=api_key or "none", base_url=base_url, max_retries=0)
            return OpenAISDKChatClient(api_key, base_url, sdk_client=sdk)
        except ImportError:
            pass
    return UrllibChatClient(api_key, base_url)


__all__ = [
    "OPTIONS_KEY", "AnthropicChatClient", "LLMConnectionError", "LLMHTTPError", "OpenAISDKChatClient",
    "UrllibChatClient", "accepts_options", "in_pyodide", "make_client", "supports_effort", "with_options",
]
