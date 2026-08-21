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
        import urllib.error
        import urllib.request

        url = f"{self.base_url}/chat/completions"
        body = json.dumps({k: v for k, v in payload.items() if v is not None}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"aquascope/{__version__}",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict) and data.get("error") and not data.get("choices"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"{url}: {msg}")
        return _wrap(data)


def in_pyodide() -> bool:
    return sys.platform == "emscripten" or "pyodide" in sys.modules


def make_client(api_key: str | None, base_url: str | None) -> Any:
    """The OpenAI SDK when it is installed (and we are not in a browser), else :class:`UrllibChatClient`."""
    import os

    if not in_pyodide() and os.environ.get("AQUASCOPE_LLM_TRANSPORT", "").lower() != "urllib":
        try:
            from openai import OpenAI

            return OpenAI(api_key=api_key or "none", base_url=base_url)
        except ImportError:
            pass
    return UrllibChatClient(api_key, base_url)


__all__ = ["LLMHTTPError", "UrllibChatClient", "in_pyodide", "make_client"]
