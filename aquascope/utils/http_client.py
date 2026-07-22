"""
Shared HTTP client with retry logic, rate-limiting, and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")

#: True when running under Pyodide/WebAssembly (e.g. the stlite browser demo).
IS_EMSCRIPTEN = sys.platform == "emscripten"

_dumps = json.dumps


class _EmscriptenResponse:
    """Just enough of the ``httpx.Response`` surface for CachedHTTPClient."""

    def __init__(self, status_code: int, headers: dict, text: str, url: str):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.TransportError(f"HTTP {self.status_code} for {self.url}")


class _EmscriptenClient:
    """Minimal ``httpx.Client`` stand-in for browser (Pyodide/WASM) runtimes.

    httpx cannot open sockets inside WebAssembly, so under Emscripten requests
    go through ``urllib.request``, which pyodide-http patches to use the
    browser's XHR/fetch. Whether a source responds then depends on its CORS
    policy — failures surface with a CORS hint rather than hanging.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        try:
            import pyodide_http

            pyodide_http.patch_all()
        except ImportError:  # pragma: no cover - only present in Pyodide
            logger.warning("pyodide_http not available; browser requests may fail")

    def _request(self, method: str, url: str, params=None, headers=None, body=None):
        import urllib.error
        import urllib.parse
        import urllib.request

        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            qs = urllib.parse.urlencode(clean, doseq=True)
            if qs:
                url = f"{url}{'&' if '?' in url else '?'}{qs}"

        req = urllib.request.Request(url, data=body, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if body is not None and not any(k.lower() == "content-type" for k in (headers or {})):
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return _EmscriptenResponse(resp.status, hdrs, text, url)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            hdrs = {k.lower(): v for k, v in (exc.headers or {}).items()}
            return _EmscriptenResponse(exc.code, hdrs, text, url)
        except (urllib.error.URLError, OSError) as exc:
            raise httpx.TransportError(
                f"Browser fetch failed for {url}. This source may not allow "
                f"cross-origin (CORS) requests from web apps — install "
                f"aquascope locally for full access. ({exc})"
            ) from exc

    def get(self, url: str, params=None, headers=None):
        return self._request("GET", url, params=params, headers=headers)

    def post(self, url: str, json=None, params=None, headers=None):
        body = None if json is None else _dumps(json).encode("utf-8")
        return self._request("POST", url, params=params, headers=headers, body=body)

    def close(self) -> None:
        pass


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_calls: int = 30, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: list[float] = []

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.period]
        if len(self._timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self._timestamps[0])
            if sleep_time > 0:
                logger.debug("Rate limit reached — sleeping %.1f s", sleep_time)
                time.sleep(sleep_time)
        self._timestamps.append(time.monotonic())


class CachedHTTPClient:
    """
    HTTP client wrapper that provides:
    - automatic retries with exponential back-off
    - optional disk caching
    - optional rate-limiting
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = 30.0,
        retries: int = 3,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int = 3600,
        rate_limiter: RateLimiter | None = None,
        verify: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_ttl = cache_ttl_seconds
        self.rate_limiter = rate_limiter

        if not verify and not IS_EMSCRIPTEN:
            import warnings

            import urllib3

            warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
            logger.warning(
                "TLS verification DISABLED for base_url=%s — only use this for "
                "open-data hosts known to ship broken cert chains.",
                base_url,
            )

        if IS_EMSCRIPTEN:
            # Browser XHR fails fast and deterministically (no flaky sockets),
            # so retry loops just waste the visitor's time. TLS verification is
            # the browser's call — `verify=False` cannot be honoured here.
            self.retries = min(self.retries, 1)
            self._client = _EmscriptenClient(timeout=timeout)
        else:
            self._client = httpx.Client(timeout=timeout, follow_redirects=True, verify=verify)

    # ── cache helpers ────────────────────────────────────────────────
    def _cache_key(self, url: str, params: dict | None) -> str:
        raw = f"{url}|{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> Any | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl:
            path.unlink(missing_ok=True)
            return None
        return json.loads(path.read_text())

    def _write_cache(self, key: str, data: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, default=str))

    # ── public API ───────────────────────────────────────────────────
    @staticmethod
    def _parse_response_json(resp: httpx.Response) -> Any:
        """
        Parse JSON from *resp*, handling common pitfalls:

        - Raises ``ValueError`` with a clear message if Content-Type indicates
          HTML or XML (the server returned an error page instead of JSON).
        - Strips a UTF-8 BOM (``\\ufeff``) and leading/trailing whitespace
          before parsing, which fixes the ``JSONDecodeError: Expecting value:
          line 5 column 1`` class of failures.
        - On ``JSONDecodeError``, logs and re-raises with the first 200 chars
          of the body so callers can diagnose unexpected formats.
        """
        content_type = resp.headers.get("content-type", "")

        if "text/html" in content_type:
            preview = resp.text[:500]
            raise ValueError(
                f"Expected JSON but received HTML (Content-Type: {content_type!r}). "
                f"The server may have returned an error page. "
                f"Response preview: {preview!r}"
            )
        if "text/xml" in content_type or "application/xml" in content_type:
            preview = resp.text[:500]
            raise ValueError(
                f"Expected JSON but received XML (Content-Type: {content_type!r}). Response preview: {preview!r}"
            )

        # Strip BOM (\ufeff) and surrounding whitespace before parsing.
        text = resp.text.lstrip("\ufeff").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            preview = text[:200]
            raise ValueError(f"JSON decode failed ({exc}). First 200 chars of response: {preview!r}") from exc

    def get_json(
        self,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        use_cache: bool = True,
    ) -> Any:
        """GET *path* and return parsed JSON, with cache + retries.

        Args:
            path: URL path (appended to ``base_url``) or a full URL.
            params: Optional query parameters.
            headers: Optional extra request headers (e.g. ``{"Accept":
                "application/json"}``).
            use_cache: Whether to read/write the disk cache.
        """
        if path.startswith(("http://", "https://")):
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path

        if use_cache:
            key = self._cache_key(url, params)
            cached = self._read_cache(key)
            if cached is not None:
                logger.debug("Cache hit for %s", url)
                return cached

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            try:
                resp = self._client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = self._parse_response_json(resp)
                if use_cache:
                    self._write_cache(key, data)
                return data
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt,
                    self.retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"All {self.retries} attempts failed for {url}") from last_exc

    def get_text(
        self,
        path: str,
        params: dict | None = None,
        headers: dict | None = None,
        use_cache: bool = True,
    ) -> str:
        """GET *path* and return the raw response body as text.

        Shares the same retry, rate-limit, and disk-cache behaviour as
        :meth:`get_json`, but does not attempt JSON parsing. Use this for
        endpoints that return CSV or other plain-text payloads.
        """
        if path.startswith(("http://", "https://")):
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path

        if use_cache:
            # Suffix the cache key so text payloads never collide with the
            # JSON cache for the same URL.
            key = f"{self._cache_key(url, params)}-text"
            path_cache = self.cache_dir / f"{key}.txt"
            if path_cache.exists():
                age = time.time() - path_cache.stat().st_mtime
                if age <= self.cache_ttl:
                    logger.debug("Cache hit for %s", url)
                    return path_cache.read_text()
                path_cache.unlink(missing_ok=True)

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            try:
                resp = self._client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                text = resp.text
                if use_cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / f"{key}.txt").write_text(text)
                return text
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt,
                    self.retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"All {self.retries} attempts failed for {url}") from last_exc

    def post_json(
        self,
        path: str,
        json_body: Any | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        use_cache: bool = True,
    ) -> Any:
        """POST *path* (optionally with a JSON body) and return parsed JSON.

        Shares the retry, rate-limit, and disk-cache behaviour of
        :meth:`get_json`. The cache key includes the JSON body, so different
        payloads to the same URL cache independently. Use this for portals that
        expose data through ``POST`` actions (e.g. the WRA gweb HydroInfo
        endpoints), where caching is important to avoid re-hammering a server
        that throttles request bursts.
        """
        if path.startswith(("http://", "https://")):
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path

        # Cache key folds in the JSON body so distinct payloads do not collide.
        cache_params = dict(params or {})
        if json_body is not None:
            cache_params["__body__"] = json_body
        if use_cache:
            key = f"{self._cache_key(url, cache_params)}-post"
            cached = self._read_cache(key)
            if cached is not None:
                logger.debug("Cache hit (POST) for %s", url)
                return cached

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
            try:
                resp = self._client.post(url, json=json_body, params=params, headers=headers)
                resp.raise_for_status()
                data = self._parse_response_json(resp)
                if use_cache:
                    self._write_cache(key, data)
                return data
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "Attempt %d/%d failed for POST %s: %s — retrying in %ds",
                    attempt,
                    self.retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"All {self.retries} attempts failed for POST {url}") from last_exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
