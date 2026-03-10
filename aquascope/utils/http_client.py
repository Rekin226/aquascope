"""
Shared HTTP client with retry logic, rate-limiting, and caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")


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
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_ttl = cache_ttl_seconds
        self.rate_limiter = rate_limiter
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

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
    def get_json(
        self,
        path: str,
        params: dict | None = None,
        use_cache: bool = True,
    ) -> Any:
        """GET *path* and return parsed JSON, with cache + retries."""
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
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
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

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
