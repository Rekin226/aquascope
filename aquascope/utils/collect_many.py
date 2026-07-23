"""Concurrent multi-source data fetching with partial-failure handling.

Provides :func:`collect_many`, a helper that runs multiple collector
calls in parallel using a ``ThreadPoolExecutor``, respects per-host
rate limiting, and returns both successes and structured failures
instead of aborting on the first error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CollectRequest:
    """A single fetch request to be executed by :func:`collect_many`.

    Attributes
    ----------
    key : str
        Unique identifier for this request (e.g. station ID or source name).
        Used to map results back to requests.
    fn : Callable
        Zero-argument callable that performs the fetch and returns data.
        Wrap your collector call in a lambda or ``functools.partial``::

            CollectRequest(
                key="usgs_01234",
                fn=lambda: usgs_collector.collect(site="01234"),
            )
    """

    key: str
    fn: Callable[[], Any]


@dataclass
class CollectResult:
    """Aggregated result from :func:`collect_many`.

    Attributes
    ----------
    successes : dict[str, Any]
        Mapping of ``request.key`` → returned data for successful requests.
    failures : list[CollectFailure]
        Structured list of failures (key + exception) for failed requests.
    """

    successes: dict[str, Any] = field(default_factory=dict)
    failures: list[CollectFailure] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        """Number of successful fetches."""
        return len(self.successes)

    @property
    def n_failure(self) -> int:
        """Number of failed fetches."""
        return len(self.failures)


@dataclass
class CollectFailure:
    """A single failed fetch request.

    Attributes
    ----------
    key : str
        The ``CollectRequest.key`` that failed.
    error : Exception
        The exception raised during the fetch.
    """

    key: str
    error: Exception


def collect_many(
    requests: list[CollectRequest],
    max_workers: int = 8,
    progress_callback: Callable[[str, bool], None] | None = None,
) -> CollectResult:
    """Run multiple collector calls concurrently with partial-failure handling.

    Executes each :class:`CollectRequest` in a thread pool.  One failing
    request does not abort the batch — failures are collected and returned
    alongside successes.  Per-host rate limiting is honoured because each
    ``CollectRequest.fn`` calls through the collector's own
    :class:`~aquascope.utils.http_client.CachedHTTPClient`, which already
    coordinates with its :class:`~aquascope.utils.http_client.RateLimiter`.

    Parameters
    ----------
    requests : list[CollectRequest]
        Fetch requests to execute.  Order is not preserved in the result.
    max_workers : int
        Maximum number of concurrent threads (default 8).  Tune down if
        rate limits are tight; tune up for many independent hosts.
    progress_callback : callable, optional
        Called after each request completes with ``(key, success)`` where
        *success* is ``True`` on success and ``False`` on failure.  Use
        this to drive a ``tqdm`` progress bar or a custom UI::

            with tqdm(total=len(requests)) as pbar:
                collect_many(
                    requests,
                    progress_callback=lambda key, ok: pbar.update(1),
                )

    Returns
    -------
    CollectResult
        Aggregated successes and failures.

    Examples
    --------
    >>> reqs = [
    ...     CollectRequest(key="a", fn=lambda: fetch_station("A")),
    ...     CollectRequest(key="b", fn=lambda: fetch_station("B")),
    ... ]
    >>> result = collect_many(reqs, max_workers=4)
    >>> result.n_success, result.n_failure
    (2, 0)
    """
    if not requests:
        return CollectResult()

    result = CollectResult()
    n = len(requests)
    logger.info("collect_many: starting %d requests with max_workers=%d", n, max_workers)

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        future_to_key = {
            executor.submit(req.fn): req.key
            for req in requests
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                data = future.result()
                result.successes[key] = data
                logger.debug("collect_many: ✓ %s", key)
                if progress_callback:
                    progress_callback(key, True)
            except Exception as exc:
                result.failures.append(CollectFailure(key=key, error=exc))
                logger.warning("collect_many: ✗ %s — %s: %s", key, type(exc).__name__, exc)
                if progress_callback:
                    progress_callback(key, False)

    logger.info(
        "collect_many: done — %d succeeded, %d failed",
        result.n_success,
        result.n_failure,
    )
    return result
