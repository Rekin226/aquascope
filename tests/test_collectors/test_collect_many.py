"""Tests for aquascope.utils.collect_many."""

from __future__ import annotations

import threading
import time

from aquascope.utils.collect_many import (
    CollectFailure,
    CollectRequest,
    CollectResult,
    collect_many,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _ok(key: str, value: object = None):
    """Return a CollectRequest that always succeeds."""
    return CollectRequest(key=key, fn=lambda: value if value is not None else key)


def _fail(key: str, exc: Exception | None = None):
    """Return a CollectRequest that always raises."""
    e = exc or RuntimeError(f"simulated failure for {key}")
    return CollectRequest(key=key, fn=lambda: (_ for _ in ()).throw(e))


# ── Basic return types ────────────────────────────────────────────────────

class TestCollectManyReturnTypes:
    def test_returns_collect_result(self):
        result = collect_many([_ok("a")])
        assert isinstance(result, CollectResult)

    def test_empty_requests_returns_empty_result(self):
        result = collect_many([])
        assert result.n_success == 0
        assert result.n_failure == 0
        assert result.successes == {}
        assert result.failures == []

    def test_successes_is_dict(self):
        result = collect_many([_ok("x", 42)])
        assert isinstance(result.successes, dict)

    def test_failures_is_list(self):
        result = collect_many([_fail("x")])
        assert isinstance(result.failures, list)


# ── All success ───────────────────────────────────────────────────────────

class TestCollectManyAllSuccess:
    def test_all_keys_present(self):
        reqs = [_ok("a", 1), _ok("b", 2), _ok("c", 3)]
        result = collect_many(reqs)
        assert set(result.successes.keys()) == {"a", "b", "c"}

    def test_values_correct(self):
        reqs = [_ok("x", 10), _ok("y", 20)]
        result = collect_many(reqs)
        assert result.successes["x"] == 10
        assert result.successes["y"] == 20

    def test_n_success_correct(self):
        reqs = [_ok(str(i)) for i in range(5)]
        result = collect_many(reqs)
        assert result.n_success == 5

    def test_n_failure_zero(self):
        reqs = [_ok(str(i)) for i in range(5)]
        result = collect_many(reqs)
        assert result.n_failure == 0


# ── Partial failure ───────────────────────────────────────────────────────

class TestCollectManyPartialFailure:
    def test_one_failure_does_not_abort_batch(self):
        reqs = [_ok("a", 1), _fail("b"), _ok("c", 3)]
        result = collect_many(reqs)
        assert result.n_success == 2
        assert result.n_failure == 1

    def test_successful_keys_present(self):
        reqs = [_ok("a", 1), _fail("b"), _ok("c", 3)]
        result = collect_many(reqs)
        assert "a" in result.successes
        assert "c" in result.successes

    def test_failed_key_not_in_successes(self):
        reqs = [_ok("a", 1), _fail("b")]
        result = collect_many(reqs)
        assert "b" not in result.successes

    def test_failure_has_correct_key(self):
        reqs = [_fail("bad_key")]
        result = collect_many(reqs)
        assert result.failures[0].key == "bad_key"

    def test_failure_has_exception(self):
        exc = ValueError("test error")
        reqs = [_fail("x", exc)]
        result = collect_many(reqs)
        assert isinstance(result.failures[0].error, ValueError)

    def test_all_fail_zero_successes(self):
        reqs = [_fail(str(i)) for i in range(4)]
        result = collect_many(reqs)
        assert result.n_success == 0
        assert result.n_failure == 4

    def test_failures_are_collect_failure_instances(self):
        reqs = [_fail("x")]
        result = collect_many(reqs)
        assert isinstance(result.failures[0], CollectFailure)


# ── Concurrency ───────────────────────────────────────────────────────────

class TestCollectManyConcurrency:
    def test_runs_concurrently(self):
        """Prove concurrency: N slow tasks finish faster than N × sleep_time."""
        n = 6
        sleep_time = 0.2

        reqs = [
            CollectRequest(key=str(i), fn=lambda: time.sleep(sleep_time) or True)
            for i in range(n)
        ]

        start = time.monotonic()
        result = collect_many(reqs, max_workers=n)
        elapsed = time.monotonic() - start

        assert result.n_success == n
        # Sequential would take n * sleep_time; concurrent should be much less
        assert elapsed < n * sleep_time * 0.75

    def test_max_workers_respected(self):
        """Never more than max_workers threads active simultaneously."""
        max_workers = 3
        active = [0]
        peak = [0]
        lock = threading.Lock()

        def task():
            with lock:
                active[0] += 1
                if active[0] > peak[0]:
                    peak[0] = active[0]
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return True

        reqs = [CollectRequest(key=str(i), fn=task) for i in range(9)]
        collect_many(reqs, max_workers=max_workers)
        assert peak[0] <= max_workers

    def test_single_worker_still_works(self):
        reqs = [_ok(str(i), i) for i in range(3)]
        result = collect_many(reqs, max_workers=1)
        assert result.n_success == 3


# ── Progress callback ─────────────────────────────────────────────────────

class TestCollectManyProgressCallback:
    def test_callback_called_for_each_request(self):
        calls = []
        reqs = [_ok("a"), _ok("b"), _fail("c")]
        collect_many(reqs, progress_callback=lambda key, ok: calls.append((key, ok)))
        assert len(calls) == 3

    def test_callback_reports_success(self):
        calls = []
        collect_many([_ok("x", 1)], progress_callback=lambda k, ok: calls.append(ok))
        assert calls == [True]

    def test_callback_reports_failure(self):
        calls = []
        collect_many([_fail("x")], progress_callback=lambda k, ok: calls.append(ok))
        assert calls == [False]

    def test_no_callback_does_not_raise(self):
        result = collect_many([_ok("a")], progress_callback=None)
        assert result.n_success == 1


# ── Edge cases ────────────────────────────────────────────────────────────

class TestCollectManyEdgeCases:
    def test_single_request_success(self):
        result = collect_many([_ok("only", 99)])
        assert result.successes["only"] == 99

    def test_single_request_failure(self):
        result = collect_many([_fail("only")])
        assert result.n_failure == 1

    def test_max_workers_larger_than_requests(self):
        """max_workers > n should not raise."""
        reqs = [_ok("a", 1)]
        result = collect_many(reqs, max_workers=100)
        assert result.n_success == 1

    def test_n_success_plus_n_failure_equals_total(self):
        reqs = [_ok("a"), _fail("b"), _ok("c"), _fail("d"), _ok("e")]
        result = collect_many(reqs)
        assert result.n_success + result.n_failure == len(reqs)
