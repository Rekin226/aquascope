"""Run model-written Python against aquascope, inside limits.

The Analyst's ten tools cover the questions we anticipated. The ones we did not
anticipate ("decadal maxima", "the ratio of these two records", "the same
analysis for every donor") need code, and the whole library is already loaded
next to the data: in the browser that is the reader's own tab, with no server
and nothing to pay for.

So this is a small, deliberate execution tool rather than a general one:

* the namespace is prepared (``aquascope``, ``workbench``, ``pandas``, ``numpy``,
  and whatever the caller passes as data), so the model does not need imports;
* imports are checked against an allow-list before anything runs, which keeps
  the obvious ways out (``os``, ``subprocess``, ``socket``, ``importlib``)
  from being a one-liner;
* output is whatever the snippet leaves in ``result``, plus anything it printed,
  both truncated;
* a wall-clock budget stops a runaway loop.

None of that makes it a security boundary. The real boundary is the one the
platform provides: in the Explorer this runs in the reader's own browser (a
WASM sandbox, their data, their machine), and in the CLI it runs with the same
rights as the aquascope process the user started. It is off unless the caller
turns it on.
"""

from __future__ import annotations

import ast
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SandboxResult", "SandboxError", "run_python", "ALLOWED_IMPORTS"]

#: Modules a snippet may import. Everything the analyses need, nothing that
#: reaches the filesystem, the network or another process.
ALLOWED_IMPORTS = frozenset({
    "aquascope", "numpy", "np", "pandas", "pd", "math", "statistics", "datetime",
    "json", "itertools", "collections", "re", "scipy", "typing", "dataclasses", "functools",
})

_BLOCKED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
})

MAX_OUTPUT_CHARS = 8_000


class SandboxError(RuntimeError):
    """The snippet was refused before it ran."""


@dataclass
class SandboxResult:
    ok: bool
    result: Any = None
    stdout: str = ""
    error: str | None = None
    seconds: float = 0.0
    figures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "seconds": round(self.seconds, 3)}
        if self.stdout:
            out["stdout"] = self.stdout
        if self.error:
            out["error"] = self.error
        if self.result is not None:
            out["result"] = self.result
        if self.figures:
            out["figures"] = self.figures
        return out


def _check(code: str) -> None:
    """Refuse a snippet that imports outside the allow-list or reaches for an escape hatch."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    allowed = ", ".join(sorted(ALLOWED_IMPORTS))
                    raise SandboxError(f"import of {alias.name!r} is not allowed here (allowed: {allowed})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise SandboxError(f"import from {node.module!r} is not allowed here")
        elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise SandboxError(f"{node.id!r} is not available here")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise SandboxError(f"dunder attribute {node.attr!r} is not available here")


def _summarise(value: Any, *, max_rows: int = 200) -> Any:
    """Turn whatever the snippet produced into something JSON can carry."""
    from aquascope.workbench import jsonable

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        return jsonable(value)

    if isinstance(value, pd.DataFrame):
        head = value.head(max_rows)
        return {"type": "table", "columns": [str(c) for c in head.columns],
                "rows": jsonable(head.to_numpy()), "n_rows": int(len(value))}
    if isinstance(value, pd.Series):
        head = value.head(max_rows)
        return {"type": "series", "name": str(value.name) if value.name is not None else None,
                "index": jsonable(list(head.index)), "values": jsonable(head.to_numpy()),
                "n": int(len(value))}
    return jsonable(value)


def run_python(
    code: str,
    *,
    data: dict[str, Any] | None = None,
    timeout_seconds: float = 20.0,
) -> SandboxResult:
    """Execute ``code`` with aquascope in scope and return what it left in ``result``.

    ``data`` is merged into the namespace: the Explorer passes the record on
    screen as ``df``, the analysis dict as ``analysis``, and so on.
    """
    _check(code)

    import numpy as np
    import pandas as pd

    import aquascope
    from aquascope import workbench

    namespace: dict[str, Any] = {
        "aquascope": aquascope, "workbench": workbench,
        "pd": pd, "pandas": pd, "np": np, "numpy": np,
        "result": None,
    }
    namespace.update(data or {})

    buffer = io.StringIO()
    started = time.perf_counter()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102 - the point of the tool
    except Exception as exc:  # noqa: BLE001 - the model gets to see the failure and fix it
        return SandboxResult(
            ok=False, error=f"{type(exc).__name__}: {exc}",
            stdout=buffer.getvalue()[:MAX_OUTPUT_CHARS], seconds=time.perf_counter() - started,
        )
    elapsed = time.perf_counter() - started
    if elapsed > timeout_seconds:
        # No pre-emption here (a thread cannot safely interrupt CPython, and the
        # browser has no signals), so a slow snippet is reported, not killed.
        return SandboxResult(
            ok=False, error=f"the snippet took {elapsed:.1f} s, over the {timeout_seconds:.0f} s budget",
            stdout=buffer.getvalue()[:MAX_OUTPUT_CHARS], seconds=elapsed,
        )

    figures = namespace.get("figures") or []
    if not isinstance(figures, list):
        figures = []
    return SandboxResult(
        ok=True,
        result=_summarise(namespace.get("result")),
        stdout=buffer.getvalue()[:MAX_OUTPUT_CHARS],
        seconds=elapsed,
        figures=figures[:4],
    )
