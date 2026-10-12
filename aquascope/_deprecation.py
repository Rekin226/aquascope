"""How a public name is retired without breaking anyone (#588, ``docs/stability.md``).

A stable name never just disappears. It is first deprecated: it keeps working, it warns, and it is listed in
:data:`DEPRECATED` with the release that deprecated it and what to use instead. It can be removed only in the next
major release, and only after it has warned for at least six months.

Three tools, all standard library:

* :data:`DEPRECATED`, the register. ``tests/test_public_api.py`` reads it: a stable name, CLI verb or flag, or MCP
  tool or argument missing from the package fails the test unless it is registered here as removable.
* :func:`moved`, for a module's ``__getattr__``: the old name keeps resolving to the new object and warns.
* :func:`deprecated`, for a function that stays but should no longer be used: calling it warns.

The warning is :class:`AquaScopeDeprecationWarning`, a ``FutureWarning``, so Python shows it by default in scripts
and notebooks (a plain ``DeprecationWarning`` is hidden outside ``__main__``, and most of AquaScope's users are
not library authors who run with warnings on).
"""

from __future__ import annotations

import functools
import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class AquaScopeDeprecationWarning(FutureWarning):
    """A public AquaScope name, CLI verb or flag, or MCP tool that still works but will be removed."""


@dataclass(frozen=True)
class Deprecation:
    """One deprecated name: the release that deprecated it, what to use instead, and the release that may remove
    it (always a major release)."""

    since: str
    use: str
    remove_in: str = "2.0"


# Keys:
#   "aquascope.module:name"       a Python name (``module:name``, as in an entry point)
#   "cli:verb" / "cli:verb --flag" a CLI verb or one of its flags ("cli:river snap --max-distance")
#   "mcp:tool" / "mcp:tool.arg"   an MCP tool or one of its arguments
# Empty at 1.0. Add an entry in the same pull request that deprecates the name, and a CHANGELOG line under
# "Deprecated".
DEPRECATED: dict[str, Deprecation] = {}


def message(key: str, register: dict[str, Deprecation] | None = None) -> str:
    """The one-line warning for ``key``: what is deprecated, since when, what to use, and when it goes."""
    entry = (DEPRECATED if register is None else register)[key]
    what = key.split(":", 1)[1] if key.startswith(("cli:", "mcp:")) else key.replace(":", ".")
    kind = "aquascope " if key.startswith("cli:") else "the MCP tool " if key.startswith("mcp:") else ""
    return (f"{kind}{what} is deprecated since {entry.since} and will be removed in {entry.remove_in}; "
            f"use {entry.use} instead.")


def warn(key: str, *, stacklevel: int = 2, register: dict[str, Deprecation] | None = None) -> None:
    """Warn that ``key`` (a :data:`DEPRECATED` key) is deprecated."""
    warnings.warn(message(key, register), AquaScopeDeprecationWarning, stacklevel=stacklevel + 1)


def moved(module: str, aliases: dict[str, str], *,
          register: dict[str, Deprecation] | None = None) -> Callable[[str], Any]:
    """A module ``__getattr__`` that keeps old names working.

    ``aliases`` maps an old name in ``module`` to its new home as ``"package.module:name"``. Each old name must
    have a :data:`DEPRECATED` entry under ``"<module>:<old name>"``. Use it at the bottom of the old module::

        __getattr__ = moved(__name__, {"old_name": "aquascope.new_module:new_name"})
    """

    def module_getattr(name: str) -> Any:
        target = aliases.get(name)
        if target is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        warn(f"{module}:{name}", register=register)
        new_module, _, new_name = target.partition(":")
        return getattr(importlib.import_module(new_module), new_name)

    return module_getattr


def deprecated(key: str, *, register: dict[str, Deprecation] | None = None) -> Callable[[F], F]:
    """Mark a function that stays in place but is deprecated: calling it warns. ``key`` is its
    :data:`DEPRECATED` key; the wrapper carries it as ``__deprecated__`` (as PEP 702's decorator does)."""

    def wrap(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            warn(key, register=register)
            return fn(*args, **kwargs)

        inner.__deprecated__ = message(key, register)  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap


def removable(key: str, version: str, register: dict[str, Deprecation] | None = None) -> bool:
    """True when ``key`` is deprecated and ``version`` is at or past the major release that may remove it."""
    entry = (DEPRECATED if register is None else register).get(key)
    if entry is None:
        return False
    return int(version.split(".")[0]) >= int(entry.remove_in.split(".")[0])
