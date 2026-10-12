"""Keep aquascope up to date: the newest release on PyPI, how this copy was installed, and the command
that upgrades it the same way.

A program does not choose how it is installed, and the right upgrade depends on that: a uv tool is
upgraded by uv, a pipx app by pipx, a venv or a conda environment by its own pip, and a development
checkout by ``git pull``, never by pip. :func:`install_kind` reads the answer off the running
interpreter, so ``aquascope update`` can run the right command instead of guessing.

The CLI face is ``aquascope update [--check] [--yes]`` and ``aquascope --version``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PYPI_URL = "https://pypi.org/pypi/aquascope/json"

#: Install kinds, in the order :func:`install_kind` tests them.
KINDS = ("editable", "uv-tool", "pipx", "conda", "venv", "system")


def installed_version() -> str:
    from aquascope import __version__

    return str(__version__)


def latest_version(timeout: float = 10.0) -> str:
    """The newest release on PyPI (pre-releases are never ``info.version``)."""
    import httpx

    resp = httpx.get(PYPI_URL, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return str(resp.json()["info"]["version"])


def _release(version: str) -> tuple[int, ...]:
    """``"0.22.0"`` -> ``(0, 22, 0)``; anything after the numeric release (``rc1``, ``.dev3``) is dropped."""
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) < len(piece):
            break
    return tuple(parts)


def is_newer(latest: str, current: str) -> bool:
    """Whether ``latest`` is a later release than ``current``."""
    try:
        from packaging.version import Version

        return bool(Version(latest) > Version(current))
    except Exception:  # packaging is not a dependency; fall back on the numeric release
        return _release(latest) > _release(current)


def _editable() -> bool:
    """Whether this aquascope runs from a source checkout: the package sits in a git clone with its
    ``pyproject.toml`` (an editable install, or the clone on ``PYTHONPATH``), or pip recorded an editable
    install (PEP 610)."""
    import aquascope

    root = Path(aquascope.__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists() and (root / ".git").exists():
        return True
    from importlib import metadata

    try:
        raw = metadata.distribution("aquascope").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable"))
    except ValueError:
        return False


def _uv_receipt(prefix: Path) -> dict[str, Any] | None:
    path = prefix / "uv-receipt.toml"
    if not path.exists():
        return None
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # Python 3.10: read the two fields this module needs from the text
        return _uv_receipt_loose(path)
    try:
        return dict(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {"tool": {}}


def _uv_receipt_loose(path: Path) -> dict[str, Any]:
    """The aquascope requirement and the Python of a uv receipt, without a TOML parser."""
    import re

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"tool": {}}
    req: dict[str, Any] = {"name": "aquascope"}
    m = re.search(r'name\s*=\s*"aquascope"[^}]*', text)
    if m:
        extras = re.search(r'extras\s*=\s*\[([^\]]*)\]', m.group(0))
        spec = re.search(r'specifier\s*=\s*"([^"]*)"', m.group(0))
        req["extras"] = re.findall(r'"([^"]+)"', extras.group(1)) if extras else []
        if spec:
            req["specifier"] = spec.group(1)
    py = re.search(r'^python\s*=\s*"([^"]+)"', text, re.M)
    return {"tool": {"requirements": [req], **({"python": py.group(1)} if py else {})}}


def install_kind(prefix: str | Path | None = None, base_prefix: str | Path | None = None) -> str:
    """How this copy of aquascope was installed: one of :data:`KINDS`."""
    pre = Path(prefix or sys.prefix)
    base = Path(base_prefix or sys.base_prefix)
    if prefix is None and _editable():
        return "editable"
    if (pre / "uv-receipt.toml").exists():
        return "uv-tool"
    parts = pre.parts
    if "pipx" in parts and "venvs" in parts:
        return "pipx"
    if (pre / "conda-meta").exists():
        return "conda"
    if pre != base:
        return "venv"
    return "system"


def upgrade_command(kind: str, *, prefix: str | Path | None = None) -> list[str] | None:
    """The command that upgrades aquascope for an install ``kind``; ``None`` for a development checkout.

    A uv tool installed with an exact pin (``aquascope==0.22.0``) is never moved by ``uv tool upgrade``,
    so it is reinstalled without the pin, with the same extras and Python.
    """
    pre = Path(prefix or sys.prefix)
    if kind == "editable":
        return None
    if kind == "uv-tool":
        tool = (_uv_receipt(pre) or {}).get("tool") or {}
        req: dict[str, Any] = next(
            (r for r in tool.get("requirements") or [] if r.get("name") == "aquascope"), {})
        if str(req.get("specifier") or "").startswith("=="):
            extras = ",".join(req.get("extras") or [])
            cmd = ["uv", "tool", "install", f"aquascope[{extras}]" if extras else "aquascope", "--force"]
            return cmd + (["--python", str(tool["python"])] if tool.get("python") else [])
        return ["uv", "tool", "upgrade", "aquascope"]
    if kind == "pipx":
        return ["pipx", "upgrade", "aquascope"]
    return [sys.executable, "-m", "pip", "install", "--upgrade", "aquascope"]


def check(*, timeout: float = 10.0) -> dict[str, Any]:
    """Everything ``aquascope update`` needs: versions, whether a newer one exists, the kind and the command.

    ``latest`` is ``None`` and ``error`` says why when PyPI cannot be reached.
    """
    kind = install_kind()
    out: dict[str, Any] = {"installed": installed_version(), "latest": None, "newer": False, "kind": kind,
                           "command": upgrade_command(kind), "error": None, "location": sys.prefix}
    try:
        out["latest"] = latest_version(timeout=timeout)
    except Exception as exc:
        out["error"] = f"could not reach PyPI: {exc}"
        return out
    out["newer"] = is_newer(out["latest"], out["installed"])
    missing = out["command"] and shutil.which(out["command"][0]) is None and out["command"][0] != sys.executable
    if missing:
        out["error"] = f"`{out['command'][0]}` is not on your PATH; install it or upgrade by hand"
    return out
