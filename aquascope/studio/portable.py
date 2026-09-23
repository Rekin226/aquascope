"""Versioned, local-only completed studies. Opening one never executes its plan."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from aquascope import __version__
from aquascope.claims import content_digest
from aquascope.studio.workspace import WORKSPACE_VERSION, Workspace

FORMAT = "aquascope-completed-study"
VERSION = 1
MAX_BYTES = 50_000_000


def dumps(ws: Workspace) -> str:
    """Include results, input tables and artifact bytes, excluding recursive bundles."""
    missing = [a.name for a in ws.artifacts if not a.data and a.id not in ("bundle", "workspace", "portable")]
    if missing:
        raise ValueError("This workspace is missing artifact bytes. Open the original complete study or full bundle "
                         "before exporting a complete study: " + ", ".join(missing[:3]))
    data = ws.to_dict(with_artifacts=True)
    data["artifacts"] = [a for a in data["artifacts"] if a["id"] not in ("bundle", "workspace", "portable")]
    envelope = {"format": FORMAT, "version": VERSION, "software_version": __version__,
                "workspace": data, "sha256": content_digest(data),
                "sharing": "Contains study inputs and results. Share this file only when you intend to disclose them."}
    text = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("The completed study exceeds the 50 MB portable-file limit; use the full bundle.")
    return text


def loads(text: str) -> Workspace:
    """Validate a portable file and restore its stored results without network or code execution."""
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("The completed study exceeds the 50 MB portable-file limit.")
    obj: dict[str, Any] = json.loads(text)
    if not isinstance(obj, dict) or obj.get("format") != FORMAT or obj.get("version") != VERSION:
        raise ValueError("Unsupported completed-study format or version.")
    data = obj.get("workspace")
    if not isinstance(data, dict) or data.get("version") != WORKSPACE_VERSION:
        raise ValueError("Unsupported workspace version.")
    if obj.get("sha256") != content_digest(data):
        raise ValueError("The completed-study contents do not match their checksum.")
    for artifact in data.get("artifacts") or []:
        name = str(artifact.get("name") or "")
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe artifact path in completed study.")
    return Workspace.from_dict(data)
