"""Where aquascope keeps its on-disk caches.

Kept apart from :mod:`aquascope.archive.catalog` (which re-exports it) so the light modules that only need a
cache folder (:mod:`aquascope.rivers`, :mod:`aquascope.nownext`) do not import the Archive package, which
pulls pandas in: the Explorer's light workers run those modules without it.
"""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """``$AQUASCOPE_CACHE_DIR``, else ``~/.cache/aquascope``, created if missing."""
    root = os.environ.get("AQUASCOPE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "aquascope")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path
