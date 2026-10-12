"""Guard: nothing in the package may call the retired USGS Water Services host (#515).

USGS shuts ``waterservices.usgs.gov`` down in Q1 2027. A call left behind would fail as a 404, which the
collectors turn into an empty frame and a quiet hole in the Archive, so this test fails the build instead.
"""

from __future__ import annotations

from pathlib import Path

import aquascope

LEGACY_HOST = "waterservices" + ".usgs.gov"  # split so this file does not match its own search


def test_no_file_under_aquascope_references_the_legacy_usgs_host():
    root = Path(aquascope.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if LEGACY_HOST in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"{LEGACY_HOST} is retired (#515); move these to api.waterdata.usgs.gov: {offenders}"
