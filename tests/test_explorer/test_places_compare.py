"""My places and Compare in the Explorer: the page, the worker op and the wiring line up.

The general asset test already checks that every id a module reaches for exists; these pin the pieces that
make the feature reachable at all (a header button, a station button, a surface the shell knows about, and a
worker op that calls the package rather than doing the science in JS).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"


def _read(*parts: str) -> str:
    return (EXPLORER.joinpath(*parts)).read_text(encoding="utf-8")


def test_the_page_has_the_places_and_compare_elements() -> None:
    ids = set(re.findall(r'id="([^"]+)"', _read("index.html")))
    need = {
        "btn-places", "places-count", "btn-place", "panel-places", "places-list", "places-empty", "places-hint",
        "places-status", "places-back", "btn-compare", "cmp-hydro-card", "plot-cmp-hydro", "cmp-fdc-card",
        "plot-cmp-fdc", "cmp-ffa-card", "plot-cmp-ffa", "cmp-summary-card", "cmp-table", "cmp-actions",
        "cmp-notes", "cmp-methods", "cmp-basis", "cmp-window",
        "pl-t-list", "pl-t-hydro", "pl-t-fdc", "pl-t-ffa", "pl-tab-list", "pl-tab-hydro", "pl-tab-fdc", "pl-tab-ffa",
    }
    assert not need - ids, sorted(need - ids)


def test_the_shell_knows_the_places_surface() -> None:
    assert '"panel-places"' in _read("src", "shell.js")


def test_the_modules_are_wired() -> None:
    assert "initPlaces()" in _read("app.js")
    assert "syncPlaceButton()" in _read("src", "panel-station.js")
    assert 'call("compare"' in _read("src", "compare.js")


def test_the_worker_calls_the_package_for_compare() -> None:
    worker = _read("worker.js")
    assert 'if (m.type === "compare") return await compare(m);' in worker
    assert "aquascope.compare" in worker and "compare_stations(" in worker


def test_every_storage_access_in_places_is_guarded() -> None:
    """A private window or blocked storage must not break the page: each localStorage touch sits in a try."""
    js = re.sub(r"//[^\n]*", "", _read("src", "places.js"))
    for m in re.finditer(r"\b(getItem|setItem|localStorage)\b", js):
        before = js[max(0, m.start() - 200): m.start()]
        assert "try" in before, f"unguarded storage access near: {js[m.start() - 60: m.end() + 20]!r}"


def test_compare_draws_with_the_shared_chart_helper() -> None:
    js = _read("src", "compare.js")
    assert 'from "./charts.js?v=__BUILD__"' in js
    assert "Plotly.react" not in js  # always through charts.plot, so the theme and PNG export apply
