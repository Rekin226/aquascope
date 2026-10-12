"""Map first (#548): a click answers in a card on the map, and the panel waits for Details.

The card writes no numbers of its own. Its status sentence for a gauge in the daily snapshot is drawn from
the snapshot's row, so it is held here to :func:`aquascope.nownext.flow_status`'s own sentence word for
word; the rest checks the wiring a browser would otherwise be the first to notice.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquascope import nownext

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"
SRC = EXPLORER / "src"

pytestmark_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _read(name: str) -> str:
    return (EXPLORER / name).read_text(encoding="utf-8")


def _snapshot_sentences(rows: list[dict], today: str) -> list[str | None]:
    core = SRC / "map-card-core.js"
    script = f"""
    const m = await import({json.dumps(core.as_uri())});
    const rows = {json.dumps(rows)};
    console.log(JSON.stringify(rows.map((r) => m.snapshotSentence(r, {{ today: {json.dumps(today)} }}))));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


@pytestmark_node
@pytest.mark.parametrize("today", ["2026-10-09", "2026-10-20"])
def test_the_card_says_a_snapshot_row_in_flow_status_words(today: str) -> None:
    rng = np.random.default_rng(548)
    days = pd.date_range("1990-01-01", "2026-10-08", freq="D")
    series = pd.Series(rng.gamma(2.0, 5.0, len(days)), index=days)
    rows, expected = [], []
    for value in (0.5, 10.0, 60.0):   # well below, about normal, well above
        st = nownext.flow_status(series, value=value, date="2026-10-08", today=today)
        assert st["class"], st
        rows.append({"cls": st["class"], "pct": st["percentile"], "date": st["date"], "n_years": st["n_years"]})
        # flow_status adds ", the latest day in the record" only when it picked the day itself; given the
        # day, as here, its sentence is the card's.
        expected.append(st["sentence"])
    assert _snapshot_sentences(rows, today) == expected


def test_the_panel_starts_folded_and_the_card_is_on_the_page() -> None:
    html = _read("index.html")
    assert re.search(r'<body class="[^"]*\bpanel-collapsed\b', html)
    assert 'id="btn-panel"' in html and 'aria-expanded="false" aria-controls="panel"' in html
    card = re.search(r'<section id="map-card"[^>]*>', html)
    assert card, "the map card's element is missing"
    for attr in ('role="dialog"', 'aria-labelledby="mc-title"', 'tabindex="-1"', "hidden"):
        assert attr in card.group(0)
    assert "initMapCard()" in _read("app.js")


def test_a_click_fills_the_panel_without_unfolding_it_and_a_tab_link_unfolds_it() -> None:
    station = (SRC / "panel-station.js").read_text(encoding="utf-8")
    point = (SRC / "panel-point.js").read_text(encoding="utf-8")
    assert 'showSurface("panel-station", { reveal: Boolean(tab) })' in station
    assert 'showSurface("panel-point", { reveal: Boolean(tab) })' in point
    # The tab is in the address only while the panel shows it, so a link with tab= still opens it there.
    url = (SRC / "url.js").read_text(encoding="utf-8")
    assert 'state.activeTab && state.panelOpen !== false' in url
    shell = (SRC / "shell.js").read_text(encoding="utf-8")
    assert '"aq:surface"' in shell and '"aq:panel"' in shell


def test_the_card_hears_the_record_and_the_river_from_the_panels() -> None:
    card = (SRC / "map-card.js").read_text(encoding="utf-8")
    for event in ('"analysis"', '"riversnap"', '"reachchange"', '"aq:surface"', '"aq:panel"'):
        assert event in card, event
    assert 'dispatchEvent(new CustomEvent("analysis"' in (SRC / "panel-station.js").read_text(encoding="utf-8")
    river = (SRC / "river.js").read_text(encoding="utf-8")
    assert 'new CustomEvent("riversnap"' in river
    assert "export function traceRiver" in river
    # Other layers open a card through the shared actions (a flood cell, a warning reach).
    assert "actions.openMapCard = openCard" in card


def test_the_card_respects_reduced_motion_and_has_a_phone_sheet() -> None:
    css = _read("style.css")
    assert ".map-card.sheet" in css
    block = css[css.index("/* ── the map card (#548)"):]
    assert "@media (prefers-reduced-motion: reduce)" in block
    assert "(max-width: 640px)" in (SRC / "map-card.js").read_text(encoding="utf-8")
