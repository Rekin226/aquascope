"""The forecast track record on the map (#785): the card's words are aquascope.forecast_skill's, word for word.

``explorer/src/track-record-core.js`` says the rows of ``verify/forecast_skill_gauges.parquet`` on the river card,
on a Floods ahead reach and in the evidence ladder. The rows are scored in Python
(:mod:`aquascope.archive.verify`), so the page writes no number of its own; this holds its sentences and bands to
:mod:`aquascope.forecast_skill` and :data:`aquascope.archive.verify.TRUST_BANDS`, and checks the wiring.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aquascope import forecast_skill as fs
from aquascope.archive import verify as vf

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"
CORE = EXPLORER / "src" / "track-record-core.js"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _js(expr: str) -> Any:
    script = f"const m = await import({json.dumps(CORE.as_uri())});\nconsole.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


def _rows() -> list[dict]:
    rows = []
    for i, (model, variant, factor, n) in enumerate([
        ("geoglows", "raw", 4.85, 84), ("geoglows", "corrected", 1.95, 84), ("glofas", "raw", 2.25, 41),
        ("glofas", "corrected", 1.04, 29), ("geoglows", "raw", 1.0, 1), ("glofas", "raw", 12.349, 200),
    ]):
        enough = n >= vf.MIN_PAIRS
        rows.append({"source": "usgs", "station_id": f"S{i // 4}", "model": model, "variant": variant, "lead_min": 1,
                     "lead_max": 3, "window_days": 30, "n_pairs": n, "first_valid": "2026-10-09",
                     "last_valid": "2026-11-07", "enough_pairs": enough, "factor": factor,
                     "trust": vf.trust(factor, enough)})
    return rows


@needs_node
def test_the_card_lines_are_the_package_lines():
    rows = _rows()
    for variant in ("corrected", "raw"):
        py = fs.track_lines(rows[:4], variant=variant)
        js = _js(f"m.trackLines({json.dumps(rows[:4])}, {{ variant: {json.dumps(variant)} }})")
        assert js == py
    assert _js(f"{json.dumps(rows)}.map((r) => m.trackLine(r))") == [fs.track_line(r) for r in rows]
    assert _js("m.trackHead()") == fs.track_head()


@needs_node
def test_the_ranking_items_are_the_package_lines():
    rank = fs.rank_models(_rows()[:4])
    items = _js(f"m.rankingRows({json.dumps(rank)})")
    assert [i["text"] for i in items] == [fs.track_line({**r, "enough_pairs": True}) for r in rank["ranked"]] + [
        fs.track_line({**r, "enough_pairs": False}) for r in rank["not_enough"]]


@needs_node
def test_the_bands_and_the_floor_are_the_package_ones():
    bands = _js("m.TRUST_BANDS")
    assert [(b["trust"], b["maxFactor"], b["words"]) for b in bands] == [
        (b["trust"], b["max_factor"], b["words"]) for b in vf.TRUST_BANDS]
    assert _js("[m.MIN_PAIRS, m.TRACK_DAYS, m.TRACK_LEADS]") == [vf.MIN_PAIRS, vf.TRACK_DAYS, list(vf.TRACK_LEADS)]
    assert _js("m.windowSql('u')").endswith(f"lead_min = {vf.TRACK_LEADS[0]} AND lead_max = {vf.TRACK_LEADS[1]}")


def test_the_page_reads_the_published_verify_prefix_and_shows_nothing_without_it():
    loader = (EXPLORER / "src" / "track-record.js").read_text(encoding="utf-8")
    assert "summary.json" in loader and "forecast_skill_gauges.parquet" in loader
    assert "if (!res.ok) return false;" in loader            # verify/ missing: no line, no badge, no legend row
    assert "shown: () => Boolean(summary)" in loader
    config = (EXPLORER / "config.js").read_text(encoding="utf-8")
    assert f'resolve/main/{vf.FOLDER}/"' in config
    assert vf.GAUGE_TABLE == "forecast_skill_gauges.parquet"


def test_the_card_floods_ahead_and_the_evidence_tab_are_wired():
    card = (EXPLORER / "src" / "map-card.js").read_text(encoding="utf-8")
    assert "trackForGauge(m.key)" in card and "trackForReach(reachId)" in card and "${trustHtml(s.trust)}" in card
    fews = (EXPLORER / "src" / "fews.js").read_text(encoding="utf-8")
    assert "trustForReach(p.river_id)" in fews
    worker = (EXPLORER / "worker.js").read_text(encoding="utf-8")
    assert 'verified=_k.get("verified") or []' in worker
    assert 'id="ev-forecasts"' in (EXPLORER / "index.html").read_text(encoding="utf-8")
    app = (EXPLORER / "app.js").read_text(encoding="utf-8")
    assert "initTrackRecord();" in app
