"""The study map in the page: the hooks studio.js, map.js and the worker carry for it."""

from __future__ import annotations

from pathlib import Path

EXPLORER = Path(__file__).resolve().parents[2] / "explorer"


def _read(*parts: str) -> str:
    return (EXPLORER.joinpath(*parts)).read_text(encoding="utf-8")


def test_the_study_draws_on_the_map_through_its_own_module():
    studio = _read("src", "studio.js")
    for hook in ('from "./study-map.js?v=__BUILD__"', "studyMapArtifact(artifact)", "clearStudyMap()",
                 "showStudyMapFor(S.ws)", "focusStudyStep(onMap.dataset.mapStep)", "data-map-step",
                 "stepsOnMapHtml(S.ws, toolLabel)", "if (drawnOnMap(f)) return"):
        assert hook in studio, hook
    mod = _read("src", "study-map.js")
    assert 'const SRC = "study-map"' in mod and "fitBoundsTo" in mod and "map.flyTo" in mod
    # the module owns its layers and names them study-*, which map.js carries across a basemap change
    assert "studyLayers(SRC)" in mod and "highlightFilters(id)" in mod
    specs = _read("src", "study-map-data.js")
    for layer in ("study-fill", "study-line", "study-points", "study-hl-line", "study-hl-points"):
        assert f'id: "{layer}"' in specs, layer
    m = _read("src", "map.js")
    assert 'id.startsWith("study-")' in m
    # the engine places the features; the page never computes a place
    data = _read("src", "study-map-data.js")
    assert "features.push(...fc.features" in data and "0.25" not in data and "0.05" not in data


def test_the_worker_sends_the_study_map_with_its_bytes():
    worker = _read("worker.js")
    assert 'with_data=art.media_type == "image/png" or art.id == "study-map"' in worker


def test_the_study_map_surface_writes_with_plain_hyphens():
    for parts in (("src", "study-map.js"), ("src", "study-map-data.js")):
        text = _read(*parts)
        assert "—" not in text and "–" not in text, parts
