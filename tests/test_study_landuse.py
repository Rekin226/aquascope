"""The Study tool uses the sketchpad's TR-55 calculation for the client's scenario."""

from __future__ import annotations

import pytest

from aquascope.methods import SiteContext, assess_method
from aquascope.sketch import landuse
from aquascope.studio import catalogue
from aquascope.study import Step, Study, run_study

FIELD = [[4.955, 45.70], [4.97, 45.70], [4.97, 45.708], [4.955, 45.708]]
SCENARIO = {"polygon": FIELD, "use": "paved", "rain_mm": 100, "cover": {"10": 1},
            "soil_group": "C", "slope_pct": 3}


def test_study_runs_the_checked_sketch_without_site_lookups(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The supplied baseline and scenario need no network lookup")

    for name in ("cover_fractions", "soil_groups", "land_slope"):
        monkeypatch.setattr(landuse, name, unexpected)
    monkeypatch.setattr("aquascope.rivers.snap_to_river", unexpected)
    expected = landuse.landuse_change(**SCENARIO, outlet=False)
    entry = catalogue.get("landuse_change")
    assert entry.methods == ["landuse_change"] and entry.tables == entry.figures == ["landuse_runoff"]
    assert entry.arguments["cover"]["type"] == "object"
    assert [s["type"] for s in entry.arguments["polygon"]["anyOf"]] == ["array", "object"]
    assert entry.compact()["arguments"]["polygon"] == "array or object"
    assert entry.arguments["use"]["enum"] == list(landuse.USES)
    assert set(entry.required) == {"polygon", "use", "rain_mm"}
    study = Study("What if the field were paved?", version=3, steps=[
        Step(id="runoff", tool=entry.id, arguments=SCENARIO, expects=entry.gates),
    ])
    run = run_study(study)
    assert run.ok and run.results[0]["gates_passed"]
    result = run.results[0]["result"]
    for key in ("before", "after", "change", "assumptions", "licences", "check"):
        assert result[key] == expected[key]
    assert result["grade"] == "screening" and result["change"]["q_mm"] > 0
    assert result["reference_check"]["ok"] and len(result["reference_check"]["rows"]) == 6


def test_site_verdict_and_plan_need_the_actual_scenario():
    assert assess_method("landuse_change", SiteContext())["status"] == "not_defensible"
    verdict = assess_method("landuse_change", SiteContext(available={"land-use scenario"}))
    assert verdict["status"] == "defensible"
    evidence = [{**verdict, "scenario": SCENARIO}]
    entry = catalogue.get("landuse_change")
    step = {"id": "runoff", "tool": entry.id, "method": entry.id, "arguments": SCENARIO, "expects": entry.gates}
    assert catalogue.validate_step(step, sufficiency=evidence) == []
    invented = {**step, "arguments": {**SCENARIO, "rain_mm": 110}}
    assert "supplied by the user" in " ".join(catalogue.validate_step(invented, sufficiency=evidence))
    unverified = {**step, "expects": [{"check": "status_is", "path": "reference_check.ok", "value": False}]}
    assert "requires the gate" in " ".join(catalogue.validate_step(unverified, sufficiency=evidence))


@pytest.mark.parametrize(("override", "why"), [
    ({"polygon": None}, "polygon"), ({"polygon": [[1, 2], [2, 3]]}, "polygon"),
    ({"use": "impervious"}, "impervious_pct"), ({"use": "city"}, "use"),
    ({"rain_mm": None}, "rain_mm"), ({"rain_mm": -1}, "rain_mm"),
    ({"soil_group": "Q"}, "soil_group"), ({"cover": {"80": 1}}, "TR-55"),
    ({"tc_hours": 24}, "tc_hours"), ({"rain_mm": float("nan")}, "rain_mm"),
    ({"use": {}}, "use"), ({"rain_mm": True}, "rain_mm"), ({"extra": None}, "extra"),
    ({"polygon": [[1, 2], [2, 3], [3, 4]]}, "polygon"),
    ({"polygon": [[1, 86], [2, 3], [3, 5]]}, "polygon"),
])
def test_unsuitable_scenarios_are_explained_before_execution(override, why):
    args = {**SCENARIO, **override}
    assert why in " ".join(catalogue.validate_step({"tool": "landuse_change", "arguments": args}))
    with pytest.raises(ValueError, match=why):
        landuse.study_landuse_change(**args)


def test_a_missing_worldcover_baseline_is_read_or_the_step_fails(monkeypatch):
    def unavailable(ring, **kwargs):
        raise OSError("WorldCover could not be read")

    monkeypatch.setattr(landuse, "cover_fractions", unavailable)
    args = {k: v for k, v in SCENARIO.items() if k != "cover"}
    run = run_study(Study("Runoff?", steps=[Step(tool="landuse_change", arguments=args)]))
    assert not run.ok and "WorldCover could not be read" in run.results[0]["error"]


def test_geojson_polygon_and_failed_reference_check(monkeypatch):
    args = {**SCENARIO, "polygon": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [FIELD]}}}
    monkeypatch.setattr(landuse, "check", lambda: {"ok": False, "rows": []})
    entry = catalogue.get("landuse_change")
    run = run_study(Study("Runoff?", version=3, steps=[Step(tool=entry.id, arguments=args, expects=entry.gates)]))
    row = run.results[0]
    assert not row["gates_passed"] and "failed" in row["result"]["check"]
