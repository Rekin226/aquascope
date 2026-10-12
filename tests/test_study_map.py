"""The study on the map: GeoJSON features per step, the aggregate, the bundle file and the faces."""

from __future__ import annotations

import json

from aquascope import study_map as sm
from aquascope.study import Step, Study, StudyRun, write_outputs

SITE = {"lat": 51.415, "lon": -0.308}
CATCH = {"latitude": 51.415, "longitude": -0.308, "sub_basin": {"hybas_id": 2120008490, "up_area": 9948.2},
         "upstream": {"n_sub_basins": 40}, "attributes": {}}
FLOW = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "unit": "m3/s"}
ANYWHERE = {"latitude": 51.415, "longitude": -0.308, "climate": {"precipitation_mm_per_year": 700.0},
            "glofas": {"stats": {"mean": 60.0}}}
SIMILAR = {"method": "combined", "k": 2, "features_used": ["area"], "latitude": 51.4, "longitude": -0.3,
           "sub_basin": {"hybas_id": 1},
           "stations": [{"source": "uk_ea", "station_id": "D1", "name": "Donor one", "latitude": 52.0,
                         "longitude": -1.0, "score": 0.9, "features": {"x": 1}},
                        {"source": "uk_ea", "station_id": "D2", "latitude": None, "longitude": None}]}


def _roles(features):
    return [f["properties"]["role"] for f in features]


def test_a_gauge_step_is_placed_from_the_station_lookup():
    feats = sm.step_features("s2", "analyze_station", FLOW, stations={("uk_ea", "3400TH"): (51.41, -0.31)})
    assert _roles(feats) == ["gauge"]
    f = feats[0]
    assert f["geometry"] == {"type": "Point", "coordinates": [-0.31, 51.41]}
    assert f["properties"] == {"role": "gauge", "label": "Kingston (uk_ea 3400TH)", "step_id": "s2",
                               "source": "uk_ea", "station_id": "3400TH"}
    assert sm.step_features("s2", "analyze_station", FLOW) == [], "no coordinates, no feature"
    assert sm.step_features("s2", "analyze_station", {"error": "x", **FLOW}) == []
    assert sm.step_features("s2", "x", None) == [] and sm.step_features("s2", "x", [1, 2]) == []


def test_the_catchment_is_its_polygon_when_given_else_its_outlet():
    outlet = sm.step_features("s1", "describe_catchment", CATCH)
    assert _roles(outlet) == ["catchment"]
    assert outlet[0]["geometry"]["type"] == "Point"
    assert outlet[0]["properties"]["label"] == "Catchment, 9,948 km2 upstream (outlet)"
    assert outlet[0]["properties"]["hybas_id"] == 2120008490 and outlet[0]["properties"]["outline"] is False
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    drawn = sm.step_features("s1", "describe_catchment", {**CATCH, "geometry": poly})
    assert drawn[0]["geometry"] == poly and drawn[0]["properties"]["outline"] is True
    # a similarity result carries the sub-basin too, but it describes no catchment
    assert "catchment" not in _roles(sm.step_features("s5", "similar_basins", SIMILAR))


def test_the_grid_cells_are_boxes_snapped_to_their_grids_and_marked_approximate():
    feats = sm.step_features("s4", "anywhere", ANYWHERE)
    assert _roles(feats) == ["grid_cell", "grid_cell"]
    era5, glofas = feats
    assert era5["properties"]["model"] == "era5" and era5["properties"]["approximate"] is True
    ring = era5["geometry"]["coordinates"][0]
    assert ring[0] == [-0.375, 51.375] and ring[2] == [-0.125, 51.625], "0.25-degree cell centred on 51.5, -0.25"
    ring = glofas["geometry"]["coordinates"][0]
    assert ring[0] == [-0.35, 51.4] and ring[2] == [-0.3, 51.45], "0.05-degree cell centred on 51.425, -0.325"
    assert "approximate" in glofas["properties"]["label"]
    # a cell the source reported is used as reported, and not marked approximate
    told = sm.step_features("s4", "anywhere", {**ANYWHERE, "cells": {"glofas": {"latitude": 51.475,
                                                                               "longitude": -0.275}}})
    g = [f for f in told if f["properties"]["model"] == "glofas"][0]
    assert g["properties"]["approximate"] is False and "approximate" not in g["properties"]["label"]
    assert g["geometry"]["coordinates"][0][0] == [-0.3, 51.45]


def test_donors_are_points_with_their_ids_and_unplaced_donors_are_skipped():
    feats = sm.step_features("s5", "similar_basins", SIMILAR)
    assert _roles(feats) == ["donor"]
    p = feats[0]["properties"]
    assert p["label"] == "Donor one (uk_ea D1)" and p["station_id"] == "D1" and p["score"] == 0.9
    assert "features" not in p, "only scalar properties"
    listed = sm.step_features("s0", "find_stations", {"stations": [{"source": "a", "station_id": "1",
                                                                     "lat": 1, "lon": 2}] * 80})
    assert _roles(listed) == ["station"] * sm.MAX_POINTS
    reg = sm.step_features("s6", "regionalize_signatures", {"similarity": {"donors": SIMILAR["stations"]}})
    assert _roles(reg) == ["donor"]


def _results():
    return [
        {"id": "s1", "tool": "describe_catchment", "ok": True, "result": CATCH},
        {"id": "s2", "tool": "analyze_station", "ok": True, "result": FLOW},
        {"id": "s3", "tool": "flood_frequency", "ok": False, "result": {"error": "short"},
         "fallback": {"tool": "anywhere", "ok": True, "result": ANYWHERE}},
        {"id": "s0", "tool": "assess_site", "ok": True,
         "result": {"point": SITE, "stations": [{"source": "uk_ea", "station_id": "3400TH", "latitude": 51.41,
                                                  "longitude": -0.31}]}},
    ]


def test_the_study_map_is_the_site_then_every_step_and_the_run_lists_the_gauges_it_needs():
    fc = sm.run_features(_results(), site=SITE)
    assert fc["type"] == "FeatureCollection"
    assert _roles(fc["features"]) == ["site", "catchment", "gauge", "grid_cell", "grid_cell", "station"]
    assert fc["features"][0]["properties"]["step_id"] is None
    fb = [f for f in fc["features"] if f["properties"]["step_id"] == "s3"]
    assert fb and all(f["properties"]["fallback"] is True for f in fb), "the fallback's features ride on its step"
    w, s, e, n = fc["bbox"]
    assert w <= -0.375 and e >= -0.125 and s <= 51.375 and n >= 51.625
    json.dumps(fc)  # JSON-safe
    assert sm.run_features([], site=None) == {"type": "FeatureCollection", "features": []}


def test_attach_puts_each_steps_own_features_on_its_record():
    results = _results()
    results[1]["map"] = {"stale": True}
    results.append({"id": "s9", "tool": "x", "ok": True, "result": {"n": 1}, "map": {"stale": True}})
    sm.attach(results, site=SITE)
    assert _roles(results[0]["map"]["features"]) == ["catchment"]
    assert _roles(results[1]["map"]["features"]) == ["gauge"]
    assert "map" not in results[4], "a step with nothing to place carries no map"


def test_write_outputs_writes_the_geojson_next_to_the_results(tmp_path):
    study = Study(question="q", steps=[Step(tool="describe_catchment", id="s1")], version=2,
                  problem={"kind": "flood_risk", "site": SITE})
    full = {"arguments": {}, "gates": [], "rationale": None, "error": None}
    run = StudyRun(study=study, results=[{**full, **_results()[0]}])
    paths = write_outputs(run, tmp_path)
    fc = json.loads((tmp_path / "study_map.geojson").read_text(encoding="utf-8"))
    assert paths["study_map.geojson"].endswith("study_map.geojson")
    assert _roles(fc["features"]) == ["site", "catchment"]
    empty = StudyRun(study=study, results=[{**full, "id": "s1", "tool": "x", "ok": True, "result": {"n": 1}}])
    assert "study_map.geojson" not in write_outputs(empty, tmp_path / "none")


def test_publish_makes_one_artifact_and_hands_it_to_the_face_only_when_it_changed():
    from aquascope.studio.workspace import Dataset, Inventory, Workspace

    ws = Workspace()
    ws.site = dict(SITE)
    ws.inventory = Inventory(site=dict(SITE), datasets=[
        Dataset(id="uk_ea:3400TH", kind="station", source="uk_ea", station_id="3400TH", lat=51.41, lon=-0.31)])
    seen: list = []
    results = _results()[:2]
    fc = sm.publish(ws, results, seen.append)
    assert _roles(fc["features"]) == ["site", "catchment", "gauge"], "the gauge is placed from the inventory"
    art = ws.artifact(sm.ARTIFACT_ID)
    assert art is not None and art.name == "study_map.geojson" and art.media_type == "application/geo+json"
    assert art.kind == "data" and json.loads(art.data)["features"][2]["properties"]["station_id"] == "3400TH"
    assert [a.id for a in seen] == ["study-map"] and results[1]["map"]["features"][0]["properties"]["role"] == "gauge"
    sm.publish(ws, results, seen.append)
    assert len(seen) == 1, "an unchanged map is not sent again"
    sm.publish(ws, _results()[:3], seen.append)
    assert len(seen) == 2 and len([a for a in ws.artifacts if a.id == "study-map"]) == 1
    # the workspace view every face can ask for
    ws.run = {"results": results}
    assert _roles(sm.workspace_features(ws.to_dict())["features"]) == ["site", "catchment", "gauge"]
    # nothing to place: no artifact
    bare = Workspace()
    bare.site = dict(SITE)
    sm.publish(bare, [{"id": "s1", "tool": "x", "ok": True, "result": {"n": 1}}], seen.append)
    assert bare.artifact(sm.ARTIFACT_ID) is None and len(seen) == 2


def test_a_callback_that_raises_does_not_stop_the_study():
    from aquascope.studio.workspace import Workspace

    ws = Workspace()
    ws.site = dict(SITE)

    def boom(_art):
        raise RuntimeError("the face fell over")

    sm.publish(ws, _results()[:1], boom)
    assert ws.artifact(sm.ARTIFACT_ID) is not None
