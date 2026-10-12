"""The Scout: datasets from the reconnaissance, the ERA5 cell, uploads through the ingest QA, no network."""

from __future__ import annotations

import aquascope.explore
from aquascope.studio.roles import scout
from aquascope.studio.workspace import Dataset, Workspace
from tests.test_studio.conftest import RICH, SAMPLES_CSV, SERIES_CSV, UNGAUGED, patched


def _ws(**tables) -> Workspace:
    ws = Workspace()
    ws.site = {"lat": 51.415, "lon": -0.308}
    ws.brief.problem, ws.brief.playbook = "drought", "drought_status"
    for k, v in tables.items():
        ws.add_table(k, v)
    return ws


def test_the_inventory_lists_every_station_variable_the_catchment_the_donors_and_era5():
    ws = _ws()
    with patched(RICH):
        inv = scout.scout(ws)
    assert ws.inventory is inv and inv.site == {"lat": 51.415, "lon": -0.308}
    ids = [d.id for d in inv.datasets]
    assert ids[:4] == ["uk_ea:3400TH:discharge", "uk_ea:3400TH:water_level", "uk_ea:R1:precipitation",
                       "uk_ea:W1:groundwater_level"]
    assert "catchment" in ids and "donors" in ids and ids[-1] == "era5"
    gauge = inv.dataset("uk_ea:3400TH:discharge")
    assert gauge.kind == "station" and gauge.years == 39.5 and gauge.distance_km == 0.4 and gauge.resolution == "daily"
    assert inv.dataset("era5").kind == "reanalysis" and inv.dataset("era5").years > 80
    assert inv.dataset("donors").n == 8 and inv.years_by_variable["precipitation"] == 36.0
    assert inv.recon is not None and inv.notes == ["served record starts 1986"]
    assert [e["event"] for e in ws.events if e["role"] == "scout"] == ["recon", "inventory"]


def test_an_ungauged_site_still_has_the_reanalysis_and_survives_a_failed_recon(monkeypatch):
    ws = _ws()
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    assert [d.kind for d in inv.datasets] == ["catchment", "donors", "reanalysis"]

    def boom(*a, **k):
        raise RuntimeError("catalog offline")

    monkeypatch.setattr(aquascope.explore, "assess_site", boom, raising=False)
    ws2 = _ws()
    inv2 = scout.scout(ws2)
    assert [d.kind for d in inv2.datasets] == ["reanalysis"]
    assert any("reconnaissance unavailable" in n for n in inv2.notes)
    assert any(e["event"] == "error" for e in ws2.events)


def test_uploads_go_through_the_ingest_qa():
    ws = _ws(**{"upload:flows.csv": SERIES_CSV, "upload:samples.csv": SAMPLES_CSV})
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    flows = inv.dataset("upload:flows.csv")
    assert flows.kind == "upload" and flows.variable == "discharge" and flows.n > 1000 and flows.years > 10
    assert flows.quality["verdict"] in ("usable", "check") and flows.quality["mapping"]["value_column"] == "flow_m3s"
    assert flows.quality["qa"]["coverage_pct"] > 0 and flows.quality["columns"] == ["date", "flow_m3s"]
    samples = inv.dataset("upload:samples.csv")
    assert samples.kind == "upload" and samples.variable is None and samples.n == 3
    assert samples.quality["verdict"] == "table" and samples.quality["columns"] == ["station", "parameter", "value",
                                                                                      "unit"]
    assert len(inv.uploads()) == 2 and any("upload:flows.csv" in n for n in inv.notes)
    assert ws.to_dict()["inventory"]["datasets"][-1]["id"] == "upload:samples.csv"


def test_the_reach_the_site_snaps_to_is_listed_as_modelled(monkeypatch):
    from aquascope import rivers

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: {
        "snapped": True, "river_id": 230399750, "distance_m": 18.0, "snap_lat": 51.4151, "snap_lon": -0.3085,
        "message": "Snapped 18 m to river reach 230399750, Strahler order 6."})
    ws = _ws()
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    reach = inv.dataset("geoglows_reach")
    assert reach.kind == "modelled" and reach.variable == "discharge" and reach.station_id == "230399750"
    assert reach.start == "1940-01-01" and reach.years > 80 and reach.distance_km == 0.018
    assert reach.note.startswith("MODELLED, not measured") and reach.quality["licence"] == "CC BY 4.0"
    assert [d.kind for d in inv.datasets] == ["catchment", "donors", "reanalysis", "modelled"]


def test_no_stream_near_the_site_is_a_note_and_no_network_is_silent(monkeypatch):
    from aquascope import rivers

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: {
        "snapped": False, "message": "No stream within 1,000 m of this point."})
    ws = _ws()
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    assert "geoglows_reach" not in [d.id for d in inv.datasets]
    assert "GEOGLOWS river network: No stream within 1,000 m of this point." in inv.notes


def test_an_unreadable_network_leaves_the_reach_out_quietly():
    # tests/conftest.py takes GEOGLOWS off the network, as an outage would
    assert scout.reach_dataset(51.4, -0.3) == (None, None)


def test_the_place_context_layers_are_listed_with_their_licences():
    ws = _ws()
    with patched(RICH):
        inv = scout.scout(ws)
    layers = [c["layer"] for c in inv.context]
    assert layers == ["flood_history", "surface_water", "flood_hazard", "dams", "rain_gauge", "actual_et", "soil"]
    floods = inv.context[0]
    assert floods["sources"] == ["Google Research", "Microsoft AI for Good Lab"]
    assert floods["licences"] == ["CC-BY-4.0", "MIT"] and "summary" not in floods  # a drought question reads none
    # listed beside the datasets, not counted among them
    assert all(d.kind != "context" for d in inv.datasets)
    assert ws.to_dict()["inventory"]["context"][3]["layer"] == "dams"


def test_a_flood_question_reads_the_flood_history(monkeypatch):
    seen = []

    def fake(lat, lon):
        seen.append((lat, lon))
        return {"summary": "3 flood events in the news within 25 km, latest 2024-01-05.",
                "news": {"available": True, "n_events": 3, "latest": "2024-01-05"}}

    monkeypatch.setattr(scout, "_read_flood_history", fake)
    ws = _ws()
    ws.brief.problem, ws.brief.playbook = "flood_risk", "flood_risk"
    with patched(RICH):
        inv = scout.scout(ws)
    assert seen == [(51.415, -0.308)]
    floods = inv.context[0]
    assert floods["events"] == 3 and floods["latest"] == "2024-01-05" and floods["summary"].startswith("3 flood")
    assert "Flood history: 3 flood events in the news within 25 km, latest 2024-01-05." in inv.notes
    assert any(e["event"] == "context" for e in ws.events if e["role"] == "scout")


def test_a_failed_flood_history_read_never_stops_the_scout(monkeypatch):
    def boom(lat, lon):
        raise RuntimeError("Archive offline")

    monkeypatch.setattr(scout, "_read_flood_history", boom)
    ws = _ws()
    ws.brief.playbook = "flood_change"
    with patched(RICH):
        inv = scout.scout(ws)
    assert "summary" not in inv.context[0]
    assert any("flood history unreadable" in e["detail"] for e in ws.events)


def test_the_flood_playbook_caveat_follows_the_events_the_scout_found(monkeypatch):
    from aquascope import playbooks as pbk

    pb = pbk.load("flood_risk")
    intake = pbk.fill_intake(pb, {})
    flagged = "Historical flood events near the site"
    monkeypatch.setattr(scout, "_read_flood_history", lambda lat, lon: {
        "summary": "2 flood events.", "radius_km": 25.0, "news": {"available": True, "n_events": 2,
                                                                   "latest": "2020-01-01"}})
    ws = _ws()
    ws.brief.playbook = "flood_risk"
    with patched(RICH):
        inv = scout.scout(ws)
    assert inv.recon["context"]["flood_history"] == {"events": 2, "latest": "2020-01-01", "radius_km": 25.0}
    with_events = pbk.caveats_for(pb, pbk.evaluation_context(pb, inv.recon, intake))
    assert any(flagged in c for c in with_events)
    # the reconnaissance alone (what HydroGym scores against) never carries it
    without = pbk.caveats_for(pb, pbk.evaluation_context(pb, RICH, intake))
    assert not any(flagged in c for c in without)


UPSTREAM = {"available": True, "regulated": True, "n_dams": 7, "total_capacity_mcm": 142.0,
            "degree_of_regulation_pct": 3.8, "dams": [{"name": "Spitallamm"}, {"name": "Raterichsboden"}],
            "summary": "Regulated upstream: 7 dams in Global Dam Watch drain to this reach, holding about 142 "
                       "million m3, about 3.8 % of a year's mean flow here (GEOGLOWS, modelled)."}


def _with_reach(monkeypatch, river_id="230260670"):
    reach = Dataset(id="geoglows_reach", kind="modelled", variable="discharge", source="GEOGLOWS v2",
                    station_id=river_id, name=f"GEOGLOWS river reach {river_id}")
    monkeypatch.setattr(scout, "reach_dataset", lambda lat, lon: (reach, None))


def test_a_supply_question_reads_the_dams_upstream_as_context(monkeypatch):
    seen = []

    def fake(lat, lon, river_id=None):
        seen.append((lat, lon, river_id))
        return dict(UPSTREAM)

    monkeypatch.setattr(scout, "_read_upstream_dams", fake)
    _with_reach(monkeypatch)
    ws = _ws()
    ws.brief.problem, ws.brief.playbook = "supply", "supply_reliability"
    with patched(RICH):
        inv = scout.scout(ws)
    assert seen == [(51.415, -0.308, "230260670")]  # the reach the site snapped to
    dams = next(c for c in inv.context if c["layer"] == "dams")
    assert dams["regulated_upstream"] is True and dams["upstream"].startswith("Regulated upstream: 7 dams")
    assert inv.recon["context"]["upstream_dams"]["degree_of_regulation_pct"] == 3.8
    assert inv.recon["context"]["upstream_dams"]["largest"] == ["Spitallamm", "Raterichsboden"]
    assert any(n.startswith("Dams upstream: Regulated upstream") for n in inv.notes)
    # the reconnaissance the caller passed in (what HydroGym scores against) is left alone
    assert "upstream_dams" not in (RICH.get("context") or {})


def test_the_dams_upstream_are_read_for_a_flood_question_but_not_a_drought_one(monkeypatch):
    calls = []
    monkeypatch.setattr(scout, "_read_upstream_dams", lambda lat, lon, river_id=None: calls.append(1) or {
        "available": False, "regulated": None, "summary": "Dams upstream are not available yet."})
    _with_reach(monkeypatch)
    ws = _ws()
    ws.brief.playbook = "flood_change"
    with patched(RICH):
        inv = scout.scout(ws)
    assert calls == [1] and "upstream_dams" not in (inv.recon.get("context") or {})
    assert "Dams upstream: Dams upstream are not available yet." in inv.notes
    with patched(RICH):
        scout.scout(_ws())  # drought_status
    assert calls == [1]


def test_a_failed_dams_upstream_read_never_stops_the_scout(monkeypatch):
    def boom(lat, lon, river_id=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(scout, "_read_upstream_dams", boom)
    _with_reach(monkeypatch)
    ws = _ws()
    ws.brief.playbook = "supply_reliability"
    with patched(RICH):
        inv = scout.scout(ws)
    assert "upstream" not in next(c for c in inv.context if c["layer"] == "dams")
    assert any("dams upstream unreadable" in e["detail"] for e in ws.events)


def test_no_reach_means_no_dams_upstream_to_read(monkeypatch):
    calls = []
    monkeypatch.setattr(scout, "_read_upstream_dams", lambda lat, lon, river_id=None: calls.append(1) or dict(UPSTREAM))
    monkeypatch.setattr(scout, "reach_dataset", lambda lat, lon: (None, "GEOGLOWS river network: no stream near"))
    ws = _ws()
    ws.brief.playbook = "flood_risk"
    with patched(RICH):
        inv = scout.scout(ws)
    assert calls == [] and not any(n.startswith("Dams upstream") for n in inv.notes)
# ── which global model to lean on near the site (#518) ──────────────────────

LEAN = {"model": "grrr", "label": "Google GRRR", "median_kge": 0.61, "n_gauges": 3,
        "sentence": "Near this site, Google GRRR tracked the gauges best: median KGE 0.61 at 3 gauges within 80 km. "
                    "Lean on Google GRRR's numbers here and read the others as a second opinion."}


def test_the_scout_reads_which_model_to_lean_on_and_notes_it(monkeypatch):
    monkeypatch.setattr(scout, "_read_model_skill", lambda lat, lon: LEAN)
    ws = _ws()
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    assert inv.models == LEAN
    assert any(n.startswith("Model skill near the site: Near this site, Google GRRR") for n in inv.notes)
    again = type(inv).from_dict(inv.to_dict())
    assert again.models == LEAN


def test_no_published_skill_leaves_the_inventory_as_it_was(monkeypatch):
    def offline(lat, lon):
        raise RuntimeError("Hub offline")

    monkeypatch.setattr(scout, "_read_model_skill", offline)
    assert scout.model_evidence(1.0, 2.0) is None
    monkeypatch.setattr(scout, "_read_model_skill", lambda lat, lon: {"model": None, "sentence": "No graded gauge."})
    ws = _ws()
    with patched(UNGAUGED):
        inv = scout.scout(ws)
    assert inv.models == {"model": None, "sentence": "No graded gauge."}
    assert not any(n.startswith("Model skill near the site") for n in inv.notes)
    assert "models" in inv.to_dict()
