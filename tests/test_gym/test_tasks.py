"""HydroGym Phase 1 tasks (#175): the tree's verdict on three fixture sites is the key, probes make unsolvable
tasks, sites split deterministically, and the suite round-trips through JSONL."""

from __future__ import annotations

from datetime import date

import pytest

from aquascope.ai_engine.team import choose_playbook, intake_hints
from aquascope.gym import tasks as gt

STATION = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "distance_km": 0.4,
           "variables": ["discharge", "water_level"], "period_start": "1883-10-01", "years": 39.5}


def recon(lat, lon, years=None, stations=None, donors=None, area=None, dams=0):
    years = years or {}
    ctx = {"years_by_variable": dict(years), "resolution_by_variable": {k: "daily" for k in years},
           "area_km2": area, "donors": donors, "ungauged": not years,
           "available": ["glofas", "temperature", "forcing"]}  # what assess_site assumes for any point on land
    return {"point": {"lat": lat, "lon": lon}, "stations": list(stations or []),
            "catchment": {"area_km2": area, "upstream_area_km2": area, "dams": dams},
            "context": ctx, "sufficiency": [], "notes": []}


LONG = {"lat": 51.415, "lon": -0.308, "source": "uk_ea", "station_id": "3400TH", "name": "Kingston",
        "kind": "gauged_long"}
SHORT = {"lat": 46.9, "lon": -110.2, "source": "usgs", "station_id": "06120500", "name": "Musselshell",
         "kind": "gauged_short"}
POINT = {"lat": 40.1234, "lon": -3.4567, "source": None, "station_id": None, "name": None, "kind": "ungauged"}


def fake_recon(lat, lon, *, point_donors=5):
    if (lat, lon) == (LONG["lat"], LONG["lon"]):
        return recon(lat, lon, {"discharge": 39.5}, [STATION], donors=8, area=9948, dams=1)
    if (lat, lon) == (SHORT["lat"], SHORT["lon"]):
        return recon(lat, lon, {"discharge": 12}, [dict(STATION, **SHORT, years=12)], donors=8, area=300)
    return recon(lat, lon, None, [], donors=point_donors, area=120)


def test_tasks_from_three_sites_and_two_playbooks_carry_the_expected_branches():
    events = []
    tasks = gt.tasks_from_playbooks([LONG, SHORT, POINT], ["flood_risk", "ungauged_flow"], recon=fake_recon,
                                    probes=0, on_event=events.append)
    key = {(t.site.get("station_id") or "point", t.playbook): t for t in tasks}
    assert len(tasks) == 6 and not any(t.unsolvable for t in tasks)
    assert key[("3400TH", "flood_risk")].expected["branch"] == "at_site"
    assert key[("06120500", "flood_risk")].expected["branch"] == "short_record"
    assert key[("point", "flood_risk")].expected["branch"] == "regional"
    assert key[("3400TH", "ungauged_flow")].expected["branch"] == "at_gauge"
    assert key[("point", "ungauged_flow")].expected["branch"] == "regional"
    t = key[("3400TH", "flood_risk")]
    assert t.expected["tools"] == ["describe_catchment", "analyze_station", "flood_frequency"]
    assert {(g["step"], g["check"]) for g in t.expected["gates"]} >= {("s2", "min_years"), ("s3", "spread_within")}
    assert t.expected["station"]["station_id"] == "3400TH" and t.intake == {"return_period": 100,
                                                                             "decision": "design flow"}
    assert t.problem == "Design flow for a road crossing at this point, 100-year return period."
    assert t.recon["context"]["years_by_variable"] == {"discharge": 39.5}, "the reconnaissance is snapshotted"
    assert t.id.startswith("flood_risk-") and t.id == gt._task_id("flood_risk", LONG, t.intake), "ids are stable"
    assert len({t.id for t in tasks}) == 6 and all(t.split in gt.SPLITS for t in tasks)
    assert events[0].startswith("site 1 uk_ea/3400TH: discharge 39.5 yr")


def test_a_site_the_data_cannot_support_makes_unsolvable_tasks():
    tasks = gt.tasks_from_playbooks([POINT], ["flood_risk", "ungauged_flow"],
                                    recon=lambda lat, lon: fake_recon(lat, lon, point_donors=2), probes=0)
    assert [t.unsolvable for t in tasks] == [True, True]
    flood, flow = tasks
    assert flow.expected["decline_kind"] == "declined" and "Fewer than three donor" in flow.expected["decline_reason"]
    assert flood.expected["decline_kind"] == "refused" and "3 donor gauges, 2 found" in flood.expected["decline_reason"]
    assert flood.expected["branch"] is None and flood.expected["tools"] == [] and flood.expected["gates"] == []


def test_probes_are_read_off_the_playbooks_decline_rules_and_rotate_over_sites():
    assert gt.decline_probes("flood_risk") == [{"intake": {"decision": "inundation extent"},
                                                "rule": "Inundation extent (which streets or fields"}]
    assert gt.decline_probes("groundwater_decline")[0]["intake"] == {"attribute_cause": True}
    assert gt.decline_probes("ungauged_flow") == [], "its only decline reads the reconnaissance"
    assert gt.decline_probes("drought_status")[0]["intake"] == {"flash_drought": True}
    assert gt.decline_probes("supply_reliability") == [{"intake": {"storage": True},
                                                        "rule": "A scheme with a reservoir needs"}]
    assert gt.decline_probes("irrigation_feasibility")[0]["intake"] == {"decision": "daily schedule"}
    tasks = gt.tasks_from_playbooks([LONG, SHORT, POINT], None, recon=fake_recon, probes=1)
    assert len(tasks) == 24, "seven playbooks and one probe per site"
    probes = [t for t in tasks if t.probe]
    assert [t.playbook for t in probes] == ["drought_status", "flood_risk", "groundwater_decline"]
    assert all(t.unsolvable and t.expected["decline_kind"] == "declined" for t in probes)
    assert probes[0].intake["flash_drought"] is True and "flash drought" in probes[0].problem
    assert probes[1].intake["decision"] == "inundation extent" and "inundation extent" in probes[1].problem
    assert probes[2].intake["attribute_cause"] is True and "why" in probes[2].problem
    # the base tasks: supply_reliability with no demand stated declines at every site, and so does
    # water_quality where no sampled parameters are within reach (three sites each), plus the three probes
    assert sum(t.unsolvable for t in tasks) == 3 + 3 + 3
    everything = gt.tasks_from_playbooks([LONG], None, recon=fake_recon, probes=None)
    assert len(everything) == 7 + 6 and sum(t.unsolvable for t in everything) == 2 + 6
    assert [t.playbook for t in gt.tasks_from_playbooks([LONG], None, recon=fake_recon, probes=0)] == [
        "drought_status", "flood_risk", "groundwater_decline", "irrigation_feasibility", "supply_reliability",
        "ungauged_flow", "water_quality"]


def test_a_site_whose_reconnaissance_fails_is_skipped_with_a_note_unless_asked_to_keep_it():
    def broken(lat, lon):
        if (lat, lon) == (LONG["lat"], LONG["lon"]):
            raise RuntimeError("BasinATLAS down")
        return fake_recon(lat, lon)

    events, skipped = [], []
    tasks = gt.tasks_from_playbooks([LONG, SHORT], ["flood_risk"], recon=broken, probes=0, on_event=events.append,
                                    skipped=skipped)
    assert [t.site["station_id"] for t in tasks] == ["06120500"], "the unreachable site makes no task"
    assert len(skipped) == 1 and skipped[0]["site"] == LONG and "BasinATLAS down" in skipped[0]["error"]
    assert events[0].startswith("site 1 uk_ea/3400TH: skipped, reconnaissance unavailable: RuntimeError")
    # kept on request: the empty snapshot makes the gauge look ungauged, which is why the default skips it
    (t,) = gt.tasks_from_playbooks([LONG], ["flood_risk"], recon=broken, probes=0, skip_unreachable=False)
    assert t.expected["branch"] == "regional" and "BasinATLAS down" in t.recon["notes"][0]


@pytest.mark.parametrize("playbook,intake", [
    ("flood_risk", {}), ("flood_risk", {"decision": "inundation extent"}),
    ("flood_risk", {"decision": "risk screening", "return_period": 50}), ("flood_risk", {"decision": "insurance"}),
    ("ungauged_flow", {}), ("ungauged_flow", {"purpose": "irrigation offtake", "statistic": "Q95"}),
    ("groundwater_decline", {}), ("groundwater_decline", {"attribute_cause": True}),
    ("drought_status", {}), ("drought_status", {"flash_drought": True}),
    ("drought_status", {"drought_concern": "agriculture"}),
    ("supply_reliability", {"demand_m3s": 2.0}), ("supply_reliability", {"demand_ml_day": 5.0, "use": "municipal"}),
    ("supply_reliability", {"demand_m3s": 3.0, "storage": True}),
    ("irrigation_feasibility", {"crop": "wheat_winter", "area_ha": 25.0, "planting_month": 10}),
    ("irrigation_feasibility", {"decision": "daily schedule"}),
])
def test_problem_texts_route_to_their_playbook_and_state_their_intake(playbook, intake):
    text = gt.problem_text(playbook, intake)
    assert choose_playbook(text) == (playbook, False), text
    hints = intake_hints(text, playbook)
    for k, v in intake.items():
        if k in ("decision", "return_period", "attribute_cause", "purpose", "statistic", "flash_drought",
                 "drought_concern", "demand_m3s", "demand_ml_day", "use", "storage", "crop", "area_ha",
                 "planting_month"):
            assert hints.get(k) == v, (text, hints)


def test_split_and_site_key_are_deterministic():
    assert gt.site_key(LONG) == "uk_ea/3400TH" and gt.site_key(POINT) == "40.1234,-3.4567"
    assert gt.split_for(LONG) == gt.split_for(dict(LONG)) and gt.split_for(LONG) in gt.SPLITS
    sites = [{"lat": 0.0, "lon": float(i)} for i in range(400)]
    share = sum(gt.split_for(s) == "test" for s in sites) / len(sites)
    assert 0.15 < share < 0.35, "about one site in four is held out"


def test_jsonl_round_trip(tmp_path):
    tasks = gt.tasks_from_playbooks([LONG, POINT], ["flood_risk"], recon=fake_recon, probes=1)
    path = gt.write_tasks(tasks, tmp_path / "tasks.jsonl")
    back = gt.read_tasks(path)
    assert [t.to_dict() for t in back] == [t.to_dict() for t in tasks]
    assert back[0].unsolvable is False and back[1].unsolvable is True and back[1].probe


def _row(source, sid, country, lat, lon, start, end, variables=("discharge",)):
    return {"source": source, "station_id": sid, "name": f"{source} {sid}", "latitude": lat, "longitude": lon,
            "variables": list(variables), "period_start": start, "period_end": end, "country": country, "extra": {}}


CATALOG = [
    _row("usgs", "A", "USA", 40.0, -100.0, "1950-01-01", "2026-01-01"),
    _row("usgs", "A1", "USA", 40.6, -99.4, "1950-01-01", "2026-01-01"),        # a cluster around A, so a point
    _row("usgs", "A2", "USA", 39.4, -99.5, "1950-01-01", "2026-01-01"),        # offset from it is surrounded
    _row("usgs", "A3", "USA", 40.5, -100.6, "1950-01-01", "2026-01-01"),
    _row("usgs", "A4", "USA", 39.5, -100.5, "1950-01-01", "2026-01-01"),
    _row("usgs", "B", "USA", 41.0, -101.0, "2015-01-01", "2026-01-01"),
    _row("usgs", "C", "USA", 42.0, -102.0, "2023-01-01", "2026-01-01"),               # too short for anything
    _row("uk_ea", "D", "GBR", 51.4, -0.3, "1883-10-01", "2026-01-01"),
    _row("uk_ea", "E", "GBR", 52.0, -1.0, "2000-01-01", "2026-01-01", ("groundwater_level",)),
    _row("uk_ea", "F", "GBR", 53.0, -2.0, "2020-01-01", "2026-01-01", ("groundwater_level",)),  # a young well
    _row("bom", "G", "AUS", -35.8, 148.4, None, None),                                # no span: not counted
    _row("hubeau_hydrometrie", "H", "FRA", 45.0, 4.8, "1990-01-01", "2026-01-01"),
    _row("taiwan_cwa", "I", "TWN", 25.0, 121.5, "2010-01-01", "2026-01-01"),
]


def test_suggest_sites_spans_kinds_sources_and_continents_and_adds_ungauged_points():
    today = date(2026, 9, 2)
    sites = gt.suggest_sites(8, seed=7, catalog=CATALOG, today=today)
    assert len(sites) == 8 and sites == gt.suggest_sites(8, seed=7, catalog=CATALOG, today=today)
    kinds = [s["kind"] for s in sites]
    assert kinds.count("ungauged") == 2 and {"gauged_long", "gauged_short", "groundwater"} <= set(kinds)
    gauged = [s for s in sites if s["kind"] != "ungauged"]
    assert {s["station_id"] for s in gauged} <= {"A", "A1", "A2", "A3", "A4", "B", "D", "E", "H", "I"}, \
        "C, F and G make no site"
    assert len({s["source"] for s in gauged}) >= 3 and len({s["continent"] for s in gauged}) >= 3
    long_ = next(s for s in gauged if s["station_id"].startswith("A"))
    assert long_["years"] == 76.0 and long_["continent"] == "north_america" and long_["kind"] == "gauged_long"
    for p in (s for s in sites if s["kind"] == "ungauged"):
        assert p["source"] is None and p["anchor"].startswith("usgs/A") and 38.5 < p["lat"] < 41.5
        assert gt._surrounded(_cells(CATALOG, today), p["lat"], p["lon"])
    only_uk = gt.suggest_sites(4, seed=1, catalog=CATALOG, sources=["uk_ea"], today=today, ungauged_share=0)
    assert {s["source"] for s in only_uk} == {"uk_ea"} and len(only_uk) == 2
    assert gt.suggest_sites(0, catalog=CATALOG) == [] and gt.suggest_sites(3, catalog=[]) == []
    # on_land is asked about every surrounded candidate; refusing them all tops the suite up with gauges.
    asked = []

    def at_sea(lat, lon):
        asked.append((lat, lon))
        return False

    topped = gt.suggest_sites(6, seed=7, catalog=CATALOG, today=today, on_land=at_sea)
    assert asked and len(topped) == 6 and not any(s["kind"] == "ungauged" for s in topped)


def _cells(catalog, today):
    cells = {}
    for row in catalog:
        v = gt._classify(row, today)
        if v and v[0] != "groundwater":
            site = gt._site_from_row(row, *v)
            cells.setdefault((int(site["lat"] // 1), int(site["lon"] // 1)), []).append(site)
    return cells


def test_a_task_whose_key_the_tree_cannot_compute_is_skipped_with_the_playbook_named():
    def truncated(lat, lon):
        # the context says 31 years of groundwater levels but no listed station carries the variable
        snap = fake_recon(lat, lon)
        snap["context"]["years_by_variable"]["groundwater_level"] = 31.2
        return snap

    skipped, events = [], []
    tasks = gt.tasks_from_playbooks([LONG], ["flood_risk", "groundwater_decline"], recon=truncated, probes=0,
                                    skipped=skipped, on_event=events.append)
    assert [t.playbook for t in tasks] == ["flood_risk"]
    assert len(skipped) == 1 and skipped[0]["playbook"] == "groundwater_decline" and skipped[0]["site"] == LONG
    assert "no key: PlaybookError: placeholder station" in skipped[0]["error"]
    assert events[-1].startswith("  groundwater_decline: skipped, no key: PlaybookError")


def test_an_open_catalog_span_runs_to_today_as_the_reconnaissance_reads_it():
    today = date(2026, 9, 3)
    assert gt._span_years("2012-11-11", None, today) == 13.8, "a station still open (hubeau, uk_ea) has a span"
    assert gt._span_years("1990-01-01", "2026-01-01", today) == 36.0
    assert gt._span_years("1990-01-01", "2030-01-01", today) == 36.7, "an end in the future is clipped to today"
    assert gt._span_years(None, "2026-01-01", today) is None and gt._span_years("2027-01-01", None, today) is None
    open_gauge = _row("hubeau_hydrometrie", "V1", "FRA", 47.9, -0.2, "1980-06-01", None)
    assert gt._classify(open_gauge, today) == ("gauged_long", 46.3)
    well = _row("uk_ea", "W1", "GBR", 51.7, -0.7, "1991-09-03", None, ("groundwater_level",))
    assert gt._classify(well, today) == ("groundwater", 35.0)
    sites = gt.suggest_sites(12, seed=3, catalog=CATALOG + [open_gauge, well], today=today, ungauged_share=0)
    assert {"V1", "W1"} <= {s["station_id"] for s in sites}, "open stations are sampled"


def test_the_land_proxy_rejects_an_island_cluster_and_accepts_a_surrounded_point():
    today = date(2026, 9, 2)
    island = [_row("usgs", f"S{i}", "USA", 15.1 + 0.05 * i, 145.7 + 0.03 * i, "1990-01-01", "2026-01-01")
              for i in range(4)]
    cells = _cells(island, today)
    assert not gt._surrounded(cells, 14.64, 145.86), "four gauges all to the north-west: the point is at sea"
    assert gt._surrounded(_cells(CATALOG, today), 40.1, -100.0)
    assert not gt._surrounded(_cells(CATALOG, today), 45.0, 4.8), "one gauge alone never surrounds"
