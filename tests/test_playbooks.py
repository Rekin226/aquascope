"""Playbooks: the shipped trees validate, pick the right branch per site, decline what they should, and
emit a study the runner executes with no model in the loop."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

import aquascope.explore
from aquascope import playbooks as pbk
from aquascope.ai_engine import team
from aquascope.gates import evaluate
from aquascope.study import parse_block_yaml, run_study

STATION = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "distance_km": 0.4,
           "variables": ["discharge", "water_level"], "period_start": "1883-10-01", "years": 39.5}
RAIN = {"source": "uk_ea", "station_id": "R1", "name": "Teddington rain", "distance_km": 2.1,
        "variables": ["precipitation"], "years": 36.0}
BORE = {"source": "uk_ea", "station_id": "W1", "name": "Bore", "distance_km": 5.0,
        "variables": ["groundwater_level"], "years": 15}
ALL_IDS = ["drought_status", "flood_risk", "groundwater_decline", "irrigation_feasibility", "supply_reliability",
           "ungauged_flow", "water_quality"]
#: What assess_site reports as reachable for any point on land.
POINT_PRODUCTS = ("glofas", "temperature", "forcing")


def recon(years=None, stations=None, donors=None, area=None, dams=0, available=POINT_PRODUCTS):
    years = years or {}
    ctx = {"years_by_variable": dict(years), "resolution_by_variable": {k: "daily" for k in years},
           "area_km2": area, "donors": donors, "ungauged": not years}
    if available is not None:
        ctx["available"] = list(available)
    return {"point": {"lat": 51.415, "lon": -0.308}, "stations": list(stations or []),
            "catchment": {"area_km2": area, "upstream_area_km2": area, "dams": dams},
            "context": ctx, "sufficiency": [], "notes": ["served record starts 1986"]}


LONG = recon({"discharge": 39.5}, [STATION], donors=8, area=9948, dams=1)
SHORT = recon({"discharge": 12}, [dict(STATION, years=12)], donors=8, area=300)
VERY_SHORT = recon({"discharge": 7}, [dict(STATION, years=7)], donors=8, area=300)
UNGAUGED = recon(None, [], donors=5, area=120)
WELL = recon({"groundwater_level": 15}, [dict(STATION, variables=["groundwater_level"], years=15)])
WQ_STATION = {"source": "usgs", "station_id": "USGS-01646500", "name": "Potomac at Little Falls", "distance_km": 7.2,
              "variables": ["discharge", "water_quality"], "period_start": "1930-03-01", "years": 96.5}
WQ_SITE = recon({"discharge": 96.5, "water_quality": 96.5}, [WQ_STATION], donors=8, area=29000)


RICH = recon({"discharge": 39.5, "precipitation": 36.0, "groundwater_level": 15}, [STATION, RAIN, BORE], donors=8,
             area=9948)
RAIN_ONLY = recon({"precipitation": 25}, [dict(RAIN, years=25)], available=("glofas",))
RAIN_MARGINAL = recon({"precipitation": 25}, [dict(RAIN, years=25)])
BORE_ONLY = recon({"groundwater_level": 15}, [BORE])
DEMAND = {"demand_m3s": 2}
CROP = {"crop": "maize", "area_ha": 20}


def test_the_six_playbooks_list_load_and_validate():
    ids = [p["id"] for p in pbk.list_playbooks()]
    assert ids == ALL_IDS
    for pid in ids:
        pb = pbk.load(pid)
        assert pbk.validate(pb) == [], pid
        assert pb.branches and pb.caveats and pb.citations and pb.declines
        desc = pbk.describe(pid)
        assert desc["id"] == pid and isinstance(desc["branches"], list)


def test_the_files_stay_within_the_yaml_subset_the_browser_reads():
    yaml = pytest.importorskip("yaml")
    for path in pbk.PLAYBOOK_DIR.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert parse_block_yaml(text) == yaml.safe_load(text), path.name


@pytest.mark.parametrize("pid, site, intake, branch, tools", [
    ("flood_risk", LONG, {"return_period": 100}, "at_site",
     ["describe_catchment", "analyze_station", "flood_frequency"]),
    ("flood_risk", SHORT, {"return_period": 100}, "short_record",
     ["describe_catchment", "analyze_station", "similar_basins", "regionalize_signatures", "anywhere"]),
    ("flood_risk", UNGAUGED, {"return_period": 100}, "regional",
     ["describe_catchment", "similar_basins", "regionalize_signatures", "anywhere"]),
    ("ungauged_flow", LONG, {}, "at_gauge", ["describe_catchment", "analyze_station", "regionalize_signatures"]),
    ("ungauged_flow", UNGAUGED, {}, "regional",
     ["describe_catchment", "similar_basins", "regionalize_signatures", "anywhere"]),
    ("groundwater_decline", WELL, {}, "well",
     ["analyze_station", "get_timeseries", "sgi_drought", "get_timeseries", "recharge"]),
    ("groundwater_decline", UNGAUGED, {}, "regional", ["anywhere"]),
    # drought: a long rain gauge with temperature, a well and a river beside it
    ("drought_status", RICH, {}, "gauge_indices", ["drought_indices", "low_flow_context", "drought_propagation"]),
    ("drought_status", RAIN_MARGINAL, {}, "gauge_indices_marginal", ["drought_indices"]),
    ("drought_status", RAIN_ONLY, {}, "gauge_spi_only", ["drought_indices"]),
    ("drought_status", LONG, {}, "reanalysis", ["drought_indices", "low_flow_context"]),
    ("drought_status", BORE_ONLY, {}, "reanalysis", ["drought_indices", "drought_propagation"]),
    ("drought_status", UNGAUGED, {}, "reanalysis", ["drought_indices"]),
    # supply: gauged long, gauged short (with the regional cross-check), ungauged
    ("supply_reliability", LONG, DEMAND, "gauged", ["describe_catchment", "analyze_station", "supply_reliability"]),
    ("supply_reliability", VERY_SHORT, DEMAND, "gauged_short",
     ["describe_catchment", "analyze_station", "supply_reliability", "supply_reliability"]),
    ("supply_reliability", UNGAUGED, DEMAND, "regional",
     ["describe_catchment", "similar_basins", "supply_reliability"]),
    # irrigation: the supply check only when a gauge is within reach
    ("irrigation_feasibility", LONG, CROP, "with_gauge", ["anywhere", "crop_water_demand", "supply_reliability"]),
    ("irrigation_feasibility", VERY_SHORT, CROP, "with_gauge", ["anywhere", "crop_water_demand", "supply_reliability"]),
    ("irrigation_feasibility", UNGAUGED, CROP, "demand_only", ["anywhere", "crop_water_demand"]),
    ("irrigation_feasibility", BORE_ONLY, CROP, "demand_only", ["anywhere", "crop_water_demand"]),
    ("water_quality", WQ_SITE, {}, "drinking", ["water_quality_samples", "who_screen", "wqi"]),
])
def test_each_playbook_selects_the_branch_the_record_supports(pid, site, intake, branch, tools):
    assert pbk.select_branch(pid, site, intake).id == branch
    study = pbk.plan(pid, site, intake, problem_text="the problem")
    assert study.version == 2 and study.author == "playbook" and study.question == "the problem"
    assert study.plan["playbook"] == pid and study.plan["branch"] == branch
    assert [s.tool for s in study.steps] == tools
    text = study.to_yaml()
    assert not re.search(r"\{\{\s*(intake|station|site|derived)\.", text), "every plan-time placeholder is resolved"
    assert all(s.id and s.rationale for s in study.steps)
    if pid in ("drought_status", "supply_reliability", "irrigation_feasibility"):
        assert all(s.method for s in study.steps if s.tool != "describe_catchment"), "every step names its method"
    assert study.plan["caveats"] and study.plan["citations"]
    assert study.plan["recon_notes"] == ["served record starts 1986"]


def test_placeholders_resolve_to_typed_values_and_prose():
    study = pbk.plan("flood_risk", LONG, {"return_period": 200})
    fetch = study.step_by_id("s3")
    assert fetch.arguments == {"source": "uk_ea", "station_id": "3400TH", "bootstrap_ci": True}
    assert study.step_by_id("s1").arguments == {"lat": 51.415, "lon": -0.308}
    rp = [g for g in fetch.expects if g["check"] == "max_return_period_factor"][0]
    assert rp["return_period"] == 200 and isinstance(rp["return_period"], int)
    assert "T = 200 year" in fetch.rationale and "39.5 years" in study.plan["rationale"]
    assert study.problem["params"] == {"return_period": 200, "decision": "design flow"}
    assert study.plan["station"]["station_id"] == "3400TH"


def test_intake_defaults_and_coercion():
    pb = pbk.load("groundwater_decline")
    filled = pbk.fill_intake(pb, {"horizon": "20", "attribute_cause": "no", "concern": "Supply"})
    assert filled == {"horizon": 20, "concern": "supply", "attribute_cause": False}
    assert pbk.fill_intake(pb, None)["attribute_cause"] is False
    with pytest.raises(pbk.Declined) as exc:
        pbk.fill_intake(pbk.load("flood_risk"), {"decision": "mapping"})
    assert exc.value.kind == "intake" and "decision" in exc.value.reason


def test_declines_print_their_own_sentence():
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("flood_risk", recon({"discharge": 12}, [dict(STATION, years=12)], donors=1), {"return_period": 100})
    assert exc.value.kind == "declined" and "36 years" in exc.value.reason and "100 years" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("flood_risk", LONG, {"decision": "inundation extent"})
    assert "out of scope" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("ungauged_flow", recon(None, [], donors=2))
    assert "three donor" in exc.value.reason and "2 found" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("groundwater_decline", WELL, {"attribute_cause": True})
    assert "pumping" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("water_quality", LONG)
    assert "Phase 3 (#188)" in exc.value.reason and "Water Quality Portal" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("water_quality", WQ_SITE, {"health_verdict": True})
    assert "health judgement" in exc.value.reason and "never sampled" in exc.value.reason
    # a long record with donors is not declined for T = 100
    assert pbk.plan("flood_risk", LONG, {"return_period": 100}).plan["branch"] == "at_site"
    # and a site with donors unknown is left to the run-time gate
    assert pbk.plan("ungauged_flow", recon(None, [], donors=None)).plan["branch"] == "regional"


def test_caveats_carry_the_evidence_sentences_and_react_to_the_site():
    flood = pbk.plan("flood_risk", LONG, {"return_period": 100}).plan["caveats"]
    assert any("Wasko et al. 2024" in c and "immature" in c for c in flood)
    assert any("upstream dams" in c for c in flood)
    no_dams = pbk.plan("flood_risk", recon({"discharge": 39.5}, [STATION], donors=8, dams=0), {"return_period": 100})
    assert not any("upstream dams" in c for c in no_dams.plan["caveats"])
    gw = pbk.load("groundwater_decline")
    assert any("Jasechko" in c for c in gw.citations)
    regional = pbk.plan(gw, UNGAUGED).plan["caveats"]
    assert any("regional signal" in c for c in regional)


def test_the_273_scenario_is_refused_at_plan_time_and_by_the_gate():
    big = recon({"discharge": 30}, [STATION], area=101033)
    tree = {"id": "calib", "title": "Calibration", "problem": "climate_change",
            "branches": [{"id": "only", "steps": [
                {"id": "s1", "tool": "analyse_table", "method": "gr4j_calibration", "arguments": {}}]}]}
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan(tree, big)
    assert exc.value.kind == "refused" and "101,033" in exc.value.reason and "ceiling" in exc.value.reason
    gate = evaluate([{"check": "max_area_km2", "value": 10000, "path": "attributes.upstream_area_km2"}],
                    {"attributes": {"upstream_area_km2": 101033.0}})
    assert not gate[0]["passed"] and "above the ceiling" in gate[0]["detail"]
    # the same tree on a small catchment plans
    assert pbk.plan(tree, recon({"discharge": 30}, [STATION], area=900, available=["forcing"])).steps


def test_an_optional_step_is_dropped_with_a_note_not_refused():
    study = pbk.plan("flood_risk", recon(None, [], donors=5, available=[]))
    assert [s.tool for s in study.steps] == ["describe_catchment", "similar_basins", "regionalize_signatures"]
    assert study.plan["notes"] and "glofas" in study.plan["notes"][0]


def test_validate_catches_the_authoring_mistakes():
    bad = {"id": "bad", "title": "Bad", "problem": "x",
           "intake": [{"name": "t", "type": "choice"}, {"name": "u", "type": "int", "default": 1}],
           "branches": [{"id": "b", "when": [{"path": "a", "op": "~", "value": 1}], "steps": [
               {"id": "s1", "tool": "teleport", "method": "warp", "arguments": {"x": "{{ intake.nope }}"},
                "expects": [{"check": "nope"}], "depends_on": ["s9"], "fallback": {"foo": 1}},
               {"id": "s1", "tool": "list_sources", "arguments": {"y": "{{ magic.x }}"}}]}],
           "declines": [{"when": [{"path": "a", "op": "in", "value": []}], "say": " "}]}
    errors = pbk.validate(bad)
    text = "\n".join(errors)
    for needle in ("a choice needs options", "unknown operator '~'", "unknown tool 'teleport'",
                   "unknown method 'warp'", "not an earlier step", "unknown check 'nope'", "a fallback is",
                   "intake.nope", "magic", "duplicate id", "says nothing"):
        assert needle in text, needle
    with pytest.raises(pbk.PlaybookError):
        pbk.load("no_such_playbook")
    with pytest.raises(pbk.PlaybookError):
        pbk.load({"id": "x"})


def test_a_gauged_branch_without_its_station_is_an_authoring_error():
    site = recon({"discharge": 30}, [dict(STATION, variables=["water_level"])], donors=5)
    with pytest.raises(pbk.PlaybookError, match="station"):
        pbk.plan("flood_risk", site, branch="at_site")


def test_the_study_a_playbook_emits_runs_with_no_model():
    study = pbk.plan("flood_risk", LONG, {"return_period": 100})
    payload = {"source": "uk_ea", "station_id": "3400TH", "unit": "m3/s", "years": 39.9, "trend": {"p_value": 0.3},
               "ffa": {"return_periods": [2, 5, 10, 25, 50, 100],
                       "fits": {"gev_lmoments": {"q": [1, 2, 3, 4, 5, 6]}, "lp3": {"q": [1, 2, 3, 4, 5, 6.5]},
                                "gev_bootstrap": {"q": [1, 2, 3, 4, 5, 6], "ci": [[5, 7]] * 6}}}}
    tools = {"describe_catchment": lambda **kw: {"sub_basin": {"hybas_id": 1}, "attributes": {}},
             "analyze_station": lambda **kw: payload, "flood_frequency": lambda **kw: payload}
    with patch("aquascope.study._tools", return_value=tools):
        run = run_study(study)
    assert run.ok and all(g["passed"] for g in run.gates) and len(run.gates) == 7
    assert "gate spread_within: passed" in run.to_markdown()


def test_the_explorer_playbook_list_is_the_package_s_own():
    """explorer/playbooks.json is generated from the YAML files; the page draws its chips from it."""
    import json
    from pathlib import Path

    from aquascope.playbooks import as_json

    data = json.loads(as_json())
    ids = [p["id"] for p in data["playbooks"]]
    assert ids == ALL_IDS
    flood = data["playbooks"][1]
    assert flood["title"] and flood["problem"] == "flood_risk"
    fields = {f["name"]: f for f in flood["intake"]}
    assert fields["return_period"]["type"] == "int" and fields["return_period"]["default"] == 100
    assert fields["decision"]["type"] == "choice" and "design flow" in fields["decision"]["options"]
    shipped = Path(__file__).resolve().parents[1] / "explorer" / "playbooks.json"
    assert shipped.read_text(encoding="utf-8") == as_json(), (
        "explorer/playbooks.json is stale: run `python -m aquascope.playbooks`"
    )


# ── the three playbooks of the second batch: drought, supply, irrigation ─────


def test_the_new_playbooks_resolve_their_placeholders_to_typed_values():
    drought = pbk.plan("drought_status", RICH, {"timescales": "3, 6,12"})
    s1, s2, s3 = drought.steps
    assert s1.arguments == {"lat": 51.415, "lon": -0.308, "source": "uk_ea", "station_id": "R1",
                            "timescales": [3, 6, 12], "years": 40}, "a list intake keeps its type"
    assert s2.arguments == {"source": "uk_ea", "station_id": "3400TH"}, "the discharge step takes the river gauge"
    assert s3.arguments["station_id"] == "W1", "the SGI step takes the well, not the branch's rain gauge"
    assert "36 years of precipitation at Teddington rain" in drought.plan["rationale"]
    assert "[3, 6, 12] months" in drought.plan["rationale"]
    assert pbk.fill_intake(pbk.load("drought_status"), None)["timescales"] == [1, 3, 12]
    with pytest.raises(pbk.Declined, match="empty list"):
        pbk.fill_intake(pbk.load("drought_status"), {"timescales": ", ,"})

    supply = pbk.plan("supply_reliability", LONG, {"demand_ml_day": 50})
    assert supply.step_by_id("s3").arguments == {"source": "uk_ea", "station_id": "3400TH", "demand_m3s": None,
                                                 "demand_ml_day": 50.0, "share": 0.1, "reserve": "q95"}
    assert supply.problem["params"]["use"] == "other" and supply.problem["params"]["storage"] is False

    irrigation = pbk.plan("irrigation_feasibility", LONG, {"crop": "Maize", "area_ha": "20", "planting_month": 5})
    s3 = irrigation.step_by_id("s3")
    assert s3.arguments["demand_m3s"] == "{{ result.s2.demand.peak_month_m3s }}", "left for the runner"
    assert s3.arguments["months"] == "{{ result.s2.season.months }}" and s3.depends_on == ["s2"]
    assert irrigation.step_by_id("s2").arguments["crop"] == "maize"
    assert irrigation.problem["params"] == {"crop": "maize", "area_ha": 20.0, "planting_month": 5, "efficiency": 0.7,
                                            "share": 0.1, "decision": "seasonal demand"}
    assert "The supply is that river at that gauge" in irrigation.plan["rationale"]


def test_the_crop_choices_are_the_fao56_table():
    from aquascope.agri.crop_water import KC_TABLE

    crop = next(f for f in pbk.load("irrigation_feasibility").intake if f.name == "crop")
    assert crop.options == sorted(KC_TABLE) and crop.default in KC_TABLE


def test_the_new_playbooks_decline_with_their_own_sentences():
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("drought_status", RICH, {"flash_drought": True})
    assert exc.value.kind == "declined" and "monthly droughts" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("supply_reliability", LONG)
    assert "State the demand" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("supply_reliability", LONG, {"demand_m3s": 1, "storage": True})
    assert "storage-yield" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("supply_reliability", recon(None, [], donors=2), {"demand_ml_day": 5})
    assert "three donor" in exc.value.reason and "2 found" in exc.value.reason
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("irrigation_feasibility", LONG, {"decision": "daily schedule"})
    assert "day-by-day" in exc.value.reason
    # a rain gauge without temperature anywhere: the registry refuses the ET0 step rather than the tree
    with pytest.raises(pbk.Declined) as exc:
        pbk.plan("irrigation_feasibility", RAIN_ONLY, CROP)
    assert exc.value.kind == "refused" and "temperature" in exc.value.reason


def test_the_new_playbooks_caveats_react_to_the_site():
    rich = pbk.plan("drought_status", RICH).plan["caveats"]
    assert any("Thornthwaite" in c for c in rich) and not any("rainfall deficit only" in c for c in rich)
    spi_only = pbk.plan("drought_status", RAIN_ONLY).plan["caveats"]
    assert any("SPEI is preferable" in c for c in spi_only) and any("shorter than the 30 years" in c for c in spi_only)
    cell = pbk.plan("drought_status", UNGAUGED).plan["caveats"]
    assert any("ERA5 cell" in c for c in cell)
    assert any("screening rule" in c for c in pbk.plan("supply_reliability", LONG, DEMAND).plan["caveats"])
    short = pbk.plan("supply_reliability", VERY_SHORT, DEMAND).plan["caveats"]
    assert any("shorter than ten years" in c for c in short)
    regional = pbk.plan("supply_reliability", UNGAUGED, DEMAND).plan["caveats"]
    assert any("leave-one-out skill" in c for c in regional)
    gauge = pbk.plan("irrigation_feasibility", LONG, CROP).plan["caveats"]
    assert any("#310" in c for c in gauge) and any("108732" in c for c in gauge)
    assert any("supply comes from the gauged river" in c for c in gauge)
    no_gauge = pbk.plan("irrigation_feasibility", UNGAUGED, CROP).plan["caveats"]
    assert any("Supply was not checked" in c for c in no_gauge)


def test_result_placeholders_are_validated_and_resolved_by_the_runner():
    tree = {"id": "chain", "title": "Chain", "problem": "irrigation",
            "branches": [{"id": "only", "steps": [
                {"id": "s1", "tool": "anywhere", "arguments": {"lat": 1, "lon": 2}},
                {"id": "s2", "tool": "supply_reliability", "arguments": {"demand_m3s": "{{ result.s1.climate.x }}"}},
                {"id": "s3", "tool": "supply_reliability", "depends_on": ["s1"],
                 "arguments": {"demand_m3s": "{{ result.s9.climate.x }}"}}]}]}
    errors = "\n".join(pbk.validate(tree))
    assert "reads result.s1, so depends_on must list s1" in errors
    assert "result.s9.climate.x names no earlier step" in errors
    good = {"id": "chain", "title": "Chain", "problem": "irrigation",
            "branches": [{"id": "only", "steps": [
                {"id": "s1", "tool": "anywhere", "arguments": {"lat": 1, "lon": 2}},
                {"id": "s2", "tool": "supply_reliability", "depends_on": ["s1"],
                 "arguments": {"demand_m3s": "{{ result.s1.climate.et0_mm_per_year }}",
                               "months": "{{ result.s1.climate.months }}",
                               "reserve": "q{{ result.s1.climate.pct }}"}}]}]}
    assert pbk.validate(good) == []
    study = pbk.plan(good, UNGAUGED)
    seen = {}

    def supply(**kw):
        seen.update(kw)
        return {"reliability": {"daily": 1.0}, "unit": "m3/s"}

    tools = {"anywhere": lambda **kw: {"climate": {"et0_mm_per_year": 612.5, "months": [4, 5], "pct": 95}},
             "supply_reliability": supply}
    with patch("aquascope.study._tools", return_value=tools):
        run = run_study(study)
    assert run.ok and seen == {"demand_m3s": 612.5, "months": [4, 5], "reserve": "q95"}
    # a reference into nothing fails the step with the reason instead of passing a placeholder string on
    bad = pbk.plan(good, UNGAUGED)
    with patch("aquascope.study._tools", return_value={"anywhere": lambda **kw: {"climate": {}},
                                                       "supply_reliability": supply}):
        run = run_study(bad)
    assert not run.ok and "nothing at 'climate.et0_mm_per_year'" in run.results[1]["error"]


# ── keyless team.solve, one run per new playbook, the tools served by fakes ──

DROUGHT = {"latitude": 51.415, "longitude": -0.308, "pet_method": "thornthwaite",
           "precipitation_source": "uk_ea R1 (gauge)",
           "station": {"source": "uk_ea", "station_id": "R1", "variable": "precipitation", "unit": "mm", "years": 36.6},
           "timescales": [1, 3, 12], "headline_timescale": 3, "headline_index": "spei", "threshold": -1.0,
           "months": 439, "start": "1990-01-01", "end": "2026-07-01", "years": 36.6, "status": "moderately_dry",
           "in_drought": True,
           "current": {"date": "2026-07-01", "spi": {"1": -0.8, "3": -1.1, "12": -0.4},
                       "spei": {"1": -1.0, "3": -1.4, "12": -0.7}},
           "indices": [{"timescale": 3, "spi": {"current": -1.1, "class": "moderately_dry", "worst": -2.6,
                                                "worst_date": "1997-08-01", "events": 21},
                        "spei": {"current": -1.4, "class": "moderately_dry", "worst": -2.9, "worst_date": "2022-08-01",
                                 "events": 24},
                        "divergence": {"current": -0.3, "mean_last_10y": -0.35, "months_spei_drier_pct": 82.0,
                                       "correlation": 0.96, "n": 437}}],
           "temperature": {"mean_c": 10.9, "trend_c_per_decade": 0.31, "p_value": 0.002, "n_years": 36},
           "methods": [{"name": "SPI", "text": "t", "citation": "McKee 1993"}], "notes": []}
LOW = {"source": "uk_ea", "station_id": "3400TH", "variable": "discharge", "unit": "m3/s", "start": "1986-08-17",
       "end": "2026-08-15", "years": 39.9, "fdc": {"q95": 12.3, "q50": 43.0, "q10": 148.0, "q05": 210.0}, "bfi": 0.71,
       "low_flow": {"7q10": 9.8},
       "recent": {"end": "2026-08-15", "last_30d_mean": 15.2, "last_30d_exceedance_pct": 88.0,
                  "last_90d_mean": 19.9, "last_90d_exceedance_pct": 80.5},
       "methods": [{"name": "FDC", "text": "t", "citation": "Vogel 1994"}]}
PROPAGATION = {"source": "uk_ea", "station_id": "W1", "unit": "m", "years": 15.2, "start": "2011-01-01",
               "end": "2026-03-01",
               "sgi": {"current": -1.3, "date": "2026-03-01", "worst": -2.2, "worst_date": "2012-03-01", "events": 4,
                       "threshold": -1.0, "in_drought": True},
               "propagation": {"best": {"timescale": 6, "lag_months": 2, "correlation": 0.74, "n": 176}},
               "methods": [{"name": "SGI", "text": "t", "citation": "Bloomfield 2013"}]}
SUPPLY = {"mode": "gauged", "source": "uk_ea", "station_id": "3400TH", "unit": "m3/s", "years": 39.9,
          "start": "1986-08-17", "end": "2026-08-15", "demand_m3s": 2.0, "share": 0.1,
          "reserve_rule": "Q95 kept in the river",
          "reserve_m3s": 12.3, "required_flow_m3s": 20.0, "fdc": {"q95": 12.3, "q50": 43.0, "q10": 148.0}, "bfi": 0.71,
          "low_flow": {"7q10": 9.8},
          "reliability": {"daily": 0.91, "annual": 0.35, "volumetric": 0.96, "days_short_per_year": 33.0,
                          "worst_year": {"year": 1976, "days_short": 121}},
          "verdict": "seasonal shortfalls", "methods": [{"name": "FDC", "text": "t", "citation": "Vogel 1994"}]}
CROPWATER = {"crop": "maize", "area_ha": 20.0, "planting_month": 4, "efficiency": 0.7, "season_days": 125,
             "years_used": [2016, 2017, 2018], "season": {"months": [4, 5, 6, 7, 8]},
             "demand": {"etc_mm": 373.0, "effective_rain_mm": 261.6, "net_irrigation_mm": 217.8,
                        "gross_irrigation_mm": 311.1, "gross_irrigation_mm_range": [281.4, 343.8], "gross_m3": 62220.0,
                        "mean_m3s": 0.00576, "peak_month_mm": 127.0, "peak_month_m3s": 0.00966},
             "methods": [{"name": "FAO-56", "text": "t", "citation": "Allen 1998"}]}
CLIMATE = {"latitude": 51.415, "longitude": -0.308,
           "climate": {"precipitation_mm_per_year": 640.0, "et0_mm_per_year": 612.0, "aridity_index": 1.05,
                       "aridity_class": "humid"}}
FLOW = {"source": "uk_ea", "station_id": "3400TH", "name": "Kingston", "unit": "m3/s", "variable": "discharge",
        "start": "1986-08-17", "end": "2026-08-15", "years": 39.9, "stats": {"mean": 65.2, "min": 3.1, "max": 520.0},
        "fdc": {"q95": 12.3, "q50": 43.0, "q10": 148.0}}


def _solve(text, recon_value, tools, **kw):
    calls = []

    def rec(name, payload):
        def f(**kwargs):
            calls.append((name, kwargs))
            return payload
        return f

    served = {name: rec(name, payload) for name, payload in tools.items()}
    with patch.object(aquascope.explore, "assess_site", create=True, return_value=recon_value), \
         patch("aquascope.study._tools", return_value=served):
        res = team.solve(text, lat=51.415, lon=-0.308, **kw)
    return res, calls


def test_keyless_drought_status_end_to_end():
    res, calls = _solve("Is this area in drought?", RICH,
                        {"drought_indices": DROUGHT, "low_flow_context": LOW, "drought_propagation": PROPAGATION})
    assert not res.declined and res.ok and res.cost == {} and res.study.plan["branch"] == "gauge_indices"
    assert [c[0] for c in calls] == ["drought_indices", "low_flow_context", "drought_propagation"]
    assert calls[0][1]["timescales"] == [1, 3, 12] and calls[2][1]["station_id"] == "W1"
    assert all(g["passed"] for g in res.gates) and len(res.gates) == 7
    for needle in ("SPEI (thornthwaite PET): -1 at 1 month, -1.4 at 3 months", "moderately dry, in drought",
                   "averages -0.35 (SPEI the drier)", "0.31 C per decade", "Q95 12.3 m3/s", "lag of 2 months",
                   "uk_ea R1"):
        assert needle in res.answer, needle
    assert all(c["passed"] for c in res.checks), [c for c in res.checks if not c["passed"]]
    md = res.to_markdown()
    assert "Thornthwaite" in md and "Vicente-Serrano" in md and "Model calls: 0" in md


def test_keyless_supply_reliability_end_to_end():
    res, calls = _solve("Can the river supply 2 m3/s to a town?", LONG,
                        {"describe_catchment": {"sub_basin": {"hybas_id": 1},
                                                "attributes": {"upstream_area_km2": 9948.0}},
                         "analyze_station": FLOW, "supply_reliability": SUPPLY})
    assert not res.declined and res.ok and res.study.plan["branch"] == "gauged"
    assert res.problem["params"]["demand_m3s"] == 2.0 and res.problem["params"]["use"] == "municipal"
    assert calls[2][1] == {"source": "uk_ea", "station_id": "3400TH", "demand_m3s": 2.0, "demand_ml_day": None,
                           "share": 0.1, "reserve": "q95"}
    assert all(g["passed"] for g in res.gates) and len(res.gates) == 6
    for needle in ("Q95 12.3 m3/s", "20 m3/s", "91 % of days", "35 % of years", "seasonal shortfalls", "1976"):
        assert needle in res.answer, needle
    assert all(c["passed"] for c in res.checks), [c for c in res.checks if not c["passed"]]
    assert "Smakhtin" in res.to_markdown()


def test_keyless_irrigation_feasibility_hands_the_demand_to_the_supply_check():
    res, calls = _solve("Can I irrigate 20 ha of maize planted in April here?", LONG,
                        {"anywhere": CLIMATE, "crop_water_demand": CROPWATER, "supply_reliability": SUPPLY})
    assert not res.declined and res.ok and res.study.plan["branch"] == "with_gauge"
    assert res.problem["params"]["crop"] == "maize" and res.problem["params"]["area_ha"] == 20.0
    assert calls[1][1]["crop"] == "maize" and calls[1][1]["planting_month"] == 4
    assert calls[2][1]["demand_m3s"] == 0.00966 and calls[2][1]["months"] == [4, 5, 6, 7, 8], "the runner filled it"
    assert all(g["passed"] for g in res.gates)
    for needle in ("311.1 mm", "62,220 m3", "0.00966 m3/s", "the supply check against the gauge follows",
                   "91 % of days"):
        assert needle in res.answer, needle
    assert all(c["passed"] for c in res.checks), [c for c in res.checks if not c["passed"]]
    assert "#310" in res.to_markdown()
    # no gauge: the demand alone, and the plan says supply was not checked
    res, calls = _solve("Can I irrigate 20 ha of maize planted in April here?", UNGAUGED,
                        {"anywhere": CLIMATE, "crop_water_demand": CROPWATER})
    assert res.ok and res.study.plan["branch"] == "demand_only" and [c[0] for c in calls] == ["anywhere",
                                                                                              "crop_water_demand"]
    assert any("Supply was not checked" in c for c in res.caveats) and "supply was not checked" in res.answer


def test_the_scout_asks_the_registry_by_the_playbooks_problem_not_its_id():
    """drought_status is the playbook, drought the problem the sufficiency table knows; the live run had crashed."""
    seen = {}

    def fake_assess(lat, lon, *, problem=None, return_period=None):
        seen["problem"] = problem
        return RICH

    with patch.object(aquascope.explore, "assess_site", create=True, side_effect=fake_assess), \
         patch("aquascope.study._tools", return_value={"drought_indices": lambda **kw: DROUGHT,
                                                       "low_flow_context": lambda **kw: LOW,
                                                       "drought_propagation": lambda **kw: PROPAGATION}):
        res = team.solve("Is this area in drought?", lat=51.415, lon=-0.308, playbook="drought_status")
    assert seen["problem"] == "drought" and res.ok
    assert not any(e["event"] == "error" for e in res.timeline if e["role"] == "scout")
def test_coerce_intake_makes_a_model_s_reply_safe():
    """The lenient twin of fill_intake: a small model's mistake costs a default, never the plan."""
    good = pbk.coerce_intake("flood_risk", {"return_period": "50", "decision": "Design Flow", "foo": 1})
    assert good == {"return_period": 50, "decision": "design flow"}      # coerced, options case-folded, foo dropped
    bad = pbk.coerce_intake("flood_risk", {"return_period": -3, "decision": "mapping"})
    assert bad == {"return_period": 100, "decision": "design flow"}      # below min 2 and outside the options
    assert pbk.coerce_intake("flood_risk", {"return_period": float("nan")})["return_period"] == 100
    assert pbk.coerce_intake("flood_risk", {"return_period": [50]})["return_period"] == 100
    assert pbk.coerce_intake("flood_risk", {"return_period": True})["return_period"] == 100
    well = pbk.coerce_intake("groundwater_decline", {"attribute_cause": "yes", "horizon": 0})
    assert well == {"horizon": 10, "concern": "other", "attribute_cause": True}
    assert pbk.coerce_intake("ungauged_flow", None) == {"purpose": "other", "statistic": "all"}
    assert pbk.coerce_intake("ungauged_flow", "not a dict") == {"purpose": "other", "statistic": "all"}
    with pytest.raises(pbk.PlaybookError):
        pbk.coerce_intake("no_such_playbook", {})


def test_intake_bounds_are_strict_for_fill_intake_and_validated():
    with pytest.raises(pbk.Declined) as exc:
        pbk.fill_intake(pbk.load("flood_risk"), {"return_period": 1})
    assert exc.value.kind == "intake" and "minimum 2" in exc.value.reason
    assert pbk.fill_intake(pbk.load("flood_risk"), {"return_period": 2})["return_period"] == 2
    broken = {
        "id": "b", "title": "B", "problem": "flood_risk",
        "intake": [{"name": "t", "type": "choice", "options": ["a"], "min": 1},
                   {"name": "u", "type": "int", "min": 5, "max": 1}],
        "branches": [{"id": "x", "steps": [{"id": "s1", "tool": "anywhere"}]}],
    }
    errors = " ".join(pbk.validate(broken))
    assert "min/max apply to int and float" in errors and "above max" in errors


def test_the_water_quality_playbook_follows_the_use_and_carries_its_guideline_caveat():
    """A station with sampled parameters within reach: samples, then the index against the use's guidelines;
    irrigation adds the FAO 29 suitability index, drinking adds the WHO screen; no station declines."""
    drinking = pbk.plan("water_quality", WQ_SITE, problem_text="Is the river water safe to drink?")
    assert drinking.plan["branch"] == "drinking" and drinking.problem["params"]["use"] == "drinking"
    assert drinking.step_by_id("s1").arguments == {"source": "usgs", "station_id": "USGS-01646500", "years": 5,
                                                   "use": "drinking"}
    assert drinking.step_by_id("s3").arguments == {"from_step": "s1", "use": "drinking"}
    gates = [(g["check"], g.get("path")) for g in drinking.step_by_id("s3").expects]
    assert gates == [("not_empty", "ccme.score"), ("min_samples", "ccme.sample_counts")]
    assert drinking.step_by_id("s3").fallback == "stop"
    assert any("WHO (2022)" in c for c in drinking.plan["caveats"])
    assert any("not a" in c and "verdict" in c for c in drinking.plan["caveats"])
    assert any("USGS daily water-quality values" in c for c in drinking.plan["caveats"])
    assert any("digitised approximations" in c for c in drinking.plan["caveats"])
    irrigation = pbk.plan("water_quality", WQ_SITE, {"use": "irrigation", "years": 3})
    assert irrigation.plan["branch"] == "irrigation"
    assert [s.tool for s in irrigation.steps] == ["water_quality_samples", "wqi", "iwqi"]
    assert irrigation.step_by_id("s1").arguments["years"] == 3
    assert irrigation.step_by_id("s2").method == "water_quality_index"
    assert irrigation.step_by_id("s3").method == "iwqi"
    assert any("FAO Irrigation and Drainage Paper 29" in c for c in irrigation.plan["caveats"])
    aquatic = pbk.plan("water_quality", WQ_SITE, {"use": "aquatic life"})
    assert aquatic.plan["branch"] == "aquatic_life"
    assert [s.tool for s in aquatic.steps] == ["water_quality_samples", "wqi"]
    assert any("aquatic-life guidelines" in c for c in aquatic.plan["caveats"])
    assert pbk.select_branch("water_quality", LONG) is None
    unnamed = recon({"water_quality": 7.5}, [dict(WQ_STATION, name=None)])
    assert "listed at USGS-01646500 (usgs USGS-01646500" in pbk.plan("water_quality", unnamed).plan["rationale"]
    assert pbk.select_branch("water_quality", WQ_SITE, {"use": "irrigation"}).id == "irrigation"
