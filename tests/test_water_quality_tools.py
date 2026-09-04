"""The water-quality leg's plumbing (#62): the workbench analyses ``wqi`` and ``iwqi``, the
``water_quality_samples`` study tool over fake collectors, the MCP face, ``from_step`` over samples,
the ``min_samples`` gate, and the template Narrator's sentences."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from aquascope import explore, mcp_server
from aquascope import workbench as wb
from aquascope.ai_engine import team
from aquascope.collectors.usgs import USGSCollector
from aquascope.collectors.wqp import WQPCollector
from aquascope.schemas.water_data import DataSource, WaterQualitySample
from aquascope.study import Step, Study, run_study, tool_names

IDX = pd.date_range("2024-01-01", periods=12, freq="MS")


def _block(param: str, value: float, unit: str) -> pd.DataFrame:
    return pd.DataFrame({"sample_datetime": IDX, "parameter": param, "value": [value] * 12, "unit": unit})


@pytest.fixture(scope="module")
def drinking_frame() -> pd.DataFrame:
    return pd.concat([_block("pH", 7.4, ""), _block("Nitrate", 60.0, "mg/L"), _block("Arsenic", 2.0, "ug/L"),
                      _block("Turbidity", 1.0, "NTU"), _block("DO", 8.0, "mg/L"), _block("Temperature", 15.0, "deg C")])


# ── workbench ───────────────────────────────────────────────────────────────


def test_wqi_is_a_workbench_analysis_with_a_ccme_headline(drinking_frame):
    assert wb.TOOLS["wqi"]["needs"] == "frame" and wb.TOOLS["iwqi"]["needs"] == "frame"
    res = wb.run("wqi", drinking_frame)
    json.dumps(res, allow_nan=False)
    assert res["index"] == "ccme_wqi" and res["score"] == 83.5 and res["category"] == "Good"
    assert res["use"] == "drinking" and res["guideline_set"] == "drinking" and res["variant"] == "auto"
    assert res["ccme"]["n_variables"] == 5 and res["ccme"]["drivers"][0]["parameter"] == "nitrate"
    assert res["nsf"]["score"] is None, "four of the nine NSF parameters: no NSF score"
    assert res["sample_counts"]["nitrate"] == 12 and res["n_samples"] == 60 and res["unit"]
    assert res["period"]["start"] == "2024-01-01"
    names = [m["name"] for m in res["methods"]]
    assert names[0] == "CCME Water Quality Index 1.0" and "WHO 2022" in names[1]
    assert "CCME (2001)" in res["methods"][0]["citation"]


def test_wqi_takes_a_use_a_variant_and_a_custom_table(drinking_frame):
    aquatic = wb.run("wqi", drinking_frame, use="aquatic life", variant="ccme")
    assert aquatic["guideline_set"] == "aquatic life" and "nsf" not in aquatic and aquatic["score"] < 83.5
    custom = wb.run("wqi", drinking_frame, guidelines={"nitrate": {"max": 100}, "ph": {"min": 6, "max": 9}})
    assert custom["guideline_set"] == "custom" and custom["score"] == 100.0
    nsf_only = wb.run("wqi", drinking_frame, variant="nsf")
    assert nsf_only["index"] is None and nsf_only["score"] is None and "ccme" not in nsf_only
    with pytest.raises(ValueError, match="Unknown use"):
        wb.run("wqi", drinking_frame, use="bathing")
    with pytest.raises(ValueError, match="Unknown variant"):
        wb.run("wqi", drinking_frame, variant="magic")


def test_iwqi_is_a_workbench_analysis(drinking_frame):
    res = wb.run("iwqi", drinking_frame)
    json.dumps(res, allow_nan=False)
    assert res["index"] == "iwqi" and res["restriction"] == "slight to moderate"
    assert res["drivers"] == ["nitrate_nitrogen"] and "sodium" in res["missing"]
    assert res["methods"][0]["name"].startswith("Irrigation water quality") and "Ayers" in res["methods"][0]["citation"]
    assert res["unit"] == "meq/L"


def test_the_mcp_analyse_table_runs_wqi_over_csv(drinking_frame):
    res = mcp_server.analyse_table(drinking_frame.to_csv(index=False), "wqi", {"use": "drinking"})
    assert res["score"] == 83.5 and res["category"] == "Good"
    listed = {a["name"] for a in mcp_server.list_analyses()["analyses"]}
    assert {"wqi", "iwqi"} <= listed


def test_describe_methods_carries_the_index_citations():
    methods = mcp_server.describe_methods()["methods"]
    assert "CCME (2001)" in methods["ccme_wqi"]["citation"]
    assert "Brown" in methods["nsf_wqi"]["citation"] and "digitised" in methods["nsf_wqi"]["text"]
    assert "Ayers" in methods["iwqi"]["citation"]
    assert "World Health Organization" in methods["who_screen"]["citation"]


# ── the samples tool ────────────────────────────────────────────────────────


def _sample(param: str, value: float, unit: str, day: int) -> WaterQualitySample:
    return WaterQualitySample(source=DataSource.USGS, station_id="USGS-01646500",
                              sample_datetime=datetime(2024, 1, day), parameter=param, value=value, unit=unit)


class _FakeUSGS:
    def __init__(self):
        self.calls: list[dict] = []

    def collect(self, **kw):
        self.calls.append(kw)
        codes = kw["parameter"].split(",")
        out = []
        for d in range(1, 8):
            if "00010" in codes:
                out.append(_sample("Temperature", 10.0 + d, "deg C", d))
            if "00400" in codes:
                out.append(_sample("pH", 7.5, "std units", d))
            if "00300" in codes:
                out.append(_sample("DO", 9.0, "mg/l", d))
        return out


def test_water_quality_samples_from_usgs_is_tidy_counted_and_attributed():
    fake = _FakeUSGS()
    with patch.object(explore, "build_collector", return_value=fake):
        res = explore.water_quality_samples("usgs", "USGS-01646500")
    call = fake.calls[0]
    assert call["collection"] == "daily" and call["statCd"] == "00003" and call["station_id"] == "USGS-01646500"
    assert call["parameter"] == "00010,00095,00300,00400" and call["days"] == pytest.approx(5 * 365.25, abs=2)
    assert res["source"] == "usgs" and res["license"] == "US-PD" and "public domain" in res["attribution"]
    assert res["n_samples"] == 21 and res["n_parameters"] == 3
    assert res["sample_counts"] == {"DO": 7, "Temperature": 7, "pH": 7}
    assert res["units"] == {"DO": "mg/l", "Temperature": "deg C", "pH": "std units"} and res["unit"]
    assert res["parameters"]["Temperature"]["median"] == 14.0 and res["start"] == "2024-01-01"
    assert res["requested"]["years"] == 5 and "statistic 00003" in res["fetch_note"]
    assert res["samples"][0] == {"datetime": "2024-01-01T00:00:00", "parameter": "DO", "value": 9.0, "unit": "mg/l"}
    json.dumps(res, allow_nan=False)
    # named parameters map to codes; years caps the window
    with patch.object(explore, "build_collector", return_value=fake):
        explore.water_quality_samples("usgs", "USGS-01646500", years=2, parameters=["pH", "dissolved oxygen"])
    assert fake.calls[1]["parameter"] == "00300,00400" and fake.calls[1]["days"] == pytest.approx(2 * 365.25, abs=2)
    with pytest.raises(ValueError, match="cover"), patch.object(explore, "build_collector", return_value=fake):
        explore.water_quality_samples("usgs", "USGS-01646500", parameters=["boron"])


def test_water_quality_samples_declines_sources_without_a_path_and_reports_empties():
    with pytest.raises(ValueError, match="no water-quality samples"):
        explore.water_quality_samples("uk_ea", "3400TH")
    with pytest.raises(ValueError, match="no water-quality fetch path yet"):
        explore.water_quality_samples("gemstat", "X")

    class Empty:
        def collect(self, **kw):
            return []

    with patch.object(explore, "build_collector", return_value=Empty()):
        res = explore.water_quality_samples("usgs", "USGS-0")
    assert res["n_samples"] == 0 and "no water-quality samples" in res["error"] and res["samples"] == []
    assert mcp_server.water_quality_samples("nope", "x")["error"].startswith("unknown source")
    assert "no water-quality samples" in mcp_server.water_quality_samples("uk_ea", "3400TH")["error"]


def test_water_quality_samples_from_wqp_asks_for_the_screening_list_only():
    class FakeWQP:
        def __init__(self):
            self.calls: list[dict] = []

        def collect(self, **kw):
            self.calls.append(kw)
            return [WaterQualitySample(source=DataSource.WQP, station_id=kw["site_id"],
                                       sample_datetime=datetime(2023, 5, d), parameter="Nitrate", value=1.0 * d,
                                       unit="mg/l as N") for d in range(1, 6)]

    fake = FakeWQP()
    with patch.object(explore, "build_collector", return_value=fake):
        res = explore.water_quality_samples("wqp", "USGS-01646500", use="irrigation")
    call = fake.calls[0]
    assert call["site_id"] == "USGS-01646500"
    assert call["characteristic_name"] == list(explore.WQP_CHARACTERISTICS["irrigation"])
    assert call["max_results"] == explore.WQ_MAX_SAMPLES and len(call["start_date"].split("-")) == 3
    assert res["n_samples"] == 5 and res["units"] == {"Nitrate": "mg/l as N"} and "capped" in res["fetch_note"]


def test_the_wqp_collector_passes_site_ids_and_several_characteristics():
    class FakeClient:
        base_url = "x"
        rate_limiter = None
        _client = object()   # no .stream: the buffered path

        def __init__(self):
            self.calls: list = []

        def get_text(self, path, params=None, use_cache=True):
            self.calls.append((path, params))
            return "Location_Identifier,Result_Characteristic,Result_Measure,Result_MeasureUnit,Activity_StartDate\n"

    client = FakeClient()
    WQPCollector(client=client).fetch_raw(site_id="USGS-01646500", characteristic_name=["pH", "Nitrate"],
                                          start_date="01-01-2020", end_date="01-01-2025", max_results=10)
    path, params = client.calls[0]
    assert path == "/Result/search" and params["siteid"] == "USGS-01646500"
    assert params["characteristicName"] == ["pH", "Nitrate"] and params["startDateLo"] == "01-01-2020"


def test_the_usgs_collector_passes_the_statistic_code_on_the_keyless_path():
    class FakeClient:
        def __init__(self):
            self.calls: list = []

        def get_json(self, url, params=None, **kw):
            self.calls.append((url, params))
            return {"value": {"timeSeries": []}}

    client = FakeClient()
    USGSCollector(api_key="DEMO_KEY", client=client).fetch_raw(station_id="USGS-01646500", days=10, collection="daily",
                                                              parameter="00010,00400", statCd="00003")
    url, params = client.calls[0]
    assert url.endswith("/nwis/dv/") and params["statCd"] == "00003" and params["parameterCd"] == "00010,00400"
    assert params["sites"] == "01646500"


# ── studies, gates, the Analyst ─────────────────────────────────────────────


def test_a_study_step_runs_wqi_on_the_samples_of_an_earlier_step():
    samples = {"source": "usgs", "station_id": "USGS-1", "unit": "mg/l", "n_samples": 24,
               "sample_counts": {"pH": 12, "Nitrate": 12},
               "samples": [{"datetime": d.isoformat(), "parameter": p, "value": v, "unit": u}
                           for d in IDX for p, v, u in (("pH", 7.2, ""), ("Nitrate", 20.0, "mg/L"))]}
    study = Study(question="q", version=2, steps=[
        Step(tool="water_quality_samples", id="s1", arguments={"source": "usgs", "station_id": "USGS-1"},
             expects=[{"check": "not_empty", "path": "samples"}, {"check": "unit_present"}]),
        Step(tool="wqi", id="s2", arguments={"from_step": "s1", "use": "drinking"},
             expects=[{"check": "not_empty", "path": "ccme.score"},
                      {"check": "min_samples", "value": 4, "path": "ccme.sample_counts"}]),
        Step(tool="iwqi", id="s3", arguments={"from_step": "s1"}),
    ])
    run = run_study(study, tools={"water_quality_samples": lambda **kw: samples})
    assert run.ok and all(g["passed"] for g in run.gates) and len(run.gates) == 4
    wqi_result = run.results[1]["result"]
    assert wqi_result["score"] == 100.0 and wqi_result["ccme"]["n_variables"] == 2
    assert run.results[2]["result"]["restriction"] == "none"
    assert {"wqi", "iwqi", "water_quality_samples"} <= set(tool_names())


def test_the_min_samples_gate_names_the_thin_parameters():
    from aquascope.gates import CHECKS, evaluate

    assert "min_samples" in CHECKS
    thin = evaluate([{"check": "min_samples", "value": 4}], {"sample_counts": {"ph": 12, "arsenic": 2}})[0]
    assert not thin["passed"] and "arsenic (2)" in thin["detail"]
    ok = evaluate([{"check": "min_samples", "value": 4, "path": "n"}], {"n": 9})[0]
    assert ok["passed"]
    none = evaluate([{"check": "min_samples", "value": 4}], {"n": 9})[0]
    assert not none["passed"] and "no sample counts" in none["detail"]


def test_the_analyst_and_mcp_expose_the_samples_tool():
    from aquascope.ai_engine.analyst import _tool_specs

    specs = {s.name: s for s in _tool_specs()}
    assert "water_quality_samples" in specs
    assert set(specs["water_quality_samples"].parameters["required"]) == {"source", "station_id"}
    assert "wqi" in specs["analyse_table"].description and "iwqi" in specs["analyse_table"].description
    assert mcp_server.water_quality_samples.__doc__ and "screening" in mcp_server.water_quality_samples.__doc__


def test_the_template_narrator_has_sentences_for_the_water_quality_tools():
    study = Study(question="q", version=2, steps=[])
    samples = {"source": "usgs", "station_id": "USGS-1", "n_samples": 24, "n_parameters": 2, "start": "2024-01-01",
               "end": "2024-12-01",
               "parameters": {"pH": {"n": 12, "unit": "std units"}, "Nitrate": {"n": 12, "unit": "mg/L"}}}
    text = " ".join(team._sentences_for("water_quality_samples", samples, study))
    assert "24 water-quality samples" in text and "usgs USGS-1" in text and "Nitrate 12 in mg/L" in text
    screen = {"rows": [{"parameter": "nitrate", "rule": "at most 50 mg/L", "n": 12, "n_exceed": 12, "pct": 100.0,
                        "status": "Alert"}], "n_alerts": 1, "n_warnings": 0}
    text = " ".join(team._sentences_for("who_screen", screen, study))
    assert "1 alert(s)" in text and "12 of 12 samples" in text and "at most 50 mg/L" in text
    wqi = {"guideline_set": "drinking", "ccme": {"score": 83.5, "category": "Good", "n_variables": 5, "n_tests": 60,
                                                 "n_failed_tests": 12, "f1": 20.0, "f2": 20.0, "f3": 3.85,
                                                 "drivers": [{"parameter": "nitrate", "n_failed": 12, "n": 12,
                                                              "guideline": "at most 50 mg/L"}],
                                                 "meets_minimum_design": True},
           "nsf": {"score": None, "n_parameters": 4, "missing": ["bod", "fecal_coliform"]}}
    text = " ".join(team._sentences_for("wqi", wqi, study))
    assert "83.5 out of 100, Good" in text and "nitrate 12 of 12 samples outside at most 50 mg/L" in text
    assert "NSF index was not computed" in text and "missing bod, fecal coliform" in text
    iwqi = {"restriction": "severe", "drivers": ["infiltration"], "missing": ["boron"],
            "indices": {"sar": {"value": 12.0, "class": "S2"}, "sodium_percent": {"value": 80.0, "class": "unsuitable"},
                        "rsc": {"value": None}, "ussl_class": "C2-S2"}}
    text = " ".join(team._sentences_for("iwqi", iwqi, study))
    assert "severe restriction on use, driven by infiltration" in text and "SAR 12 (S2)" in text
    assert "USSL class C2-S2" in text and "not judged: boron" in text
    assert team.choose_playbook("Is the river water safe to drink?") == ("water_quality", False)
    assert team.intake_hints("Is the river water safe to drink?", "water_quality") == {"use": "drinking"}
    assert team.intake_hints("water quality for irrigating the orchard", "water_quality")["use"] == "irrigation"


# ── the team, keyless ───────────────────────────────────────────────────────

WQ_STATION = {"source": "usgs", "station_id": "USGS-01646500", "name": "Potomac at Little Falls", "distance_km": 7.2,
              "variables": ["discharge", "water_quality"], "years": 96.5}
WQ_RECON = {"point": {"lat": 38.9, "lon": -77.05}, "stations": [WQ_STATION], "catchment": {"upstream_area_km2": 29940},
            "context": {"years_by_variable": {"discharge": 96.5, "water_quality": 96.5},
                        "resolution_by_variable": {"discharge": "daily", "water_quality": "daily"},
                        "area_km2": 29940, "donors": 8, "available": ["glofas"], "ungauged": False},
            "sufficiency": [], "notes": ["Record resolution is not in the catalog; daily is assumed."]}
NO_WQ_RECON = {**WQ_RECON, "stations": [dict(WQ_STATION, variables=["discharge"])],
               "context": {**WQ_RECON["context"], "years_by_variable": {"discharge": 96.5}}}


def _samples_payload() -> dict:
    rows = []
    for d in pd.date_range("2021-01-01", periods=60, freq="MS"):
        rows += [{"datetime": d.isoformat(), "parameter": "pH", "value": 7.6, "unit": "std units"},
                 {"datetime": d.isoformat(), "parameter": "DO", "value": 9.5, "unit": "mg/l"},
                 {"datetime": d.isoformat(), "parameter": "Temperature", "value": 14.0, "unit": "deg C"},
                 {"datetime": d.isoformat(), "parameter": "Conductivity", "value": 320.0, "unit": "uS/cm @25C"}]
    return {"source": "usgs", "station_id": "USGS-01646500", "agency": "U.S. Geological Survey", "license": "US-PD",
            "attribution": "U.S. Geological Survey (public domain)", "n_samples": 240, "n_parameters": 4,
            "parameters": {"Conductivity": {"n": 60, "unit": "uS/cm @25C"}, "DO": {"n": 60, "unit": "mg/l"},
                           "Temperature": {"n": 60, "unit": "deg C"}, "pH": {"n": 60, "unit": "std units"}},
            "sample_counts": {"Conductivity": 60, "DO": 60, "Temperature": 60, "pH": 60},
            "unit": "mg/l", "start": "2021-01-01", "end": "2025-12-01", "years": 4.92, "samples": rows,
            "fetch_note": "USGS daily mean values", "methods": []}


def _solve(problem: str, recon: dict, **kwargs):
    import aquascope.explore

    calls: list = []

    def samples(**kw):
        calls.append(kw)
        return _samples_payload()

    with patch.object(aquascope.explore, "assess_site", return_value=recon):
        res = team.solve(problem, lat=38.9, lon=-77.05, playbook="water_quality",
                         tools={"water_quality_samples": samples}, **kwargs)
    return res, calls


def test_keyless_solve_runs_the_water_quality_playbook_end_to_end():
    res, calls = _solve("Is the river water safe to drink?", WQ_RECON)
    assert not res.declined and res.ok and res.cost == {} and res.model is None
    assert res.study.plan["playbook"] == "water_quality" and res.study.plan["branch"] == "drinking"
    assert res.problem["params"]["use"] == "drinking" and res.problem["params"]["health_verdict"] is False
    assert calls == [{"source": "usgs", "station_id": "USGS-01646500", "years": 5, "use": "drinking"}]
    assert [r["tool"] for r in res.run.results] == ["water_quality_samples", "who_screen", "wqi"]
    assert all(g["passed"] for g in res.gates) and [g["check"] for g in res.gates] == [
        "not_empty", "unit_present", "not_empty", "min_samples"]
    wqi_result = res.run.results[2]["result"]
    assert wqi_result["score"] == 100.0 and wqi_result["ccme"]["n_variables"] == 2, "pH and DO have WHO values"
    assert "CCME Water Quality Index 1.0" in res.answer and "100 out of 100, Excellent" in res.answer
    assert "USGS-01646500" in res.answer and "NSF index was not computed" in res.answer
    assert all(c["passed"] for c in res.checks), [c for c in res.checks if not c["passed"]]
    md = res.to_markdown()
    for needle in ("## Caveats", "not a", "verdict", "digitised approximations", "WHO (2022)", "## Data",
                   "usgs / USGS-01646500", "US-PD", "CCME (2001)", "Brown, R. M.", "Model calls: 0"):
        assert needle in md, needle
    assert not res.not_established, res.not_established


def test_keyless_solve_for_irrigation_adds_the_suitability_index():
    res, calls = _solve("Is this river water fit for irrigating the orchard?", WQ_RECON, intake={"years": 3})
    assert res.study.plan["branch"] == "irrigation" and calls[0]["use"] == "irrigation" and calls[0]["years"] == 3
    assert [r["tool"] for r in res.run.results] == ["water_quality_samples", "wqi", "iwqi"]
    assert res.ok and all(g["passed"] for g in res.gates)
    iwqi_result = res.run.results[2]["result"]
    assert iwqi_result["restriction"] == "none" and iwqi_result["drivers"] == [], "EC 0.32 dS/m and pH only"
    assert iwqi_result["indices"]["sar"]["value"] is None and "sodium" in iwqi_result["missing"]
    assert "Irrigation suitability (FAO 29): no restriction on use from the sampled parameters" in res.answer
    assert "Not sampled, so not judged" in res.answer and "sodium" in res.answer
    assert any("FAO Irrigation and Drainage Paper 29" in c for c in res.caveats)


def test_solve_declines_without_sampled_parameters_within_reach():
    res, calls = _solve("Is the river water safe to drink?", NO_WQ_RECON)
    assert res.declined and calls == [] and res.run is None or (res.run is not None and not res.run.results)
    assert "Phase 3 (#188)" in res.declined_reason and "Water Quality Portal" in res.declined_reason
    assert "**Declined.**" in res.to_markdown()
    verdict, _ = _solve("Is the river water safe to drink?", WQ_RECON, intake={"health_verdict": True})
    assert verdict.declined and "health judgement" in verdict.declined_reason


def test_the_who_screen_reads_agency_labels_and_units_through_the_shared_vocabulary():
    """USGS labels dissolved oxygen "DO" and WQP reports arsenic in ug/L: the screen has to see both."""
    df = pd.DataFrame({"sample_datetime": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
                       "parameter": ["DO"] * 4 + ["Arsenic"] * 4, "value": [9.0, 3.0, 8.0, 8.5, 2.0, 20.0, 5.0, 1.0],
                       "unit": ["mg/l"] * 4 + ["ug/L"] * 4})
    rows = {r["parameter"]: r for r in wb.who_screen(df)["rows"]}
    assert rows["dissolved_oxygen"]["n_exceed"] == 1 and rows["dissolved_oxygen"]["rule"] == "at least 5.0 mg/L"
    assert rows["arsenic"]["n_exceed"] == 1 and rows["arsenic"]["status"] == "Alert", "20 ug/L is 0.02 mg/L"
    saturation = pd.DataFrame({"parameter": ["DO"], "value": [95.0], "unit": ["% saturation"]})
    assert wb.who_screen(saturation)["rows"] == [], "percent saturation is not a concentration"
    from aquascope.utils.parameters import convert_value, resolve_parameter

    assert resolve_parameter("00300", "mg/l") == ("dissolved_oxygen", 1.0)
    assert convert_value("arsenic", 20.0, "ug/L") == (0.02, None)
    assert convert_value("lead", 1.0, "furlongs") == (None, "furlongs")
