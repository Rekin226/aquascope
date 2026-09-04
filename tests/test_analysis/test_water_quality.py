"""Water quality indices (#62): the CCME worked example to the published number, NSF over the nine
parameters with digitised curves, FAO 29 irrigation classes at their boundaries, and unit handling."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aquascope.analysis import water_quality as wq

DATES = pd.to_datetime([
    "1997-01-07", "1997-02-04", "1997-03-04", "1997-04-08", "1997-05-06", "1997-06-03",
    "1997-07-08", "1997-08-05", "1997-09-02", "1997-10-07", "1997-11-04", "1997-12-01",
])

#: Table 1 of the CCME WQI 1.0 User's Manual: North Saskatchewan River at Devon, 1997 ("L" values kept
#: at the detection limit, as the manual counts them). Ten variables, 103 tests, four exceedances.
DEVON_1997 = {
    ("DO", "mg/L"): [11.4, 11.0, 11.5, 12.5, 10.4, 8.9, 8.5, 7.5, 9.2, 11.0, 12.1, 13.3],
    ("pH", ""): [8.0, 7.9, 7.9, 7.9, 8.1, 8.2, 8.3, 8.2, 8.2, 8.1, 8.0, 8.0],
    ("TP", "mg/L"): [0.006, 0.005, 0.006, 0.0581, 0.042, 0.108, 0.017, 0.008, 0.006, 0.008, 0.006, 0.004],
    ("TN", "mg/L"): [0.160, 0.170, 0.132, 0.428, 0.250, 0.707, 0.153, 0.153, 0.130, 0.093, 0.296, 0.054],
    ("FC", "#/dL"): [4, 42, 4, 4, 4, 26, 9, 8, 12, 12, 8, 4],
    ("As", "mg/L"): [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0006, 0.0002, 0.0002, 0.0003, 0.0002, 0.0002,
                     0.0002],
    ("Pb", "mg/L"): [0.0004, 0.0094, 0.0003, 0.0008, 0.0008, 0.0013, 0.0004, 0.0003, 0.0018, 0.0011, 0.0051,
                     0.0003],
    ("Hg", "ug/L"): [0.05] * 6 + [None] + [0.05] * 5,
    ("2,4-D", "ug/L"): [0.005, None, None, 0.004, None, None, None, 0.005, None, 0.005, None, None],
    ("Lindane", "ug/L"): [0.005, None, None, 0.005, None, None, None, 0.005, None, 0.005, None, None],
}
DEVON_GUIDELINES = {
    "DO": {"min": 5, "unit": "mg/L"}, "pH": {"min": 6.5, "max": 9.0}, "TP": {"max": 0.05, "unit": "mg/L"},
    "TN": {"max": 1, "unit": "mg/L"}, "FC": {"max": 400, "unit": "#/dL"}, "As": {"max": 0.05, "unit": "mg/L"},
    "Pb": {"max": 0.004, "unit": "mg/L"}, "Hg": {"max": 0.1, "unit": "ug/L"}, "2,4-D": {"max": 4, "unit": "ug/L"},
    "Lindane": {"max": 0.01, "unit": "ug/L"},
}


def _long(table: dict, dates=DATES) -> pd.DataFrame:
    rows = []
    for (param, unit), values in table.items():
        for d, v in zip(dates, values):
            if v is not None:
                rows.append({"sample_datetime": d, "parameter": param, "value": v, "unit": unit})
    return pd.DataFrame(rows)


def _block(param: str, values, unit: str, n: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="MS")
    vals = list(values) if not isinstance(values, (int, float)) else [values] * n
    return pd.DataFrame({"sample_datetime": idx, "parameter": param, "value": vals, "unit": unit})


def _strict(result: dict) -> None:
    json.dumps(result, allow_nan=False)


# ── CCME WQI 1.0 ────────────────────────────────────────────────────────────


def test_ccme_reproduces_the_manual_s_worked_example():
    """CCME (2001) User's Manual: F1 = 20, F2 = 3.9, F3 = 2.8, WQI = 88, Good."""
    res = wq.wqi_ccme(_long(DEVON_1997), DEVON_GUIDELINES)
    assert res["n_variables"] == 10 and res["n_tests"] == 103
    assert res["n_failed_variables"] == 2 and res["n_failed_tests"] == 4
    assert res["f1"] == 20.0 and res["f2"] == pytest.approx(3.88, abs=0.01)
    assert res["f3"] == pytest.approx(2.8, abs=0.05)
    assert round(res["score"]) == 88 and res["category"] == "Good"
    assert [d["parameter"] for d in res["drivers"]] == ["lead", "total_phosphorus"]
    assert res["meets_minimum_design"] is True and res["guideline_set"] == "custom"
    assert res["period"] == {"start": "1997-01-07", "end": "1997-12-01", "years": 0.9}
    assert res["sample_counts"]["mercury"] == 11 and res["sample_counts"]["2,4-d"] == 4
    assert res["input"]["unrecognised"] == [] and res["input"]["unconvertible_units"] == {}
    assert "CCME (2001)" in res["citation"]
    _strict(res)


def test_ccme_categories_span_the_five_bands():
    assert [wq.ccme_category(s) for s in (100, 95, 94.9, 80, 79.9, 65, 64.9, 45, 44.9, 0)] == [
        "Excellent", "Excellent", "Good", "Good", "Fair", "Fair", "Marginal", "Marginal", "Poor", "Poor"]


def test_ccme_over_the_shipped_drinking_set_is_perfect_when_nothing_exceeds():
    df = pd.concat([_block("pH", 7.4, ""), _block("Nitrate", 8.0, "mg/L"), _block("Arsenic", 2.0, "ug/L"),
                    _block("Turbidity", 1.0, "NTU"), _block("Conductivity", 400, "uS/cm")])
    res = wq.wqi_ccme(df, "drinking")
    assert res["score"] == 100.0 and res["category"] == "Excellent"
    assert res["n_variables"] == 4, "conductivity has no WHO guideline and is not counted"
    assert res["guideline_set"] == "drinking" and res["sample_counts"] == {
        "arsenic": 12, "nitrate": 12, "ph": 12, "turbidity": 12}
    assert any("sampled parameters that have a guideline" in n for n in res["notes"])


def test_ccme_a_minimum_guideline_and_a_zero_guideline_count_as_excursions():
    df = pd.concat([_block("DO", [2.5] * 6 + [8.0] * 6, "mg/L"), _block("E. coli", [0] * 11 + [10], "CFU/100mL"),
                    _block("pH", 7.0, ""), _block("Lead", 0.001, "mg/L")])
    res = wq.wqi_ccme(df, "drinking")
    do = next(v for v in res["variables"] if v["parameter"] == "dissolved_oxygen")
    assert do["n_failed"] == 6 and do["worst_excursion"] == pytest.approx(1.0, abs=0.01)   # 5 / 2.5 - 1
    ec = next(v for v in res["variables"] if v["parameter"] == "e_coli")
    assert ec["n_failed"] == 1 and ec["worst_excursion"] == pytest.approx(9.0)             # against 1 CFU
    assert res["f1"] == 50.0 and res["f2"] == pytest.approx(100 * 7 / 48, abs=0.01)
    assert res["f3"] == pytest.approx(23.8, abs=0.1)          # nse = (6 x 1.0 + 9) / 48
    assert res["score"] == pytest.approx(66.9, abs=0.1) and res["category"] == "Fair"


def test_ccme_says_when_there_is_nothing_to_index_and_when_the_design_is_thin():
    empty = wq.wqi_ccme(pd.DataFrame({"parameter": ["zzz"], "value": [1.0], "unit": [""]}), "drinking")
    assert empty["score"] is None and empty["category"] is None and empty["n_variables"] == 0
    assert "not computed" in empty["notes"][0]
    thin = wq.wqi_ccme(pd.concat([_block("pH", 7.0, "", n=2), _block("Nitrate", 60.0, "mg/L", n=2)]), "drinking")
    assert thin["score"] is not None and thin["meets_minimum_design"] is False
    assert any("at least 4" in n for n in thin["notes"]) and any("Fewer than 4 samples" in n for n in thin["notes"])
    _strict(thin)


def test_ccme_guideline_sets_are_named_and_a_custom_table_is_keyed_by_any_spelling():
    assert set(wq.GUIDELINE_SETS) == {"drinking", "irrigation", "aquatic life"}
    assert wq.guideline_set("aquatic_life") is wq.CCME_AQUATIC_LIFE_GUIDELINES
    with pytest.raises(ValueError, match="unknown guideline set"):
        wq.guideline_set("bathing")
    # the WHO screen's values are the drinking set's
    assert wq.WHO_DRINKING_GUIDELINES["nitrate"] == {"max": 50.0, "unit": "mg/L"}
    assert wq.WHO_DRINKING_GUIDELINES["ph"] == {"min": 6.5, "max": 8.5, "unit": "pH units"}
    assert wq.WHO_DRINKING_GUIDELINES["dissolved_oxygen"] == {"min": 5.0, "unit": "mg/L"}
    # a bound given with its own unit is converted like a sample
    table = wq._normalise_guidelines({"Total phosphorus": {"max": 50, "unit": "ug/L"}, "Nitrate-N": {"max": 10}})
    assert table["total_phosphorus"]["max"] == pytest.approx(0.05)
    assert table["nitrate"]["max"] == pytest.approx(44.268)


# ── NSF WQI ─────────────────────────────────────────────────────────────────

NINE = lambda: pd.concat([  # noqa: E731 - a fresh frame per test
    _block("DO", 9.0, "mg/L"), _block("Temperature", 15.0, "deg C"), _block("pH", 7.3, "std units"),
    _block("Fecal coliform", 5, "cfu/100mL"), _block("BOD5", 1.0, "mg/L"), _block("Nitrate", 1.0, "mg/L"),
    _block("Total phosphate", 0.05, "mg/L"), _block("Turbidity", 2.0, "NTU"), _block("Total solids", 80, "mg/L"),
    _block("Temperature change", 0.5, "deg C"),
])


def test_nsf_weights_sum_to_one_and_every_parameter_has_a_curve():
    assert sum(wq.NSF_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(wq.NSF_WEIGHTS) == set(wq.NSF_CURVES)
    assert wq.nsf_sub_index("ph", 7.3) == 93.0 and wq.nsf_sub_index("ph", 2.0) == 0.0
    assert wq.nsf_sub_index("fecal_coliform", 10) == 80.0 and wq.nsf_sub_index("fecal_coliform", 0.1) == 97.0
    assert wq.nsf_sub_index("temperature_change", -5) == wq.nsf_sub_index("temperature_change", 5) == 75.0


def test_nsf_with_all_nine_parameters_at_clean_values_scores_high_and_says_the_curves_are_digitised():
    res = wq.wqi_nsf(NINE())
    assert res["complete"] is True and res["missing"] == [] and res["n_parameters"] == 9
    assert res["score"] >= 90 and res["category"] == "Excellent"
    do = res["sub_indices"]["dissolved_oxygen_saturation"]
    assert do["value"] == pytest.approx(89.7, abs=0.2), "9 mg/L at 15 deg C is about 90 % saturation"
    assert "digitised approximations" in res["notes"][-1] and "Brown" in res["citation"]
    _strict(res)


def test_nsf_at_guideline_values_gives_a_mid_score():
    """Every parameter at (about) its drinking or surface-water guideline value: a medium water."""
    df = pd.concat([
        _block("DO", 5.0, "mg/L"), _block("Temperature", 20.0, "deg C"), _block("pH", 8.5, ""),
        _block("Fecal coliform", 200, "cfu/100mL"), _block("BOD", 5.0, "mg/L"), _block("Nitrate", 50.0, "mg/L"),
        _block("Total phosphate", 1.0, "mg/L"), _block("Turbidity", 5.0, "NTU"), _block("Total solids", 500, "mg/L"),
        _block("Temperature change", 5.0, "deg C"),
    ])
    res = wq.wqi_nsf(df)
    assert res["complete"] is True
    assert 40 <= res["score"] <= 65, res["score"]
    assert res["category"] in ("Medium", "Bad")
    assert [d["parameter"] for d in res["drivers"]][:2] == ["nitrate", "total_solids"] or res["drivers"]


def test_nsf_stand_ins_and_renormalised_weights_are_declared():
    df = pd.concat([_block("DO", 3.0, "mg/L"), _block("pH", 9.5, ""), _block("E. coli", 5000, "MPN/100mL"),
                    _block("BOD", 15.0, "mg/L"), _block("Nitrate", 40.0, "mg/l as N"), _block("TP", 0.8, "mg/L"),
                    _block("Turbidity", 60.0, "NTU"), _block("TDS", 600, "mg/L")])
    res = wq.wqi_nsf(df)
    assert res["complete"] is False and res["missing"] == ["temperature_change"] and res["weights_renormalised"]
    assert res["score"] < 35 and res["category"] in ("Bad", "Very bad")
    assert res["sub_indices"]["fecal_coliform"]["from"].startswith("E. coli")
    assert res["sub_indices"]["phosphate"]["value"] == pytest.approx(0.8 * 3.066, rel=1e-3)
    assert res["sub_indices"]["nitrate"]["value"] == pytest.approx(40 * 4.4268, rel=1e-3), "as N became NO3"
    text = " ".join(res["notes"])
    assert "20 deg C" in text and "renormalised" in text and "stand in" in text
    # a reference temperature turns the sampled temperature into the change parameter
    with_ref = wq.wqi_nsf(pd.concat([df, _block("Temperature", 24.0, "deg C")]), reference_temperature=20)
    assert with_ref["complete"] is True and with_ref["sub_indices"]["temperature_change"]["value"] == 4.0


def test_nsf_needs_five_parameters_to_report_a_score():
    res = wq.wqi_nsf(pd.concat([_block("pH", 7.0, ""), _block("Turbidity", 1.0, "NTU")]))
    assert res["score"] is None and res["category"] is None and res["n_parameters"] == 2
    assert any("fewer than 5" in n for n in res["notes"])
    assert wq.wqi_nsf(pd.DataFrame({"parameter": ["zzz"], "value": [1.0]}))["score"] is None


# ── irrigation (FAO 29) ─────────────────────────────────────────────────────

MEQ = wq.EQUIVALENT_WEIGHTS


def _ions(na=3.0, ca=1.0, mg=1.0, hco3=1.0, ec=700, **extra) -> pd.DataFrame:
    """Ions given in meq/L (as mg/L in the table), EC in uS/cm."""
    parts = [_block("Sodium", na * MEQ["sodium"], "mg/L"), _block("Calcium", ca * MEQ["calcium"], "mg/L"),
             _block("Magnesium", mg * MEQ["magnesium"], "mg/L"),
             _block("Bicarbonate", hco3 * MEQ["bicarbonate"], "mg/L"),
             _block("Specific conductance", ec, "uS/cm")]
    for name, (value, unit) in extra.items():
        parts.append(_block(name, value, unit))
    return pd.concat(parts)


def test_iwqi_computes_sar_sodium_percent_and_rsc_in_meq():
    res = wq.iwqi(_ions(na=3.0, ca=1.0, mg=1.0, hco3=3.25, ec=700))
    assert res["indices"]["sar"]["value"] == pytest.approx(3.0, abs=0.01)   # 3 / sqrt((1 + 1) / 2)
    assert res["indices"]["sodium_percent"]["value"] == pytest.approx(60.0, abs=0.1)
    assert res["indices"]["rsc"]["value"] == pytest.approx(1.25, abs=0.01)
    assert res["ions_meq_l"]["sodium"] == pytest.approx(3.0, abs=0.01)
    assert res["statistic"] == "median" and "potassium" in res["missing"] and "boron" in res["missing"]
    assert "Ayers" in res["citation"] and "Richards" in res["citation"]
    _strict(res)


@pytest.mark.parametrize("value, klass", [(1.24, "safe"), (1.25, "marginal"), (2.5, "marginal"), (2.51, "unsuitable")])
def test_rsc_classes_at_their_boundaries(value, klass):
    assert wq.rsc_class(value) == klass


@pytest.mark.parametrize("value, klass", [(19.9, "excellent"), (20, "good"), (40, "permissible"), (60, "doubtful"),
                                          (80, "unsuitable"), (95, "unsuitable")])
def test_sodium_percent_classes_at_their_boundaries(value, klass):
    assert wq.sodium_percent_class(value) == klass


@pytest.mark.parametrize("ec_us, restriction", [(699, "none"), (700, "slight to moderate"),
                                                (3000, "slight to moderate"), (3001, "severe")])
def test_fao_salinity_restriction_at_its_boundaries(ec_us, restriction):
    res = wq.iwqi(_ions(ec=ec_us))
    assert res["components"]["salinity_ec"]["restriction"] == restriction


def test_fao_infiltration_reads_sar_against_ec_and_the_worst_component_wins():
    # SAR 3 at EC 0.7 dS/m: infiltration none (EC above 0.7 is needed, 0.7 is not above): slight to moderate
    res = wq.iwqi(_ions(na=3.0, ca=1.0, mg=1.0, ec=700))
    assert res["components"]["infiltration"]["restriction"] == "slight to moderate"
    assert res["components"]["sodium_toxicity"]["restriction"] == "slight to moderate"   # SAR 3 is the boundary
    assert res["restriction"] == "slight to moderate" and "salinity_ec" in res["drivers"]
    # SAR 12 at EC 0.4 dS/m: severe infiltration hazard, severe sodium toxicity
    hot = wq.iwqi(_ions(na=12.0, ca=1.0, mg=1.0, ec=400))
    assert hot["components"]["infiltration"]["restriction"] == "severe"
    assert hot["restriction"] == "severe" and hot["drivers"] == ["infiltration", "sodium_toxicity"]
    assert hot["indices"]["sar"]["class"] == "S2" and hot["indices"]["ussl_class"] == "C2-S2"
    # low-salinity water: FAO 29 reads EC below 0.7 dS/m as no salinity hazard but, at SAR 0 to 3, as a
    # slight to moderate infiltration hazard (low-salt water disperses the soil); nothing else restricts
    clean = wq.iwqi(_ions(na=1.0, ca=2.0, mg=1.0, hco3=1.0, ec=690, Boron=(0.2, "mg/L"), Chloride=(50, "mg/L"),
                          pH=(7.5, ""), Nitrate=(5.0, "mg/L")))
    assert clean["restriction"] == "slight to moderate" and clean["drivers"] == ["infiltration"]
    assert {k: v["restriction"] for k, v in clean["components"].items() if k != "infiltration"} == {
        "salinity_ec": "none", "salinity_tds": None, "sodium_toxicity": "none", "chloride_toxicity": "none",
        "boron_toxicity": "none", "nitrate_nitrogen": "none", "bicarbonate": "none", "ph": "none"}
    assert clean["components"]["chloride_toxicity"]["value"] == pytest.approx(50 / 35.45, rel=1e-3)
    assert clean["components"]["nitrate_nitrogen"]["value"] == pytest.approx(5.0 / 4.4268, rel=1e-3)


def test_iwqi_says_what_it_cannot_judge():
    res = wq.iwqi(pd.DataFrame({"parameter": ["pH", "pH"], "value": [7.0, 7.2], "unit": ["", ""]}))
    assert res["restriction"] == "none" and res["indices"]["sar"]["value"] is None
    assert "sodium" in res["missing"] and any("SAR" in n for n in res["notes"])
    nothing = wq.iwqi(pd.DataFrame({"parameter": ["zzz"], "value": [1.0]}))
    assert nothing["restriction"] is None and any("no restriction is judged" in n for n in nothing["notes"])
    alk = wq.iwqi(_ions(hco3=0).pipe(lambda d: d[d["parameter"] != "Bicarbonate"])
                  .pipe(lambda d: pd.concat([d, _block("Alkalinity", 100, "mg/L")])))
    assert alk["ions_meq_l"]["bicarbonate"] == pytest.approx(100 / 50.04, rel=1e-3)


# ── parameters and units ────────────────────────────────────────────────────


@pytest.mark.parametrize("name, unit, key, factor", [
    ("Dissolved oxygen (DO)", "mg/l", "dissolved_oxygen", 1.0),
    ("Dissolved oxygen (DO)", "% saturation", "dissolved_oxygen_saturation", 1.0),
    ("Specific conductance", "uS/cm @25C", "conductivity", 1.0),
    ("Nitrate", "mg/l as N", "nitrate", 4.4268),
    ("Nitrate-N", "mg/L", "nitrate", 4.4268),
    ("Nitrate", "mg/L", "nitrate", 1.0),
    ("Escherichia coli", "MPN/100ml", "e_coli", 1.0),
    ("00400", "std units", "ph", 1.0),
    ("Temperature, water", "deg C", "temperature", 1.0),
    ("NH3-N", "mg/L", "ammonia", 1.0),
    ("Total dissolved solids", "mg/L", "tds", 1.0),
    ("Phosphorus, total", "mg/L", "total_phosphorus", 1.0),
    ("Something else", "mg/L", None, 1.0),
])
def test_parameter_names_resolve_to_the_vocabulary(name, unit, key, factor):
    assert wq.resolve_parameter(name, unit) == (key, factor)


def test_units_are_converted_or_the_sample_is_dropped_and_counted():
    df = pd.DataFrame({
        "sample_datetime": pd.to_datetime(["2024-01-01"] * 7),
        "parameter": ["Mercury", "Conductivity", "Temperature", "Nitrate", "Lead", "Turbidity", "Unknown thing"],
        "value": [5.0, 1.2, 68.0, 12.0, 3.0, 4.0, 1.0],
        "unit": ["ug/L", "mS/cm", "deg F", "mg/l as N", "furlongs", "NTU", "mg/L"],
    })
    frame, report = wq.normalise_samples(df)
    by = frame.set_index("parameter")["value"]
    assert by["mercury"] == pytest.approx(0.005) and frame.set_index("parameter")["unit"]["mercury"] == "mg/L"
    assert by["conductivity"] == pytest.approx(1200.0) and by["temperature"] == pytest.approx(20.0)
    assert by["nitrate"] == pytest.approx(53.12, abs=0.01)
    assert "lead" not in by.index and report["unconvertible_units"] == {"lead": {"furlongs": 1}}
    assert report["unrecognised"] == ["Unknown thing"] and report["n_in"] == 7 and report["n_used"] == 5
    # the WHO screen's mercury and nitrate values are compared in mg/L as NO3
    res = wq.wqi_ccme(df, "drinking")
    failed = {v["parameter"]: v["n_failed"] for v in res["variables"]}
    assert failed == {"mercury": 1, "nitrate": 1, "turbidity": 0}


def test_a_wide_table_and_missing_units_are_handled():
    wide = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="MS"), "pH": [7.0, 7.1, 7.2],
                         "Nitrate": [10.0, 20.0, 30.0], "notes": ["a", "b", "c"]})
    frame, report = wq.normalise_samples(wide)
    assert sorted(frame["parameter"].unique()) == ["nitrate", "ph"] and len(frame) == 6
    assert report["assumed_units"]["nitrate"] == "unit assumed mg/L"
    assert frame["datetime"].notna().all()
    empty, rep = wq.normalise_samples(pd.DataFrame())
    assert empty.empty and rep["n_in"] == 0
