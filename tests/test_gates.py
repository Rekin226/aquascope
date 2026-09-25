"""Gates: every check in the vocabulary, passing and failing, over a tool payload."""

from __future__ import annotations

import pytest

from aquascope.gates import CHECKS, evaluate, resolve_path

PAYLOAD = {
    "years": 39.5,
    "unit": "m3/s",
    "n": 14555,
    "trend": {"p_value": 0.4},
    "ffa": {
        "return_periods": [2, 5, 10, 25, 50, 100],
        "fits": {
            "gev_lmoments": {"q": [250, 330, 380, 440, 480, 520]},
            "lp3": {"q": [252, 335, 388, 452, 500, 548], "ci": [[200, 300]] * 5 + [[410, 690]]},
            "gev_bootstrap": {"q": [250, 330, 380, 440, 480, 520], "ci": [[210, 290]] * 5 + [[None, 650]]},
        },
    },
    "stations": [{"source": "usgs", "station_id": "1"}, {"source": "usgs", "station_id": "2"}],
    "k": 2,
    "validation": {"nse": 0.71, "kge": 0.55},
    "attributes": {"upstream_area_km2": 101033.0},
    "sufficiency": [{"method": "gr4j_calibration", "status": "not_defensible"},
                    {"method": "flow_duration", "status": "defensible"}],
    "sample_counts": {"ph": 12, "nitrate": 3},
}


def _one(check: dict, payload=PAYLOAD) -> dict:
    rows = evaluate([check], payload)
    assert len(rows) == 1 and set(rows[0]) >= {"check", "passed", "detail"}
    return rows[0]


def test_paths_take_dots_indexes_and_selectors():
    assert resolve_path(PAYLOAD, "ffa.fits.lp3.q.5") == 548
    assert resolve_path(PAYLOAD, "ffa.fits.lp3.q[5]") == 548
    assert resolve_path(PAYLOAD, "ffa.fits.lp3.q[-1]") == 548
    assert resolve_path(PAYLOAD, "stations[1].station_id") == "2"
    assert resolve_path(PAYLOAD, "sufficiency[method=gr4j_calibration].status") == "not_defensible"
    assert resolve_path(PAYLOAD, "nope.deeper") is None
    assert resolve_path(PAYLOAD, "stations[9]") is None
    assert resolve_path(PAYLOAD, None) is PAYLOAD


@pytest.mark.parametrize("check, ok, bad", [
    ({"check": "min_years", "value": 20}, True, {"check": "min_years", "value": 40}),
    ({"check": "max_return_period_factor", "value": 3, "return_period": 100}, True,
     {"check": "max_return_period_factor", "value": 2, "return_period": 100}),
    ({"check": "ci_finite", "path": "ffa.fits.lp3.ci", "return_period": 100}, True,
     {"check": "ci_finite", "path": "ffa.fits.gev_bootstrap.ci", "return_period": 100}),
    ({"check": "spread_within", "value": 0.25, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
      "return_period": 100}, True,
     {"check": "spread_within", "value": 0.02, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
      "return_period": 100}),
    ({"check": "nse_min", "value": 0.5, "path": "validation.nse"}, True,
     {"check": "nse_min", "value": 0.9, "path": "validation.nse"}),
    ({"check": "kge_min", "value": 0.5, "path": "validation.kge"}, True,
     {"check": "kge_min", "value": 0.6, "path": "validation.kge"}),
    ({"check": "not_empty", "path": "trend"}, True, {"check": "not_empty", "path": "glofas"}),
    ({"check": "unit_present"}, True, {"check": "unit_present", "path": "trend.unit"}),
    ({"check": "max_area_km2", "value": 200000, "path": "attributes.upstream_area_km2"}, True,
     {"check": "max_area_km2", "value": 10000, "path": "attributes.upstream_area_km2"}),
    ({"check": "min_donors", "value": 2, "path": "stations"}, True, {"check": "min_donors", "value": 3, "path": "k"}),
    ({"check": "status_is", "path": "sufficiency[method=flow_duration].status", "value": "defensible"}, True,
     {"check": "status_is", "path": "sufficiency[method=gr4j_calibration].status",
      "value": ["defensible", "marginal"]}),
    ({"check": "min_samples", "value": 3}, True, {"check": "min_samples", "value": 4}),
])
def test_every_check_passes_and_fails(check, ok, bad):
    assert ok
    good = _one(check)
    assert good["passed"], good
    failed = _one(bad)
    assert not failed["passed"], failed
    assert failed["detail"]


def test_details_quote_the_numbers_a_reader_needs():
    row = _one({"check": "max_return_period_factor", "value": 3, "return_period": 200})
    assert not row["passed"] and "T = 200" in row["detail"] and "118" in row["detail"]
    row = _one({"check": "spread_within", "value": 0.25, "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
                "return_period": 100})
    assert "5%" in row["detail"] and "T = 100" in row["detail"]
    row = _one({"check": "max_area_km2", "value": 10000, "path": "attributes.upstream_area_km2"})
    assert "101,033" in row["detail"] and "ceiling" in row["detail"]


def test_return_period_lookup_needs_the_fitted_periods():
    row = _one({"check": "ci_finite", "path": "ffa.fits.lp3.ci", "return_period": 500})
    assert not row["passed"] and "500" in row["detail"]
    row = _one({"check": "ci_finite", "path": "ci", "return_period": 100}, {"ci": [[1, 2]]})
    assert not row["passed"] and "return_periods" in row["detail"]


def test_an_error_payload_fails_every_gate_except_status():
    rows = evaluate([{"check": "min_years", "value": 1}, {"check": "not_empty", "path": "x"}], {"error": "boom"})
    assert [r["passed"] for r in rows] == [False, False] and "boom" in rows[0]["detail"]


def test_unknown_or_malformed_gates_fail_loudly():
    rows = evaluate([{"check": "nope"}, {"value": 3}, "min_years"], PAYLOAD)
    assert [r["passed"] for r in rows] == [False, False, False]
    assert "unknown check" in rows[0]["detail"] and "min_years" in rows[0]["detail"]
    assert set(CHECKS) == {"min_years", "max_return_period_factor", "ci_finite", "spread_within", "nse_min", "kge_min",
                           "not_empty", "unit_present", "max_area_km2", "min_donors", "status_is", "min_samples",
                           "fit_envelopes_max", "sampling_density", "trend_on_series", "cross_check_ratio"}


def test_empty_expects_is_no_gate():
    assert evaluate([], PAYLOAD) == [] and evaluate(None, PAYLOAD) == []


# ── reviewer's-eye checks (#416) ──

FFA = {"record_max": {"value": 800.0, "year": 1894, "empirical_return_period": 144.0, "n_years": 143},
       "return_periods": [2, 5, 10, 25, 50, 100],
       "fits": {"gev_lmoments": {"q": [300, 400, 460, 540, 600, 652], "at_record_max": 690.0,
                                 "q_by_T": {"100": 652.0}},
                "lp3": {"q": [310, 405, 470, 550, 590, 624], "at_record_max": 660.0}},
       "amax_trend": {"on": "annual maxima", "p_value": 0.41, "tau": 0.05}}


def test_fit_envelopes_max_compares_the_record_maximum_with_the_fit_at_its_return_period():
    ok = evaluate([{"check": "fit_envelopes_max", "path": "ffa", "value": 0.25}], {"ffa": FFA})[0]
    assert ok["passed"] and "ratio 1.16" in ok["detail"] and "1894" in ok["detail"] and "144" in ok["detail"]
    bad = evaluate([{"check": "fit_envelopes_max", "path": "ffa", "value": 0.10}], {"ffa": FFA})[0]
    assert not bad["passed"] and "sits above the fit" in bad["detail"]
    lp3 = evaluate([{"check": "fit_envelopes_max", "path": "ffa", "value": 0.25, "fit": "lp3"}], {"ffa": FFA})[0]
    assert lp3["passed"] and "lp3" in lp3["detail"] and "ratio 1.21" in lp3["detail"]
    none = evaluate([{"check": "fit_envelopes_max", "path": "ffa"}], {"ffa": {"fits": {}}})[0]
    assert not none["passed"] and "no record maximum" in none["detail"]


def test_sampling_density_refuses_a_resolution_the_record_cannot_carry():
    sparse = {"sampling": {"n": 227, "span_years": 48.7, "per_year": 4.66, "inferred_resolution": "sparse"}}
    daily = {"sampling": {"n": 51943, "span_years": 142.9, "per_year": 363.5, "inferred_resolution": "daily"}}
    bad = evaluate([{"check": "sampling_density", "value": "daily"}], sparse)[0]
    assert not bad["passed"] and "4.66 a year" in bad["detail"] and "daily claimed" in bad["detail"]
    assert "sparser than the resolution assumed" in bad["detail"]
    ok = evaluate([{"check": "sampling_density", "value": "daily"}], daily)[0]
    assert ok["passed"] and "about daily" in ok["detail"]
    monthly = evaluate([{"check": "sampling_density", "value": "monthly"}], sparse)[0]
    assert not monthly["passed"], "five a year is not monthly either"
    rate = evaluate([{"check": "sampling_density", "value": 4}], sparse)[0]
    assert rate["passed"] and "4 a year needed" in rate["detail"]
    assert not evaluate([{"check": "sampling_density", "value": "daily"}], {})[0]["passed"]


def test_trend_on_series_reads_the_test_on_the_series_it_names():
    ok = evaluate([{"check": "trend_on_series", "value": 0.05}], {"ffa": FFA})[0]
    assert ok["passed"] and "annual maxima" in ok["detail"] and "p = 0.41" in ok["detail"]
    trending = {"ffa": {"amax_trend": {"on": "annual maxima", "p_value": 0.004, "tau": 0.31}}}
    bad = evaluate([{"check": "trend_on_series", "value": 0.05}], trending)[0]
    assert not bad["passed"] and "significant trend in the annual maxima" in bad["detail"] and "caveat" in bad["detail"]
    other = evaluate([{"check": "trend_on_series", "path": "trend", "value": 0.05}],
                     {"trend": {"on": "annual mean", "p_value": 0.9, "tau": 0.0}})[0]
    assert other["passed"] and "annual mean" in other["detail"]
    assert not evaluate([{"check": "trend_on_series"}], {"ffa": {}})[0]["passed"]


def test_cross_check_ratio_compares_a_cross_check_with_a_reference_number():
    gev = {"q": [200, 540], "q_by_T": {"2": 200, "100": 540}}
    payload = {"glofas": {"ffa": {"return_periods": [2, 100], "fits": {"gev_lmoments": gev}}}}
    gate = {"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T", "value": 0.5,
            "reference": {"100": 652.0}, "return_period": 100}
    ok = evaluate([gate], payload)[0]
    assert ok["passed"] and "ratio 0.83" in ok["detail"] and "T = 100" in ok["detail"]
    far = evaluate([dict(gate, reference={"100": 2000.0})], payload)[0]
    assert not far["passed"] and "disagrees" in far["detail"]
    assert "within the allowed factor of 1.50" in ok["detail"]
    assert "outside the allowed factor of 1.50" in far["detail"] and "within" not in far["detail"]
    scalar = evaluate([{"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q",
                        "reference": 600.0, "return_period": 100, "value": 0.5}], payload)[0]
    assert scalar["passed"] and "540" in scalar["detail"]
    missing = evaluate([dict(gate, reference=None)], payload)[0]
    assert not missing["passed"] and "did not resolve" in missing["detail"]
