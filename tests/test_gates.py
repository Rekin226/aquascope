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
                           "not_empty", "unit_present", "max_area_km2", "min_donors", "status_is", "min_samples"}


def test_empty_expects_is_no_gate():
    assert evaluate([], PAYLOAD) == [] and evaluate(None, PAYLOAD) == []
