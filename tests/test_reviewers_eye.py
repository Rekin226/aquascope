"""The payload side of the reviewer's-eye gates (#416): the record maximum and the fit at its return period,
the annual-maxima trend, the sampling block, a gate that references an earlier step, and the verify check
that holds a stationarity claim to the maxima."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from aquascope.ai_engine.verify import verify
from aquascope.explore import _sampling, analyze_series
from aquascope.study import Step, Study, run_study


def _daily(years: int = 40, seed: int = 7, trend: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1980-01-01", periods=365 * years, freq="D")
    base = np.exp(rng.normal(3.0, 0.6, len(idx)))
    drift = 1.0 + trend * (np.arange(len(idx)) / len(idx))
    return pd.Series(base * drift, index=idx)


def test_the_flood_payload_carries_the_record_maximum_the_fit_there_and_the_maxima_trend():
    out = analyze_series(_daily(), "discharge", "m3/s")
    ffa = out["ffa"]
    rec = ffa["record_max"]
    assert rec["n_years"] >= 38 and rec["empirical_return_period"] == rec["n_years"] + 1
    assert rec["value"] == max(out["annual_max"]["v"]) and rec["year"] in out["annual_max"]["year"]
    gev = ffa["fits"]["gev_lmoments"]
    # T_max = n + 1 (about 41 years) sits between the 25- and the 50-year quantiles
    assert gev["q"][3] < gev["at_record_max"] < gev["q"][4] and gev["q_by_T"]["100"] == gev["q"][-1]
    assert len(gev["q"]) == len(ffa["return_periods"]) == 6, "the extra return period is not in the list"
    assert ffa["fits"]["lp3"]["at_record_max"] > 0 and "100" in ffa["fits"]["lp3"]["q_by_T"]
    am = ffa["amax_trend"]
    assert am["on"] == "annual maxima" and 0 <= am["p_value"] <= 1 and am["n_years"] == rec["n_years"]
    assert out["sampling"]["inferred_resolution"] == "daily" and out["sampling"]["per_year"] > 350


def test_a_trending_record_shows_in_the_maxima_test_and_a_sparse_record_in_the_sampling_block():
    out = analyze_series(_daily(trend=3.0), "discharge", "m3/s")
    assert out["ffa"]["amax_trend"]["p_value"] < 0.05 and out["ffa"]["amax_trend"]["trend"] != "no trend"
    assert _sampling(227, 48.7) == {"n": 227, "span_years": 48.7, "per_year": 4.66,
                                    "inferred_resolution": "quarterly"}
    assert _sampling(10, 48.7)["inferred_resolution"] == "sparse"
    assert _sampling(600, 48.7)["inferred_resolution"] == "monthly"
    assert _sampling(30, 0.0)["per_year"] == 30.0


def test_a_gate_may_reference_an_earlier_steps_result():
    calls: list = []

    def fit(**kw):
        calls.append("fit")
        return {"ffa": {"fits": {"gev_lmoments": {"q_by_T": {"100": 652.0}}}}}

    def cross(**kw):
        calls.append("cross")
        return {"glofas": {"ffa": {"fits": {"gev_lmoments": {"q_by_T": {"100": 540.0}}}}}}

    study = Study(question="q", version=2, steps=[
        Step(tool="fit", id="s1"),
        Step(tool="cross", id="s2", expects=[{"check": "cross_check_ratio", "value": 0.5, "return_period": 100,
                                              "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                                              "reference": "{{ result.s1.ffa.fits.gev_lmoments.q_by_T }}"}]),
        Step(tool="cross", id="s3", expects=[{"check": "cross_check_ratio", "value": 0.5, "return_period": 100,
                                              "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                                              "reference": "{{ result.s9.nothing }}"}]),
    ])
    with patch("aquascope.study._tools", return_value={"fit": fit, "cross": cross}):
        run = run_study(study)
    g2 = run.results[1]["gates"][0]
    assert g2["passed"] and "ratio 0.83" in g2["detail"]
    g3 = run.results[2]["gates"][0]
    assert not g3["passed"] and "did not resolve" in g3["detail"], "an unresolvable reference fails the gate only"
    assert calls == ["fit", "cross", "cross"]


def test_verify_holds_a_stationarity_claim_to_the_test_on_the_maxima():
    trending = [{"name": "flood_frequency", "arguments": {}, "ok": True, "payload": {
        "station_id": "3400TH", "years": 40.0, "unit": "m3/s",
        "trend": {"on": "annual mean", "p_value": 0.9, "tau": 0.0},
        "ffa": {"return_periods": [100], "fits": {"gev_lmoments": {"q": [652.0]}},
                "amax_trend": {"on": "annual maxima", "p_value": 0.004, "tau": 0.3}}}}]
    text = ("No trend was found in the annual means (p = 0.9), so the record of annual maxima is treated as "
            "stationary; the 100-year flood at 3400TH is 652 m3/s (90 % interval 600 to 700 m3/s).")
    out = verify(text, trending, question="the 100-year flood")
    names = {c.name: c for c in out.checks}
    assert "stationarity_matches_the_maxima" in names and not names["stationarity_matches_the_maxima"].passed
    assert "finds a trend (p = 0.004)" in names["stationarity_matches_the_maxima"].detail
    calm = [dict(trending[0], payload={**trending[0]["payload"],
                                       "ffa": {**trending[0]["payload"]["ffa"],
                                               "amax_trend": {"on": "annual maxima", "p_value": 0.6, "tau": 0.02}}})]
    out2 = verify(text, calm, question="the 100-year flood")
    assert {c.name: c.passed for c in out2.checks}["stationarity_matches_the_maxima"]
    claim = ("The annual maxima show an increasing trend, so the 100-year flood at 3400TH of 652 m3/s "
             "(interval 600 to 700 m3/s) is a floor.")
    out3 = verify(claim, calm, question="the 100-year flood")
    assert not {c.name: c.passed for c in out3.checks}["stationarity_matches_the_maxima"]
    assert "stationarity_matches_the_maxima" not in {c.name for c in verify(text, [], question="q").checks}
