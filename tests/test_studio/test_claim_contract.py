"""Regressions for the estimator/uncertainty mismatch reproduced in the adoption audit."""

from __future__ import annotations

import io
import json

from aquascope.studio.deliverables.workbook import workbook_bytes
from aquascope.studio.roles import author, interpreter
from aquascope.trend_series import mark_reported_trend, reported_trend
from tests.test_studio.conftest import FLOW, fake_tools
from tests.test_studio.test_critic_author import _ran


def test_equal_rounded_estimates_do_not_make_different_estimators_share_an_interval():
    ws = _ran()
    numbers = author.key_numbers(ws.study_obj(), ws.run["results"])
    lm = next(k for k in numbers if k["label"] == "100-year return level, GEV (L-moments)")
    mle = next(k for k in numbers if k["label"] == "100-year return level, GEV (MLE with L-moments fallback)")
    lp3 = next(k for k in numbers if k["label"] == "100-year return level, Log-Pearson III")
    assert lm["value"] == mle["value"] == 520
    assert interpreter._interval_for(lm) is None
    assert interpreter._interval_for(mle) == [420, 650]
    assert interpreter._interval_for(lp3) == [410, 690]
    assert lm["evidence"]["result_id"] != mle["evidence"]["result_id"]


def test_a_model_cannot_attach_an_unrelated_interval_or_replace_the_claim():
    ws = _ran()
    rules = interpreter.rules_findings(ws)
    out = interpreter.validate_findings(ws, {
        "findings": rules["findings"],
        "decision": {"value": 520, "band": [420, 650], "answer": "Adopt 520 m3/s with band 420 to 650 m3/s."},
    }, rules=rules)
    assert out["decision"]["value"] == 520 and out["decision"]["band"] is None
    assert "band 420" not in out["decision"]["answer"]
    assert out["decision"]["evidence"]["estimator"] == "gev_lmoments"


def test_narrated_intervals_must_name_and_match_the_fitted_estimator():
    results = [{"payload": {"ffa": {"fits": {
        "gev_lmoments": {"q": [583.2]},
        "gev_bootstrap": {"q": [539.2], "ci": [[403.8, 767.1]], "ci_level": .9},
    }}}}]
    assert not author._interval_claim_supported("GEV is 583.2 with a 90% band of 403.8 to 767.1.", results)
    assert not author._interval_claim_supported("GEV MLE is 583.2 with a 90% band of 403.8 to 767.1.", results)
    assert author._interval_claim_supported("GEV MLE is 539.2 with a 90% band of 403.8 to 767.1.", results)
    assert not author._interval_claim_supported("GEV MLE is 539.2 with a 95% band of 403.8 to 767.1.", results)


def test_live_audit_numbers_keep_their_own_uncertainty_and_provenance_in_exports():
    flow = json.loads(json.dumps(FLOW))
    flow["data_snapshot"] = "sha256:fixture"
    flow["ffa"]["fits"]["gev_lmoments"]["q"][5] = 583.2
    flow["ffa"]["fits"]["gev_bootstrap"]["q"][5] = 539.2
    flow["ffa"]["fits"]["gev_bootstrap"]["ci"][5] = [403.8, 767.1]
    ws = _ran(tools=fake_tools([], flood_frequency=flow, analyze_station=flow))
    findings = interpreter.interpret(ws, None)
    report = author.author_report(ws, None)
    d = report["decision"]
    assert d["value"] == 583.2 and d["band"] is None
    assert d["evidence"]["dataset"]["snapshot"] == "sha256:fixture"
    assert d["evidence"]["dataset"]["start"] == flow["start"]
    assert d["evidence"]["aggregation"].startswith("annual maxima")
    assert report["findings"] == findings["findings"]
    import openpyxl

    book = openpyxl.load_workbook(io.BytesIO(workbook_bytes(ws)), read_only=True)
    rows = list(book["Findings"].values)
    assert rows[0][-1] == "evidence"
    claim = next(r for r in rows[1:] if "583.2" in str(r[2]))
    assert json.loads(claim[-1])["estimator"] == "gev_lmoments"


def test_missing_maxima_never_silently_substitutes_mean_trend():
    from aquascope.studio.deliverables.figures import _trend

    payload = {"trend": {"on": "annual mean", "p_value": 0.01}}
    assert reported_trend(payload, flood=True) is None
    assert not mark_reported_trend(payload, flood=True)
    assert reported_trend(payload)["unavailable"]
    assert "p_value" not in reported_trend(payload)
    assert _trend(payload, "m3/s", None) is None


def test_review_caveats_are_limitations_not_scientific_assumptions():
    ws = _ran()
    study = ws.study_obj()
    study.plan["caveats"] = ["Independent hydrologist review pending."]
    ws.set_study(study)
    decision = interpreter.rules_findings(ws)["decision"]
    assert "Independent hydrologist review pending." in decision["limitations"]
    assert "Independent hydrologist review pending." not in decision["conditions"]


def test_opposing_trends_follow_the_flood_quantity():
    payload = {"trend": {"on": "annual mean", "p_value": 0.8, "sens_slope_per_year": -2},
               "ffa": {"amax_trend": {"p_value": 0.001, "sens_slope_per_year": 10}}}
    assert mark_reported_trend(payload, flood=True)
    assert reported_trend(payload)["p_value"] == 0.001
    assert reported_trend(payload)["sens_slope_per_year"] == 10
