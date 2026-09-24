"""The four trust bugs of the live keyless Study at USGS 01013500 ("How big is the 100-year flood here and is it
getting worse?"): the verdict said established above a "Not established" box, the GloFAS cross-check read a
tributary's cell, the trend quoted was the annual mean's, and the record was the archive's 40-year copy (#270).
Nothing here touches the network."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope import explore
from aquascope.gates import evaluate
from aquascope.schemas.water_data import DataSource, StreamflowReading
from aquascope.studio.roles import author, critic, interpreter
from aquascope.trend_series import is_flood_question, mark_reported_trend, reported_trend
from tests.test_studio.conftest import ANYWHERE, FLOW, fake_tools
from tests.test_studio.test_critic_author import _ran

# ── (a) one verdict ──────────────────────────────────────────────────────────


def _tributary_cell() -> dict:
    """GloFAS at the gauge's own coordinates: a tributary, 16.6 m3/s against a 520 m3/s at-site 100-year flow."""
    g = json.loads(json.dumps(ANYWHERE))
    g["glofas"]["ffa"]["fits"]["gev_lmoments"]["q_by_T"] = {"2": 5, "5": 8, "10": 10, "25": 13, "50": 15,
                                                             "100": 16.6}
    return g


def test_a_failed_cross_check_on_the_headline_is_not_established_anywhere():
    """Regression: the badge read ESTABLISHED while the same answer listed the failed gate and ended in a
    "Not established" box. The grade is computed once and a failed gate that bears on the headline caps it."""
    ws = _ran(tools=fake_tools([], anywhere=_tributary_cell()))
    failed = [g for g in ws.run["failed_gates"] if g["check"] == "cross_check_ratio"]
    assert failed and failed[0]["step"] == "s4", "the cross-check disagreed"
    out = interpreter.interpret(ws, None)
    grade, primary = interpreter.grade_for_study(ws)
    assert primary == "s3" and grade == "indicative"
    assert out["decision"]["grade"] == "indicative" and "(indicative)" in out["decision"]["answer"]
    assert [g["step"] for g in interpreter.headline_gates(ws, primary)] == ["s4"]
    assert any("cross_check_ratio" in c for c in out["decision"]["conditions"])
    missing = critic.not_established(ws)
    assert any("cross_check_ratio" in m for m in missing), "the box lists it"
    report = author.author_report(ws, None)
    assert report["grade"] == "indicative", "the badge and the box read the same verdict"


def test_every_gate_passed_is_still_established():
    ws = _ran()
    grade, primary = interpreter.grade_for_study(ws)
    assert primary == "s3" and grade == "established"
    assert interpreter.headline_gates(ws, primary) == []


def test_a_skipped_cross_check_does_not_lower_the_grade():
    g = json.loads(json.dumps(ANYWHERE))
    g["glofas"] = {"comparable": False, "note": "no comparable model cell: nothing within a factor 2.",
                   "cell": {"comparable": False}}
    ws = _ran(tools=fake_tools([], anywhere=g))
    s4 = next(r for r in ws.run["results"] if r["id"] == "s4")
    cc = next(x for x in s4["gates"] if x["check"] == "cross_check_ratio")
    assert cc["passed"] and cc["skipped"] and "no comparable model cell" in cc["detail"]
    assert not ws.run["failed_gates"]
    out = interpreter.interpret(ws, None)
    assert out["decision"]["grade"] == "established"
    agree = next(c for c in out["consistency"] if "cross_check_ratio" in c["a"])
    assert agree["agree"] is None


# ── (b) the GloFAS cell snapped to the gauge's river ─────────────────────────


class _FakeFlood:
    """The Open-Meteo flood API for a list of coordinates: one answer per point, the main stem east of the site."""

    def __init__(self, main_mean: float = 50.0, fail: bool = False):
        self.calls: list[dict] = []
        self.main_mean = main_mean
        self.fail = fail

    def fetch_raw(self, *, latitude, longitude, start_date=None, end_date=None, daily=None):
        self.calls.append({"latitude": latitude, "longitude": longitude, "start_date": start_date})
        if self.fail:
            raise RuntimeError("HTTP 429")
        lats = [float(x) for x in str(latitude).split(",")]
        lons = [float(x) for x in str(longitude).split(",")]
        if len(lats) == 1:  # the full-record fetch at the chosen cell
            days = pd.date_range("2000-01-01", "2025-12-31", freq="D")
            rng = np.random.default_rng(3)
            q = (self.main_mean * np.exp(rng.normal(0, 0.6, len(days)))).round(3)
            return {"latitude": lats[0], "longitude": lons[0],
                    "daily": {"time": [d.strftime("%Y-%m-%d") for d in days], "river_discharge": q.tolist()}}
        out = []
        for la, lo in zip(lats, lons):
            east = lo - lons[len(lons) // 2]
            mean = self.main_mean if abs(east - 0.1) < 1e-6 and abs(la - lats[len(lats) // 2]) < 1e-6 else 2.0
            out.append({"latitude": la, "longitude": lo, "daily": {"time": ["2024-01-01", "2024-01-02"],
                                                                     "river_discharge": [mean, mean]}})
        return out


def test_the_cell_is_snapped_to_the_one_whose_mean_flow_matches_the_gauge():
    fake = _FakeFlood(main_mean=48.0)
    with patch.object(explore, "build_collector", return_value=fake):
        cell = explore.snap_glofas_cell(46.7, -68.6, 50.0)
    assert len(str(fake.calls[0]["latitude"]).split(",")) == 25, "one request for a 5 x 5 window"
    assert cell["comparable"] and cell["lat"] == 46.7 and cell["lon"] == pytest.approx(-68.5)
    assert cell["ratio"] == pytest.approx(0.96) and cell["offset_km"] > 5
    assert "closest to the gauge's" in cell["why"] and cell["n_probed"] == 25


def test_no_cell_within_tolerance_says_so():
    with patch.object(explore, "build_collector", return_value=_FakeFlood(main_mean=400.0)):
        cell = explore.snap_glofas_cell(46.7, -68.6, 50.0)
    assert cell["comparable"] is False and cell["why"].startswith("no comparable model cell")


def test_anywhere_reads_the_snapped_cell_and_records_it():
    fake = _FakeFlood(main_mean=48.0)
    with patch.object(explore, "build_collector", return_value=fake):
        out = explore.anywhere(46.7, -68.6, years=20, match_mean_flow=50.0, area_km2=2320)
    g = out["glofas"]
    assert g["comparable"] is True and g["cell"]["lon"] == pytest.approx(-68.5) and g["cell"]["area_km2"] == 2320
    assert fake.calls[-1]["latitude"] == 46.7 and fake.calls[-1]["longitude"] == pytest.approx(-68.5)
    assert "ffa" in g and any("Cell snapped to the gauge" in n for n in out["notes"])
    json.dumps(out)


def test_anywhere_with_no_comparable_cell_skips_the_cross_check_gate():
    fake = _FakeFlood(main_mean=400.0)
    with patch.object(explore, "build_collector", return_value=fake):
        out = explore.anywhere(46.7, -68.6, years=20, match_mean_flow=50.0)
    g = out["glofas"]
    assert g["comparable"] is False and "ffa" not in g and g["note"].startswith("no comparable model cell")
    gates = evaluate([{"check": "not_empty", "path": "glofas"},
                      {"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                       "reference": {"100": 583.0}, "return_period": 100, "value": 0.5}], out)
    assert all(x["passed"] for x in gates)
    assert gates[1]["skipped"] and "no comparable model cell" in gates[1]["detail"]


def test_a_failed_probe_is_no_comparable_cell_not_a_tributary():
    with patch.object(explore, "build_collector", return_value=_FakeFlood(fail=True)):
        out = explore.anywhere(46.7, -68.6, years=20, match_mean_flow=50.0)
    assert out["glofas"]["comparable"] is False and "probe failed" in out["glofas"]["note"]


def test_the_point_card_without_a_gauge_is_unchanged():
    fake = _FakeFlood(main_mean=48.0)
    with patch.object(explore, "build_collector", return_value=fake):
        out = explore.anywhere(46.7, -68.6, years=20)
    assert all("," not in str(c["latitude"]) for c in fake.calls) and "cell" not in out["glofas"]


def test_the_flood_playbook_and_a_models_cross_check_ask_for_the_snap():
    from aquascope import playbooks as pbk
    from aquascope.studio.catalogue import repair_cross_checks
    from tests.test_playbooks import LONG

    study = pbk.plan("flood_risk", LONG, {"return_period": 100})
    s4 = study.step_by_id("s4")
    assert s4.tool == "anywhere" and s4.arguments["match_mean_flow"] == "{{ result.s3.stats.mean }}"
    steps = [{"id": "s3", "tool": "flood_frequency", "method": "at_site_flood_frequency", "arguments": {}},
             {"id": "s4", "tool": "anywhere", "arguments": {"lat": 1, "lon": 2},
              "expects": [{"check": "cross_check_ratio", "path": "glofas.ffa.fits.gev_lmoments.q_by_T",
                           "reference": "{{ result.<the flood_frequency step>.ffa.fits.gev_lmoments.q_by_T }}"}]}]
    repair_cross_checks(steps)
    assert steps[1]["arguments"]["match_mean_flow"] == "{{ result.s3.stats.mean }}"


# ── (c) the flood trend is the annual maxima's ───────────────────────────────


def test_which_questions_are_about_the_floods():
    assert is_flood_question("flood_risk", "is it getting worse?")
    assert is_flood_question(None, "How big is the 100-year flood here and is it getting worse?")
    assert not is_flood_question("supply_reliability", "will the flood of demand be met?")
    assert not is_flood_question(None, "Is the mean flow declining?")


def test_reported_trend_picks_the_series():
    p = json.loads(json.dumps(FLOW))
    assert reported_trend(p)["on"] == "annual mean"
    assert reported_trend(p, flood=True)["on"] == "annual maxima"
    assert mark_reported_trend(p, flood=True) and reported_trend(p)["p_value"] == 0.37
    q = json.loads(json.dumps(FLOW))
    assert not mark_reported_trend(q, flood=False) and "trend_reported" not in q


def test_a_flood_study_quotes_the_annual_maxima_trend():
    ws = _ran(problem="How big is the 100-year flood here and is it getting worse?")
    s2 = next(r for r in ws.run["results"] if r["id"] == "s2")
    assert s2["result"]["trend_reported"]["on"] == "annual maxima"
    report = author.author_report(ws, None)
    labels = {k["label"]: k for k in report["key_numbers"]}
    assert "Mann-Kendall p-value (annual maxima)" in labels
    assert "Mann-Kendall p-value (annual mean)" not in labels
    text = json.dumps(report)
    assert "Mann-Kendall on the annual maxima" in text and "Mann-Kendall on the annual mean" not in text


def test_a_supply_study_keeps_the_annual_mean():
    ws = _ran(problem="Can the river supply 2 m3/s reliably?", playbook="supply_reliability",
              intake={"demand_m3s": 2.0})
    for r in ws.run["results"]:
        assert "trend_reported" not in (r.get("result") or {})


def test_the_trend_figure_draws_the_maxima_for_a_flood_question():
    pytest.importorskip("matplotlib")
    from aquascope.studio.deliverables import figures

    p = json.loads(json.dumps(FLOW))
    p["annual_max"] = {"year": list(range(1990, 2020)), "v": [100.0 + 2 * i for i in range(30)]}
    p["series"] = {"t": ["2000-01-01", "2001-01-01", "2002-01-01", "2003-01-01"], "v": [1, 2, 3, 4]}
    mark_reported_trend(p, flood=True)
    fig, caption = figures.make("trend", p)
    assert caption.startswith("Annual maximum") and "annual maxima" in caption
    figures.close(fig)
    q = json.loads(json.dumps(FLOW))
    q["series"] = p["series"]
    drawn = figures.make("trend", q)
    assert drawn is not None and drawn[1].startswith("Annual mean")
    figures.close(drawn[0])


# ── (d) the full record, not the archive's 40-year copy (#270) ───────────────


def _today() -> date:
    return datetime.now(timezone.utc).date()


class _USGS:
    def __init__(self, start: str | None, fail: bool = False):
        self.start, self.fail, self.calls = start, fail, []

    def collect(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("NWIS 503")
        if kw.get("parameter") != "00060" or self.start is None:
            return []
        idx = pd.date_range(self.start, _today(), freq="D")
        return [StreamflowReading(source=DataSource.USGS, station_id="USGS-01013500",
                                  reading_datetime=t.to_pydatetime(), discharge_cms=50.0 + (i % 90),
                                  source_type="in_situ") for i, t in enumerate(idx)]


def _archive_copy(years: int = 40) -> pd.Series:
    idx = pd.date_range(end=_today(), periods=int(365.25 * years), freq="D")
    return pd.Series(np.linspace(40, 60, len(idx)), index=idx)


def test_a_short_archive_copy_is_replaced_by_the_agencys_full_record():
    agency = _USGS("1903-10-01")
    with patch("aquascope.archive.observations.fetch_archived_series", return_value=_archive_copy()), \
            patch.object(explore, "build_collector", return_value=agency):
        out = explore.fetch_series("usgs", "USGS-01013500", period_start="1903-07-29")
    assert out["series"].index.min().date() == date(1903, 10, 1)
    assert agency.calls[0]["days"] == (_today() - date(1903, 7, 29)).days
    assert "USGS daily values (NWIS); full record requested (from 1903-07-29" in out["note"]
    assert "archive holds only" in out["note"] and "full record came from the agency" in out["note"]


def test_the_archive_copy_stands_when_the_agency_has_nothing_earlier_or_fails():
    hit = _archive_copy()
    for agency in (_USGS(None), _USGS(None, fail=True)):
        with patch("aquascope.archive.observations.fetch_archived_series", return_value=hit), \
                patch.object(explore, "build_collector", return_value=agency):
            out = explore.fetch_series("usgs", "USGS-01013500", period_start="1903-07-29")
        assert len(out["series"]) == len(hit) and "From the AquaScope archive" in out["note"]
        assert "The catalog lists this station from 1903-07-29" in out["note"]


def test_a_capped_request_or_a_full_archive_copy_never_calls_the_agency():
    class Never:
        def __getattr__(self, name):
            raise AssertionError("the agency must not be called")

    with patch("aquascope.archive.observations.fetch_archived_series", return_value=_archive_copy()), \
            patch.object(explore, "build_collector", return_value=Never()):
        capped = explore.fetch_series("usgs", "USGS-01013500", years=20, period_start="1903-07-29")
        start = (_today() - timedelta(days=int(40 * 365.25) - 30)).isoformat()
        whole = explore.fetch_series("usgs", "USGS-01013500", period_start=start)
    assert "last 20 years requested" in capped["note"] and "From the AquaScope archive" in whole["note"]


def test_the_browser_does_not_call_an_agency_it_cannot_reach(monkeypatch):
    class Never:
        def __getattr__(self, name):
            raise AssertionError("the agency must not be called from the browser")

    monkeypatch.setattr(explore, "IS_EMSCRIPTEN", True)
    with patch("aquascope.archive.observations.fetch_archived_series", return_value=_archive_copy()), \
            patch.object(explore, "build_collector", return_value=Never()):
        out = explore.fetch_series("greece_openhi", "8425", period_start="1960-01-01")
    assert "From the AquaScope archive" in out["note"]


def test_trend_check_reads_the_series_the_report_quotes():
    """A flood report quotes the annual-maxima test; the check must not hold it to the annual-mean p."""
    from aquascope.ai_engine.verify import verify

    record = {"trend": {"p_value": 0.01, "on": "annual mean"}}  # the record step: annual means only
    flood = {"trend": {"p_value": 0.01, "on": "annual mean"},
             "trend_reported": {"p_value": 0.32, "on": "annual maxima"}}
    answer = "Mann-Kendall on the annual maxima: not significant at the 5 % level (p = 0.32), no trend in the floods."
    v = verify(answer, [{"ok": True, "name": "analyze_station", "payload": record},
                        {"ok": True, "name": "flood_frequency", "payload": flood}])
    check = next(c for c in v.checks if c.name == "trend_matches_the_test")
    assert check.passed
    # and the check still bites: calling the peaks' trend significant at p = 0.32 is caught
    wrong = "Mann-Kendall on the annual maxima: a significant rising trend in the flood peaks (p = 0.32)."
    v = verify(wrong, [{"ok": True, "name": "analyze_station", "payload": record},
                       {"ok": True, "name": "flood_frequency", "payload": flood}])
    assert not next(c for c in v.checks if c.name == "trend_matches_the_test").passed
