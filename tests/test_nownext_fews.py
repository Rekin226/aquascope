"""The FEWS view (#556): threshold classes, the ensemble plume from the members, and the forecast gauges.

Every network seam is replaced: GEOGLOWS (``rivers._fetch_json``, ``rivers.forecast_ensemble``,
``rivers.forecast_stats``, ``rivers.snap_to_river``) and the published Floods ahead issue.
"""

from __future__ import annotations

import numpy as np
import pytest

from aquascope import nownext, rivers
from aquascope.archive import warnings as fw

Q = {2: 100.0, 5: 150.0, 10: 200.0, 25: 260.0, 50: 300.0, 100: 350.0}
RID = 760716396


def _members(days: int = 3, n: int = 51, base: float = 90.0, step: float = 30.0) -> tuple[list[str], dict]:
    """3-hourly times over ``days`` UTC days, member k flowing base + step * day + k (high-res member 52 at 999)."""
    times = [f"2026-10-{10 + d:02d}T{h:02d}:00:00+00:00" for d in range(days) for h in range(0, 24, 3)]
    members = {}
    for k in range(1, n + 1):
        members[f"ensemble_{k:02d}"] = [base + step * (i // 8) + k for i in range(len(times))]
    members["ensemble_52"] = [999.0] * len(times)
    return times, members


# ── threshold classes: the Floods ahead rule ─────────────────────────────────


def test_threshold_map_reads_every_form_and_drops_the_unusable():
    assert nownext.threshold_map({2: 100, "5": 150.0, "q10": 200}) == {2: 100.0, 5: 150.0, 10: 200.0}
    assert nownext.threshold_map({"return_periods": [2, 5], "q": [100, None]}) == {2: 100.0}
    assert nownext.threshold_map([100, 150, float("nan"), -1]) == {2: 100.0, 5: 150.0}
    assert nownext.threshold_map({"q7": 3, "source": "x"}) == {}
    assert nownext.threshold_map(None) == {}


@pytest.mark.parametrize("value", [None, float("nan"), 0, 99.9, 100, 149, 150, 201, 260, 349, 350, 1e9])
def test_threshold_class_agrees_with_the_floods_ahead_layer(value):
    assert nownext.threshold_class(value, Q) == fw.classify(value, Q)
    assert nownext.THRESHOLD_YEARS == fw.RETURN_PERIODS and nownext.DAILY_CODES == fw.DAILY_CODES


def test_without_a_2_year_flow_nothing_is_classed():
    assert nownext.threshold_class(1e9, {5: 1.0}) == 0
    assert nownext.daily_code([0, 2, 5, 100, 0]) == fw.daily_string([0, 2, 5, 100, 0]) == "01260"


# ── the plume from the members ──────────────────────────────────────────────


def test_ensemble_daily_averages_each_member_per_utc_day_then_takes_quantiles():
    times, members = _members()
    members["ensemble_01"][3] = None                  # a gap is skipped, not counted as zero
    out = nownext.ensemble_daily(times, members, thresholds=Q)
    assert out["date"] == ["2026-10-10", "2026-10-11", "2026-10-12"] and out["n_members"] == 51
    assert out["initialized"] == "2026-10-10T00:00Z"
    day0 = np.array([90.0 + k for k in range(1, 52)])
    for key, want in (("median", np.median(day0)), ("p25", np.percentile(day0, 25)),
                      ("p75", np.percentile(day0, 75)), ("min", 91.0), ("max", 141.0), ("mean", day0.mean())):
        assert out[key][0] == pytest.approx(want)
    assert out["high_res"] == [999.0, 999.0, 999.0]        # member 52 rides apart, not in the statistics
    # members past each flow, per day: day 0 flows are 91..141, so 42 reach 100 and none 150
    assert out["members_at"]["2"][0] == 42 and out["members_at"]["5"] == [0, 22, 51]
    # each member's own 15-day peak (day 2: 151..201)
    assert out["share"]["2"] == 1.0 and out["share"]["10"] == pytest.approx(2 / 51, abs=1e-4)


def test_ensemble_daily_trims_a_last_day_no_member_reaches_and_copes_with_nothing():
    times, members = _members(days=2)
    out = nownext.ensemble_daily(times, members, days=15)
    assert len(out["date"]) == 2 and "members_at" not in out
    assert nownext.ensemble_daily([], {})["date"] == []


def test_forecast_ensemble_is_a_thin_fetch(monkeypatch):
    times, members = _members(days=1)
    raw = {"datetime": times, **{k: [("" if i == 1 else v) for i, v in enumerate(vals)] for k, vals in members.items()},
           "metadata": {"gen_date": "2026-10-10T12:06:45+00:00"}}
    seen = {}
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: seen.update(url=url, params=params) or raw)
    out = rivers.forecast_ensemble(RID)
    assert seen["url"].endswith(f"/forecastensemble/{RID}") and out["modelled"] is True
    assert "date" not in seen["params"]
    rivers.forecast_ensemble(RID, "2026-10-09")
    assert seen["params"]["date"] == "20261009"           # the run the map shows
    assert len(out["members"]) == 52 and out["members"]["ensemble_01"][1] is None
    assert out["generated"].startswith("2026-10-10")
    monkeypatch.setattr(rivers, "_fetch_json", lambda url, params=None: {})
    assert rivers.forecast_ensemble(RID)["error"]


def _ens(days: int = 3, **kw) -> dict:
    times, members = _members(days=days, **kw)
    return {"datetime": times, "members": members, "generated": "2026-10-10T12:00:00+00:00", "url": "u"}


def test_plume_classes_each_day_like_the_layer_and_counts_the_members(monkeypatch):
    monkeypatch.setattr(rivers, "forecast_ensemble", lambda rid, run=None: _ens())
    obs = {"t": ["2026-09-01", "2026-10-01", "2026-10-09", "2026-10-11", "2026-11-30"], "v": [1, 2, 3, None, 5]}
    res = nownext.plume(RID, thresholds={**{f"q{t}": q for t, q in Q.items()}, "source": "the layer",
                                         "licence": "CC BY-NC-SA 4.0"}, obs=obs)
    assert res["from"] == "members" and res["issued"] == "2026-10-10" and res["modelled"] is True
    # ensemble mean per day: 116, 146, 176 against 100 / 150 / 200
    assert res["class_daily"] == [2, 2, 5] and res["daily"] == "112" and res["rp"] == 5
    assert res["peak"] == pytest.approx(176.0) and res["peak_date"] == "2026-10-12"
    assert res["first_date"] == "2026-10-10"
    assert res["thresholds"]["q"][0] == 100.0 and res["thresholds"]["source"] == "the layer"
    assert res["thresholds"]["licence"] == "CC BY-NC-SA 4.0"
    assert res["sentence"] == "The ensemble mean peaks at 176 m³/s on 12 October, above the 5-year flow (150 m³/s)."
    assert res["members_line"] == ("In the 3 days, all 51 members reach the 2-year flow, all the 5-year flow and 2 "
                                   "the 10-year flow.")
    # the gauge's record from 21 days before the run to its last day; 1 September is too early, 30 November late
    assert res["observed"] == {"t": ["2026-10-01", "2026-10-09"], "v": [2.0, 3.0]}


def test_plume_reads_the_run_the_map_shows_and_falls_back_to_the_newest(monkeypatch):
    asked = []

    def fake(rid, run=None):
        asked.append(run)
        return {"error": "no such run"} if run else _ens()

    monkeypatch.setattr(rivers, "forecast_ensemble", fake)
    res = nownext.plume(RID, thresholds=Q, run="2026-10-01")
    assert asked == ["2026-10-01", None] and res["issued"] == "2026-10-10"
    assert any("did not answer, so this is the newest run" in n for n in res["notes"])


def test_plume_falls_back_to_the_statistics_and_says_so(monkeypatch):
    monkeypatch.setattr(rivers, "forecast_ensemble", lambda rid, run=None: {"error": "no members"})
    times = [f"2026-10-10T{h:02d}:00:00+00:00" for h in range(0, 24, 3)]
    flows = [120.0] * len(times)
    monkeypatch.setattr(rivers, "forecast_stats", lambda rid: {
        "datetime": times, "flow_avg": flows, "flow_med": flows, "flow_25p": flows, "flow_75p": flows,
        "flow_min": flows, "flow_max": flows, "high_res": flows})
    res = nownext.plume(RID, thresholds=Q)
    assert res["from"] == "statistics" and res["rp"] == 2 and "members_at" not in res and res["members_line"] == ""
    assert any("statistics" in n for n in res["notes"])
    monkeypatch.setattr(rivers, "forecast_stats", lambda rid: {"error": "down"})
    assert nownext.plume(RID, thresholds=Q)["error"] == "no members"


def test_plume_finds_thresholds_in_the_issue_then_the_simulated_record(monkeypatch):
    monkeypatch.setattr(rivers, "forecast_ensemble", lambda rid, run=None: _ens())
    monkeypatch.setattr(nownext, "_issue_thresholds", lambda rid: dict(Q))
    res = nownext.plume(RID)
    assert res["thresholds"]["source"].startswith("the Floods ahead issue") and res["rp"] == 5
    assert "NC-SA" in res["thresholds"]["licence"]

    monkeypatch.setattr(nownext, "_issue_thresholds", lambda rid: {})
    assert nownext.plume(RID)["thresholds"] is None and nownext.plume(RID)["rp"] == 0
    monkeypatch.setattr(nownext, "_reach_history", lambda rid: ({}, "series"))
    monkeypatch.setattr(nownext, "_thresholds", lambda s: {"return_periods": [2, 5], "q": [100, 150],
                                                            "method": "Log-Pearson III"})
    res = nownext.plume(RID, history=True)
    assert res["thresholds"]["q"] == [100.0, 150.0] and res["thresholds"]["source"].startswith("Log-Pearson III")


def test_plume_snaps_a_point_and_refuses_nothing(monkeypatch):
    monkeypatch.setattr(rivers, "forecast_ensemble", lambda rid, run=None: _ens())
    snapped = {"snapped": True, "river_id": 760000007}
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, prefer="main": snapped)
    assert nownext.plume(lat=1.0, lon=2.0, thresholds=Q)["river_id"] == 760000007
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, prefer="main": {"snapped": False, "message": "far"})
    assert nownext.plume(lat=1.0, lon=2.0)["error"] == "far"
    with pytest.raises(ValueError):
        nownext.plume()


# ── the forecast gauges ─────────────────────────────────────────────────────


def _issued(source="usgs", sid="X", corrected=True, q2=100.0, flows=(80.0, 120.0, 160.0)):
    rows = []
    for i, v in enumerate(flows):
        row = {"issue_date": "2026-10-09", "source": source, "station_id": sid, "river_id": 42, "model": "geoglows",
               "init_date": "2026-10-08", "valid_date": f"2026-10-{9 + i:02d}", "mean": v / 2, "median": v / 2,
               "p25": v / 2 - 1, "p75": v / 2 + 1, "min": 0.0, "max": v, "kge_raw": 0.2, "kge_corrected": 0.6,
               "reach_mean_ratio": 0.9, "gauge_q2": q2, "gauge_q5": 150.0, "gauge_q10": None}
        if corrected:
            row.update({"mean_c": v, "median_c": v, "p25_c": v - 5, "p75_c": v + 5, "min_c": v - 9, "max_c": v + 50})
        rows.append(row)
    return rows


def test_forecast_points_class_each_gauge_against_its_own_flows():
    rows = [*_issued(), *_issued(sid="Y", corrected=False), *_issued(sid="Z", q2=None),
            {**_issued()[0], "model": "glofas", "station_id": "G"}]
    res = nownext.forecast_points(list(reversed(rows)))
    assert res["n"] == 3 and res["issue_date"] == "2026-10-09"
    x, y, z = res["points"]
    assert x["key"] == "usgs/X" and x["corrected"] is True and x["median"] == [80.0, 120.0, 160.0]
    assert x["class_daily"] == [0, 2, 5] and x["rp"] == 5 and x["daily"] == "012" and x["issued"] == "2026-10-08"
    assert x["first_date"] == "2026-10-10" and x["peak"] == 160.0 and x["peak_date"] == "2026-10-11"
    assert x["thresholds"]["q"] == [100.0, 150.0] and x["kge_corrected"] == 0.6 and x["classed"] is True
    assert y["corrected"] is False and y["mean"] == [40.0, 60.0, 80.0] and y["rp"] == 0   # raw, below
    assert z["classed"] is False and z["thresholds"] is None and z["rp"] == 0
    assert res["counts"]["5"] == 1 and res["counts"]["0"] == 2
    assert nownext.forecast_points([])["n"] == 0
    assert "note" not in x


def test_forecast_points_say_when_the_correction_scored_worse():
    rows = [{**r, "kge_raw": 0.49, "kge_corrected": -0.05} for r in _issued()]
    point = nownext.forecast_points(rows)["points"][0]
    assert point["corrected"] is True and "scored worse" in point["note"]
    raw = [{**r, "kge_raw": 0.49, "kge_corrected": -0.05} for r in _issued(corrected=False)]
    assert "note" not in nownext.forecast_points(raw)["points"][0]   # the raw plume needs no warning
