"""Now and next (#517): today against normal, the 15-day forecast, the correction to a gauge and its skill.

Every network seam is replaced: GEOGLOWS (``rivers.forecast_stats`` and ``nownext._reach_history``), Open-Meteo
(``nownext._fetch_json``), the GloFAS cell match and the agency fetch.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aquascope import explore, nownext, rivers


def _seasonal(start: str = "1990-01-01", end: str = "2026-10-08", seed: int = 1, scale: float = 1.0) -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    doy = idx.dayofyear.to_numpy()
    base = 50 + 30 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    flood = rng.gamma(1.2, 8.0, len(idx))
    return pd.Series(scale * (base + flood), index=idx)


# ── flow_status ──────────────────────────────────────────────────────────────


def test_flow_status_ranks_the_latest_day_against_the_same_days_in_other_years():
    s = _seasonal()
    s.iloc[-1] = s.iloc[-1] + 500  # a flood today
    st = nownext.flow_status(s, today="2026-10-08")
    assert st["date"] == "2026-10-08" and st["class"] == "much_above" and st["percentile"] > 90
    assert st["n_years"] == 36  # 1990 to 2025: the target year is left out of its own reference
    assert st["sentence"].startswith("Flow is much above normal for 8 October (")
    assert st["sentence"].endswith("percentile of 36 years).")
    assert set(st["normal"]) == {"p10", "p25", "p50", "p75", "p90"}
    assert st["recent"]["t"][-1] == "2026-10-08" and len(st["recent"]["t"]) == 31


@pytest.mark.parametrize(("value", "cls"), [(-1e9, "much_below"), (1e9, "much_above")])
def test_flow_status_with_a_value_from_elsewhere(value, cls):
    st = nownext.flow_status(_seasonal(), "2026-07-01", value=value, today="2026-07-01")
    assert st["class"] == cls and st["value"] == pytest.approx(value)


def test_the_five_classes_follow_the_rounded_percentile():
    assert [nownext._class_of(p)["id"] for p in (0, 9.4, 9.6, 24.4, 24.6, 75.4, 75.6, 90.4, 90.6, 100)] == [
        "much_below", "much_below", "below", "below", "normal", "normal", "above", "above", "much_above",
        "much_above"]


def test_flow_status_says_so_when_the_record_is_too_short():
    st = nownext.flow_status(_seasonal("2019-01-01"), today="2026-10-08")
    assert st["class"] is None and st["n_years"] == 7
    assert "Only 7 years have values within 7 days of 8 October" in st["error"]
    assert st["sentence"] == st["error"]


def test_flow_status_on_a_stale_record_speaks_in_the_past():
    st = nownext.flow_status(_seasonal(end="2024-03-03"), today="2026-10-08")
    assert st["age_days"] > 2
    assert st["sentence"].startswith("Flow was ") and "on 3 March 2024 (" in st["sentence"]
    assert st["sentence"].endswith(", the latest day in the record.")


def test_flow_status_on_29_february_uses_the_28th_in_common_years():
    st = nownext.flow_status(_seasonal(end="2024-02-29"), today="2024-02-29")
    assert st["n_years"] >= 30 and st["class"] is not None


def test_flow_status_refuses_rainfall_and_an_empty_record():
    assert "rainfall" in nownext.flow_status(_seasonal(), variable="precipitation")["error"]
    assert nownext.flow_status(pd.Series(dtype=float))["error"]
    assert "no value on" in nownext.flow_status(_seasonal(end="2026-01-01"), "2026-05-01")["error"]


def test_flow_status_reads_a_t_v_dict_and_names_the_level():
    s = _seasonal()
    d = {"t": [x.strftime("%Y-%m-%d") for x in s.index], "v": list(s.values)}
    st = nownext.flow_status(d, variable="water_level", unit="m", today="2026-10-08")
    assert st["sentence"].startswith("Water level is ")


def test_ordinals():
    assert [nownext._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 82, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "82nd", "100th"]


# ── top_up and station_status ───────────────────────────────────────────────


def test_top_up_adds_only_the_agencys_newer_days(monkeypatch):
    copy = _seasonal(end="2026-10-01")
    seen = {}

    def fake(source, sid, years=None, prefer_archive=True, variable=None, **kw):
        seen.update(source=source, years=years, prefer_archive=prefer_archive, variable=variable)
        return {"series": _seasonal("2025-10-08", "2026-10-07")}

    monkeypatch.setattr(explore, "fetch_series", fake)
    s, note = nownext.top_up(copy, "usgs", "USGS-1", today="2026-10-08")
    assert seen == {"source": "usgs", "years": 1, "prefer_archive": False, "variable": "discharge"}
    assert s.index.max() == pd.Timestamp("2026-10-07") and len(s) == len(copy) + 6
    assert s.loc["2026-09-30"] == pytest.approx(copy.loc["2026-09-30"])  # the copy's own days are kept
    assert note == "Added 6 recent days from the agency, to 2026-10-07."


def test_top_up_leaves_a_fresh_record_alone_and_never_raises(monkeypatch):
    monkeypatch.setattr(explore, "fetch_series", lambda *a, **k: pytest.fail("no fetch for a fresh record"))
    assert nownext.top_up(_seasonal(end="2026-10-07"), "usgs", "x", today="2026-10-08")[1] == ""
    assert "cannot be asked" in nownext.top_up(_seasonal(end="2026-10-01"), "grdc", "x", today="2026-10-08")[1]

    def boom(*a, **k):
        raise RuntimeError("agency down")

    monkeypatch.setattr(explore, "fetch_series", boom)
    s, note = nownext.top_up(_seasonal(end="2026-10-01"), "usgs", "x", today="2026-10-08")
    assert s.index.max() == pd.Timestamp("2026-10-01") and "could not be asked" in note


def test_station_status_fetches_tops_up_and_ranks(monkeypatch):
    calls = []

    def fake(source, sid, years=None, prefer_archive=True, variable=None, **kw):
        calls.append(prefer_archive)
        if prefer_archive:
            return {"series": _seasonal(end="2026-10-05"), "variable": "discharge", "unit": "m3/s", "note": "archive"}
        return {"series": _seasonal("2026-01-01", "2026-10-08")}

    monkeypatch.setattr(explore, "fetch_series", fake)
    st = nownext.station_status("usgs", "USGS-1", today="2026-10-08")
    assert calls == [True, False] and st["date"] == "2026-10-08" and st["unit"] == "m3/s"
    assert st["source"] == "usgs" and st["top_up"].startswith("Added 3 recent days")


# ── the forecast ────────────────────────────────────────────────────────────

RID = 230260670


def _stats() -> dict:
    times = [f"2026-10-07T{h:02d}:00:00+00:00" for h in range(0, 24, 3)] + \
            [f"2026-10-08T{h:02d}:00:00+00:00" for h in range(0, 24, 3)]
    flows = [100.0] * 8 + [200.0] * 8
    return {"river_id": RID, "datetime": times, "flow_avg": flows, "flow_med": flows,
            "flow_25p": [v - 10 for v in flows], "flow_75p": [v + 10 for v in flows],
            "flow_min": [v - 50 for v in flows], "flow_max": [v + 900 for v in flows],
            "high_res": [None] + flows[1:], "generated": "2026-10-08T08:00:00+00:00", "url": "u", "source": "s"}


def _glofas_raw(lat=46.95, lon=7.45):
    return {"latitude": lat, "longitude": lon, "daily": {
        "time": ["2026-10-08", "2026-10-09"], "river_discharge_mean": [150.0, 160.0],
        "river_discharge_median": [150.0, 155.0], "river_discharge_p25": [140, 150], "river_discharge_p75": [160, 170],
        "river_discharge_min": [120, 130], "river_discharge_max": [190, 210]}}


@pytest.fixture
def models(monkeypatch):
    seen = {"glofas": []}
    monkeypatch.setattr(rivers, "forecast_stats", lambda rid: _stats())
    hist = _seasonal("1940-01-01", "2026-10-01", seed=3, scale=2.0)
    monkeypatch.setattr(nownext, "_reach_history", lambda rid: ({"river_id": rid}, hist))

    def fake_json(url, params=None):
        seen["glofas"].append(params)
        return _glofas_raw(params["latitude"], params["longitude"])

    monkeypatch.setattr(nownext, "_fetch_json", fake_json)
    monkeypatch.setattr(explore, "snap_glofas_cell", lambda lat, lon, ref, **kw: {
        "lat": 46.9, "lon": 7.5, "offset_km": 5.2, "flow_magnitude_matches": True})
    seen["hist"] = hist
    return seen


def test_forecast_reads_both_models_daily_with_the_reachs_thresholds(models, monkeypatch):
    # The status is for the first forecast day (7 October 2026); pin "today" to the fixture's run date so the
    # sentence reads "is" rather than "was" whatever the calendar says.
    monkeypatch.setattr(nownext, "_today", lambda: date(2026, 10, 8))
    fc = nownext.forecast(46.95, 7.45, river_id=RID)
    g = fc["geoglows"]
    assert g["date"] == ["2026-10-07", "2026-10-08"] and g["mean"] == [100.0, 200.0]
    assert g["max"] == [1000.0, 1100.0] and g["high_res"] == [100.0, 200.0]  # a missing hour is left out
    assert g["licence"] == "CC BY 4.0" and g["generated"].startswith("2026-10-08")
    gl = fc["glofas"]
    assert gl["date"] == ["2026-10-08", "2026-10-09"] and gl["mean"] == [150.0, 160.0]
    assert "picked by mean flow, 5.2 km away" in gl["cell_note"]
    assert models["glofas"][0]["latitude"] == 46.9 and models["glofas"][0]["forecast_days"] == 15
    thr = fc["thresholds"]
    assert thr["return_periods"] == [2, 5, 10, 25, 50, 100] and len(thr["q"]) == 6
    assert thr["q"] == sorted(thr["q"]) and thr["method"].startswith("Log-Pearson III") and thr["n_years"] >= 80
    assert fc["status"]["sentence"].startswith("Simulated flow is ")
    assert fc["sentence"].startswith("The GEOGLOWS ensemble mean peaks at 200 m³/s on 8 October, under the 2-year")
    assert fc["modelled"] is True and "correction" not in fc


def test_forecast_says_when_a_peak_passes_a_threshold():
    thr = {"return_periods": [2, 5, 10], "q": [100.0, 150.0, 300.0]}
    s = nownext._peak_sentence({"date": ["2026-10-09"], "mean": [180.0]}, thr, what="It")
    assert s == "It peaks at 180 m³/s on 9 October, above the 5-year flow (150 m³/s)."


def test_forecast_with_a_gauge_corrects_and_scores(models):
    obs = models["hist"] / 2.0 + 1.0
    fc = nownext.forecast(river_id=RID, obs=obs, glofas_at=(1.0, 2.0))
    assert models["glofas"][0]["latitude"] == 1.0
    corr = fc["correction"]
    assert corr["by"] == "month" and corr["forecast"]["date"] == ["2026-10-07", "2026-10-08"]
    assert corr["forecast"]["mean"][1] < 200.0  # mapped onto the gauge, which runs at half the model
    assert fc["gauge_thresholds"]["source"] == "the gauge's observed record"
    assert fc["sentence"].startswith("Corrected to the gauge, the GEOGLOWS ensemble mean peaks at")
    assert any(m["name"].startswith("Flow-duration quantile mapping") for m in fc["methods"])


def test_forecast_where_there_is_no_river_is_glofas_only(models, monkeypatch):
    monkeypatch.setattr(rivers, "forecast_stats", lambda rid: pytest.fail("no GEOGLOWS without a reach"))
    fc = nownext.forecast(46.6, 7.9, snap=False)
    assert fc["river_id"] is None and "geoglows" not in fc and fc["glofas"]["mean"] == [150.0, 160.0]
    assert any("only GloFAS" in n for n in fc["notes"])
    assert fc["sentence"].startswith("The GloFAS ensemble mean peaks at 160")


def test_forecast_snaps_a_point_and_survives_a_failed_model(models, monkeypatch):
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0, **kw: {
        "snapped": True, "river_id": RID, "snap_lat": 46.95, "snap_lon": 7.45, "message": "Snapped."})

    def boom(rid):
        raise RuntimeError("500")

    monkeypatch.setattr(rivers, "forecast_stats", boom)
    fc = nownext.forecast(46.9, 7.4)
    assert fc["river_id"] == RID and "did not answer" in fc["geoglows"]["error"] and fc["glofas"]["mean"]
    with pytest.raises(ValueError):
        nownext.forecast()


def test_a_quick_forecast_skips_the_simulated_record(models, monkeypatch):
    monkeypatch.setattr(nownext, "_reach_history", lambda rid: pytest.fail("the quick forecast reads no history"))
    fc = nownext.forecast(46.95, 7.45, river_id=RID, history=False, glofas=False)
    assert fc["history"] is False and fc["geoglows"]["mean"] == [100.0, 200.0] and "glofas" not in fc
    assert "thresholds" not in fc and "status" not in fc
    assert fc["sentence"] == "The GEOGLOWS ensemble mean peaks at 200 m³/s on 8 October."


def test_a_full_forecast_reuses_the_geoglows_part_it_is_given(models, monkeypatch):
    quick = nownext.forecast(46.95, 7.45, river_id=RID, history=False, glofas=False)
    monkeypatch.setattr(rivers, "forecast_stats", lambda rid: pytest.fail("GEOGLOWS was read again"))
    fc = nownext.forecast(46.95, 7.45, river_id=RID, known_geoglows=quick["geoglows"])
    assert fc["history"] is True and fc["geoglows"]["mean"] == [100.0, 200.0] and fc["thresholds"]["q"]


def test_daily_geoglows_needs_no_pandas_and_matches_the_hourly_means():
    stats = {"datetime": ["2026-10-07T21:00:00Z", "2026-10-07T22:00:00+00:00", "2026-10-08T02:00:00+03:00",
                          "not a time"],
             "flow_avg": [1.0, None, 5.0, 9.0], "flow_med": [None, None, 2.0, 3.0], "flow_max": [1.0]}
    d = nownext._daily_geoglows(stats, 15)
    # 02:00 at +03:00 is 23:00 UTC on the 7th, so every readable hour falls on one day
    assert d["date"] == ["2026-10-07"] and d["mean"] == [3.0] and d["median"] == [2.0] and "max" not in d
    assert d["initialized"] == "2026-10-07T21:00Z"
    assert nownext._daily_geoglows({"datetime": []}, 15) == {"date": []}


def test_daily_geoglows_caps_the_days():
    d = nownext._daily_geoglows(_stats(), 1)
    assert d["date"] == ["2026-10-07"] and d["p25"] == [90.0]
    assert d["initialized"] == "2026-10-07T00:00Z"  # the run's start, not when the API answered


def test_forecast_says_when_the_reach_is_far_from_the_gauges_river(models):
    fc = nownext.forecast(river_id=RID, obs=models["hist"] * 10.0, glofas_at=(1.0, 2.0))
    chk = fc["reach_check"]
    assert chk["matches"] is False and chk["ratio"] == pytest.approx(0.1, rel=1e-3)
    assert "may be on another river than this reach" in chk["note"]
    near = nownext.forecast(river_id=RID, obs=models["hist"] * 1.5, glofas_at=(1.0, 2.0))["reach_check"]
    assert near["matches"] is True and near["note"] is None


# ── corrected to the gauge ──────────────────────────────────────────────────


def test_correction_recovers_a_biased_model_and_scores_it_on_held_out_years():
    model = _seasonal("2000-01-01", "2025-12-31", seed=5)
    obs = model * 0.5
    fcst = {"date": ["2026-01-05", "2026-07-05"], "mean": [float(model.loc["2025-01-05"]),
                                                          float(model.loc["2025-07-05"])], "p25": [None, 10.0]}
    res = nownext.correct_to_gauge(model, obs, fcst)
    sk = res["skill"]
    assert res["by"] == "month" and sk["fit_period"]["end"] < sk["score_period"]["start"]
    assert sk["raw"]["beta"] == pytest.approx(2.0, abs=0.01) and sk["raw"]["pbias"] == pytest.approx(100.0, abs=1)
    assert sk["corrected"]["kge"] > 0.9 > sk["raw"]["kge"]
    assert res["forecast"]["mean"][0] == pytest.approx(fcst["mean"][0] * 0.5, rel=0.05)
    assert res["forecast"]["p25"][0] is None
    assert res["skill_line"] == (f"Corrected forecast: KGE {sk['corrected']['kge']:.2f} on the "
                                 f"{sk['score_period']['start'][:4]}-2025 hindcast, raw {sk['raw']['kge']:.2f}.")


def test_skill_counts_hits_and_false_alarms_above_the_two_year_flow():
    model = _seasonal("2000-01-01", "2025-12-31", seed=5)
    sk = nownext.hindcast_skill(model, model * 0.5, threshold=60.0)
    raw, cor = sk["raw"], sk["corrected"]
    assert raw["days_above"] == cor["days_above"] > 0
    assert raw["hit_rate"] == 1.0 and raw["false_alarms"] > 0  # the raw model, twice as high, always says "above"
    assert cor["hit_rate"] > 0.8 and cor["false_alarm_ratio"] < raw["false_alarm_ratio"]
    assert 0 <= raw["false_alarm_rate"] <= 1
    assert sk["skill_detail"].startswith(f"Bias {cor['pbias']:+.0f} % (raw +100 %). Days above the 2-year flow "
                                         f"caught: {cor['hits']} of {cor['days_above']} (raw {raw['hits']})")


def test_correction_needs_enough_shared_years():
    model = _seasonal("2024-01-01", "2025-12-31")
    res = nownext.correct_to_gauge(model, model, None)
    assert "share 2.0 years" in res["error"] and "needs 3" in res["error"]
    assert "no days" in nownext.correct_to_gauge(model, pd.Series(dtype=float))["error"]


def test_correction_falls_back_to_one_curve_when_a_month_is_thin():
    model = _seasonal("2010-01-01", "2025-12-31")
    obs = (model * 0.7)[model.index.month != 7]
    res = nownext.correct_to_gauge(model, obs, model.iloc[-3:])
    assert res["by"] == "year" and "one flow-duration curve" in res["note"]
    assert len(res["forecast"]["mean"]) == 3


def test_beyond_the_curve_the_end_ratio_is_kept():
    mapping = {"by": "year", "curves": {0: (np.array([1.0, 2.0, 4.0]), np.array([2.0, 4.0, 8.0]))}}
    out = nownext._map_values(mapping, [8.0, 0.5, math.nan], [1, 1, 1])
    assert out[0] == pytest.approx(16.0) and out[1] == pytest.approx(1.0) and math.isnan(out[2])


# ── now: one call ───────────────────────────────────────────────────────────


def test_now_for_a_station_gives_status_and_the_corrected_forecast(models, monkeypatch):
    from aquascope.archive import catalog

    gauge = models["hist"]["1980":] / 2.0
    monkeypatch.setattr(explore, "fetch_series", lambda source, sid, **kw: {
        "series": gauge, "variable": "discharge", "unit": "m3/s", "note": "archive"})
    prefs = []

    def snap(lat, lon, max_distance_m=1000.0, prefer="main"):
        prefs.append(prefer)
        return {"snapped": True, "river_id": RID, "snap_lat": lat, "snap_lon": lon, "message": "Snapped 20 m."}

    monkeypatch.setattr(rivers, "snap_to_river", snap)
    catalog.set_catalog([{"source": "usgs", "station_id": "USGS-1", "latitude": 40.0, "longitude": -75.0,
                          "name": "A RIVER"}])
    try:
        res = nownext.now(station="usgs/USGS-1", refresh=False)
    finally:
        catalog.set_catalog(None)
    assert res["station"]["name"] == "A RIVER" and res["status"]["class"] in {c["id"] for c in nownext.STATUS_CLASSES}
    assert "recent" not in res["status"]
    assert res["forecast"]["river_id"] == RID and res["forecast"]["correction"]["skill"]["corrected"]["kge"] is not None
    assert res["sentence"].startswith(res["status"]["sentence"])
    assert prefs == ["nearest"]  # a gauge's position is on its own river


def test_now_needs_a_proper_station():
    with pytest.raises(ValueError):
        nownext.now(station="no-slash")


def test_today_is_utc():
    assert isinstance(nownext._today(), date)


def test_a_river_id_without_a_position_says_glofas_needs_one(models):
    fc = nownext.forecast(river_id=RID)
    assert fc["glofas"]["error"].startswith("GloFAS is read at a position") and fc["geoglows"]["mean"]
