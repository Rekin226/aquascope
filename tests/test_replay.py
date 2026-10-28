"""Replay an event (#659): the record's pieces, the story, the report, and Route it with a real hydrograph."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from aquascope import replay, routing
from tests.test_routing import TOY, TOY_IDS, toy  # noqa: F401 - the toy network fixture

# ── the window ──────────────────────────────────────────────────────────────


def test_the_window_runs_from_before_the_event_to_after_it_and_never_past_yesterday():
    a, b = replay.event_window("2024-11-19", "2024-11-29")
    assert (a, b) == (date(2024, 11, 11), date(2024, 12, 5))
    a, b = replay.event_window("2024-01-01", "2024-12-31")
    assert (b - a).days + 1 == replay.MAX_DAYS
    today = date.today()
    a, b = replay.event_window(today - timedelta(days=2))
    assert b == today - timedelta(days=1)


# ── Route it with a real hydrograph ─────────────────────────────────────────


def test_route_takes_a_real_hydrograph_and_starts_every_reach_in_flow(toy):  # noqa: F811
    q = [100.0] * 24 + [100.0 + 20 * h for h in range(1, 25)] + [580.0 - 10 * h for h in range(1, 49)]
    res = routing.route(TOY_IDS["A"], inflow_cms=q, inflow_source="a gauge", along=False)
    assert res["ok"] and res["inflow"]["shape"] == "series" and res["inflow"]["source"] == "a gauge"
    assert res["hours"] == len(q) - 1 and res["inflow"]["peak_cms"] == 580.0
    first, last = res["reaches"][0], res["reaches"][-1]
    assert first["peak_h"] == 47.0 and last["peak_h"] > first["peak_h"]
    assert last["peak_cms"] <= 580.0 and last["peak_cms"] > 100.0
    assert res["message"].startswith("A hydrograph peaking at 580")
    # The flow already in the river is there from the first hour (no filling from empty).
    flat = routing.route(TOY_IDS["A"], inflow_cms=[100.0] * 30, along=False)
    assert all(r["peak_cms"] == pytest.approx(100.0) for r in flat["reaches"])
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], inflow_cms=[1.0], along=False)
    with pytest.raises(ValueError):
        routing.route(TOY_IDS["A"], inflow_cms=[1.0, -2.0], along=False)


# ── the rain ────────────────────────────────────────────────────────────────


def test_the_basin_goes_to_climateserv_as_cells_holding_the_upstream_catchments(toy):  # noqa: F811
    cells = replay.basin_cells(TOY_IDS["C"])
    assert cells["geometry"]["type"] == "MultiPolygon" and cells["cell_deg"] == 0.1
    assert cells["n_reaches"] == len(TOY) and cells["area_km2"] == pytest.approx(330.0)
    xs = [x for poly in cells["geometry"]["coordinates"] for x, _ in poly[0]]
    assert min(xs) <= 8.0 and max(xs) >= 8.4


def test_the_rain_is_read_from_climateserv_once_it_is_done(monkeypatch):
    calls = []
    answers = {
        "progress": [[42.0], [100.0]],
        "data": {"data": [{"date": "11/23/2024", "value": {"avg": 41.94}},
                          {"date": "11/22/2024", "value": {"avg": 3.0}},
                          {"date": "11/24/2024", "value": {"avg": 12.5}}, {"date": "bad", "value": {"avg": 1}}]},
    }

    def fake(url):
        calls.append(url)
        if "Progress" in url:
            return answers["progress"].pop(0)
        return answers["data"]

    monkeypatch.setitem(replay.FETCH, "json", fake)
    piece = {"piece": "rain", "request": "abc", "pending": True, "source": "s", "licence": "l", "coverage": "c",
             "basin": {"area_km2": 6884.0}, "window": ["2024-11-22", "2024-11-24"]}
    once = replay.rain_poll(piece)
    assert once["pending"] and once["progress"] == 42.0
    done = replay.rain_poll(once)
    assert done["available"] and done["total_mm"] == 57.4 and done["peak"] == {"date": "2024-11-23", "mm": 41.9}
    assert [d for d, _ in done["daily_mm"]] == ["2024-11-22", "2024-11-23", "2024-11-24"]
    assert all("climateserv.servirglobal.net" in u and "id=abc" in u for u in calls)


def test_a_rain_request_names_imerg_v07_and_the_basin_average(monkeypatch):
    seen = {}
    monkeypatch.setitem(replay.FETCH, "json", lambda url: seen.setdefault("url", url) and ["req-1"])
    rid = replay.rain_request({"type": "MultiPolygon", "coordinates": []}, date(2024, 11, 11), date(2024, 12, 5))
    assert rid == "req-1"
    assert "datatype=226" in seen["url"] and "operationtype=5" in seen["url"]
    assert "begintime=11%2F11%2F2024" in seen["url"]


# ── the forecast archive ────────────────────────────────────────────────────


def test_the_forecast_column_is_found_by_the_region_of_each_chunk(monkeypatch):
    chunks = {0: np.array([110000001, 110000002, 120000001]), 1: np.array([120000005, 230000001, 230567275]),
              2: np.array([230000009, 410000001, 0])}
    meta = {"rivid/.zarray": {"shape": [8], "chunks": [3]}}
    monkeypatch.setattr(replay, "_RIVID", {})
    monkeypatch.setitem(replay.FETCH, "bytes", lambda url: url.encode())
    monkeypatch.setattr(replay, "_decode", lambda raw, m: chunks[int(raw.decode().rsplit("/", 1)[1])])
    assert replay._forecast_column(230567275, "base", meta) == 5
    assert replay._forecast_column(230000009, "base", meta) == 6
    assert replay._forecast_column(230000002, "base", meta) is None


def test_the_forecast_says_what_it_said_for_the_peak_day_at_each_lead(monkeypatch):
    def fake_run(rid, run):
        start = date(int(run[:4]), int(run[4:6]), int(run[6:8]))
        ens = np.tile(np.arange(15, dtype=float) * 10, (51, 1)) + np.arange(51)[:, None]
        return {"start": start, "ens": ens, "hres": ens[0]}

    monkeypatch.setattr(replay, "_forecast_run", fake_run)
    curve = {"T": [2, 5], "q": [40.0, 80.0]}
    fc = replay.forecasts(1, "2024-11-26", curve=curve, observed_cms=100.0)
    assert fc["available"] and [r["lead_days"] for r in fc["runs"]] == [1, 3, 7]
    one = fc["runs"][0]
    assert one["issued"] == "2024-11-25" and one["median_cms"] == 35.0 and (one["low_cms"], one["high_cms"]) == (10, 60)
    assert one["share_above_q2"] == pytest.approx(21 / 51, abs=1e-3) and one["median_vs_observed_pct"] == -65.0
    assert fc["runs"][2]["median_cms"] == 95.0
    early = replay.forecasts(1, "2024-07-01")
    assert not early["available"] and "archive starts" in early["note"]


# ── Floods past: the candidates ─────────────────────────────────────────────


def test_candidates_group_the_reports_into_runs_of_days(monkeypatch):
    from aquascope.context import floods_past

    ev = [("2024-11-23", "2024-11-24"), ("2024-11-25", None), ("2024-11-27", None), ("2024-11-13", "2024-11-15"),
          ("2024-11-01", None), ("2024-11-30", None)]
    events = [{"start": s, "end": e, "lat": 52.0 + k * 0.01, "lon": -2.2, "area_km2": 1.0}
              for k, (s, e) in enumerate(ev)]
    monkeypatch.setattr(floods_past, "flood_events_month", lambda *a, **k: {
        "ok": True, "available": True, "window": "Nov 2024", "news": {"events": events}})
    c = replay.candidates(52.0, -2.2, month="2024-11")
    assert [(x["start"], x["end"], x["reports"]) for x in c["candidates"]] == [
        ("2024-11-23", "2024-11-27", 3), ("2024-11-01", "2024-11-01", 1), ("2024-11-13", "2024-11-15", 1),
        ("2024-11-30", "2024-11-30", 1)]
    assert c["candidates"][0]["rank"] == 1 and "Groundsource" in c["source"]


# ── the story and the report ────────────────────────────────────────────────


def _record():
    obs = [["2024-11-24", 324.5], ["2024-11-25", 423.4], ["2024-11-26", 432.6], ["2024-11-27", 428.0]]
    mod = [["2024-11-24", 80.0], ["2024-11-25", 101.0], ["2024-11-26", 112.2], ["2024-11-27", 110.0]]
    basin = {"bbox": [-3.9, 52.0, -2.2, 52.9], "area_km2": 6884.0, "cell_deg": 0.1, "cells": 80,
             "geometry": {"type": "MultiPolygon", "coordinates": []}}
    return {
        "made": "2026-10-27T12:00", "name": "Saxons Lode, November 2024",
        "event": {"start": "2024-11-19", "end": "2024-11-29", "lat": 52.02, "lon": -2.36, "reports": 169,
                  "from": "Floods past"},
        "window": ["2024-11-11", "2024-12-05"], "reach": {"river_id": 230567275, "how": "given"},
        "peak": {"date": "2024-11-26", "cms": 432.6, "from": "the daily mean observed at Saxons Lode",
                 "river_id": 230567275},
        "focus": {"lat": 52.0496, "lon": -2.2005, "name": "Saxons Lode", "river_id": 230567275},
        "pieces": {
            "gauges": {"available": True, "source": "s", "licence": "l", "coverage": "c", "gauges": [
                {"name": "Saxons Lode", "source": "uk_ea", "station_id": "x", "lat": 52.0496, "lon": -2.2005,
                 "observed": obs, "modelled": mod, "obs_peak": {"date": "2024-11-26", "cms": 432.6},
                 "model_peak": {"date": "2024-11-26", "cms": 112.2}, "peak_error_pct": -74.1, "timing_days": 0}]},
            "model": {"available": True, "peak": {"date": "2024-11-26", "cms": 112.2}, "return_period": 2.4,
                      "daily": mod, "river_id": 230567275, "source": "s", "licence": "l", "coverage": "c"},
            "rain": {"available": True, "total_mm": 114.2, "peak": {"date": "2024-11-23", "mm": 41.9},
                     "basin": basin, "window": ["2024-11-11", "2024-12-05"], "daily_mm": [["2024-11-23", 41.9]],
                     "source": "IMERG", "licence": "NASA", "coverage": "c"},
            "forecast": {"available": True, "peak_date": "2024-11-26", "source": "s", "licence": "l", "coverage": "c",
                         "runs": [{"lead_days": 1, "issued": "2024-11-25", "median_cms": 63.4, "low_cms": 62.9,
                                   "high_cms": 65.0, "q2_cms": 133.0, "share_above_q2": 0.0,
                                   "median_vs_observed_pct": -85.3},
                                  {"lead_days": 7, "issued": "2024-11-19", "note": "no run issued that day"}]},
            "wave": {"available": True, "inflow_source": "the daily flow observed at Saxons Lode",
                     "inflow_peak_cms": 432.6, "places": [
                         {"name": "Over Farm", "kind": "gauge", "lag_h": 10.0, "peak_cms": 431.0, "along_km": 30.0,
                          "peak_at": "2024-11-26T10:00"},
                         {"name": "Epney", "kind": "gauge", "lag_h": 18.0, "peak_cms": 430.0, "along_km": 50.0,
                          "peak_at": "2024-11-26T18:00"}],
                     "geometry": {"type": "LineString", "coordinates": [[-2.2, 52.05], [-2.3, 51.8]]},
                     "source": "Route it", "licence": "l", "coverage": "c"},
            "flood": {"available": False, "pending": True, "flow_cms": 432.6, "river_id": 230567275,
                      "flow_source": "the daily mean observed at Saxons Lode", "licence": "l"},
        },
    }


def test_every_number_in_a_note_comes_from_the_record():
    st = replay.story(_record())
    assert st["schema"] == "aquascope.story" and st["version"] == 1
    kinds = [s["kind"] for s in st["stops"]]
    assert kinds == ["event", "rain", "gauge", "forecast", "forecast", "wave", "flood", "exposure", "end"]
    notes = {s["id"]: s["note"] for s in st["stops"]}
    assert notes["event"] == "169 reports of flooding in the news near Saxons Lode, 19 Nov 2024 to 29 Nov 2024."
    assert "114 mm fell on the 6,884 km² upstream" in notes["rain"] and "23 Nov 2024, brought 42 mm" in notes["rain"]
    assert notes["gauge"].startswith("Saxons Lode peaked at 433 m³/s on 26 Nov 2024 (daily mean). GEOGLOWS simulated "
                                     "112 m³/s on 26 Nov 2024: -74 %, on the same day.")
    assert "a 2-year flow on its own record" in notes["gauge"]
    assert "the forecast's median was 63 m³/s (members 63 m³/s to 65 m³/s), -85 % against the gauge; 0 % of members" \
        in notes["forecast-1"]
    assert "no archived forecast (no run issued that day)" in notes["forecast-7"]
    assert "the peak reaches Epney 18 h later, at 430 m³/s" in notes["wave"]
    for s in st["stops"]:
        c = s["camera"]
        assert len(c["center"]) == 2 and 3 <= c["zoom"] <= 13 and 0 <= c["pitch"] <= 60
    json.dumps(st)


def test_the_flood_stops_carry_a_template_the_page_fills_or_the_numbers_given():
    st = replay.story(_record())
    flood = next(s for s in st["stops"] if s["kind"] == "flood")
    expo = next(s for s in st["stops"] if s["kind"] == "exposure")
    assert flood["needs"] == "flood" and "{wet_km2}" in flood["template"] and flood["run"]["flow_cms"] == 432.6
    assert expo["needs"] == "exposure" and "{buildings}" in expo["template"]
    done = replay.story(_record(), flood_result={"wet_km2": 41.2, "max_depth_m": 6.04, "exposure": {
        "available": True, "buildings": 1234, "roads_km": 45.67, "people": 2050.4}})
    notes = {s["id"]: s["note"] for s in done["stops"]}
    assert notes["flood"].startswith("At 433 m³/s (the daily mean observed at Saxons Lode), Flood it puts 41.2 km² "
                                     "under water, 6.0 m at the deepest. A screening flood")
    assert notes["exposure"] == "Under that water: 1,234 buildings, 45.7 km of road and about 2,050 people."


def test_a_missing_piece_says_so_and_the_story_goes_on():
    rec = _record()
    rec["pieces"]["rain"] = {"available": False, "note": "IMERG starts on 1 Jun 2000.", "licence": "NASA"}
    rec["pieces"]["forecast"] = {"available": False, "runs": [], "note": "No archived forecast covers this peak."}
    st = replay.story(rec)
    notes = {s["id"]: s["note"] for s in st["stops"]}
    assert notes["rain"] == "Basin rain not available: IMERG starts on 1 Jun 2000."
    assert notes["forecast"] == "Forecast not available: No archived forecast covers this peak."
    assert "Not available: rain, forecast." in notes["end"]
    assert {s["piece"]: s["available"] for s in st["sources"]}["rain"] is False


def test_the_report_has_the_numbers_the_sources_and_the_licences():
    md = replay.report(_record())
    assert md.startswith("# Replay: Saxons Lode, November 2024")
    assert "| Saxons Lode (uk_ea x) | 433 m³/s, 2024-11-26 | 112 m³/s, 2024-11-26 | -74 % | +0.0 d |" in md
    assert "| 2024-11-25 | 1 d | 63 m³/s | 63 m³/s to 65 m³/s |" in md
    assert "| Epney (gauge) | 50.0 | 2024-11-26T18:00 | 18.0 | 430 m³/s |" in md
    assert "Groundsource news events (Google), CC BY 4.0" in md and "none was written by a model" in md


def test_a_gauge_far_down_another_river_is_not_the_event_s(toy, monkeypatch):  # noqa: F811
    stem = replay._stem(TOY_IDS["B"])
    assert stem[TOY_IDS["B"]] == 0 and stem[TOY_IDS["C"]] == 1
    assert all(v != 0 for k, v in stem.items() if k != TOY_IDS["B"])


def test_a_flood_near_a_tributary_is_the_main_river_s(monkeypatch):
    """A news event's centre beside a tributary (the Leadon beside the Severn): the reach picked is the main
    river's, the largest upstream area within 25 km, at its reach nearest the centre, not the nearer tributary."""
    from aquascope import rivers

    ids = {"M0": 230900001, "M1": 230900002, "M2": 230900003, "M3": 230900004, "T0": 230900005, "T1": 230900006}
    ds = {"M0": "M1", "M1": "M2", "M2": "M3", "M3": None, "T0": "T1", "T1": "M2"}
    where = {"M0": (-2.30, 52.20), "M1": (-2.25, 52.10), "M2": (-2.25, 52.00), "M3": (-2.25, 51.80),
             "T0": (-2.45, 52.08), "T1": (-2.40, 52.05)}
    area = {"M0": 1e9, "M1": 1e9, "M2": 1e9, "M3": 1e9, "T0": 5e7, "T1": 5e7}
    names = list(ids)
    net = rivers._Network(230, [ids[n] for n in names], [ids[ds[n]] if ds[n] else -1 for n in names],
                          [area[n] for n in names], [where[n][0] for n in names], [where[n][1] for n in names])
    monkeypatch.setattr(rivers, "_network_for", lambda rid, lat=None, lon=None: net)
    # The tiles see the tributary near the centre, with a higher order than anything else within 12 km.
    near = [{"river_id": ids["T0"], "strahler_order": 3, "distance_m": 700.0, "lat": 52.08, "lon": -2.445}]
    monkeypatch.setattr(rivers, "reaches_near", lambda lat, lon, max_distance_m=2000.0, limit=6: near)
    got = replay._reach_for(None, 52.08, -2.44)
    assert got["river_id"] == ids["M1"] and "largest upstream area" in got["how"]
    # When the main river's line is within reach of the tiles, its nearest line wins.
    near2 = near + [{"river_id": ids["M2"], "strahler_order": 5, "distance_m": 9000.0, "lat": 52.0, "lon": -2.3}]
    monkeypatch.setattr(rivers, "reaches_near", lambda lat, lon, max_distance_m=2000.0, limit=6: near2)
    assert replay._reach_for(None, 52.08, -2.44)["river_id"] == ids["M2"]
    # Without the network the Strahler pick stands.
    monkeypatch.setattr(rivers, "_network_for", lambda rid, lat=None, lon=None: (_ for _ in ()).throw(OSError("x")))
    assert replay._reach_for(None, 52.08, -2.44)["river_id"] == ids["M2"]
