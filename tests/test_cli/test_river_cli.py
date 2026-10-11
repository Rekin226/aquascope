"""`aquascope river snap|record|area|trace`, the MCP tools and the Analyst tool over aquascope.rivers (#516)."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from aquascope import cli, rivers

SNAP = {"lat": 46.948, "lon": 7.452, "snapped": True, "river_id": 230260670, "strahler_order": 5,
        "distance_m": 199.8, "snap_lat": 46.9498, "snap_lon": 7.4521, "max_distance_m": 1000.0,
        "message": "Snapped 200 m to river reach 230260670, Strahler order 5."}
HILLSIDE = {"lat": 46.6, "lon": 7.9, "snapped": False, "river_id": None, "max_distance_m": 200.0,
            "nearest": {"river_id": 230172557, "distance_m": 554.0},
            "message": "No stream within 200 m of this point. The nearest mapped reach is 230172557, 554 m away."}
RECORD = {"river_id": 230260670, "modelled": True, "start": "1940-01-01", "end": "2026-09-30", "years": 86.7,
          "stats": {"mean": 120.5, "max": 1038.7}, "fdc": {"q95": 40.1, "q50": 98.2, "q10": 230.4},
          "ffa": {"n_years": 86, "return_periods": [2, 100],
                  "fits": {"gev_lmoments": {"q": [615.0, 1127.3]},
                           "lp3": {"q": [621.3, 1087.9], "ci": [[589.9, 654.5], [990.2, 1195.4]]}}},
          "notes": ["Simulated by the GEOGLOWS v2 hydrologic model, not measured."],
          "attribution": rivers.ATTRIBUTION, "series": {"t": [], "v": []}}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "river", *argv])
    cli.main()


def test_river_snap_prints_the_reach(monkeypatch, capsys):
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0, **kw: SNAP)
    _run(monkeypatch, "snap", "46.948", "7.452")
    out = capsys.readouterr().out
    assert "Snapped 200 m to river reach 230260670" in out and "46.94980" in out


def test_river_snap_json_on_a_hillside(monkeypatch, capsys):
    seen = {}

    def fake(lat, lon, max_distance_m=1000.0, prefer="main", area_km2=None):
        seen.update(max=max_distance_m, prefer=prefer, area=area_km2)
        return HILLSIDE

    monkeypatch.setattr(rivers, "snap_to_river", fake)
    _run(monkeypatch, "snap", "46.6", "7.9", "--max-distance", "200", "--json")
    assert json.loads(capsys.readouterr().out)["snapped"] is False and seen["max"] == 200.0
    assert seen["prefer"] == "main" and seen["area"] is None
    _run(monkeypatch, "snap", "46.6", "7.9", "--nearest", "--area", "2941")
    assert seen["prefer"] == "nearest" and seen["area"] == 2941.0


def test_river_record_prints_the_modelled_table_and_writes_the_csv(monkeypatch, capsys, tmp_path):
    def fake(rid, years=None, store=None, **kw):
        assert rid == 230260670 and years == 30
        if store is not None:
            store["series"] = pd.Series([1.5, 2.25], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
        return dict(RECORD)

    monkeypatch.setattr(rivers, "reach_record", fake)
    out_csv = tmp_path / "reach.csv"
    _run(monkeypatch, "record", "230260670", "--years", "30", "--csv", str(out_csv))
    out = capsys.readouterr().out
    assert "MODELLED daily discharge, 1940-01-01 to 2026-09-30" in out
    assert "100-yr  GEV 1,127" in out and "90 % CI 990.2 to 1,195" in out
    assert "GEOGLOWS v2" in out
    assert out_csv.read_text().splitlines() == ["date,discharge_m3_per_s_modelled", "2020-01-01,1.5", "2020-01-02,2.25"]


def test_river_record_at_a_hillside_exits_with_the_reason(monkeypatch, capsys):
    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: HILLSIDE)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "record", "--at", "46.6", "7.9")
    assert exc.value.code == 1 and "No stream within 200 m" in capsys.readouterr().out


def test_river_trace_lists_the_gauges_and_writes_geojson(monkeypatch, capsys, tmp_path):
    trace = {"message": "320 reaches, 1,078 km to the outlet.", "upstream_area_km2": 3019.4,
             "geometry": {"type": "LineString", "coordinates": [[7.45, 46.95], [4.08, 51.95]]},
             "geometry_licence": "TDX-Hydro, CC BY-SA 4.0", "length_km": 1078.0,
             "gauges": [{"along_km": 222.1, "source": "pegelonline", "station_id": "x", "name": "BASEL"}],
             "dams": [{"along_km": 23.8, "name": "Muehleberg", "capacity_mcm": 25.0, "purpose": "Hydroelectricity"}],
             "dams_info": {"summary": "1 dam within 2 km of the path."},
             "countries_info": {"summary": "Crosses Switzerland, Germany, France and Netherlands."},
             "upstream_dams": {"summary": "Regulated upstream: 7 dams in Global Dam Watch drain to this reach."},
             "notes": ["A note."]}
    seen = {}

    def fake(rid, gauge_km=2.0, dam_km=2.0):
        seen.update(gauge_km=gauge_km, dam_km=dam_km)
        return trace

    monkeypatch.setattr(rivers, "trace_downstream", fake)
    path = tmp_path / "trace.geojson"
    _run(monkeypatch, "trace", "230260670", "--geojson", str(path), "--dam-km", "1.5")
    out = capsys.readouterr().out
    assert "1,078 km to the outlet" in out and "3,019 km2" in out and "pegelonline/x" in out
    assert "km    23.8  Muehleberg  25.0 million m3  Hydroelectricity" in out and seen["dam_km"] == 1.5
    assert "Crosses Switzerland" in out and "Upstream: Regulated upstream: 7 dams" in out
    fc = json.loads(path.read_text())
    assert fc["features"][0]["properties"]["licence"].startswith("TDX-Hydro")


def test_river_area(monkeypatch, capsys):
    monkeypatch.setattr(rivers, "upstream_area", lambda rid: {
        "upstream_area_km2": 3019.4, "n_reaches_upstream": 950, "vpu": 209, "note": "Approximate."})
    _run(monkeypatch, "area", "230260670")
    assert "3,019.4 km2 drain to it, 950 reaches upstream" in capsys.readouterr().out


def test_river_area_at_a_point_passes_where_the_reach_is(monkeypatch, capsys):
    """With --at, the snapped position goes along, so the right processing unit is read first."""
    seen = {}

    def fake(rid, lat=None, lon=None):
        seen.update(rid=rid, lat=lat, lon=lon)
        return {"upstream_area_km2": 3019.4, "n_reaches_upstream": 950, "vpu": 209, "note": "Approximate."}

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: SNAP)
    monkeypatch.setattr(rivers, "upstream_area", fake)
    _run(monkeypatch, "area", "--at", "46.948", "7.452")
    assert seen == {"rid": 230260670, "lat": 46.9498, "lon": 7.4521}


def test_river_dams_lists_the_dams_upstream(monkeypatch, capsys):
    seen = {}

    def fake(rid, with_flow=True, lat=None, lon=None):
        seen.update(rid=rid, with_flow=with_flow, lat=lat)
        return {"summary": "Regulated upstream: 1 dam in Global Dam Watch drains to this reach.",
                "dams": [{"name": "Spitallamm", "capacity_mcm": 101.0, "purpose": None}],
                "note": "Approximate.", "source": {"attribution": "Global Dam Watch database v1.0, CC BY 4.0"}}

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: SNAP)
    monkeypatch.setattr(rivers, "upstream_dams", fake)
    _run(monkeypatch, "dams", "--at", "46.948", "7.452", "--no-flow")
    out = capsys.readouterr().out
    assert "River reach 230260670: Regulated upstream: 1 dam" in out
    assert "Spitallamm" in out and "101.0 million m3" in out
    assert seen == {"rid": 230260670, "with_flow": False, "lat": 46.9498} and "CC BY 4.0" in out


def test_river_upstream_and_downstream_print_the_ids(monkeypatch, capsys):
    seen = {}

    def up(rid, max_n=20_000, lat=None, lon=None):
        seen.update(up=(rid, max_n, lat))
        return {"ids": list(range(230000001, 230000021)), "message": "20 reaches drain to river reach 230000001."}

    def down(rid, max_n=5_000, lat=None, lon=None):
        seen.update(down=(rid, max_n))
        return {"ids": [230000001, 230000002], "message": "2 reaches from river reach 230000001 to the outlet."}

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0: SNAP)
    monkeypatch.setattr(rivers, "upstream_ids", up)
    monkeypatch.setattr(rivers, "downstream_ids", down)
    _run(monkeypatch, "upstream", "--at", "46.948", "7.452", "--max", "50")
    out = capsys.readouterr().out
    assert "20 reaches drain" in out and "230000006 ... 230000015" in out
    assert seen["up"] == (230260670, 50, 46.9498)
    _run(monkeypatch, "downstream", "230000001", "--json")
    assert json.loads(capsys.readouterr().out)["ids"] == [230000001, 230000002]
    assert seen["down"] == (230000001, 5000)


def test_river_needs_a_reach_or_a_point(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "record")
    assert exc.value.code == 2


# ── the other faces ──────────────────────────────────────────────────────────


def test_mcp_tools_wrap_the_engine(monkeypatch):
    from aquascope import mcp_server as m

    monkeypatch.setattr(rivers, "snap_to_river", lambda lat, lon, max_distance_m=1000.0, **kw: SNAP)
    assert m.snap_to_river(46.948, 7.452)["river_id"] == 230260670
    full = {**RECORD, "fdc": {"q95": 1, "q50": 2, "q10": 3, "exceedance": [1], "q": [1]}}
    monkeypatch.setattr(rivers, "reach_record", lambda *a, **k: json.loads(json.dumps(full)))
    res = m.reach_record(river_id=230260670)
    assert "series" not in res and set(res["fdc"]) == {"q95", "q50", "q10"}
    reaches = [{"river_id": i} for i in range(100)]
    monkeypatch.setattr(rivers, "trace_downstream", lambda *a, **k: {"reaches": list(reaches)})
    res = m.trace_downstream(230260670)
    assert len(res["reaches"]) == 40 and "100 reaches" in res["reaches_note"]
    dams = [{"name": f"d{i}", "along_km": float(i), "capacity_mcm": float(i)} for i in range(50)]
    monkeypatch.setattr(rivers, "trace_downstream", lambda *a, **k: {"dams": list(dams)})
    res = m.trace_downstream(230260670)
    assert len(res["dams"]) == 30 and res["dams"][0]["name"] == "d20" and "50 dams" in res["dams_note"]
    monkeypatch.setattr(rivers, "upstream_dams", lambda rid, lat=None, lon=None, with_flow=True: {
        "river_id": rid, "with_flow": with_flow})
    assert m.upstream_dams(230260670, with_flow=False) == {"river_id": 230260670, "with_flow": False}
    monkeypatch.setattr(rivers, "upstream_ids", lambda rid, max_n=0, lat=None, lon=None: {"n": max_n})
    assert m.upstream_ids(230260670)["n"] == 200 and m.upstream_ids(230260670, max_n=10**9)["n"] == 20_000
    monkeypatch.setattr(rivers, "downstream_ids", lambda rid, max_n=0, lat=None, lon=None: {"n": max_n})
    assert m.downstream_ids(230260670)["n"] == 5000


def test_the_mcp_server_registers_the_river_tools():
    import asyncio

    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert {"snap_to_river", "reach_record", "upstream_area", "trace_downstream", "upstream_dams",
            "upstream_ids", "downstream_ids"} <= names


def test_the_analyst_tool_and_the_team_sentence(monkeypatch):
    from aquascope.ai_engine import analyst, team
    from aquascope.study import Study

    spec = {s.name: s for s in analyst._tool_specs()}["reach_record"]
    monkeypatch.setattr(rivers, "reach_record", lambda *a, **k: json.loads(json.dumps(
        {**RECORD, "snap": {"distance_m": 18.0}})))
    payload = spec.func(lat=46.948, lon=7.452)
    assert "series" not in payload
    study = Study(question="q", steps=[], problem={"params": {"return_period": 100}})
    text = " ".join(team._sentences_for("reach_record", payload, study))
    assert "MODELLED, not measured" in text and "reach 230260670, 18 m from the site" in text
    assert "100-year GEV 1,127 m3/s" in text
    err = team._sentences_for("reach_record", {"error": "No stream within 1 km of this point."}, study)
    assert err == ["GEOGLOWS reach record not used: No stream within 1 km of this point."]
