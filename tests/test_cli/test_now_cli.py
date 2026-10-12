"""`aquascope now` and the MCP tools over aquascope.nownext (#517): thin faces over one engine."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from aquascope import cli, mcp_server, nownext

FC = {
    "river_id": 750170607, "snap": {"message": "Snapped 23 m to river reach 750170607."},
    "geoglows": {"date": ["2026-10-07", "2026-10-08"], "mean": [0.6, 1.9], "p25": [0.5, 0.7], "p75": [0.7, 1.7]},
    "glofas": {"date": ["2026-10-08", "2026-10-09"], "mean": [6.9, 6.4]},
    "thresholds": {"return_periods": [2, 5], "q": [133.8, 195.7], "source": "GEOGLOWS", "method": "LP3"},
    "gauge_thresholds": {"return_periods": [2, 5], "q": [206.8, 323.3], "source": "the gauge", "method": "LP3"},
    "correction": {"forecast": {"date": ["2026-10-07", "2026-10-08"], "mean": [0.28, 3.1]},
                   "skill_line": "Corrected forecast: KGE 0.47 on the 1992-2026 hindcast, raw 0.21.",
                   "skill": {"by": "month"}},
    "sentence": "Corrected to the gauge, the GEOGLOWS ensemble mean peaks at 3.1 m³/s on 8 October.",
}
NOW = {"station": {"source": "usgs", "station_id": "USGS-01350000", "variable": "discharge",
                   "name": "SCHOHARIE CREEK"},
       "status": {"sentence": "Flow is normal for 7 October (72nd percentile of 123 years).", "class": "normal",
                  "top_up": "Added 3 recent days from the agency, to 2026-10-07."},
       "forecast": FC, "sentence": "x"}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "now", *argv])
    cli.main()


def test_now_for_a_station_prints_status_table_thresholds_and_skill(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake(lat, lon, **kw):
        seen.update(lat=lat, lon=lon, **kw)
        return NOW

    monkeypatch.setattr(nownext, "now", fake)
    out_csv = tmp_path / "f.csv"
    _run(monkeypatch, "--station", "usgs/USGS-01350000", "--csv", str(out_csv))
    out = capsys.readouterr().out
    assert seen["station"] == "usgs/USGS-01350000" and seen["lat"] is None and seen["correct"] is True
    assert "Flow is normal for 7 October" in out and "Added 3 recent days" in out
    assert "corrected" in out and "2026-10-09" in out
    assert "Thresholds from the gauge, LP3: 2-yr 206.8, 5-yr 323.3" in out
    assert "KGE 0.47" in out and "CC BY 4.0" in out
    lines = out_csv.read_text().splitlines()
    assert lines[0] == "model,date,mean,median,p25,p75,min,max,high_res,corrected"
    assert lines[2].startswith("geoglows,2026-10-08,1.9,,0.7,1.7,,,,3.1") and lines[-1].startswith("glofas,")


def test_now_at_a_point_and_as_json(monkeypatch, capsys):
    seen = {}

    def fake(lat, lon, **kw):
        seen.update(lat=lat, lon=lon, **kw)
        return {"station": None, "status": None, "forecast": {**FC, "correction": None}, "sentence": "s"}

    monkeypatch.setattr(nownext, "now", fake)
    _run(monkeypatch, "46.948", "7.452", "--days", "10", "--raw", "--json")
    assert json.loads(capsys.readouterr().out)["sentence"] == "s"
    assert (seen["lat"], seen["lon"], seen["days"], seen["correct"]) == (46.948, 7.452, 10, False)
    assert seen["history"] is True
    _run(monkeypatch, "46.948", "7.452", "--quick", "--json")
    capsys.readouterr()
    assert seen["history"] is False


def test_now_without_a_place_says_what_to_give(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch)
    assert exc.value.code == 2 and "Give LAT LON" in capsys.readouterr().out


def test_the_mcp_tools_are_registered_and_dispatch(monkeypatch):
    names = {t.name for t in asyncio.run(mcp_server.build_server().list_tools())}
    assert {"flow_status", "flow_forecast", "correct_to_gauge"} <= names

    monkeypatch.setattr(nownext, "station_status", lambda source, sid, date=None: {
        "sentence": "Flow is normal.", "recent": {"t": [], "v": []}, "source": source, "date": date})
    st = mcp_server.flow_status("usgs", "USGS-1", date="2026-01-01")
    assert st["sentence"] == "Flow is normal." and "recent" not in st and st["date"] == "2026-01-01"

    calls = []
    monkeypatch.setattr(nownext, "now", lambda lat=None, lon=None, **kw: calls.append((lat, lon, kw)) or NOW)
    assert mcp_server.flow_forecast(46.9, 7.4, days=7)["forecast"] is FC
    assert calls[-1][2]["history"] is True
    mcp_server.flow_forecast(46.9, 7.4, quick=True)
    assert calls[-1][2]["history"] is False
    corr = mcp_server.correct_to_gauge("usgs", "USGS-1")
    assert calls[-1][2]["station"] == "usgs/USGS-1"
    assert corr["correction"]["skill_line"].startswith("Corrected forecast: KGE 0.47")
    assert corr["raw"] is FC["geoglows"] and corr["gauge_thresholds"]["q"][0] == 206.8

    def bad(*a, **k):
        raise ValueError("give the station as source/station_id")

    monkeypatch.setattr(nownext, "now", bad)
    assert "source/station_id" in mcp_server.flow_forecast(station="x")["error"]
