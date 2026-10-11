"""`aquascope now --plume` and the MCP tool `forecast_plume` over aquascope.nownext.plume (#556): thin faces."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from aquascope import cli, mcp_server, nownext

PLUME = {
    "river_id": 760716396, "issued": "2026-10-10", "n_members": 51, "date": ["2026-10-10", "2026-10-11"],
    "median": [2213.0, 2000.0], "p25": [2200.0, 1990.0], "p75": [2220.0, 2010.0], "min": [2190.0, None],
    "max": [2240.0, 2050.0], "class_daily": [5, 2], "members_at": {"2": [51, 51]},
    "thresholds": {"return_periods": [2, 5], "q": [1290.0, 2030.0], "source": "the Floods ahead issue"},
    "sentence": "The ensemble mean peaks at 2,213 m³/s on 10 October, above the 5-year flow (2,030 m³/s).",
    "members_line": "In the 2 days, all 51 members reach the 2-year flow and all the 5-year flow.",
    "attribution": "GEOGLOWS v2 forecast (GEOGloWS ECMWF Streamflow Service), CC BY 4.0",
}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["aquascope", "now", *argv])
    cli.main()


def test_now_plume_prints_the_days_classes_thresholds_and_members(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(nownext, "plume", lambda rid, **kw: seen.update(rid=rid, **kw) or PLUME)
    _run(monkeypatch, "--river-id", "760716396", "--plume")
    out = capsys.readouterr().out
    assert seen["rid"] == 760716396 and seen["history"] is True and seen["lat"] is None
    assert "River reach 760716396, GEOGLOWS run of 2026-10-10 (51 members, modelled)" in out
    assert "2026-10-10" in out and "5-yr" in out and "2,190 - 2,240" in out and "- - 2,050" in out
    assert "Thresholds from the Floods ahead issue: 2-yr 1,290, 5-yr 2,030" in out
    assert "all 51 members reach the 2-year flow" in out and "not an official warning" in out


def test_now_plume_as_json_at_a_point_and_refuses_a_station(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(nownext, "plume", lambda rid, **kw: seen.update(rid=rid, **kw) or PLUME)
    _run(monkeypatch, "39.18", "-96.27", "--plume", "--json")
    assert json.loads(capsys.readouterr().out)["river_id"] == 760716396
    assert seen["rid"] is None and (seen["lat"], seen["lon"]) == (39.18, -96.27)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--station", "usgs/X", "--plume")
    assert exc.value.code == 2 and "river reach" in capsys.readouterr().out


def test_the_mcp_tool_is_registered_and_drops_the_per_day_member_counts(monkeypatch):
    names = {t.name for t in asyncio.run(mcp_server.build_server().list_tools())}
    assert "forecast_plume" in names
    monkeypatch.setattr(nownext, "plume", lambda rid, **kw: dict(PLUME))
    res = mcp_server.forecast_plume(760716396)
    assert res["members_line"].startswith("In the 2 days") and "members_at" not in res

    def bad(*a, **k):
        raise ValueError("give a river_id, or lat and lon")

    monkeypatch.setattr(nownext, "plume", bad)
    assert "river_id" in mcp_server.forecast_plume()["error"]
