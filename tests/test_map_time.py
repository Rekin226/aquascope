"""Time on the map (#522): the dated layers, their valid dates, and a time-lapse's frames.

One engine, three faces: the functions in aquascope.map_time, the MCP tools, the
`aquascope layers` CLI verb, and the Explorer's layers.js, which carries the same
first and last days (checked here). No network: the GIBS capabilities are a fixture.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from aquascope import cli
from aquascope import map_time as mt

ROOT = Path(__file__).resolve().parents[1]
LAYERS_JS = ROOT / "explorer" / "src" / "layers.js"

CAPS = """<Capabilities><Contents>
<Layer><ows:Identifier>IMERG_Precipitation_Rate</ows:Identifier>
<Dimension><ows:Identifier>Time</ows:Identifier><Default>2026-10-06</Default>
<Value>2000-06-01/2025-10-22/P1D</Value><Value>2025-10-30/2026-10-06/P1D</Value></Dimension></Layer>
<Layer><ows:Identifier>GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI</ows:Identifier>
<Dimension><ows:Identifier>Time</ows:Identifier><Default>2022-07-01</Default>
<Value>2002-04-04/2002-04-04/P28D</Value><Value>2002-05-02/2002-05-02/P17D</Value>
<Value>2002-08-01/2003-04-01/P1M</Value></Dimension></Layer>
<Layer><ows:Identifier>NoTime_Layer</ows:Identifier></Layer>
<Layer><ows:Identifier>Later_Layer</ows:Identifier>
<Dimension><Default>2020-01-01</Default><Value>2020-01-01/2020-02-01/P1D</Value></Dimension></Layer>
</Contents></Capabilities>"""


# ── dates and steps ─────────────────────────────────────────────────────────


def test_steps_move_by_day_week_and_calendar_month():
    assert mt.step_date("2024-12-31", "day") == "2025-01-01"
    assert mt.step_date("2024-01-01", "week", 2) == "2024-01-15"
    assert mt.step_date("2024-01-31", "month") == "2024-02-29"
    assert mt.step_date("2023-01-31", "month") == "2023-02-28"
    assert mt.step_date("2024-03-15", "month", -3) == "2023-12-15"
    with pytest.raises(ValueError):
        mt.step_date("2024-01-01", "hour")
    with pytest.raises(ValueError):
        mt.parse_date("14/07/2021")
    assert mt.parse_date("2021-07-14T06:00:00Z") == date(2021, 7, 14)


def test_frames_include_both_ends_do_not_drift_and_cap():
    assert mt.frame_dates("2024-01-31", "2024-04-30", "month")["dates"] == [
        "2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]
    assert mt.frame_dates("2024-01-03", "2024-01-01")["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    capped = mt.frame_dates("2000-01-01", "2024-01-01", "day", max_frames=5)
    assert capped == {"dates": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
                      "truncated": True}


def test_monthly_layers_ask_for_the_first_of_the_month_or_their_own_first_day():
    storage = mt._layer("storage")
    assert mt.layer_date(storage, "2010-05-17") == "2010-05-01"
    assert mt.layer_date(storage, "2002-04-20") == "2002-04-04"
    assert mt.layer_date(mt._layer("precip"), "2010-05-17") == "2010-05-17"
    assert mt.tile_template(storage, "2010-05-17").endswith(
        "/GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI/default/2010-05-01/"
        "GoogleMapsCompatible_Level6/{z}/{y}/{x}.png")
    assert not mt.in_range(storage, "2024-01-01")
    assert mt.in_range(mt._layer("precip"), "2024-01-01", today=date(2026, 10, 8))


def test_grace_asks_for_a_day_gibs_has_an_image_for():
    # Checked against GIBS tiles on 2026-10-08: the first of the month is a 404 for these months.
    storage = mt._layer("storage")
    assert mt.layer_date(storage, "2004-02-15") == "2004-02-04"
    assert mt.layer_date(storage, "2004-02-01") == "2004-02-04"
    assert mt.layer_date(storage, "2011-02-20") == "2011-02-08"
    assert mt.layer_date(storage, "2004-01-20") == "2004-01-01"
    assert mt.layer_date(storage, "2016-01-30") == "2016-01-29"
    assert mt.image_for(storage, "2002-06-15") is None
    assert mt.image_for(storage, "2017-09-01") is None
    assert mt.image_for(storage, "2022-08-01") is None, "the capabilities run to 2022-12, the tiles do not"
    assert not mt.in_range(storage, "2017-09-01") and mt.in_range(storage, "2004-02-01")
    assert all(not p.startswith("2020-01-20") for p in mt.GRACE_PERIODS), "the inverted interval is left out"


# ── the catalogue ───────────────────────────────────────────────────────────


def test_dated_layers_lists_every_layer_with_a_range_and_a_licence():
    out = mt.dated_layers()
    ids = [lay["id"] for lay in out["layers"]]
    assert ids == ["daily", "precip", "soil", "snow", "lst", "storage"]
    assert out["steps"] == ["day", "week", "month"] and out["max_frames"] == 60
    for lay in out["layers"]:
        assert lay["since"] and lay["licence"] and lay["attribution"]
        assert "{date}" in lay["tile_template"] and lay["tile_template"].startswith(mt.GIBS)
    assert next(lay for lay in out["layers"] if lay["id"] == "storage")["until"] == "2022-07-01"
    assert "live" not in out
    json.dumps(out)


def test_capabilities_are_read_per_layer_and_gaps_found():
    found = mt.parse_capabilities(CAPS, ["IMERG_Precipitation_Rate",
                                         "GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI",
                                         "NoTime_Layer", "Missing_Layer"])
    assert set(found) == {"IMERG_Precipitation_Rate", "GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI"}
    imerg = found["IMERG_Precipitation_Rate"]
    assert imerg["default"] == "2026-10-06"
    assert mt.interval_gaps(imerg["intervals"]) == [["2025-10-23", "2025-10-30"]]
    grace = found["GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI"]
    # 4 April + 28 days covers to 1 May; 2 May + 17 days to 18 May; then nothing until August.
    assert mt.interval_gaps(grace["intervals"]) == [["2002-05-19", "2002-08-01"]]


def test_live_mode_adds_intervals_gaps_and_the_latest_day(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_text(self, url):
            assert url == mt.CAPABILITIES_URL
            return CAPS

        def close(self):
            pass

    monkeypatch.setattr("aquascope.utils.http_client.CachedHTTPClient", FakeClient)
    out = mt.dated_layers(live=True)
    assert out["live"] is True
    precip = next(lay for lay in out["layers"] if lay["id"] == "precip")
    assert precip["latest"] == "2026-10-06" and precip["gaps"] == [["2025-10-23", "2025-10-30"]]
    assert "intervals" not in next(lay for lay in out["layers"] if lay["id"] == "soil")


def test_live_mode_keeps_the_recorded_ranges_when_gibs_is_unreachable(monkeypatch):
    def boom():
        raise RuntimeError("GIBS down")

    monkeypatch.setattr(mt, "_live_intervals", boom)
    out = mt.dated_layers(live=True)
    assert "GIBS down" in out["live_error"] and "live" not in out
    assert len(out["layers"]) == 6


# ── time-lapse frames ───────────────────────────────────────────────────────


def test_frames_skip_dates_a_layer_cannot_show():
    out = mt.layer_frames("soil", "2015-03-29", "2015-04-02")
    assert [f["date"] for f in out["frames"]] == ["2015-03-31", "2015-04-01", "2015-04-02"]
    assert out["skipped"] == 2 and out["truncated"] is False
    smap = f"{mt.GIBS}/SMAP_L4_Analyzed_Root_Zone_Soil_Moisture/default/2015-03-31/"
    assert out["frames"][0]["tiles"].startswith(smap)
    assert out["licence"].startswith("Open (NASA)")


def test_monthly_frames_are_one_per_month_and_stop_at_the_last_month():
    out = mt.layer_frames("storage", "2022-05-01", "2022-09-30", step="week")
    assert [f["layer_date"] for f in out["frames"]] == ["2022-05-01", "2022-06-01", "2022-07-01"]
    empty = mt.layer_frames("storage", "2024-01-01", "2024-06-01", step="month")
    assert empty["frames"] == [] and "no data" in empty["note"]


def test_monthly_frames_skip_the_gaps_inside_the_range():
    out = mt.layer_frames("storage", "2002-04-15", "2002-09-15", step="month")
    assert [f["layer_date"] for f in out["frames"]] == ["2002-04-04", "2002-05-02", "2002-08-01", "2002-09-01"]
    assert out["skipped"] == 2   # 15 June and 15 July 2002 have no image


def test_a_range_from_long_before_the_layer_still_finds_its_frames():
    out = mt.layer_frames("precip", "1900-01-01", "2000-06-03")
    assert [f["date"] for f in out["frames"]] == ["2000-06-01", "2000-06-02", "2000-06-03"]
    assert out["skipped"] == (date(2000, 6, 1) - date(1900, 1, 1)).days
    weekly = mt.layer_frames("soil", "2015-03-01", "2015-04-20", step="week")
    assert [f["date"] for f in weekly["frames"]] == ["2015-04-05", "2015-04-12", "2015-04-19"]
    assert weekly["skipped"] == 5
    monthly = mt.layer_frames("soil", "2014-01-31", "2015-05-31", step="month")
    assert [f["date"] for f in monthly["frames"]] == ["2015-03-31", "2015-04-30", "2015-05-31"]
    assert monthly["skipped"] == 14


def test_frames_are_capped_and_bad_input_is_an_error_not_a_crash():
    out = mt.layer_frames("precip", "2001-01-01", "2010-01-01", max_frames=500)
    assert len(out["frames"]) == 60 and out["truncated"] is True
    assert "unknown layer" in mt.layer_frames("rain", "2020-01-01", "2020-02-01")["error"]
    assert "step" in mt.layer_frames("precip", "2020-01-01", "2020-02-01", step="hour")["error"]
    assert "YYYY-MM-DD" in mt.layer_frames("precip", "yesterday", "2020-02-01")["error"]
    assert "whole number" in mt.layer_frames("precip", "2020-01-01", "2020-02-01", max_frames="many")["error"]


# ── the faces ───────────────────────────────────────────────────────────────


def test_the_mcp_server_offers_both_tools():
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert {"dated_layers", "layer_frames"} <= names
    assert m.layer_frames("snow", "2020-01-01", "2020-01-03")["frames"][0]["date"] == "2020-01-01"
    assert m.dated_layers()["layers"][0]["id"] == "daily"


def test_cli_lists_the_layers(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "storage" in out and "2002-04-04 to 2022-07-01" in out and "Steps: day, week, month" in out


def test_cli_prints_frames_or_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "frames", "precip", "--start", "2020-01-01",
                                      "--end", "2020-01-15", "--step", "week"])
    cli.main()
    out = capsys.readouterr().out
    assert "3 frames" in out and "2020-01-08" in out and "NASA" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "frames", "precip", "--start", "2020-01-01",
                                      "--end", "2020-01-02", "--json"])
    cli.main()
    assert len(json.loads(capsys.readouterr().out)["frames"]) == 2


def test_cli_frames_error_exits_non_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "frames", "nope", "--start", "2020-01-01",
                                      "--end", "2020-01-02"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1 and "unknown layer" in capsys.readouterr().out


# ── the Explorer face agrees with the package ───────────────────────────────


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_explorer_carries_the_same_dated_layers_and_ranges():
    script = f"""
    const m = await import({json.dumps(LAYERS_JS.as_uri())});
    const dated = [...m.BASEMAPS, ...m.OVERLAYS].filter((l) => l.time)
      .map((l) => ({{ id: l.id, since: l.since || null, until: l.until || null, monthly: Boolean(l.monthly),
                      tile: l.tiles[0], periods: l.periods || null }}));
    console.log(JSON.stringify(dated));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    js = {row["id"]: row for row in json.loads(out.stdout)}
    py = {lay["id"]: lay for lay in mt.DATED_LAYERS}
    assert set(js) == set(py), "every dated layer on the map is in aquascope.map_time, and only those"
    for lid, lay in py.items():
        assert js[lid]["since"] == lay["since"], lid
        assert js[lid]["until"] == lay["until"], lid
        assert js[lid]["monthly"] == (lay["cadence"] == "month"), lid
        assert lay["gibs_layer"] in js[lid]["tile"] and lay["matrix"] in js[lid]["tile"], lid
        assert js[lid]["periods"] == (list(lay["periods"]) if lay.get("periods") else None), lid


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_explorer_picks_the_same_grace_image_for_every_day():
    script = f"""
    const m = await import({json.dumps(LAYERS_JS.as_uri())});
    const s = m.overlayById("storage");
    const out = {{}};
    const stop = new Date(Date.UTC(2023, 0, 1));
    for (let d = new Date(Date.UTC(2002, 0, 1)); d < stop; d.setUTCDate(d.getUTCDate() + 1)) {{
      const iso = d.toISOString().slice(0, 10);
      out[iso] = m.imageFor(s, iso);
    }}
    console.log(JSON.stringify(out));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    storage = mt._layer("storage")
    differ = [d for d, v in js.items() if v != (lambda img: img.isoformat() if img else None)(mt.image_for(storage, d))]
    assert len(js) > 7000 and differ == []
