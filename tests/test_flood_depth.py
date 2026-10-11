"""Flood depth where floods are forecast (#554): the map chosen, the tiles, the clip and the three faces.

One engine, three faces: aquascope.flood_depth.flood_depth_overlay, the MCP tool of the same name,
`aquascope layers depth`, and the Explorer's flood-depth-core.js, which keeps the same tile list, radius rule and
colour ramp (checked here with node). No network: the depth files are built in memory and the stream tiles are
replaced by a fixture.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aquascope import cli
from aquascope import flood_depth as fd
from aquascope.utils import cog
from tests.cog_builder import build_tiff

ROOT = Path(__file__).resolve().parents[1]
DEPTH_JS = ROOT / "explorer" / "src" / "flood-depth-core.js"

# A reach on a tributary of the Lena (the 2026-10-09 issue had it at its 100-year flow), and its tile.
RID, LON, LAT = 340470321, 121.206, 60.6537
TILE = "ID226_N70_E120"
MANIFEST = {"issue_date": "2026-10-09", "valid_to": "2026-10-23", "run": "2026100900"}


def _reach(rid=RID, lon=LON, lat=LAT, rp=100, daily="466666665432110", order=6):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"river_id": rid, "rp": rp, "peak": 1400.0, "q2": 247.0, "day": "2026-10-12",
                           "share": 0.9, "order": order, "daily": daily, "gauges": ""}}


ISSUE = {"manifest": MANIFEST, "features": [_reach(), _reach(rid=340000001, lon=121.5, lat=60.9, rp=5, daily="2" * 15),
                                            _reach(rid=340000002, lon=121.3, lat=60.8, rp=25, daily="4" * 15)]}


class _Resp:
    def __init__(self, status: int, content: bytes):
        self.status_code = status
        self.content = content


class _Web:
    """The depth files by URL, with HTTP ranges, for the COG reader."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def get(self, url: str, headers: dict | None = None, timeout: float | None = None) -> _Resp:
        data = self.files.get(url)
        if data is None:
            return _Resp(404, b"")
        start, end = (int(x) for x in headers["Range"].split("=", 1)[1].split("-"))
        return _Resp(206, data[start:end + 1])


@pytest.fixture
def web(monkeypatch):
    fake = _Web()
    monkeypatch.setattr(cog, "_httpx_client", lambda: fake)
    cog._OPEN.clear()
    # No stream tiles: the clip is the plain circle unless a test hands lines in.
    monkeypatch.setattr(fd, "river_lines", lambda *a, **k: None)
    yield fake
    cog._OPEN.clear()


def _depth_file(web, rp: int, wet: dict[tuple[float, float], float]) -> None:
    """A small depth file near the reach (0.01 degree pixels), dry (-9999) apart from patches at the given points."""
    x0, y0, d = 121.0, 60.8, 0.01
    arr = np.full((40, 80), -9999.0, dtype="float32")
    for (lon, lat), v in wet.items():
        r, c = int((y0 - lat) / d), int((lon - x0) / d)
        arr[r - 1:r + 2, c - 1:c + 2] = v   # a 3 x 3 patch, so sampling every second pixel still finds it
        arr[r, c] = v
    web.files[fd.depth_url(TILE, rp)] = build_tiff(arr, tile=(16, 16), compression=8, predictor=3,
                                                   geotransform=(x0, d, y0, -d), nodata=-9999)


# ── the rules ───────────────────────────────────────────────────────────────


def test_the_map_is_the_largest_return_period_not_above_the_forecast():
    assert [fd.depth_return_period(c) for c in (2, 5, 10, 25, 50, 100)] == [None, None, 10, 20, 50, 100]
    assert fd.depth_return_period(None) is None and fd.depth_return_period("x") is None
    assert fd.depth_return_period(1000) == 500


def test_tiles_are_found_by_their_top_left_corner():
    assert len(fd.DEPTH_TILES) == 271 and len(set(fd.DEPTH_TILES)) == 271
    assert fd.tile_for(LAT, LON) == TILE
    assert fd.tile_for(53.5, 113.5) == "ID215_N60_E110"
    assert fd.tile_for(0.0, -150.0) is None, "open sea"
    assert fd.tile_bounds(TILE) == [120.0, 60.0, 130.0, 70.0]
    assert fd.depth_url(TILE, 20).endswith("/depth-rp20/ID226_N70_E120/ID226_N70_E120_RP20_depth.tif")
    assert fd.tiles_for_bbox([119.5, 59.5, 120.5, 60.5]) == [n for n in (
        fd.tile_for(60.2, 119.9), fd.tile_for(60.2, 120.1), fd.tile_for(59.9, 119.9), fd.tile_for(59.9, 120.1)) if n]


def test_the_radius_grows_with_the_river_and_the_ramp_runs_light_to_deep():
    assert [fd.reach_radius_km(o) for o in (3, 5, 6, 7, 12, None)] == [2.5, 3.0, 4.5, 6.0, 10.0, 3.0]
    assert fd.ramp_color(0.01) is None and fd.ramp_color(None) is None
    assert fd.ramp_color(1.0) == (106, 174, 232, 0.74)
    assert fd.ramp_color(50) == (23, 63, 153, 0.93)
    assert fd.ramp_color(2.0) == (77, 149, 218, 0.8)


def test_the_clip_keeps_the_reach_and_leaves_a_larger_river_out():
    own = [(LON - 0.02, LAT, LON + 0.02, LAT)]
    big = [(LON - 0.1, LAT - 0.03, LON + 0.1, LAT - 0.03)]   # a larger river about 3.3 km south
    box, inside = fd.clip_area(LON, LAT, 4.5, (own, big))
    assert box[0] < LON - 0.02 and box[2] > LON + 0.02
    assert inside(LON, LAT + 0.01)                 # up the valley from the reach
    assert not inside(LON, LAT - 0.025)            # nearer the larger river
    assert not inside(LON, LAT + 0.05)             # past the radius
    _, plain = fd.clip_area(LON, LAT, 4.5)
    assert plain(LON, LAT - 0.025), "without lines it is the circle"


# ── the overlay ─────────────────────────────────────────────────────────────


def test_a_forecast_reach_gets_its_map_tiles_and_depth(web):
    _depth_file(web, 100, {(LON, LAT): 2.4, (LON + 0.02, LAT + 0.01): 5.5, (LON + 0.5, LAT): 9.0})
    res = fd.flood_depth_overlay(RID, warnings=ISSUE)
    assert res["return_period"] == 100 and res["available"]
    assert res["reach"]["forecast_class"] == 100 and res["reach"]["strahler_order"] == 6
    assert res["wording"] == "may flood in the next 15 days, model estimate"
    assert [t["name"] for t in res["tiles"]] == [TILE]
    gdal = res["tiles"][0]["gdal"]
    assert gdal.startswith("gdal_translate -projwin ") and "/vsicurl/https://" in gdal
    assert res["extent"]["clip"]["radius_km"] == 4.5 and res["extent"]["clip"]["along"] == "point"
    d = res["depth"]
    assert d["at_point_m"] == pytest.approx(2.4) and d["max_m"] == pytest.approx(5.5), "9 m is outside the circle"
    assert "pass its 100-year flow" in res["summary"] and "up to 5.5 m deep" in res["summary"]
    assert res["summary"].endswith("May flood in the next 15 days, model estimate.")
    assert res["licence"] == "CC BY 4.0"
    assert "CC BY 4.0" in res["licence_wording"]["copyright"]
    assert res["licence_wording"]["readme"] == "no restrictions, free and open Copernicus product"


def test_a_day_of_the_forecast_steps_the_map(web):
    _depth_file(web, 20, {(LON, LAT): 0.8})
    day0 = fd.flood_depth_overlay(RID, day="2026-10-09", warnings=ISSUE)
    assert day0["reach"]["forecast_class"] == 25 and day0["return_period"] == 20
    assert "the nearest map at or below it" in day0["summary"]
    late = fd.flood_depth_overlay(RID, day="2026-10-20", warnings=ISSUE, sample=False)
    assert late["reach"]["forecast_class"] == 5 and late["available"] is False and late["tiles"] == []
    assert "no 2- or 5-year maps" in late["summary"]
    out = fd.flood_depth_overlay(RID, day="2026-11-30", warnings=ISSUE)
    assert "outside the forecast" in out["error"]


def test_another_return_period_and_the_errors(web):
    res = fd.flood_depth_overlay(RID, return_period=500, warnings=ISSUE, sample=False)
    assert res["return_period"] == 500 and res["tiles"][0]["url"].endswith("_RP500_depth.tif")
    assert "no 5-year depth map" in fd.flood_depth_overlay(RID, return_period=5, warnings=ISSUE)["error"]
    assert "river_id" in fd.flood_depth_overlay(warnings=ISSUE)["error"]
    assert "river_id" in fd.flood_depth_overlay(RID, bbox=[1, 2, 3, 4], warnings=ISSUE)["error"]
    assert "not in the Floods ahead issue" in fd.flood_depth_overlay(123456789, warnings=ISSUE)["error"]
    assert "2 x 2 degrees" in fd.flood_depth_overlay(bbox=[100, 50, 105, 51], warnings=ISSUE)["error"]
    empty = fd.flood_depth_overlay(RID, warnings={"manifest": {}, "features": []})
    assert "no Floods ahead issue" in empty["error"]


def test_a_box_lists_the_tiles_and_the_forecast_reaches_in_it(web):
    res = fd.flood_depth_overlay(bbox=[121.0, 60.5, 121.6, 61.0], warnings=ISSUE)
    assert res["return_period"] == 100 and [t["name"] for t in res["tiles"]] == [TILE]
    assert [r["river_id"] for r in res["forecast_reaches"]] == [RID, 340000002], "the 5-year reach has no map"
    assert res["forecast_reaches"][1]["depth_return_period"] == 20
    assert "2 reaches here may flood" in res["summary"]
    sea = fd.flood_depth_overlay(bbox=[-150.5, 0.0, -150.0, 0.5], return_period=10, warnings=ISSUE)
    assert sea["available"] is False and "No depth map" in sea["summary"]


def test_a_local_run_folder_is_read(tmp_path, web):
    folder = tmp_path / "forecasts" / "warnings"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(MANIFEST))
    (folder / "latest.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": ISSUE["features"]}))
    res = fd.flood_depth_overlay(RID, local=str(tmp_path), sample=False)
    assert res["forecast"]["issue_date"] == "2026-10-09" and res["return_period"] == 100


# ── the faces ───────────────────────────────────────────────────────────────


def test_the_mcp_server_offers_the_tool(monkeypatch):
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "flood_depth_overlay" in names
    seen = {}

    def fake(river_id=None, **kw):
        seen.update(kw, river_id=river_id)
        return {"ok": True}

    monkeypatch.setattr(fd, "flood_depth_overlay", fake)
    assert m.flood_depth_overlay(RID, day="2026-10-12") == {"ok": True}
    assert seen == {"river_id": RID, "bbox": None, "return_period": None, "day": "2026-10-12"}


def test_cli_prints_the_map_and_exits_non_zero_on_an_error(monkeypatch, capsys):
    monkeypatch.setattr(fd, "_load_warnings", lambda warnings, local, repo_id: (MANIFEST, ISSUE["features"]))
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "depth", str(RID), "--no-sample"])
    cli.main()
    out = capsys.readouterr().out
    assert "100-year depth map" in out and "_RP100_depth.tif" in out and "CC BY 4.0" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "depth", "--bbox", "121", "60.5", "121.6", "61",
                                      "--rp", "50", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["return_period"] == 50
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "depth", "1", "--no-sample"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_ask_the_map_can_turn_it_on_and_off():
    from aquascope import map_commands as mc

    on = mc.parse_command("show the flood depth")
    assert on["matched"] and {"type": "set_layer", "layer": "flood_depth", "on": True} in on["actions"]
    off = mc.parse_command("hide inundation")
    assert {"type": "set_layer", "layer": "flood_depth", "on": False} in off["actions"]
    assert mc.parse_command("show floods ahead")["actions"][0]["layer"] == "floods_ahead"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_explorer_keeps_the_same_tiles_radius_and_ramp():
    script = f"""
    const m = await import({json.dumps(DEPTH_JS.as_uri())});
    console.log(JSON.stringify({{ tiles: m.DEPTH_TILES, ramp: m.RAMP, base: m.DEPTH_BASE, rps: m.DEPTH_RETURN_PERIODS,
      label: m.DEPTH_LABEL, url: m.depthUrl("{TILE}", 20), tile: m.tileFor({LAT}, {LON}),
      radius: [3, 5, 6, 7, 12, null].map(m.reachRadiusKm), map: [2, 5, 10, 25, 50, 100].map(m.depthReturnPeriod),
      colors: [0.01, 0.5, 1, 2, 7, 50].map(m.rampColor) }}));
    """
    out = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                                    check=True).stdout)
    assert out["tiles"] == list(fd.DEPTH_TILES)
    assert out["ramp"] == fd.RAMP
    assert out["base"] == fd.DEPTH_BASE and out["rps"] == list(fd.DEPTH_RETURN_PERIODS) and out["label"] == fd.LABEL
    assert out["url"] == fd.depth_url(TILE, 20) and out["tile"] == TILE
    assert out["radius"] == [fd.reach_radius_km(o) for o in (3, 5, 6, 7, 12, None)]
    assert out["map"] == [fd.depth_return_period(c) for c in (2, 5, 10, 25, 50, 100)]
    for got, d in zip(out["colors"], (0.01, 0.5, 1, 2, 7, 50)):
        want = fd.ramp_color(d)
        assert (got is None and want is None) or (tuple(got[:3]) == want[:3] and got[3] == pytest.approx(want[3]))
