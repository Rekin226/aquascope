"""Change land use (#661): aquascope.sketch.landuse, its CLI verb, its MCP tool and the change_landuse map action.

The checks are TR-55's own worked examples (USDA SCS 1986, Urban Hydrology for Small Watersheds, 2nd ed.,
210-VI-TR-55): examples 2-1 to 2-4 (figures 2-5 to 2-8: CN 70, 75, 77, 74 and runoff 2.81, 3.28, 3.48, 3.19 in from
6.0 in of rain) and example 4-1 (worksheet 4: 345 cfs), and NEH 630 subpart F's lag-method example (Tc 1.14 h).
No test reads the network: the cover, the soil and the slope are given or faked.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from aquascope import map_commands as mc
from aquascope.sketch import landuse as lu

ROOT = Path(__file__).resolve().parents[1]
FIELD = [[4.955, 45.70], [4.97, 45.70], [4.97, 45.708], [4.955, 45.708]]


def test_check_passes():
    res = lu.check()
    assert res["ok"], res["rows"]
    assert len(res["rows"]) == 6


@pytest.mark.parametrize("cn,q_in", [(70, 2.81), (75, 3.28), (77, 3.48)])
def test_runoff_equation_matches_tr55_examples(cn, q_in):
    assert round(lu.runoff_depth(6.0 * 25.4, cn) / 25.4, 2) == q_in


def test_runoff_edges():
    assert lu.runoff_depth(5.0, 60) == 0.0                 # all of it held by Ia
    assert lu.runoff_depth(80.0, 100) == 80.0
    with pytest.raises(ValueError):
        lu.runoff_depth(10, 0)


def test_composite_cn_figures():
    assert lu.composite_cn(61, 35) == pytest.approx(73.95)          # TR-55 example 2-3, Memphis B: 74
    assert lu.composite_cn(74, 35) == pytest.approx(82.4)           # Loring C: 82
    assert lu.composite_cn(74, 25, 0.5) == pytest.approx(78.5)      # example 2-4, read 78 off figure 2-4
    with pytest.raises(ValueError):
        lu.composite_cn(74, 40, 0.5)                                 # figure 2-4 only under 30% impervious


def test_example_4_1_peak():
    run = lu.peak_discharge(250 / 247.105381, 6.0 * 25.4, 75, 1.53, distribution="II")
    assert run["ia_mm"] / 25.4 == pytest.approx(0.667, abs=1e-3)    # table 4-1
    assert run["ia_p"] == pytest.approx(0.11, abs=0.002)
    assert run["qu"] == pytest.approx(270, abs=2)                     # read off exhibit 4-II
    assert run["peak_cfs"] == pytest.approx(345, rel=0.01)
    assert run["peak_cms"] == pytest.approx(run["peak_cfs"] * 0.028316846592, rel=1e-9)


def test_unit_peak_limits_and_pond_factor():
    u = lu.unit_peak(20.0, 0.05, "II")
    assert u["held"] == ["Tc", "Ia/P"] and u["tc_h"] == 10.0 and u["ia_p"] == 0.10
    assert lu.pond_factor(0) == 1.0 and lu.pond_factor(0.3) == 0.97 and lu.pond_factor(12) == 0.72
    with pytest.raises(ValueError):
        lu.unit_peak(1, 0.2, "V")


def test_lag_method_example():
    assert round(lu.lag_tc(63, 4.79, length_m=3865 * 0.3048), 2) == 1.14
    # the flow length from the area (equation 630.15-5): l = 209 A^0.6, A in acres
    a_km2 = 100 / 247.105381
    assert lu.lag_tc(70, 2.0, area_km2=a_km2) == pytest.approx(lu.lag_tc(70, 2.0, length_m=209 * 100 ** 0.6 * 0.3048))


def test_change_offline():
    res = lu.landuse_change(FIELD, "residential_1_4", rain_mm=100, cover={40: 0.6, 10: 0.3, 80: 0.1},
                            soil_group="C", slope_pct=3.0, outlet=False)
    assert res["ok"] and res["soils"] == {"C": 1.0}
    # before: the land classes (cropland 85, woods 70 for C), water as pond and swamp
    assert res["before"]["cn"] == pytest.approx((0.6 * 85 + 0.3 * 70) / 0.9, abs=0.05)
    assert res["pond_pct"] == pytest.approx(10.0) and res["before"]["fp"] == 0.72
    assert res["after"]["cn"] == 83.0 and res["after"]["fp"] == 1.0
    assert res["change"]["peak_cms"] > 0 and res["route"]["peak_cms"] == res["change"]["peak_cms"]
    # the triangle Route it gets carries the added volume
    tri = 0.5 * res["route"]["peak_cms"] * res["route"]["duration_h"] * 3600
    assert tri == pytest.approx(res["change"]["volume_m3"], rel=0.01)
    assert res["area_km2"] == pytest.approx(1.04, abs=0.02)
    assert "TR-55" in res["method"] and res["licences"]["landcover"]["licence"] == "CC BY 4.0"
    json.dumps(res)


def test_impervious_use_and_bad_inputs():
    res = lu.landuse_change(FIELD, "impervious", impervious_pct=50, cover={30: 1.0}, soil_group="B", slope_pct=2,
                            outlet=False)
    assert res["after"]["cn"] == pytest.approx(lu.composite_cn(61, 50))
    for kw in ({"use": "castle"}, {"use": "impervious"}, {"use": "paved", "rain_mm": 0},
               {"use": "paved", "soil_group": "E"}, {"use": "paved", "distribution": "IV"}):
        use = kw.pop("use")
        with pytest.raises(ValueError):
            lu.landuse_change(FIELD, use, cover={30: 1.0}, soil_group=kw.pop("soil_group", "B"), slope_pct=2,
                              outlet=False, **kw)
    with pytest.raises(ValueError):
        lu.landuse_change(FIELD, "paved", cover={80: 1.0}, soil_group="B", slope_pct=2, outlet=False)
    with pytest.raises(ValueError):
        lu.landuse_change(FIELD[:2], "paved", cover={30: 1.0}, soil_group="B", slope_pct=2, outlet=False)


def _png(rgb: np.ndarray) -> bytes:
    """A plain 8-bit RGB PNG, every row unfiltered."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].astype("u1").tobytes() for r in range(h))

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_cover_from_the_tiles_colours():
    # every tile: the left half cropland's colour, the right half tree cover's
    tile = np.zeros((256, 256, 3), dtype="u1")
    tile[:, :128] = (0xF0, 0x96, 0xFF)
    tile[:, 128:] = (0x00, 0x64, 0x00)
    data = _png(tile)
    urls = []

    def fetch(url):
        urls.append(url)
        return data

    res = lu.cover_fractions(FIELD, fetch=fetch)
    assert set(res["fractions"]) == {10, 40} and sum(res["fractions"].values()) == pytest.approx(1.0)
    assert res["unclassified"] == 0 and all("esa-worldcover-map-10m-2021" in u for u in urls)


def test_soil_groups_from_the_cog(monkeypatch):
    class Img:
        transform = (-180.0, 1 / 480, 0.0, 84.0, 0.0, -1 / 480)

    class Fake:
        images = [Img()]

        def read_window(self, c0, r0, c1, r1):
            win = np.full((r1 - r0 + 1, c1 - c0 + 1), 3.0)
            win[:, : win.shape[1] // 2] = 4.0
            return win

    import aquascope.utils.cog as cog

    monkeypatch.setattr(cog, "open_cog", lambda url, fetch=None: Fake())
    res = lu.soil_groups(FIELD)
    assert set(res["fractions"]) == {"C", "D"} and sum(res["fractions"].values()) == pytest.approx(1.0)


def test_explorer_lists_match_the_package():
    core = (ROOT / "explorer" / "src" / "landuse-core.js").read_text(encoding="utf-8")
    for key in lu.USES:
        if key in lu.TR55:
            assert f'["{key}", "{lu.TR55[key]["label"]}"]' in core, key
        else:
            assert f'["{key}", ' in core
    for code, c in lu.WORLDCOVER.items():
        assert re.search(rf"\b{code}: \"{c['color']}\"", core), code
    assert f"rain_mm: {lu.DEFAULT_RAIN_MM:g}" in core


@pytest.mark.parametrize("text,want", [
    ("change this area to houses", {"type": "change_landuse", "area": "drawn", "use": "residential_1_4"}),
    ("turn this field into a car park with 80 mm of rain",
     {"type": "change_landuse", "area": "drawn", "use": "paved", "rain_mm": 80.0}),
    ("what if this area is built over", {"type": "change_landuse", "area": "drawn", "use": "residential_1_4"}),
    ("plant woods here", {"type": "change_landuse", "area": "drawn", "use": "woods_good"}),
    ("convert this area into a shopping mall", {"type": "change_landuse", "area": "drawn", "use": "commercial"}),
])
def test_change_landuse_phrases(text, want):
    res = mc.parse_command(text, "2026-10-27")
    assert res["actions"] == [want], res
    assert mc.describe_action(res["actions"][0]).startswith("Change the area drawn to ")


def test_change_landuse_validation():
    ok, err = mc.validate_action({"type": "change_landuse", "use": "industrial", "polygon": FIELD, "soil_group": "b"})
    assert err is None and ok["polygon"] == FIELD and ok["soil_group"] == "B"
    assert mc.validate_action({"type": "change_landuse", "use": "industrial"})[0]["area"] == "drawn"
    assert "impervious_pct" in mc.validate_action({"type": "change_landuse", "use": "impervious"})[1]
    assert "use must be" in mc.validate_action({"type": "change_landuse", "use": "moon base"})[1]
    schema = json.dumps(mc.action_schema())
    assert "change_landuse" in schema and "woods_good" in schema
    assert mc.LANDUSE_LIMITS["rain_mm"] == (1.0, 2000.0)
    assert set(mc.LANDUSE_WORDS) <= set(lu.USES)


def test_cli_and_mcp(capsys, monkeypatch):
    from aquascope.cli import main
    from aquascope.mcp_server import change_landuse

    calls = {}
    real = lu.landuse_change

    def fake(polygon, use, **kw):
        calls.update(kw, use=use)
        return real(polygon, use, rain_mm=kw.get("rain_mm", 100), cover={40: 1.0}, soil_group="C",
                                 slope_pct=2, outlet=False)

    monkeypatch.setattr(lu, "landuse_change", fake)
    out = change_landuse(FIELD, "paved", rain_mm=60)
    assert out["ok"] and calls["use"] == "paved" and calls["rain_mm"] == 60
    argv = sys.argv
    try:
        sys.argv = ["aquascope", "landuse", "--polygon", json.dumps(FIELD), "--use", "commercial", "--rain", "120"]
        main()
        sys.argv = ["aquascope", "landuse", "--check"]
        main()
        sys.argv = ["aquascope", "landuse", "--uses"]
        main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "as commercial and business" in text and "TR-55 example 4-1" in text and "woods_good" in text
