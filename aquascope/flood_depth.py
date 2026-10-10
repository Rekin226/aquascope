"""Flood depth where floods are forecast (#554): the RAS Mapper view of a forecast, keyless.

When the daily Floods ahead issue (#546, :mod:`aquascope.archive.warnings`) says a river reach will pass its 10-, 25-,
50- or 100-year flow in the next 15 days, :func:`flood_depth_overlay` says which JRC flood depth map shows what that
could look like around the reach, where to read it, and (optionally) how deep it gets there.

The depth maps are the JRC CEMS-GloFAS global river flood hazard maps v2.1.2: modelled water depth (m) at 3 arc-seconds
(about 90 m) for floods of 10, 20, 50, 75, 100, 200 and 500 years, made with LISFLOOD (flows) and LISFLOOD-FP
(inundation). There are no 2- or 5-year maps. They are read from N. Lebovits's Cloud-Optimized GeoTIFF mirror on
Source Cooperative (``data.source.coop/nlebovits/jrc-glofas/depth-rp<RP>/<tile>/<tile>_RP<RP>_depth.tif``): 271 tiles
of 10 x 10 degrees per return period, 512-pixel internal tiles with overviews, deflate, float32, nodata -9999, CORS
``*`` with byte ranges (checked 2026-10-10 from the Explorer's origin). A file is 17 to 180 MB, so nothing reads a
whole one: the CLI and the MCP tool read the pixels around the reach, and the Explorer reads the same windows in a
worker (``explorer/src/flood-depth-core.js`` and ``flood-depth-worker.js``, which keep the same tile list, the same
radius rule and the same colour ramp as this module; a test keeps them equal).

Which map: the largest return period that does not exceed the forecast's class. A reach forecast to pass its 25-year
flow gets the 20-year map; 10, 50 and 100 get their own. Around the reach means within a radius that grows with the
river (:func:`reach_radius_km`), faded at the edge on the map.

What it is not: a flood simulation of this event. It is a precomputed hazard map chosen by a forecast, so it is a
model estimate twice over: the forecast (GEOGLOWS) and the hazard map (JRC, a different model) can each be wrong, and
the return periods of the two models are not the same floods. The label says so every time:
"may flood in the next 15 days, model estimate".

Licence, read on 2026-10-10 from JRC's own files: the dataset's ``copyright.txt`` says any copyright and sui generis
right "is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence" with credit given and
changes indicated; the README's licence line (A02) says "no restrictions, free and open Copernicus product". Both are
recorded in :data:`LICENCE_WORDING`; the mirror lists CC-BY-4.0 too.

Pure standard library apart from the HTTP client and the COG reader, so it imports on a bare install.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

DEPTH_BASE = "https://data.source.coop/nlebovits/jrc-glofas"
DEPTH_RETURN_PERIODS = (10, 20, 50, 75, 100, 200, 500)
#: The return periods the Floods ahead forecast classes (aquascope.archive.warnings.RETURN_PERIODS).
FORECAST_CLASSES = (2, 5, 10, 25, 50, 100)
FORECAST_DAYS = 15
LABEL = "may flood in the next 15 days, model estimate"
VERSION = "v2.1.2"
#: Degrees per pixel at full resolution (3 arc-seconds).
PIXEL_DEG = 1.0 / 1200.0
TILE_DEG = 10
NODATA = -9999.0
#: Depths shallower than this (m) are left dry on the map.
MIN_DEPTH_M = 0.05
CHECKED = "2026-10-10"

JRC_FTP = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS"
LICENCE = "CC BY 4.0"
LICENCE_WORDING = {
    "copyright": ("(c) European Union, 1995-2026. ... Any copyright and/or sui generis right on the dataset is "
                  "licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. Reuse is "
                  "allowed provided appropriate credit is given and any changes are indicated."),
    "copyright_url": f"{JRC_FTP}/copyright.txt",
    "readme": "no restrictions, free and open Copernicus product",
    "readme_url": f"{JRC_FTP}/flood_hazard/README.txt",
    "mirror": "the Source Cooperative mirror's README lists CC-BY-4.0",
    "checked": CHECKED,
}
ATTRIBUTION = ("Flood depth: JRC CEMS-GloFAS global river flood hazard maps v2.1.2, (c) European Union, CC BY 4.0 "
               "(Baugh et al. 2024); Cloud-Optimized GeoTIFF mirror by N. Lebovits on Source Cooperative")
CITATION = ("Baugh, C., Colonese, J., D'Angelo, C., Dottori, F., Neal, J., Prudhomme, C., Salamon, P. (2024): "
            "Modelled flood inundation for different return period scenarios at the global scale. European "
            "Commission, Joint Research Centre (JRC)")
HOMEPAGE = "https://data.jrc.ec.europa.eu/collection/id-0054"

METHOD = ("The JRC CEMS-GloFAS flood depth map whose return period is the largest not above the reach's forecast "
          "class (10 -> 10-year, 25 -> 20-year, 50 -> 50-year, 100 -> 100-year; there are no 2- or 5-year maps), "
          "read within {radius} of the reach's point in the GEOGLOWS tables. The class comes from the daily Floods "
          "ahead issue: the ensemble-mean daily flow against the reach's own GEOGLOWS return-period flows.")
NOT = ("A precomputed hazard map chosen by a forecast, not a flood simulation of this event: a model estimate twice "
       "over (the GEOGLOWS forecast and the JRC hazard map, two different models whose return periods are not the "
       "same floods). The maps cover rivers draining more than about 1,000 km2 and leave out permanent water; "
       "JRC warns that depths over 10 m on small channels can be artefacts. For warnings, follow your national "
       "hydrological or meteorological service.")

#: The colour ramp of the map and the legend: depth (m) -> colour and opacity. Light to deep blue, the JRC
#: classification's break points (1, 3 and 10 m) as stops.
RAMP: list[dict[str, Any]] = [
    {"depth_m": 0.05, "hex": "#bfe0f7", "alpha": 0.55},
    {"depth_m": 1.0, "hex": "#6aaee8", "alpha": 0.74},
    {"depth_m": 3.0, "hex": "#2f7ccc", "alpha": 0.86},
    {"depth_m": 10.0, "hex": "#173f99", "alpha": 0.93},
]

#: Every tile of the mirror, "ID<n>_<N|S><top lat>_<E|W><left lon>" (the same 271 for each return period; listed on
#: 2026-10-10 from the mirror and JRC's all_urls.txt).
DEPTH_TILES: tuple[str, ...] = (
    "ID1_N70_W180", "ID2_N80_W170", "ID3_N70_W170", "ID4_N60_W170", "ID5_N80_W160", "ID6_N70_W160",
    "ID7_N60_W160", "ID8_N80_W150", "ID9_N70_W150", "ID10_N60_W150", "ID11_N80_W140", "ID12_N70_W140",
    "ID13_N60_W140", "ID14_N80_W130", "ID15_N70_W130", "ID16_N60_W130", "ID17_N50_W130", "ID18_N40_W130",
    "ID19_N80_W120", "ID20_N70_W120", "ID21_N60_W120", "ID22_N50_W120", "ID23_N40_W120", "ID24_N30_W120",
    "ID25_N80_W110", "ID26_N70_W110", "ID27_N60_W110", "ID28_N50_W110", "ID29_N40_W110", "ID30_N30_W110",
    "ID31_N20_W110", "ID32_N90_W100", "ID33_N80_W100", "ID34_N70_W100", "ID35_N60_W100", "ID36_N50_W100",
    "ID37_N40_W100", "ID38_N30_W100", "ID39_N20_W100", "ID40_N90_W90", "ID41_N80_W90", "ID42_N70_W90",
    "ID43_N60_W90", "ID44_N50_W90", "ID45_N40_W90", "ID46_N30_W90", "ID47_N20_W90", "ID48_N10_W90",
    "ID49_N0_W90", "ID50_N90_W80", "ID51_N80_W80", "ID52_N70_W80", "ID53_N60_W80", "ID54_N50_W80",
    "ID55_N40_W80", "ID56_N30_W80", "ID57_N20_W80", "ID58_N10_W80", "ID59_N0_W80", "ID60_S10_W80",
    "ID61_S20_W80", "ID62_S30_W80", "ID63_S40_W80", "ID64_S50_W80", "ID65_N90_W70", "ID66_N80_W70",
    "ID67_N70_W70", "ID68_N60_W70", "ID69_N50_W70", "ID70_N20_W70", "ID71_N10_W70", "ID72_N0_W70",
    "ID73_S10_W70", "ID74_S20_W70", "ID75_S30_W70", "ID76_S40_W70", "ID77_S50_W70", "ID78_N80_W60",
    "ID79_N70_W60", "ID80_N60_W60", "ID81_N50_W60", "ID82_N10_W60", "ID83_N0_W60", "ID84_S10_W60",
    "ID85_S20_W60", "ID86_S30_W60", "ID87_N80_W50", "ID88_N70_W50", "ID89_N60_W50", "ID90_N10_W50",
    "ID91_N0_W50", "ID92_S10_W50", "ID93_S20_W50", "ID94_N80_W40", "ID95_N70_W40", "ID96_N0_W40",
    "ID97_S10_W40", "ID98_N80_W30", "ID99_N70_W30", "ID100_N80_W20", "ID101_N70_W20", "ID102_N60_W20",
    "ID103_N30_W20", "ID104_N20_W20", "ID105_N10_W20", "ID106_N60_W10", "ID107_N50_W10", "ID108_N40_W10",
    "ID109_N30_W10", "ID110_N20_W10", "ID111_N10_W10", "ID112_N70_W0", "ID113_N60_W0", "ID114_N50_W0",
    "ID115_N40_W0", "ID116_N30_W0", "ID117_N20_W0", "ID118_N10_W0", "ID119_N0_W0", "ID120_N70_E10",
    "ID121_N60_E10", "ID122_N50_E10", "ID123_N40_E10", "ID124_N30_E10", "ID125_N20_E10", "ID126_N10_E10",
    "ID127_N0_E10", "ID128_S10_E10", "ID129_S20_E10", "ID130_S30_E10", "ID131_N80_E20", "ID132_N70_E20",
    "ID133_N60_E20", "ID134_N50_E20", "ID135_N40_E20", "ID136_N30_E20", "ID137_N20_E20", "ID138_N10_E20",
    "ID139_N0_E20", "ID140_S10_E20", "ID141_S20_E20", "ID142_S30_E20", "ID143_N80_E30", "ID144_N70_E30",
    "ID145_N60_E30", "ID146_N50_E30", "ID147_N40_E30", "ID148_N30_E30", "ID149_N20_E30", "ID150_N10_E30",
    "ID151_N0_E30", "ID152_S10_E30", "ID153_S20_E30", "ID154_S30_E30", "ID155_N70_E40", "ID156_N60_E40",
    "ID157_N50_E40", "ID158_N40_E40", "ID159_N30_E40", "ID160_N20_E40", "ID161_N10_E40", "ID162_N0_E40",
    "ID163_S10_E40", "ID164_S20_E40", "ID165_N70_E50", "ID166_N60_E50", "ID167_N50_E50", "ID168_N40_E50",
    "ID169_N30_E50", "ID170_N20_E50", "ID171_N10_E50", "ID172_S10_E50", "ID173_N80_E60", "ID174_N70_E60",
    "ID175_N60_E60", "ID176_N50_E60", "ID177_N40_E60", "ID178_N30_E60", "ID179_N80_E70", "ID180_N70_E70",
    "ID181_N60_E70", "ID182_N50_E70", "ID183_N40_E70", "ID184_N30_E70", "ID185_N20_E70", "ID186_N10_E70",
    "ID187_N80_E80", "ID188_N70_E80", "ID189_N60_E80", "ID190_N50_E80", "ID191_N40_E80", "ID192_N30_E80",
    "ID193_N20_E80", "ID194_N10_E80", "ID195_N80_E90", "ID196_N70_E90", "ID197_N60_E90", "ID198_N50_E90",
    "ID199_N40_E90", "ID200_N30_E90", "ID201_N20_E90", "ID202_N10_E90", "ID203_N0_E90", "ID204_N80_E100",
    "ID205_N70_E100", "ID206_N60_E100", "ID207_N50_E100", "ID208_N40_E100", "ID209_N30_E100", "ID210_N20_E100",
    "ID211_N10_E100", "ID212_N0_E100", "ID213_N80_E110", "ID214_N70_E110", "ID215_N60_E110", "ID216_N50_E110",
    "ID217_N40_E110", "ID218_N30_E110", "ID219_N20_E110", "ID220_N10_E110", "ID221_N0_E110", "ID222_S10_E110",
    "ID223_S20_E110", "ID224_S30_E110", "ID225_N80_E120", "ID226_N70_E120", "ID227_N60_E120", "ID228_N50_E120",
    "ID229_N40_E120", "ID230_N30_E120", "ID231_N20_E120", "ID232_N10_E120", "ID233_N0_E120", "ID234_S10_E120",
    "ID235_S20_E120", "ID236_S30_E120", "ID237_N80_E130", "ID238_N70_E130", "ID239_N60_E130", "ID240_N50_E130",
    "ID241_N40_E130", "ID242_N0_E130", "ID243_S10_E130", "ID244_S20_E130", "ID245_S30_E130", "ID246_N80_E140",
    "ID247_N70_E140", "ID248_N60_E140", "ID249_N50_E140", "ID250_N40_E140", "ID251_N0_E140", "ID252_S10_E140",
    "ID253_S20_E140", "ID254_S30_E140", "ID255_S40_E140", "ID256_N80_E150", "ID257_N70_E150", "ID258_N60_E150",
    "ID259_N0_E150", "ID260_S10_E150", "ID261_S20_E150", "ID262_S30_E150", "ID263_N80_E160", "ID264_N70_E160",
    "ID265_N60_E160", "ID266_S40_E160", "ID267_N80_E170", "ID268_N70_E170", "ID269_N60_E170", "ID270_S30_E170",
    "ID271_S40_E170",
)


def _tile_key(name: str) -> tuple[int, int]:
    _, la, lo = name.split("_")
    top = int(la[1:]) * (1 if la[0] == "N" else -1)
    left = int(lo[1:]) * (1 if lo[0] == "E" else -1)
    return top, left


_TILE_INDEX: dict[tuple[int, int], str] = {_tile_key(n): n for n in DEPTH_TILES}


def tile_for(lat: float, lon: float) -> str | None:
    """The tile holding a point, or None over the sea and outside the maps (60 S to 80 N)."""
    top = int(math.ceil(lat / TILE_DEG) * TILE_DEG)
    left = int(math.floor(lon / TILE_DEG) * TILE_DEG)
    return _TILE_INDEX.get((top, left))


def tile_bounds(name: str) -> list[float]:
    """A tile's nominal bounds, [west, south, east, north] (the files reach a few pixels past them)."""
    top, left = _tile_key(name)
    return [float(left), float(top - TILE_DEG), float(left + TILE_DEG), float(top)]


def depth_url(name: str, return_period: int) -> str:
    return f"{DEPTH_BASE}/depth-rp{int(return_period)}/{name}/{name}_RP{int(return_period)}_depth.tif"


def tiles_for_bbox(bbox: list[float] | tuple[float, ...]) -> list[str]:
    """The tiles a box touches, west to east then north to south."""
    west, south, east, north = (float(x) for x in bbox)
    out = []
    for top in range(int(math.ceil(north / TILE_DEG) * TILE_DEG), int(math.ceil(south / TILE_DEG) * TILE_DEG) - 1,
                     -TILE_DEG):
        for left in range(int(math.floor(west / TILE_DEG) * TILE_DEG), int(math.floor(east / TILE_DEG) * TILE_DEG) + 1,
                          TILE_DEG):
            name = _TILE_INDEX.get((top, left))
            if name and top - TILE_DEG < north and top > south and left < east and left + TILE_DEG > west:
                out.append(name)
    return out


def depth_return_period(forecast_class: Any) -> int | None:
    """The depth map for a forecast class: the largest return period not above it, or None below 10 years."""
    try:
        c = float(forecast_class or 0)
    except (TypeError, ValueError):
        return None
    fits = [rp for rp in DEPTH_RETURN_PERIODS if rp <= c]
    return fits[-1] if fits else None


def reach_radius_km(order: Any) -> float:
    """How far around a reach the depth is drawn: 2 km per Strahler order above 3 (order 5: 4 km), 3 to 12 km."""
    try:
        o = int(order)
    except (TypeError, ValueError):
        o = 5
    return float(min(12, max(3, 2 * (o - 3))))


def disk_bbox(lon: float, lat: float, radius_km: float) -> list[float]:
    """[west, south, east, north] of a circle of ``radius_km`` around a point."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.05, math.cos(math.radians(lat))))
    return [round(lon - dlon, 5), round(max(-90.0, lat - dlat), 5),
            round(lon + dlon, 5), round(min(90.0, lat + dlat), 5)]


def ramp_color(depth_m: float) -> tuple[int, int, int, float] | None:
    """The map's colour for a depth: (r, g, b, alpha), or None for dry (the same as flood-depth-core.js)."""
    if depth_m is None or not depth_m >= MIN_DEPTH_M:
        return None
    stops = [(s["depth_m"], tuple(int(s["hex"][i:i + 2], 16) for i in (1, 3, 5)), s["alpha"]) for s in RAMP]
    if depth_m >= stops[-1][0]:
        d, rgb, a = stops[-1]
        return (*rgb, a)
    for (d0, c0, a0), (d1, c1, a1) in zip(stops, stops[1:]):
        if depth_m <= d1:
            t = max(0.0, (depth_m - d0) / (d1 - d0))
            rgb = tuple(round(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
            return (*rgb, round(a0 + (a1 - a0) * t, 3))
    return None  # pragma: no cover - the loop always returns


# ── the forecast ────────────────────────────────────────────────────────────


def _load_warnings(warnings: dict[str, Any] | None, local: str | None, repo_id: str) -> tuple[dict, list]:
    """(manifest, features) of the Floods ahead issue: handed in, from a local run folder, or from the Archive."""
    if warnings is not None:
        return dict(warnings.get("manifest") or {}), list(warnings.get("features") or [])
    import json
    from pathlib import Path

    from aquascope.archive.warnings import FOLDER

    if local:
        root = Path(local) / FOLDER if (Path(local) / FOLDER).exists() else Path(local)
        man = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        geo = json.loads((root / "latest.geojson").read_text()) if (root / "latest.geojson").exists() else {}
    else:
        from aquascope.archive.forecasts import read_published_json

        man = read_published_json(f"{FOLDER}/manifest.json", repo_id)
        geo = read_published_json(f"{FOLDER}/latest.geojson", repo_id) if man else {}
    return man, list(geo.get("features") or [])


def _day_index(issue_date: str | None, day: str | None) -> int:
    """Forecast day 0 to 14 for an ISO date, -1 for the 15-day peak; ValueError outside the forecast."""
    if not day:
        return -1
    if not issue_date:
        raise ValueError("the Floods ahead issue has no issue date")
    i = (date.fromisoformat(str(day)[:10]) - date.fromisoformat(issue_date)).days
    if not 0 <= i < FORECAST_DAYS:
        end = date.fromisoformat(issue_date) + timedelta(days=FORECAST_DAYS - 1)
        raise ValueError(f"{day} is outside the forecast, which runs from {issue_date} to {end.isoformat()}")
    return i


def class_on(props: dict[str, Any], i: int) -> int:
    """A reach's forecast class on day ``i`` (its ``daily`` string), or its 15-day peak class for -1."""
    if i < 0:
        return int(props.get("rp") or 0)
    from aquascope.archive.warnings import DAILY_CODES

    s = str(props.get("daily") or "")
    try:
        return DAILY_CODES[int(s[i])] if i < len(s) else 0
    except (ValueError, IndexError):
        return 0


def _reach_out(f: dict[str, Any], i: int) -> dict[str, Any]:
    p = f.get("properties") or {}
    lon, lat = (f.get("geometry") or {}).get("coordinates") or (None, None)
    cls = class_on(p, i)
    return {"river_id": int(p["river_id"]), "lat": lat, "lon": lon, "strahler_order": p.get("order"),
            "forecast_class": cls, "peak_class": int(p.get("rp") or 0), "peak_cms": p.get("peak"),
            "q2_cms": p.get("q2"), "peak_date": p.get("day"), "share": p.get("share"), "daily": p.get("daily"),
            "depth_return_period": depth_return_period(cls)}


# ── reading the depth ───────────────────────────────────────────────────────


def sample_disk(lon: float, lat: float, radius_km: float, return_period: int, *, step: int = 2) -> dict[str, Any]:
    """The depth at the point and the deepest and wet share within ``radius_km``, every ``step``-th pixel of the
    full-resolution map (byte-range reads of the few 512-pixel blocks involved)."""
    from aquascope.utils.cog import COGNotFound, open_cog

    west, south, east, north = disk_bbox(lon, lat, radius_km)
    at_point: float | None = None
    deepest: float | None = None
    deepest_at: list[float] | None = None
    wet = total = 0
    coslat = math.cos(math.radians(lat))
    for name in tiles_for_bbox([west, south, east, north]):
        try:
            cog = open_cog(depth_url(name, return_period))
        except COGNotFound:
            continue
        tw, ts, te, tn = tile_bounds(name)
        x0, dx, _, y0, _, dy = cog.images[0].transform
        c0 = max(0, int((max(west, tw) - x0) / dx))
        c1 = min(cog.images[0].width - 1, int((min(east, te) - x0) / dx))
        r0 = max(0, int((y0 - min(north, tn)) / -dy))
        r1 = min(cog.images[0].height - 1, int((y0 - max(south, ts)) / -dy))
        pixels = []
        for r in range(r0, r1 + 1, step):
            py = y0 + (r + 0.5) * dy
            for c in range(c0, c1 + 1, step):
                px = x0 + (c + 0.5) * dx
                if math.hypot((px - lon) * coslat, py - lat) * 111.32 <= radius_km:
                    pixels.append((c, r, px, py))
        cog.prefetch([(c, r) for c, r, _, _ in pixels])
        for c, r, px, py in pixels:
            v = cog.read_pixel(c, r)
            total += 1
            if v is None or not float(v) >= MIN_DEPTH_M:
                continue
            wet += 1
            if deepest is None or float(v) > deepest:
                deepest, deepest_at = float(v), [round(px, 5), round(py, 5)]
        if tile_for(lat, lon) == name:
            v = cog.value_at(lon, lat)
            at_point = round(float(v), 2) if v is not None and float(v) >= MIN_DEPTH_M else None
    return {"at_point_m": at_point, "max_m": None if deepest is None else round(deepest, 2), "max_at": deepest_at,
            "wet_share": round(wet / total, 3) if total else None, "pixels_read": total,
            "pixel_m": round(90 * step), "radius_km": radius_km}


def _tiles_out(bbox: list[float], rp: int) -> list[dict[str, Any]]:
    w, s, e, n = bbox
    out = []
    for name in tiles_for_bbox(bbox):
        tw, ts, te, tn = tile_bounds(name)
        win = [round(max(w, tw), 5), round(max(s, ts), 5), round(min(e, te), 5), round(min(n, tn), 5)]
        url = depth_url(name, rp)
        out.append({"name": name, "url": url, "bounds": [tw, ts, te, tn], "window": win,
                    "gdal": f"gdal_translate -projwin {win[0]} {win[3]} {win[2]} {win[1]} /vsicurl/{url} "
                            f"depth_rp{rp}_{name}.tif"})
    return out


def _base(rp: int | None) -> dict[str, Any]:
    return {
        "layer": "flood_depth", "label": "Flood depth", "wording": LABEL, "return_period": rp,
        "available_return_periods": list(DEPTH_RETURN_PERIODS), "units": "m", "nodata": NODATA,
        "pixel": "3 arc-seconds (about 90 m)", "legend": [dict(s) for s in RAMP], "version": VERSION,
        "attribution": ATTRIBUTION, "licence": LICENCE, "licence_wording": dict(LICENCE_WORDING),
        "citation": CITATION, "homepage": HOMEPAGE, "not": NOT,
    }


def _check_rp(return_period: Any) -> int | None:
    if return_period in (None, ""):
        return None
    rp = int(return_period)
    if rp not in DEPTH_RETURN_PERIODS:
        raise ValueError(f"no {rp}-year depth map: JRC publishes {', '.join(map(str, DEPTH_RETURN_PERIODS))} years "
                         "(no 2- or 5-year maps)")
    return rp


def _check_bbox(bbox: Any) -> list[float]:
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox is west, south, east, north in degrees")
    w, s, e, n = (float(x) for x in bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError("bbox is west, south, east, north in degrees, west < east and south < north")
    if e - w > 2 or n - s > 2:
        raise ValueError("keep the box within 2 x 2 degrees: the maps are 90 m pixels")
    return [w, s, e, n]


def flood_depth_overlay(river_id: int | str | None = None, *, bbox: list[float] | None = None,
                        return_period: int | None = None, day: str | None = None, radius_km: float | None = None,
                        sample: bool = True, warnings: dict[str, Any] | None = None, local: str | None = None,
                        repo_id: str = "Rekin226/aquascope-gauges") -> dict[str, Any]:
    """Flood depth where a flood is forecast: the JRC depth map around a Floods ahead reach, or over a box.

    With ``river_id`` (a GEOGLOWS reach in today's Floods ahead issue) the map is the one that matches the reach's
    forecast class (its 15-day peak, or its class on ``day``, an ISO date inside the forecast), clipped to a circle
    around the reach (:func:`reach_radius_km`, or ``radius_km``). ``return_period`` picks another map (10, 20, 50,
    75, 100, 200 or 500). With ``bbox`` ([west, south, east, north], up to 2 x 2 degrees) the map is the
    ``return_period`` one (default 100) and the answer lists the forecast reaches inside the box that have a depth
    map. ``sample`` reads the depth at the reach, the deepest pixel and the wet share within the circle.

    Returns the tiles (URL, window, a GDAL command for the window), the extent and its clip, the forecast behind the
    choice, the legend ramp, the method, what it is not, and the licence (CC BY 4.0, with JRC's own wording).
    """
    try:
        rp_asked = _check_rp(return_period)
        if (river_id is None) == (bbox is None):
            raise ValueError("give a river_id from Floods ahead, or a bbox")
        box = _check_bbox(bbox) if bbox is not None else None
    except (TypeError, ValueError) as exc:
        return {"layer": "flood_depth", "error": str(exc)}

    issue: dict[str, Any] = {}
    features: list[dict[str, Any]] = []
    warn_error = None
    try:
        manifest, features = _load_warnings(warnings, local, repo_id)
        issue = {k: manifest.get(k) for k in ("issue_date", "valid_to", "run", "made", "smoke") if k in manifest}
    except Exception as exc:  # noqa: BLE001 - the box mode works without a forecast
        warn_error = f"{type(exc).__name__}: {exc}"
    try:
        i = _day_index(issue.get("issue_date"), day) if (day and features) else -1
    except ValueError as exc:
        return {"layer": "flood_depth", "error": str(exc), "forecast": issue}

    if box is not None:
        rp = rp_asked or 100
        out = _base(rp)
        reaches = []
        for f in features:
            lon, lat = (f.get("geometry") or {}).get("coordinates") or (None, None)
            if lat is None or not (box[1] <= lat <= box[3] and box[0] <= lon <= box[2]):
                continue
            r = _reach_out(f, i)
            if r["depth_return_period"]:
                reaches.append(r)
        reaches.sort(key=lambda r: -r["forecast_class"])
        out.update(extent={"bbox": box, "clip": None}, tiles=_tiles_out(box, rp), forecast=issue or None,
                   forecast_reaches=reaches[:50], forecast_reaches_n=len(reaches), day=day,
                   method=METHOD.format(radius="a radius that grows with the river"))
        if warn_error:
            out["forecast_error"] = warn_error
        out["available"] = bool(out["tiles"])
        out["summary"] = (f"The {rp}-year JRC depth map over this box: {len(out['tiles'])} tile"
                          f"{'' if len(out['tiles']) == 1 else 's'}"
                          + (f"; {len(reaches)} reach{'' if len(reaches) == 1 else 'es'} here {LABEL}."
                             if reaches else ".")) if out["tiles"] else "No depth map covers this box (sea, or "\
            "outside 60 S to 80 N)."
        return out

    try:
        rid = int(str(river_id).strip())
    except ValueError:
        return {"layer": "flood_depth", "error": f"not a GEOGLOWS river_id: {river_id!r}"}
    if not features:
        return {"layer": "flood_depth", "error": "no Floods ahead issue could be read"
                + (f" ({warn_error})" if warn_error else "") + "; give a bbox and a return_period instead"}
    hit = next((f for f in features if int((f.get("properties") or {}).get("river_id", -1)) == rid), None)
    if hit is None:
        return {"layer": "flood_depth", "forecast": issue,
                "error": f"reach {rid} is not in the Floods ahead issue of {issue.get('issue_date')} (it lists the "
                         "reaches expected to reach their 2-year flow); give a bbox and a return_period instead"}
    reach = _reach_out(hit, i)
    rp = rp_asked or reach["depth_return_period"]
    out = _base(rp)
    radius = float(radius_km) if radius_km else reach_radius_km(reach["strahler_order"])
    extent = disk_bbox(reach["lon"], reach["lat"], radius)
    when = f"on {day}" if i >= 0 else f"(peak on {reach['peak_date']})"
    out.update(reach=reach, forecast=issue, day=day, method=METHOD.format(radius=f"{radius:g} km"),
               extent={"bbox": extent, "clip": {"type": "disk", "center": [reach["lon"], reach["lat"]],
                                                "radius_km": radius}})
    cls = reach["forecast_class"]
    lead = (f"River reach {rid} is forecast to reach its {cls}-year flow {when}" if cls >= 2 else
            f"River reach {rid} is forecast to stay below its 2-year flow on {day}")
    if rp is None:
        out.update(available=False, tiles=[],
                   summary=f"{lead}: below the 10-year flow, the smallest JRC depth map, so there is no depth map "
                           "to show (JRC has no 2- or 5-year maps).")
        return out
    out["tiles"] = _tiles_out(extent, rp)
    out["available"] = bool(out["tiles"])
    nearest = rp_asked is None and rp != cls
    chosen = f"the {rp}-year depth map" + (" (the nearest map at or below it)" if nearest else "")
    summary = f"{lead}; {chosen} around it"
    if sample and out["tiles"]:
        try:
            out["depth"] = sample_disk(reach["lon"], reach["lat"], radius, rp)
        except Exception as exc:  # noqa: BLE001 - the tiles and the extent still stand
            out["depth_error"] = f"{type(exc).__name__}: {exc}"
    d = out.get("depth") or {}
    if d.get("max_m") is not None:
        share = d.get("wet_share") or 0
        summary += f", within {radius:g} km: up to {d['max_m']:.1f} m deep, {share * 100:.0f} % of the area wet"
    elif d:
        summary += f": dry within {radius:g} km in this map"
    out["summary"] = summary + f". {LABEL[0].upper()}{LABEL[1:]}."
    return out
