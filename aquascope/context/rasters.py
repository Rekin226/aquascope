"""The raster context layers, each read pixel by pixel from Cloud-Optimized GeoTIFFs (#520).

* surface water since 1984: JRC Global Surface Water v1.5 occurrence and occurrence change;
* flood depth at return periods: the JRC CEMS-GloFAS hazard maps v2.1.2 (community COG mirror);
* soil: SoilGrids 2.0 texture and the water held between 33 and 1500 kPa (1 km aggregates);
* actual evapotranspiration: FAO WaPOR v3 annual AETI at 300 m.

Every endpoint here answers range requests with CORS open to any origin (checked 2026-10-08 from the
Explorer's own origin), so the same functions run in the browser worker.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aquascope.context._common import (
    IS_EMSCRIPTEN,
    check_bbox,
    check_point,
    failed,
    grid_points,
    http_client,
    layer_result,
    memo,
)
from aquascope.context.projection import homolosine
from aquascope.utils.cog import COGNotFound, open_cog

logger = logging.getLogger(__name__)

__all__ = [
    "actual_et",
    "actual_et_area",
    "flood_hazard",
    "flood_hazard_area",
    "soil",
    "soil_area",
    "surface_water",
    "surface_water_area",
    "texture_class",
]


def _parallel(fns: list[Callable[[], Any]], workers: int = 8) -> list[Any]:
    """Run small reads concurrently in CPython; one after another in the browser (no threads there)."""
    if IS_EMSCRIPTEN or len(fns) < 2:
        return [f() for f in fns]
    with ThreadPoolExecutor(max_workers=min(workers, len(fns))) as pool:
        return list(pool.map(lambda f: f(), fns))


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ── surface water (JRC GSW v1.5, 1984-2024) ──────────────────────────────────

GSW_BASE = "https://storage.googleapis.com/water-world/download2024/VER1-5"
#: The map tiles of the same release, for the Explorer's overlay.
GSW_TILES = "https://storage.googleapis.com/water-world/tiles2024/occurrence/{z}/{x}/{y}.png"
GSW_PERIOD = "1984-2024"
GSW_EPOCHS = ("1984-1999", "2000-2024")


def gsw_url(layer: str, lat: float, lon: float) -> str:
    """The 10 x 10 degree GSW v1.5 file holding a point; files are named by their top-left corner."""
    top = int(math.ceil(lat / 10.0) * 10)
    left = int(math.floor(lon / 10.0) * 10)
    lat_tag = f"{top}N" if top >= 0 else f"{-top}S"
    lon_tag = f"{left}E" if left >= 0 else f"{-left}W"
    return f"{GSW_BASE}/{layer}/{layer}_{lon_tag}_{lat_tag}_v1_5_2024.tif"


def _gsw_value(layer: str, lat: float, lon: float) -> int | None:
    try:
        v = open_cog(gsw_url(layer, lat, lon)).value_at(lon, lat)
    except COGNotFound:
        return None  # the open ocean and the poles have no file
    return None if v is None else int(v)


def _neighbourhood_max(lat: float, lon: float, radius_px: int = 10, step: int = 2) -> int | None:
    """The highest occurrence within about 300 m (every second 30 m pixel), from the tile already read."""
    try:
        cog = open_cog(gsw_url("occurrence", lat, lon))
    except COGNotFound:
        return None
    ij = cog.index(lon, lat)
    if ij is None:
        return None
    best: int | None = None
    img = cog.image(0)
    pixels = [(ij[0] + dc, ij[1] + dr) for dr in range(-radius_px, radius_px + 1, step)
              for dc in range(-radius_px, radius_px + 1, step)]
    pixels = [(c, r) for c, r in pixels if 0 <= r < img.height and 0 <= c < img.width]
    cog.prefetch(pixels)  # GSW files are one LZW strip per row: one request for the rows instead of eleven
    for c, r in pixels:
        v = cog.read_pixel(c, r)
        if v is None or v > 100:
            continue
        best = v if best is None else max(best, v)
    return best


def _occurrence(raw: int | None) -> int | None:
    if raw is None:
        return 0  # an empty (sparse) block: GDAL writes those for all-land strips
    return raw if raw <= 100 else None


def _change(raw: int | None) -> tuple[int | None, str]:
    if raw is None:
        return None, "no data"
    if raw <= 200:
        return raw - 100, "computed"
    return None, {253: "not water", 254: "no homologous months to compare", 255: "no data"}.get(raw, "no data")


def _change_words(change: int | None) -> str:
    """The occurrence change intensity in words (it is a normalised difference, not percentage points)."""
    if change is None:
        return ""
    if change >= 100:
        return ", all of it since 2000 (never in 1984-1999)"
    if change <= -100:
        return ", none of it since 2000"
    if change == 0:
        return ", as often since 2000 as in 1984-1999"
    word = "more" if change > 0 else "less"
    return f", {word} often since 2000 than in 1984-1999 (normalised change {change:+d} %)"


def surface_water(lat: float, lon: float) -> dict[str, Any]:
    """How often a 30 m pixel was water from 1984 to 2024, and how that changed (JRC Global Surface Water v1.5).

    ``occurrence_pct`` is the share of valid monthly observations that saw water; ``change_norm_pct`` is
    GSW's occurrence change intensity between 1984-1999 and 2000-2024, the normalised difference of the two
    epochs' occurrence over homologous months (-100: water only before 2000, 0: no change, +100: water only
    since 2000; the downloadable GeoTIFF holds this ``change_norm`` band, not the absolute difference);
    ``nearby_max_occurrence_pct`` is the highest occurrence within about 300 m, so a click beside a river
    still finds it.
    """
    lat, lon = check_point(lat, lon)
    try:
        try:
            occ_raw = open_cog(gsw_url("occurrence", lat, lon)).value_at(lon, lat)
        except COGNotFound:
            return layer_result("surface_water", ["surface_water"], ok=True, lat=lat, lon=lon, occurrence_pct=None,
                                change_norm_pct=None, period=GSW_PERIOD,
                                summary="No Global Surface Water tile here (open sea or the poles).")
        change, change_status = _change(_gsw_value("change", lat, lon))
        nearby = _neighbourhood_max(lat, lon)
    except Exception as exc:  # noqa: BLE001 - surfaced as the layer's error
        return failed("surface_water", ["surface_water"], exc)
    occ = _occurrence(None if occ_raw is None else int(occ_raw))
    if occ is None:
        summary = "No valid Landsat observations for this pixel."
    elif occ == 0:
        summary = "Never mapped as water from 1984 to 2024 at this 30 m pixel"
        if nearby:
            summary += f"; water within about 300 m up to {nearby} % of the time"
        summary += "."
    else:
        summary = f"Water {occ} % of the time from 1984 to 2024" + _change_words(change) + "."
    return layer_result(
        "surface_water", ["surface_water"], ok=True, lat=lat, lon=lon, occurrence_pct=occ, change_norm_pct=change,
        change_status=change_status, nearby_max_occurrence_pct=nearby, period=GSW_PERIOD,
        change_epochs=list(GSW_EPOCHS), pixel_m=30, summary=summary,
    )


def surface_water_area(west: float, south: float, east: float, north: float, *, n: int = 3) -> dict[str, Any]:
    """Surface water over a box, sampled at an n x n grid of points: the mean occurrence and how many were ever wet."""
    bbox = check_bbox(west, south, east, north)
    try:
        occ = [_occurrence(_gsw_value("occurrence", la, lo)) for la, lo in grid_points(bbox, n)]
    except Exception as exc:  # noqa: BLE001
        return failed("surface_water", ["surface_water"], exc)
    valid = [v for v in occ if v is not None]
    wet = sum(1 for v in valid if v > 0)
    mean = _mean(list(valid))
    summary = (f"{wet} of {len(valid)} sample points were ever water from 1984 to 2024"
               + (f" (mean occurrence {mean:.0f} %)" if mean is not None and wet else "") + ".") if valid else \
        "No Global Surface Water data in this box."
    return layer_result("surface_water", ["surface_water"], ok=True, bbox=list(bbox), samples=len(occ),
                        sampled_occurrence_pct=occ, mean_occurrence_pct=mean, points_ever_wet=wet,
                        period=GSW_PERIOD, summary=summary)


# ── flood hazard (JRC CEMS-GloFAS v2.1.2 via Source Cooperative) ─────────────

HAZARD_BASE = "https://data.source.coop/nlebovits/jrc-glofas"
RETURN_PERIODS = (10, 20, 50, 75, 100, 200, 500)
_TILE_RE = re.compile(r"/RP10/(ID\d+)_([NS])(\d+)_([EW])(\d+)_RP10_depth\.tif")


def _hazard_index() -> dict[tuple[int, int], str]:
    """(top latitude, left longitude) of each 10-degree tile -> its "ID105_N10_W20" name, from the mirror's list."""
    def build() -> dict[tuple[int, int], str]:
        text = http_client().get_text(f"{HAZARD_BASE}/all_urls.txt")
        out: dict[tuple[int, int], str] = {}
        for m in _TILE_RE.finditer(text):
            tid, ns, la, ew, lo = m.groups()
            top = int(la) * (1 if ns == "N" else -1)
            left = int(lo) * (1 if ew == "E" else -1)
            out[(top, left)] = f"{tid}_{ns}{la}_{ew}{lo}"
        return out

    return memo("hazard_index", build)


def hazard_url(lat: float, lon: float, rp: int) -> str | None:
    top = int(math.ceil(lat / 10.0) * 10)
    left = int(math.floor(lon / 10.0) * 10)
    name = _hazard_index().get((top, left))
    if name is None:
        return None
    return f"{HAZARD_BASE}/depth-rp{rp}/{name}/{name}_RP{rp}_depth.tif"


def _depth(lat: float, lon: float, rp: int) -> float | None:
    url = hazard_url(lat, lon, rp)
    if url is None:
        return None
    try:
        v = open_cog(url).value_at(lon, lat)
    except COGNotFound:
        return None
    return None if v is None or v <= 0 else round(float(v), 2)


def flood_hazard(lat: float, lon: float, *,
                 return_periods: tuple[int, ...] | list[int] = RETURN_PERIODS) -> dict[str, Any]:
    """Modelled river flood depth (m) at a point for each return period (JRC CEMS-GloFAS hazard maps v2.1.2).

    The 500-year map is read first: where it is dry every smaller flood is dry too, so a dry point costs one
    file. A blank is "outside the mapped extent", not proof that a small stream cannot flood.
    """
    lat, lon = check_point(lat, lon)
    rps = sorted({int(r) for r in return_periods if int(r) in RETURN_PERIODS}) or list(RETURN_PERIODS)
    try:
        if hazard_url(lat, lon, rps[-1]) is None:
            return layer_result("flood_hazard", ["flood_hazard"], ok=True, lat=lat, lon=lon, depth_m={},
                                summary="No flood hazard map covers this point (open sea, or outside 60 S to 80 N).")
        deepest = _depth(lat, lon, rps[-1])
        depths: dict[str, float | None] = {str(rp): None for rp in rps}
        depths[str(rps[-1])] = deepest
        if deepest is not None:
            got = _parallel([lambda rp=rp: _depth(lat, lon, rp) for rp in rps[:-1]])
            for rp, d in zip(rps[:-1], got):
                depths[str(rp)] = d
    except Exception as exc:  # noqa: BLE001
        return failed("flood_hazard", ["flood_hazard"], exc)
    first = next((rp for rp in rps if depths[str(rp)] is not None), None)
    if first is None:
        summary = f"Outside the modelled flood extent up to the {rps[-1]}-year flood (large rivers only)."
    else:
        d100 = depths.get("100")
        summary = f"Floods in the model from the {first}-year flood"
        summary += f"; {d100:.1f} m deep at the 100-year flood." if d100 is not None else \
            f"; {depths[str(rps[-1])]:.1f} m deep at the {rps[-1]}-year flood."
    return layer_result("flood_hazard", ["flood_hazard"], ok=True, lat=lat, lon=lon, depth_m=depths,
                        first_flooded_return_period=first, pixel_m=90, summary=summary)


def flood_hazard_area(west: float, south: float, east: float, north: float, *, n: int = 3,
                      return_period: int = 100) -> dict[str, Any]:
    """Share of an n x n grid of sample points inside the modelled flood extent at one return period."""
    bbox = check_bbox(west, south, east, north)
    try:
        depths = [_depth(la, lo, return_period) for la, lo in grid_points(bbox, n)]
    except Exception as exc:  # noqa: BLE001
        return failed("flood_hazard", ["flood_hazard"], exc)
    wet = [d for d in depths if d is not None]
    summary = (f"{len(wet)} of {len(depths)} sample points flood in the {return_period}-year map"
               + (f" (up to {max(wet):.1f} m)" if wet else "") + ".")
    return layer_result("flood_hazard", ["flood_hazard"], ok=True, bbox=list(bbox), return_period=return_period,
                        sampled_depth_m=depths, points_flooded=len(wet), summary=summary)


# ── soil (SoilGrids 2.0, 1 km aggregates in Homolosine) ──────────────────────

SOILGRIDS_BASE = "https://files.isric.org/soilgrids/latest/data_aggregated/1000m"
#: (depth label, thickness in m) of the SoilGrids standard layers used here.
TEXTURE_DEPTHS = (("0-5cm", 0.05), ("5-15cm", 0.10), ("15-30cm", 0.15))
AWC_DEPTHS = (("0-5cm", 0.05), ("5-15cm", 0.10), ("15-30cm", 0.15), ("30-60cm", 0.30), ("60-100cm", 0.40))


def soilgrids_url(prop: str, depth: str) -> str:
    return f"{SOILGRIDS_BASE}/{prop}/{prop}_{depth}_mean_1000.tif"


def _soil_value(prop: str, depth: str, x: float, y: float) -> float | None:
    return open_cog(soilgrids_url(prop, depth)).value_at(x, y)


def texture_class(sand: float, silt: float, clay: float) -> str:
    """USDA soil texture class from sand, silt and clay in percent (the texture triangle)."""
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    return "clay"


def _soil_at(x: float, y: float) -> dict[str, Any]:
    jobs: list[tuple[str, str]] = [(p, d) for p in ("clay", "sand") for d, _ in TEXTURE_DEPTHS]
    jobs += [(p, d) for p in ("wv0033", "wv1500") for d, _ in AWC_DEPTHS]
    got = _parallel([lambda p=p, d=d: _soil_value(p, d, x, y) for p, d in jobs])
    v = dict(zip(jobs, got))
    out: dict[str, Any] = {}
    # texture: thickness-weighted over 0-30 cm, g/kg -> %; silt closes the sum (SoilGrids fractions are compositional)
    tw = [(v[("clay", d)], v[("sand", d)], t) for d, t in TEXTURE_DEPTHS]
    if all(c is not None and s is not None for c, s, _ in tw):
        total = sum(t for *_, t in tw)
        clay = sum(c * t for c, _, t in tw) / total / 10.0
        sand = sum(s * t for _, s, t in tw) / total / 10.0
        silt = max(0.0, 100.0 - clay - sand)
        out.update(clay_pct=round(clay, 1), sand_pct=round(sand, 1), silt_pct=round(silt, 1),
                   texture=texture_class(sand, silt, clay))
    # available water: (33 kPa - 1500 kPa) in mm per m, times the layer thickness
    layers = []
    awc = 0.0
    complete = True
    for d, t in AWC_DEPTHS:
        fc, wp = v[("wv0033", d)], v[("wv1500", d)]
        if fc is None or wp is None:
            complete = False
            continue
        layers.append({"depth": d, "field_capacity": round(fc / 1000.0, 3), "wilting_point": round(wp / 1000.0, 3)})
        awc += max(0.0, fc - wp) * t
    out["layers"] = layers
    out["available_water_mm"] = round(awc) if complete and layers else None
    return out


def soil(lat: float, lon: float) -> dict[str, Any]:
    """Soil texture of the top 30 cm and the plant-available water of the top metre (SoilGrids 2.0, 1 km).

    Available water is the volumetric water content at field capacity (33 kPa) minus wilting point
    (1500 kPa), summed over 0-100 cm, in mm. SoilGrids models soil on land outside water, ice and cities.
    """
    lat, lon = check_point(lat, lon)
    try:
        x, y = homolosine(lon, lat)
        res = _soil_at(x, y)
    except Exception as exc:  # noqa: BLE001
        return failed("soil", ["soil"], exc)
    if "texture" not in res and res.get("available_water_mm") is None:
        return layer_result("soil", ["soil"], ok=True, lat=lat, lon=lon, summary="No SoilGrids estimate here "
                            "(water, ice, a city or outside the mapped land).", **res)
    parts = []
    if "texture" in res:
        parts.append(f"{res['texture'].capitalize()} topsoil ({res['sand_pct']:.0f} % sand, {res['silt_pct']:.0f} % "
                     f"silt, {res['clay_pct']:.0f} % clay)")
    if res.get("available_water_mm") is not None:
        parts.append(f"holds about {res['available_water_mm']} mm of plant-available water in the top metre")
    return layer_result("soil", ["soil"], ok=True, lat=lat, lon=lon, resolution_m=1000, depth_texture="0-30 cm",
                        depth_water="0-100 cm", summary="; ".join(parts) + ".", **res)


def soil_area(west: float, south: float, east: float, north: float, *, n: int = 2) -> dict[str, Any]:
    """Mean available water and the commonest texture over an n x n grid of sample points."""
    bbox = check_bbox(west, south, east, north)
    try:
        samples = [_soil_at(*homolosine(lo, la)) for la, lo in grid_points(bbox, n)]
    except Exception as exc:  # noqa: BLE001
        return failed("soil", ["soil"], exc)
    awc = _mean([s.get("available_water_mm") for s in samples])
    textures = [s["texture"] for s in samples if s.get("texture")]
    common = max(set(textures), key=textures.count) if textures else None
    summary = (f"Mostly {common}" if common else "No texture estimate") + (
        f"; about {awc:.0f} mm of plant-available water in the top metre (mean of {len(samples)} points)."
        if awc is not None else ".")
    return layer_result("soil", ["soil"], ok=True, bbox=list(bbox), samples=len(samples), texture=common,
                        mean_available_water_mm=None if awc is None else round(awc), summary=summary)


# ── actual evapotranspiration (FAO WaPOR v3, annual, 300 m) ──────────────────

WAPOR_BUCKET = "fao-gismgr-wapor-3-data"
WAPOR_PREFIX = "DATA/WAPOR-3/MAPSET/L1-AETI-A/"
WAPOR_BASE = f"https://storage.googleapis.com/{WAPOR_BUCKET}/{WAPOR_PREFIX}"


def wapor_years() -> list[int]:
    """The years with an annual AETI file, from the bucket listing (cached a week)."""
    def build() -> list[int]:
        data = http_client().get_json(
            f"https://storage.googleapis.com/storage/v1/b/{WAPOR_BUCKET}/o",
            params={"prefix": WAPOR_PREFIX, "fields": "items(name)"},
        )
        years = []
        for item in (data or {}).get("items") or []:
            m = re.search(r"L1-AETI-A\.(\d{4})\.tif$", str(item.get("name", "")))
            if m:
                years.append(int(m.group(1)))
        return sorted(set(years))

    return memo("wapor_years", build)


def wapor_url(year: int) -> str:
    return f"{WAPOR_BASE}WAPOR-3.L1-AETI-A.{int(year)}.tif"


def _aeti(lat: float, lon: float, year: int) -> float | None:
    v = open_cog(wapor_url(year)).value_at(lon, lat)
    return None if v is None else round(float(v), 1)


def actual_et(lat: float, lon: float, *, years: int = 1) -> dict[str, Any]:
    """Annual actual evapotranspiration and interception (mm/yr) at a 300 m pixel, FAO WaPOR v3.

    ``years`` is how many of the latest years to read (each is one more file); the mean is over those read.
    """
    lat, lon = check_point(lat, lon)
    try:
        available = wapor_years()
        if not available:
            raise RuntimeError("the WaPOR listing returned no annual files")
        chosen = available[-max(1, min(int(years or 1), len(available))):]
        vals = _parallel([lambda y=y: _aeti(lat, lon, y) for y in chosen])
    except Exception as exc:  # noqa: BLE001
        return failed("actual_et", ["actual_et"], exc)
    series = {str(y): v for y, v in zip(chosen, vals)}
    got = [v for v in vals if v is not None]
    if not got:
        return layer_result("actual_et", ["actual_et"], ok=True, lat=lat, lon=lon, annual_mm=series,
                            years_available=available, summary="No WaPOR estimate here (water or no data).")
    latest_year, latest = max((y, v) for y, v in zip(chosen, vals) if v is not None)
    mean = sum(got) / len(got)
    summary = f"About {latest:,.0f} mm of water left as evapotranspiration in {latest_year}"
    if len(got) > 1:
        summary += f" (mean {mean:,.0f} mm a year over {len(got)} years)"
    return layer_result("actual_et", ["actual_et"], ok=True, lat=lat, lon=lon, annual_mm=series,
                        latest_year=latest_year, latest_mm=latest, mean_mm=round(mean, 1),
                        years_available=available, pixel_m=300, summary=summary + ".")


def actual_et_area(west: float, south: float, east: float, north: float, *, n: int = 3) -> dict[str, Any]:
    """Mean of the latest annual AETI over an n x n grid of sample points."""
    bbox = check_bbox(west, south, east, north)
    try:
        year = wapor_years()[-1]
        vals = [_aeti(la, lo, year) for la, lo in grid_points(bbox, n)]
    except Exception as exc:  # noqa: BLE001
        return failed("actual_et", ["actual_et"], exc)
    mean = _mean(list(vals))
    n_ok = len([v for v in vals if v is not None])
    summary = (f"About {mean:,.0f} mm of evapotranspiration in {year} (mean of {n_ok} sample points)."
               if mean is not None else "No WaPOR estimate in this box.")
    return layer_result("actual_et", ["actual_et"], ok=True, bbox=list(bbox), year=year, sampled_mm=vals,
                        mean_mm=None if mean is None else round(mean, 1), summary=summary)
