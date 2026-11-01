"""Draw a land-use change on the map and see the runoff and peak-flow change (#661). Experimental.

The SCS curve-number method as USDA's Technical Release 55 gives it (TR-55, *Urban Hydrology for Small Watersheds*,
second edition, June 1986, 210-VI-TR-55):

* **Runoff** (chapter 2, equations 2-1 to 2-4): ``Q = (P - Ia)**2 / (P - Ia + S)`` for ``P > Ia``, with
  ``S = 1000/CN - 10`` (inches) and ``Ia = 0.2 S``.
* **Curve numbers** (tables 2-2a to 2-2c) for each cover and hydrologic soil group, area-weighted; a share of
  impervious area by figure 2-3 (connected) and figure 2-4 (part unconnected), whose equations are in appendix F.
* **Peak discharge**, the graphical method (chapter 4): ``qp = qu Am Q Fp``, with the unit peak discharge ``qu``
  from the equation behind exhibit 4 (appendix F, table F-1) for the time of concentration and ``Ia/P``
  (interpolated linearly between the table's ``Ia/P`` rows, held at the limiting values outside them, as TR-55
  recommends) and the pond and swamp factor ``Fp`` of table 4-2.
* **Time of concentration**: the watershed lag method of USDA NRCS, National Engineering Handbook part 630,
  subpart F (amended June 2025), equation 630.15-4b, with the flow length from the drawn area (equation 630.15-5)
  and the average land slope read from the terrain.

What the drawn area holds now comes from data, read for the polygon:

* **Land cover**: ESA WorldCover 2021 (10 m, the Explorer's land-cover layer), each class given the TR-55 cover the
  table :data:`WORLDCOVER` names (built-up as impervious, CN 98, because at 10 m a built-up pixel is the roof or the
  pavement itself; TR-55's urban districts already average the lawns that WorldCover maps apart). Water, wetland and
  mangroves count as ponds and swamps (``Fp``); snow and ice, and moss and lichen, have no TR-55 cover and are left
  out of the curve number.
* **Hydrologic soil group**: HYSOGs250m (Ross et al., 2018), in the Natural Capital Alliance's cloud-optimised copy,
  where the dual groups (A/D, B/D, C/D, D/D: undrained) are group D. Or the group you pick.
* **Terrain** for the slope: the Terrarium tiles the Explorer uses.

Cover and soil are crossed as if independent inside the polygon (each cover class takes the polygon's mix of soil
groups). The drawn area is treated as its own small watershed: its peak before and after, and the change, which can be
routed down the river with Route it from the reach it drains to (:func:`aquascope.rivers.snap_to_river`).

Every number is computed here; :func:`check` reproduces TR-55's worked examples 2-1 to 2-4 and 4-1 and NEH 630's
lag-method example.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = [
    "DISTRIBUTIONS", "HYSOGS_URL", "LICENCES", "TR55", "USES", "WORLDCOVER", "WORLDCOVER_TILES",
    "check", "composite_cn", "cover_fractions", "landuse_change", "lag_tc", "land_slope", "peak_discharge",
    "pond_factor", "runoff_depth", "soil_groups", "unit_peak",
]

#: TR-55's curve numbers used here, (A, B, C, D), with the table they come from. Average runoff condition, Ia = 0.2S.
TR55: dict[str, dict[str, Any]] = {
    "residential_1_8": {"label": "Residential, 1/8 acre lots or town houses (65% impervious)", "table": "2-2a",
                        "cn": (77, 85, 90, 92), "impervious_pct": 65},
    "residential_1_4": {"label": "Residential, 1/4 acre lots (38% impervious)", "table": "2-2a",
                        "cn": (61, 75, 83, 87), "impervious_pct": 38},
    "residential_1_2": {"label": "Residential, 1/2 acre lots (25% impervious)", "table": "2-2a",
                        "cn": (54, 70, 80, 85), "impervious_pct": 25},
    "residential_1_acre": {"label": "Residential, 1 acre lots (20% impervious)", "table": "2-2a",
                           "cn": (51, 68, 79, 84), "impervious_pct": 20},
    "commercial": {"label": "Commercial and business (85% impervious)", "table": "2-2a",
                   "cn": (89, 92, 94, 95), "impervious_pct": 85},
    "industrial": {"label": "Industrial (72% impervious)", "table": "2-2a", "cn": (81, 88, 91, 93),
                   "impervious_pct": 72},
    "paved": {"label": "Paved parking lots, roofs, driveways", "table": "2-2a", "cn": (98, 98, 98, 98),
              "impervious_pct": 100},
    "open_space_good": {"label": "Open space, lawns and parks, good condition (grass cover > 75%)", "table": "2-2a",
                        "cn": (39, 61, 74, 80)},
    "newly_graded": {"label": "Newly graded areas (no vegetation)", "table": "2-2a", "cn": (77, 86, 91, 94)},
    "row_crops": {"label": "Row crops, straight row, good condition", "table": "2-2b", "cn": (67, 78, 85, 89)},
    "fallow_bare": {"label": "Fallow, bare soil", "table": "2-2b", "cn": (77, 86, 91, 94)},
    "pasture_good": {"label": "Pasture, grassland or range, good condition", "table": "2-2c",
                     "cn": (39, 61, 74, 80)},
    "meadow": {"label": "Meadow, continuous grass mowed for hay", "table": "2-2c", "cn": (30, 58, 71, 78)},
    "brush_fair": {"label": "Brush, brush-weed-grass mixture, fair condition", "table": "2-2c",
                   "cn": (35, 56, 70, 77)},
    "woods_good": {"label": "Woods, good condition", "table": "2-2c", "cn": (30, 55, 70, 77)},
}
#: The uses a reader can draw, in the order the Explorer lists them. "impervious" takes a percentage instead
#: (open space in good condition with that share of connected impervious area, TR-55 figure 2-3).
USES: tuple[str, ...] = ("residential_1_8", "residential_1_4", "residential_1_2", "residential_1_acre",
                         "commercial", "industrial", "paved", "newly_graded", "open_space_good", "row_crops",
                         "pasture_good", "meadow", "woods_good", "impervious")

#: ESA WorldCover 2021 v200 classes: code -> (label, colour in its map, TR-55 cover or the role it plays).
WORLDCOVER: dict[int, dict[str, Any]] = {
    10: {"label": "Tree cover", "color": "#006400", "tr55": "woods_good"},
    20: {"label": "Shrubland", "color": "#ffbb22", "tr55": "brush_fair"},
    30: {"label": "Grassland", "color": "#ffff4c", "tr55": "pasture_good"},
    40: {"label": "Cropland", "color": "#f096ff", "tr55": "row_crops"},
    50: {"label": "Built-up", "color": "#fa0000", "tr55": "paved"},
    60: {"label": "Bare / sparse vegetation", "color": "#b4b4b4", "tr55": "fallow_bare"},
    70: {"label": "Snow and ice", "color": "#f0f0f0", "tr55": None},
    80: {"label": "Permanent water bodies", "color": "#0064c8", "tr55": "pond"},
    90: {"label": "Herbaceous wetland", "color": "#0096a0", "tr55": "pond"},
    95: {"label": "Mangroves", "color": "#00cf75", "tr55": "pond"},
    100: {"label": "Moss and lichen", "color": "#fae6a0", "tr55": None},
}
#: The Explorer's land-cover layer, ESA WorldCover 2021 on Terrascope's WMTS (cross-origin reads allowed).
WORLDCOVER_TILES = ("https://wmts.terrascope.be/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
                    "&LAYER=esa-worldcover-map-10m-2021-v2_map&STYLE=default&TILEMATRIXSET=EPSG:3857"
                    "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/png&TIME=2021-01-01")
#: HYSOGs250m, reclassified to 1-4 (A-D) by the Natural Capital Alliance, a cloud-optimised GeoTIFF (EPSG:4326,
#: 1/480 degree) that answers cross-origin range reads.
HYSOGS_URL = ("https://data.naturalcapitalalliance.stanford.edu/download/global/HYSOGs250m/"
              "HYSOGs250m_Soil_Groups_reclassified.tif")
GROUPS = ("A", "B", "C", "D")

LICENCES: dict[str, dict[str, str]] = {
    "landcover": {"name": "ESA WorldCover 10 m 2021 v200", "licence": "CC BY 4.0",
                  "credit": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) "
                            "processed by ESA WorldCover consortium", "url": "https://esa-worldcover.org"},
    "soil": {"name": "HYSOGs250m (Ross et al. 2018, ORNL DAAC, doi:10.3334/ORNLDAAC/1566)",
             "licence": "ORNL DAAC: openly shared, without restriction (EOSDIS data use); the reclassified copy read "
                        "here: CC BY 4.0, Natural Capital Alliance, Stanford University",
             "credit": "Ross, C.W. et al. (2018), Global Hydrologic Soil Groups (HYSOGs250m) for Curve Number-Based "
                       "Runoff Modeling, ORNL DAAC", "url": "https://doi.org/10.3334/ORNLDAAC/1566"},
    "terrain": {"name": "Terrarium tiles (Mapzen/Tilezen joerd on AWS Open Data)",
                "licence": "open (see the joerd attribution list)",
                "credit": "SRTM, GMTED2010, ETOPO1, national lidar and others",
                "url": "https://github.com/tilezen/joerd/blob/master/docs/attribution.md"},
}

#: TR-55 appendix F, table F-1: the coefficients of log10(qu) = C0 + C1 log10(Tc) + C2 log10(Tc)**2 per rainfall
#: distribution and Ia/P.
TABLE_F1: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "I": ((0.10, 2.30550, -0.51429, -0.11750), (0.20, 2.23537, -0.50387, -0.08929),
          (0.25, 2.18219, -0.48488, -0.06589), (0.30, 2.10624, -0.45695, -0.02835),
          (0.35, 2.00303, -0.40769, 0.01983), (0.40, 1.87733, -0.32274, 0.05754),
          (0.45, 1.76312, -0.15644, 0.00453), (0.50, 1.67889, -0.06930, 0.0)),
    "IA": ((0.10, 2.03250, -0.31583, -0.13748), (0.20, 1.91978, -0.28215, -0.07020),
           (0.25, 1.83842, -0.25543, -0.02597), (0.30, 1.72657, -0.19826, 0.02633),
           (0.50, 1.63417, -0.09100, 0.0)),
    "II": ((0.10, 2.55323, -0.61512, -0.16403), (0.30, 2.46532, -0.62257, -0.11657),
           (0.35, 2.41896, -0.61594, -0.08820), (0.40, 2.36409, -0.59857, -0.05621),
           (0.45, 2.29238, -0.57005, -0.02281), (0.50, 2.20282, -0.51599, -0.01259)),
    "III": ((0.10, 2.47317, -0.51848, -0.17083), (0.30, 2.39628, -0.51202, -0.13245),
            (0.35, 2.35477, -0.49735, -0.11985), (0.40, 2.30726, -0.46541, -0.11094),
            (0.45, 2.24876, -0.41314, -0.11508), (0.50, 2.17772, -0.36803, -0.09525)),
}
DISTRIBUTIONS = tuple(TABLE_F1)
#: TR-55 table 4-2: the adjustment factor Fp for pond and swamp areas spread through the watershed (percent, Fp).
TABLE_4_2 = ((0.0, 1.00), (0.2, 0.97), (1.0, 0.87), (3.0, 0.75), (5.0, 0.72))

#: A default design storm: a 24-hour depth for the reader to replace with the local design depth. It is not a
#: statistic of any place.
DEFAULT_RAIN_MM = 100.0
CFS_TO_CMS = 0.028316846592
SQKM_PER_SQMI = 2.589988110336
MM_PER_IN = 25.4
EARTH_RADIUS_M = 6_371_008.8

METHOD = ("SCS curve number, TR-55 (USDA 1986): runoff Q = (P - 0.2S)²/(P + 0.8S), S = 1000/CN - 10 in; the "
          "drawn area's peak by TR-55's graphical method, qp = qu Am Q Fp, with Tc from NEH 630's lag equation "
          "(15-4b) on the terrain's slope.")
ASSUMPTIONS = ("The drawn area as its own small, homogeneous watershed, average runoff condition (Ia = 0.2S), one "
               "24-hour design storm with an SCS rainfall distribution, cover and soil crossed as if independent "
               "inside it. TR-55's limits apply: CN above 40, Tc 0.1 to 10 h, no snowmelt. A screening answer.")
CHECK_LINE = ("Checked against TR-55's worked examples 2-1 to 2-4 (runoff 2.81, 3.28, 3.48 and 3.19 in from 6 in of "
              "rain; the equation gives 2.81, 3.28, 3.48 and 3.18) and 4-1 (345 cfs; 345), and NEH 630's lag-method "
              "example (1.14 h; 1.14).")


# ── TR-55's equations ───────────────────────────────────────────────────────


def _cn(value: Any, name: str = "CN") -> float:
    cn = float(value)
    if not 1.0 <= cn <= 100.0:
        raise ValueError(f"{name} must be from 1 to 100, got {cn:g}")
    return cn


def runoff_depth(p_mm: float, cn: float) -> float:
    """TR-55 equations 2-3 and 2-4: the runoff depth (mm) from ``p_mm`` of rain on curve number ``cn``."""
    cn = _cn(cn)
    p = float(p_mm) / MM_PER_IN
    if p < 0:
        raise ValueError("rain must be 0 or more")
    if cn >= 100.0:
        return float(p_mm)
    s = 1000.0 / cn - 10.0
    ia = 0.2 * s
    return 0.0 if p <= ia else (p - ia) ** 2 / (p - ia + s) * MM_PER_IN


def composite_cn(cn_pervious: float, impervious_pct: float, unconnected_ratio: float = 0.0) -> float:
    """TR-55 figure 2-3 (connected impervious area) and figure 2-4 (part of it unconnected, total under 30%),
    appendix F: ``CNc = CNp + (Pimp/100)(98 - CNp)(1 - 0.5R)``."""
    cnp = _cn(cn_pervious, "the pervious CN")
    p = float(impervious_pct)
    r = float(unconnected_ratio)
    if not 0.0 <= p <= 100.0 or not 0.0 <= r <= 1.0:
        raise ValueError("impervious_pct is 0 to 100 and unconnected_ratio 0 to 1")
    if r and p > 30.0:
        raise ValueError("figure 2-4 (unconnected impervious area) holds for a total impervious area under 30%")
    return cnp + p / 100.0 * (98.0 - cnp) * (1.0 - 0.5 * r)


def unit_peak(tc_h: float, ia_p: float, distribution: str = "II") -> dict[str, Any]:
    """TR-55 exhibit 4 through its equation (appendix F, table F-1): the unit peak discharge ``qu`` (csm/in) for a
    time of concentration ``tc_h`` (held to 0.1 to 10 h) and ``ia_p`` (held to the table's rows; linear between
    them). Returns ``{qu, tc_h, ia_p, held}``, ``held`` naming what was held to a limit."""
    if distribution not in TABLE_F1:
        raise ValueError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}")
    rows = TABLE_F1[distribution]
    held = []
    tc = float(tc_h)
    if tc < 0.1 or tc > 10.0:
        held.append("Tc")
        tc = min(10.0, max(0.1, tc))
    r = float(ia_p)
    if r < rows[0][0] or r > rows[-1][0]:
        held.append("Ia/P")
        r = min(rows[-1][0], max(rows[0][0], r))
    lt = math.log10(tc)

    def q(row: tuple[float, float, float, float]) -> float:
        return 10 ** (row[1] + row[2] * lt + row[3] * lt * lt)

    for lo, hi in zip(rows, rows[1:]):
        if lo[0] <= r <= hi[0]:
            w = (r - lo[0]) / (hi[0] - lo[0])
            qu = q(lo) + w * (q(hi) - q(lo))
            break
    else:  # pragma: no cover - r is held inside the rows
        qu = q(rows[0])
    return {"qu": qu, "tc_h": tc, "ia_p": r, "held": held}


def pond_factor(pond_pct: float) -> float:
    """TR-55 table 4-2: ``Fp`` for the percentage of pond and swamp area, rounded to the nearest table value
    (5% and above take the last row)."""
    pct = max(0.0, float(pond_pct))
    return min(TABLE_4_2, key=lambda row: abs(row[0] - min(pct, 5.0)))[1]


def peak_discharge(area_km2: float, rain_mm: float, cn: float, tc_h: float, *, distribution: str = "II",
                   pond_pct: float = 0.0) -> dict[str, Any]:
    """TR-55's graphical method (chapter 4): ``qp = qu Am Q Fp`` for an area of ``area_km2``, ``rain_mm`` of 24-hour
    rain, curve number ``cn`` and ``tc_h``. Returns ``{peak_cms, peak_cfs, q_mm, ia_mm, ia_p, qu, fp, tc_h, held}``."""
    cn = _cn(cn)
    p_in = float(rain_mm) / MM_PER_IN
    if p_in <= 0:
        raise ValueError("rain must be above 0")
    s = 1000.0 / cn - 10.0
    ia = 0.2 * s
    q_mm = runoff_depth(rain_mm, cn)
    u = unit_peak(tc_h, ia / p_in, distribution)
    fp = pond_factor(pond_pct)
    am = float(area_km2) / SQKM_PER_SQMI
    cfs = u["qu"] * am * (q_mm / MM_PER_IN) * fp
    return {"peak_cms": cfs * CFS_TO_CMS, "peak_cfs": cfs, "q_mm": q_mm, "ia_mm": ia * MM_PER_IN,
            "ia_p": ia / p_in, "qu": u["qu"], "fp": fp, "tc_h": u["tc_h"], "held": u["held"]}


def lag_tc(cn: float, slope_pct: float, *, area_km2: float | None = None, length_m: float | None = None) -> float:
    """NEH 630 subpart F, equation 630.15-4b: ``Tc = l**0.8 (S + 1)**0.7 / (1140 Y**0.5)`` hours, with ``l`` the flow
    length in feet (``length_m``, or from the area by equation 630.15-5, ``l = 209 A**0.6``, A in acres), ``S`` from
    the curve number (held to 50 to 95, as the handbook asks) and ``Y`` the average land slope in percent."""
    y = float(slope_pct)
    if y <= 0:
        raise ValueError("the land slope must be above 0%")
    if length_m is not None:
        length_ft = float(length_m) / 0.3048
    elif area_km2 is not None:
        length_ft = 209.0 * (float(area_km2) * 247.105381) ** 0.6
    else:
        raise ValueError("give the flow length or the area")
    s = 1000.0 / min(95.0, max(50.0, _cn(cn))) - 10.0
    return length_ft ** 0.8 * (s + 1.0) ** 0.7 / (1140.0 * y ** 0.5)


# ── the polygon and the data under it ───────────────────────────────────────


def _ring(polygon: Any) -> np.ndarray:
    if isinstance(polygon, Mapping):
        geom = polygon.get("geometry", polygon)
        coords = geom.get("coordinates")
        if geom.get("type") == "Polygon":
            polygon = coords[0]
        elif geom.get("type") == "MultiPolygon":
            polygon = coords[0][0]
        else:
            raise ValueError("a land-use change needs a Polygon")
    pts = [] if polygon is None else polygon
    ring = np.asarray([[float(p[0]), float(p[1])] for p in pts], dtype="f8").reshape(-1, 2)
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError("draw a polygon of at least three [lon, lat] points")
    if np.any(np.abs(ring[:, 1]) > 85) or np.any(np.abs(ring[:, 0]) > 180):
        raise ValueError("the polygon must be on the map (latitudes within 85 degrees)")
    return ring


def _inside(ring: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd rule for points (x, y) against a ring of [x, y]."""
    out = np.zeros(np.shape(x), dtype=bool)
    n = len(ring)
    for k in range(n):
        x1, y1 = ring[k]
        x2, y2 = ring[(k + 1) % n]
        if y1 == y2:
            continue
        hit = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        out ^= hit
    return out


def polygon_area_km2(polygon: Any) -> float:
    """The polygon's area on the sphere (km²), by the spherical excess of its ring (good to a fraction of a percent
    for the areas drawn here)."""
    ring = _ring(polygon)
    lon, lat = np.radians(ring[:, 0]), np.radians(ring[:, 1])
    lon2, lat2 = np.roll(lon, -1), np.roll(lat, -1)
    a = np.sum((lon2 - lon) * (2 + np.sin(lat) + np.sin(lat2))) * EARTH_RADIUS_M ** 2 / 2.0
    return abs(float(a)) / 1e6


def _get(url: str) -> bytes:
    """GET a whole file, once more if the first try fails (a tile server's hiccup should not end the run)."""
    try:
        return _get_once(url)
    except OSError:
        return _get_once(url)


def _get_once(url: str) -> bytes:
    """GET a whole file: httpx in CPython, urllib (patched by pyodide-http) in the Explorer's worker."""
    if sys.platform == "emscripten":  # pragma: no cover - only under Pyodide
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - https data hosts
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - a JsException from the browser's XHR is not an OSError
            raise OSError(f"{url}: {exc}") from exc
    import httpx

    try:
        r = httpx.get(url, timeout=60, follow_redirects=True,
                      headers={"User-Agent": "aquascope (https://github.com/Rekin226/aquascope)"})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise OSError(f"{url}: {exc}") from exc
    return r.content


def _world_px(lon: np.ndarray, lat: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    s = 256 * 2 ** zoom
    lat_r = np.radians(lat)
    return (np.asarray(lon) + 180.0) / 360.0 * s, (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / math.pi) / 2 * s


def _mosaic(ring: np.ndarray, zoom: int, url: str, decode: Callable[[bytes], np.ndarray],
            fetch: Callable[[str], bytes]) -> tuple[np.ndarray, np.ndarray, int]:
    """The tiles under the ring at ``zoom``, decoded and set side by side, and the mask of their pixels whose
    centres are inside the ring. Returns (mosaic rows x cols x ..., mask, tiles read)."""
    px, py = _world_px(ring[:, 0], ring[:, 1], zoom)
    tx0, tx1 = int(px.min() // 256), int(px.max() // 256)
    ty0, ty1 = int(py.min() // 256), int(py.max() // 256)
    n = 2 ** zoom
    rows = []
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            row.append(decode(fetch(url.format(z=zoom, x=tx % n, y=ty))))
        rows.append(np.concatenate(row, axis=1))
    mosaic = np.concatenate(rows, axis=0)
    gy, gx = np.mgrid[0:mosaic.shape[0], 0:mosaic.shape[1]]
    mask = _inside(np.column_stack([px - tx0 * 256, py - ty0 * 256]), gx + 0.5, gy + 0.5)
    return mosaic, mask, (tx1 - tx0 + 1) * (ty1 - ty0 + 1)


def _zoom_for(ring: np.ndarray, most_px: int, top: int, bottom: int = 6) -> int:
    for z in range(top, bottom - 1, -1):
        px, py = _world_px(ring[:, 0], ring[:, 1], z)
        if max(np.ptp(px), np.ptp(py)) <= most_px:
            return z
    return bottom


def _rgb(data: bytes) -> np.ndarray:
    from aquascope.floodsim import decode_png_rgb

    return decode_png_rgb(data)


def cover_fractions(polygon: Any, *, fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """ESA WorldCover 2021 inside the polygon, read from the Explorer's land-cover tiles (their colours are the
    product's own legend): ``{fractions: {code: share}, zoom, pixels, tiles, unclassified}``."""
    ring = _ring(polygon)
    zoom = _zoom_for(ring, 640, 14)
    mosaic, mask, tiles = _mosaic(ring, zoom, WORLDCOVER_TILES, _rgb, fetch or _get)
    if not mask.any():   # smaller than a pixel: the pixel under its centre
        mask = np.zeros(mask.shape, dtype=bool)
        cx, cy = _world_px(np.array([ring[:, 0].mean()]), np.array([ring[:, 1].mean()]), zoom)
        tx0 = int(_world_px(ring[:, 0], ring[:, 1], zoom)[0].min() // 256)
        ty0 = int(_world_px(ring[:, 0], ring[:, 1], zoom)[1].min() // 256)
        mask[int(cy[0] - ty0 * 256), int(cx[0] - tx0 * 256)] = True
    rgb = mosaic[mask].astype("i4")
    key = rgb[:, 0] * 65536 + rgb[:, 1] * 256 + rgb[:, 2]
    counts: dict[int, int] = {}
    for code, c in WORLDCOVER.items():
        h = int(c["color"][1:], 16)
        n = int(np.count_nonzero(key == h))
        if n:
            counts[code] = n
    known = sum(counts.values())
    if not known:
        raise ValueError("no land cover under the polygon (outside WorldCover, or only open sea)")
    return {"fractions": {k: v / known for k, v in sorted(counts.items())}, "zoom": zoom, "pixels": int(mask.sum()),
            "tiles": tiles, "unclassified": int(mask.sum()) - known,
            "source": LICENCES["landcover"]["name"], "licence": LICENCES["landcover"]["licence"]}


def soil_groups(polygon: Any, *, fetch: Any = None) -> dict[str, Any]:
    """HYSOGs250m inside the polygon (pixel centres; the pixel under the centre when the polygon is smaller than
    one): ``{fractions: {"A": share, ...}, pixels}``. The dual groups are D in the copy read (undrained)."""
    from aquascope.utils.cog import open_cog

    ring = _ring(polygon)
    cog = open_cog(HYSOGS_URL, fetch=fetch)
    img = cog.images[0]
    x0, dx, _, y0, _, dy = img.transform
    c0 = int(math.floor((ring[:, 0].min() - x0) / dx))
    c1 = int(math.floor((ring[:, 0].max() - x0) / dx))
    r0 = int(math.floor((ring[:, 1].max() - y0) / dy))
    r1 = int(math.floor((ring[:, 1].min() - y0) / dy))
    win = np.asarray(cog.read_window(c0, r0, c1, r1), dtype="f8")
    if win.size == 0:
        raise ValueError("the polygon is outside HYSOGs250m (56 S to 84 N)")
    rr, cc = np.mgrid[0:win.shape[0], 0:win.shape[1]]
    lon = x0 + (c0 + cc + 0.5) * dx
    lat = y0 + (r0 + rr + 0.5) * dy
    mask = _inside(ring, lon, lat)
    if not mask.any():
        mx, my = ring[:, 0].mean(), ring[:, 1].mean()
        mask = np.zeros(win.shape, dtype=bool)
        r_c = min(win.shape[0] - 1, max(0, int((my - y0) / dy) - r0))
        c_c = min(win.shape[1] - 1, max(0, int((mx - x0) / dx) - c0))
        mask[r_c, c_c] = True
    vals = win[mask]
    vals = vals[np.isfinite(vals) & (vals >= 1) & (vals <= 4)]
    if not vals.size:
        raise ValueError("no soil group under the polygon (water or no data): pick one")
    frac = {g: float(np.count_nonzero(vals == k + 1)) / vals.size for k, g in enumerate(GROUPS)}
    return {"fractions": {g: f for g, f in frac.items() if f > 0}, "pixels": int(vals.size),
            "source": LICENCES["soil"]["name"], "licence": LICENCES["soil"]["licence"]}


def land_slope(polygon: Any, *, fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """The polygon's average land slope (%) on the Terrarium terrain (the mean gradient over its cells, at a zoom
    that puts about 64 cells across it, 13 at most), and its lowest point (where it drains, for the outlet)."""
    from aquascope.floodsim import terrarium_elevation

    ring = _ring(polygon)
    zoom = _zoom_for(ring, 64, 13, 4)
    z_rgb, mask, _ = _mosaic(ring, zoom, "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png",
                             _rgb, fetch or _get)
    z = terrarium_elevation(z_rgb)
    lat_mid = float(ring[:, 1].mean())
    cell = 2 * math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat_mid)) / (256 * 2 ** zoom)
    gy, gx = np.gradient(z, cell)
    slope = np.hypot(gx, gy) * 100.0
    if mask.sum() < 4:
        raise ValueError("the polygon is too small to read a slope: give one")
    low = np.argmin(np.where(mask, z, np.inf))
    r, c = np.unravel_index(low, z.shape)
    px, py = _world_px(ring[:, 0], ring[:, 1], zoom)
    wx, wy = int(px.min() // 256) * 256 + c + 0.5, int(py.min() // 256) * 256 + r + 0.5
    s = 256 * 2 ** zoom
    lon = wx / s * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * wy / s))))
    return {"slope_pct": float(np.mean(slope[mask])), "cell_m": cell, "zoom": zoom, "lowest": [float(lon), lat],
            "lowest_m": float(z[r, c])}


# ── the change ──────────────────────────────────────────────────────────────


def _use_cn(use: str, soils: Mapping[str, float], impervious_pct: float | None) -> tuple[float, str]:
    if use == "impervious":
        if impervious_pct is None:
            raise ValueError("use 'impervious' needs impervious_pct")
        cns = [composite_cn(TR55["open_space_good"]["cn"][GROUPS.index(g)], impervious_pct) for g in soils]
        cn = sum(f * c for f, c in zip(soils.values(), cns))
        return cn, (f"Open space in good condition with {float(impervious_pct):g}% connected impervious area "
                    "(figure 2-3)")
    if use not in TR55:
        raise ValueError(f"use must be one of {', '.join(USES)}")
    t = TR55[use]
    return sum(f * t["cn"][GROUPS.index(g)] for g, f in soils.items()), f"{t['label']} (table {t['table']})"


def landuse_change(polygon: Any, use: str, *, rain_mm: float = DEFAULT_RAIN_MM, distribution: str = "II",
                   soil_group: str | None = None, impervious_pct: float | None = None,
                   cover: Mapping[Any, float] | None = None, slope_pct: float | None = None,
                   tc_hours: float | None = None, outlet: bool = True,
                   fetch: Callable[[str], bytes] | None = None) -> dict[str, Any]:
    """The runoff and the peak flow of a drawn area before and after it becomes ``use`` (a key of :data:`USES`), for
    a 24-hour design storm of ``rain_mm`` with SCS distribution ``distribution`` (I, IA, II or III).

    The area's cover now is read from WorldCover (or given as ``cover``, {WorldCover code: share}), its soil groups
    from HYSOGs250m (or ``soil_group``, A to D, for the whole area), its slope from the terrain (or ``slope_pct``),
    and Tc from NEH 630's lag equation before and after (or ``tc_hours`` for both). ``outlet`` snaps the area's
    lowest point to the GEOGLOWS reach it drains to. Returns the curve numbers, the runoff depths and volumes, the
    peaks, the change, the reach, and a triangle for Route it (the added peak, over the time that carries the added
    volume), with every source and licence. Numbers from code."""
    ring = _ring(polygon)
    if distribution not in TABLE_F1:
        raise ValueError(f"distribution must be one of {', '.join(DISTRIBUTIONS)}")
    p = float(rain_mm)
    if not 1.0 <= p <= 2000.0:
        raise ValueError("rain_mm must be from 1 to 2000 mm")
    area = polygon_area_km2(ring)
    if area <= 0:
        raise ValueError("the polygon has no area")
    notes: list[str] = []
    sources: dict[str, Any] = {}

    if cover is None:
        cv = cover_fractions(ring, fetch=fetch)
        cover = cv["fractions"]
        sources["landcover"] = {k: cv[k] for k in ("zoom", "pixels", "tiles", "unclassified")}
    cover = {int(k): float(v) for k, v in cover.items() if float(v) > 0}
    if any(k not in WORLDCOVER for k in cover):
        raise ValueError(f"cover codes must be WorldCover's: {', '.join(map(str, WORLDCOVER))}")
    tot = sum(cover.values())
    cover = {k: v / tot for k, v in cover.items()}

    if soil_group:
        g = str(soil_group).upper()
        if g not in GROUPS:
            raise ValueError("soil_group is A, B, C or D")
        soils = {g: 1.0}
        sources["soil"] = {"picked": g}
    else:
        sg = soil_groups(ring)
        soils = sg["fractions"]
        sources["soil"] = {"pixels": sg["pixels"]}

    land = {k: v for k, v in cover.items() if WORLDCOVER[k]["tr55"] not in (None, "pond")}
    pond_pct = 100.0 * sum(v for k, v in cover.items() if WORLDCOVER[k]["tr55"] == "pond")
    left_out = [WORLDCOVER[k]["label"] for k in cover if WORLDCOVER[k]["tr55"] is None]
    if left_out:
        notes.append(f"{', '.join(left_out)}: no TR-55 cover, left out of the curve number.")
    if not land:
        raise ValueError("nothing under the polygon has a TR-55 cover (water, wetland, snow or moss only)")
    lw = sum(land.values())
    rows = []
    cn_before = 0.0
    for k, v in sorted(land.items(), key=lambda kv: -kv[1]):
        key = WORLDCOVER[k]["tr55"]
        cn_k = sum(f * TR55[key]["cn"][GROUPS.index(g)] for g, f in soils.items())
        cn_before += v / lw * cn_k
        rows.append({"code": k, "label": WORLDCOVER[k]["label"], "share": round(v, 4), "tr55": TR55[key]["label"],
                     "table": TR55[key]["table"], "cn": round(cn_k, 1)})
    for k, v in cover.items():
        if WORLDCOVER[k]["tr55"] in (None, "pond"):
            rows.append({"code": k, "label": WORLDCOVER[k]["label"], "share": round(v, 4),
                         "tr55": "pond and swamp (Fp)" if WORLDCOVER[k]["tr55"] == "pond" else "left out",
                         "table": "4-2" if WORLDCOVER[k]["tr55"] == "pond" else "", "cn": None})
    cn_after, use_label = _use_cn(use, soils, impervious_pct)

    tc_note = "given"
    lowest = None
    if tc_hours is None:
        if slope_pct is None:
            sl = land_slope(ring, fetch=fetch)
            slope_pct = max(sl["slope_pct"], 0.5)
            lowest = sl["lowest"]
            if sl["slope_pct"] < 0.5:
                notes.append("The terrain is flatter than 0.5%; 0.5% is used in the lag equation.")
        tc_b = lag_tc(cn_before, slope_pct, area_km2=area)
        tc_a = lag_tc(cn_after, slope_pct, area_km2=area)
        tc_note = f"NEH 630 lag equation, slope {slope_pct:.1f}%"
    else:
        tc_b = tc_a = float(tc_hours)
    before = peak_discharge(area, p, cn_before, tc_b, distribution=distribution, pond_pct=pond_pct)
    after = peak_discharge(area, p, cn_after, tc_a, distribution=distribution, pond_pct=0.0)
    for name, run in (("before", before), ("after", after)):
        if run["held"]:
            notes.append(f"{name.capitalize()}: {' and '.join(run['held'])} held to TR-55's limits.")
    if min(cn_before, cn_after) <= 40:
        notes.append("A curve number of 40 or less: TR-55 asks for another procedure there.")
    if area > 19.0 * SQKM_PER_SQMI:
        notes.append("Larger than the lag equation was fitted for (most watersheds under 2,000 acres, a reasonable "
                     "upper limit about 19 mi², NEH 630): read the peak as a rough guide.")

    vol_b = before["q_mm"] / 1000 * area * 1e6
    vol_a = after["q_mm"] / 1000 * area * 1e6
    added_peak = after["peak_cms"] - before["peak_cms"]
    added_vol = vol_a - vol_b
    route = None
    if added_peak > 0 and added_vol > 0:
        route = {"peak_cms": round(added_peak, 3), "duration_h": round(2.0 * added_vol / added_peak / 3600.0, 2),
                 "shape": "triangle"}

    reach = None
    if outlet:
        from aquascope.rivers import snap_to_river

        at = lowest or [float(ring[:, 0].mean()), float(ring[:, 1].mean())]
        try:
            sn = snap_to_river(at[1], at[0], max_distance_m=5000.0, prefer="nearest")
            if sn.get("snapped"):
                reach = {"river_id": sn["river_id"], "lat": sn["snap_lat"], "lon": sn["snap_lon"],
                         "distance_m": sn["distance_m"], "strahler_order": sn["strahler_order"], "from": at}
            else:
                notes.append(sn.get("message") or "No river reach found below the area.")
        except (ValueError, RuntimeError, OSError) as exc:
            notes.append(f"The reach below the area could not be read: {exc}")

    def run_out(run: dict[str, Any], cn: float, tc: float) -> dict[str, Any]:
        return {"cn": round(cn, 1), "q_mm": round(run["q_mm"], 1), "peak_cms": round(run["peak_cms"], 3),
                "tc_h": round(tc, 2), "ia_p": round(run["ia_p"], 3), "qu_csm_in": round(run["qu"], 1), "fp": run["fp"]}

    out = {
        "ok": True, "area_km2": round(area, 4), "use": use, "use_label": use_label, "rain_mm": p,
        "distribution": distribution, "soils": {g: round(f, 4) for g, f in soils.items()}, "cover": rows,
        "pond_pct": round(pond_pct, 2), "before": run_out(before, cn_before, tc_b),
        "after": run_out(after, cn_after, tc_a),
        "change": {"cn": round(cn_after - cn_before, 1), "q_mm": round(after["q_mm"] - before["q_mm"], 1),
                   "peak_cms": round(added_peak, 3), "volume_m3": round(added_vol, 0),
                   "peak_pct": round(100.0 * added_peak / before["peak_cms"], 1) if before["peak_cms"] > 0 else None},
        "tc_from": tc_note, "reach": reach, "route": route, "sources": sources, "licences": LICENCES,
        "method": METHOD, "assumptions": ASSUMPTIONS, "check": CHECK_LINE, "notes": notes,
    }
    out["summary"] = summary(out)
    return out


def summary(res: Mapping[str, Any]) -> str:
    """One plain sentence about a :func:`landuse_change` result."""
    b, a = res["before"], res["after"]
    pct = res["change"]["peak_pct"]
    more = f" ({pct:+.0f}%)" if pct is not None else ""
    return (f"{res['area_km2']:.2f} km² as {res['use_label'].split(' (')[0].lower()}: CN {b['cn']:.0f} to "
            f"{a['cn']:.0f}, runoff {b['q_mm']:.0f} to {a['q_mm']:.0f} mm from {res['rain_mm']:g} mm of rain, "
            f"the area's peak {b['peak_cms']:.2f} to {a['peak_cms']:.2f} m³/s{more}.")


# ── the check ───────────────────────────────────────────────────────────────

#: TR-55 chapter 2's examples (Heavenly Acres, Dyer County, Tennessee; 250 acres; 25-year 24-hour rain, 6.0 in):
#: the curve number and runoff each worksheet prints (figures 2-5 to 2-8).
#: The runoff is printed to 0.01 in, read from table 2-1 or figure 2-1 (CN 74 is not a row of table 2-1): the
#: equation is held to that. Each line: (the CN printed, acres, and for a figure-2-3 or 2-4 line the pervious CN,
#: the percent impervious and the unconnected ratio it was read with).
TR55_EXAMPLES = {
    "2-1": {"parts": [(61, 30, None), (74, 70, None)], "cn": 70, "q_in": 2.81},
    "2-2": {"parts": [(70, 75, None), (80, 100, None), (74, 75, None)], "cn": 75, "q_in": 3.28},
    "2-3": {"parts": [(74, 75, (61, 35, 0.0)), (82, 100, (74, 35, 0.0)), (74, 75, None)], "cn": 77, "q_in": 3.48},
    "2-4": {"parts": [(70, 75, None), (78, 100, (74, 25, 0.5)), (74, 75, None)], "cn": 74, "q_in": 3.19},
}
#: TR-55 example 4-1 (worksheet 4): the developed watershed of example 2-2, Am 0.39 mi² (250 acres), CN 75,
#: Tc 1.53 h, type II, 6.0 in; printed Ia 0.667 in, Ia/P 0.11, qu 270 csm/in, Q 3.28 in, Fp 1.0, qp 345 cfs.
TR55_EX_4_1 = {"acres": 250.0, "cn": 75, "tc_h": 1.53, "p_in": 6.0, "qp_cfs": 345.0, "qu": 270.0}
#: NEH 630 subpart F, 630.74 A (Mawney Brook, Kent County, Rhode Island): l 3,865 ft, CN 63, Y 4.79%, Tc 1.14 h.
NEH_LAG = {"length_ft": 3865.0, "cn": 63, "slope_pct": 4.79, "tc_h": 1.14}


def check() -> dict[str, Any]:
    """TR-55's worked examples 2-1 to 2-4 (curve number and runoff) and 4-1 (peak), and NEH 630's lag example,
    recomputed. Returns ``{ok, rows, sources}``."""
    rows = []
    for name, ex in TR55_EXAMPLES.items():
        total = prod = 0.0
        figs = []
        for printed, acres, fig in ex["parts"]:
            if fig:
                # TR-55 reads a whole CN off figure 2-3 or 2-4; its equation (appendix F) is within half a unit
                figs.append((printed, composite_cn(*fig)))
            total += acres
            prod += printed * acres
        cn_w = round(prod / total)
        q_in = runoff_depth(6.0 * MM_PER_IN, cn_w) / MM_PER_IN
        fig_words = "".join(f"; figure CN {c:.1f} (printed {pr})" for pr, c in figs)
        rows.append({"what": f"TR-55 example {name}: weighted CN and runoff from 6.0 in",
                     "printed": f"CN {ex['cn']}, Q {ex['q_in']:.2f} in", "computed": f"CN {cn_w}, Q {q_in:.2f} in"
                     + fig_words, "ok": cn_w == ex["cn"] and abs(q_in - ex["q_in"]) <= 0.0101
                     and all(abs(c - pr) <= 0.5 for pr, c in figs)})
    e = TR55_EX_4_1
    run = peak_discharge(e["acres"] / 247.105381, e["p_in"] * MM_PER_IN, e["cn"], e["tc_h"], distribution="II")
    rows.append({"what": "TR-55 example 4-1: graphical peak discharge", "printed": f"qu {e['qu']:.0f} csm/in, "
                 f"qp {e['qp_cfs']:.0f} cfs", "computed": f"qu {run['qu']:.0f} csm/in, qp {run['peak_cfs']:.0f} cfs",
                 "ok": abs(run["peak_cfs"] - e["qp_cfs"]) / e["qp_cfs"] < 0.01 and abs(run["qu"] - e["qu"]) <= 2})
    n = NEH_LAG
    tc = lag_tc(n["cn"], n["slope_pct"], length_m=n["length_ft"] * 0.3048)
    rows.append({"what": "NEH 630 subpart F, 630.74 A: lag-method Tc", "printed": f"{n['tc_h']:.2f} h",
                 "computed": f"{tc:.2f} h", "ok": round(tc, 2) == n["tc_h"]})
    return {"ok": all(r["ok"] for r in rows), "rows": rows, "sources": [
        "USDA Soil Conservation Service (1986). Urban Hydrology for Small Watersheds, Technical Release 55, "
        "2nd ed. (210-VI-TR-55): examples 2-1 to 2-4 (figures 2-5 to 2-8) and 4-1 (worksheet 4), appendix F.",
        "USDA NRCS (2025). National Engineering Handbook, part 630 Hydrology, subpart F, Time of Concentration "
        "(amended June 2025): equation 630.15-4b and example 630.74 A.",
    ]}
