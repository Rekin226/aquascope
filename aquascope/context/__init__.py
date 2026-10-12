"""Place context (#520): what a hydrologist asks first about a point or an area.

One function per layer, each ``(lat, lon) -> dict`` with an area variant ``(west, south, east, north) ->
dict``, and :func:`place_context` / :func:`area_context` to run several at once:

=================  ============================================  ===================================
layer              what it answers                               data (licence)
=================  ============================================  ===================================
``flood_history``  what flooded here before, with dates          Groundsource (CC BY 4.0), Microsoft
                                                                 Sentinel-1 floods 2014-2024 (MIT)
``surface_water``  how often this ground has been water, 1984+   JRC Global Surface Water v1.5
``flood_hazard``   modelled flood depth at return periods        JRC CEMS-GloFAS hazard maps v2.1.2
``dams``           dams and reservoirs nearby                    Global Dam Watch v1.0 (CC BY 4.0)
``soil``           texture and plant-available water             SoilGrids 2.0 (CC BY 4.0)
``actual_et``      how much water leaves as evapotranspiration   FAO WaPOR v3 (CC BY 4.0)
``rain_gauge``     the nearest real rain gauge and its record    NOAA GHCN-Daily (CC0)
=================  ============================================  ===================================

Every result carries a one-line ``summary`` and the licence and attribution of what it read. Everything
runs keyless in CPython and in the Explorer's Pyodide worker (HTTP range reads, no GDAL). The Global
Water Watch reservoir series is deliberately absent: its data licence is not confirmed.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aquascope.context._common import IS_EMSCRIPTEN, check_bbox, check_point
from aquascope.context.events import dams, dams_area, flood_history, flood_history_area, radar_recurrence
from aquascope.context.gauges import rain_gauge, rain_gauges_area
from aquascope.context.rasters import (
    actual_et,
    actual_et_area,
    flood_hazard,
    flood_hazard_area,
    soil,
    soil_area,
    surface_water,
    surface_water_area,
)

__all__ = [
    "AREA_LAYERS",
    "LAYERS",
    "LAYER_SOURCES",
    "actual_et",
    "area_context",
    "area_layer",
    "dams",
    "flood_hazard",
    "flood_history",
    "layer",
    "place_context",
    "radar_recurrence",
    "rain_gauge",
    "soil",
    "surface_water",
]

#: The order the layers are listed in, cheapest and most asked-for first.
LAYERS: dict[str, Callable[..., dict[str, Any]]] = {
    "flood_history": flood_history,
    "surface_water": surface_water,
    "flood_hazard": flood_hazard,
    "dams": dams,
    "rain_gauge": rain_gauge,
    "actual_et": actual_et,
    "soil": soil,
}

#: The registry entries (``aquascope.registry.CONTEXT_LAYERS``) behind each layer, for licences and credits.
LAYER_SOURCES: dict[str, tuple[str, ...]] = {
    "flood_history": ("groundsource", "microsoft_floods"),
    "surface_water": ("surface_water",),
    "flood_hazard": ("flood_hazard",),
    "dams": ("dams",),
    "rain_gauge": ("rain_gauge",),
    "actual_et": ("actual_et",),
    "soil": ("soil",),
}

AREA_LAYERS: dict[str, Callable[..., dict[str, Any]]] = {
    "flood_history": flood_history_area,
    "surface_water": surface_water_area,
    "flood_hazard": flood_hazard_area,
    "dams": dams_area,
    "rain_gauge": rain_gauges_area,
    "actual_et": actual_et_area,
    "soil": soil_area,
}


def _pick(layers: Any, table: dict[str, Any]) -> list[str]:
    if layers is None or layers == "all" or layers == []:
        return list(table)
    if isinstance(layers, str):
        layers = [p.strip() for p in layers.split(",") if p.strip()]
    unknown = [name for name in layers if name not in table]
    if unknown:
        raise ValueError(f"unknown context layer(s) {unknown}; choose from {list(table)}")
    return [name for name in table if name in layers]


def _run(calls: list[tuple[str, Callable[[], dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    if IS_EMSCRIPTEN or len(calls) < 2:
        return {name: fn() for name, fn in calls}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {name: pool.submit(fn) for name, fn in calls}
        return {name: fut.result() for name, fut in futures.items()}


def layer(name: str, lat: float, lon: float, **kwargs: Any) -> dict[str, Any]:
    """One layer by name at a point (the Explorer asks for them one at a time, so the card fills as they land)."""
    if name not in LAYERS:
        raise ValueError(f"unknown context layer {name!r}; choose from {list(LAYERS)}")
    return LAYERS[name](lat, lon, **kwargs)


def area_layer(name: str, west: float, south: float, east: float, north: float, **kwargs: Any) -> dict[str, Any]:
    """One layer by name over a box (the Explorer's area card asks for them one at a time too)."""
    if name not in AREA_LAYERS:
        raise ValueError(f"unknown context layer {name!r}; choose from {list(AREA_LAYERS)}")
    return AREA_LAYERS[name](*check_bbox(west, south, east, north), **kwargs)


def place_context(lat: float, lon: float, layers: list[str] | str | None = None) -> dict[str, Any]:
    """The context of a point: every layer (or the ones named), each with a one-line summary and its licence."""
    lat, lon = check_point(lat, lon)
    names = _pick(layers, LAYERS)
    results = _run([(n, lambda n=n: LAYERS[n](lat, lon)) for n in names])
    return {
        "lat": lat, "lon": lon, "layers": results,
        "summary": [results[n].get("summary") for n in names if results[n].get("summary")],
        "attribution": sorted({s["attribution"] for r in results.values() for s in r.get("sources", [])}),
    }


def area_context(west: float, south: float, east: float, north: float,
                 layers: list[str] | str | None = None) -> dict[str, Any]:
    """The context of a box. Vector layers count what is inside; raster layers are sampled on a small grid."""
    bbox = check_bbox(west, south, east, north)
    names = _pick(layers, AREA_LAYERS)
    results = _run([(n, lambda n=n: AREA_LAYERS[n](*bbox)) for n in names])
    return {
        "bbox": list(bbox), "layers": results,
        "summary": [results[n].get("summary") for n in names if results[n].get("summary")],
        "attribution": sorted({s["attribution"] for r in results.values() for s in r.get("sources", [])}),
    }
