"""What a study puts on the map, as GeoJSON.

Every step result can say where it happened: the gauge it read, the
catchment it described, the reanalysis or model grid cell it sampled, the
donor gauges it borrowed from, the stations it listed. This module reads
those places out of the payloads the tools already return and writes them
as GeoJSON features with ``{"role", "label", "step_id"}`` properties, so
every face gets the same map: the Explorer draws the features as each step
lands, the bundle carries ``study_map.geojson``, ``aquascope run`` writes it
next to ``results.json``, and the MCP tools return it with the workspace.

Pure Python (no shapely, no geopandas), so it runs in the Pyodide worker.
Every reader is defensive: a shape it does not know yields no feature,
never an exception.

Roles:

``site``       the point the study was started at (no step)
``gauge``      the station a step read
``catchment``  the catchment a step described: its polygon when the payload
               carries one, else its outlet point
``grid_cell``  an ERA5 or GloFAS grid cell as a box (``approximate`` when it
               was snapped from the point rather than reported by the source)
``donor``      a donor gauge of a similarity or regionalisation step
``station``    a station a search or the reconnaissance listed
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "ARTIFACT_ID", "FILE_NAME", "MEDIA_TYPE", "ROLES", "attach", "feature_collection", "publish", "run_features",
    "step_features", "to_geojson", "workspace_features",
]

ROLES = ("site", "gauge", "catchment", "grid_cell", "donor", "station")
ARTIFACT_ID = "study-map"
FILE_NAME = "study_map.geojson"
MEDIA_TYPE = "application/geo+json"

#: Grid spacing in degrees of the cells a study samples through Open-Meteo: ERA5 (the ``/era5`` endpoint,
#: 0.25 degrees) and GloFAS v4 (0.05 degrees, about 5 km). The snapped box is marked approximate.
GRIDS: dict[str, dict[str, Any]] = {
    "era5": {"res": 0.25, "offset": 0.0, "label": "ERA5 cell (0.25 degrees, approximate)"},
    "glofas": {"res": 0.05, "offset": 0.025, "label": "GloFAS cell (about 5 km, approximate)"},
}
#: At most this many listed stations or donors per step, nearest first as the tool ordered them.
MAX_POINTS = 50

_STATION_KEYS = ("source", "station_id", "name", "distance_km", "score", "similarity_distance", "area_km2")


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _lat_lon(d: Any) -> tuple[float, float] | None:
    """(lat, lon) from ``latitude``/``longitude``, ``lat``/``lon`` or a ``point`` dict; None when out of range."""
    if not isinstance(d, dict):
        return None
    p = d.get("point") if isinstance(d.get("point"), dict) else d
    lat = _num(p.get("lat", p.get("latitude")))
    lon = _num(p.get("lon", p.get("longitude")))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _point(lat: float, lon: float) -> dict[str, Any]:
    return {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]}


def _feature(geometry: dict[str, Any], role: str, label: str, step_id: str | None, **props: Any) -> dict[str, Any]:
    properties = {"role": role, "label": label, "step_id": step_id}
    properties.update({k: v for k, v in props.items() if v is not None})
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _is_area(geom: Any) -> bool:
    return (isinstance(geom, dict) and geom.get("type") in ("Polygon", "MultiPolygon")
            and isinstance(geom.get("coordinates"), list) and bool(geom["coordinates"]))


def _cell_box(lat: float, lon: float, grid: str) -> dict[str, Any]:
    g = GRIDS[grid]
    res, off = g["res"], g["offset"]
    clat = math.floor((lat - off) / res + 0.5) * res + off
    clon = math.floor((lon - off) / res + 0.5) * res + off
    h = res / 2
    s, n = max(clat - h, -90.0), min(clat + h, 90.0)
    w, e = clon - h, clon + h
    ring = [[round(w, 6), round(s, 6)], [round(e, 6), round(s, 6)], [round(e, 6), round(n, 6)],
            [round(w, 6), round(n, 6)], [round(w, 6), round(s, 6)]]
    return {"type": "Polygon", "coordinates": [ring]}


def _station_label(st: dict[str, Any]) -> str:
    sid = st.get("station_id") or st.get("id")
    name = st.get("station_name") or st.get("name")
    if name and sid and str(name).strip() and str(name).strip() != str(sid):
        return f"{str(name).strip()} ({st.get('source') or ''} {sid})".replace("( ", "(")
    if st.get("source") and sid:
        return f"{st['source']} {sid}"
    return str(sid or name or "station")


#: Tools whose listed stations are donors (ranked by similarity), not just neighbours.
DONOR_TOOLS = ("similar_basins", "regionalize_signatures")


def _listed(tool: str | None, payload: dict[str, Any]) -> tuple[str, list[Any]]:
    """The stations a payload lists and the role they play: donors of a similarity search, else stations."""
    sim = payload.get("similarity")
    if isinstance(sim, dict) and isinstance(sim.get("donors"), list):
        return "donor", sim["donors"]
    if isinstance(payload.get("donors"), list):
        return "donor", payload["donors"]
    if isinstance(payload.get("stations"), list):
        donor = tool in DONOR_TOOLS or "features_used" in payload
        return ("donor" if donor else "station"), payload["stations"]
    return "station", []


def step_features(step_id: str | None, tool: str | None, payload: Any, *, site: dict[str, Any] | None = None,
                  stations: dict[tuple[str, str], tuple[float, float]] | None = None) -> list[dict[str, Any]]:
    """The GeoJSON features one step result puts on the map (possibly none).

    ``stations`` maps ``(source, station_id)`` to ``(lat, lon)`` for a payload that names its gauge without
    coordinates (``analyze_station`` does); ``site`` places a catchment or a cell whose payload has no point.
    """
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    out: list[dict[str, Any]] = []
    here = _lat_lon(payload) or _lat_lon(site)

    # the gauge the step read
    src, sid = payload.get("source"), payload.get("station_id")
    st = payload.get("station") if isinstance(payload.get("station"), dict) else None
    if (not src or not sid) and st:
        src, sid = st.get("source"), st.get("station_id")
    if src and sid:
        at = _lat_lon(payload) or _lat_lon(st) or (stations or {}).get((str(src), str(sid)))
        if at is not None:
            out.append(_feature(_point(*at), "gauge", _station_label({**(st or {}), **payload}), step_id,
                                source=str(src), station_id=str(sid)))

    # the catchment it described
    sb = payload.get("sub_basin") if isinstance(payload.get("sub_basin"), dict) else None
    if sb is not None and (tool == "describe_catchment" or isinstance(payload.get("upstream"), dict)):
        # the upstream polygon when the producer drew one; the sub-basin's own polygon is only the outlet's
        # level-12 cell, not the catchment, so it is not used
        geom = next((g for g in (payload.get("geometry"), (payload.get("catchment") or {}).get("geometry")
                                 if isinstance(payload.get("catchment"), dict) else None) if _is_area(g)), None)
        up = _num(sb.get("up_area"))
        label = f"Catchment, {up:,.0f} km2 upstream" if up is not None else "Catchment"
        if geom is None and here is not None:
            geom = _point(*here)
            label += " (outlet)"
        if geom is not None:
            out.append(_feature(geom, "catchment", label, step_id, hybas_id=sb.get("hybas_id"), up_area_km2=up,
                                outline=_is_area(geom)))

    # the grid cells it sampled
    cells = payload.get("cells") if isinstance(payload.get("cells"), dict) else {}
    for grid, key in (("era5", "climate"), ("glofas", "glofas")):
        if not isinstance(payload.get(key), dict):
            continue
        reported = _lat_lon(cells.get(grid)) or _lat_lon(payload[key].get("cell"))
        at = reported or here
        if at is None:
            continue
        box = _cell_box(*at, grid)
        label = GRIDS[grid]["label"] if reported is None else GRIDS[grid]["label"].replace(", approximate", "")
        out.append(_feature(box, "grid_cell", label, step_id, model=grid, resolution_deg=GRIDS[grid]["res"],
                            approximate=reported is None))

    # the donors or the stations it listed
    role, rows = _listed(tool, payload)
    n = 0
    for row in rows:
        if n >= MAX_POINTS:
            break
        if not isinstance(row, dict):
            continue
        at = _lat_lon(row)
        if at is None and row.get("source") and row.get("station_id"):
            at = (stations or {}).get((str(row["source"]), str(row["station_id"])))
        if at is None:
            continue
        extra = {k: row.get(k) for k in _STATION_KEYS if k != "name" and _scalar(row.get(k))}
        out.append(_feature(_point(*at), role, _station_label(row), step_id, **extra))
        n += 1
    return out


def _scalar(x: Any) -> bool:
    return x is not None and isinstance(x, (str, int, float, bool)) and not (isinstance(x, float) and
                                                                            not math.isfinite(x))


def feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """A FeatureCollection with its ``bbox`` ([west, south, east, north]) when it has any coordinates."""
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": list(features)}
    xs: list[float] = []
    ys: list[float] = []

    def walk(c: Any) -> None:
        if isinstance(c, (list, tuple)) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
            xs.append(float(c[0]))
            ys.append(float(c[1]))
        elif isinstance(c, (list, tuple)):
            for x in c:
                walk(x)

    for f in features:
        walk((f.get("geometry") or {}).get("coordinates"))
    if xs:
        fc["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    return fc


def _station_lookup(results: list[dict[str, Any]],
                    stations: dict[tuple[str, str], tuple[float, float]] | None) -> dict[tuple[str, str],
                                                                                      tuple[float, float]]:
    """Coordinates by (source, station_id): the ones given, plus every station a payload of the run lists
    (the reconnaissance lists the gauges a later step reads)."""
    out = dict(stations or {})
    for rec in results:
        for p in (rec.get("result"), (rec.get("fallback") or {}).get("result")
                  if isinstance(rec.get("fallback"), dict) else None):
            if not isinstance(p, dict):
                continue
            for row in p.get("stations") or []:
                if isinstance(row, dict) and row.get("source") and row.get("station_id"):
                    at = _lat_lon(row)
                    if at is not None:
                        out.setdefault((str(row["source"]), str(row["station_id"])), at)
    return out


def _entries(rec: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    out = [(rec, False)]
    if isinstance(rec.get("fallback"), dict):
        out.append((rec["fallback"], True))
    return out


def attach(results: list[dict[str, Any]], *, site: dict[str, Any] | None = None,
           stations: dict[tuple[str, str], tuple[float, float]] | None = None) -> None:
    """Put each step's own features on its run record as ``rec["map"]`` (a FeatureCollection), or drop the key
    when the step put nothing on the map. The fallback's features ride on the step, marked ``fallback``."""
    lookup = _station_lookup(results, stations)
    for rec in results:
        if not isinstance(rec, dict):
            continue
        feats: list[dict[str, Any]] = []
        sid = rec.get("id")
        for entry, is_fb in _entries(rec):
            if not entry.get("ok") and not isinstance(entry.get("result"), dict):
                continue
            for f in step_features(sid, entry.get("tool"), entry.get("result"), site=site, stations=lookup):
                if is_fb:
                    f["properties"]["fallback"] = True
                feats.append(f)
        if feats:
            rec["map"] = feature_collection(feats)
        else:
            rec.pop("map", None)


def run_features(results: list[dict[str, Any]], *, site: dict[str, Any] | None = None,
                 stations: dict[tuple[str, str], tuple[float, float]] | None = None) -> dict[str, Any]:
    """The whole study on the map: the site, then every step's features in run order."""
    lookup = _station_lookup(results, stations)
    feats: list[dict[str, Any]] = []
    at = _lat_lon(site)
    for rec in results:
        if not isinstance(rec, dict):
            continue
        for entry, is_fb in _entries(rec):
            for f in step_features(rec.get("id"), entry.get("tool"), entry.get("result"), site=site,
                                   stations=lookup):
                if is_fb:
                    f["properties"]["fallback"] = True
                feats.append(f)
    if at is not None:
        feats.insert(0, _feature(_point(*at), "site", "Study site", None))
    return feature_collection(feats)


def _ws_stations(ws: Any) -> dict[tuple[str, str], tuple[float, float]]:
    inv = getattr(ws, "inventory", None)
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for d in getattr(inv, "datasets", None) or []:
        if d.source and d.station_id and _num(d.lat) is not None and _num(d.lon) is not None:
            out.setdefault((str(d.source), str(d.station_id)), (float(d.lat), float(d.lon)))
    return out


def workspace_features(ws: Any) -> dict[str, Any]:
    """The study map of a Studio workspace (an object or its ``to_dict``): the site and every step result."""
    if isinstance(ws, dict):
        from aquascope.studio.workspace import Workspace

        ws = Workspace.from_dict(ws)
    results = list((ws.run or {}).get("results") or [])
    return run_features(results, site=ws.site, stations=_ws_stations(ws))


def to_geojson(fc: dict[str, Any]) -> str:
    return json.dumps(fc, ensure_ascii=False, default=str, indent=1) + "\n"


def publish(ws: Any, results: list[dict[str, Any]], on_artifact: Any = None) -> dict[str, Any]:
    """The Analysts' hook after each round of results: every record gets its own ``map``, and the whole study
    map becomes the ``study_map.geojson`` artifact (replaced by id), handed to ``on_artifact`` when it changed
    so a face can draw it as the steps land. Returns the FeatureCollection."""
    from aquascope.studio.workspace import Artifact

    stations = _ws_stations(ws)
    attach(results, site=ws.site, stations=stations)
    fc = run_features(results, site=ws.site, stations=stations)
    if not any(f["properties"]["role"] != "site" for f in fc["features"]):
        return fc
    data = to_geojson(fc).encode("utf-8")
    old = ws.artifact(ARTIFACT_ID)
    if old is not None and old.data == data:
        return fc
    art = ws.add_artifact(Artifact(
        id=ARTIFACT_ID, kind="data", name=FILE_NAME, data=data, media_type=MEDIA_TYPE,
        caption="Where the study looked, as GeoJSON: the site, the gauges, the catchment, the grid cells and "
                "the donor gauges, each with the step that used it.",
        meta={"features": len(fc["features"])}))
    if on_artifact:
        try:
            on_artifact(art)
        except Exception:  # noqa: BLE001 - a face's callback must not stop the study
            pass
    return fc
