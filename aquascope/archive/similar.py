"""Nearest similar gauged basins: prediction in ungauged basins, the practical half (#53).

Given any point on land (or any station), find the gauged catchments in the
Archive that look most like its catchment: same size class, climate, relief,
land cover, soils, human pressure. That is the "physical similarity" donor
selection of regionalisation (Bloeschl et al. 2013, Runoff Prediction in
Ungauged Basins; Oudin et al. 2008 for why spatial proximity is a strong
baseline), and it is what a consultant does by hand before transferring
anything to an ungauged site.

Two tables make it instant:

``basins/station_catchments.parquet``
    one row per catalog station: ``source, station_id, hybas_id, up_area,
    sub_area, area_km2, area_source, attribute_scope`` and the catchment
    attributes of :data:`aquascope.archive.basins.ATTRIBUTE_GUIDE` (real
    units). ``area_km2`` is the agency's catchment area when the catalog
    carries one (``extra.catchment_area_km2``: USGS, UK EA, Hub'Eau), else
    BasinATLAS ``up_area``. ``attribute_scope`` says which BasinATLAS values
    were taken: ``upstream`` when the gauge closes (most of) its sub-basin's
    catchment, ``local`` when the agency area shows the gauge drains only a
    corner of the sub-basin (a small creek inside a big river's sub-basin
    would otherwise inherit the big river's attributes). Built weekly by the
    harvest workflow with :func:`assign_station_catchments` (a spatial join of
    the catalog against the BasinATLAS FlatGeobuf), published next to the catalog.

The search standardises a chosen feature set over the table (z-scores),
measures Euclidean distance in that space (``method="similarity"``), or
great-circle distance (``"proximity"``), or both combined (``"combined"``:
similarity distance plus proximity in units of 500 km), and returns the
``k`` closest stations that measure discharge, with the per-feature deltas
so the reader can see why they were chosen.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquascope.archive.basins import ATTRIBUTION, LICENSE, row_catchment_attributes
from aquascope.registry import SOURCES

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
TABLE_NAME = "station_catchments.parquet"

# feature key -> (attribute key in the guide / table column, transform, weight)
FEATURES: dict[str, tuple[str, str | None, float]] = {
    "log_area": ("area_km2", "log10", 1.5),
    "elevation": ("elevation_m", None, 1.0),
    "slope": ("slope_deg", "log1p", 1.0),
    "precipitation": ("precipitation_mm_yr", None, 1.5),
    "aridity": ("aridity_index", "log1p", 1.5),
    "temperature": ("temperature_c", None, 1.0),
    "snow": ("snow_cover_pct", None, 1.0),
    "forest": ("forest_pct", None, 0.7),
    "cropland": ("cropland_pct", None, 0.7),
    "urban": ("urban_pct", "log1p", 0.7),
    "clay": ("clay_pct", None, 0.5),
    "sand": ("sand_pct", None, 0.5),
    "population_density": ("population_density", "log1p", 0.5),
    "regulation": ("degree_of_regulation_pct", "log1p", 0.7),
}
PROXIMITY_SCALE_KM = 500.0


def table_url(repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/basins/{TABLE_NAME}"


# ── build (harvest workflow) ────────────────────────────────────────────────


def assign_station_catchments(
    catalog: list[dict[str, Any]],
    fgb_path: str | Path,
    attributes_path: str | Path,
    out_path: str | Path,
    *,
    batch: int = 20_000,
) -> pd.DataFrame:
    """Spatial-join every catalog station to its level-12 sub-basin and tabulate the catchment attributes.

    Needs geopandas + pyogrio (``basins`` extra) and the local ``lev12.fgb`` and
    ``lev12_attributes.parquet``. Stations in the sea or off coverage get no row.
    """
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import Point

    from aquascope.archive.basins import TOPOLOGY_COLUMNS, load_attributes

    t0 = time.perf_counter()
    pts = gpd.GeoDataFrame(
        {"source": [r["source"] for r in catalog], "station_id": [r["station_id"] for r in catalog]},
        geometry=[Point(float(r["longitude"]), float(r["latitude"])) for r in catalog], crs="EPSG:4326",
    )
    # Tile the world so the million polygons never sit in memory at once: read the sub-basins
    # intersecting each 20 x 20 degree tile that holds stations, join those points, move on.
    joined_parts = []
    xs, ys = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()
    tile = 20.0
    tiles = sorted({(math.floor(x / tile) * tile, math.floor(y / tile) * tile) for x, y in zip(xs, ys)})
    for tx, ty in tiles:
        in_tile = pts[(xs >= tx) & (xs < tx + tile) & (ys >= ty) & (ys < ty + tile)]
        if in_tile.empty:
            continue
        polys = pyogrio.read_dataframe(str(fgb_path), bbox=(tx - 0.01, ty - 0.01, tx + tile + 0.01, ty + tile + 0.01))
        if polys.empty:
            continue
        polys.columns = [c.lower() if c.upper() in TOPOLOGY_COLUMNS else c for c in polys.columns]
        keep = [c for c in ("hybas_id", "up_area", "sub_area", "next_down") if c in polys.columns]
        polys = polys[keep + [polys.geometry.name]]
        part = gpd.sjoin(in_tile, polys, how="inner", predicate="within")
        joined_parts.append(part[~part.index.duplicated(keep="first")])
        logger.info("  tile %+.0f,%+.0f: %d stations, %d sub-basins, %d matched",
                    tx, ty, len(in_tile), len(polys), len(part))
    joined = pd.concat(joined_parts) if joined_parts else pts.iloc[0:0].assign(hybas_id=[], up_area=[], sub_area=[])
    joined = joined[~joined.index.duplicated(keep="first")]
    logger.info("station join: %d of %d stations fall in a sub-basin (%.0fs)", len(joined), len(pts),
                time.perf_counter() - t0)

    ids = sorted({int(x) for x in joined["hybas_id"].dropna().unique()})
    rows: list[dict[str, Any]] = []
    attrs_by_id: dict[int, dict[str, Any]] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        df = load_attributes(chunk, path=attributes_path)
        for rec in df.to_dict("records"):
            attrs_by_id[int(rec["hybas_id"])] = rec
    agency_area = {(r["source"], r["station_id"]): (r.get("extra") or {}).get("catchment_area_km2") for r in catalog}
    for rec in joined.drop(columns=[joined.geometry.name]).to_dict("records"):
        hid = int(rec["hybas_id"])
        up_area = float(rec.get("up_area") or 0.0)
        sub_area = float(rec.get("sub_area") or 0.0)
        agency = agency_area.get((rec["source"], rec["station_id"]))
        try:
            agency = float(agency) if agency is not None and float(agency) > 0 else None
        except (TypeError, ValueError):
            agency = None
        scope, area, area_source = "upstream", up_area, "basinatlas_up_area"
        if agency is not None:
            area, area_source = agency, "agency"
            if up_area and agency < 0.5 * up_area:
                scope = "local"  # the gauge drains a fraction of the sub-basin's catchment
        base = {"source": rec["source"], "station_id": rec["station_id"], "hybas_id": hid,
                "up_area": up_area, "sub_area": sub_area, "area_km2": float(area), "area_source": area_source,
                "attribute_scope": scope}
        base.update(row_catchment_attributes(attrs_by_id.get(hid, {}), scope=scope))
        rows.append(base)
    out = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("station catchments: %d rows, %d columns -> %s (%.0fs)", len(out), out.shape[1], out_path,
                time.perf_counter() - t0)
    return out


# ── read side ───────────────────────────────────────────────────────────────


def load_station_catchments(*, repo_id: str = DEFAULT_REPO_ID, refresh: bool = False,
                            path: str | Path | None = None) -> pd.DataFrame:
    """The station -> catchment table (downloaded once a day into the cache)."""
    if path is None:
        from aquascope.archive.catalog import _download, cache_dir

        path = _download(table_url(repo_id), cache_dir() / f"{repo_id.replace('/', '__')}__{TABLE_NAME}", refresh)
    return pd.read_parquet(path)


def _transform(values: pd.Series | np.ndarray | float, how: str | None):
    v = np.asarray(values, dtype=float)
    if how == "log10":
        return np.log10(np.clip(v, 1e-3, None))
    if how == "log1p":
        return np.log1p(np.clip(v, 0, None))
    return v


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def similar_basins(
    target: dict[str, Any],
    *,
    k: int = 10,
    method: str = "similarity",
    require_discharge: bool = True,
    sources: list[str] | None = None,
    features: list[str] | None = None,
    exclude: tuple[str, str] | None = None,
    table: pd.DataFrame | None = None,
    catalog: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank gauged stations by how much their catchment resembles ``target``.

    ``target`` carries the catchment attributes under the guide keys (as
    :func:`row_catchment_attributes` / ``describe_catchment()["attributes"]``
    give them) plus ``latitude`` / ``longitude`` (needed for proximity).
    ``method``: ``similarity`` (attribute space), ``proximity`` (distance on the
    ground) or ``combined``. ``exclude`` drops one ``(source, station_id)`` (the
    target station itself). Returns the ranking with per-feature deltas.
    """
    from aquascope.archive.catalog import load_stations

    if method not in ("similarity", "proximity", "combined"):
        raise ValueError("method must be similarity, proximity or combined")
    tab = table if table is not None else load_station_catchments()
    cat = catalog if catalog is not None else load_stations()
    meta = {(r["source"], r["station_id"]): r for r in cat}
    keys = [(s, i) for s, i in zip(tab["source"], tab["station_id"])]
    if require_discharge:
        mask = np.array(["discharge" in ((meta.get(kk) or {}).get("variables") or []) for kk in keys])
    else:
        mask = np.ones(len(tab), dtype=bool)
    if sources:
        mask &= tab["source"].isin(sources).to_numpy()
    if exclude:
        mask &= ~((tab["source"] == exclude[0]) & (tab["station_id"] == exclude[1])).to_numpy()
    cand = tab[mask].reset_index(drop=True)
    if cand.empty:
        return {"n_candidates": 0, "stations": [], "method": method}

    tval = _target_values(target)
    feats = [f for f in (features or list(FEATURES)) if f in FEATURES]
    used = [f for f in feats if FEATURES[f][0] in cand.columns and tval.get(f) is not None]
    dist_attr = np.zeros(len(cand))
    contributions: dict[str, np.ndarray] = {}
    if method in ("similarity", "combined") and used:
        parts = []
        for f in used:
            col, how, w = FEATURES[f]
            x = _transform(cand[col].to_numpy(dtype=float), how)
            mu, sd = np.nanmean(x), np.nanstd(x)
            sd = sd if sd and not math.isnan(sd) else 1.0
            z = (x - mu) / sd
            zt = (_transform(tval[f], how) - mu) / sd
            d = np.where(np.isnan(z), 3.0, np.abs(z - zt)) * w  # missing values are penalised, not ignored
            contributions[f] = d
            parts.append(d ** 2)
        dist_attr = np.sqrt(np.sum(parts, axis=0) / max(sum(FEATURES[f][2] ** 2 for f in used), 1e-9))
    dist_km = None
    if method in ("proximity", "combined"):
        lat0, lon0 = float(target.get("latitude")), float(target.get("longitude"))
        lats = np.array([float((meta.get(kk) or {}).get("latitude", np.nan)) for kk in
                         zip(cand["source"], cand["station_id"])])
        lons = np.array([float((meta.get(kk) or {}).get("longitude", np.nan)) for kk in
                         zip(cand["source"], cand["station_id"])])
        dist_km = _haversine_km(lat0, lon0, lats, lons)
    if method == "similarity":
        score = dist_attr
    elif method == "proximity":
        score = np.nan_to_num(dist_km, nan=1e9)
    else:
        score = np.sqrt(dist_attr ** 2 + (np.nan_to_num(dist_km, nan=1e9) / PROXIMITY_SCALE_KM) ** 2)
    order = np.argsort(score)[: max(1, int(k))]
    out = []
    for i in order:
        row = cand.iloc[i]
        st = meta.get((row["source"], row["station_id"]), {})
        deltas = {}
        for f in used:
            col = FEATURES[f][0]
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                deltas[f] = {"station": round(float(v), 3), "target": round(float(tval[f]), 3)}
        out.append({
            "source": row["source"], "station_id": row["station_id"], "name": st.get("name"),
            "latitude": st.get("latitude"), "longitude": st.get("longitude"),
            "agency": SOURCES[row["source"]].agency if row["source"] in SOURCES else None,
            "period_start": st.get("period_start"), "period_end": st.get("period_end"),
            "score": round(float(score[i]), 4),
            "similarity_distance": round(float(dist_attr[i]), 4) if used else None,
            "distance_km": round(float(dist_km[i]), 1) if dist_km is not None and not math.isnan(dist_km[i]) else None,
            "hybas_id": int(row["hybas_id"]), "up_area_km2": round(float(row["up_area"]), 1),
            "features": deltas,
        })
    return {
        "method": method, "features_used": used, "n_candidates": int(len(cand)), "k": len(out),
        "target": {f: tval[f] for f in used}, "stations": out,
        "license": LICENSE, "attribution": ATTRIBUTION,
        "methods": [{
            "name": "Similar gauged basins (physical similarity / spatial proximity)",
            "text": "Gauged stations ranked by weighted Euclidean distance in standardised BasinATLAS catchment "
                    "attribute space (area, relief, climate, land cover, soils, human pressure), by great-circle "
                    "distance, or both; the donor-selection step of regionalisation.",
            "citation": "Bloeschl, G., Sivapalan, M., Wagener, T., Viglione, A., Savenije, H. (eds.) (2013). Runoff "
                        "Prediction in Ungauged Basins. Cambridge University Press; Oudin, L. et al. (2008). Spatial "
                        "proximity, physical similarity, regression and ungaged catchments. Water Resour. Res. 44, "
                        "W03413. Attributes: " + ATTRIBUTION,
        }],
    }


def _target_drainage_area(target: dict[str, Any]) -> float | None:
    """Use the outlet area for a point and the gauge's own drainage area for a station row."""
    basin = target.get("sub_basin")
    outlet_area = basin.get("up_area") if isinstance(basin, dict) else None
    attrs = target.get("attributes")
    if isinstance(attrs, dict):
        # In a single-row catchment description, area_km2 is only the polygon's area.
        candidates = (attrs.get("upstream_area_km2"), outlet_area, attrs.get("up_area"), attrs.get("area_km2"))
    else:
        candidates = (target.get("area_km2"), target.get("up_area"), target.get("upstream_area_km2"), outlet_area)
    for value in candidates:
        if isinstance(value, dict):
            value = value.get("value")
        if value is None or isinstance(value, bool):
            continue
        try:
            area = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(area) and area > 0:
            return area
    return None


def _target_values(target: dict[str, Any]) -> dict[str, float | None]:
    """Feature values from a target dict (flat guide keys, or describe_catchment's nested attributes)."""
    nested = target.get("attributes")
    attrs = nested if isinstance(nested, dict) else target
    out: dict[str, float | None] = {}
    for f, (col, _how, _w) in FEATURES.items():
        if col == "area_km2":
            out[f] = _target_drainage_area(target)
            continue
        v = attrs.get(col)
        if isinstance(v, dict):
            v = v.get("value")
        try:
            out[f] = None if v is None else float(v)
        except (TypeError, ValueError):
            out[f] = None
    return out


def similar_for_point(
    lat: float, lon: float, *, k: int = 10, method: str = "combined", desc: dict[str, Any] | None = None,
    **kw: Any
) -> dict[str, Any]:
    """Describe the catchment of a point (BasinATLAS) and rank the gauged basins most like it.

    ``desc`` is a ``describe_catchment`` result already in hand (the browser builds one from the sub-basin
    row it read itself, where BasinATLAS cannot be opened); with it, and ``table`` / ``catalog`` in ``kw``,
    nothing here touches a file.
    """
    if desc is None:
        from aquascope.archive.basins import describe_catchment

        desc = describe_catchment(lat, lon, upstream=False)
    if desc.get("error"):
        return {"latitude": lat, "longitude": lon, "error": desc["error"]}
    target = {"latitude": lat, "longitude": lon, "attributes": desc.get("attributes", {}),
              "sub_basin": desc.get("sub_basin")}
    res = similar_basins(target, k=k, method=method, **kw)
    res.update({"latitude": lat, "longitude": lon, "sub_basin": desc.get("sub_basin"),
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return res


def similar_for_station(
    source: str, station_id: str, *, k: int = 10, method: str = "combined", **kw: Any
) -> dict[str, Any]:
    """Rank the gauged basins most like a station's own catchment (the station itself excluded)."""
    from aquascope.archive.catalog import load_stations

    tab = kw.pop("table", None)
    tab = tab if tab is not None else load_station_catchments()
    row = tab[(tab["source"] == source) & (tab["station_id"] == station_id)]
    if row.empty:
        return {"source": source, "station_id": station_id,
                "error": "station has no catchment row (not on land in BasinATLAS, or not in the catalog)"}
    cat = kw.pop("catalog", None)
    cat = cat if cat is not None else load_stations()
    st = next((r for r in cat if r["source"] == source and r["station_id"] == station_id), {})
    target = {**row.iloc[0].to_dict(), "latitude": st.get("latitude"), "longitude": st.get("longitude")}
    res = similar_basins(target, k=k, method=method, exclude=(source, station_id), table=tab, catalog=cat, **kw)
    res.update({"source": source, "station_id": station_id})
    return res


__all__ = ["FEATURES", "TABLE_NAME", "assign_station_catchments", "load_station_catchments", "similar_basins",
           "similar_for_point", "similar_for_station", "table_url"]
