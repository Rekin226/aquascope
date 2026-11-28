"""Screen GEOGLOWS VPU 514 (New Zealand) against SWORD v17b river reaches, offline.

Run `python -m benchmarks.swot_crosswalk --help` from the checkout.
The report and source downloads are in benchmarks/swot_crosswalk.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

NZTM = "EPSG:2193"
RADIUS_M = 500
AREA_FACTOR = 2
TIE_M = 100
# Independent station coordinates and catchment areas, sourced in the benchmark notes.
SITES = {
    "Buller at Te Kuha (93203)": (-41.8342457, 171.6993625, 6350),
    "Grey at Dobson (91401)": (-42.4532880, 171.2978045, 3830),
    "Wairau at Tuamarina (5870110)": (-41.44, 173.96, 3430),
}


def _sample(lines: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Sample across the total length, including every part of a multipart channel."""
    parts, parent = shapely.get_parts(lines, return_index=True)
    lengths = shapely.length(parts)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    first = np.r_[True, np.diff(parent) != 0]
    offsets = starts[first]
    targets = offsets[:, None] + shapely.length(lines)[:, None] * fractions
    indices = np.searchsorted(ends, targets)
    points: np.ndarray = shapely.line_interpolate_point(parts[indices], targets - starts[indices])
    return points


def crosswalk(streams: gpd.GeoDataFrame, sword: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a decision per GEOGLOWS reach and the evidence for every nearby SWORD river."""
    streams = streams.to_crs(NZTM).reset_index(drop=True)
    rivers = sword.loc[sword["type"] == 1].to_crs(NZTM).reset_index(drop=True)
    for frame, key in [(streams, "LINKNO"), (rivers, "reach_id")]:
        if frame[key].isna().any() or not frame[key].is_unique:
            raise ValueError(f"{key} must identify one reach per row")
        if not frame.geometry.geom_type.isin(["LineString", "MultiLineString"]).all() or (frame.length <= 0).any():
            raise ValueError(f"{key} has an empty or non-linear geometry")

    lines = streams.geometry.to_numpy()
    reaches = rivers.geometry.to_numpy()
    pairs = shapely.STRtree(reaches).query(lines, predicate="dwithin", distance=RADIUS_M)
    src, dst = pairs
    geog_area = streams.DSContArea.to_numpy(dtype=float) / 1e6
    sword_area = rivers.facc.to_numpy(dtype=float)[dst]
    ratio = np.full(len(src), np.nan)
    known = (np.isfinite(geog_area[src]) & (geog_area[src] > 0) & np.isfinite(sword_area) & (sword_area > 0))
    np.divide(geog_area[src], sword_area, where=known, out=ratio)
    samples = _sample(lines, np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    dist = shapely.distance(samples[src], reaches[dst, None])

    candidates = pd.DataFrame({
        "river_id": streams.LINKNO.to_numpy()[src], "sword_id": rivers.reach_id.to_numpy()[dst],
        "river_name": rivers.river_name.to_numpy()[dst], "sword_km2": sword_area, "area_ratio": ratio,
        "near_points": np.sum(dist <= RADIUS_M, axis=1), "mean_distance_m": dist.mean(axis=1),
    }).sort_values(["river_id", "mean_distance_m", "sword_id"], ignore_index=True)
    candidates["geometry_ok"] = candidates.near_points >= 4
    candidates["area_ok"] = candidates.area_ratio.between(1 / AREA_FACTOR, AREA_FACTOR)
    eligible = candidates[candidates.geometry_ok & candidates.area_ok]
    best = eligible.drop_duplicates("river_id").set_index("river_id")
    second = eligible[eligible.groupby("river_id").cumcount() == 1].set_index("river_id")
    ambiguous = (second.mean_distance_m - best.mean_distance_m).le(TIE_M)

    links = streams[["LINKNO", "strmOrder"]].rename(columns={"LINKNO": "river_id", "strmOrder": "strahler_order"})
    links["geoglows_km2"] = geog_area
    links = links.set_index("river_id")
    links["status"] = "no_nearby_river"
    links.loc[links.index.isin(candidates.river_id), "status"] = "geometry_failed"
    near = candidates[candidates.geometry_ok]
    links.loc[links.index.isin(near.river_id), "status"] = "area_missing"
    links.loc[links.index.isin(near.loc[near.area_ratio.notna(), "river_id"]), "status"] = "area_failed"
    links.loc[best.index, "status"] = "matched"
    links.loc[ambiguous.index[ambiguous], "status"] = "ambiguous"
    links["sword_id"] = best.sword_id.reindex(links.index).astype("Int64").where(links.status == "matched")
    return links.sort_index().reset_index(), candidates


def report(streams: gpd.GeoDataFrame, sword: gpd.GeoDataFrame, links: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    """Coverage denominators and two independent station checks; a match is a candidate, not ground truth."""
    streams = streams.to_crs(NZTM)
    rivers = sword.loc[sword["type"] == 1].to_crs(NZTM)
    matched = links[links.status == "matched"].merge(candidates, on=["river_id", "sword_id"], validate="one_to_one")
    summary = {}
    for name, subset in [("all", links), ("strahler_5_plus", links[links.strahler_order >= 5])]:
        counts = {k: int(v) for k, v in subset.status.value_counts().items()}
        rate = 100 * counts.get("matched", 0) / len(subset) if len(subset) else None
        summary[name] = {"geoglows_reaches": len(subset), "counts": counts,
                         "match_percent": round(rate, 2) if rate is not None else None}
    transform = Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    sites = {}
    for name, (lat, lon, area) in SITES.items():
        point = shapely.Point(*transform.transform(lon, lat))
        g = streams.loc[streams.distance(point).idxmin()]
        s = rivers.loc[rivers.distance(point).idxmin()]
        decision = links.loc[links.river_id == g.LINKNO].iloc[0]
        sites[name] = {
            "lat": lat, "lon": lon, "reported_km2": area, "river_id": int(g.LINKNO),
            "geoglows_distance_m": round(g.geometry.distance(point)), "geoglows_km2": round(g.DSContArea / 1e6, 1),
            "nearest_sword_id": int(s.reach_id), "nearest_sword_name": s.river_name,
            "sword_distance_m": round(s.geometry.distance(point)), "sword_km2": round(float(s.facc), 1),
            "status": decision.status,
            "assigned_sword_id": None if pd.isna(decision.sword_id) else int(decision.sword_id),
        }
    return {"sword_rows": len(sword), "sword_rivers": len(rivers), "coverage": summary,
            "distinct_matched_sword_rivers": int(matched.sword_id.nunique()),
            "area_ratio_quantiles": {str(p): round(float(v), 3) for p, v in matched.area_ratio.quantile(
                [0.1, 0.5, 0.9]).items()} if len(matched) else {},
            "stations": sites}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", required=True, type=Path, help="GEOGLOWS streams_514.gpkg")
    parser.add_argument("--sword", required=True, type=Path, help="SWORD oc_sword_reaches_hb57_v17b.shp")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args()
    streams = gpd.read_file(args.streams, engine="pyogrio", columns=["LINKNO", "DSContArea", "strmOrder"])
    sword = gpd.read_file(args.sword, engine="pyogrio", columns=["reach_id", "facc", "type", "river_name"])
    links, candidates = crosswalk(streams, sword)
    args.out.mkdir(parents=True, exist_ok=True)
    links.to_csv(args.out / "crosswalk.csv", index=False)
    candidates.to_csv(args.out / "candidates.csv", index=False)
    summary = report(streams, sword, links, candidates)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
