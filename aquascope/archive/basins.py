"""Catchments for the whole world: BasinATLAS (HydroATLAS v1.0, CC BY 4.0) in the Archive.

HydroBASINS level-12 sub-basins (about 1.0 million polygons, ~130 km² each)
carry, in BasinATLAS, some 280 hydro-environmental attributes each: climate,
runoff, land cover, soils, population, dams. Caravan's static attributes are
exactly these. HydroATLAS is CC BY 4.0 (Linke et al. 2019), unlike the bare
HydroBASINS core product whose licence forbids stand-alone redistribution, so
BasinATLAS is what the Archive mirrors, under ``basins/``:

``basins/lev12.fgb``
    every level-12 sub-basin polygon (simplified) as FlatGeobuf, spatially
    indexed, so a point-in-polygon lookup over HTTPS reads a few kilobytes
``basins/lev12_topology.parquet``
    ``hybas_id, next_down, next_sink, main_bas, sub_area, up_area, pfaf_id,
    endo, coast, order, lat, lon``: the routing graph and centroids (small)
``basins/lev12_attributes.parquet``
    every BasinATLAS attribute per sub-basin, sorted by ``hybas_id`` so a
    row-group lookup is cheap
``basins/lev12.pmtiles``, ``basins/lev06.pmtiles``
    vector tiles for the Explorer (attributes limited to the routing keys)

Two jobs live here: :func:`build_basins` turns the BasinATLAS FileGDB into
those files (run by ``.github/workflows/basins.yml``), and the read side
(:func:`sub_basin_at`, :func:`upstream_ids`, :func:`catchment_attributes`,
:func:`describe_catchment`) answers "which catchment is this point in, what
is upstream, and what does the catchment look like" from the published files.

Attribution required by the licence (also embedded in every file's metadata):
Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global
hydro-environmental sub-basin and river reach characteristics at high spatial
resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
LEVEL = 12
LAYER = f"BasinATLAS_v10_lev{LEVEL:02d}"
COARSE_LEVEL = 6

ATTRIBUTION = (
    "HydroATLAS v1.0 (BasinATLAS), CC BY 4.0. Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). "
    "Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. "
    "Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6"
)
LICENSE = "CC-BY-4.0"

TOPOLOGY_COLUMNS = ("HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "SUB_AREA", "UP_AREA", "PFAF_ID", "ENDO",
                    "COAST", "ORDER", "ORDER_")
# The FileGDB stores the ids as doubles (they exceed int32); we publish them as int64.
_ID_COLUMNS = ("hybas_id", "next_down", "next_sink", "main_bas", "pfaf_id")

# The attributes a hydrologist asks about first. BasinATLAS field names end in a
# scope + statistic code: "s" = this sub-basin, "u" = everything upstream (total
# or area-weighted), "p" = at the pour point; "av" average, "se" spatial extent
# (%), "yr" annual, "su" sum. Where an upstream ("u"/"p") field exists we read it
# from the outlet sub-basin's row (that is what BasinATLAS precomputed); otherwise
# the sub-basin ("s") field is aggregated over the upstream set as noted.
ATTRIBUTE_GUIDE: dict[str, tuple[str, str | None, str, str, str]] = {
    # key: (sub-basin field, upstream field or None, unit, aggregation of the sub-basin field, label)
    "elevation_m": ("ele_mt_sav", "ele_mt_uav", "m", "area", "mean elevation"),
    "slope_deg": ("slp_dg_sav", "slp_dg_uav", "degrees", "area", "mean slope"),
    "precipitation_mm_yr": ("pre_mm_syr", "pre_mm_uyr", "mm/yr", "area", "annual precipitation (WorldClim)"),
    "pet_mm_yr": ("pet_mm_syr", "pet_mm_uyr", "mm/yr", "area", "annual potential evapotranspiration"),
    "aet_mm_yr": ("aet_mm_syr", "aet_mm_uyr", "mm/yr", "area", "annual actual evapotranspiration"),
    "aridity_index": ("ari_ix_sav", "ari_ix_uav", "P/PET", "area", "aridity index (P/PET)"),
    "temperature_c": ("tmp_dc_syr", "tmp_dc_uyr", "°C", "area", "mean annual air temperature"),
    "snow_cover_pct": ("snw_pc_syr", "snw_pc_uyr", "%", "area", "annual snow cover extent"),
    "runoff_mm_yr": ("run_mm_syr", None, "mm/yr", "area", "annual land-surface runoff"),
    "discharge_m3s": ("dis_m3_pyr", "dis_m3_pyr", "m3/s", "outlet", "mean annual natural discharge at the outlet"),
    "forest_pct": ("for_pc_sse", "for_pc_use", "%", "area", "forest cover"),
    "cropland_pct": ("crp_pc_sse", "crp_pc_use", "%", "area", "cropland"),
    "pasture_pct": ("pst_pc_sse", "pst_pc_use", "%", "area", "pasture"),
    "urban_pct": ("urb_pc_sse", "urb_pc_use", "%", "area", "urban extent"),
    "irrigated_pct": ("ire_pc_sse", "ire_pc_use", "%", "area", "irrigated area"),
    "glacier_pct": ("gla_pc_sse", "gla_pc_use", "%", "area", "glacier extent"),
    "wetland_pct": ("wet_pc_sg1", "wet_pc_ug1", "%", "area", "wetlands (all classes)"),
    "lake_pct": ("lka_pc_sse", "lka_pc_use", "%", "area", "lake area"),
    "karst_pct": ("kar_pc_sse", "kar_pc_use", "%", "area", "karst extent"),
    "clay_pct": ("cly_pc_sav", "cly_pc_uav", "%", "area", "clay fraction in soil"),
    "silt_pct": ("slt_pc_sav", "slt_pc_uav", "%", "area", "silt fraction in soil"),
    "sand_pct": ("snd_pc_sav", "snd_pc_uav", "%", "area", "sand fraction in soil"),
    "soil_organic_carbon_t_ha": ("soc_th_sav", "soc_th_uav", "t/ha", "area", "soil organic carbon"),
    "soil_water_pct": ("swc_pc_syr", "swc_pc_uyr", "%", "area", "annual soil water content"),
    "groundwater_table_cm": ("gwt_cm_sav", None, "cm", "area", "groundwater table depth"),
    "population_density": ("ppd_pk_sav", "ppd_pk_uav", "people/km2", "area", "population density"),
    "population": ("pop_ct_ssu", "pop_ct_usu", "people", "sum", "population count"),
    "degree_of_regulation_pct": ("dor_pc_pva", "dor_pc_pva", "%", "outlet", "degree of regulation by reservoirs"),
    "human_footprint_2009": ("hft_ix_s09", "hft_ix_u09", "index 0-50", "area", "human footprint (2009)"),
    "reservoir_volume_mcm": ("rev_mc_usu", "rev_mc_usu", "million m3", "outlet", "reservoir volume upstream"),
}
# BasinATLAS attribute families (first six characters of the column name), from the BasinATLAS
# Catalog v1.0: display name, unit as stored, and the multiplier the catalog says the stored
# integers carry ("x10" means value 10 is 1 unit). Population counts are stored in thousands.
FIELDS: dict[str, dict[str, Any]] = {
    "dis_m3": {"id": "H01", "name": "Natural Discharge", "unit": "cubic meters per second", "stored_x": 1},
    "run_mm": {"id": "H02", "name": "Land Surface Runoff", "unit": "millimeters", "stored_x": 1},
    "inu_pc": {"id": "H03", "name": "Inundation Extent", "unit": "percent cover", "stored_x": 1},
    "lka_pc": {"id": "H04", "name": "Limnicity (Percent Lake Area)", "unit": "percent cover (x10)", "stored_x": 10},
    "lkv_mc": {"id": "H05", "name": "Lake Volume", "unit": "million cubic meters", "stored_x": 1},
    "rev_mc": {"id": "H06", "name": "Reservoir Volume", "unit": "million cubic meters", "stored_x": 1},
    "dor_pc": {"id": "H07", "name": "Degree of Regulation", "unit": "percent (x10)", "stored_x": 10},
    "ria_ha": {"id": "H08", "name": "River Area", "unit": "hectares", "stored_x": 1},
    "riv_tc": {"id": "H09", "name": "River Volume", "unit": "thousand cubic meters", "stored_x": 1},
    "gwt_cm": {"id": "H10", "name": "Groundwater Table Depth", "unit": "centimeters", "stored_x": 1},
    "ele_mt": {"id": "P01", "name": "Elevation", "unit": "meters a.s.l.", "stored_x": 1},
    "slp_dg": {"id": "P02", "name": "Terrain Slope", "unit": "degrees (x10)", "stored_x": 10},
    "sgr_dk": {"id": "P03", "name": "Stream Gradient", "unit": "decimeters per km", "stored_x": 1},
    "clz_cl": {"id": "C01", "name": "Climate Zones", "unit": "classes (18)", "stored_x": 1},
    "cls_cl": {"id": "C02", "name": "Climate Strata", "unit": "classes (125)", "stored_x": 1},
    "tmp_dc": {"id": "C03", "name": "Air Temperature", "unit": "degrees Celsius (x10)", "stored_x": 10},
    "pre_mm": {"id": "C04", "name": "Precipitation", "unit": "millimeters", "stored_x": 1},
    "pet_mm": {"id": "C05", "name": "Potential Evapotranspiration", "unit": "millimeters", "stored_x": 1},
    "aet_mm": {"id": "C06", "name": "Actual Evapotranspiration", "unit": "millimeters", "stored_x": 1},
    "ari_ix": {"id": "C07", "name": "Global Aridity Index", "unit": "index value (x100)", "stored_x": 100},
    "cmi_ix": {"id": "C08", "name": "Climate Moisture Index", "unit": "index value (x100)", "stored_x": 100},
    "snw_pc": {"id": "C09", "name": "Snow Cover Extent", "unit": "percent cover", "stored_x": 1},
    "glc_cl": {"id": "L01", "name": "Land Cover Classes", "unit": "classes (22)", "stored_x": 1},
    "glc_pc": {"id": "L02", "name": "Land Cover Extent", "unit": "percent cover", "stored_x": 1},
    "pnv_cl": {"id": "L03", "name": "Potential Natural Vegetation Classes", "unit": "classes (15)", "stored_x": 1},
    "pnv_pc": {"id": "L04", "name": "Potential Natural Vegetation Extent", "unit": "percent cover", "stored_x": 1},
    "wet_cl": {"id": "L05", "name": "Wetland Classes", "unit": "classes (12)", "stored_x": 1},
    "wet_pc": {"id": "L06", "name": "Wetland Extent", "unit": "percent cover", "stored_x": 1},
    "for_pc": {"id": "L07", "name": "Forest Cover Extent", "unit": "percent cover", "stored_x": 1},
    "crp_pc": {"id": "L08", "name": "Cropland Extent", "unit": "percent cover", "stored_x": 1},
    "pst_pc": {"id": "L09", "name": "Pasture Extent", "unit": "percent cover", "stored_x": 1},
    "ire_pc": {"id": "L10", "name": "Irrigated Area Extent (Equipped)", "unit": "percent cover", "stored_x": 1},
    "gla_pc": {"id": "L11", "name": "Glacier Extent", "unit": "percent cover", "stored_x": 1},
    "prm_pc": {"id": "L12", "name": "Permafrost Extent", "unit": "percent cover", "stored_x": 1},
    "pac_pc": {"id": "L13", "name": "Protected Area Extent", "unit": "percent cover", "stored_x": 1},
    "tbi_cl": {"id": "L14", "name": "Terrestrial Biomes", "unit": "classes (14)", "stored_x": 1},
    "tec_cl": {"id": "L15", "name": "Terrestrial Ecoregions", "unit": "classes (846)", "stored_x": 1},
    "fmh_cl": {"id": "L16", "name": "Freshwater Major Habitat Types", "unit": "classes (13)", "stored_x": 1},
    "fec_cl": {"id": "L17", "name": "Freshwater Ecoregions", "unit": "classes (426)", "stored_x": 1},
    "cly_pc": {"id": "S01", "name": "Clay Fraction in Soil", "unit": "percent", "stored_x": 1},
    "slt_pc": {"id": "S02", "name": "Silt Fraction in Soil", "unit": "percent", "stored_x": 1},
    "snd_pc": {"id": "S03", "name": "Sand Fraction in Soil", "unit": "percent", "stored_x": 1},
    "soc_th": {"id": "S04", "name": "Organic Carbon Content in Soil", "unit": "tonnes/hectare", "stored_x": 1},
    "swc_pc": {"id": "S05", "name": "Soil Water Content", "unit": "percent", "stored_x": 1},
    "lit_cl": {"id": "S06", "name": "Lithological Classes", "unit": "classes (16)", "stored_x": 1},
    "kar_pc": {"id": "S07", "name": "Karst Area Extent", "unit": "percent cover", "stored_x": 1},
    "ero_kh": {"id": "S08", "name": "Soil Erosion (RUSLE-based)", "unit": "kg/hectare per year", "stored_x": 1},
    "pop_ct": {"id": "A01", "name": "Population Count", "unit": "count (thousands)", "stored_x": 1},
    "ppd_pk": {"id": "A02", "name": "Population Density", "unit": "people per km²", "stored_x": 1},
    "urb_pc": {"id": "A03", "name": "Urban Extent", "unit": "percent cover", "stored_x": 1},
    "nli_ix": {"id": "A04", "name": "Nighttime Lights", "unit": "index value (x100)", "stored_x": 100},
    "rdd_mk": {"id": "A05", "name": "Road Density", "unit": "meters per km²", "stored_x": 1},
    "hft_ix": {"id": "A06", "name": "Human Footprint", "unit": "index value (x10)", "stored_x": 10},
    "gad_id": {"id": "A07", "name": "Global Administrative Areas", "unit": "ID number", "stored_x": 1},
    "gdp_ud": {"id": "A08", "name": "Gross Domestic Product (PPP)", "unit": "US dollars", "stored_x": 1},
    "hdi_ix": {"id": "A09", "name": "Human Development Index", "unit": "index value (x1000)", "stored_x": 1000},
}
NODATA = -9999


def field_info(column: str) -> dict[str, Any]:
    """Name, unit and stored multiplier for any BasinATLAS column (``pre_mm_uyr`` -> the ``pre_mm`` family)."""
    fam = column[:6]
    info = dict(FIELDS.get(fam, {"id": "", "name": fam, "unit": "", "stored_x": 1}))
    suffix = column[7:] if len(column) > 7 else ""
    info["scope"] = {"s": "sub-basin", "u": "upstream", "p": "pour point"}.get(suffix[:1], "")
    return info


def scale_value(column: str, value: float) -> tuple[float, str]:
    """Undo the catalog's storage multiplier: (value in real units, unit label)."""
    fam = column[:6]
    info = FIELDS.get(fam)
    if info is None:
        return float(value), ""
    unit = info["unit"].split(" (x")[0]
    val = float(value) / float(info["stored_x"] or 1)
    if fam == "pop_ct":
        val, unit = val * 1000.0, "people"
    return val, unit


@dataclass
class BasinsBuildReport:
    built_at: str
    n_basins: int
    files: dict[str, int]
    seconds: float
    attribution: str = ATTRIBUTION
    license: str = LICENSE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def basins_url(filename: str, repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/basins/{filename}"


# ── build (workflow) ────────────────────────────────────────────────────────


def _require_pyogrio():
    from aquascope.utils.imports import require

    return require("pyogrio", feature="basins build", group="basins")


def build_basins(
    gdb_path: str | Path,
    out_dir: str | Path,
    *,
    simplify_deg: float = 0.0005,
    batch: int = 50_000,
    max_features: int | None = None,
    write_fgb: bool = False,
) -> BasinsBuildReport:
    """Turn the BasinATLAS FileGDB into the Archive's ``basins/`` parquet files.

    Reads the level-12 layer in batches (about 1 M polygons) and streams the
    full attribute table into ``lev12_attributes.parquet`` (one row group per
    batch; the source is ordered by ``HYBAS_ID`` so row-group statistics prune
    lookups) while collecting the routing columns and a representative point
    per sub-basin into ``lev12_topology.parquet``. ``write_fgb`` also writes
    the simplified polygons as an indexed FlatGeobuf from Python (fine for
    tests and small extracts; the workflow streams it with ``ogr2ogr``
    instead, which needs no memory). ``max_features`` limits the read.
    """
    pyogrio = _require_pyogrio()
    from aquascope.utils.imports import require

    pa = require("pyarrow", feature="basins build", group="basins")
    pq = require("pyarrow.parquet", feature="basins build", group="basins")

    t0 = time.perf_counter()
    out = Path(out_dir) / "basins"
    out.mkdir(parents=True, exist_ok=True)
    info = pyogrio.read_info(str(gdb_path), layer=LAYER)
    total = int(info["features"]) if max_features is None else min(int(info["features"]), max_features)
    logger.info("BasinATLAS %s: %d features, %d fields", LAYER, total, len(info["fields"]))
    meta = {b"aquascope": json.dumps({"source": "BasinATLAS v1.0", "layer": LAYER, "license": LICENSE,
                                       "attribution": ATTRIBUTION}).encode()}

    topo_frames: list[pd.DataFrame] = []
    fgb_frames: list[Any] = []
    attr_path = out / f"lev{LEVEL:02d}_attributes.parquet"
    writer = None
    n_done = 0
    for start in range(0, total, batch):
        n = min(batch, total - start)
        gdf = pyogrio.read_dataframe(str(gdb_path), layer=LAYER, skip_features=start, max_features=n)
        gdf.columns = [c.lower().rstrip("_") if c.upper() in TOPOLOGY_COLUMNS else c for c in gdf.columns]
        for c in _ID_COLUMNS:
            if c in gdf.columns:
                gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0).astype("int64")
        cent = gdf.geometry.representative_point()
        topo = pd.DataFrame({c: gdf[c] for c in dict.fromkeys(x.lower().rstrip("_") for x in TOPOLOGY_COLUMNS)
                             if c in gdf.columns})
        topo["lat"] = cent.y.round(5).to_numpy()
        topo["lon"] = cent.x.round(5).to_numpy()
        topo_frames.append(topo)
        attrs = pd.DataFrame(gdf.drop(columns=[gdf.geometry.name])).sort_values("hybas_id")
        table = pa.Table.from_pandas(attrs, preserve_index=False)
        if writer is None:
            schema = table.schema.with_metadata({**(table.schema.metadata or {}), **meta})
            writer = pq.ParquetWriter(attr_path, schema, compression="zstd")
        writer.write_table(table.cast(writer.schema), row_group_size=20_000)
        if write_fgb:
            keep = [c for c in ("hybas_id", "next_down", "main_bas", "sub_area", "up_area", "pfaf_id")
                    if c in gdf.columns]
            slim = gdf[keep + [gdf.geometry.name]].copy()
            if simplify_deg:
                slim[gdf.geometry.name] = slim.geometry.simplify(simplify_deg, preserve_topology=True)
            fgb_frames.append(slim)
        n_done += n
        logger.info("  %d / %d", n_done, total)
    if writer is not None:
        writer.close()
    files: dict[str, int] = {attr_path.name: attr_path.stat().st_size} if attr_path.exists() else {}

    topo_df = pd.concat(topo_frames, ignore_index=True).sort_values("hybas_id").reset_index(drop=True)
    topo_table = pa.Table.from_pandas(topo_df, preserve_index=False)
    topo_table = topo_table.replace_schema_metadata({**(topo_table.schema.metadata or {}), **meta})
    p = out / f"lev{LEVEL:02d}_topology.parquet"
    pq.write_table(topo_table, p, compression="zstd", row_group_size=200_000)
    files[p.name] = p.stat().st_size

    if write_fgb and fgb_frames:
        import geopandas as gpd

        fgb_path = out / f"lev{LEVEL:02d}.fgb"
        all_fgb = gpd.GeoDataFrame(pd.concat(fgb_frames, ignore_index=True), crs=fgb_frames[0].crs)
        pyogrio.write_dataframe(all_fgb, str(fgb_path), driver="FlatGeobuf", layer_options={"SPATIAL_INDEX": "YES"})
        files[fgb_path.name] = fgb_path.stat().st_size

    report = BasinsBuildReport(
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), n_basins=int(len(topo_df)),
        files=files, seconds=round(time.perf_counter() - t0, 1),
    )
    (out / "build.json").write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    logger.info("basins built: %d sub-basins in %.0fs", report.n_basins, report.seconds)
    return report


# ── read side ───────────────────────────────────────────────────────────────


def _cache_path(name: str, repo_id: str) -> Path:
    from aquascope.archive.catalog import cache_dir

    return cache_dir() / f"{repo_id.replace('/', '__')}__basins__{name}"


def load_topology(
    *, repo_id: str = DEFAULT_REPO_ID, refresh: bool = False, path: str | Path | None = None
) -> pd.DataFrame:
    """The level-12 routing table (about 1 M rows), downloaded once a day into the cache."""
    from aquascope.archive.catalog import _download

    if path is None:
        path = _download(basins_url(f"lev{LEVEL:02d}_topology.parquet", repo_id),
                         _cache_path(f"lev{LEVEL:02d}_topology.parquet", repo_id), refresh)
    return pd.read_parquet(path)


class Topology:
    """Upstream/downstream navigation over the level-12 graph, built once from the topology frame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.set_index("hybas_id", drop=False)
        self.children: dict[int, list[int]] = defaultdict(list)
        for hid, nd in zip(df["hybas_id"].to_numpy(), df["next_down"].to_numpy()):
            if nd:
                self.children[int(nd)].append(int(hid))

    def upstream_ids(self, hybas_id: int, *, include_self: bool = True, limit: int = 200_000) -> list[int]:
        seen = {int(hybas_id)}
        order = [int(hybas_id)] if include_self else []
        q = deque([int(hybas_id)])
        while q and len(seen) < limit:
            for c in self.children.get(q.popleft(), ()):
                if c not in seen:
                    seen.add(c)
                    order.append(c)
                    q.append(c)
        return order

    def downstream_ids(self, hybas_id: int, *, limit: int = 5_000) -> list[int]:
        out: list[int] = []
        cur = int(hybas_id)
        while len(out) < limit:
            nd = int(self.df.at[cur, "next_down"]) if cur in self.df.index else 0
            if not nd:
                break
            out.append(nd)
            cur = nd
        return out


def sub_basin_at(
    lat: float, lon: float, *, repo_id: str = DEFAULT_REPO_ID, fgb_path: str | Path | None = None
) -> dict[str, Any] | None:
    """The level-12 sub-basin containing the point, from the indexed FlatGeobuf (one small range read).

    Returns ``{"hybas_id", "next_down", "main_bas", "sub_area", "up_area", "pfaf_id"}`` or None.
    Needs pyogrio and shapely (``basins`` extra).
    """
    pyogrio = _require_pyogrio()
    from shapely.geometry import Point

    src = str(fgb_path) if fgb_path else f"/vsicurl/{basins_url(f'lev{LEVEL:02d}.fgb', repo_id)}"
    d = 0.02
    gdf = pyogrio.read_dataframe(src, bbox=(lon - d, lat - d, lon + d, lat + d))
    if gdf.empty:
        return None
    gdf.columns = [c.lower() if c.upper() in TOPOLOGY_COLUMNS else c for c in gdf.columns]  # ogr2ogr keeps HYBAS_ID
    pt = Point(lon, lat)
    hit = gdf[gdf.contains(pt)]
    if hit.empty:
        hit = gdf.iloc[[gdf.distance(pt).argmin()]]
    row = hit.iloc[0]
    return {k: (int(row[k]) if k in ("hybas_id", "next_down", "main_bas", "pfaf_id") else float(row[k]))
            for k in ("hybas_id", "next_down", "main_bas", "sub_area", "up_area", "pfaf_id") if k in hit.columns}


class _HttpRangeFile:
    """A minimal seekable, read-only file over HTTP range requests (for pyarrow's ``PythonFile``).

    Lets ``ParquetFile`` read the footer and only the row groups it needs from
    the 250 MB attributes parquet on the Hub instead of downloading it all.
    """

    READ_AHEAD = 8 * 1024 * 1024  # column chunks of one row group are contiguous: fetch them in one go

    def __init__(self, url: str, timeout: float = 60):
        import urllib.request

        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._buf = b""
        self._buf_start = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            self._size = int(resp.headers.get("Content-Length") or 0)
        self.closed = False
        self.requests = 0

    def size(self) -> int:
        return self._size

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def _fetch(self, start: int, end: int) -> bytes:
        import urllib.request

        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        self.requests += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - fixed https host
            return resp.read()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        end = min(self._size, self._pos + n)
        bstart, bend = self._buf_start, self._buf_start + len(self._buf)
        if not (bstart <= self._pos and end <= bend):
            # refill: at least what was asked, read ahead when reading forward through a row group
            want_end = min(self._size, max(end, self._pos + self.READ_AHEAD))
            self._buf = self._fetch(self._pos, want_end - 1)
            self._buf_start = self._pos
        off = self._pos - self._buf_start
        data = self._buf[off: off + (end - self._pos)]
        self._pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def load_attributes(
    hybas_ids: list[int],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    path: str | Path | None = None,
    columns: list[str] | None = None,
    max_row_groups: int = 40,
) -> pd.DataFrame:
    """BasinATLAS attribute rows for the given sub-basins.

    Reads only the row groups whose ``hybas_id`` statistics cover the ids:
    from ``path`` when given, otherwise straight from the Hub through HTTP
    range requests (footer + a few row groups, not the whole file). Ids are
    spatially clustered, so a catchment usually sits in a handful of groups;
    when more than ``max_row_groups`` would be needed the nearest ones to the
    outlet (the largest id block) are read and the rest skipped.
    """
    from aquascope.utils.imports import require

    pa = require("pyarrow", feature="basins", group="archive")
    pq = require("pyarrow.parquet", feature="basins", group="archive")

    ids = sorted({int(x) for x in hybas_ids})
    if not ids:
        return pd.DataFrame(columns=["hybas_id"])
    remote = None
    if path is None:
        remote = _HttpRangeFile(basins_url(f"lev{LEVEL:02d}_attributes.parquet", repo_id))
        source: Any = pa.PythonFile(remote, mode="r")
    else:
        source = str(path)
    pf = pq.ParquetFile(source, pre_buffer=True)
    md = pf.metadata
    idx = pf.schema_arrow.get_field_index("hybas_id")
    wanted: list[int] = []
    lo, hi = ids[0], ids[-1]
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(idx).statistics
        if st is None or not st.has_min_max:
            wanted.append(g)
            continue
        gmin, gmax = int(st.min), int(st.max)
        if gmax < lo or gmin > hi:
            continue
        # any id inside this group's range?
        import bisect

        k = bisect.bisect_left(ids, gmin)
        if k < len(ids) and ids[k] <= gmax:
            wanted.append(g)
    if len(wanted) > max_row_groups:
        logger.info("basins: %d row groups needed, reading %d", len(wanted), max_row_groups)
        wanted = wanted[-max_row_groups:]
    if not wanted:
        return pd.DataFrame(columns=["hybas_id"])
    table = pf.read_row_groups(wanted, columns=columns)
    if remote is not None:
        logger.info("basins: %d row groups in %d range requests", len(wanted), remote.requests)
    df = table.to_pandas()
    return df[df["hybas_id"].isin(ids)].reset_index(drop=True)


def catchment_attributes(ids: list[int], attrs: pd.DataFrame, outlet: int | None = None) -> dict[str, Any]:
    """Catchment attributes for a set of sub-basins per :data:`ATTRIBUTE_GUIDE`.

    When the outlet row carries BasinATLAS's own upstream field it is used
    as is (``source: "basinatlas_upstream"``); otherwise the sub-basin field is
    aggregated over ``ids`` (area-weighted mean, sum, or outlet value).
    """
    if attrs.empty:
        return {}
    df = attrs.set_index("hybas_id")
    df = df[df.index.isin(ids)]
    if df.empty:
        return {}
    w = df["sub_area"].astype(float).clip(lower=0) if "sub_area" in df.columns else pd.Series(1.0, index=df.index)
    wsum = float(w.sum()) or 1.0
    if outlet is None:
        outlet = int(df["up_area"].astype(float).idxmax()) if "up_area" in df.columns else int(df.index[0])
    outlet = int(outlet)
    out: dict[str, Any] = {
        "n_sub_basins": int(len(df)),
        "area_km2": round(float(w.sum()), 1),
        "outlet_hybas_id": outlet,
    }
    if "up_area" in df.columns and outlet in df.index:
        out["upstream_area_km2"] = round(float(df.at[outlet, "up_area"]), 1)
    complete = len(df) == 1 or ("up_area" in df.columns and outlet in df.index
                                and abs(float(df.at[outlet, "up_area"]) - float(w.sum())) < 0.05 * float(w.sum()))
    for key, (s_field, u_field, unit, how, label) in ATTRIBUTE_GUIDE.items():
        val = None
        source = None
        has_u = u_field and u_field in df.columns and outlet in df.index
        if has_u and pd.notna(df.at[outlet, u_field]) and float(df.at[outlet, u_field]) > NODATA:
            val, source, field = float(df.at[outlet, u_field]), "basinatlas_upstream", u_field
        elif s_field in df.columns:
            col = pd.to_numeric(df[s_field], errors="coerce")
            col = col.mask(col <= NODATA)
            if col.isna().all():
                continue
            field = s_field
            if how == "area":
                val = float((col.fillna(0) * w).sum() / wsum)
                source = "area_weighted_mean" if len(df) > 1 else "sub_basin"
            elif how == "sum":
                val, source = float(col.fillna(0).sum()), "sum" if len(df) > 1 else "sub_basin"
            elif how == "max":
                val, source = float(col.max()), "max"
            else:
                val, source = float(col.get(outlet, col.iloc[0])), "outlet"
        if val is None:
            continue
        val, catalog_unit = scale_value(field, val)
        entry = {"value": round(val, 2), "unit": unit or catalog_unit, "label": label, "field": field, "source": source}
        if source == "area_weighted_mean" and not complete:
            entry["note"] = "aggregated over the sub-basins returned; the upstream set may be capped"
        out[key] = entry
    return out


def row_catchment_attributes(row: dict[str, Any] | pd.Series, *, scope: str = "upstream") -> dict[str, float]:
    """Flat ``{guide key: value}`` for one BasinATLAS row.

    ``scope="upstream"`` takes the upstream (``_u``/``_p``) field where it
    exists: the catchment closed at the row's sub-basin outlet.
    ``scope="local"`` takes the sub-basin's own (``_s``) field: right for a
    gauge that drains only a corner of its sub-basin. Real units (storage
    multipliers undone, ``NODATA`` dropped). Used to tabulate every gauged
    station's catchment for similarity search.
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()
    out: dict[str, float] = {}
    for key, (s_field, u_field, _unit, _how, _label) in ATTRIBUTE_GUIDE.items():
        val = None
        order = (u_field, s_field) if scope == "upstream" else (s_field, u_field)
        for f in order:
            if f and f in row and row[f] is not None:
                try:
                    v = float(row[f])
                except (TypeError, ValueError):
                    continue
                if not math.isnan(v) and v > NODATA:
                    val, field = v, f
                    break
        if val is None:
            continue
        out[key] = round(scale_value(field, val)[0], 4)
    return out


def describe_catchment(
    lat: float, lon: float, *, repo_id: str = DEFAULT_REPO_ID, upstream: bool = True, max_sub_basins: int = 20_000
) -> dict[str, Any]:
    """Which sub-basin a point sits in, what drains to it, and the catchment's HydroATLAS attributes.

    Runs entirely on the Archive's ``basins/`` files. ``upstream=False`` describes the local
    level-12 sub-basin only. Returns a JSON-safe dict with ``sub_basin``, ``upstream``
    (ids count and area) and ``attributes`` (see :data:`ATTRIBUTE_GUIDE`), plus licence
    and attribution.
    """
    sb = sub_basin_at(lat, lon, repo_id=repo_id)
    if sb is None:
        return {"latitude": lat, "longitude": lon,
                "error": "no BasinATLAS sub-basin contains this point (ocean, or outside coverage)"}
    ids = [sb["hybas_id"]]
    note = "local level-12 sub-basin only"
    if upstream:
        topo = Topology(load_topology(repo_id=repo_id))
        ids = topo.upstream_ids(sb["hybas_id"], limit=max_sub_basins)
        note = f"catchment upstream of the sub-basin containing the point ({len(ids)} level-12 sub-basins)"
        if len(ids) >= max_sub_basins:
            note += "; truncated at the sub-basin cap, attributes cover the nearest part of the basin"
    attrs = load_attributes(ids, repo_id=repo_id)
    return {
        "latitude": lat,
        "longitude": lon,
        "sub_basin": sb,
        "upstream": {"n_sub_basins": len(ids), "note": note},
        "attributes": catchment_attributes(ids, attrs, outlet=sb["hybas_id"]),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "methods": [_METHOD],
    }


_METHOD = {
    "name": "Catchment attributes from BasinATLAS (HydroATLAS v1.0)",
    "text": "Level-12 HydroBASINS sub-basins traced upstream through the NEXT_DOWN routing field; "
            "attributes aggregated as area-weighted means (or outlet / total values where the field is "
            "already an upstream aggregate).",
    "citation": ATTRIBUTION,
}

_SUB_BASIN_KEYS = ("hybas_id", "next_down", "main_bas", "sub_area", "up_area", "pfaf_id", "approximate")


def describe_catchment_from_row(
    lat: float | None,
    lon: float | None,
    sub_basin: dict[str, Any] | None,
    row: dict[str, Any] | None,
    *,
    n_upstream: int | None = None,
) -> dict[str, Any]:
    """:func:`describe_catchment` for a caller that has already found the sub-basin and read its row.

    The Explorer reads BasinATLAS with DuckDB and FlatGeobuf on its main
    thread, where pyogrio cannot run, and its Python worker cannot. It hands
    the sub-basin it found (``hybas_id``, ``up_area``, ``sub_area``, ...) and
    the outlet's raw attribute row here and gets the payload every other face
    gets, built by the same :func:`catchment_attributes`. The outlet row's
    upstream (``_u``) fields already describe the catchment closed at that
    outlet, so a single row gives the same attributes a full upstream walk
    aggregates to; ``n_upstream`` is the count of sub-basins the caller
    walked, for the note.
    """
    sb = {k: v for k, v in dict(sub_basin or {}).items() if k in _SUB_BASIN_KEYS and v is not None}
    plat = float(lat) if lat is not None else None
    plon = float(lon) if lon is not None else None
    if sb.get("hybas_id") is None:
        return {"latitude": plat, "longitude": plon,
                "error": "no BasinATLAS sub-basin contains this point (ocean, or outside coverage)"}
    hybas_id = int(sb["hybas_id"])
    sb["hybas_id"] = hybas_id
    frame = pd.DataFrame([{**dict(row or {}), "hybas_id": hybas_id}])
    for col in ("sub_area", "up_area"):
        if col not in frame.columns and sb.get(col) is not None:
            frame[col] = float(sb[col])
    n = int(n_upstream) if n_upstream else 1
    note = (f"catchment upstream of the sub-basin containing the point ({n} level-12 sub-basins); "
            "attributes from the outlet's upstream fields" if n > 1
            else "attributes from the outlet sub-basin's upstream fields")
    if sb.get("approximate"):
        note += "; nearest sub-basin, the point is not inside one"
    return {
        "latitude": plat,
        "longitude": plon,
        "sub_basin": sb,
        "upstream": {"n_sub_basins": n, "note": note},
        "attributes": catchment_attributes([hybas_id], frame, outlet=hybas_id),
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "methods": [_METHOD],
    }


__all__ = [
    "ATTRIBUTE_GUIDE", "ATTRIBUTION", "LICENSE", "Topology", "basins_url", "build_basins", "catchment_attributes",
    "describe_catchment", "describe_catchment_from_row", "load_attributes", "load_topology", "sub_basin_at",
]
