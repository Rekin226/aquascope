"""BasinATLAS in the Archive: build from a FileGDB-like layer, then point -> sub-basin -> upstream -> attributes."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from aquascope.archive import basins  # noqa: E402

# A tiny river: four level-12 sub-basins. 1 and 2 drain into 3, 3 drains into 4 (the outlet, coastal).
TOPO = pd.DataFrame({
    "hybas_id": [1, 2, 3, 4, 9],
    "next_down": [3, 3, 4, 0, 0],
    "next_sink": [4, 4, 4, 4, 9],
    "main_bas": [4, 4, 4, 4, 9],
    "sub_area": [10.0, 20.0, 30.0, 40.0, 5.0],
    "up_area": [10.0, 20.0, 60.0, 100.0, 5.0],
    "pfaf_id": [111, 112, 113, 114, 200],
    "endo": [0, 0, 0, 0, 0],
    "coast": [0, 0, 0, 1, 1],
    "order": [1, 1, 2, 3, 1],
    "lat": [1.5, 1.5, 0.5, 0.5, 5.5],
    "lon": [0.5, 1.5, 0.5, 1.5, 5.5],
})
ATTRS = pd.DataFrame({
    "hybas_id": [1, 2, 3, 4, 9],
    "sub_area": [10.0, 20.0, 30.0, 40.0, 5.0],
    "up_area": [10.0, 20.0, 60.0, 100.0, 5.0],
    "ele_mt_sav": [1000, 800, 400, 100, 50],
    "pre_mm_syr": [2000, 1500, 1000, 800, 600],
    "tmp_dc_syr": [50, 80, 120, 150, 200],
    "ari_ix_sav": [150, 120, 90, 70, 60],
    "for_pc_sse": [90, 60, 30, 10, 0],
    "pop_ct_ssu": [100, 200, 3000, 40000, 5],
    "dis_m3_pyr": [0.5, 1.0, 3.0, 5.0, 0.1],
    "dor_pc_pva": [0, 0, 5, 12, 0],
})


def test_topology_walks_upstream_and_downstream():
    topo = basins.Topology(TOPO)
    assert sorted(topo.upstream_ids(4)) == [1, 2, 3, 4]
    assert topo.upstream_ids(3, include_self=False) == [1, 2]
    assert topo.upstream_ids(1) == [1] and topo.upstream_ids(9) == [9]
    assert topo.downstream_ids(1) == [3, 4] and topo.downstream_ids(4) == []
    assert len(topo.upstream_ids(4, limit=2)) == 2  # the cap holds


def test_catchment_attributes_aggregate_per_guide():
    ids = basins.Topology(TOPO).upstream_ids(4)
    out = basins.catchment_attributes(ids, ATTRS, outlet=4)
    assert out["n_sub_basins"] == 4 and out["area_km2"] == 100.0 and out["upstream_area_km2"] == 100.0
    # area-weighted mean elevation: (10*1000 + 20*800 + 30*400 + 40*100) / 100 = 420
    assert out["elevation_m"]["value"] == 420.0 and out["elevation_m"]["unit"] == "m"
    assert out["temperature_c"]["value"] == pytest.approx((10 * 50 + 20 * 80 + 30 * 120 + 40 * 150) / 100 / 10)
    assert out["aridity_index"]["value"] == pytest.approx((10 * 150 + 20 * 120 + 30 * 90 + 40 * 70) / 100 / 100)
    assert out["population"]["value"] == 43_300_000.0 and out["population"]["source"] == "sum"  # stored in thousands
    assert out["discharge_m3s"]["value"] == 5.0  # outlet value
    assert out["degree_of_regulation_pct"]["value"] == 1.2  # outlet value; DOR is stored x10
    assert "glacier_pct" not in out  # field absent: skipped, not invented
    # the local sub-basin only
    local = basins.catchment_attributes([1], ATTRS, outlet=1)
    assert local["n_sub_basins"] == 1 and local["forest_pct"]["value"] == 90.0
    assert basins.catchment_attributes([42], ATTRS) == {}


def test_load_attributes_prunes_by_id(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = tmp_path / "attrs.parquet"
    pq.write_table(pa.Table.from_pandas(ATTRS, preserve_index=False), p, row_group_size=2)
    df = basins.load_attributes([2, 4], path=p, columns=["hybas_id", "ele_mt_sav"])
    assert sorted(df["hybas_id"]) == [2, 4] and list(df.columns) == ["hybas_id", "ele_mt_sav"]


def test_build_and_lookup_roundtrip(tmp_path, monkeypatch):
    """Synthetic BasinATLAS layer (a FileGDB stand-in in GeoPackage) -> build_basins -> sub_basin_at + describe."""
    pytest.importorskip("pyogrio")
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    # squares: 1 = (0,1)-(1,2), 2 = (1,1)-(2,2), 3 = (0,0)-(1,1), 4 = (1,0)-(2,1), 9 = (5,5)-(6,6)
    geoms = [box(0, 1, 1, 2), box(1, 1, 2, 2), box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)]
    layer = gpd.GeoDataFrame(
        {c.upper(): TOPO[c] for c in ("hybas_id", "next_down", "next_sink", "main_bas", "sub_area", "up_area",
                                       "pfaf_id", "endo", "coast", "order")}
        | {k: ATTRS[k] for k in ("ele_mt_sav", "pre_mm_syr", "tmp_dc_syr", "for_pc_sse", "pop_ct_ssu")},
        geometry=geoms, crs="EPSG:4326",
    )
    src = tmp_path / "BasinATLAS_v10.gpkg"
    layer.to_file(src, layer=basins.LAYER, driver="GPKG")

    report = basins.build_basins(src, tmp_path / "archive", batch=2, write_fgb=True)
    assert report.n_basins == 5
    assert set(report.files) >= {"lev12_topology.parquet", "lev12_attributes.parquet", "lev12.fgb"}
    out = tmp_path / "archive" / "basins"
    topo = pd.read_parquet(out / "lev12_topology.parquet")
    assert list(topo["hybas_id"]) == [1, 2, 3, 4, 9] and topo.loc[0, "lat"] == pytest.approx(1.5)
    attrs = pd.read_parquet(out / "lev12_attributes.parquet")
    assert "ele_mt_sav" in attrs.columns and "hybas_id" in attrs.columns and len(attrs) == 5
    build = json.loads((out / "build.json").read_text())
    assert build["license"] == "CC-BY-4.0" and "Linke" in build["attribution"]

    # point lookups against the local FlatGeobuf
    sb = basins.sub_basin_at(1.5, 0.5, fgb_path=out / "lev12.fgb")
    assert sb["hybas_id"] == 1 and sb["next_down"] == 3 and sb["up_area"] == 10.0
    assert basins.sub_basin_at(20, 20, fgb_path=out / "lev12.fgb") is None

    # describe_catchment end to end on the local files
    orig = basins.sub_basin_at
    monkeypatch.setattr(basins, "sub_basin_at", lambda lat, lon, **kw: orig(lat, lon, fgb_path=out / "lev12.fgb"))
    monkeypatch.setattr(basins, "load_topology", lambda **kw: topo)
    monkeypatch.setattr(basins, "load_attributes", lambda ids, **kw: attrs[attrs["hybas_id"].isin(ids)])
    res = basins.describe_catchment(0.5, 1.5)  # inside sub-basin 4, the outlet
    assert res["sub_basin"]["hybas_id"] == 4 and res["upstream"]["n_sub_basins"] == 4
    assert res["attributes"]["elevation_m"]["value"] == 420.0 and res["license"] == "CC-BY-4.0"
    assert res["methods"][0]["name"].startswith("Catchment attributes")
    local = basins.describe_catchment(1.5, 1.5, upstream=False)
    assert local["upstream"]["n_sub_basins"] == 1 and local["attributes"]["forest_pct"]["value"] == 60.0



def test_describe_catchment_from_row_builds_the_same_payload_from_what_a_page_read():
    """The Explorer finds the sub-basin and reads its row itself; the package turns them into the payload."""
    from aquascope.archive.basins import ATTRIBUTION, describe_catchment_from_row

    sub_basin = {"hybas_id": 2120000010, "next_down": 2120000020, "up_area": 9948.3, "sub_area": 120.5,
                 "approximate": False, "geometry": "must not survive"}
    row = {"hybas_id": 2120000010, "ele_mt_uav": 120, "ele_mt_sav": 40, "pre_mm_uyr": 700, "dor_pc_pva": 50,
           "dis_m3_pyr": 65.0, "for_pc_use": -9999}
    out = describe_catchment_from_row(51.415, -0.308, sub_basin, row, n_upstream=311)
    assert out["latitude"] == 51.415 and out["sub_basin"]["hybas_id"] == 2120000010
    assert "geometry" not in out["sub_basin"]
    attrs = out["attributes"]
    assert attrs["upstream_area_km2"] == 9948.3
    assert attrs["elevation_m"]["value"] == 120 and attrs["elevation_m"]["source"] == "basinatlas_upstream"
    assert attrs["precipitation_mm_yr"]["value"] == 700
    assert attrs["degree_of_regulation_pct"]["value"] == 5.0, "stored x10"
    assert "forest_pct" not in attrs, "NODATA is dropped, not read as -9999 %"
    assert out["upstream"]["n_sub_basins"] == 311 and out["attribution"] == ATTRIBUTION
    assert out["methods"][0]["citation"] == ATTRIBUTION

    missing = describe_catchment_from_row(0.0, 0.0, None, None)
    assert "error" in missing and missing["latitude"] == 0.0
