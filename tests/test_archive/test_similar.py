"""Nearest similar gauged basins over the station -> catchment table (#53, the practical half)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import similar
from aquascope.archive.basins import describe_catchment_from_row, row_catchment_attributes


def _table():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(60):
        rows.append({
            "source": "usgs" if i % 2 else "uk_ea", "station_id": f"S{i}", "hybas_id": 100 + i,
            "up_area": float(10 ** rng.uniform(1.5, 4.5)), "sub_area": 100.0, "attribute_scope": "upstream",
            "elevation_m": float(rng.uniform(20, 2500)), "slope_deg": float(rng.uniform(0.5, 25)),
            "precipitation_mm_yr": float(rng.uniform(300, 2500)), "aridity_index": float(rng.uniform(0.2, 3)),
            "temperature_c": float(rng.uniform(-2, 24)), "snow_cover_pct": float(rng.uniform(0, 60)),
            "forest_pct": float(rng.uniform(0, 90)), "cropland_pct": float(rng.uniform(0, 80)),
            "urban_pct": float(rng.uniform(0, 30)), "clay_pct": float(rng.uniform(10, 40)),
            "sand_pct": float(rng.uniform(20, 70)), "population_density": float(rng.uniform(0, 500)),
            "degree_of_regulation_pct": float(rng.uniform(0, 100)),
        })
    # a near-twin of the target and a far-away twin
    twin = dict(rows[0], station_id="TWIN", hybas_id=999, up_area=1000.0, elevation_m=300.0, slope_deg=3.0,
                precipitation_mm_yr=900.0, aridity_index=0.9, temperature_c=10.0, snow_cover_pct=5.0,
                forest_pct=30.0, cropland_pct=40.0, urban_pct=3.0, clay_pct=22.0, sand_pct=40.0,
                population_density=80.0, degree_of_regulation_pct=5.0)
    rows.append(twin)
    df = pd.DataFrame(rows)
    df["area_km2"] = df["up_area"]
    df["area_source"] = "basinatlas_up_area"
    return df


def _catalog(table):
    rng = np.random.default_rng(1)
    cat = []
    for r in table.to_dict("records"):
        lat, lon = float(rng.uniform(35, 60)), float(rng.uniform(-120, 10))
        if r["station_id"] == "TWIN":
            lat, lon = 51.5, -0.1
        cat.append({"source": r["source"], "station_id": r["station_id"], "name": f"Station {r['station_id']}",
                    "latitude": lat, "longitude": lon,
                    "variables": ["discharge"] if r["station_id"] != "S1" else ["water_level"],
                    "period_start": "1980-01-01", "period_end": None})
    return cat


TARGET = {"latitude": 51.4, "longitude": -0.3, "area_km2": 1050.0, "elevation_m": 310.0, "slope_deg": 3.2,
          "precipitation_mm_yr": 880.0, "aridity_index": 0.95, "temperature_c": 10.3, "snow_cover_pct": 4.0,
          "forest_pct": 28.0, "cropland_pct": 42.0, "urban_pct": 4.0, "clay_pct": 21.0, "sand_pct": 41.0,
          "population_density": 90.0, "degree_of_regulation_pct": 4.0}


def test_similarity_ranks_the_twin_first_and_explains_it():
    tab = _table()
    cat = _catalog(tab)
    res = similar.similar_basins(TARGET, k=5, method="similarity", table=tab, catalog=cat)
    assert res["stations"][0]["station_id"] == "TWIN"
    assert res["k"] == 5 and res["n_candidates"] == 60  # S1 has no discharge and is excluded
    top = res["stations"][0]
    assert top["features"]["precipitation"] == {"station": 900.0, "target": 880.0}
    assert top["similarity_distance"] < res["stations"][1]["similarity_distance"]
    assert set(res["features_used"]) == set(similar.FEATURES)
    assert res["methods"][0]["citation"].startswith("Bloeschl")


def test_proximity_and_combined_use_the_ground_distance():
    tab = _table()
    cat = _catalog(tab)
    prox = similar.similar_basins(TARGET, k=3, method="proximity", table=tab, catalog=cat)
    assert prox["stations"][0]["station_id"] == "TWIN" and prox["stations"][0]["distance_km"] < 20
    comb = similar.similar_basins(TARGET, k=3, method="combined", table=tab, catalog=cat)
    assert comb["stations"][0]["station_id"] == "TWIN" and comb["stations"][0]["distance_km"] is not None
    with pytest.raises(ValueError):
        similar.similar_basins(TARGET, method="magic", table=tab, catalog=cat)


def test_filters_sources_exclusion_and_missing_values():
    tab = _table()
    cat = _catalog(tab)
    only_usgs = similar.similar_basins(TARGET, k=4, sources=["usgs"], table=tab, catalog=cat)
    assert all(s["source"] == "usgs" for s in only_usgs["stations"])
    excl = similar.similar_basins(TARGET, k=3, exclude=("uk_ea", "TWIN"), table=tab, catalog=cat)
    assert all(s["station_id"] != "TWIN" for s in excl["stations"])
    # a station with a missing attribute is penalised, not dropped or crashed on
    tab2 = tab.copy()
    tab2.loc[tab2["station_id"] == "TWIN", "forest_pct"] = np.nan
    res = similar.similar_basins(TARGET, k=2, table=tab2, catalog=cat)
    assert res["stations"][0]["station_id"] == "TWIN" and "forest" not in res["stations"][0]["features"]
    # a target missing a feature just uses the rest
    t2 = {k: v for k, v in TARGET.items() if k != "snow_cover_pct"}
    res = similar.similar_basins(t2, k=1, table=tab, catalog=cat)
    assert "snow" not in res["features_used"] and res["stations"][0]["station_id"] == "TWIN"
    # nothing left after filters
    assert similar.similar_basins(TARGET, sources=["nope"], table=tab, catalog=cat)["stations"] == []


def test_target_from_describe_catchment_shape():
    desc_like = {"latitude": 51.4, "longitude": -0.3, "sub_basin": {"hybas_id": 1, "up_area": 1050.0},
                 "attributes": {"upstream_area_km2": 1050.0, "elevation_m": {"value": 310.0, "unit": "m"},
                                "precipitation_mm_yr": {"value": 880.0}, "aridity_index": {"value": 0.95}}}
    vals = similar._target_values(desc_like)
    assert vals["log_area"] == 1050.0 and vals["elevation"] == 310.0 and vals["snow"] is None
    flat = {"area_km2": 20.0, "up_area": 900.0, "elevation_m": 100.0}
    assert similar._target_values(flat)["log_area"] == 20.0  # a station row: its own area, not the sub-basin's


def test_point_donors_rank_by_outlet_area_in_single_polygon_description():
    sb = {"hybas_id": 1, "up_area": 1000.0, "sub_area": 20.0}
    desc = describe_catchment_from_row(51.4, -0.3, sb, sb)
    assert desc["attributes"]["area_km2"] == 20.0
    tab = _table()
    tab.loc[tab["station_id"] == "S0", "area_km2"] = 20.0
    out = similar.similar_for_point(51.4, -0.3, k=1, method="similarity", desc=desc,
                                    features=["log_area"], table=tab, catalog=_catalog(tab))
    best = out["stations"][0]
    assert best["station_id"] == "TWIN"
    assert best["features"]["log_area"] == {"station": 1000.0, "target": 1000.0}


@pytest.mark.parametrize(("attrs", "sub_basin", "expected"), [
    ({"area_km2": 20}, {"up_area": 900}, 900),
    ({"area_km2": 20, "upstream_area_km2": 900}, {}, 900),
    ({"area_km2": 20, "upstream_area_km2": float("nan")}, {"up_area": 900}, 900),
    ({"area_km2": {"value": 20, "unit": "km²"}}, {"up_area": 0}, 20),
    ({"area_km2": -9999}, {"up_area": float("inf")}, None),
])
def test_point_area_uses_a_valid_fallback(attrs, sub_basin, expected):
    assert similar._target_values({"attributes": attrs, "sub_basin": sub_basin})["log_area"] == expected


def test_similar_for_station_excludes_itself(monkeypatch):
    tab = _table()
    cat = _catalog(tab)
    res = similar.similar_for_station("uk_ea", "TWIN", k=3, method="similarity", table=tab, catalog=cat)
    assert res["source"] == "uk_ea" and all(s["station_id"] != "TWIN" for s in res["stations"])
    assert "error" in similar.similar_for_station("usgs", "nope", table=tab, catalog=cat)


def test_row_catchment_attributes_takes_upstream_fields():
    row = {"ele_mt_sav": 100, "ele_mt_uav": 400, "pre_mm_syr": 700, "pre_mm_uyr": 750, "run_mm_syr": 20,
           "slp_dg_uav": 35, "pop_ct_usu": 12.5, "dor_pc_pva": 120, "soc_th_uav": -9999, "soc_th_sav": 30}
    out = row_catchment_attributes(row)
    assert out["elevation_m"] == 400 and out["precipitation_mm_yr"] == 750 and out["runoff_mm_yr"] == 20
    assert out["slope_deg"] == 3.5 and out["population"] == 12500 and out["degree_of_regulation_pct"] == 12.0
    assert out["soil_organic_carbon_t_ha"] == 30  # NODATA upstream falls back to the local value
    local = row_catchment_attributes(row, scope="local")
    assert local["elevation_m"] == 100 and local["precipitation_mm_yr"] == 700 and local["slope_deg"] == 3.5


def test_assign_station_catchments_on_a_synthetic_layer(tmp_path):
    pytest.importorskip("pyogrio")
    gpd = pytest.importorskip("geopandas")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely.geometry import box

    layer = gpd.GeoDataFrame(
        {"HYBAS_ID": [1, 2], "UP_AREA": [50.0, 900.0], "SUB_AREA": [50.0, 100.0], "NEXT_DOWN": [2, 0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326",
    )
    fgb = tmp_path / "lev12.fgb"
    layer.to_file(fgb, driver="FlatGeobuf")
    ap = tmp_path / "attrs.parquet"
    catalog = [{"source": "usgs", "station_id": "A", "latitude": 0.5, "longitude": 0.5},
               {"source": "usgs", "station_id": "B", "latitude": 0.5, "longitude": 1.5,
                "extra": {"catchment_area_km2": 850.0}},
               {"source": "usgs", "station_id": "C", "latitude": 0.6, "longitude": 1.6,
                "extra": {"catchment_area_km2": 12.0}},  # a creek inside sub-basin 2
               {"source": "usgs", "station_id": "SEA", "latitude": 5.0, "longitude": 5.0}]
    attrs = pd.DataFrame([{"hybas_id": 1, "ele_mt_uav": 500, "pre_mm_uyr": 1000, "for_pc_use": 40},
                          {"hybas_id": 2, "ele_mt_uav": 200, "ele_mt_sav": 120, "pre_mm_uyr": 800, "pre_mm_syr": 650,
                           "for_pc_use": 10, "for_pc_sse": 55}])
    pq.write_table(pa.Table.from_pandas(attrs, preserve_index=False), ap)
    out = similar.assign_station_catchments(catalog, fgb, ap, tmp_path / "station_catchments.parquet")
    assert sorted(out["station_id"]) == ["A", "B", "C"]
    by = out.set_index("station_id")
    assert by.loc["A", "hybas_id"] == 1 and by.loc["A", "up_area"] == 50.0 and by.loc["A", "elevation_m"] == 500
    assert by.loc["A", "area_km2"] == 50.0 and by.loc["A", "area_source"] == "basinatlas_up_area"
    assert by.loc["A", "attribute_scope"] == "upstream" and by.loc["A", "forest_pct"] == 40
    # B closes most of sub-basin 2's catchment: agency area, upstream attributes
    assert by.loc["B", "area_km2"] == 850.0 and by.loc["B", "area_source"] == "agency"
    assert by.loc["B", "attribute_scope"] == "upstream" and by.loc["B", "elevation_m"] == 200
    # C drains 12 km2 of a 900 km2 catchment: local sub-basin attributes, its own area
    assert by.loc["C", "attribute_scope"] == "local" and by.loc["C", "area_km2"] == 12.0
    assert by.loc["C", "elevation_m"] == 120 and by.loc["C", "forest_pct"] == 55
    assert by.loc["C", "precipitation_mm_yr"] == 650
    back = pd.read_parquet(tmp_path / "station_catchments.parquet")
    assert len(back) == 3 and "precipitation_mm_yr" in back.columns
