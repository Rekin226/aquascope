"""Regionalisation of flow signatures (#53, the predictive half): donor signatures, transfer, leave-one-out skill."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import regionalize as rg
from tests.test_archive.test_similar import TARGET, _catalog, _table


def _daily_flow(area_km2: float, precip: float, years: int = 12, seed: int = 0) -> pd.Series:
    """A synthetic daily discharge (m3/s) whose mean runoff scales with precipitation and area."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=365 * years, freq="D")
    doy = idx.dayofyear.to_numpy()
    mm = 0.4 * precip / 365.25 * (1 + 0.6 * np.sin(2 * np.pi * (doy - 60) / 365.25))
    mm = mm * np.exp(rng.normal(0, 0.5, len(idx)))
    return pd.Series(mm * area_km2 / 86.4, index=idx)


def _signatures(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(table.to_dict("records")):
        q = _daily_flow(r["area_km2"], r["precipitation_mm_yr"], seed=i)
        sig = rg.station_signature(q, r["area_km2"], r["precipitation_mm_yr"])
        assert sig is not None
        rows.append({"source": r["source"], "station_id": r["station_id"], "area_km2": r["area_km2"],
                     "area_source": r["area_source"], "hybas_id": r["hybas_id"], **sig})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def world():
    tab = _table()
    return tab, _catalog(tab), _signatures(tab)


def test_station_signature_units_and_thresholds():
    q = _daily_flow(1000.0, 900.0, years=12)
    sig = rg.station_signature(q, 1000.0, 900.0)
    assert sig["n_years"] >= 11 and sig["start"] == "2000-01-01"
    # mean flow in mm/d ~ 0.4 * 900 / 365.25 ~ 1 mm/d (with the lognormal noise inflating the mean a little)
    assert 0.8 < sig["q_mean_mm"] < 1.6 and sig["q95_mm"] < sig["q_median_mm"] < sig["q05_mm"] < sig["q_annual_max_mm"]
    assert 0.3 < sig["runoff_ratio"] < 0.7 and 0 <= sig["baseflow_index"] <= 1 and sig["zero_flow_fraction"] == 0
    assert sig["seasonality_index"] > 0 and sig["flashiness_index"] > 0
    # too short, no area, or nothing left -> None
    assert rg.station_signature(q[: 365 * 3], 1000.0) is None
    assert rg.station_signature(q, 0.0) is None
    assert rg.station_signature(q, 1000.0, min_years=5)["n_years"] >= 11
    # NaN-only fields are None so the parquet/JSON stays clean
    assert rg.station_signature(q, 1000.0)["runoff_ratio"] is None


def test_compute_station_signatures_from_bundles(world):
    tab, _cat, _sig = world
    sub = tab.head(3)
    frames = []
    for i, r in enumerate(sub.to_dict("records")):
        q = _daily_flow(r["area_km2"], r["precipitation_mm_yr"], seed=i)
        frames.append(pd.DataFrame({"station_id": r["station_id"], "date": q.index, "value": q.to_numpy()}))
    # a station without a catchment row is skipped, one with a short record too
    frames.append(pd.DataFrame({"station_id": "GHOST", "date": q.index, "value": q.to_numpy()}))
    short = _daily_flow(500.0, 800.0, years=2)
    frames.append(pd.DataFrame({"station_id": "S5", "date": short.index, "value": short.to_numpy()}))
    bundles = {"uk_ea": pd.concat([f for f in frames if f["station_id"].iloc[0] in ("S0", "S2", "GHOST", "S5")]),
               "usgs": pd.concat([f for f in frames if f["station_id"].iloc[0] == "S1"]), "empty": pd.DataFrame()}
    out = rg.compute_station_signatures(bundles, tab)
    assert set(out["station_id"]) == {"S0", "S1", "S2"} and {"q_mean_mm", "hybas_id", "area_km2"} <= set(out.columns)
    assert out.loc[out["station_id"] == "S1", "source"].iloc[0] == "usgs"


def test_similarity_transfer_recovers_the_twin_and_carries_a_band(world):
    tab, cat, sig = world
    res = rg.regionalize(TARGET, k=5, method="similarity", signatures=sig, table=tab, catalog=cat)
    est = res["estimates"]
    assert res["method"] == "similarity" and res["similarity"]["donors"][0]["station_id"] == "TWIN"
    assert set(est) >= {"q_mean_mm", "q95_mm", "baseflow_index", "runoff_ratio"}
    twin = sig[sig["station_id"] == "TWIN"].iloc[0]
    e = est["q_mean_mm"]
    assert e["low"] <= e["value"] <= e["high"] and e["unit"] == "mm/d" and e["n_donors"] == 5
    assert e["donor_min"] <= twin["q_mean_mm"] <= e["donor_max"]
    # the target's precipitation (880 mm) implies ~1 mm/d with runoff ratio 0.4: the weighted mean lands nearby
    assert 0.5 < e["value"] < 2.5
    assert res["methods"][0]["citation"].startswith("Bloeschl")
    # exclude drops the twin from the donor pool
    res2 = rg.regionalize(TARGET, k=5, method="similarity", signatures=sig, table=tab, catalog=cat,
                          exclude=("uk_ea", "TWIN"))
    assert all(d["station_id"] != "TWIN" for d in res2["similarity"]["donors"])


def test_point_compares_five_and_ten_without_changing_the_requested_transfer(world, monkeypatch, capsys):
    from argparse import Namespace

    from aquascope.cli import cmd_basins

    tab, cat, sig = world
    kw = dict(table=tab, catalog=cat, signatures=sig)
    desc = {"attributes": TARGET}
    point = rg.regionalize_point(1, 2, k=3, desc=desc, skill=None, **kw)
    target = {"latitude": 1, "longitude": 2, **desc}
    reference = {k: rg.regionalize(target, k=k, **kw) for k in (3, 5, 10)}
    assert point["similarity"] == reference[3]["similarity"]
    comparison = point["donor_count_sensitivity"]
    rows = comparison["estimates"]
    donors = comparison["donors"]
    assert [(d["source"], d["station_id"], d["score"]) for d in donors] == [
        (d["source"], d["station_id"], d["score"]) for d in reference[10]["similarity"]["donors"]]
    assert set(rows) == {"q_mean_mm", "q95_mm", "q05_mm"}
    for name, row in rows.items():
        assert (row["n5"], row["n10"]) == (5, 10)
        assert row["k5"] == reference[5]["estimates"][name]["value"]
        assert row["k10"] == reference[10]["estimates"][name]["value"]
        assert row["change_pct"] == pytest.approx(100 * (row["k10"] / row["k5"] - 1), abs=0.1)
        weights = [1 / (d["score"] + 0.05) for d in donors]
        values = np.log(np.clip([d["signatures"][name] for d in donors], 1e-3, None))
        assert row["k10"] == round(np.exp(np.average(values, weights=weights)), 4)
    off = rg.regionalize_point(1, 2, k=3, desc=desc, skill=None, compare_donor_counts=False, **kw)
    assert "donor_count_sensitivity" not in off and off["similarity"] == point["similarity"]
    no_features = rg.regionalize_point(1, 2, desc={}, skill=None, **kw)
    assert "donor_count_sensitivity" not in no_features

    monkeypatch.setattr(rg, "regionalize_point", lambda *args, **kwargs: point)
    cmd_basins(Namespace(basins_cmd="regionalize", lat=1, lon=2, k=3, method="similarity", json=False))
    output = capsys.readouterr().out
    assert "5 vs 10 donors" in output and "Q95" in output


def test_donor_sensitivity_requires_complete_sets_per_signature(world):
    tab, cat, sig = world
    kw = dict(desc={"attributes": TARGET}, skill=None)
    few = rg.regionalize_point(1, 2, table=tab.head(6), catalog=cat, signatures=sig, **kw)
    row = few["donor_count_sensitivity"]["estimates"]["q95_mm"]
    assert row["n5"] == 5 and row["n10"] == len(few["similarity"]["donors"]) < 10 and row["k5"] is not None
    assert row["k10"] is None and row["change_pct"] is None

    ranked = rg.regionalize_point(1, 2, table=tab, catalog=cat, signatures=sig, **kw)
    first = ranked["similarity"]["donors"][0]
    sig = sig.copy()
    first_row = (sig.source == first["source"]) & (sig.station_id == first["station_id"])
    sig.loc[first_row, "q95_mm"] = np.nan
    sig.loc[first_row, "q05_mm"] = -np.inf
    partial = rg.regionalize_point(1, 2, method="both", table=tab, catalog=cat, signatures=sig, **kw)
    comparison = partial["donor_count_sensitivity"]
    row = comparison["estimates"]["q95_mm"]
    assert (row["n5"], row["n10"]) == (4, 9) and row["k5"] is None and row["k10"] is None
    assert comparison["estimates"]["q05_mm"]["n10"] == 9
    assert comparison["donors"][0]["signatures"]["q05_mm"] is None
    assert comparison["estimates"]["q_mean_mm"]["change_pct"] is not None
    assert partial["estimates"]["q05_mm"]["n_donors"] == 9
    assert json.dumps(partial, allow_nan=False)
    ridge = rg.regionalize_point(1, 2, method="regression", table=tab, catalog=cat, signatures=sig, **kw)
    assert "donor_count_sensitivity" not in ridge


def test_regression_transfer_tracks_precipitation(world):
    tab, cat, sig = world
    wet = dict(TARGET, precipitation_mm_yr=2200.0)
    dry = dict(TARGET, precipitation_mm_yr=400.0)
    rw = rg.regionalize(wet, method="regression", signatures=sig, table=tab, catalog=cat)
    rd = rg.regionalize(dry, method="regression", signatures=sig, table=tab, catalog=cat)
    assert rw["regression"]["n_fit"] == len(tab) and "q_mean_mm" in rw["estimates"]
    assert rw["estimates"]["q_mean_mm"]["value"] > rd["estimates"]["q_mean_mm"]["value"] * 2
    assert rw["estimates"]["q_mean_mm"]["residual_sd"] > 0
    both = rg.regionalize(TARGET, method="both", signatures=sig, table=tab, catalog=cat)
    assert "similarity" in both and "regression" in both and both["estimates"] == both["similarity"]["estimates"]
    with pytest.raises(ValueError):
        rg.regionalize(TARGET, method="magic", signatures=sig, table=tab, catalog=cat)


def test_regionalize_handles_an_empty_or_disjoint_pool(world):
    tab, cat, sig = world
    assert "error" in rg.regionalize(TARGET, signatures=sig.iloc[0:0], table=tab, catalog=cat)
    other = sig.assign(station_id=lambda d: "X" + d["station_id"])
    assert "error" in rg.regionalize(TARGET, signatures=other, table=tab, catalog=cat)


def test_loo_skill_reports_per_signature_scores(world):
    tab, cat, sig = world
    skill = rg.loo_skill(sig, tab, cat, k=5, max_stations=20)
    assert skill["n_stations"] == 20 and skill["n_signature_stations"] == len(sig) and skill["k"] == 5
    for m in ("similarity", "regression"):
        per = skill["methods"][m]
        assert "q_mean_mm" in per and per["q_mean_mm"]["n"] == 20 and per["q_mean_mm"]["space"] == "log"
        assert -5 < per["q_mean_mm"]["nse"] <= 1 and 0 <= per["q_mean_mm"]["median_ape"] < 5
    # mean flow is driven by precipitation, which is a feature: the regression should have some skill
    assert skill["methods"]["regression"]["q_mean_mm"]["r2"] > 0.3
    json.dumps(skill)  # serialisable as published


def test_regionalize_point_wraps_describe_catchment(monkeypatch, world):
    tab, cat, sig = world
    monkeypatch.setattr("aquascope.archive.basins.describe_catchment",
                        lambda lat, lon, **kw: {"sub_basin": {"hybas_id": 1, "up_area": 1050.0},
                                                "attributes": {k: {"value": v} for k, v in TARGET.items()
                                                               if k not in ("latitude", "longitude")}})
    res = rg.regionalize_point(51.4, -0.3, k=5, signatures=sig, table=tab, catalog=cat,
                               skill={"methods": {"similarity": {"q_mean_mm": {"nse": 0.5, "median_ape": 0.3}}}})
    assert res["latitude"] == 51.4 and res["sub_basin"]["hybas_id"] == 1 and "run_at" in res
    assert res["skill"]["by_signature"]["q_mean_mm"]["nse"] == 0.5 and "q_mean_mm" in res["estimates"]
    monkeypatch.setattr("aquascope.archive.basins.describe_catchment", lambda lat, lon, **kw: {"error": "sea"})
    assert rg.regionalize_point(0, 0, skill=None)["error"] == "sea"


def test_urls():
    assert rg.signatures_url().endswith("basins/station_signatures.parquet")
    assert rg.skill_url("a/b") == "https://huggingface.co/datasets/a/b/resolve/main/basins/regionalization_skill.json"


def test_the_point_paths_take_a_description_already_in_hand(world):
    """The browser builds describe_catchment's result itself and hands the tables over: no file is opened."""
    from unittest.mock import patch

    from aquascope.archive import similar as sm

    tab, cat, sig = world
    row = tab.iloc[-1]  # the twin
    desc = {"attributes": {k: {"value": float(row[k])} for k in (
        "elevation_m", "slope_deg", "precipitation_mm_yr", "aridity_index", "temperature_c", "snow_cover_pct",
        "forest_pct", "cropland_pct", "urban_pct", "clay_pct", "sand_pct", "population_density",
        "degree_of_regulation_pct")}, "sub_basin": {"hybas_id": 999, "up_area": 1000.0}}
    with patch("aquascope.archive.basins.describe_catchment", side_effect=AssertionError("must not be called")):
        near = sm.similar_for_point(TARGET["latitude"], TARGET["longitude"], k=5, method="similarity", desc=desc,
                                    table=tab, catalog=cat)
        est = rg.regionalize_point(TARGET["latitude"], TARGET["longitude"], k=5, method="similarity", desc=desc,
                                   table=tab, signatures=sig, catalog=cat, skill=None)
    assert near["stations"] and near["sub_basin"]["hybas_id"] == 999 and near["k"] == 5
    assert est["estimates"]["q_mean_mm"]["value"] > 0 and est["sub_basin"]["hybas_id"] == 999
