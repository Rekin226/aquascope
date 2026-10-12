"""Caravan-format export from the Archive: indices match Caravan's own code, the layout validates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope.archive import caravan


def _forcing_frame(days=365 * 3, seed=1, start="2000-01-01"):
    idx = pd.date_range(start, periods=days, freq="D", name="date")
    rng = np.random.default_rng(seed)
    doy = idx.dayofyear.to_numpy()
    p = np.where(rng.random(days) < 0.35, rng.gamma(1.5, 6.0, days), 0.0)
    t = 8 + 12 * np.sin((doy - 100) / 365 * 2 * np.pi) + rng.normal(0, 2, days)
    pet = np.clip(1.5 + 2.5 * np.sin((doy - 100) / 365 * 2 * np.pi), 0.2, None)
    return pd.DataFrame({
        "total_precipitation_sum": p,
        "potential_evaporation_sum_FAO_PENMAN_MONTEITH": pet,
        "temperature_2m_mean": t,
        "temperature_2m_min": t - 4,
        "temperature_2m_max": t + 4,
        "surface_solar_radiation_downwards_mean": 150 + 100 * np.sin((doy - 100) / 365 * 2 * np.pi),
    }, index=idx)


def _caravan_reference(df):
    """Verbatim logic of caravan_utils.calculate_climate_indices (FAO-PM branch) as an independent oracle."""
    p_mean = df["total_precipitation_sum"].mean()
    pet_mean = df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].mean()
    precipitation, pet = df["total_precipitation_sum"], df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"]
    mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
    mean_monthly_pet = pet.groupby(pet.index.month).mean()
    p_gt_et = 1 - mean_monthly_pet.loc[mean_monthly_precip > mean_monthly_pet] / mean_monthly_precip.loc[
        mean_monthly_precip > mean_monthly_pet]
    srs = pd.Series(np.zeros((12), dtype=np.float32), index=mean_monthly_pet.index, name="dummy")
    p_eq_et = srs.loc[mean_monthly_precip == mean_monthly_pet]
    p_lt_et = mean_monthly_precip.loc[mean_monthly_precip < mean_monthly_pet] / mean_monthly_pet.loc[
        mean_monthly_precip < mean_monthly_pet] - 1
    monthly = pd.concat([p_gt_et, p_eq_et, p_lt_et])
    mean_monthly_temp = df["temperature_2m_mean"].groupby(df.index.month).mean()
    frac_snow = mean_monthly_precip.loc[mean_monthly_temp < 0].sum() / mean_monthly_precip.sum()
    high_prec_freq = len(df.loc[df["total_precipitation_sum"] >= 5 * p_mean]) / len(df)
    low_prec_freq = len(df.loc[df["total_precipitation_sum"] < 1]) / len(df)

    def split_list(a_list):
        new_list, start = [], 0
        for index, value in enumerate(a_list):
            if index < len(a_list) - 1:
                if a_list[index + 1] > value + 1:
                    end = index + 1
                    new_list.append(a_list[start:end])
                    start = end
            else:
                new_list.append(a_list[start:len(a_list)])
        return new_list

    precip = df["total_precipitation_sum"].values
    groups = split_list(list(np.where(precip < 1)[0]))
    low_dur = np.mean([len(g) for g in groups]) if groups else 0.0
    groups = split_list(list(np.where(precip >= 5 * p_mean)[0]))
    high_dur = np.mean([len(g) for g in groups]) if groups else 0.0
    return {"p_mean": p_mean, "pet_mean_FAO_PM": pet_mean, "aridity_FAO_PM": pet_mean / p_mean, "frac_snow": frac_snow,
            "moisture_index_FAO_PM": monthly.mean(), "seasonality_FAO_PM": monthly.max() - monthly.min(),
            "high_prec_freq": high_prec_freq, "high_prec_dur": high_dur, "low_prec_freq": low_prec_freq,
            "low_prec_dur": low_dur}


def test_climate_indices_match_caravan_reference():
    df = _forcing_frame()
    ours = caravan.climate_indices(df, start=df.index.min(), end=df.index.max())
    ref = _caravan_reference(df)
    for k, v in ref.items():
        assert ours[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    assert 0 <= ours["frac_snow"] <= 1 and ours["high_prec_dur"] >= 1 and ours["low_prec_dur"] >= 1
    # the Caravan reference period is applied when the frame covers it
    long = _forcing_frame(days=365 * 45, start="1979-01-01")
    d = long.loc[caravan.CARAVAN_START:caravan.CARAVAN_END]
    assert caravan.climate_indices(long)["p_mean"] == pytest.approx(d["total_precipitation_sum"].mean())


def test_hydroatlas_catchment_row_prefers_upstream_fields():
    raw = {"hybas_id": 1, "ele_mt_sav": 100, "ele_mt_uav": 400, "run_mm_syr": 30, "dis_m3_pyr": 12.5,
           "for_pc_sse": 10, "for_pc_use": None, "up_area": 999.0, "next_down": 0}
    row = caravan._hydroatlas_catchment_row(raw)
    assert row == {"ele_mt_sav": 400, "run_mm_syr": 30, "dis_m3_pyr": 12.5, "for_pc_sse": 10}


def test_gauge_ids_and_area_helpers():
    assert caravan.gauge_id_for("aquascope_usgs", "USGS-01646500") == "aquascope_usgs_USGS-01646500"
    assert caravan.gauge_id_for("x", "a b/c") == "x_a_b_c"


def test_export_end_to_end_with_fakes(tmp_path, monkeypatch):
    """Two archived stations (one too short), forcing/basins/areas faked; files, units and validation checked."""
    catalog = [
        {"source": "uk_ea", "station_id": "S1", "name": "Thames at Somewhere", "latitude": 51.4, "longitude": -0.3,
         "variables": ["discharge"], "country": "United Kingdom", "extra": {}},
        {"source": "uk_ea", "station_id": "S2", "name": "Short record", "latitude": 52.0, "longitude": -1.0,
         "variables": ["discharge"], "country": "United Kingdom", "extra": {}},
        {"source": "uk_ea", "station_id": "S3", "name": "Not archived", "latitude": 53.0, "longitude": -2.0,
         "variables": ["discharge"], "country": "United Kingdom", "extra": {}},
    ]
    idx1 = pd.date_range("1990-01-01", periods=365 * 15, freq="D")
    idx2 = pd.date_range("2015-01-01", periods=400, freq="D")
    obs = pd.concat([
        pd.DataFrame({"station_id": "S1", "date": idx1, "value": 20.0}),
        pd.DataFrame({"station_id": "S2", "date": idx2, "value": 5.0}),
    ], ignore_index=True)

    def fake_forcing(lat, lon, start, end, *, models=None):
        idx = pd.date_range(start, end, freq="D", name="date")
        return _forcing_frame(days=len(idx), start=str(start)), "best_match"

    monkeypatch.setattr(caravan, "fetch_forcing", fake_forcing)
    monkeypatch.setattr(caravan, "station_area_km2", lambda source, st, collectors: (1000.0, "uk_ea_catchmentArea")
                        if st["station_id"] == "S1" else (None, ""))
    monkeypatch.setattr("aquascope.archive.basins.sub_basin_at",
                        lambda lat, lon, **kw: {"hybas_id": 2120000010, "next_down": 0, "main_bas": 2120000010,
                                                "sub_area": 120.0, "up_area": 1200.0, "pfaf_id": 1})
    monkeypatch.setattr("aquascope.archive.basins.load_attributes",
                        lambda ids, **kw: pd.DataFrame([{"hybas_id": 2120000010, "sub_area": 120.0, "up_area": 1200.0,
                                                         "ele_mt_sav": 90, "ele_mt_uav": 210, "pre_mm_syr": 700,
                                                         "pre_mm_uyr": 750, "for_pc_sse": 12, "for_pc_use": 20,
                                                         "dis_m3_pyr": 19.0}]))
    report = caravan.export_caravan("uk_ea", tmp_path, catalog=catalog, observations=obs, min_years=10,
                                    end=pd.Timestamp("2004-12-30").date())
    by = {g.station_id: g for g in report.gauges}
    assert by["S1"].ok and by["S1"].area_km2 == 1000.0 and by["S1"].hybas_id == 2120000010
    assert not by["S2"].ok and "too short" in by["S2"].error
    assert "S3" not in by  # not archived and fetch_missing is off: not picked
    assert report.n_ok == 1

    prefix = "aquascope_uk_ea"
    ts = pd.read_csv(tmp_path / "timeseries" / "csv" / prefix / "aquascope_uk_ea_S1.csv", index_col="date",
                     parse_dates=["date"])
    assert ts.index.min() == pd.Timestamp("1981-01-01") and ts.index.max() == pd.Timestamp("2004-12-30")
    assert list(ts.columns) == [c for c, *_ in caravan.FORCING] + ["streamflow"]
    # 20 m3/s over 1000 km2 = 20 * 86.4 / 1000 = 1.728 mm/d, rounded to 2 decimals; NaN before the record starts
    assert ts.loc["1995-06-01", "streamflow"] == pytest.approx(1.73)
    assert np.isnan(ts.loc["1985-06-01", "streamflow"])

    other = pd.read_csv(tmp_path / "attributes" / prefix / f"attributes_other_{prefix}.csv", index_col="gauge_id")
    assert list(other.index) == ["aquascope_uk_ea_S1"]
    assert other.loc["aquascope_uk_ea_S1", "area"] == 1000.0 and other.loc["aquascope_uk_ea_S1", "gauge_lat"] == 51.4
    assert other.loc["aquascope_uk_ea_S1", "area_source"] == "uk_ea_catchmentArea"
    assert other.loc["aquascope_uk_ea_S1", "forcing_model"] == "best_match"
    hydro = pd.read_csv(tmp_path / "attributes" / prefix / f"attributes_hydroatlas_{prefix}.csv", index_col="gauge_id")
    assert hydro.loc["aquascope_uk_ea_S1", "ele_mt_sav"] == 210 and hydro.loc["aquascope_uk_ea_S1", "for_pc_sse"] == 20
    assert "ele_mt_uav" not in hydro.columns and "hybas_id" not in hydro.columns
    raw = pd.read_csv(tmp_path / "attributes" / prefix / f"attributes_basinatlas_raw_{prefix}.csv",
                      index_col="gauge_id")
    assert raw.loc["aquascope_uk_ea_S1", "ele_mt_uav"] == 210
    cara = pd.read_csv(tmp_path / "attributes" / prefix / f"attributes_caravan_{prefix}.csv", index_col="gauge_id")
    for c in caravan.CARAVAN_INDEX_REQUIRED:
        assert c in cara.columns
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    sub = prov["subdatasets"][prefix]
    assert sub["n_gauges"] == 1 and sub["forcing"]["models"] == "best_match"
    assert (tmp_path / "licenses" / f"{prefix}.md").read_text(encoding="utf-8").count("CC BY 4.0") >= 1

    res = caravan.validate_caravan(tmp_path, prefix)
    assert res["ok"], res["problems"]

    # a broken tree is caught
    ts.loc["1995-06-01", "streamflow"] = -1
    ts.to_csv(tmp_path / "timeseries" / "csv" / prefix / "aquascope_uk_ea_S1.csv")
    res = caravan.validate_caravan(tmp_path, prefix)
    assert not res["ok"] and any("negative" in p for p in res["problems"])


def test_export_uses_basinatlas_area_when_agency_has_none(tmp_path, monkeypatch):
    catalog = [{"source": "usgs", "station_id": "USGS-1", "name": "X", "latitude": 40.0, "longitude": -100.0,
                "variables": ["discharge"], "country": "United States", "extra": {}}]
    idx = pd.date_range("2000-01-01", periods=365 * 12, freq="D")
    obs = pd.DataFrame({"station_id": "USGS-1", "date": idx, "value": 8.64})
    monkeypatch.setattr(caravan, "station_area_km2", lambda *a, **k: (None, ""))
    monkeypatch.setattr("aquascope.archive.basins.sub_basin_at",
                        lambda lat, lon, **kw: {"hybas_id": 7, "up_area": 100.0})
    monkeypatch.setattr("aquascope.archive.basins.load_attributes", lambda ids, **kw: pd.DataFrame())
    report = caravan.export_caravan("usgs", tmp_path, catalog=catalog, observations=obs, forcing=False)
    g = report.gauges[0]
    assert g.ok and g.area_source.startswith("basinatlas") and g.area_km2 == 100.0
    ts = pd.read_csv(tmp_path / "timeseries" / "csv" / "aquascope_usgs" / "aquascope_usgs_USGS-1.csv", index_col="date")
    assert list(ts.columns) == ["streamflow"]
    assert ts["streamflow"].dropna().iloc[0] == pytest.approx(7.46)  # 8.64 m3/s * 86.4 / 100 km2 = 7.46496
    assert not (tmp_path / "attributes" / "aquascope_usgs" / "attributes_caravan_aquascope_usgs.csv").exists()

    with pytest.raises(ValueError):
        caravan.export_caravan("taiwan_cwa", tmp_path, catalog=catalog, observations=obs)
