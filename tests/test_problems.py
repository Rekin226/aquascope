"""The site-level tools the playbooks call (aquascope.problems), run offline against fake ERA5 and fake records:
what each returns, the arithmetic it does, and how it says no."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope import problems as pr

DAYS = pd.date_range("1985-01-01", "2026-08-20", freq="D")
_PHASE = 2 * np.pi * (DAYS.dayofyear / 365.25)
_RNG = np.random.default_rng(0)
ERA5 = pd.DataFrame({
    "precipitation_sum": _RNG.gamma(0.6, 3.0, len(DAYS)) * (1 + 0.4 * np.sin(_PHASE)),
    "temperature_2m_mean": (10 + 8 * np.sin(_PHASE - np.pi / 2) + _RNG.normal(0, 2, len(DAYS))
                            + np.linspace(0, 2, len(DAYS))),
    "et0_fao_evapotranspiration": np.clip(2.0 + 1.8 * np.sin(_PHASE - np.pi / 2) + _RNG.normal(0, 0.3, len(DAYS)),
                                          0.1, None),
}, index=DAYS)
FLOW = pd.Series(np.clip(40 + 30 * np.sin(_PHASE + np.pi / 2) + _RNG.gamma(1.5, 10, len(DAYS)), 2, None), index=DAYS)
_P_MONTH = ERA5["precipitation_sum"].resample("MS").sum()
# the water table follows the 6-month rain sum, three months late
LEVELS = (_P_MONTH.rolling(6).mean().shift(3) / 10 + 20).dropna()


def fake_era5(lat, lon, *, years=None, start=None, end=None,
              variables=("precipitation_sum", "temperature_2m_mean", "et0_fao_evapotranspiration")):
    sub = ERA5.loc[str(start) if start else None: str(end) if end else None, list(variables)]
    if years is not None and start is None:
        sub = sub.iloc[-int(years * 365.25):]
    return sub.copy(), {"source": "ERA5 via Open-Meteo (fake)", "start": str(sub.index[0].date()),
                        "end": str(sub.index[-1].date()), "elevation_m": 12.0, "n_days": len(sub)}


def fake_fetch(source, station_id, *, years=None, variable=None, **kw):
    if station_id == "EMPTY":
        return {"series": None, "variable": variable, "unit": "", "note": "nothing", "requested": {}}
    if variable == "precipitation":
        s = ERA5["precipitation_sum"].loc["1990":]
    elif variable == "groundwater_level":
        s = LEVELS
    else:
        s = FLOW
    unit = {"precipitation": "mm", "groundwater_level": "m"}.get(variable, "m3/s")
    return {"series": s, "variable": variable or "discharge", "unit": unit, "note": "fake", "requested": {}}


@pytest.fixture(autouse=True)
def _offline():
    with patch.object(pr, "era5_daily", fake_era5), patch.object(pr, "fetch_series", fake_fetch):
        yield


def _strict(payload: dict) -> None:
    json.dumps(payload, allow_nan=False)


# ── drought ─────────────────────────────────────────────────────────────────


def test_drought_indices_for_the_era5_cell():
    out = pr.drought_indices(51.4, -0.3, years=40)
    _strict(out)
    assert out["precipitation_source"] == "ERA5 cell via Open-Meteo" and out["pet_method"] == "thornthwaite"
    assert out["timescales"] == [1, 3, 12] and out["headline_timescale"] == 3 and out["headline_index"] == "spei"
    assert 39 <= out["years"] <= 41 and set(out["current"]["spei"]) == {"1", "3", "12"}
    assert out["temperature"]["trend_c_per_decade"] > 0.3, "two degrees over forty years is read as warming"
    assert out["status"] in ("normal", "moderately_dry", "severely_dry", "extremely_dry", "moderately_wet", "very_wet",
                             "extremely_wet") and isinstance(out["in_drought"], bool)
    names = [m["name"] for m in out["methods"]]
    assert names[:3] == ["Standardized Precipitation Index", "Standardized Precipitation-Evapotranspiration Index",
                         "Thornthwaite potential evapotranspiration"]
    assert any("ERA5 cell" in n for n in out["notes"])


def test_drought_indices_at_a_rain_gauge_uses_its_record_and_era5_pet():
    out = pr.drought_indices(51.4, -0.3, source="uk_ea", station_id="R1", timescales=[3, 12])
    _strict(out)
    assert out["precipitation_source"] == "uk_ea R1 (gauge)" and out["station"]["station_id"] == "R1"
    assert 36 <= out["station"]["years"] <= 37 and out["timescales"] == [3, 12]
    row = next(r for r in out["indices"] if r["timescale"] == 12)
    assert row["divergence"]["mean_last_10y"] < 0, "warming: SPEI reads drier than SPI"
    assert row["spi"]["worst"] < -1.5 and row["spei"]["events"] >= 1


def test_drought_indices_pet_options_and_refusals():
    fao = pr.drought_indices(51.4, -0.3, pet="fao56")
    assert fao["pet_method"] == "fao56" and fao["headline_index"] == "spei"
    spi_only = pr.drought_indices(51.4, -0.3, pet="none")
    assert spi_only["pet_method"] == "none" and spi_only["headline_index"] == "spi"
    assert spi_only["current"]["spei"] == {}
    assert "unknown source" in pr.drought_indices(51.4, -0.3, source="nope", station_id="x")["error"]
    empty = pr.drought_indices(51.4, -0.3, source="uk_ea", station_id="EMPTY")
    assert "error" not in empty and any("no precipitation" in n for n in empty["notes"]), "falls back to the cell"

    def broken(*a, **k):
        raise RuntimeError("offline")

    with patch.object(pr, "era5_daily", broken):
        assert "ERA5 climate unavailable" in pr.drought_indices(51.4, -0.3)["error"]
        gauge = pr.drought_indices(51.4, -0.3, source="uk_ea", station_id="R1")
        assert gauge["pet_method"] == "none" and gauge["headline_index"] == "spi", "the gauge still gives SPI"


def test_drought_propagation_recovers_the_lag_built_into_the_levels():
    out = pr.drought_propagation("uk_ea", "W1", 51.4, -0.3)
    _strict(out)
    assert out["unit"] == "m" and out["years"] > 35 and out["sgi"]["n"] > 400
    best = out["propagation"]["best"]
    assert best["timescale"] == 6 and best["lag_months"] == 3 and best["correlation"] > 0.9
    assert set(out["propagation"]["by_timescale"]) == {"1", "3", "6", "12", "24"}
    assert set(out["series"]) == {"index", "step", "sgi", "spi"}
    assert [m["name"] for m in out["methods"]][2] == "SPI to SGI drought propagation"
    assert "no groundwater levels" in pr.drought_propagation("uk_ea", "EMPTY", 51.4, -0.3)["error"].lower()


# ── low flows and supply ────────────────────────────────────────────────────


def test_low_flow_context_reads_the_record():
    out = pr.low_flow_context("uk_ea", "3400TH")
    _strict(out)
    fdc = out["fdc"]
    assert fdc["q95"] < fdc["q50"] < fdc["q10"] < fdc["q05"] and 0 < out["bfi"] < 1
    assert out["low_flow"]["7q10"] < fdc["q95"] * 1.2 and out["unit"] == "m3/s"
    assert 0 <= out["recent"]["last_30d_exceedance_pct"] <= 100 and out["recent"]["end"] == "2026-08-20"
    assert "Low-flow frequency (7Q10)" in [m["name"] for m in out["methods"]]


def test_supply_reliability_gauged_applies_the_reserve_and_the_share():
    out = pr.supply_reliability(demand_m3s=2.0, source="uk_ea", station_id="3400TH")
    _strict(out)
    assert out["mode"] == "gauged" and out["reserve_rule"] == "Q95 kept in the river"
    assert out["reserve_m3s"] == pytest.approx(out["fdc"]["q95"])
    assert out["required_flow_m3s"] == pytest.approx(max(2.0 / 0.1, out["fdc"]["q95"] + 2.0))
    rel = out["reliability"]
    assert 0 < rel["daily"] < 1 and rel["daily"] <= rel["daily_reserve_only"] and 0 <= rel["annual"] <= 1
    assert rel["volumetric"] >= rel["daily"] and rel["worst_year"]["days_short"] >= rel["days_short_per_year"]
    assert out["verdict"] in ("reliable", "mostly reliable", "seasonal shortfalls", "unreliable")
    # the share alone binds here: a flow of 20 m3/s is needed for 2 m3/s at a 10 % share
    daily = FLOW.resample("D").mean()
    assert rel["daily"] == pytest.approx(float((daily >= out["required_flow_m3s"]).mean()), abs=0.01)


def test_supply_reliability_converts_ml_day_and_restricts_to_months():
    out = pr.supply_reliability(demand_ml_day=50, source="uk_ea", station_id="3400TH", months=[6, 7, 8],
                                reserve="none", share=0.5)
    assert out["demand_m3s"] == pytest.approx(50 * 1000 / 86400, rel=1e-3) and out["demand_given_as"] == "ML/day"
    assert out["months"] == [6, 7, 8] and out["reserve_rule"] == "no reserve" and out["reliability"]["daily"] == 1.0
    fixed = pr.supply_reliability(demand_m3s=1, source="uk_ea", station_id="3400TH", reserve="12.5")
    assert fixed["reserve_m3s"] == 12.5 and "12.5" in fixed["reserve_rule"]


def test_supply_reliability_regional_reads_a_band_off_three_flow_duration_points():
    reg = {"method": "similarity", "n_donors_available": 5, "estimates": {
        "q95_mm": {"value": 0.2, "low": 0.1, "high": 0.4}, "q_median_mm": {"value": 0.9, "low": 0.6, "high": 1.3},
        "q05_mm": {"value": 4.0, "low": 3.0, "high": 6.0}, "q_mean_mm": {"value": 1.3, "low": 1.0, "high": 1.7}},
        "skill": {"by_signature": {"q95_mm": {"nse": 0.55}}},
        "methods": [{"name": "donors", "text": "t", "citation": "c"}]}
    with patch("aquascope.mcp_server.describe_catchment", return_value={"attributes": {"upstream_area_km2": 500.0}}), \
         patch("aquascope.mcp_server.regionalize_signatures", return_value=reg):
        out = pr.supply_reliability(demand_m3s=0.05, lat=51.4, lon=-0.3)
        _strict(out)
        assert out["mode"] == "regional" and out["area_km2"] == 500.0 and out["n_donors"] == 5
        assert out["signatures_m3s"]["q95"]["value"] == pytest.approx(0.2 * 500 / 86.4)
        assert out["signatures_m3s"]["q95"]["loo_nse"] == 0.55
        rel = out["reliability"]
        assert rel["low"] <= rel["daily"] <= rel["high"] and 0.05 <= rel["daily"] <= 0.95
        assert "donors" in [m["name"] for m in out["methods"]]
        big = pr.supply_reliability(demand_m3s=50.0, lat=51.4, lon=-0.3)
        assert big["reliability"]["daily"] == 0.05 and big["verdict"] == "unreliable"
    with patch("aquascope.mcp_server.describe_catchment", return_value={"error": "BasinATLAS down"}):
        assert "BasinATLAS down" in pr.supply_reliability(demand_m3s=1, lat=51.4, lon=-0.3)["error"]


def test_supply_reliability_says_what_it_needs():
    assert "demand" in pr.supply_reliability()["error"]
    assert "source and station_id" in pr.supply_reliability(demand_m3s=1)["error"]
    assert "share" in pr.supply_reliability(demand_m3s=1, source="uk_ea", station_id="x", share=1.5)["error"]
    assert "months" in pr.supply_reliability(demand_m3s=1, source="uk_ea", station_id="x", months=[13])["error"]
    assert "unknown source" in pr.supply_reliability(demand_m3s=1, source="nope", station_id="x")["error"]
    bad_reserve = pr.supply_reliability(demand_m3s=1, source="uk_ea", station_id="3400TH", reserve="lots")
    assert "reserve" in bad_reserve["error"]


def test_exceedance_interpolates_in_log_space_and_clamps():
    curve = [(0.95, 1.0), (0.5, 10.0), (0.05, 100.0)]
    assert pr._exceedance_of(0.5, curve) == 0.95 and pr._exceedance_of(500.0, curve) == 0.05
    assert pr._exceedance_of(10.0, curve) == pytest.approx(0.5)
    assert pr._exceedance_of(np.sqrt(10.0), curve) == pytest.approx(0.725, abs=1e-6)


# ── irrigation ──────────────────────────────────────────────────────────────


def test_crop_water_demand_averages_the_seasons_and_converts_to_volume_and_rate():
    out = pr.crop_water_demand(51.4, -0.3, crop="maize", area_ha=20, planting_month=4)
    _strict(out)
    assert out["crop"] == "maize" and out["season_days"] == 125 and out["season"]["months"] == [4, 5, 6, 7, 8]
    assert len(out["years_used"]) >= 9 and out["supply_checked"] is False
    d = out["demand"]
    assert d["gross_irrigation_mm"] == pytest.approx(d["net_irrigation_mm"] / 0.7, rel=0.01)
    assert d["gross_m3"] == pytest.approx(d["gross_irrigation_mm"] / 1000 * 20 * 1e4, rel=1e-3)
    assert d["mean_m3s"] == pytest.approx(d["gross_m3"] / (125 * 86400), rel=1e-2)
    assert d["peak_month_m3s"] > d["mean_m3s"] and d["gross_irrigation_mm_range"][0] <= d["gross_irrigation_mm"]
    assert "maize on 20 ha" in out["text"] and "m3/s in the peak month" in out["text"]
    assert any("#310" in n for n in out["notes"]) and any("108732" in n for n in out["notes"])
    assert out["kc"] == {"initial": 0.3, "mid": 1.2, "late": 0.6}


def test_crop_water_demand_refusals():
    assert "unknown crop" in pr.crop_water_demand(51.4, -0.3, crop="kale", area_ha=1, planting_month=4)["error"]
    assert "planting_month" in pr.crop_water_demand(51.4, -0.3, crop="maize", area_ha=1, planting_month=13)["error"]
    assert "efficiency" in pr.crop_water_demand(51.4, -0.3, crop="maize", area_ha=1, planting_month=4,
                                                efficiency=0)["error"]
    assert pr.crop_water_demand(51.4, -0.3, crop="Sugar Cane", area_ha=1, planting_month=4)["crop"] == "sugarcane"
