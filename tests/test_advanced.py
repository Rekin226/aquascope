"""The advanced study steps (aquascope.advanced) on synthetic records: no network, every call stubbed."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from aquascope import advanced as adv
from aquascope.gates import evaluate


def _daily(years: int = 60, *, trend: float = 0.0, shift_at: int | None = None, shift: float = 0.0,
           seed: int = 1, start: str = "1960-01-01") -> pd.Series:
    """A daily flow record with a seasonal cycle, lognormal noise and an optional trend or step in the floods."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(years * 365.25), freq="D")
    doy = idx.dayofyear.to_numpy()
    base = 20 + 15 * np.sin(2 * np.pi * (doy - 80) / 365.25) ** 2
    t = (idx.year - idx.year[0]).to_numpy(dtype=float)
    scale = 1 + trend * t
    if shift_at is not None:
        scale = scale * np.where(idx.year >= idx.year[0] + shift_at, 1 + shift, 1.0)
    q = base * scale * rng.lognormal(0, 0.35, len(idx))
    # one flood a year, Gumbel-sized, riding on the same trend or step
    for y in sorted(set(idx.year)):
        day = pd.Timestamp(f"{y}-04-15") + pd.Timedelta(days=int(rng.integers(0, 30)))
        if day in idx:
            k = idx.get_loc(day)
            q[k] += scale[k] * (80 + 25 * rng.gumbel())
    return pd.Series(q, index=idx)


@pytest.fixture
def record(monkeypatch):
    """Serve a synthetic record through the module's fetch (the Archive-first fetch is never called)."""
    store: dict[str, pd.Series] = {}

    def fake_fetch(source, station_id, years, variable):
        return {"series": store[station_id], "variable": "discharge", "unit": "m3/s", "note": "synthetic"}

    adv._RECORDS.clear()
    monkeypatch.setattr(adv, "_fetch", fake_fetch)
    return store


# ── change points ──


def test_a_clean_record_reads_stationary(record):
    record["clean"] = _daily(60, seed=3)
    out = adv.change_points("usgs", "clean")
    assert out["n"] >= 55 and out["stationary"] is True
    assert "not contradicted" in out["verdict"]
    assert evaluate([{"check": "stationary"}], out)[0]["passed"]
    json.dumps(out)  # JSON-safe


def test_a_step_change_is_found_near_its_year(record):
    record["step"] = _daily(60, shift_at=30, shift=0.8, seed=4)
    out = adv.change_points("usgs", "step")
    assert out["stationary"] is False and out["pettitt"]["significant"]
    assert abs(out["pettitt"]["change_year"] - 1990) <= 4
    assert out["pettitt"]["change_pct"] > 30
    assert "step change around" in out["verdict"]
    assert not evaluate([{"check": "stationary"}], out)[0]["passed"]


def test_too_short_a_record_is_an_error_not_a_verdict(record):
    record["short"] = _daily(6)
    out = adv.change_points("usgs", "short")
    assert "error" in out and "at least 10" in out["error"]


# ── the nonstationary GEV ──


def test_a_rising_flood_record_prefers_the_nonstationary_fit(record):
    record["rising"] = _daily(70, trend=0.02, seed=5)
    out = adv.nonstationary_flood("usgs", "rising", return_periods=[100], n_boot=30)
    assert out["likelihood_ratio"]["significant"] and out["preferred"] == "nonstationary"
    row = out["table"][-1]
    assert row["last_year"] > row["first_year"] and row["change_first_to_last_pct"] > 20
    assert out["horizon_year"] == out["last_year"] + 25 and row["horizon"] > row["last_year"]
    ci = out["nonstationary"]["ci_last_year"]
    assert ci["lower"] < row["last_year"] < ci["upper"] and len(ci["ci"]) == 2
    assert any("sensitivity" in n for n in out["notes"])


def test_a_stationary_record_keeps_the_stationary_answer(record):
    record["flat"] = _daily(70, seed=6)
    out = adv.nonstationary_flood("usgs", "flat", return_periods=[50], n_boot=0)
    assert out["preferred"] == "stationary" and "stationary 50-year flow" in out["verdict"]
    assert out["likelihood_ratio"]["statistic"] >= 0.0


def test_the_nonstationary_fit_is_never_worse_than_the_stationary_one():
    """The library fix: both fits start from the stationary MLE, so the LR statistic cannot be negative and the
    shape keeps scipy's sign (a heavy tail gives a larger 100-year level than a light one)."""
    from scipy.stats import genextreme

    from aquascope.hydrology.flood_frequency import fit_nonstationary_gev

    years = np.arange(1950, 2020, dtype=float)
    heavy = genextreme.rvs(-0.2, loc=100, scale=30, size=len(years), random_state=11)
    light = genextreme.rvs(0.2, loc=100, scale=30, size=len(years), random_state=11)
    rh = fit_nonstationary_gev(heavy, years, return_periods=[100])
    rl = fit_nonstationary_gev(light, years, return_periods=[100])
    for r in (rh, rl):
        assert r.lr_statistic >= 0.0 and 0.0 <= r.p_value <= 1.0 and r.aic_stationary > 0
    assert rh.shape < 0 < rl.shape, "scipy convention: negative c is the heavy tail"
    assert rh.return_levels[100].mean() > rl.return_levels[100].mean()
    c, loc, scale = rh.stationary_params
    assert abs(c - genextreme.fit(heavy)[0]) < 0.1


# ── peaks over threshold ──


def test_pot_finds_about_the_asked_number_of_peaks(record):
    record["pot"] = _daily(40, seed=7)
    out = adv.pot_flood("usgs", "pot", events_per_year=2.0, return_periods=[10, 100])
    assert out["n_peaks"] >= 10 and 0.8 <= out["peaks_per_year"] <= 3.5
    assert out["gpd"]["q_by_T"]["100"] > out["gpd"]["q_by_T"]["10"] > out["threshold"]
    assert out["annual_max_gev"]["q_by_T"]["100"] is not None


def test_declustering_merges_a_run_of_days_into_one_event():
    idx = pd.date_range("2000-01-01", periods=30, freq="D")
    s = pd.Series(1.0, index=idx)
    s.iloc[[3, 4, 5]] = [10, 12, 11]   # one event
    s.iloc[20] = 9                     # another
    peaks = adv._decluster(s, 5.0, 7)
    assert list(peaks.values) == [12.0, 9.0]


# ── the catchment model ──


def _forcing_for(q: pd.Series, area_km2: float, *, cold: bool = False, seed: int = 8) -> pd.DataFrame:
    """ERA5-like forcing whose rainfall drives the synthetic flow (so GR4J has something to find)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(q.index[0] - pd.Timedelta(days=400), q.index[-1], freq="D")
    doy = idx.dayofyear.to_numpy()
    p = rng.gamma(0.6, 6.0, len(idx))
    pet = 2.5 + 2.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)
    t = (-8 + 14 * season) if cold else (13 + 8 * season)
    return pd.DataFrame({"p": p, "pet": np.clip(pet, 0.2, None), "t": t}, index=idx)


def test_the_catchment_model_calibrates_validates_and_runs_scenarios(monkeypatch):
    from aquascope.models.rainfall_runoff import GR4J

    idx = pd.date_range("2004-01-01", "2023-12-31", freq="D")
    f = _forcing_for(pd.Series(0.0, index=idx), 500.0)
    truth = GR4J(x1=300, x2=0.5, x3=80, x4=1.8).simulate(f["p"], f["pet"], warmup_days=365).streamflow
    q_cms = (truth * 500.0 / 86.4).loc[idx]
    adv._RECORDS.clear()
    monkeypatch.setattr(adv, "_fetch", lambda *a: {"series": q_cms, "variable": "discharge", "unit": "m3/s",
                                                   "note": "synthetic"})
    monkeypatch.setattr(adv, "_forcing", lambda lat, lon, start, end: f.loc[str(start):str(end)])
    out = adv.catchment_model("usgs", "m", 45.0, 5.0, area_km2=500.0, maxiter=12)
    assert "error" not in out, out.get("error")
    assert out["snow"]["used"] is False
    assert out["validation"]["kge"] > 0.9 and out["skill_grade"] == "good"
    assert set(out["params"]) == {"X1", "X2", "X3", "X4"}
    labels = [r["label"] for r in out["scenarios"]]
    assert labels[0] == "10% less rain" and len(labels) == 4
    drier, wetter = out["scenarios"][0], out["scenarios"][1]
    assert drier["mean_flow_change_pct"] < 0 < wetter["mean_flow_change_pct"]
    assert len(out["pet_climatology"]) == 366
    assert evaluate([{"check": "kge_min", "value": 0.5, "path": "validation.kge"}], out)[0]["passed"]


def test_the_catchment_model_needs_an_area():
    out = adv.catchment_model("usgs", "x", 45.0, 5.0, area_km2=None)
    assert "area" in out["error"]


def test_a_snowy_catchment_gets_a_snow_store(monkeypatch):
    from aquascope.models.rainfall_runoff import GR4J, degree_day_snow

    idx = pd.date_range("2004-01-01", "2023-12-31", freq="D")
    f = _forcing_for(pd.Series(0.0, index=idx), 800.0, cold=True, seed=9)
    liquid = degree_day_snow(f["p"], f["t"], ddf=3.5, t0=0.5)
    truth = GR4J(x1=250, x2=0.0, x3=90, x4=2.0).simulate(liquid, f["pet"], warmup_days=365).streamflow
    q_cms = (truth * 800.0 / 86.4).loc[idx]
    adv._RECORDS.clear()
    monkeypatch.setattr(adv, "_fetch", lambda *a: {"series": q_cms, "variable": "discharge", "unit": "m3/s",
                                                   "note": "synthetic"})
    monkeypatch.setattr(adv, "_forcing", lambda lat, lon, start, end: f.loc[str(start):str(end)])
    out = adv.catchment_model("usgs", "s", 60.0, 10.0, area_km2=800.0, maxiter=12)
    assert out["snow"]["used"] is True and {"DDF", "T0"} <= set(out["params"])
    assert out["validation"]["kge"] > 0.8
    warm = next(r for r in out["scenarios"] if r["label"] == "2 °C warmer")
    assert warm["dt_c"] == 2.0 and warm["dpet_pct"] == 6.0


def test_degree_day_snow_stores_winter_and_melts_in_spring():
    from aquascope.models.rainfall_runoff import degree_day_snow

    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    p = pd.Series([10, 10, 0, 0, 0, 0], index=idx, dtype=float)
    t = pd.Series([-5, -5, -5, 5, 5, 5], index=idx, dtype=float)
    out = degree_day_snow(p, t, ddf=4.0, t0=0.0)
    assert out.iloc[:3].sum() == 0.0            # all snow, no melt below freezing
    assert list(out.iloc[3:]) == [20.0, 0.0, 0.0]  # 4 mm/C/day x 5 C = 20 mm melts the whole 20 mm pack
    assert math.isclose(out.sum(), p.sum())      # mass balance


# ── CMIP6 projections ──


def _cmip6_frames(models=("A", "B", "C", "D"), *, wetter: float = 1.05, warmer: float = 1.5) -> dict:
    idx = pd.date_range("1984-01-01", "2049-12-31", freq="D")
    rng = np.random.default_rng(12)
    out = {}
    for i, m in enumerate(models):
        fut = idx.year >= 2020
        p = rng.gamma(0.6, 5.0, len(idx)) * np.where(fut, wetter + 0.01 * i, 1.0)
        t = 8 + 10 * np.sin(2 * np.pi * idx.dayofyear / 365.25) + np.where(fut, warmer, 0.0)
        frame = pd.DataFrame({"p": p, "t": t}, index=idx)
        frame["pet_oudin"] = adv.oudin_pet(frame["t"], 45.0)
        out[m] = frame
    return out


def test_climate_projection_reports_the_ensemble_and_its_spread(monkeypatch):
    monkeypatch.setattr(adv, "_cmip6", lambda *a, **k: _cmip6_frames())
    out = adv.climate_projection(45.0, 5.0)
    assert out["n_models"] == 4 and len(out["models"]) == 4
    e = out["ensemble"]["precip_change_pct"]
    assert e["n"] == 4 and e["n_up"] == 4 and 4 < e["median"] < 10
    assert out["ensemble"]["temp_change_c"]["median"] == pytest.approx(1.5, abs=0.05)
    assert out["ensemble"]["pet_change_pct"]["median"] > 0
    assert "mean_flow_change_pct" not in out["ensemble"], "no GR4J parameters, no flow changes"
    assert "4 of 4 agree on more" in out["verdict"]
    assert any("SSP5-8.5" in n for n in out["notes"])
    assert evaluate([{"check": "min_models", "value": 3}], out)[0]["passed"]


def test_climate_projection_drives_a_calibrated_model(monkeypatch):
    monkeypatch.setattr(adv, "_cmip6", lambda *a, **k: _cmip6_frames(wetter=1.15, warmer=0.2))
    params = {"X1": 300, "X2": 0.0, "X3": 80, "X4": 1.8}
    out = adv.climate_projection(45.0, 5.0, params=params, area_km2=500.0, pet_climatology=[2.5] * 366)
    e = out["ensemble"]["mean_flow_change_pct"]
    assert e["n"] == 4 and e["median"] > 5, "15% more rain in every model raises the mean flow"
    assert out["ensemble"]["flood_change_pct"]["n"] == 4


def test_too_few_models_is_an_error(monkeypatch):
    monkeypatch.setattr(adv, "_cmip6", lambda *a, **k: _cmip6_frames(models=("A", "B")))
    out = adv.climate_projection(45.0, 5.0)
    assert "at least 3" in out["error"]


def test_oudin_is_zero_below_minus_five_and_grows_with_temperature():
    idx = pd.date_range("2020-06-21", periods=3, freq="D")
    pet = adv.oudin_pet(pd.Series([-10.0, 10.0, 25.0], index=idx), 45.0)
    assert pet.iloc[0] == 0.0 and 0 < pet.iloc[1] < pet.iloc[2] < 10


def test_the_climate_collector_builds_the_request(monkeypatch):
    from aquascope.collectors.openmeteo import OpenMeteoCollector

    seen = {}
    col = OpenMeteoCollector(mode="climate")
    monkeypatch.setattr(col.client, "get_json", lambda url, params=None: seen.update(url=url, **params) or {})
    col.fetch_raw(latitude=1, longitude=2, start_date="1990-01-01", end_date="2000-01-01", models="A,B",
                  daily=["precipitation_sum"])
    assert seen["url"].startswith("https://climate-api.open-meteo.com") and seen["models"] == "A,B"
    assert "timezone" not in seen
    with pytest.raises(ValueError):
        col.fetch_raw(latitude=1, longitude=2)


# ── more than one gauge ──


def test_regional_flood_matches_the_studys_gauge_in_the_pool(monkeypatch):
    fake = {"summary": {"n_studied": 6}, "headline": "6 gauges", "field_significance": {}, "notes": [],
            "methods": [], "regional_frequency": {"n_sites": 6, "heterogeneity": {"H": 0.7, "class": "homogeneous"},
                                                  "regional": {"growth_curve": {"2": 0.9, "100": 2.1}}},
            "sites": [{"key": "usgs/USGS-01013500", "source": "usgs", "station_id": "USGS-01013500",
                       "q100": 583.2, "regional_q100": 494.8, "record_years": 40.0, "n_amax": 39}]}
    import aquascope.area_study as area

    monkeypatch.setattr(area, "study_area", lambda **kw: fake)
    out = adv.regional_flood(47.2, -68.6, source="usgs", station_id="01013500")
    assert out["target"]["pooled_q100"] == 494.8 and out["n_pooled"] == 6
    assert "acceptably homogeneous" in out["verdict"]
    assert evaluate([{"check": "min_sites", "value": 5}], out)[0]["passed"]


def test_every_advanced_tool_is_a_study_step_with_a_catalogue_entry():
    from aquascope.studio import catalogue
    from aquascope.study import tool_names

    names = set(tool_names())
    for tool in ("change_points", "nonstationary_flood", "pot_flood", "catchment_model", "climate_projection",
                 "regional_flood", "compare_gauges"):
        assert tool in names
        entry = catalogue.get(tool)
        assert entry is not None and entry.kind in ("station", "site")
        for m in entry.methods:
            from aquascope.methods import METHODS

            assert METHODS[m].tool == tool, (tool, m)


def test_a_refused_projection_is_said_plainly_and_not_repeated(monkeypatch):
    calls = []

    def refuse(*a, **k):
        calls.append(1)
        raise RuntimeError("TransportError: HTTP 429 for https://climate-api.open-meteo.com/v1/climate")

    adv._CMIP6_CACHE.clear()
    monkeypatch.setattr(adv, "_cmip6_fetch", refuse)
    first = adv.climate_projection(10.0, 20.0)
    second = adv.climate_projection(10.0, 20.0, return_period=50)
    assert "free allowance" in first["error"] and "free allowance" in second["error"]
    assert calls == [1], "the fallback's identical request is answered from the remembered refusal"
    adv._CMIP6_CACHE.clear()
