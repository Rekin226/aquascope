"""AquaScope Bench Phase 0 (#175): basins, the calibration env (gymnasium API), the baselines and the leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope import gym as hg
from aquascope.gym.env import _BoundsBox


@pytest.fixture(scope="module")
def basin():
    return hg.synthetic_basin(0, years=6)


def test_synthetic_basin_is_reproducible_and_well_formed(basin):
    again = hg.synthetic_basin(0, years=6)
    pd.testing.assert_frame_equal(basin.frame, again.frame)
    assert list(basin.frame.columns[:3]) == ["precip", "pet", "q_obs"] and basin.frame.index.freq is not None
    assert basin.meta["true_params"]["X1"] == 320.0 and 5.9 < basin.n_years < 6.1
    s = basin.summary()
    assert 0.2 < s["runoff_ratio"] < 0.9 and s["q_missing_fraction"] == 0 and s["split_date"] < s["end"]
    assert len(basin.calibration) + len(basin.validation) == len(basin.frame)
    with pytest.raises(ValueError):
        hg.Basin("x", "x", basin.frame.head(100))
    with pytest.raises(ValueError):
        hg.Basin("x", "x", basin.frame.drop(columns=["pet"]))


def test_basin_fills_gaps_in_forcing_but_keeps_flow_gaps():
    f = hg.synthetic_basin(2, years=4).frame.copy()
    f = f.drop(f.index[10:20])  # a hole in the record
    f.loc[f.index[100:105], "precip"] = np.nan
    f.loc[f.index[200:230], "q_obs"] = np.nan
    b = hg.Basin("g", "gappy", f, split_date="2002-06-01")
    assert b.frame.index.is_monotonic_increasing and len(b.frame) == len(f) + 10
    assert b.frame["precip"].isna().sum() == 0 and b.frame["pet"].isna().sum() == 0
    assert b.frame["q_obs"].isna().sum() == 40 and b.split_date == pd.Timestamp("2002-06-01")


def test_env_follows_the_gym_api_and_scores_the_truth_well(basin):
    env = hg.CalibrationEnv(basin, objective="nse", max_steps=3)
    obs, info = env.reset(seed=0)
    assert obs.shape == (16,) and obs.dtype == np.float32 and info["basin"]["id"] == "synthetic-0"
    assert info["param_names"] == ["X1", "X2", "X3", "X4"] and len(info["obs_names"]) == 16
    assert obs[9] == -1.0 and obs[15] == 0.0  # no action yet, no steps used
    obs, r, term, trunc, info = env.step(basin.meta["true_params"])
    assert r > 0.9 and not term and not trunc and info["steps_left"] == 2
    assert 0 <= obs[9] <= 1 and obs[13] == pytest.approx(r, abs=1e-6) and info["validation"]["nse"] > 0.9
    obs, r2, term, trunc, info = env.step([1500, 5, 500, 10])  # a bad corner of the space
    assert r2 < r and env.best["reward"] == pytest.approx(r) and obs[14] == pytest.approx(r, abs=1e-6)
    obs, r3, term, trunc, info = env.step(np.array([1e9, -1e9, 0, 0]))  # clipped, not rejected
    assert trunc and info["params"]["X1"] == 1500.0 and info["params"]["X2"] == -10.0
    with pytest.raises(RuntimeError):
        env.step([1, 1, 1, 1])
    assert "best nse" in env.render() and env.n_simulations == 3
    env.reset()
    with pytest.raises(ValueError):
        env.step([1, 2, 3])
    with pytest.raises(ValueError):
        hg.CalibrationEnv(basin, objective="rmse")


def test_env_cycles_basins_and_unit_actions(basin):
    other = hg.synthetic_basin(1, years=6)
    env = hg.CalibrationEnv([basin, other], max_steps=2, unit_actions=True)
    assert np.allclose(env.action_space.low, 0) and np.allclose(env.action_space.high, 1)
    _, info = env.reset()
    assert info["basin"]["id"] == "synthetic-0"
    _, info = env.reset()
    assert info["basin"]["id"] == "synthetic-1"
    _, info = env.reset(options={"basin": "synthetic-0"})
    assert env.basin.id == "synthetic-0"
    with pytest.raises(KeyError):
        env.reset(options={"basin": "nope"})
    _, r, *_ = env.step([0.2, 0.6, 0.15, 0.2])
    assert env.best["params"]["X1"] == pytest.approx(1 + 0.2 * 1499)
    ev = env.evaluate(basin.meta["true_params"])
    assert ev["calibration"]["nse"] > 0.9 and env.step_count == 1  # evaluate spends no step
    u = env.to_unit(basin.meta["true_params"])
    assert env.from_unit(u) == pytest.approx(basin.meta["true_params"], rel=1e-6)


def test_bounds_box_shim_behaves_like_a_space():
    box = _BoundsBox(np.array([0, -1]), np.array([1, 1]))
    box.seed(0)
    x = box.sample()
    assert box.contains(x) and not box.contains(np.array([2, 0])) and box.shape == (2,)


@pytest.mark.skipif(not hg.HAS_GYMNASIUM, reason="gymnasium not installed")
def test_gymnasium_env_checker_accepts_it(basin):
    from gymnasium.utils.env_checker import check_env

    check_env(hg.CalibrationEnv(basin, max_steps=3), skip_render_check=True)
    check_env(hg.CalibrationEnv(basin, max_steps=3, unit_actions=True), skip_render_check=True)


def test_baselines_and_leaderboard(basin):
    env = hg.CalibrationEnv(basin, max_steps=12)
    env.reset(seed=0)
    rs = hg.random_search(env, seed=0)
    assert rs["agent"] == "random_search" and rs["steps"] == 12 and rs["best_reward"] is not None
    env.reset(seed=0)
    nm = hg.nelder_mead(env)
    assert nm["steps"] == 12 and nm["best_reward"] > rs["best_reward"] - 0.3  # a sane local search
    env.reset(seed=0)
    de = hg.differential_evolution(env, maxiter=3, popsize=4)
    assert de["steps"] <= 12 and de["simulator_calls"] >= 12 and de["best_reward"] > 0.5
    table = hg.run_leaderboard([basin], ["random_search", "nelder_mead"], max_steps=6, seeds=(0, 1))
    assert len(table) == 4 and set(table["agent"]) == {"random_search", "nelder_mead"}
    assert {"best_reward", "val_nse", "val_kge", "simulator_calls", "X1"} <= set(table.columns)
    assert hg.episode_table(env).shape[0] == env.step_count


def test_make_and_suggest_basins(monkeypatch, basin):
    env = hg.make()
    assert env.basin.id == "synthetic-0"
    sig = pd.DataFrame([
        {"source": "usgs", "station_id": "A", "n_years": 30, "zero_flow_fraction": 0.0, "runoff_ratio": 0.5,
         "area_source": "agency", "area_km2": 100, "start": "1990-01-01", "end": "2020-01-01", "q_mean_mm": 1.0,
         "baseflow_index": 0.6},
        {"source": "uk_ea", "station_id": "B", "n_years": 40, "zero_flow_fraction": 0.0, "runoff_ratio": 0.4,
         "area_source": "agency", "area_km2": 50, "start": "1980-01-01", "end": "2020-01-01", "q_mean_mm": 1.2,
         "baseflow_index": 0.7},
        {"source": "usgs", "station_id": "C", "n_years": 12, "zero_flow_fraction": 0.0, "runoff_ratio": 0.5,
         "area_source": "basinatlas_up_area", "area_km2": 900, "start": "2008-01-01", "end": "2020-01-01",
         "q_mean_mm": 1.0, "baseflow_index": 0.6},
        {"source": "usgs", "station_id": "D", "n_years": 30, "zero_flow_fraction": 0.2, "runoff_ratio": 0.5,
         "area_source": "agency", "area_km2": 100, "start": "1990-01-01", "end": "2020-01-01", "q_mean_mm": 0.1,
         "baseflow_index": 0.2},
        {"source": "usgs", "station_id": "E", "n_years": 30, "zero_flow_fraction": 0.0, "runoff_ratio": 0.5,
         "area_source": "agency", "area_km2": 100, "start": "1990-01-01", "end": "2020-01-01", "q_mean_mm": 1.0,
         "baseflow_index": 0.6},
    ])
    cat = pd.DataFrame([{"source": "usgs", "station_id": "E", "snow_cover_pct": 60.0},
                        {"source": "usgs", "station_id": "A", "snow_cover_pct": 5.0}])
    out = hg.suggest_basins(5, signatures=sig, catchments=cat)
    ids = [r["station_id"] for r in out]
    assert set(ids[:2]) == {"A", "B"} and "D" not in ids and "E" not in ids and "C" not in ids  # D, E, C
    loose = [r["station_id"] for r in hg.suggest_basins(5, signatures=sig, catchments=cat, max_snow_pct=None,
                                                         min_years=10)]
    assert loose[0] == "B" and "E" in loose and "C" in loose and "D" not in loose
    assert [r["station_id"] for r in hg.suggest_basins(3, sources=["uk_ea"], signatures=sig, catchments=cat)] == ["B"]
    assert hg.suggest_basins(3, signatures=sig.iloc[0:0]) == []


def test_load_basin_from_archive_pieces(monkeypatch, tmp_path):
    catalog = [{"source": "usgs", "station_id": "S1", "name": "Test Creek", "latitude": 40.0, "longitude": -100.0,
                "extra": {"catchment_area_km2": 250.0}, "agency": "USGS", "license": "US-PD"}]
    idx = pd.date_range("2001-01-01", "2006-12-31", freq="D")
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"station_id": "S1", "date": idx, "value": rng.gamma(2, 3, len(idx))})  # m3/s
    monkeypatch.setattr("aquascope.archive.bundles.load_observations", lambda *a, **k: obs)
    monkeypatch.setattr("aquascope.archive.similar.load_station_catchments",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    calls = []

    def fake_forcing(lat, lon, start, end, **kw):
        calls.append((lat, lon, start, end))
        i = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"total_precipitation_sum": rng.gamma(0.8, 4, len(i)),
                             "potential_evaporation_sum_FAO_PENMAN_MONTEITH": 2.0 + 0 * np.arange(len(i))},
                            index=i), "best_match"

    monkeypatch.setattr("aquascope.archive.caravan.fetch_forcing", fake_forcing)
    b = hg.load_basin("usgs", "S1", cache_dir=tmp_path, catalog=catalog)
    assert b.id == "usgs/S1" and b.area_km2 == 250.0 and b.meta["area_source"] == "agency"
    assert calls and calls[0][2] == idx[0].date() and len(b.frame) == len(idx)
    assert b.frame["q_obs"].iloc[0] == pytest.approx(obs["value"].iloc[0] * 86.4 / 250.0)
    assert (tmp_path / "usgs__S1__auto__auto.parquet").exists()
    b2 = hg.load_basin("usgs", "S1", cache_dir=tmp_path, catalog=catalog, split="2005-01-01")
    assert b2.meta["cached"] and len(calls) == 1 and b2.split_date == pd.Timestamp("2005-01-01")
    with pytest.raises(KeyError):
        hg.load_basin("usgs", "nope", cache_dir=tmp_path, catalog=catalog)
    with pytest.raises(ValueError):
        hg.load_basin("usgs", "S1", cache_dir=tmp_path, refresh=True,
                      catalog=[{**catalog[0], "extra": {}}])  # no area anywhere
    env = hg.CalibrationEnv(b, max_steps=2)
    env.reset()
    _, r, *_ = env.step({"X1": 300, "X2": 0, "X3": 80, "X4": 2})
    assert isinstance(r, float)
