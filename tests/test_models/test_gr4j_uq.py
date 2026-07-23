"""Tests for GR4J quantile prediction intervals (#77, epic #71).

Validates the residual and ensemble UQ paths and their coverage using the
probabilistic metrics from #76."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.analysis.metrics import picp
from aquascope.models.rainfall_runoff import (
    GR4J,
    GR4JProbabilisticResult,
    predict_quantiles,
    residual_quantile_bands,
)

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _synthetic_inputs(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    precip = pd.Series(rng.gamma(0.6, 6.0, n), index=idx)
    doy = idx.dayofyear.to_numpy()
    pet = pd.Series(2.5 + 1.5 * np.sin(2 * np.pi * doy / 365.0), index=idx)
    # "Observed" = a known GR4J run plus a little multiplicative noise.
    truth = GR4J(x1=320.0, x2=-1.0, x3=70.0, x4=1.6).simulate(precip, pet, warmup_days=0)
    obs = truth.streamflow * (1.0 + rng.normal(0.0, 0.1, n))
    obs = obs.clip(lower=0.0)
    return precip, pet, obs


class TestResidualQuantileBands:
    def test_additive_offsets_match_residual_quantiles(self):
        idx = pd.date_range("2020-01-01", periods=200)
        sim = pd.Series(np.full(200, 5.0), index=idx)
        rng = np.random.default_rng(1)
        resid = rng.normal(0.0, 1.0, 200)
        obs = pd.Series(5.0 + resid, index=idx)
        bands = residual_quantile_bands(sim, obs, [0.05, 0.5, 0.95])
        assert bands[0.5].iloc[0] == pytest.approx(5.0 + np.quantile(resid, 0.5), abs=1e-9)
        assert bands[0.95].iloc[0] == pytest.approx(5.0 + np.quantile(resid, 0.95), abs=1e-9)

    def test_monotonic_and_non_negative(self):
        idx = pd.date_range("2020-01-01", periods=100)
        sim = pd.Series(np.linspace(0.0, 3.0, 100), index=idx)
        rng = np.random.default_rng(2)
        obs = pd.Series(np.clip(sim.values + rng.normal(0, 2, 100), 0, None), index=idx)
        bands = residual_quantile_bands(sim, obs, QUANTILES)
        lo, hi = bands[0.05].values, bands[0.95].values
        assert np.all(lo <= bands[0.5].values + 1e-9)
        assert np.all(bands[0.5].values <= hi + 1e-9)
        assert np.all(lo >= 0.0)

    def test_heteroscedastic_widens_at_high_flow(self):
        idx = pd.date_range("2020-01-01", periods=300)
        sim = pd.Series(np.linspace(1.0, 50.0, 300), index=idx)
        rng = np.random.default_rng(3)
        obs = pd.Series(sim.values * (1.0 + rng.normal(0, 0.2, 300)), index=idx).clip(lower=0)
        bands = residual_quantile_bands(sim, obs, QUANTILES, heteroscedastic=True)
        width = bands[0.95].values - bands[0.05].values
        assert width[-1] > width[0]  # wider band at high flow

    def test_raises_without_finite_pairs(self):
        idx = pd.date_range("2020-01-01", periods=5)
        sim = pd.Series([np.nan] * 5, index=idx)
        obs = pd.Series([1.0] * 5, index=idx)
        with pytest.raises(ValueError):
            residual_quantile_bands(sim, obs, QUANTILES)


class TestPredictQuantiles:
    def test_residual_structure_and_coverage(self):
        precip, pet, obs = _synthetic_inputs()
        res = predict_quantiles(
            precip, pet, obs, quantiles=QUANTILES,
            method="residual", warmup_days=120, maxiter=4, seed=0,
        )
        assert isinstance(res, GR4JProbabilisticResult)
        assert res.method == "residual"
        assert set(res.quantiles) == set(QUANTILES)
        # median is the 0.5 band
        assert res.median.equals(res.quantiles[0.5])
        # bands non-negative and ordered
        assert np.all(res.quantiles[0.05].values >= 0.0)
        assert np.all(res.quantiles[0.05].values <= res.quantiles[0.95].values + 1e-9)
        # in-sample coverage of the central 90% band should be near nominal
        ev = slice(120, None)
        cov = picp(
            obs.values[ev], res.quantiles[0.05].values[ev], res.quantiles[0.95].values[ev]
        )
        assert cov >= 0.8

    def test_ensemble_method_runs(self):
        precip, pet, obs = _synthetic_inputs(n=400)
        res = predict_quantiles(
            precip, pet, obs, quantiles=(0.1, 0.5, 0.9),
            method="ensemble", warmup_days=90, maxiter=3, n_members=8, seed=1,
        )
        assert res.method == "ensemble"
        assert np.all(res.quantiles[0.1].values <= res.quantiles[0.9].values + 1e-9)

    def test_invalid_method_raises(self):
        precip, pet, obs = _synthetic_inputs(n=200)
        with pytest.raises(ValueError):
            predict_quantiles(precip, pet, obs, method="bogus", maxiter=2)
