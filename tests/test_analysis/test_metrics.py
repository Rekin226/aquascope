"""Tests for the shared model-evaluation metrics module."""

import math

import numpy as np
import pytest

from aquascope.analysis.metrics import kge, log_nse, nse, pbias, r2, rmse


class TestNSE:
    def test_perfect_fit(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert nse(obs, obs) == pytest.approx(1.0)

    def test_mean_benchmark_is_zero(self):
        """Predicting the observed mean for every step gives NSE == 0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.full_like(obs, obs.mean())
        assert nse(obs, sim) == pytest.approx(0.0, abs=1e-10)

    def test_hand_computed(self):
        """obs=[1,2,3,4,5], sim=[1,2,3,4,6].

        ss_res = (0+0+0+0+1) = 1
        ss_tot = sum((obs-3)^2) = 4+1+0+1+4 = 10
        NSE = 1 - 1/10 = 0.9
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        assert nse(obs, sim) == pytest.approx(0.9)

    def test_constant_observed_returns_nan(self):
        """Zero variance in observed (ss_tot == 0) is undefined."""
        obs = np.array([5.0, 5.0, 5.0])
        sim = np.array([4.0, 5.0, 6.0])
        assert math.isnan(nse(obs, sim))

    def test_nan_pairs_dropped(self):
        obs = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        # Only indices 0, 1, 4 are valid in both arrays — a perfect match.
        assert nse(obs, sim) == pytest.approx(1.0)

    def test_empty_after_masking_returns_nan(self):
        obs = np.array([np.nan, np.nan])
        sim = np.array([1.0, 2.0])
        assert math.isnan(nse(obs, sim))


class TestLogNSE:
    def test_perfect_fit(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert log_nse(obs, obs) == pytest.approx(1.0)

    def test_handles_zero_flow(self):
        """Zero values shouldn't raise (epsilon guards log(0))."""
        obs = np.array([0.0, 1.0, 2.0, 3.0])
        sim = np.array([0.0, 1.0, 2.0, 3.0])
        result = log_nse(obs, sim)
        assert not math.isnan(result)
        assert result == pytest.approx(1.0)


class TestKGE:
    def test_perfect_fit(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert kge(obs, obs) == pytest.approx(1.0)

    def test_hand_computed_bias_only(self):
        """sim is obs scaled by 2x: same correlation/variability ratio
        shape, but beta (mean ratio) = 2.

        obs=[1,2,3,4,5], sim=[2,4,6,8,10]
        r = 1.0 (perfectly correlated)
        alpha = std(sim)/std(obs) = 2.0
        beta = mean(sim)/mean(obs) = 2.0
        KGE = 1 - sqrt((1-1)^2 + (2-1)^2 + (2-1)^2) = 1 - sqrt(2) ≈ -0.4142
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = obs * 2
        assert kge(obs, sim) == pytest.approx(1 - math.sqrt(2), abs=1e-6)

    def test_zero_variance_observed_returns_nan(self):
        obs = np.array([5.0, 5.0, 5.0])
        sim = np.array([4.0, 5.0, 6.0])
        assert math.isnan(kge(obs, sim))

    def test_single_value_returns_nan(self):
        """Correlation is undefined with fewer than 2 points."""
        obs = np.array([5.0])
        sim = np.array([5.0])
        assert math.isnan(kge(obs, sim))


class TestPBIAS:
    def test_perfect_fit_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert pbias(obs, obs) == pytest.approx(0.0)

    def test_hand_computed_overestimation(self):
        """obs sum = 15, sim sum = 18 -> PBIAS = 100*(18-15)/15 = 20.0"""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([2.0, 3.0, 4.0, 4.0, 5.0])
        assert pbias(obs, sim) == pytest.approx(20.0)

    def test_underestimation_is_negative(self):
        obs = np.array([2.0, 4.0, 6.0])
        sim = np.array([1.0, 2.0, 3.0])
        assert pbias(obs, sim) < 0

    def test_zero_observed_sum_returns_nan(self):
        obs = np.array([-1.0, 1.0])
        sim = np.array([0.0, 0.0])
        assert math.isnan(pbias(obs, sim))


class TestRMSE:
    def test_perfect_fit_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(obs, obs) == pytest.approx(0.0)

    def test_hand_computed(self):
        """errors = [1,1,1], squared = [1,1,1], mean = 1, sqrt = 1.0"""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([2.0, 3.0, 4.0])
        assert rmse(obs, sim) == pytest.approx(1.0)

    def test_always_non_negative(self):
        obs = np.array([5.0, 3.0, 8.0])
        sim = np.array([1.0, 9.0, 2.0])
        assert rmse(obs, sim) >= 0


class TestR2:
    def test_perfect_fit(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2(obs, obs) == pytest.approx(1.0)

    def test_matches_nse(self):
        """This implementation defines r2 identically to nse."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        assert r2(obs, sim) == pytest.approx(nse(obs, sim))
