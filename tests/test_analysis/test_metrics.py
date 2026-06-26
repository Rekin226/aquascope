"""Tests for the shared model-evaluation metrics module."""

import math

import numpy as np
import pytest

from aquascope.analysis.metrics import (
    crps_ensemble,
    crps_from_quantiles,
    kge,
    log_nse,
    mpiw,
    nse,
    pbias,
    picp,
    pinball_loss,
    r2,
    rmse,
)


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


class TestPinballLoss:
    def test_median_equals_half_abs_error(self):
        # q=0.5: loss reduces to 0.5 * |error|
        loss = pinball_loss(np.array([10.0]), np.array([8.0]), 0.5)
        assert loss == pytest.approx(1.0)

    def test_under_prediction_high_quantile(self):
        # obs>pred (under-prediction), q=0.9 -> penalty 0.9 * error
        loss = pinball_loss(np.array([10.0]), np.array([8.0]), 0.9)
        assert loss == pytest.approx(1.8)

    def test_over_prediction_high_quantile(self):
        # obs<pred (over-prediction), q=0.9 -> penalty 0.1 * |error|
        loss = pinball_loss(np.array([8.0]), np.array([10.0]), 0.9)
        assert loss == pytest.approx(0.2)

    def test_perfect_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert pinball_loss(obs, obs, 0.7) == pytest.approx(0.0)

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError):
            pinball_loss(np.array([1.0]), np.array([1.0]), 1.0)

    def test_nan_aware(self):
        loss = pinball_loss(
            np.array([10.0, np.nan]), np.array([8.0, 5.0]), 0.5
        )
        assert loss == pytest.approx(1.0)


class TestPICP:
    def test_partial_coverage(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower = np.zeros(5)
        upper = np.full(5, 3.0)
        # 1,2,3 inside; 4,5 outside -> 0.6
        assert picp(obs, lower, upper) == pytest.approx(0.6)

    def test_full_coverage(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert picp(obs, np.zeros(3), np.full(3, 10.0)) == pytest.approx(1.0)

    def test_nominal_coverage_on_calibrated_data(self):
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(20000)
        lower = np.full_like(obs, -1.6449)  # N(0,1) 5% quantile
        upper = np.full_like(obs, 1.6449)  # N(0,1) 95% quantile
        assert picp(obs, lower, upper) == pytest.approx(0.90, abs=0.02)


class TestMPIW:
    def test_mean_width(self):
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 5.0])
        assert mpiw(lower, upper) == pytest.approx(3.0)  # (2 + 4) / 2

    def test_normalized_by_observed_range(self):
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 5.0])
        observed = np.array([0.0, 10.0])  # range 10
        assert mpiw(lower, upper, observed=observed) == pytest.approx(0.3)

    def test_nan_aware(self):
        lower = np.array([0.0, np.nan])
        upper = np.array([2.0, 5.0])
        assert mpiw(lower, upper) == pytest.approx(2.0)


class TestCRPSEnsemble:
    def test_deterministic_ensemble_reduces_to_mae(self):
        # All members equal -> spread term is 0 -> CRPS == |mean - obs|
        obs = np.array([10.0])
        ens = np.array([[8.0, 8.0, 8.0]])
        assert crps_ensemble(obs, ens) == pytest.approx(2.0)

    def test_perfect_deterministic_is_zero(self):
        obs = np.array([5.0, 6.0])
        ens = np.array([[5.0, 5.0], [6.0, 6.0]])
        assert crps_ensemble(obs, ens) == pytest.approx(0.0)

    def test_two_member_spread(self):
        # obs=0, members {-1, 1}: term1 = mean(1,1)=1; spread = (|−1−1|*2)/(2*4)=0.5
        obs = np.array([0.0])
        ens = np.array([[-1.0, 1.0]])
        assert crps_ensemble(obs, ens) == pytest.approx(0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            crps_ensemble(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    def test_nan_observation_dropped(self):
        obs = np.array([10.0, np.nan])
        ens = np.array([[8.0, 8.0], [0.0, 0.0]])
        assert crps_ensemble(obs, ens) == pytest.approx(2.0)


class TestCRPSFromQuantiles:
    def test_perfect_prediction_is_zero(self):
        obs = np.array([1.0, 2.0, 3.0])
        preds = {q: obs.copy() for q in (0.1, 0.5, 0.9)}
        assert crps_from_quantiles(obs, preds) == pytest.approx(0.0)

    def test_constant_offset_matches_pinball_integral(self):
        # All quantiles predict 8 while obs is 10, on a symmetric grid:
        # 2 * mean_q (q * 2) over q in {0.1..0.9} = 2 * (2 * 0.5) = 2.0 = |10-8|.
        obs = np.array([10.0])
        grid = np.round(np.arange(0.1, 0.95, 0.1), 2)
        preds = {float(q): np.array([8.0]) for q in grid}
        assert crps_from_quantiles(obs, preds) == pytest.approx(2.0, abs=1e-9)

    def test_empty_returns_nan(self):
        assert math.isnan(crps_from_quantiles(np.array([1.0]), {}))
