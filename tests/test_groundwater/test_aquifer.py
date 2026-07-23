"""Tests for aquifer hydraulics and pumping test analysis."""

from __future__ import annotations

import numpy as np
import pytest

from aquascope.groundwater.aquifer import (
    AquiferParams,
    SafeYieldResult,
    cooper_jacob,
    estimate_transmissivity,
    safe_yield,
    theis_drawdown,
    theis_recovery,
)


class TestTheisDrawdown:
    def setup_method(self):
        self.T = 500.0  # m²/day
        self.S = 0.001
        self.Q = 1000.0  # m³/day
        self.r = 100.0  # m

    def test_drawdown_increases_with_time(self):
        t = np.array([0.1, 1.0, 10.0, 100.0])
        s = theis_drawdown(self.T, self.S, self.Q, self.r, t)
        assert all(s[i] < s[i + 1] for i in range(len(s) - 1))

    def test_drawdown_positive(self):
        t = np.array([0.01, 0.1, 1.0, 10.0])
        s = theis_drawdown(self.T, self.S, self.Q, self.r, t)
        assert np.all(s > 0)

    def test_drawdown_decreases_with_distance(self):
        t = np.array([10.0])
        s_near = theis_drawdown(self.T, self.S, self.Q, 50.0, t)
        s_far = theis_drawdown(self.T, self.S, self.Q, 200.0, t)
        assert s_near[0] > s_far[0]

    def test_drawdown_proportional_to_pumping_rate(self):
        t = np.array([1.0])
        s1 = theis_drawdown(self.T, self.S, 1000.0, self.r, t)
        s2 = theis_drawdown(self.T, self.S, 2000.0, self.r, t)
        np.testing.assert_allclose(s2, 2.0 * s1, rtol=1e-10)

    def test_known_solution(self):
        # For large t (small u), W(u) ≈ -0.5772 - ln(u)
        # Use a case where u is small enough for analytical verification
        t_val, s_val, q_val, r_val = 100.0, 0.0001, 500.0, 10.0
        t = np.array([100.0])
        s = theis_drawdown(t_val, s_val, q_val, r_val, t)
        u = r_val**2 * s_val / (4 * t_val * t[0])
        from scipy.special import exp1

        expected = q_val / (4 * np.pi * t_val) * exp1(u)
        np.testing.assert_allclose(s[0], expected, rtol=1e-10)

    def test_negative_transmissivity_raises(self):
        with pytest.raises(ValueError, match="Transmissivity"):
            theis_drawdown(-1.0, self.S, self.Q, self.r, 1.0)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="positive"):
            theis_drawdown(self.T, self.S, self.Q, self.r, -1.0)


class TestCooperJacob:
    def setup_method(self):
        self.T = 500.0
        self.S = 0.001
        self.Q = 1000.0
        self.r = 100.0

    def test_approximates_theis_for_small_u(self):
        # For large t, Cooper-Jacob should be close to Theis
        t = np.array([100.0, 500.0, 1000.0])
        s_theis = theis_drawdown(self.T, self.S, self.Q, self.r, t)
        s_cj = cooper_jacob(self.T, self.S, self.Q, self.r, t)
        np.testing.assert_allclose(s_cj, s_theis, rtol=0.01)

    def test_drawdown_increases_with_time(self):
        t = np.array([10.0, 100.0, 1000.0])
        s = cooper_jacob(self.T, self.S, self.Q, self.r, t)
        assert all(s[i] < s[i + 1] for i in range(len(s) - 1))


class TestTheisRecovery:
    def setup_method(self):
        self.T = 500.0
        self.Q = 1000.0
        self.tp = 1.0  # 1 day pumping

    def test_residual_drawdown_decreases(self):
        t = np.array([1.5, 2.0, 5.0, 10.0])
        s_prime = theis_recovery(self.T, self.Q, t, self.tp)
        assert all(s_prime[i] > s_prime[i + 1] for i in range(len(s_prime) - 1))

    def test_residual_positive(self):
        t = np.array([1.1, 2.0, 5.0])
        s_prime = theis_recovery(self.T, self.Q, t, self.tp)
        assert np.all(s_prime > 0)

    def test_t_less_than_tp_raises(self):
        with pytest.raises(ValueError, match="greater than"):
            theis_recovery(self.T, self.Q, np.array([0.5]), self.tp)


class TestEstimateTransmissivity:
    def setup_method(self):
        # Generate synthetic pumping test data from known parameters
        self.T_true = 500.0
        self.S_true = 0.001
        self.Q = 1000.0
        self.r = 100.0
        self.time = np.logspace(-2, 2, 50)
        self.drawdown = theis_drawdown(self.T_true, self.S_true, self.Q, self.r, self.time)

    def test_cooper_jacob_estimate(self):
        # Filter to late-time data where CJ is valid
        mask = self.time > 1.0
        result = estimate_transmissivity(
            self.time[mask], self.drawdown[mask], self.Q, self.r, method="cooper_jacob"
        )
        assert isinstance(result, AquiferParams)
        assert result.method == "cooper_jacob"
        np.testing.assert_allclose(result.transmissivity, self.T_true, rtol=0.15)

    def test_theis_estimate(self):
        result = estimate_transmissivity(
            self.time, self.drawdown, self.Q, self.r, method="theis"
        )
        assert isinstance(result, AquiferParams)
        assert result.method == "theis"
        np.testing.assert_allclose(result.transmissivity, self.T_true, rtol=0.05)
        np.testing.assert_allclose(result.storativity, self.S_true, rtol=0.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_transmissivity(self.time, self.drawdown, self.Q, self.r, method="invalid")


class TestSafeYield:
    def test_sustainable(self):
        result = safe_yield(area_km2=100.0, recharge_mm=200.0, current_extraction_mm=100.0)
        assert isinstance(result, SafeYieldResult)
        assert result.assessment == "sustainable"
        assert result.ratio < 0.7

    def test_at_risk(self):
        result = safe_yield(area_km2=100.0, recharge_mm=200.0, current_extraction_mm=160.0)
        assert result.assessment == "at risk"

    def test_unsustainable(self):
        result = safe_yield(area_km2=100.0, recharge_mm=200.0, current_extraction_mm=250.0)
        assert result.assessment == "unsustainable"
        assert result.ratio > 1.0

    def test_zero_recharge_raises(self):
        with pytest.raises(ValueError, match="positive"):
            safe_yield(area_km2=100.0, recharge_mm=0.0, current_extraction_mm=50.0)
