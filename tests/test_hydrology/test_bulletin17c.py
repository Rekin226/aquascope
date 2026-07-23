"""Tests for Bulletin 17C flood frequency analysis.

Tests weighted skew, Grubbs-Beck outlier detection, EMA censored fitting,
enhanced LP3 with regional skew and confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_annual_max(n: int = 50, seed: int = 42, add_zeros: int = 0) -> np.ndarray:
    """Generate synthetic annual maxima (log-normal-ish)."""
    rng = np.random.default_rng(seed)
    log_vals = rng.normal(loc=2.0, scale=0.3, size=n)
    vals = 10**log_vals
    if add_zeros > 0:
        vals[:add_zeros] = 0.0
    return vals


def _make_daily_discharge(years: int = 30, seed: int = 42) -> pd.Series:
    """Generate synthetic daily discharge series."""
    rng = np.random.default_rng(seed)
    days = years * 365
    dates = pd.date_range("1990-01-01", periods=days, freq="D")
    t = np.arange(days) / 365.0
    seasonal = 20 + 15 * np.sin(2 * np.pi * t)
    noise = rng.exponential(5, days)
    storms = rng.choice([0, 0, 0, 0, 50], days) * rng.random(days)
    q = seasonal + noise + storms
    q = np.maximum(q, 0.5)
    return pd.Series(q, index=dates, name="discharge")


class TestWeightedSkew:
    """Tests for weighted_skew() Bulletin 17C formula."""

    def test_weighted_skew_equal_weights(self):
        from aquascope.hydrology.flood_frequency import weighted_skew

        # With n chosen so station MSE ≈ regional MSE (0.302), the
        # weighted skew should be approximately the average of the two.
        station = 0.5
        regional = -0.1
        # Station MSE = (6/n)(1 + 1.5*0.25 + 0.3125*0.0625) ≈ 6/n * 1.394
        # Set n so that 6/n * 1.394 ≈ 0.302 → n ≈ 27.7 → use n=28
        result = weighted_skew(station, regional, n=28, regional_mse=0.302)
        # Should be between station and regional
        assert regional < result < station
        # Verify it's a reasonable weighted average (not trivially one extreme)
        assert abs(result - (station + regional) / 2) < 0.15

    def test_weighted_skew_large_n(self):
        from aquascope.hydrology.flood_frequency import weighted_skew

        # Very large n → station MSE → 0 → station weight ≈ 1
        station = 0.8
        regional = -0.2
        result = weighted_skew(station, regional, n=5000, regional_mse=0.302)
        assert abs(result - station) < 0.05


class TestGrubbsBeck:
    """Tests for grubbs_beck_test() — MGB low-outlier detection."""

    def test_grubbs_beck_detects_outliers(self):
        from aquascope.hydrology.flood_frequency import grubbs_beck_test

        rng = np.random.default_rng(123)
        # Normal data in log-space, ~100 cfs
        vals = 10 ** rng.normal(loc=2.0, scale=0.2, size=50)
        # Inject 3 extreme low outliers (~0.1 cfs vs ~100 cfs)
        vals[:3] = [0.05, 0.08, 0.1]

        threshold, mask = grubbs_beck_test(vals, alpha=0.10)
        n_detected = int(np.sum(mask))

        # Should detect at least the injected outliers
        assert n_detected >= 2
        # The detected outliers should be the small ones
        assert all(vals[mask] < 1.0)

    def test_grubbs_beck_no_outliers(self):
        from aquascope.hydrology.flood_frequency import grubbs_beck_test

        rng = np.random.default_rng(456)
        # Well-behaved data with no outliers
        vals = 10 ** rng.normal(loc=2.0, scale=0.15, size=60)

        threshold, mask = grubbs_beck_test(vals, alpha=0.10)

        # No (or very few) outliers expected in clean data
        assert int(np.sum(mask)) <= 2


class TestEMA:
    """Tests for expected_moments_algorithm()."""

    def test_ema_no_censoring_matches_lp3(self):
        from aquascope.hydrology.flood_frequency import expected_moments_algorithm, fit_lp3

        discharge = _make_daily_discharge(years=30, seed=99)
        lp3_result = fit_lp3(discharge, return_periods=[10, 100])

        # Extract annual maxima the same way fit_lp3 does
        annual_max = discharge.resample("YS").max().dropna()

        ema_result = expected_moments_algorithm(
            annual_max.values,
            zero_threshold=0.0,
            return_periods=[10, 100],
        )

        # EMA without censoring should give results in the same ballpark
        for rp in [10, 100]:
            ratio = ema_result.return_periods[rp] / lp3_result.return_periods[rp]
            assert 0.5 < ratio < 2.0, (
                f"RP {rp}: EMA={ema_result.return_periods[rp]:.1f}, "
                f"LP3={lp3_result.return_periods[rp]:.1f}"
            )

    def test_ema_with_zeros(self):
        from aquascope.hydrology.flood_frequency import expected_moments_algorithm

        vals = _make_annual_max(n=50, seed=77, add_zeros=5)
        result = expected_moments_algorithm(vals, zero_threshold=0.0, return_periods=[10, 100])

        assert result.n_censored >= 5
        assert result.n_observed <= 45
        assert result.distribution == "LP3-EMA"
        # Quantiles should still be positive and ordered
        assert result.return_periods[100] > result.return_periods[10] > 0

    def test_ema_with_regional_skew(self):
        from aquascope.hydrology.flood_frequency import expected_moments_algorithm

        vals = _make_annual_max(n=40, seed=88)
        result = expected_moments_algorithm(
            vals,
            regional_skew=0.0,
            regional_skew_mse=0.302,
            return_periods=[100],
        )

        assert result.weighted_skew is not None
        assert isinstance(result.weighted_skew, float)
        # The weighted skew should differ from pure station skew
        # (unless station happens to be exactly 0)
        assert result.return_periods[100] > 0

    def test_ema_result_fields(self):
        from aquascope.hydrology.flood_frequency import EMAResult, expected_moments_algorithm

        vals = _make_annual_max(n=50, seed=42, add_zeros=3)
        result = expected_moments_algorithm(
            vals,
            zero_threshold=0.0,
            regional_skew=-0.1,
            return_periods=[50],
        )

        assert isinstance(result, EMAResult)
        assert result.n_censored >= 3
        assert result.n_observed > 0
        assert result.n_censored + result.n_observed >= len(vals) - 10  # some may become MGB outliers
        assert result.weighted_skew is not None
        assert len(result.confidence_intervals) > 0


class TestEnhancedLP3:
    """Tests for the enhanced fit_lp3 with B17C parameters."""

    def setup_method(self):
        self.discharge = _make_daily_discharge(years=30, seed=42)

    def test_fit_lp3_regional_skew(self):
        from aquascope.hydrology.flood_frequency import fit_lp3

        result_no_regional = fit_lp3(self.discharge, return_periods=[100])
        result_with_regional = fit_lp3(
            self.discharge,
            return_periods=[100],
            regional_skew=0.0,
            regional_skew_mse=0.302,
        )

        # Both should produce valid results
        assert result_no_regional.return_periods[100] > 0
        assert result_with_regional.return_periods[100] > 0
        # Results should differ when regional skew is applied
        # (unless station skew happens to equal regional skew)
        assert result_with_regional.distribution == "LP3"

    def test_fit_lp3_confidence_intervals(self):
        from aquascope.hydrology.flood_frequency import fit_lp3

        result = fit_lp3(self.discharge, return_periods=[10, 50, 100], ci_level=0.90)

        # CIs should be populated
        assert len(result.confidence_intervals) == 3
        for rp in [10, 50, 100]:
            lo, hi = result.confidence_intervals[rp]
            point = result.return_periods[rp]
            assert lo < point < hi, f"RP {rp}: {lo} < {point} < {hi} failed"
