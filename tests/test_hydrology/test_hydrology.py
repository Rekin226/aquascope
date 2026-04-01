"""Tests for aquascope.hydrology module.

Tests flow duration curves, baseflow separation, recession analysis,
and flood frequency analysis using synthetic hydrological data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_discharge(years: int = 10, seed: int = 42) -> pd.Series:
    """Generate synthetic daily discharge with seasonal pattern + noise."""
    rng = np.random.default_rng(seed)
    days = years * 365
    dates = pd.date_range("2010-01-01", periods=days, freq="D")

    # Seasonal base + random storms
    t = np.arange(days) / 365.0
    seasonal = 20 + 15 * np.sin(2 * np.pi * t)  # spring peak
    noise = rng.exponential(5, days)
    storms = rng.choice([0, 0, 0, 0, 50], days) * rng.random(days)
    q = seasonal + noise + storms
    q = np.maximum(q, 0.5)  # no zero flow

    return pd.Series(q, index=dates, name="discharge")


class TestFlowDuration:
    """Tests for flow_duration_curve and low_flow_stat."""

    def setup_method(self):
        self.q = _make_discharge()

    def test_fdc_returns_result(self):
        from aquascope.hydrology import flow_duration_curve

        result = flow_duration_curve(self.q)
        assert len(result.exceedance) == len(result.discharge)
        assert len(result.percentiles) > 0

    def test_fdc_percentiles_decreasing(self):
        from aquascope.hydrology import flow_duration_curve

        result = flow_duration_curve(self.q)
        assert result.percentiles[5] > result.percentiles[95]

    def test_fdc_custom_percentiles(self):
        from aquascope.hydrology import flow_duration_curve

        result = flow_duration_curve(self.q, percentiles=[50, 95])
        assert set(result.percentiles.keys()) == {50, 95}

    def test_low_flow_7q10(self):
        from aquascope.hydrology import low_flow_stat

        val = low_flow_stat(self.q, n_day=7, return_period=10)
        assert val > 0
        assert val < self.q.median()

    def test_low_flow_30q5(self):
        from aquascope.hydrology import low_flow_stat

        val = low_flow_stat(self.q, n_day=30, return_period=5)
        assert val > 0

    def test_low_flow_insufficient_data(self):
        from aquascope.hydrology import low_flow_stat

        short = self.q.iloc[:365]  # Only 1 year
        with pytest.raises(ValueError, match="≥3"):
            low_flow_stat(short)


class TestBaseflow:
    """Tests for Lyne-Hollick and Eckhardt baseflow separation."""

    def setup_method(self):
        self.q = _make_discharge()

    def test_lyne_hollick_returns_result(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        assert "baseflow" in result.df.columns
        assert "quickflow" in result.df.columns
        assert 0 < result.bfi < 1

    def test_lyne_hollick_components_sum(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        total_sum = result.df["baseflow"].sum() + result.df["quickflow"].sum()
        np.testing.assert_allclose(total_sum, result.df["total"].sum(), rtol=0.01)

    def test_eckhardt_returns_result(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q)
        assert "baseflow" in result.df.columns
        assert 0 < result.bfi < 1
        assert result.method == "eckhardt"

    def test_eckhardt_bfi_max_effect(self):
        from aquascope.hydrology import eckhardt

        low_bfi = eckhardt(self.q, bfi_max=0.25)
        high_bfi = eckhardt(self.q, bfi_max=0.80)
        assert low_bfi.bfi < high_bfi.bfi

    def test_baseflow_never_exceeds_total(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        assert (result.df["baseflow"] <= result.df["total"] + 1e-10).all()


class TestRecession:
    """Tests for recession analysis."""

    def setup_method(self):
        self.q = _make_discharge()

    def test_identify_recessions(self):
        from aquascope.hydrology import identify_recessions

        segments = identify_recessions(self.q)
        assert len(segments) > 0
        for seg in segments:
            assert len(seg.discharge) >= 5
            assert seg.start < seg.end

    def test_recession_analysis_full(self):
        from aquascope.hydrology import recession_analysis

        result = recession_analysis(self.q)
        assert result.recession_constant > 0
        assert result.half_life_days > 0
        assert 0 < result.r_squared <= 1.0

    def test_recession_empty_data(self):
        from aquascope.hydrology import recession_analysis

        short = pd.Series([10.0, 10.0, 10.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        result = recession_analysis(short)
        assert len(result.segments) == 0


class TestFloodFrequency:
    """Tests for GEV and LP3 flood frequency analysis."""

    def setup_method(self):
        self.q = _make_discharge(years=30)

    def test_fit_gev(self):
        from aquascope.hydrology import fit_gev

        result = fit_gev(self.q)
        assert result.distribution == "GEV"
        assert 100 in result.return_periods
        assert result.return_periods[100] > result.return_periods[2]

    def test_fit_gev_confidence_intervals(self):
        from aquascope.hydrology import fit_gev

        result = fit_gev(self.q)
        assert len(result.confidence_intervals) > 0
        for rp, (lo, hi) in result.confidence_intervals.items():
            assert lo < hi

    def test_fit_lp3(self):
        from aquascope.hydrology import fit_lp3

        result = fit_lp3(self.q)
        assert result.distribution == "LP3"
        assert 100 in result.return_periods
        assert result.return_periods[100] > result.return_periods[2]

    def test_gev_insufficient_data(self):
        from aquascope.hydrology import fit_gev

        short = self.q.iloc[:365 * 3]
        with pytest.raises(ValueError, match="≥5"):
            fit_gev(short)

    def test_fit_gev_custom_return_periods(self):
        from aquascope.hydrology import fit_gev

        result = fit_gev(self.q, return_periods=[10, 50])
        assert set(result.return_periods.keys()) == {10, 50}
