"""Tests for aquascope.hydrology.signatures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.hydrology.signatures import (
    SignatureReport,
    baseflow_index_simple,
    compare_signatures,
    compute_signatures,
    flashiness_index,
    flow_elasticity,
    recession_constant,
    seasonality_index,
    similarity_score,
)


def _daily_index(n_days: int, start: str = "2000-01-01") -> pd.DatetimeIndex:
    """Create a DatetimeIndex of *n_days* starting from *start*."""
    return pd.date_range(start, periods=n_days, freq="D")


def _constant_series(value: float = 10.0, n_days: int = 730) -> pd.Series:
    """Return a constant daily discharge series."""
    return pd.Series(value, index=_daily_index(n_days), name="Q")


def _seasonal_series(n_days: int = 730) -> pd.Series:
    """Return a synthetic seasonal discharge series peaking in month 6."""
    idx = _daily_index(n_days)
    doy = idx.dayofyear.values.astype(float)
    # Peak near day 166 (mid-June, month 6)
    q = 10 + 40 * np.sin(2 * np.pi * (doy - 75) / 365.25)
    q = np.maximum(q, 0.5)
    return pd.Series(q, index=idx, name="Q")


def _make_report(**overrides: float) -> SignatureReport:
    """Build a dummy SignatureReport with sensible defaults."""
    defaults = {
        "mean_flow": 10.0,
        "median_flow": 9.0,
        "q5": 2.0,
        "q95": 18.0,
        "q5_q95_ratio": 0.11,
        "cv": 0.5,
        "iqr": 8.0,
        "high_flow_frequency": 5.0,
        "high_flow_duration": 3.0,
        "q_peak_mean": 4.0,
        "low_flow_frequency": 10.0,
        "low_flow_duration": 6.0,
        "baseflow_index": 0.6,
        "zero_flow_fraction": 0.0,
        "peak_month": 6,
        "seasonality_index": 0.3,
        "rising_limb_density": 0.45,
        "flashiness_index": 0.2,
        "mean_recession_constant": 0.05,
        "runoff_ratio": 0.4,
        "elasticity": 1.5,
    }
    defaults.update(overrides)
    return SignatureReport(**defaults)


class TestComputeSignatures:
    """Tests for the main compute_signatures function."""

    def test_compute_signatures_returns_report(self):
        """All fields of SignatureReport are populated with finite values."""
        q = _seasonal_series(n_days=730)
        report = compute_signatures(q)

        assert isinstance(report, SignatureReport)
        for field_name in [
            "mean_flow", "median_flow", "q5", "q95", "cv", "iqr",
            "high_flow_frequency", "high_flow_duration", "q_peak_mean",
            "low_flow_frequency", "baseflow_index", "zero_flow_fraction",
            "peak_month", "seasonality_index", "rising_limb_density",
            "flashiness_index", "mean_recession_constant",
        ]:
            val = getattr(report, field_name)
            assert val is not None, f"{field_name} should not be None"
            assert np.isfinite(val), f"{field_name} should be finite"

        # Without precip these should be None
        assert report.runoff_ratio is None
        assert report.elasticity is None

    def test_insufficient_data_raises(self):
        """Fewer than 365 values should raise ValueError."""
        q = pd.Series(np.ones(100), index=_daily_index(100))
        with pytest.raises(ValueError, match="365"):
            compute_signatures(q)

    def test_with_precipitation(self):
        """When precipitation is supplied, runoff_ratio and elasticity are populated."""
        idx = _daily_index(730)
        q = pd.Series(np.random.default_rng(42).uniform(5, 15, 730), index=idx)
        p = pd.Series(np.random.default_rng(7).uniform(2, 20, 730), index=idx)
        report = compute_signatures(q, precipitation=p)

        assert report.runoff_ratio is not None
        assert report.runoff_ratio > 0
        assert report.elasticity is not None


class TestFlashinessIndex:
    """Tests for flashiness_index."""

    def test_flashiness_constant_flow(self):
        """Constant flow should yield FI = 0."""
        q = _constant_series(10.0, 730)
        assert flashiness_index(q) == pytest.approx(0.0)

    def test_flashiness_variable_flow(self):
        """Variable flow should produce FI > 0."""
        rng = np.random.default_rng(0)
        q = pd.Series(rng.uniform(1, 100, 730), index=_daily_index(730))
        fi = flashiness_index(q)
        assert fi > 0


class TestSeasonalityIndex:
    """Tests for seasonality_index."""

    def test_seasonality_uniform(self):
        """Uniform monthly flow should give an index close to 0."""
        q = _constant_series(10.0, 365 * 3)
        si, _ = seasonality_index(q)
        assert si < 0.05

    def test_seasonality_concentrated(self):
        """All flow in one month should give index close to 1."""
        idx = _daily_index(365)
        q_vals = np.zeros(365)
        # Put all flow in July (days 182–212)
        q_vals[181:212] = 100.0
        q = pd.Series(q_vals, index=idx)
        si, _ = seasonality_index(q)
        assert si > 0.8

    def test_peak_month(self):
        """Seasonal series peaking near June should report month 6 or 7."""
        q = _seasonal_series(n_days=365 * 3)
        _, peak = seasonality_index(q)
        assert peak in (6, 7)


class TestBaseflowIndex:
    """Tests for baseflow_index_simple."""

    def test_baseflow_index_range(self):
        """BFI should be between 0 and 1."""
        q = _seasonal_series(730)
        bfi = baseflow_index_simple(q)
        assert 0.0 <= bfi <= 1.0


class TestRecessionConstant:
    """Tests for recession_constant."""

    def test_recession_constant_positive(self):
        """Recession constant should be positive for decaying flow."""
        # Create an exponentially decaying series with some noise
        idx = _daily_index(730)
        t = np.arange(730, dtype=float)
        # Repeat decay pattern every 60 days
        q_vals = 50 * np.exp(-0.05 * (t % 60))
        q = pd.Series(q_vals, index=idx)
        k = recession_constant(q, min_length=5)
        assert k > 0


class TestHighLowFlowFrequency:
    """Tests for high/low flow frequencies inside compute_signatures."""

    def test_high_flow_frequency(self):
        """Known data with spikes should show high_flow_frequency > 0."""
        idx = _daily_index(730)
        q_vals = np.full(730, 10.0)
        # Inject spikes above 3*median = 30
        q_vals[100:110] = 50.0
        q_vals[400:415] = 60.0
        q = pd.Series(q_vals, index=idx)
        report = compute_signatures(q)
        assert report.high_flow_frequency > 0

    def test_low_flow_frequency(self):
        """Known data with low periods should show low_flow_frequency > 0."""
        idx = _daily_index(730)
        q_vals = np.full(730, 10.0)
        # Insert very low flow (< 0.2 * median = 2.0)
        q_vals[200:230] = 1.0
        q_vals[500:540] = 0.5
        q = pd.Series(q_vals, index=idx)
        report = compute_signatures(q)
        assert report.low_flow_frequency > 0


class TestZeroFlowFraction:
    """Tests for zero_flow_fraction."""

    def test_zero_flow_fraction(self):
        """Series with known zero-flow days should report correct fraction."""
        idx = _daily_index(730)
        q_vals = np.full(730, 5.0)
        q_vals[0:73] = 0.0  # 10 % zeros
        q = pd.Series(q_vals, index=idx)
        report = compute_signatures(q)
        assert report.zero_flow_fraction == pytest.approx(73 / 730, abs=1e-6)


class TestFlowElasticity:
    """Tests for flow_elasticity."""

    def test_flow_elasticity(self):
        """Synthetic proportional precip–flow relationship yields finite elasticity."""
        idx = _daily_index(365 * 3)
        rng = np.random.default_rng(99)
        p = pd.Series(rng.uniform(2, 10, len(idx)), index=idx)
        q = pd.Series(0.4 * p.values + rng.normal(0, 0.3, len(idx)), index=idx)
        q = q.clip(lower=0)
        e = flow_elasticity(q, p)
        assert np.isfinite(e)

    def test_runoff_ratio_with_precip(self):
        """Runoff ratio should be within a reasonable range for synthetic data."""
        idx = _daily_index(730)
        rng = np.random.default_rng(42)
        p = pd.Series(rng.uniform(5, 20, 730), index=idx)
        q = pd.Series(rng.uniform(2, 8, 730), index=idx)
        report = compute_signatures(q, precipitation=p)
        assert report.runoff_ratio is not None
        assert 0 < report.runoff_ratio < 5  # generous bound


class TestCompareSignatures:
    """Tests for compare_signatures."""

    def test_compare_signatures_same(self):
        """Comparing identical reports should yield all-zero differences."""
        r = _make_report()
        diff = compare_signatures(r, r)
        for key, val in diff.items():
            assert val == pytest.approx(0.0), f"{key} should be 0"

    def test_compare_signatures_different(self):
        """Comparing different reports should yield non-zero differences for changed fields."""
        r1 = _make_report(mean_flow=10.0)
        r2 = _make_report(mean_flow=20.0)
        diff = compare_signatures(r1, r2)
        assert diff["mean_flow"] > 0


class TestSimilarityScore:
    """Tests for similarity_score."""

    def test_similarity_score_identical(self):
        """Identical reports should have similarity of 0."""
        r = _make_report()
        assert similarity_score(r, r) == pytest.approx(0.0)

    def test_similarity_score_different(self):
        """Different reports should have similarity > 0."""
        r1 = _make_report(mean_flow=10.0, baseflow_index=0.3)
        r2 = _make_report(mean_flow=50.0, baseflow_index=0.9)
        assert similarity_score(r1, r2) > 0
