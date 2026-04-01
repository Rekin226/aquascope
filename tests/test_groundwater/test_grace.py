"""Tests for GRACE satellite groundwater storage estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquascope.groundwater.grace import DepletionResult, GRACEProcessor, GWSAnomaly, GWSResult, TrendResult


def _monthly_index(n: int = 120, start: str = "2002-04-01") -> pd.DatetimeIndex:
    """Create a monthly DatetimeIndex."""
    return pd.date_range(start, periods=n, freq="MS")


class TestComputeGWS:
    def setup_method(self):
        self.proc = GRACEProcessor(area_km2=50_000)
        self.idx = _monthly_index(60)

    def test_gws_equals_tws_minus_sm_minus_sw(self):
        tws = pd.Series(np.ones(60) * 10.0, index=self.idx)
        sm = pd.Series(np.ones(60) * 3.0, index=self.idx)
        sw = pd.Series(np.ones(60) * 2.0, index=self.idx)
        result = self.proc.compute_gws(tws, sm, sw)
        assert isinstance(result, GWSResult)
        np.testing.assert_allclose(result.gws.values, 5.0)

    def test_gws_mean_and_std(self):
        np.random.seed(42)
        tws = pd.Series(np.random.randn(60) * 10.0 + 50.0, index=self.idx)
        sm = pd.Series(np.random.randn(60) * 3.0 + 20.0, index=self.idx)
        sw = pd.Series(np.random.randn(60) * 2.0 + 5.0, index=self.idx)
        result = self.proc.compute_gws(tws, sm, sw)
        assert abs(result.mean_gws - result.gws.mean()) < 1e-10
        assert abs(result.std_gws - result.gws.std()) < 1e-10

    def test_empty_series_raises(self):
        import pytest

        empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            self.proc.compute_gws(empty, empty, empty)

    def test_mismatched_lengths_raises(self):
        import pytest

        a = pd.Series([1.0, 2.0])
        b = pd.Series([1.0])
        with pytest.raises(ValueError, match="same length"):
            self.proc.compute_gws(a, b, b)


class TestTrendAnalysis:
    def setup_method(self):
        self.proc = GRACEProcessor()
        self.idx = _monthly_index(120)

    def test_linear_trend_detected(self):
        trend_vals = np.linspace(100, 50, 120)  # declining
        gws = pd.Series(trend_vals, index=self.idx)
        result = self.proc.trend_analysis(gws)
        assert isinstance(result, TrendResult)
        assert result.slope < 0
        assert result.r_squared > 0.99

    def test_seasonal_amplitude_computed(self):
        t = np.arange(120)
        vals = 5.0 * np.sin(2 * np.pi * t / 12) + 50.0
        gws = pd.Series(vals, index=self.idx)
        result = self.proc.trend_analysis(gws)
        assert result.seasonal_amplitude is not None
        assert result.seasonal_amplitude > 1.0

    def test_short_series_raises(self):
        import pytest

        gws = pd.Series([1.0, 2.0], index=_monthly_index(2))
        with pytest.raises(ValueError, match="at least 3"):
            self.proc.trend_analysis(gws)


class TestAnomalyDetection:
    def setup_method(self):
        self.proc = GRACEProcessor()

    def test_detects_extreme_values(self):
        np.random.seed(0)
        idx = _monthly_index(100)
        vals = np.random.randn(100) * 5.0
        vals[50] = -50.0  # extreme depletion
        vals[75] = 50.0  # extreme surplus
        gws = pd.Series(vals, index=idx)
        anomalies = self.proc.anomaly_detection(gws, threshold_sigma=2.0)
        assert len(anomalies) >= 2
        types = {a.anomaly_type for a in anomalies}
        assert "depletion" in types
        assert "surplus" in types

    def test_no_anomalies_in_uniform_data(self):
        idx = _monthly_index(50)
        gws = pd.Series(np.ones(50) * 10.0, index=idx)
        anomalies = self.proc.anomaly_detection(gws)
        assert len(anomalies) == 0

    def test_anomaly_has_correct_fields(self):
        idx = _monthly_index(50)
        vals = np.zeros(50)
        vals[25] = 100.0
        gws = pd.Series(vals, index=idx)
        anomalies = self.proc.anomaly_detection(gws, threshold_sigma=2.0)
        assert len(anomalies) > 0
        a = anomalies[0]
        assert isinstance(a, GWSAnomaly)
        assert hasattr(a, "date")
        assert hasattr(a, "z_score")


class TestDepletionRate:
    def setup_method(self):
        self.proc = GRACEProcessor(area_km2=100_000)

    def test_declining_series_negative_rate(self):
        idx = _monthly_index(120)
        # Linear decline: 100mm over 10 years → ~-10 mm/year
        vals = np.linspace(50, -50, 120)
        gws = pd.Series(vals, index=idx)
        result = self.proc.depletion_rate(gws)
        assert isinstance(result, DepletionResult)
        assert result.rate_mm_per_year < 0

    def test_km3_conversion(self):
        idx = _monthly_index(120)
        vals = np.linspace(50, -50, 120)
        gws = pd.Series(vals, index=idx)
        result = self.proc.depletion_rate(gws)
        assert result.rate_km3_per_year is not None
        assert result.rate_km3_per_year < 0

    def test_confidence_interval(self):
        idx = _monthly_index(60)
        vals = np.linspace(10, -10, 60) + np.random.default_rng(42).normal(0, 1, 60)
        gws = pd.Series(vals, index=idx)
        result = self.proc.depletion_rate(gws)
        ci_low, ci_high = result.confidence_interval
        assert ci_low < ci_high
        assert ci_low <= result.rate_mm_per_year <= ci_high
